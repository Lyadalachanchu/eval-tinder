"""Grade traces with a frozen grader version; persist every result as a GradingRun.

Predictions are stored alongside, never over, human labels. Cache keys include
project, policy epoch, manifest hash, rendered-case hash, and execution settings;
cache hits create a new GradingRun that points at the original for provenance.
Audits bypass the cache by default and record that choice.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.config import Settings, get_settings
from eval_tinder.db.enums import ExposureKind, ExposureStatus, GradingPurpose, GradingStatus
from eval_tinder.db.models import GradingRun, GraderVersion, PartitionAssignment, Project, TraceSnapshot
from eval_tinder.domain.manifest import GraderManifest
from eval_tinder.domain.rendering import CaseDocument, render_project_context, render_trace
from eval_tinder.grader.runtime import GradeResult, grade
from eval_tinder.grader.signature import build_module
from eval_tinder.ids import hash_value
from eval_tinder.llm.budget import BudgetGuard
from eval_tinder.llm.factory import MeteredLM, build_grading_lm, new_budget_guard
from eval_tinder.services.review import record_exposure

log = logging.getLogger(__name__)

PURPOSE_EXPOSURE = {
    GradingPurpose.PROBE: ExposureKind.PROBE,
    GradingPurpose.POOL: ExposureKind.POOL,
    GradingPurpose.BULK: ExposureKind.BULK_GRADING,
    GradingPurpose.OPTIMIZATION: ExposureKind.OPTIMIZATION,
    GradingPurpose.DEV_EVALUATION: ExposureKind.OPTIMIZATION,
}


class GradingError(RuntimeError):
    pass


class SealedMaterial(GradingError):
    pass


def cache_key_for(project: Project, manifest: GraderManifest, case: CaseDocument) -> str:
    return hash_value(
        {
            "project_id": project.id,
            "policy_epoch": project.policy_epoch,
            "manifest_hash": manifest.manifest_hash,
            "case_hash": case.text_hash,
            "execution": manifest.model_config_.sanitized().model_dump(),
        }
    )


@dataclass
class GraderRuntime:
    """A grader version bound to a model and (optionally) a budget guard."""

    grader: GraderVersion
    manifest: GraderManifest
    module: Any
    lm: Any
    project_context: str

    @classmethod
    def build(
        cls,
        project: Project,
        grader: GraderVersion,
        *,
        settings: Settings | None = None,
        budget: BudgetGuard | None = None,
        lm: Any | None = None,
    ) -> "GraderRuntime":
        settings = settings or get_settings()
        manifest = GraderManifest.from_dict(grader.manifest)
        base_lm = lm or build_grading_lm(manifest.model_config_, settings=settings)
        if budget is not None:
            base_lm = MeteredLM(base_lm, budget, role="grading")
        return cls(
            grader=grader,
            manifest=manifest,
            module=build_module(manifest.instruction_text),
            lm=base_lm,
            project_context=render_project_context(project.description, manifest.immutable_policy_context),
        )

    def grade_case(self, case: CaseDocument, *, max_case_chars: int) -> GradeResult:
        result = grade(
            self.module, self.lm, case=case, project_context=self.project_context, max_case_chars=max_case_chars
        )
        result.prompt_hash = self.manifest.prompt_hash
        return result


def _assert_not_sealed(session: Session, trace: TraceSnapshot, purpose: str) -> None:
    assignment = session.scalar(
        select(PartitionAssignment).where(
            PartitionAssignment.project_id == trace.project_id, PartitionAssignment.group_id == trace.group_id
        )
    )
    if assignment is not None and assignment.exposure_status == ExposureStatus.QUARANTINED:
        # Quarantined evidence (cross-partition duplicates) is invalid for every purpose, audits included.
        raise SealedMaterial(f"group {trace.group_id} is QUARANTINED; not available for {purpose}")
    if purpose == GradingPurpose.AUDIT:
        return  # the audit service grades its own sealed sample
    if assignment is not None and assignment.exposure_status == ExposureStatus.SEALED:
        raise SealedMaterial(f"group {trace.group_id} is {assignment.exposure_status}; not available for {purpose}")
    if assignment is not None and assignment.partition == "AUDIT_RESERVE" and purpose != GradingPurpose.EXPERIMENT:
        raise SealedMaterial(f"AUDIT_RESERVE material is not available for {purpose}")


def _lookup_cache(session: Session, key: str) -> GradingRun | None:
    return session.scalar(
        select(GradingRun)
        .where(GradingRun.cache_key == key, GradingRun.status == GradingStatus.OK, GradingRun.cache_hit_of.is_(None))
        .order_by(GradingRun.created_at)
        .limit(1)
    )


def _persist_hit(
    session: Session, project: Project, runtime: GraderRuntime, trace: TraceSnapshot, key: str, hit: GradingRun, *,
    purpose: str, job_id: str | None, audit_run_id: str | None,
) -> GradingRun:
    run = GradingRun(
        project_id=project.id, grader_id=runtime.grader.id, trace_id=trace.id, purpose=purpose,
        prompt_hash=hit.prompt_hash, cache_key=key, cache_hit_of=hit.id, status=hit.status,
        verdict=hit.verdict, evidence=hit.evidence, explanation=hit.explanation, usage={"cached": True},
        latency_ms=0, error=None, attempt=hit.attempt, job_id=job_id, audit_run_id=audit_run_id,
    )
    session.add(run)
    # A cache hit is still a use of this group for `purpose`: exposure history must not depend on
    # whether the provider was actually called, or a cached probe/pool/bulk use would go unrecorded.
    _record_purpose_exposure(session, project, trace, purpose, job_id)
    session.flush()
    return run


def _persist_result(
    session: Session, project: Project, runtime: GraderRuntime, trace: TraceSnapshot, key: str, result: GradeResult, *,
    purpose: str, job_id: str | None, audit_run_id: str | None,
) -> GradingRun:
    run = GradingRun(
        project_id=project.id,
        grader_id=runtime.grader.id,
        trace_id=trace.id,
        purpose=purpose,
        prompt_hash=result.prompt_hash,
        cache_key=key,
        status=result.status,
        verdict=result.verdict,
        evidence=result.evidence,
        explanation=result.explanation,
        usage=result.usage,
        latency_ms=result.latency_ms,
        error=result.error,
        attempt=result.attempts,
        job_id=job_id,
        audit_run_id=audit_run_id,
    )
    session.add(run)
    if result.status != GradingStatus.BUDGET_EXHAUSTED:  # a refused call never touched the model
        _record_purpose_exposure(session, project, trace, purpose, job_id)
    session.flush()
    return run


def grade_trace(
    session: Session,
    project: Project,
    runtime: GraderRuntime,
    trace: TraceSnapshot,
    *,
    purpose: str,
    job_id: str | None = None,
    audit_run_id: str | None = None,
    use_cache: bool = True,
    settings: Settings | None = None,
) -> GradingRun:
    settings = settings or get_settings()
    _assert_not_sealed(session, trace, purpose)
    case = render_trace(trace)
    key = cache_key_for(project, runtime.manifest, case)
    if use_cache:
        hit = _lookup_cache(session, key)
        if hit is not None:
            return _persist_hit(session, project, runtime, trace, key, hit, purpose=purpose, job_id=job_id,
                                audit_run_id=audit_run_id)
    result = runtime.grade_case(case, max_case_chars=settings.max_case_chars)
    return _persist_result(session, project, runtime, trace, key, result, purpose=purpose, job_id=job_id,
                           audit_run_id=audit_run_id)


def _record_purpose_exposure(
    session: Session, project: Project, trace: TraceSnapshot, purpose: str, job_id: str | None
) -> None:
    exposure_kind = PURPOSE_EXPOSURE.get(GradingPurpose(purpose))
    if exposure_kind is not None:
        record_exposure(session, project.id, trace.group_id, exposure_kind, job_id)


def grade_many(
    session: Session,
    project: Project,
    runtime: GraderRuntime,
    traces: Iterable[TraceSnapshot],
    *,
    purpose: str,
    job_id: str | None = None,
    audit_run_id: str | None = None,
    use_cache: bool = True,
    on_progress=None,
    settings: Settings | None = None,
    concurrency: int | None = None,
) -> list[GradingRun]:
    """Grade many traces; provider calls run in a thread pool, persistence stays sequential and ordered.

    Cache lookups happen before dispatch, results are written in input order, and the
    budget guard (thread-safe) still caps every call. Once the budget is spent later traces
    get cache hits (free) or a BUDGET_EXHAUSTED record without any model call or exposure.
    """
    settings = settings or get_settings()
    workers = max(1, int(concurrency if concurrency is not None else settings.grading_concurrency))
    traces = list(traces)
    runs: list[GradingRun] = []
    done = 0
    chunk_size = max(1, workers * 2)
    for start_idx in range(0, len(traces), chunk_size):
        chunk = traces[start_idx : start_idx + chunk_size]
        prepared: list[tuple[TraceSnapshot, Any, str, GradingRun | None]] = []
        for trace in chunk:
            _assert_not_sealed(session, trace, purpose)
            case = render_trace(trace)
            key = cache_key_for(project, runtime.manifest, case)
            hit = _lookup_cache(session, key) if use_cache else None
            prepared.append((trace, case, key, hit))
        misses = [i for i, (_, _, _, hit) in enumerate(prepared) if hit is None]
        results: dict[int, GradeResult] = {}
        if misses:
            if workers == 1 or len(misses) == 1:
                for i in misses:
                    results[i] = runtime.grade_case(prepared[i][1], max_case_chars=settings.max_case_chars)
            else:
                with ThreadPoolExecutor(max_workers=min(workers, len(misses))) as pool:
                    futures = {
                        pool.submit(runtime.grade_case, prepared[i][1], max_case_chars=settings.max_case_chars): i
                        for i in misses
                    }
                    for fut in as_completed(futures):
                        results[futures[fut]] = fut.result()
        for i, (trace, _case, key, hit) in enumerate(prepared):
            if hit is not None:
                run = _persist_hit(session, project, runtime, trace, key, hit, purpose=purpose, job_id=job_id,
                                   audit_run_id=audit_run_id)
            else:
                run = _persist_result(session, project, runtime, trace, key, results[i], purpose=purpose,
                                      job_id=job_id, audit_run_id=audit_run_id)
            runs.append(run)
            done += 1
            if on_progress is not None:
                on_progress(done)
    return runs


def latest_runs_for(session: Session, grader_id: str, trace_ids: list[str]) -> dict[str, GradingRun]:
    """Most recent GradingRun per trace for a grader (for display and exports)."""
    out: dict[str, GradingRun] = {}
    rows = session.scalars(
        select(GradingRun)
        .where(GradingRun.grader_id == grader_id, GradingRun.trace_id.in_(trace_ids))
        .order_by(GradingRun.created_at)
    )
    for r in rows:
        out[r.trace_id] = r
    return out


def new_guard(settings: Settings | None = None, **overrides: int) -> BudgetGuard:
    return new_budget_guard(settings, **overrides)
