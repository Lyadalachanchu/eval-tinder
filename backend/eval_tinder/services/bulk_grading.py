"""Bulk (shadow) grading with exactly one frozen grader manifest per job.

Rules
- A bulk job grades latest TRAIN and/or DEV traces only. Sealed or quarantined
  groups and AUDIT_RESERVE material are never bulk graded; explicit trace ids
  are validated against the same rule at enqueue time.
- The resolved trace ids and the manifest hash are frozen in the job payload:
  the worker grades exactly that list with exactly that manifest.
- Every prediction is a ``GradingRun`` stored alongside, never over, human
  labels. Predictions are tagged MACHINE, carry the grader version, and report
  their audit/automation status. No per-case confidence percentage exists.
- Budget exhaustion ends the job in BUDGET_EXHAUSTED with only the graded
  prefix persisted; cancellation ends it in CANCELLED. Neither claims completion.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.config import Settings, get_settings
from eval_tinder.db.enums import (
    AuditState,
    AutomationState,
    ExposureStatus,
    GradingPurpose,
    GradingStatus,
    JobKind,
    Partition,
)
from eval_tinder.db.models import (
    AuditRun,
    AutomationPolicy,
    GraderVersion,
    GradingRun,
    Job,
    PartitionAssignment,
    Project,
    TraceSnapshot,
)
from eval_tinder.domain.manifest import GraderManifest, pipeline_hash
from eval_tinder.llm.budget import BudgetExhausted
from eval_tinder.llm.factory import new_budget_guard
from eval_tinder.services import jobs as job_service
from eval_tinder.services.grading import GraderRuntime, grade_many
from eval_tinder.services.projects import get_grader
from eval_tinder.worker.main import JobCancelled

log = logging.getLogger(__name__)

BULK_PARTITIONS: tuple[str, ...] = (Partition.TRAIN.value, Partition.DEV.value)
HIDDEN_EXPOSURE: tuple[str, ...] = (ExposureStatus.SEALED.value, ExposureStatus.QUARANTINED.value)

# Heartbeat / progress / commit cadence (traces). Tests lower it to exercise cancellation.
PROGRESS_EVERY = 10

# Audit status vocabulary attached to every prediction. "AUDITED" means a
# COMPLETE or SPENT audit of this exact pipeline exists and its predeclared gate
# passed; it never means the prediction itself was checked by a human.
AUDIT_STATUS_UNAUDITED = "UNAUDITED"
AUDIT_STATUS_IN_PROGRESS = "AUDIT_IN_PROGRESS"
AUDIT_STATUS_AUDITED = "AUDITED"
AUDIT_STATUS_GATE_FAILED = "AUDIT_GATE_FAILED"
AUDIT_STATUS_COMPLETE = "AUDIT_COMPLETE"  # report exists; gate outcome not recorded
AUDIT_STATUS_INVALIDATED = "AUDIT_INVALIDATED"

RELEASED_AUDIT_STATES: tuple[str, ...] = (AuditState.COMPLETE.value, AuditState.SPENT.value)


class BulkGradingError(ValueError):
    pass


# ---------------------------------------------------------------- target resolution


def bulk_eligible_traces(session: Session, project: Project, partition: str | None) -> list[TraceSnapshot]:
    """Latest TRAIN/DEV traces whose groups are neither sealed nor quarantined (never AUDIT_RESERVE)."""
    if partition is not None and partition not in BULK_PARTITIONS:
        raise BulkGradingError(
            f"bulk grading is limited to {list(BULK_PARTITIONS)}; {partition!r} is sealed audit material"
        )
    partitions = [partition] if partition else list(BULK_PARTITIONS)
    stmt = (
        select(TraceSnapshot)
        .join(
            PartitionAssignment,
            (PartitionAssignment.project_id == TraceSnapshot.project_id)
            & (PartitionAssignment.group_id == TraceSnapshot.group_id),
        )
        .where(
            TraceSnapshot.project_id == project.id,
            TraceSnapshot.is_latest.is_(True),
            PartitionAssignment.partition.in_(partitions),
            PartitionAssignment.exposure_status.notin_(list(HIDDEN_EXPOSURE)),
        )
        .order_by(TraceSnapshot.group_id, TraceSnapshot.external_id, TraceSnapshot.id)
    )
    return list(session.scalars(stmt))


def _validate_explicit_trace_ids(
    session: Session, project: Project, trace_ids: list[str], partition: str | None
) -> list[TraceSnapshot]:
    if not trace_ids:
        raise BulkGradingError("trace_ids must not be empty")
    unique = list(dict.fromkeys(trace_ids))
    rows = {
        t.id: t
        for t in session.scalars(
            select(TraceSnapshot).where(TraceSnapshot.project_id == project.id, TraceSnapshot.id.in_(unique))
        )
    }
    missing = [tid for tid in unique if tid not in rows]
    if missing:
        raise BulkGradingError(f"unknown trace id(s) for this project: {missing[:5]}")
    assignments = {
        a.group_id: a
        for a in session.scalars(
            select(PartitionAssignment).where(
                PartitionAssignment.project_id == project.id,
                PartitionAssignment.group_id.in_({t.group_id for t in rows.values()}),
            )
        )
    }
    ordered: list[TraceSnapshot] = []
    for tid in unique:
        trace = rows[tid]
        assignment = assignments.get(trace.group_id)
        if assignment is None:
            raise BulkGradingError(f"trace {tid} has no partition assignment")
        if assignment.partition == Partition.AUDIT_RESERVE:
            raise BulkGradingError(f"trace {tid} is AUDIT_RESERVE material and cannot be bulk graded")
        if assignment.exposure_status in HIDDEN_EXPOSURE:
            raise BulkGradingError(
                f"trace {tid} belongs to a {assignment.exposure_status} group and cannot be bulk graded"
            )
        if partition is not None and assignment.partition != partition:
            raise BulkGradingError(f"trace {tid} is in {assignment.partition}, not {partition}")
        if not trace.is_latest:
            raise BulkGradingError(f"trace {tid} is not the latest revision of {trace.external_id!r}")
        ordered.append(trace)
    return ordered


def enqueue_bulk_grading(
    session: Session,
    project: Project,
    *,
    grader_id: str,
    partition: str | None,
    trace_ids: list[str] | None,
    idempotency_key: str,
    settings: Settings | None = None,
) -> Job:
    """Freeze one grader manifest and one explicit trace list into a BULK_GRADING job."""
    settings = settings or get_settings()
    existing = session.scalar(select(Job).where(Job.idempotency_key == idempotency_key))
    if existing is not None:
        return existing
    grader = get_grader(session, grader_id)
    if grader.project_id != project.id:
        raise BulkGradingError("grader belongs to another project")
    if partition is not None and partition not in BULK_PARTITIONS:
        raise BulkGradingError(
            f"bulk grading is limited to {list(BULK_PARTITIONS)}; {partition!r} is sealed audit material"
        )
    if trace_ids is not None:
        traces = _validate_explicit_trace_ids(session, project, list(trace_ids), partition)
    else:
        traces = bulk_eligible_traces(session, project, partition)
    manifest = GraderManifest.from_dict(grader.manifest)
    payload = {
        "project_id": project.id,
        "grader_id": grader.id,
        "manifest_hash": manifest.manifest_hash,
        "pipeline_hash": pipeline_hash(manifest),
        "policy_epoch": project.policy_epoch,
        "partition": partition,
        "trace_ids": [t.id for t in traces],
        "explicit_trace_ids": trace_ids is not None,
        "purpose": GradingPurpose.BULK.value,
    }
    return job_service.enqueue(
        session,
        kind=JobKind.BULK_GRADING,
        payload=payload,
        idempotency_key=idempotency_key,
        project_id=project.id,
        payload_ref=grader.id,
        max_attempts=1,
    )


# ---------------------------------------------------------------- worker handler


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _traces_in_order(session: Session, project_id: str, trace_ids: list[str]) -> list[TraceSnapshot]:
    rows = {
        t.id: t
        for t in session.scalars(
            select(TraceSnapshot).where(TraceSnapshot.project_id == project_id, TraceSnapshot.id.in_(trace_ids))
        )
    }
    missing = [tid for tid in trace_ids if tid not in rows]
    if missing:
        raise BulkGradingError(f"trace(s) vanished since enqueue: {missing[:5]}")
    return [rows[tid] for tid in trace_ids]


def _cancellable(traces: Iterable[TraceSnapshot], ctx) -> Iterator[TraceSnapshot]:
    """Yield traces one at a time, checking for cancellation between them."""
    for trace in traces:
        ctx.check_cancelled()
        yield trace


def _tally(counts: dict[str, int], run: GradingRun) -> None:
    counts["graded"] += 1
    if run.status == GradingStatus.OK:
        counts["ok"] += 1
    else:
        counts["errors"] += 1
    if run.verdict == "REVIEW":
        counts["review"] += 1
    if run.cache_hit_of is not None:
        counts["cache_hits"] += 1
    counts["verdicts"][run.verdict] = counts["verdicts"].get(run.verdict, 0) + 1


def bulk_grading_job_handler(job: Job, ctx) -> dict[str, Any]:
    payload = job.payload or {}
    trace_ids: list[str] = list(payload.get("trace_ids") or [])
    total = len(trace_ids)
    every = int(payload.get("progress_every") or PROGRESS_EVERY)
    settings = ctx.settings
    budget = new_budget_guard(settings)

    with ctx.session() as s:
        project = s.get(Project, payload["project_id"])
        grader = s.get(GraderVersion, payload["grader_id"])
        if project is None or grader is None:
            raise BulkGradingError("project or grader missing for bulk grading job")
        if grader.manifest_hash != payload.get("manifest_hash"):
            raise BulkGradingError("grader manifest changed since the job was enqueued; refusing to grade")
        runtime = GraderRuntime.build(project, grader, settings=settings, budget=budget)
        s.expunge(project)
        s.expunge(grader)

    counts: dict[str, Any] = {"graded": 0, "ok": 0, "review": 0, "errors": 0, "cache_hits": 0, "verdicts": {}}
    done = 0

    def publish_progress(**extra: Any) -> None:
        ctx.progress(done=done, total=total, counts=counts, **extra)

    if total == 0:
        publish_progress(complete=True)
        return {"complete": True, "total": 0, "partial": False, **counts, "usage": budget.snapshot()}

    for chunk in _chunks(trace_ids, max(1, every)):
        ctx.heartbeat(force=True)
        if ctx.cancel_requested:
            publish_progress(cancelled=True)
            raise JobCancelled(f"job {job.id} cancelled after {done}/{total} traces")
        exhausted: list[GradingRun] = []
        with ctx.session() as s:
            project = s.get(Project, payload["project_id"])
            traces = _traces_in_order(s, project.id, chunk)
            runs: list[GradingRun] = []
            cancelled: JobCancelled | None = None
            try:
                runs = grade_many(
                    s,
                    project,
                    runtime,
                    _cancellable(traces, ctx),
                    purpose=GradingPurpose.BULK,
                    job_id=job.id,
                    use_cache=True,
                    settings=settings,
                )
            except JobCancelled as e:
                cancelled = e
                # Runs graded before the cancellation are already flushed in this session/transaction.
                runs = list(
                    s.scalars(
                        select(GradingRun).where(GradingRun.job_id == job.id, GradingRun.trace_id.in_(chunk))
                    )
                )
            for run in runs:
                if run.status == GradingStatus.BUDGET_EXHAUSTED:
                    # No result was produced for this trace: keep only the graded prefix.
                    exhausted.append(run)
                    s.delete(run)
                    continue
                _tally(counts, run)
                done += 1
            s.commit()
        if cancelled is not None:
            publish_progress(cancelled=True)
            raise cancelled
        publish_progress(partial=bool(exhausted))
        if exhausted:
            raise BudgetExhausted(
                f"provider budget exhausted after {done}/{total} traces; "
                f"{total - done} trace(s) were not graded: {budget.snapshot()}"
            )
    publish_progress(complete=True)
    return {"complete": True, "partial": False, "total": total, **counts, "usage": budget.snapshot()}


# ---------------------------------------------------------------- statuses and predictions


def _gate_passed(report: dict[str, Any] | None) -> bool | None:
    """Best-effort read of a predeclared gate outcome from an audit report (None when not recorded)."""
    if not isinstance(report, dict):
        return None
    for key in ("gate_passed", "passed"):
        if isinstance(report.get(key), bool):
            return report[key]
    gate = report.get("gate")
    if isinstance(gate, dict):
        for key in ("passed", "gate_passed", "ok"):
            if isinstance(gate.get(key), bool):
                return gate[key]
    return None


def audits_for_pipeline(session: Session, grader: GraderVersion) -> list[AuditRun]:
    phash = pipeline_hash(GraderManifest.from_dict(grader.manifest))
    return list(
        session.scalars(
            select(AuditRun)
            .where(AuditRun.project_id == grader.project_id, AuditRun.pipeline_hash == phash)
            .order_by(AuditRun.created_at)
        )
    )


def audit_summaries(session: Session, grader: GraderVersion) -> list[dict[str, Any]]:
    """COMPLETE/SPENT audits of this grader's exact pipeline (never in-progress audit material)."""
    return [
        {
            "audit_id": a.id,
            "state": a.state,
            "pipeline_hash": a.pipeline_hash,
            "gate_passed": _gate_passed(a.report),
            "policy_epoch": a.policy_epoch,
            "planned_sample_size": len(a.locked_sample_ids or []),
            "completed_at": a.completed_at.isoformat() if a.completed_at else None,
        }
        for a in audits_for_pipeline(session, grader)
        if a.state in RELEASED_AUDIT_STATES
    ]


def audit_status_from_audits(audits: Iterable[Any]) -> str:
    """Collapse audit rows/summaries into one status. Prefers the strongest evidence available."""
    states: set[str] = set()
    for a in audits:
        state = a.get("state") if isinstance(a, dict) else getattr(a, "state", None)
        gate = a.get("gate_passed") if isinstance(a, dict) else _gate_passed(getattr(a, "report", None))
        if state in RELEASED_AUDIT_STATES:
            if gate is True:
                states.add(AUDIT_STATUS_AUDITED)
            elif gate is False:
                states.add(AUDIT_STATUS_GATE_FAILED)
            else:
                states.add(AUDIT_STATUS_COMPLETE)
        elif state in (AuditState.LOCKED, AuditState.IN_REVIEW):
            states.add(AUDIT_STATUS_IN_PROGRESS)
        elif state == AuditState.INVALIDATED:
            states.add(AUDIT_STATUS_INVALIDATED)
    for status in (
        AUDIT_STATUS_AUDITED,
        AUDIT_STATUS_GATE_FAILED,
        AUDIT_STATUS_COMPLETE,
        AUDIT_STATUS_IN_PROGRESS,
        AUDIT_STATUS_INVALIDATED,
    ):
        if status in states:
            return status
    return AUDIT_STATUS_UNAUDITED


def audit_status_for(session: Session, grader: GraderVersion) -> str:
    return audit_status_from_audits(audits_for_pipeline(session, grader))


def automation_policy_for(session: Session, project: Project, grader: GraderVersion) -> AutomationPolicy | None:
    """The project's automation policy, only when it was written for this grader's exact pipeline hash."""
    if not project.automation_policy_id:
        return None
    policy = session.get(AutomationPolicy, project.automation_policy_id)
    if policy is None:
        return None
    if policy.pipeline_hash != pipeline_hash(GraderManifest.from_dict(grader.manifest)):
        return None
    return policy


def automation_summary(session: Session, project: Project, grader: GraderVersion) -> dict[str, Any]:
    policy = automation_policy_for(session, project, grader)
    if policy is None:
        return {"state": AutomationState.DISABLED.value, "pipeline_hash": None, "policy_id": None}
    return {
        "state": policy.state,
        "policy_id": policy.id,
        "pipeline_hash": policy.pipeline_hash,
        "audit_id": policy.audit_id,
        "supported_scope": policy.supported_scope or {},
        "permitted_verdicts": policy.permitted_verdicts or [],
        "risk_targets": policy.risk_targets or {},
        "gate_result": policy.gate_result or {},
        "enabled_by": policy.enabled_by,
        "reason": policy.reason,
        "updated_at": policy.updated_at.isoformat() if policy.updated_at else None,
    }


def automation_status_for(session: Session, project: Project, grader: GraderVersion) -> str:
    policy = automation_policy_for(session, project, grader)
    return policy.state if policy is not None else AutomationState.DISABLED.value


def prediction_dict(
    run: GradingRun,
    trace: TraceSnapshot,
    *,
    manifest_hash: str,
    audit_status: str,
    automation_status: str,
) -> dict[str, Any]:
    """The public shape of one machine prediction. No confidence figure exists or is invented."""
    return {
        "trace_id": trace.id,
        "external_id": trace.external_id,
        "group_id": trace.group_id,
        "grader_id": run.grader_id,
        "manifest_hash": manifest_hash,
        "grading_run_id": run.id,
        "verdict": run.verdict,
        "status": run.status,
        "explanation": run.explanation,
        "evidence": run.evidence or [],
        "error": run.error,
        "purpose": run.purpose,
        "cache_hit_of": run.cache_hit_of,
        "kind": "MACHINE",
        "provisional": automation_status != AutomationState.ENABLED,
        "audit_status": audit_status,
        "automation_status": automation_status,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


def predictions_for(
    session: Session,
    project: Project,
    *,
    grader_id: str,
    trace_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Latest prediction per browsable trace for one grader version, tagged MACHINE.

    Only TRAIN/DEV traces in non-sealed, non-quarantined groups are listed; audit
    material never appears here, and AUDIT-purpose runs are never surfaced.
    """
    grader = get_grader(session, grader_id)
    if grader.project_id != project.id:
        raise BulkGradingError("grader belongs to another project")
    traces = bulk_eligible_traces(session, project, None)
    if trace_ids is not None:
        wanted = set(trace_ids)
        hidden = wanted - {t.id for t in traces}
        if hidden:
            raise BulkGradingError(
                f"{len(hidden)} requested trace(s) are not browsable: unknown, sealed, quarantined, or audit material"
            )
        traces = [t for t in traces if t.id in wanted]
    if not traces:
        return []
    runs: dict[str, GradingRun] = {}
    for r in session.scalars(
        select(GradingRun)
        .where(
            GradingRun.grader_id == grader.id,
            GradingRun.trace_id.in_([t.id for t in traces]),
            GradingRun.purpose != GradingPurpose.AUDIT,
        )
        .order_by(GradingRun.created_at, GradingRun.id)
    ):
        runs[r.trace_id] = r  # last one wins: the most recent non-audit run per trace
    audit_status = audit_status_for(session, grader)
    automation_status = automation_status_for(session, project, grader)
    out = []
    for t in traces:
        run = runs.get(t.id)
        if run is None:
            continue
        out.append(
            prediction_dict(
                run, t, manifest_hash=grader.manifest_hash, audit_status=audit_status,
                automation_status=automation_status,
            )
        )
    return out
