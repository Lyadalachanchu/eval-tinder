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
  prefix persisted (a trace that was never graded leaves no run and no exposure
  event); cancellation ends it in CANCELLED. Neither claims completion.

Status vocabulary attached to every prediction
- ``audit_status`` describes the grader's *pipeline* (not the single
  prediction): the latest released (COMPLETE/SPENT) audit of this exact
  pipeline hash decides it. AUDITED means that audit's predeclared gate passed.
- ``automation_status`` describes whether the project's explicit enablement
  applies to *this* prediction: ENABLED only when the policy is ENABLED for this
  exact pipeline hash, its audit belongs to the current policy epoch and is not
  invalidated, and the trace lies inside the policy's supported scope
  (PRODUCTION source, task types, time window). Otherwise DISABLED,
  INVALIDATED, or OUT_OF_SCOPE, with ``automation_reason`` saying why.
- ``provisional`` is False only for an ENABLED prediction whose status is OK
  and whose verdict is one of the policy's permitted verdicts. Everything else
  is provisional and needs a human.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    SourceType,
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
from eval_tinder.services.grading import GraderRuntime, grade_trace
from eval_tinder.services.projects import get_grader
from eval_tinder.worker.main import JobCancelled

log = logging.getLogger(__name__)

BULK_PARTITIONS: tuple[str, ...] = (Partition.TRAIN.value, Partition.DEV.value)
HIDDEN_EXPOSURE: tuple[str, ...] = (ExposureStatus.SEALED.value, ExposureStatus.QUARANTINED.value)

# Heartbeat / progress / commit cadence (traces). Tests lower it to exercise cancellation.
PROGRESS_EVERY = 10

# Audit status vocabulary attached to every prediction. "AUDITED" means the latest released
# (COMPLETE or SPENT) audit of this exact pipeline is COMPLETE and its predeclared gate passed;
# it never means the prediction itself was checked by a human.
AUDIT_STATUS_UNAUDITED = "UNAUDITED"
AUDIT_STATUS_IN_PROGRESS = "AUDIT_IN_PROGRESS"
AUDIT_STATUS_AUDITED = "AUDITED"
AUDIT_STATUS_GATE_FAILED = "AUDIT_GATE_FAILED"
AUDIT_STATUS_COMPLETE = "AUDIT_COMPLETE"  # report exists; gate outcome not recorded
AUDIT_STATUS_SPENT = "AUDIT_SPENT"  # historical only: the audit already influenced a revision
AUDIT_STATUS_INVALIDATED = "AUDIT_INVALIDATED"
AUDIT_STATUSES: tuple[str, ...] = (
    AUDIT_STATUS_UNAUDITED,
    AUDIT_STATUS_IN_PROGRESS,
    AUDIT_STATUS_AUDITED,
    AUDIT_STATUS_GATE_FAILED,
    AUDIT_STATUS_COMPLETE,
    AUDIT_STATUS_SPENT,
    AUDIT_STATUS_INVALIDATED,
)

# Per-prediction automation vocabulary: the three policy states plus OUT_OF_SCOPE, which marks a
# prediction of an ENABLED pipeline on a trace the policy's supported scope does not cover.
AUTOMATION_OUT_OF_SCOPE = "OUT_OF_SCOPE"
AUTOMATION_STATUSES: tuple[str, ...] = (
    AutomationState.DISABLED.value,
    AutomationState.ENABLED.value,
    AutomationState.INVALIDATED.value,
    AUTOMATION_OUT_OF_SCOPE,
)

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


def existing_job_for_key(session: Session, project: Project, idempotency_key: str, *, kind: str) -> Job | None:
    """Project- and kind-scoped idempotency lookup (raises JobError for a foreign key)."""
    from eval_tinder.services import jobs as job_service

    return job_service.find_existing(session, idempotency_key, project_id=project.id, kind=kind)

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
    existing = existing_job_for_key(session, project, idempotency_key, kind=JobKind.BULK_GRADING.value)
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


def _tally(counts: dict[str, Any], run: GradingRun) -> None:
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
    every = max(1, int(payload.get("progress_every") or PROGRESS_EVERY))
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

    for chunk in _chunks(trace_ids, every):
        ctx.heartbeat(force=True)
        if ctx.cancel_requested:
            publish_progress(cancelled=True)
            raise JobCancelled(f"job {job.id} cancelled after {done}/{total} traces")
        exhausted = False
        cancelled: JobCancelled | None = None
        with ctx.session() as s:
            project = s.get(Project, payload["project_id"])
            traces = _traces_in_order(s, project.id, chunk)
            for trace in traces:
                try:
                    ctx.check_cancelled()
                except JobCancelled as e:
                    cancelled = e
                    break
                # One savepoint per trace: a trace the budget refused produced no result, so neither its
                # run nor the exposure event ``grade_trace`` recorded for it may survive.
                savepoint = s.begin_nested()
                run = grade_trace(
                    s, project, runtime, trace, purpose=GradingPurpose.BULK, job_id=job.id, use_cache=True,
                    settings=settings,
                )
                if run.status == GradingStatus.BUDGET_EXHAUSTED:
                    savepoint.rollback()
                    exhausted = True
                    break
                savepoint.commit()
                _tally(counts, run)
                done += 1
            s.commit()  # the graded prefix of this chunk is durable even when we stop below
        if cancelled is not None:
            publish_progress(cancelled=True)
            raise cancelled
        if exhausted:
            publish_progress(partial=True)
            raise BudgetExhausted(
                f"provider budget exhausted after {done}/{total} traces; "
                f"{total - done} trace(s) were not graded: {budget.snapshot()}"
            )
        publish_progress(partial=False)
    publish_progress(complete=True)
    return {"complete": True, "partial": False, "total": total, **counts, "usage": budget.snapshot()}


# ---------------------------------------------------------------- audit status


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


def _gate_failed_checks(report: dict[str, Any] | None) -> list[str]:
    gate = report.get("gate") if isinstance(report, dict) else None
    failed = gate.get("failed") if isinstance(gate, dict) else None
    return [str(x) for x in failed] if isinstance(failed, list) else []


def audits_for_pipeline(session: Session, grader: GraderVersion) -> list[AuditRun]:
    """Every audit of this grader's exact pipeline hash, oldest first."""
    phash = pipeline_hash(GraderManifest.from_dict(grader.manifest))
    return list(
        session.scalars(
            select(AuditRun)
            .where(AuditRun.project_id == grader.project_id, AuditRun.pipeline_hash == phash)
            .order_by(AuditRun.created_at, AuditRun.id)
        )
    )


def audit_status_of(audit: Any) -> str:
    """The status one audit row (or summary dict) contributes; see ``audit_status_from_audits``."""
    if isinstance(audit, Mapping):
        state = audit.get("state")
        gate = audit.get("gate_passed")
        has_report = bool(audit.get("report_present", audit.get("gate_passed") is not None))
    else:
        state = getattr(audit, "state", None)
        report = getattr(audit, "report", None)
        gate = _gate_passed(report)
        has_report = report is not None
    if state == AuditState.SPENT:
        return AUDIT_STATUS_SPENT
    if state == AuditState.COMPLETE:
        if gate is True:
            return AUDIT_STATUS_AUDITED
        if gate is False:
            return AUDIT_STATUS_GATE_FAILED
        return AUDIT_STATUS_COMPLETE if has_report else AUDIT_STATUS_IN_PROGRESS
    if state in (AuditState.LOCKED, AuditState.IN_REVIEW):
        return AUDIT_STATUS_IN_PROGRESS
    if state == AuditState.INVALIDATED:
        return AUDIT_STATUS_INVALIDATED
    return AUDIT_STATUS_UNAUDITED


def audit_summary(audit: AuditRun) -> dict[str, Any]:
    return {
        "audit_id": audit.id,
        "state": audit.state,
        "status": audit_status_of(audit),
        "pipeline_hash": audit.pipeline_hash,
        "gate_passed": _gate_passed(audit.report),
        "gate_failed_checks": _gate_failed_checks(audit.report),
        "report_present": audit.report is not None,
        "report_version": audit.report_version,
        "policy_epoch": audit.policy_epoch,
        "planned_sample_size": len(audit.locked_sample_ids or []),
        "created_at": audit.created_at.isoformat() if audit.created_at else None,
        "completed_at": audit.completed_at.isoformat() if audit.completed_at else None,
    }


def audit_summaries(session: Session, grader: GraderVersion) -> list[dict[str, Any]]:
    """COMPLETE/SPENT audits of this grader's exact pipeline, oldest first (never in-progress material)."""
    return [audit_summary(a) for a in audits_for_pipeline(session, grader) if a.state in RELEASED_AUDIT_STATES]


def audit_status_from_audits(audits: Iterable[Any]) -> str:
    """Collapse audit rows/summaries (oldest first) into one status.

    The *latest released* (COMPLETE or SPENT) audit decides: a later failed audit is never hidden
    behind an earlier passing one, and a later passing audit supersedes an earlier failure. Only
    when no audit was released does an in-progress or invalidated audit show through.
    """
    latest_released: str | None = None
    fallback: str | None = None
    for a in audits:
        status = audit_status_of(a)
        if status in (AUDIT_STATUS_AUDITED, AUDIT_STATUS_GATE_FAILED, AUDIT_STATUS_COMPLETE, AUDIT_STATUS_SPENT):
            latest_released = status
        elif status == AUDIT_STATUS_IN_PROGRESS or (status == AUDIT_STATUS_INVALIDATED and fallback is None):
            fallback = status
    return latest_released or fallback or AUDIT_STATUS_UNAUDITED


def audit_status_for(session: Session, grader: GraderVersion) -> str:
    return audit_status_from_audits(audits_for_pipeline(session, grader))


# ---------------------------------------------------------------- automation status


def _parse_ts(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def scope_exclusion_reason(trace: TraceSnapshot | None, scope: Mapping[str, Any] | None) -> str | None:
    """Why ``trace`` lies outside a policy's supported scope, or None when it is covered.

    An audit establishes evidence only for its declared population and window, so the scope is
    checked strictly: a missing timestamp cannot establish membership in a declared window.
    """
    if trace is None:
        return "trace unavailable; scope membership cannot be established"
    scope = scope or {}
    expected_source = scope.get("source_type") or SourceType.PRODUCTION.value
    if trace.source_type != expected_source:
        return f"source_type {trace.source_type} is outside the supported scope ({expected_source} only)"
    task_types = scope.get("task_types")
    if task_types:
        task_type = (trace.metadata_ or {}).get("task_type")
        if str(task_type) not in {str(t) for t in task_types}:
            return f"task_type {task_type!r} is outside the supported task types {sorted(map(str, task_types))}"
    window = scope.get("time_window") or {}
    start = _parse_ts(window.get("start")) if isinstance(window, Mapping) else None
    end = _parse_ts(window.get("end")) if isinstance(window, Mapping) else None
    if start is not None or end is not None:
        if trace.timestamp is None:
            return "trace has no timestamp; membership in the supported time window cannot be established"
        stamp = trace.timestamp if trace.timestamp.tzinfo else trace.timestamp.replace(tzinfo=timezone.utc)
        if start is not None and stamp < start:
            return f"timestamp {stamp.isoformat()} precedes the supported window start {start.isoformat()}"
        if end is not None and stamp > end:
            return f"timestamp {stamp.isoformat()} is after the supported window end {end.isoformat()}"
    return None


@dataclass(frozen=True)
class AutomationContext:
    """The project's automation policy as it applies to one grader's pipeline, evaluated for display.

    ``stored_state`` is what the policy row says; ``effective_state`` additionally demands that the
    policy's audit still belongs to the current policy epoch and was not invalidated. A stale
    enablement is reported as INVALIDATED here even before the automation service records it.
    """

    stored_state: str = AutomationState.DISABLED.value
    effective_state: str = AutomationState.DISABLED.value
    reason: str | None = "no automation policy exists for this pipeline hash"
    policy_id: str | None = None
    pipeline_hash: str | None = None
    audit_id: str | None = None
    audit_policy_epoch: int | None = None
    supported_scope: dict[str, Any] = field(default_factory=dict)
    permitted_verdicts: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.effective_state == AutomationState.ENABLED

    def status_for(self, trace: TraceSnapshot | None, run: GradingRun | None = None) -> tuple[str, bool, str | None]:
        """``(automation_status, provisional, reason)`` for one prediction on ``trace``."""
        if not self.enabled:
            return self.effective_state, True, self.reason
        excluded = scope_exclusion_reason(trace, self.supported_scope)
        if excluded is not None:
            return AUTOMATION_OUT_OF_SCOPE, True, excluded
        if run is not None:
            if run.status != GradingStatus.OK:
                return (
                    AutomationState.ENABLED.value, True,
                    f"operational status {run.status} yields no automatic decision (effective REVIEW)",
                )
            if run.verdict not in self.permitted_verdicts:
                return (
                    AutomationState.ENABLED.value, True,
                    f"verdict {run.verdict} is not among the permitted automatic verdicts "
                    f"{list(self.permitted_verdicts)}",
                )
        return AutomationState.ENABLED.value, False, None


def automation_policy_for(session: Session, project: Project, grader: GraderVersion) -> AutomationPolicy | None:
    """The project's automation policy, only when it was written for this grader's exact pipeline hash."""
    if not project.automation_policy_id:
        return None
    policy = session.get(AutomationPolicy, project.automation_policy_id)
    if policy is None or policy.project_id != project.id:
        return None
    if policy.pipeline_hash != pipeline_hash(GraderManifest.from_dict(grader.manifest)):
        return None
    return policy


def automation_context(session: Session, project: Project, grader: GraderVersion) -> AutomationContext:
    policy = automation_policy_for(session, project, grader)
    if policy is None:
        return AutomationContext()
    audit = session.get(AuditRun, policy.audit_id) if policy.audit_id else None
    stored = policy.state
    effective = stored
    reason: str | None = policy.reason or None
    if stored == AutomationState.ENABLED:
        if audit is None or audit.project_id != project.id:
            effective, reason = AutomationState.INVALIDATED.value, "the enabling audit no longer exists"
        elif audit.pipeline_hash != policy.pipeline_hash:
            effective, reason = (
                AutomationState.INVALIDATED.value,
                "the enabling audit certified a different pipeline hash",
            )
        elif audit.policy_epoch != project.policy_epoch:
            effective, reason = (
                AutomationState.INVALIDATED.value,
                f"policy epoch changed from {audit.policy_epoch} (audited) to {project.policy_epoch}; "
                "enablement needs a new audit under the current policy",
            )
        elif audit.state == AuditState.INVALIDATED:
            effective, reason = (
                AutomationState.INVALIDATED.value,
                "the enabling audit's report was invalidated by a judgment correction",
            )
        else:
            reason = None
    elif stored == AutomationState.DISABLED:
        reason = reason or "automation explicitly disabled"
    return AutomationContext(
        stored_state=stored,
        effective_state=effective,
        reason=reason,
        policy_id=policy.id,
        pipeline_hash=policy.pipeline_hash,
        audit_id=policy.audit_id,
        audit_policy_epoch=audit.policy_epoch if audit is not None else None,
        supported_scope=dict(policy.supported_scope or {}),
        permitted_verdicts=tuple(str(v) for v in (policy.permitted_verdicts or [])),
    )


def automation_summary(session: Session, project: Project, grader: GraderVersion) -> dict[str, Any]:
    ctx = automation_context(session, project, grader)
    policy = session.get(AutomationPolicy, ctx.policy_id) if ctx.policy_id else None
    if policy is None:
        return {
            "state": AutomationState.DISABLED.value,
            "stored_state": AutomationState.DISABLED.value,
            "reason": ctx.reason,
            "pipeline_hash": None,
            "policy_id": None,
        }
    return {
        "state": ctx.effective_state,
        "stored_state": ctx.stored_state,
        "reason": ctx.reason,
        "policy_id": policy.id,
        "pipeline_hash": policy.pipeline_hash,
        "audit_id": policy.audit_id,
        "audit_policy_epoch": ctx.audit_policy_epoch,
        "project_policy_epoch": project.policy_epoch,
        "supported_scope": policy.supported_scope or {},
        "permitted_verdicts": policy.permitted_verdicts or [],
        "risk_targets": policy.risk_targets or {},
        "gate_result": policy.gate_result or {},
        "enabled_by": policy.enabled_by,
        "updated_at": policy.updated_at.isoformat() if policy.updated_at else None,
        "note": (
            "Enablement applies only to the exact frozen pipeline hash, the audited policy epoch, and traces "
            "inside supported_scope; predictions elsewhere are provisional."
        ),
    }


def automation_status_for(session: Session, project: Project, grader: GraderVersion) -> str:
    """The pipeline-level effective automation state (DISABLED, ENABLED, or INVALIDATED)."""
    return automation_context(session, project, grader).effective_state


# ---------------------------------------------------------------- predictions


def prediction_dict(
    run: GradingRun,
    trace: TraceSnapshot,
    *,
    manifest_hash: str,
    audit_status: str,
    automation: AutomationContext,
) -> dict[str, Any]:
    """The public shape of one machine prediction. No confidence figure exists or is invented."""
    automation_status, provisional, reason = automation.status_for(trace, run)
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
        "provisional": provisional,
        "audit_status": audit_status,
        "automation_status": automation_status,
        "automation_policy_state": automation.effective_state,
        "automation_reason": reason,
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
    # Traces with an open blind review request must not show any machine verdict (committee votes on
    # unjudged selected cases would otherwise be reconstructable before the expert judges them).
    from eval_tinder.services.review import open_request_trace_ids

    blind = open_request_trace_ids(session, project.id)
    traces = [t for t in traces if t.id not in blind]
    runs: dict[str, GradingRun] = {}
    for r in session.scalars(
        select(GradingRun)
        .where(
            GradingRun.grader_id == grader.id,
            GradingRun.trace_id.in_([t.id for t in traces]),
            GradingRun.purpose.in_([GradingPurpose.BULK, GradingPurpose.EXPERIMENT]),
        )
        .order_by(GradingRun.created_at, GradingRun.id)
    ):
        runs[r.trace_id] = r  # last one wins: the most recent bulk run per trace
    audit_status = audit_status_for(session, grader)
    automation = automation_context(session, project, grader)
    out = []
    for t in traces:
        run = runs.get(t.id)
        if run is None:
            continue
        out.append(
            prediction_dict(run, t, manifest_hash=grader.manifest_hash, audit_status=audit_status,
                            automation=automation)
        )
    return out
