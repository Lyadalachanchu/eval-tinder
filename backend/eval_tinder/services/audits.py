"""Independent production audits: lock a blind sample, grade it with a frozen pipeline, report evidence.

Invariants (plan sections 3, 4, 11, 15)
- Everything is frozen *before* any audit label exists: grader (pipeline hash),
  policy epoch, population, sampling plan, planned sample size and risk targets.
  No default acceptable error target is invented; the user supplies every one.
- The sample is a uniform random draw of one designated target response per
  UNTOUCHED production group in AUDIT_RESERVE. Synthetic material and groups that
  were ever inspected, probed, graded or exported are never eligible. If fewer
  groups are available than planned the caller must lower ``planned_n``
  explicitly; the sample is never shrunk silently.
- Sampled groups are sealed (``AUDIT_SEALED``): ordinary browsing, review
  batches, probes, pools, bulk grading and exports cannot touch them. Sealed
  groups never become UNTOUCHED again; releasing them marks them INSPECTED.
- The sample is fixed: skipped cases stay in the sample (they are re-offered,
  never replaced), a report is incomplete until every locked case has a
  judgment, and a correction invalidates the derived report and any dependent
  enablement while preserving the original report and the correction history.
- Machine predictions on the sample are ``GradingRun`` rows tagged with the
  audit; they never become human labels and are never shown through review
  endpoints. Zero denominators are ``NOT_ESTIMABLE`` (never 0% error).
- Once an optimization run is created after an audit completed, that audit is
  SPENT: it stays a historical report but cannot certify a revised grader.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.config import Settings, get_settings
from eval_tinder.db.enums import (
    AuditState,
    ExposureKind,
    ExposureStatus,
    GradingPurpose,
    GradingStatus,
    HumanVerdict,
    JobKind,
    MachineVerdict,
    Partition,
    ReviewPurpose,
    ReviewRequestState,
    SelectionCategory,
    SourceType,
)
from eval_tinder.db.models import (
    AuditRun,
    GradingRun,
    GraderVersion,
    HumanJudgment,
    Job,
    PartitionAssignment,
    Project,
    ReviewRequest,
    TraceSnapshot,
    UsageRecord,
)
from eval_tinder.domain.manifest import GraderManifest, pipeline_hash
from eval_tinder.domain.metrics import (
    NOT_ESTIMABLE,
    ConfusionTable,
    baselines,
    bonferroni_confidence,
    build_confusion,
    compute_metrics,
    error_upper_bound,
    sampling_design_supported,
)
from eval_tinder.ids import utcnow
from eval_tinder.services import jobs as job_service
from eval_tinder.services import review as review_service
from eval_tinder.services.grading import GraderRuntime, grade_many, new_guard
from eval_tinder.services.projects import get_grader

log = logging.getLogger(__name__)

REPORT_KIND = "AUDIT_EVIDENCE"
WEIGHTING = "group-weighted"
MISSING_PREDICTION_STATUS = "MISSING_PREDICTION"

UNRESOLVED_RULES = ("block", "count_as_error")
JOINT_ALLOCATIONS = ("bonferroni",)
_DT_MIN = datetime.min.replace(tzinfo=timezone.utc)

RISK_TARGET_KEYS = {
    "permitted_verdicts",
    "max_error_rate",
    "min_coverage",
    "confidence",
    "gate_false_pass_rate",
    "joint_allocation",
    "unresolved_automatic_rule",
}
REQUIRED_RISK_TARGETS = ("permitted_verdicts", "max_error_rate", "min_coverage", "confidence")
POPULATION_KEYS = {"source_type", "partition", "time_window", "task_types"}
SAMPLING_PLAN_KEYS = {"unit", "method", "independence_assumption_documented", "independence_note", "use_cache"}


class AuditError(ValueError):
    pass


# ----------------------------------------------------------------- validation


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_risk_targets(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize user-declared risk targets. Every gate parameter must be explicit; nothing is invented."""
    if not isinstance(raw, Mapping):
        raise AuditError(
            "risk_targets must be an object declaring permitted_verdicts, max_error_rate, min_coverage and "
            "confidence (no default acceptable error target is invented)"
        )
    problems: list[str] = []
    unknown = sorted(set(raw) - RISK_TARGET_KEYS)
    if unknown:
        problems.append(f"unknown risk target keys {unknown}; allowed: {sorted(RISK_TARGET_KEYS)}")
    missing = [k for k in REQUIRED_RISK_TARGETS if raw.get(k) is None]
    if missing:
        problems.append(
            f"missing risk targets {missing}: the user must declare permitted verdicts, maximum error rate, "
            "minimum coverage and confidence level before the audit; no default is invented"
        )
    permitted_raw = raw.get("permitted_verdicts")
    permitted: list[str] = []
    if permitted_raw is not None:
        if not isinstance(permitted_raw, (list, tuple)) or not permitted_raw:
            problems.append("permitted_verdicts must be a non-empty list drawn from ['PASS', 'FAIL']")
        else:
            bad = [v for v in permitted_raw if v not in ("PASS", "FAIL")]
            if bad:
                problems.append(f"permitted_verdicts may only contain PASS and FAIL, got {bad}")
            for v in permitted_raw:
                if v in ("PASS", "FAIL") and v not in permitted:
                    permitted.append(str(v))
    for key, low_ok in (("max_error_rate", False), ("confidence", False), ("min_coverage", True)):
        value = raw.get(key)
        if value is None:
            continue
        if not _is_number(value):
            problems.append(f"{key} must be a number")
        elif low_ok and not 0 <= value <= 1:
            problems.append(f"{key} must be within [0, 1], got {value}")
        elif not low_ok and not 0 < value < 1:
            problems.append(f"{key} must be strictly between 0 and 1, got {value}")
    gate_fp = raw.get("gate_false_pass_rate")
    if gate_fp is not None and (not _is_number(gate_fp) or not 0 < gate_fp < 1):
        problems.append(f"gate_false_pass_rate must be null or strictly between 0 and 1, got {gate_fp!r}")
    joint = raw.get("joint_allocation")
    if joint is not None and joint not in JOINT_ALLOCATIONS:
        problems.append(f"joint_allocation must be null or one of {list(JOINT_ALLOCATIONS)}, got {joint!r}")
    if gate_fp is not None and joint is None:
        problems.append(
            "joint_allocation ('bonferroni') is required when both the error-rate and the false-pass bounds are "
            "gated: several marginal bounds do not form a joint guarantee"
        )
    rule = raw.get("unresolved_automatic_rule", "block")
    if rule not in UNRESOLVED_RULES:
        problems.append(f"unresolved_automatic_rule must be one of {list(UNRESOLVED_RULES)}, got {rule!r}")
    if problems:
        raise AuditError("invalid risk targets: " + "; ".join(problems))
    return {
        "permitted_verdicts": permitted,
        "max_error_rate": float(raw["max_error_rate"]),
        "min_coverage": float(raw["min_coverage"]),
        "confidence": float(raw["confidence"]),
        "gate_false_pass_rate": None if gate_fp is None else float(gate_fp),
        "joint_allocation": joint,
        "unresolved_automatic_rule": rule,
    }


def parse_iso_datetime(value: Any, label: str) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as e:
            raise AuditError(f"{label} is not an ISO 8601 timestamp: {value!r}") from e
    else:
        raise AuditError(f"{label} must be an ISO 8601 string")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def validate_population(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize the declared population: production traces in AUDIT_RESERVE, optional window/task filter."""
    raw = raw or {}
    if not isinstance(raw, Mapping):
        raise AuditError("population must be an object")
    problems: list[str] = []
    unknown = sorted(set(raw) - POPULATION_KEYS)
    if unknown:
        problems.append(f"unknown population keys {unknown}; allowed: {sorted(POPULATION_KEYS)}")
    source = raw.get("source_type", SourceType.PRODUCTION.value)
    if source != SourceType.PRODUCTION.value:
        problems.append(f"source_type must be PRODUCTION (synthetic cases never enter an audit), got {source!r}")
    partition = raw.get("partition", Partition.AUDIT_RESERVE.value)
    if partition != Partition.AUDIT_RESERVE.value:
        problems.append(f"partition is fixed to AUDIT_RESERVE for audits, got {partition!r}")
    window = raw.get("time_window")
    start = end = None
    if window is not None:
        if not isinstance(window, Mapping) or set(window) - {"start", "end"}:
            problems.append("time_window must be an object with optional 'start' and 'end' ISO timestamps")
        else:
            try:
                start = parse_iso_datetime(window.get("start"), "time_window.start")
                end = parse_iso_datetime(window.get("end"), "time_window.end")
            except AuditError as e:
                problems.append(str(e))
            if start is not None and end is not None and start > end:
                problems.append("time_window.start must not be after time_window.end")
    task_types = raw.get("task_types")
    if task_types is not None:
        if not isinstance(task_types, (list, tuple)) or not task_types or not all(
            isinstance(t, str) and t for t in task_types
        ):
            problems.append("task_types must be null or a non-empty list of strings")
        else:
            task_types = sorted(set(task_types))
    if problems:
        raise AuditError("invalid population: " + "; ".join(problems))
    return {
        "source_type": SourceType.PRODUCTION.value,
        "partition": Partition.AUDIT_RESERVE.value,
        "time_window": {
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
        },
        "task_types": list(task_types) if task_types else None,
    }


def validate_sampling_plan(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AuditError("sampling_plan must be an object declaring unit, method and the independence assumption")
    problems: list[str] = []
    unknown = sorted(set(raw) - SAMPLING_PLAN_KEYS)
    if unknown:
        problems.append(f"unknown sampling plan keys {unknown}; allowed: {sorted(SAMPLING_PLAN_KEYS)}")
    if raw.get("unit") != "group":
        problems.append(f"unit must be 'group' (one designated target response per group), got {raw.get('unit')!r}")
    if raw.get("method") != "uniform_random":
        problems.append(f"method must be 'uniform_random', got {raw.get('method')!r}")
    documented = raw.get("independence_assumption_documented", False)
    if not isinstance(documented, bool):
        problems.append("independence_assumption_documented must be a boolean")
    note = raw.get("independence_note", "")
    if note is None:
        note = ""
    if not isinstance(note, str):
        problems.append("independence_note must be a string")
    elif documented is True and not note.strip():
        problems.append("independence_note must explain the independence assumption when it is declared documented")
    if raw.get("use_cache") not in (None, False):
        problems.append("use_cache must be false for audits: every audit prediction is a fresh provider call")
    if problems:
        raise AuditError("invalid sampling plan: " + "; ".join(problems))
    return {
        "unit": "group",
        "method": "uniform_random",
        "independence_assumption_documented": bool(documented),
        "independence_note": note.strip(),
        "use_cache": False,
    }


# ----------------------------------------------------------------- eligibility


def _in_population(trace: TraceSnapshot, population: Mapping[str, Any]) -> bool:
    if trace.source_type != SourceType.PRODUCTION.value:
        return False
    task_types = population.get("task_types")
    if task_types:
        if str((trace.metadata_ or {}).get("task_type")) not in set(task_types):
            return False
    window = population.get("time_window") or {}
    start = parse_iso_datetime(window.get("start"), "time_window.start")
    end = parse_iso_datetime(window.get("end"), "time_window.end")
    if start is not None or end is not None:
        if trace.timestamp is None:
            return False  # membership in a declared window cannot be established
        if start is not None and trace.timestamp < start:
            return False
        if end is not None and trace.timestamp > end:
            return False
    return True


def _graded_group_ids(session: Session, project_id: str) -> set[str]:
    rows = session.execute(
        select(TraceSnapshot.group_id)
        .join(GradingRun, GradingRun.trace_id == TraceSnapshot.id)
        .where(TraceSnapshot.project_id == project_id)
        .distinct()
    )
    return {r[0] for r in rows}


def eligible_audit_targets(session: Session, project: Project, population: Mapping[str, Any]) -> list[TraceSnapshot]:
    """One designated target response per eligible group, ordered by group id.

    Eligible groups are UNTOUCHED groups in AUDIT_RESERVE containing only PRODUCTION
    traces that were never inspected, probed, graded or exported. The target is the
    latest revision with the greatest ``(timestamp, external_id)`` among the group's
    traces matching the population filters.
    """
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
            PartitionAssignment.partition == Partition.AUDIT_RESERVE.value,
            PartitionAssignment.exposure_status == ExposureStatus.UNTOUCHED.value,
        )
        .order_by(TraceSnapshot.group_id, TraceSnapshot.external_id)
    )
    by_group: dict[str, list[TraceSnapshot]] = defaultdict(list)
    for trace in session.scalars(stmt):
        by_group[trace.group_id].append(trace)
    graded = _graded_group_ids(session, project.id)
    targets: list[TraceSnapshot] = []
    for group_id in sorted(by_group):
        members = by_group[group_id]
        if group_id in graded:
            continue
        if any(m.source_type != SourceType.PRODUCTION.value for m in members):
            continue  # a synthetic revision contaminates the whole group
        candidates = [m for m in members if _in_population(m, population)]
        if not candidates:
            continue
        targets.append(max(candidates, key=lambda m: (m.timestamp or _DT_MIN, m.external_id)))
    return targets


def _resolved_window(population: dict[str, Any], targets: list[TraceSnapshot]) -> dict[str, Any]:
    declared = population.get("time_window") or {"start": None, "end": None}
    stamps = [t.timestamp for t in targets if t.timestamp is not None]
    observed_start = min(stamps).isoformat() if stamps else None
    observed_end = max(stamps).isoformat() if stamps else None
    return {
        "start": declared.get("start") or observed_start,
        "end": declared.get("end") or observed_end,
        "declared": dict(declared),
        "observed": {"start": observed_start, "end": observed_end, "sampled_without_timestamp": len(targets) - len(stamps)},
    }


# ----------------------------------------------------------------- locking


def _find_by_idempotency_key(session: Session, project: Project, key: str) -> AuditRun | None:
    return session.scalar(
        select(AuditRun).where(
            AuditRun.project_id == project.id, AuditRun.sampling_plan["idempotency_key"].as_string() == key
        )
    )


def grading_job_key(audit_id: str) -> str:
    return f"audit-grading:{audit_id}"


def lock_audit(
    session: Session,
    project: Project,
    *,
    grader_id: str,
    planned_n: int,
    seed: int | None,
    population: Mapping[str, Any] | None,
    sampling_plan: Mapping[str, Any] | None,
    risk_targets: Mapping[str, Any] | None,
    idempotency_key: str,
    settings: Settings | None = None,
) -> tuple[AuditRun, Job]:
    """Freeze pipeline, scope, sample and targets; seal the sample; queue blind review and machine grading."""
    settings = settings or get_settings()
    if not idempotency_key:
        raise AuditError("idempotency_key is required")
    existing = _find_by_idempotency_key(session, project, idempotency_key)
    if existing is not None:
        job = session.get(Job, existing.grading_job_id) if existing.grading_job_id else None
        if job is None:
            raise AuditError(f"audit {existing.id} has no grading job")
        return existing, job
    if isinstance(planned_n, bool) or not isinstance(planned_n, int) or planned_n < 1:
        raise AuditError("planned_n must be a positive integer declared before review")
    targets_spec = validate_risk_targets(risk_targets)
    population_def = validate_population(population)
    plan = validate_sampling_plan(sampling_plan)
    grader = get_grader(session, grader_id)
    if grader.project_id != project.id:
        raise AuditError("grader belongs to another project")
    manifest = GraderManifest.from_dict(grader.manifest)
    frozen_hash = pipeline_hash(manifest)

    eligible = eligible_audit_targets(session, project, population_def)
    if len(eligible) < planned_n:
        raise AuditError(
            f"only {len(eligible)} eligible untouched production group(s) are available in AUDIT_RESERVE for the "
            f"declared population, but planned_n={planned_n}. Lower planned_n explicitly (or import more "
            "production groups); the sample is never shrunk silently."
        )
    if seed is None:
        seed = random.SystemRandom().randrange(1, 2**31 - 1)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AuditError("seed must be an integer")
    rng = random.Random(seed)
    sampled = rng.sample(eligible, planned_n)
    # Every eligible group is UNTOUCHED by construction; record the fact the intervals depend on.
    assignments = {
        a.group_id: a
        for a in session.scalars(
            select(PartitionAssignment).where(
                PartitionAssignment.project_id == project.id,
                PartitionAssignment.group_id.in_([t.group_id for t in sampled]),
            )
        )
    }
    fresh = all(assignments[t.group_id].exposure_status == ExposureStatus.UNTOUCHED.value for t in sampled)
    if not fresh:  # pragma: no cover - eligibility guarantees freshness; keep the invariant explicit
        raise AuditError("sampled groups are not all untouched; refusing to lock a non-fresh sample")

    audit = AuditRun(
        project_id=project.id,
        grader_id=grader.id,
        pipeline_hash=frozen_hash,
        policy_epoch=project.policy_epoch,
        population_definition={
            **population_def,
            "time_window": _resolved_window(population_def, sampled),
            "weighting": WEIGHTING,
            "sampling_unit": "one designated target response per independent production group",
            "eligible_group_count": len(eligible),
        },
        sampling_plan={
            **plan,
            "fresh_groups": True,
            "seed": seed,
            "planned_n": planned_n,
            "eligible_group_count": len(eligible),
            "idempotency_key": idempotency_key,
            "manifest_hash": grader.manifest_hash,
        },
        risk_targets=targets_spec,
        locked_sample_ids=sorted(t.id for t in sampled),
        state=AuditState.LOCKED.value,
        report=None,
        report_version=0,
        report_history=[],
        correction_history=[],
    )
    session.add(audit)
    session.flush()
    # Seal first, then create the blind requests: create_requests records AUDIT_REVIEW exposure, which must
    # not pre-empt the SEALED status.
    for trace in sampled:
        review_service.record_exposure(session, project.id, trace.group_id, ExposureKind.AUDIT_SEALED.value, audit.id)
    review_service.create_requests(
        session,
        project,
        sorted(sampled, key=lambda t: t.id),
        purpose=ReviewPurpose.AUDIT.value,
        category=SelectionCategory.AUDIT.value,
        batch_id=audit.id,
        audit_run_id=audit.id,
    )
    job = job_service.enqueue(
        session,
        kind=JobKind.AUDIT_GRADING.value,
        payload={"audit_id": audit.id, "project_id": project.id, "use_cache": False},
        idempotency_key=grading_job_key(audit.id),
        project_id=project.id,
        payload_ref=audit.id,
        max_attempts=3,
    )
    audit.grading_job_id = job.id
    audit.state = AuditState.IN_REVIEW.value
    session.flush()
    return audit, job


def get_audit(session: Session, audit_id: str) -> AuditRun:
    audit = session.get(AuditRun, audit_id)
    if audit is None:
        from eval_tinder.services.projects import NotFound

        raise NotFound(f"audit {audit_id} not found")
    return audit


def list_audits(session: Session, project_id: str) -> list[AuditRun]:
    return list(
        session.scalars(select(AuditRun).where(AuditRun.project_id == project_id).order_by(AuditRun.created_at))
    )


# ----------------------------------------------------------------- machine grading job


def audit_grading_job_handler(job: Job, ctx) -> dict[str, Any]:
    """Grade the locked sample with the frozen grader. Fresh calls only; predictions stay machine-only."""
    audit_id = job.payload["audit_id"]
    with ctx.session() as s:
        audit = s.get(AuditRun, audit_id)
        if audit is None:
            raise AuditError(f"audit {audit_id} not found")
        project = s.get(Project, audit.project_id)
        grader = s.get(GraderVersion, audit.grader_id)
        assert project is not None and grader is not None
        manifest = GraderManifest.from_dict(grader.manifest)
        if pipeline_hash(manifest) != audit.pipeline_hash:
            raise AuditError("grader pipeline hash no longer matches the frozen audit pipeline; refusing to grade")
        done = {
            r.trace_id
            for r in s.scalars(select(GradingRun).where(GradingRun.audit_run_id == audit.id))
            if r.status != GradingStatus.BUDGET_EXHAUSTED.value
        }
        pending_ids = [tid for tid in audit.locked_sample_ids if tid not in done]
        traces = [s.get(TraceSnapshot, tid) for tid in pending_ids]
        missing = [tid for tid, t in zip(pending_ids, traces, strict=True) if t is None]
        if missing:
            raise AuditError(f"locked traces missing from the database: {missing}")
        guard = new_guard(ctx.settings)
        runtime = GraderRuntime.build(project, grader, settings=ctx.settings, budget=guard)
        total = len(pending_ids)

        def on_progress(n: int) -> None:
            if n % 5 == 0 or n == total:
                s.commit()
                ctx.progress(graded=n, total=total, already_graded=len(done))
            ctx.check_cancelled()

        ctx.progress(graded=0, total=total, already_graded=len(done))
        try:
            grade_many(
                s,
                project,
                runtime,
                traces,
                purpose=GradingPurpose.AUDIT.value,
                job_id=ctx.job_id,
                audit_run_id=audit.id,
                use_cache=False,
                on_progress=on_progress,
                settings=ctx.settings,
            )
        finally:
            usage = guard.snapshot()
            s.add(
                UsageRecord(
                    project_id=project.id,
                    job_id=ctx.job_id,
                    role="grading",
                    model=manifest.model_config_.model,
                    calls=int(usage.get("calls") or 0),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                )
            )
            s.commit()
        if guard.exhausted:
            from eval_tinder.llm.budget import BudgetExhausted

            raise BudgetExhausted("audit grading budget exhausted; the sample is partially graded, not complete")
        return {
            "audit_id": audit.id,
            "graded": total,
            "already_graded": len(done),
            "use_cache": False,
            "usage": usage,
            "note": "Predictions are MACHINE outputs of the frozen grader; they are never human labels.",
        }


# ----------------------------------------------------------------- blind review


def next_audit_review(session: Session, audit: AuditRun, *, owner: str, lease_seconds: int) -> ReviewRequest | None:
    """Claim the next locked request. Blind; never draws a replacement for a skipped case."""
    if audit.state in (AuditState.SPENT.value,):
        return None
    stmt = (
        select(ReviewRequest)
        .where(
            ReviewRequest.audit_run_id == audit.id,
            ReviewRequest.state.in_([ReviewRequestState.OPEN.value, ReviewRequestState.LEASED.value]),
        )
        .order_by(ReviewRequest.created_at, ReviewRequest.id)
    )
    for req in session.scalars(stmt):
        try:
            return review_service.claim(session, req.id, owner=owner, lease_seconds=lease_seconds)
        except review_service.LeaseConflict:
            continue
    # The sample is fixed: a skipped case comes back until it is judged or explicitly marked CANNOT_JUDGE.
    skipped = list(
        session.scalars(
            select(ReviewRequest)
            .where(ReviewRequest.audit_run_id == audit.id, ReviewRequest.state == ReviewRequestState.SKIPPED.value)
            .order_by(ReviewRequest.created_at, ReviewRequest.id)
        )
    )
    for req in skipped:
        req.state = ReviewRequestState.OPEN.value
        session.flush()
        try:
            return review_service.claim(session, req.id, owner=owner, lease_seconds=lease_seconds)
        except review_service.LeaseConflict:
            continue
    return None


# ----------------------------------------------------------------- report


@dataclass
class _CaseRow:
    trace_id: str
    human: HumanJudgment | None
    machine: GradingRun | None
    request: ReviewRequest | None

    @property
    def machine_status(self) -> str:
        return self.machine.status if self.machine is not None else MISSING_PREDICTION_STATUS

    @property
    def machine_vote(self) -> str:
        """Effective machine verdict: only OK statuses vote; everything else is REVIEW."""
        if self.machine is None or self.machine.status != GradingStatus.OK.value:
            return MachineVerdict.REVIEW.value
        return self.machine.verdict


def _collect_rows(session: Session, audit: AuditRun) -> list[_CaseRow]:
    locked = list(audit.locked_sample_ids or [])
    if not locked:
        return []
    requests: dict[str, ReviewRequest] = {}
    for req in session.scalars(
        select(ReviewRequest).where(ReviewRequest.audit_run_id == audit.id).order_by(ReviewRequest.created_at)
    ):
        requests[req.trace_id] = req
    judgments: dict[str, HumanJudgment] = {}
    for j in session.scalars(
        select(HumanJudgment).where(
            HumanJudgment.trace_id.in_(locked),
            HumanJudgment.policy_epoch == audit.policy_epoch,
            HumanJudgment.superseded_by_id.is_(None),
        )
    ):
        judgments[j.trace_id] = j
    machines: dict[str, GradingRun] = {}
    for run in session.scalars(
        select(GradingRun).where(GradingRun.audit_run_id == audit.id).order_by(GradingRun.created_at, GradingRun.id)
    ):
        machines[run.trace_id] = run  # latest wins
    return [_CaseRow(tid, judgments.get(tid), machines.get(tid), requests.get(tid)) for tid in sorted(locked)]


def _bound(k: int, n: int, confidence: float, *, supported: bool) -> float | str | None:
    if not supported:
        return None
    if n == 0:
        return NOT_ESTIMABLE
    return error_upper_bound(k, n, confidence)


def _report_body(session: Session, audit: AuditRun) -> dict[str, Any]:
    rows = _collect_rows(session, audit)
    targets = audit.risk_targets or {}
    plan = audit.sampling_plan or {}
    confusion_rows: list[tuple[str, str, str]] = []
    human_unresolved: list[dict[str, Any]] = []
    unresolved_auto: list[str] = []
    operational: list[dict[str, Any]] = []
    judged = skipped = pending = 0
    for row in rows:
        if row.machine is None or row.machine.status != GradingStatus.OK.value:
            operational.append(
                {
                    "trace_id": row.trace_id,
                    "status": row.machine_status,
                    "error": row.machine.error if row.machine is not None else "no prediction recorded for this audit",
                    "effective_verdict": MachineVerdict.REVIEW.value,
                }
            )
        if row.human is None:
            state = row.request.state if row.request is not None else None
            if state == ReviewRequestState.SKIPPED.value:
                skipped += 1
                reason = "skipped by the reviewer; no judgment yet (the case stays in the fixed sample, no replacement)"
            else:
                pending += 1
                reason = "no judgment"
            human_unresolved.append(
                {"trace_id": row.trace_id, "kind": "NO_JUDGMENT", "reason": reason, "request_state": state}
            )
            continue
        judged += 1
        confusion_rows.append((row.human.verdict, row.machine_vote, row.machine_status))
        if row.human.verdict == HumanVerdict.CANNOT_JUDGE.value:
            entry = {
                "trace_id": row.trace_id,
                "kind": "CANNOT_JUDGE",
                "reason": row.human.cannot_judge_reason,
                "judgment_id": row.human.id,
                "machine_verdict": row.machine_vote,
            }
            human_unresolved.append(entry)
            if row.machine_vote in (MachineVerdict.PASS.value, MachineVerdict.FAIL.value):
                unresolved_auto.append(row.trace_id)
    table = build_confusion(confusion_rows)
    metrics = compute_metrics(table)
    locked_n = len(rows)
    complete = locked_n > 0 and judged == locked_n

    supported, design_reason = sampling_design_supported(plan)
    confidence = float(targets["confidence"])
    gate_fp = targets.get("gate_false_pass_rate")
    n_bounds = 2 if gate_fp is not None else 1
    per_bound = bonferroni_confidence(confidence, 2) if gate_fp is not None else confidence
    rule = targets.get("unresolved_automatic_rule", "block")
    counted_as_errors = len(unresolved_auto) if rule == "count_as_error" else 0
    err_k = table.pass_fail + table.fail_pass + counted_as_errors
    err_n = table.machine_binary_on_determinate + counted_as_errors
    fp_k = table.fail_pass
    fp_n = table.pass_pass + table.fail_pass
    intervals = {
        "supported": supported,
        "reason": design_reason,
        "method": "one-sided exact binomial (Clopper-Pearson) upper bound",
        "confidence": confidence,
        "n_bounds": n_bounds,
        "joint_allocation": targets.get("joint_allocation"),
        "per_bound_confidence": per_bound,
        "automatic_error_rate_upper": _bound(err_k, err_n, per_bound, supported=supported),
        "automatic_error_rate_k": err_k,
        "automatic_error_rate_n": err_n,
        "false_pass_rate_upper": _bound(fp_k, fp_n, per_bound, supported=supported),
        "false_pass_k": fp_k,
        "false_pass_n": fp_n,
        "unresolved_counted_as_errors": counted_as_errors,
        "note": (
            "Bounds hold only under the declared sampling design (independent uniform draw of fresh groups); "
            "unsupported designs receive descriptive metrics only."
        ),
    }
    notes = [
        "Development agreement is not audit evidence; every number here comes from the locked, blind audit sample.",
        "Intervals assume independent uniform group sampling as declared; genuine independence is a documented "
        "assumption, not something unique identifiers prove.",
        f"Metrics are {WEIGHTING}: one designated target response per sampled group, not all-output-weighted.",
        "Zero denominators are NOT_ESTIMABLE; they never mean 0% error or 100% reliability.",
        "Machine predictions are MACHINE outputs of the frozen grader and never become human labels.",
    ]
    if counted_as_errors:
        notes.append(
            f"{counted_as_errors} unresolved automatic decision(s) (human CANNOT_JUDGE, machine PASS/FAIL) are counted "
            "as errors in the automatic error bound under unresolved_automatic_rule='count_as_error'."
        )
    if operational:
        notes.append(
            f"{len(operational)} operational failure(s) are reported separately and treated as REVIEW for coverage."
        )
    if not supported:
        notes.append("Sampling design unsupported: descriptive metrics only; statistical enablement is disabled.")
    population = audit.population_definition or {}
    return {
        "kind": REPORT_KIND,
        "audit_id": audit.id,
        "grader_id": audit.grader_id,
        "complete": complete,
        "planned_n": int((plan or {}).get("planned_n") or locked_n),
        "locked_n": locked_n,
        "judged_n": judged,
        "skipped_n": skipped,
        "pending_n": pending,
        "counts": metrics["counts"],
        "table": table.as_table(),
        "metrics": {k: v for k, v in metrics.items() if k != "counts"},
        "baselines": baselines(table),
        "human_unresolved": human_unresolved,
        "unresolved_automatic_decisions": unresolved_auto,
        "operational_failures": operational,
        "intervals": intervals,
        "scope": {
            "population": population,
            "weighting": WEIGHTING,
            "time_window": population.get("time_window"),
            "pipeline_hash": audit.pipeline_hash,
            "policy_epoch": audit.policy_epoch,
            "grader_id": audit.grader_id,
            "sampling_plan": {k: v for k, v in plan.items() if k != "idempotency_key"},
        },
        "risk_targets": dict(targets),
        "notes": notes,
        "computed_at": utcnow().isoformat(),
        "_table": table,
    }


def _gate(body: dict[str, Any], audit: AuditRun, state: str) -> dict[str, Any]:
    targets = audit.risk_targets or {}
    table: ConfusionTable = body["_table"]
    intervals = body["intervals"]
    metrics = body["metrics"]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "audit_complete",
        body["complete"],
        f"{body['judged_n']}/{body['locked_n']} locked cases judged; {body['skipped_n']} skipped, {body['pending_n']} pending",
    )
    check(
        "no_pending_correction",
        state != AuditState.INVALIDATED.value,
        "a judgment correction invalidated the report; recompute to derive a new version"
        if state == AuditState.INVALIDATED.value
        else "no correction pending",
    )
    check(
        "audit_not_spent",
        state != AuditState.SPENT.value,
        "this audit is SPENT: its results already influenced a revision; it remains a historical report only"
        if state == AuditState.SPENT.value
        else "audit not spent",
    )
    check("sampling_design_supported", intervals["supported"], intervals["reason"])
    gate_fp = targets.get("gate_false_pass_rate")
    denominators = {
        "automatic_error_rate": intervals["automatic_error_rate_n"],
        "automatic_coverage_determinate": table.human_determinate,
    }
    if gate_fp is not None:
        denominators["false_pass_rate"] = intervals["false_pass_n"]
    empty = sorted(k for k, n in denominators.items() if n == 0)
    check(
        "denominators_estimable",
        not empty,
        f"zero denominators (NOT_ESTIMABLE): {empty}" if empty else f"denominators {denominators}",
    )
    max_err = targets.get("max_error_rate")
    upper = intervals["automatic_error_rate_upper"]
    check(
        "automatic_error_rate_bound",
        isinstance(upper, float) and max_err is not None and upper <= max_err,
        f"upper bound {upper!r} vs max_error_rate {max_err} (k={intervals['automatic_error_rate_k']}, "
        f"n={intervals['automatic_error_rate_n']}, per-bound confidence {intervals['per_bound_confidence']})",
    )
    if gate_fp is not None:
        fp_upper = intervals["false_pass_rate_upper"]
        check(
            "false_pass_rate_bound",
            isinstance(fp_upper, float) and fp_upper <= gate_fp,
            f"upper bound {fp_upper!r} vs gate_false_pass_rate {gate_fp} (k={intervals['false_pass_k']}, "
            f"n={intervals['false_pass_n']}, per-bound confidence {intervals['per_bound_confidence']})",
        )
    coverage = metrics["automatic_coverage_determinate"]["value"]
    min_cov = targets.get("min_coverage")
    check(
        "automatic_coverage",
        isinstance(coverage, float) and min_cov is not None and coverage >= min_cov,
        f"automatic coverage on human-determinate cases {coverage!r} vs min_coverage {min_cov}",
    )
    rule = targets.get("unresolved_automatic_rule", "block")
    n_unresolved = len(body["unresolved_automatic_decisions"])
    if rule == "block":
        check(
            "unresolved_automatic_decisions",
            n_unresolved == 0,
            f"{n_unresolved} unresolved automatic decision(s); rule 'block' requires 0",
        )
    else:
        check(
            "unresolved_automatic_decisions",
            True,
            f"{n_unresolved} unresolved automatic decision(s) counted as errors under rule 'count_as_error'",
        )
    passed = all(c["passed"] for c in checks)
    report = {k: v for k, v in body.items() if k != "_table"}
    report["state"] = state
    report["gate"] = {
        "passed": passed,
        "checks": checks,
        "failed": [c["name"] for c in checks if not c["passed"]],
        "note": "The gate is the predeclared product rule for explicit enablement, not a significance test.",
    }
    return report


def compute_report(session: Session, audit: AuditRun, *, effective_state: str | None = None) -> dict[str, Any]:
    """Derive the evidence report from stored judgments and audit predictions. Persists nothing."""
    return _gate(_report_body(session, audit), audit, effective_state or audit.state)


def _next_state(current: str, complete: bool) -> str:
    if current == AuditState.SPENT.value:
        return current
    return AuditState.COMPLETE.value if complete else AuditState.IN_REVIEW.value


def persist_report(session: Session, audit: AuditRun) -> AuditRun:
    """Store a new report version, preserving the previous one in ``report_history``."""
    body = _report_body(session, audit)
    new_state = _next_state(audit.state, body["complete"])
    report = _gate(body, audit, new_state)
    now = utcnow()
    history = list(audit.report_history or [])
    if audit.report is not None:
        history.append(
            {
                "event": "report_superseded",
                "report_version": audit.report_version,
                "state": audit.state,
                "at": now.isoformat(),
                "report": audit.report,
            }
        )
    audit.report_version = int(audit.report_version or 0) + 1
    report["report_version"] = audit.report_version
    corrections = list(audit.correction_history or [])
    for entry in corrections:
        if entry.get("recomputed_in_report_version") is None:
            entry["recomputed_in_report_version"] = audit.report_version
    audit.correction_history = corrections
    audit.report_history = history
    audit.report = report
    previous_state = audit.state
    audit.state = new_state
    if new_state == AuditState.COMPLETE.value and (previous_state != AuditState.COMPLETE.value or audit.completed_at is None):
        audit.completed_at = now
    session.flush()
    return audit


def report_history_versions(audit: AuditRun) -> list[int]:
    return [int(e["report_version"]) for e in (audit.report_history or []) if "report_version" in e]


# ----------------------------------------------------------------- corrections, spending, release


def correct_audit_judgment(
    session: Session,
    audit: AuditRun,
    judgment_id: str,
    *,
    verdict: str,
    explanation: str = "",
    cannot_judge_reason: str | None = None,
    reviewer_id: str,
    idempotency_key: str,
) -> HumanJudgment:
    """Append a superseding judgment; the derived report and dependent enablement are invalidated."""
    prior = session.get(HumanJudgment, judgment_id)
    if prior is None:
        raise AuditError(f"judgment {judgment_id} not found")
    if prior.trace_id not in set(audit.locked_sample_ids or []) or prior.policy_epoch != audit.policy_epoch:
        raise AuditError("judgment does not belong to this audit's locked sample and policy epoch")
    replay = session.scalar(
        select(HumanJudgment).where(
            HumanJudgment.project_id == prior.project_id, HumanJudgment.idempotency_key == idempotency_key
        )
    )
    if replay is not None:
        return replay
    new = review_service.correct_judgment(
        session,
        judgment_id,
        verdict=verdict,
        explanation=explanation,
        cannot_judge_reason=cannot_judge_reason,
        reviewer_id=reviewer_id,
        idempotency_key=idempotency_key,
    )
    corrections = list(audit.correction_history or [])
    corrections.append(
        {
            "judgment_id": judgment_id,
            "new_judgment_id": new.id,
            "at": utcnow().isoformat(),
            "previous_report_version": audit.report_version,
            "previous_state": audit.state,
            "reviewer_id": reviewer_id,
            "previous_verdict": prior.verdict,
            "verdict": new.verdict,
            "recomputed_in_report_version": None,
        }
    )
    audit.correction_history = corrections
    if audit.state != AuditState.SPENT.value:
        audit.state = AuditState.INVALIDATED.value
    session.flush()
    from eval_tinder.services import automation as automation_service

    automation_service.invalidate_for_audit(
        session, audit, reason=f"audit judgment {judgment_id} corrected (new judgment {new.id}); report invalidated"
    )
    return new


def mark_spent(session: Session, audit: AuditRun, *, reason: str) -> AuditRun:
    """A spent audit remains a historical report but can no longer certify anything."""
    if audit.state == AuditState.SPENT.value:
        return audit
    history = list(audit.report_history or [])
    history.append(
        {"event": "spent", "reason": reason, "at": utcnow().isoformat(), "previous_state": audit.state}
    )
    audit.report_history = history
    audit.state = AuditState.SPENT.value
    session.flush()
    if audit.report is not None:
        persist_report(session, audit)  # stored gate now fails "audit_not_spent"; the old report stays in history
    return audit


def mark_completed_audits_spent(session: Session, project: Project, *, reason: str) -> list[AuditRun]:
    spent = []
    for audit in list_audits(session, project.id):
        if audit.state == AuditState.COMPLETE.value:
            mark_spent(session, audit, reason=reason)
            spent.append(audit)
    return spent


def release_audit_groups(session: Session, audit: AuditRun) -> AuditRun:
    """Unseal the sampled groups. They were inspected: SEALED becomes INSPECTED, never UNTOUCHED."""
    if audit.state in (AuditState.LOCKED.value, AuditState.IN_REVIEW.value):
        raise AuditError("audit material stays sealed while the audit is in review; complete or spend it first")
    traces = list(session.scalars(select(TraceSnapshot).where(TraceSnapshot.id.in_(audit.locked_sample_ids or []))))
    for trace in traces:
        review_service.record_exposure(
            session, audit.project_id, trace.group_id, ExposureKind.AUDIT_RELEASED.value, audit.id
        )
        assignment = session.scalar(
            select(PartitionAssignment).where(
                PartitionAssignment.project_id == audit.project_id, PartitionAssignment.group_id == trace.group_id
            )
        )
        if assignment is not None and assignment.exposure_status == ExposureStatus.SEALED.value:
            assignment.exposure_status = ExposureStatus.INSPECTED.value
    history = list(audit.report_history or [])
    history.append({"event": "released", "at": utcnow().isoformat(), "groups": len({t.group_id for t in traces})})
    audit.report_history = history
    session.flush()
    return audit


def judged_count(session: Session, audit: AuditRun) -> tuple[int, int]:
    """``(judged, unresolved)`` where unresolved counts missing judgments plus CANNOT_JUDGE ones."""
    locked = list(audit.locked_sample_ids or [])
    if not locked:
        return 0, 0
    judgments = list(
        session.scalars(
            select(HumanJudgment).where(
                HumanJudgment.trace_id.in_(locked),
                HumanJudgment.policy_epoch == audit.policy_epoch,
                HumanJudgment.superseded_by_id.is_(None),
            )
        )
    )
    judged = len({j.trace_id for j in judgments})
    cannot = sum(1 for j in judgments if j.verdict == HumanVerdict.CANNOT_JUDGE.value)
    return judged, (len(locked) - judged) + cannot
