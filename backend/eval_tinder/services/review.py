"""Human review: batches, leases, blind idempotent judgments, exposure history.

Invariants
- SKIP creates no judgment. CANNOT_JUDGE requires a category and is never FAIL.
- One active judgment per trace and policy epoch; corrections append a
  superseding judgment and never overwrite the earlier one.
- Selection reasons and candidate predictions stay hidden until the expert has
  judged; DEV and AUDIT reviews stay blind afterwards too.
- Audit material is sealed: it is never offered through ordinary batches.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from eval_tinder.db.enums import (
    CannotJudgeReason,
    ExposureKind,
    ExposureStatus,
    HumanVerdict,
    Partition,
    ReviewPurpose,
    ReviewRequestState,
    SelectionCategory,
)
from eval_tinder.db.models import (
    ExposureEvent,
    HumanJudgment,
    PartitionAssignment,
    Project,
    ReviewRequest,
    TraceSnapshot,
)
from eval_tinder.domain.rendering import estimate_reading_length
from eval_tinder.ids import new_id, utcnow

PURPOSE_TO_PARTITION = {
    ReviewPurpose.TRAIN: Partition.TRAIN,
    ReviewPurpose.DEV: Partition.DEV,
    ReviewPurpose.AUDIT: Partition.AUDIT_RESERVE,
}
PURPOSE_TO_EXPOSURE = {
    ReviewPurpose.TRAIN: ExposureKind.TRAIN_REVIEW,
    ReviewPurpose.DEV: ExposureKind.DEV_REVIEW,
    ReviewPurpose.AUDIT: ExposureKind.AUDIT_REVIEW,
}


class ReviewError(ValueError):
    pass


class LeaseConflict(ReviewError):
    pass


class StaleSnapshot(ReviewError):
    pass


# ---------------------------------------------------------------- exposure


def record_exposure(session: Session, project_id: str, group_id: str, kind: str, reference_id: str | None) -> None:
    session.add(ExposureEvent(project_id=project_id, group_id=group_id, kind=kind, reference_id=reference_id))
    assignment = session.scalar(
        select(PartitionAssignment).where(
            PartitionAssignment.project_id == project_id, PartitionAssignment.group_id == group_id
        )
    )
    if assignment is not None and assignment.exposure_status == ExposureStatus.UNTOUCHED:
        if kind == ExposureKind.AUDIT_SEALED:
            assignment.exposure_status = ExposureStatus.SEALED
        elif kind not in (ExposureKind.AUDIT_RELEASED,):
            assignment.exposure_status = ExposureStatus.INSPECTED
    if assignment is not None and kind == ExposureKind.QUARANTINED:
        assignment.exposure_status = ExposureStatus.QUARANTINED
    session.flush()


def quarantine_group(session: Session, project_id: str, group_id: str, *, reason: str) -> None:
    record_exposure(session, project_id, group_id, ExposureKind.QUARANTINED, None)
    session.flush()


def sealed_group_ids(session: Session, project_id: str) -> set[str]:
    rows = session.scalars(
        select(PartitionAssignment.group_id).where(
            PartitionAssignment.project_id == project_id,
            PartitionAssignment.exposure_status.in_([ExposureStatus.SEALED, ExposureStatus.QUARANTINED]),
        )
    )
    return set(rows)


# ---------------------------------------------------------------- judgments


def active_judgment_for(session: Session, trace_id: str, policy_epoch: int) -> HumanJudgment | None:
    return session.scalar(
        select(HumanJudgment).where(
            HumanJudgment.trace_id == trace_id,
            HumanJudgment.policy_epoch == policy_epoch,
            HumanJudgment.superseded_by_id.is_(None),
        )
    )


def active_judgments(session: Session, project: Project, partition: str | None = None) -> list[tuple[TraceSnapshot, HumanJudgment]]:
    stmt = (
        select(TraceSnapshot, HumanJudgment)
        .join(HumanJudgment, HumanJudgment.trace_id == TraceSnapshot.id)
        .where(
            HumanJudgment.project_id == project.id,
            HumanJudgment.policy_epoch == project.policy_epoch,
            HumanJudgment.superseded_by_id.is_(None),
        )
        .order_by(TraceSnapshot.id)
    )
    if partition is not None:
        stmt = stmt.join(
            PartitionAssignment,
            (PartitionAssignment.project_id == TraceSnapshot.project_id)
            & (PartitionAssignment.group_id == TraceSnapshot.group_id),
        ).where(PartitionAssignment.partition == partition)
    return [(t, j) for t, j in session.execute(stmt).all()]


def resolved_label_counts(session: Session, project: Project) -> dict[str, dict[str, int]]:
    """Counts of active PASS/FAIL/CANNOT_JUDGE labels per partition at the current epoch."""
    out: dict[str, dict[str, int]] = {p.value: {"PASS": 0, "FAIL": 0, "CANNOT_JUDGE": 0, "resolved": 0} for p in Partition}
    for p in Partition:
        for _, j in active_judgments(session, project, p.value):
            out[p.value][j.verdict] += 1
            if j.verdict in (HumanVerdict.PASS, HumanVerdict.FAIL):
                out[p.value]["resolved"] += 1
    return out


def labeled_group_ids(session: Session, project: Project) -> set[str]:
    rows = session.execute(
        select(TraceSnapshot.group_id)
        .join(HumanJudgment, HumanJudgment.trace_id == TraceSnapshot.id)
        .where(HumanJudgment.project_id == project.id, HumanJudgment.policy_epoch == project.policy_epoch)
    )
    return {r[0] for r in rows}


def open_request_trace_ids(session: Session, project_id: str) -> set[str]:
    rows = session.scalars(
        select(ReviewRequest.trace_id).where(
            ReviewRequest.project_id == project_id,
            ReviewRequest.state.in_([ReviewRequestState.OPEN, ReviewRequestState.LEASED]),
        )
    )
    return set(rows)


def unresolved_context_trace_ids(session: Session, project: Project) -> set[str]:
    """Traces whose active judgment is CANNOT_JUDGE (routed to context/policy repair, not re-queued)."""
    rows = session.scalars(
        select(HumanJudgment.trace_id).where(
            HumanJudgment.project_id == project.id,
            HumanJudgment.policy_epoch == project.policy_epoch,
            HumanJudgment.superseded_by_id.is_(None),
            HumanJudgment.verdict == HumanVerdict.CANNOT_JUDGE,
        )
    )
    return set(rows)


# ---------------------------------------------------------------- candidate traces


def strata_of(trace: TraceSnapshot) -> dict[str, str]:
    md = trace.metadata_ or {}
    strata: dict[str, str] = {}
    for key in ("task_type", "language"):
        if md.get(key):
            strata[key] = str(md[key])
    status = None
    for call in trace.tool_calls or []:
        result = call.get("result") if isinstance(call, dict) else None
        if isinstance(result, dict) and "status" in result:
            status = str(result["status"])
            break
    strata["tool_result"] = status or "none"
    return strata


def eligible_traces(
    session: Session,
    project: Project,
    partition: str,
    *,
    exclude_labeled_groups: bool = True,
    exclude_open_requests: bool = True,
    exclude_unresolved: bool = True,
    production_only: bool = False,
) -> list[TraceSnapshot]:
    """Latest traces in a partition that ordinary workflows may touch (never sealed/quarantined)."""
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
            PartitionAssignment.partition == partition,
            PartitionAssignment.exposure_status.notin_([ExposureStatus.SEALED, ExposureStatus.QUARANTINED]),
        )
        .order_by(TraceSnapshot.group_id, TraceSnapshot.external_id)
    )
    if production_only:
        stmt = stmt.where(TraceSnapshot.source_type == "PRODUCTION")
    traces = list(session.scalars(stmt))
    excluded_groups = labeled_group_ids(session, project) if exclude_labeled_groups else set()
    excluded_traces = open_request_trace_ids(session, project.id) if exclude_open_requests else set()
    if exclude_unresolved:
        excluded_traces |= unresolved_context_trace_ids(session, project)
    return [t for t in traces if t.group_id not in excluded_groups and t.id not in excluded_traces]


# ---------------------------------------------------------------- batches


@dataclass
class BatchSpec:
    purpose: str
    kind: str  # SEED | DEV_RANDOM | ACTIVE
    size: int
    seed: int


def _one_per_group(traces: list[TraceSnapshot], rng: random.Random) -> list[TraceSnapshot]:
    by_group: dict[str, list[TraceSnapshot]] = defaultdict(list)
    for t in traces:
        by_group[t.group_id].append(t)
    picks = []
    for gid in sorted(by_group):
        members = by_group[gid]
        picks.append(members[rng.randrange(len(members))])
    return picks


def varied_seed_sample(traces: list[TraceSnapshot], size: int, seed: int) -> list[TraceSnapshot]:
    """Round-robin across strata (task type / tool result / language) for a varied seed set."""
    rng = random.Random(seed)
    candidates = _one_per_group(traces, rng)
    rng.shuffle(candidates)
    buckets: dict[str, list[TraceSnapshot]] = defaultdict(list)
    for t in candidates:
        key = "|".join(f"{k}={v}" for k, v in sorted(strata_of(t).items()))
        buckets[key].append(t)
    keys = sorted(buckets)
    rng.shuffle(keys)
    picks: list[TraceSnapshot] = []
    while len(picks) < size and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(picks) < size:
                picks.append(buckets[k].pop())
    return picks


def random_sample(traces: list[TraceSnapshot], size: int, seed: int) -> list[TraceSnapshot]:
    rng = random.Random(seed)
    candidates = _one_per_group(traces, rng)
    rng.shuffle(candidates)
    return candidates[:size]


def create_requests(
    session: Session,
    project: Project,
    traces: list[TraceSnapshot],
    *,
    purpose: str,
    category: str,
    batch_id: str | None = None,
    selection_round_id: str | None = None,
    audit_run_id: str | None = None,
    reasons: dict[str, dict[str, Any]] | None = None,
) -> list[ReviewRequest]:
    batch_id = batch_id or new_id()
    requests = []
    for t in traces:
        req = ReviewRequest(
            project_id=project.id,
            trace_id=t.id,
            purpose=purpose,
            selection_round_id=selection_round_id,
            audit_run_id=audit_run_id,
            batch_id=batch_id,
            selection_category=category,
            selection_reason=(reasons or {}).get(t.id, {}),
            expected_reading_length=estimate_reading_length(t),
            state=ReviewRequestState.OPEN,
        )
        session.add(req)
        requests.append(req)
        record_exposure(session, project.id, t.group_id, PURPOSE_TO_EXPOSURE[ReviewPurpose(purpose)], batch_id)
    session.flush()
    return requests


def create_review_batch(session: Session, project: Project, spec: BatchSpec) -> list[ReviewRequest]:
    if spec.purpose == ReviewPurpose.AUDIT:
        raise ReviewError("audit review requests are created only by the audit service")
    partition = PURPOSE_TO_PARTITION[ReviewPurpose(spec.purpose)]
    pool = eligible_traces(session, project, partition)
    if spec.kind == "SEED":
        if spec.purpose != ReviewPurpose.TRAIN:
            raise ReviewError("seed batches draw from TRAIN")
        picks = varied_seed_sample(pool, spec.size, spec.seed)
        category = SelectionCategory.SEED
    elif spec.kind == "DEV_RANDOM":
        if spec.purpose != ReviewPurpose.DEV:
            raise ReviewError("DEV batches are random DEV samples")
        picks = random_sample(pool, spec.size, spec.seed)
        category = SelectionCategory.DEV_RANDOM
    elif spec.kind == "RANDOM":
        picks = random_sample(pool, spec.size, spec.seed)
        category = SelectionCategory.RANDOM
    else:
        raise ReviewError(f"unknown batch kind {spec.kind!r}; active batches come from the selection service")
    return create_requests(session, project, picks, purpose=spec.purpose, category=category)


def dev_topup_target(resolved_train: int, *, cap: int = 40, floor: int = 8) -> int:
    import math

    return min(cap, max(floor, math.ceil(resolved_train / 4)))


# ---------------------------------------------------------------- leases and submissions


def get_request(session: Session, request_id: str, *, for_update: bool = False) -> ReviewRequest:
    # ``ReviewRequest.trace`` is eagerly joined (LEFT OUTER JOIN). PostgreSQL refuses a bare
    # ``FOR UPDATE`` on the nullable side of an outer join, so lock only the request row itself
    # (``FOR UPDATE OF review_requests``); that is the row the lease/judgment invariants protect.
    lock = {"of": ReviewRequest} if for_update else None
    req = session.get(ReviewRequest, request_id, with_for_update=lock)
    if req is None:
        raise ReviewError(f"review request {request_id} not found")
    return req


def claim(session: Session, request_id: str, *, owner: str, lease_seconds: int) -> ReviewRequest:
    req = get_request(session, request_id, for_update=True)
    now = utcnow()
    if req.state == ReviewRequestState.JUDGED:
        raise LeaseConflict("request already judged")
    if req.state not in (ReviewRequestState.OPEN, ReviewRequestState.LEASED):
        raise LeaseConflict(f"request is {req.state}")
    if (
        req.state == ReviewRequestState.LEASED
        and req.lease_owner != owner
        and req.lease_expiry is not None
        and req.lease_expiry > now
    ):
        raise LeaseConflict(f"request leased by another reviewer until {req.lease_expiry.isoformat()}")
    req.state = ReviewRequestState.LEASED
    req.lease_owner = owner
    req.lease_expiry = now + timedelta(seconds=lease_seconds)
    session.flush()
    return req


def release(session: Session, request_id: str, *, owner: str) -> ReviewRequest:
    req = get_request(session, request_id, for_update=True)
    if req.state == ReviewRequestState.LEASED and req.lease_owner == owner:
        req.state = ReviewRequestState.OPEN
        req.lease_owner = None
        req.lease_expiry = None
        session.flush()
    return req


def skip(session: Session, request_id: str, *, owner: str) -> ReviewRequest:
    req = get_request(session, request_id, for_update=True)
    if req.state == ReviewRequestState.JUDGED:
        raise ReviewError("request already judged")
    if req.lease_owner not in (None, owner):
        raise LeaseConflict("request leased by another reviewer")
    req.state = ReviewRequestState.SKIPPED
    req.lease_owner = None
    req.lease_expiry = None
    session.flush()
    return req


def submit_judgment(
    session: Session,
    request_id: str,
    *,
    verdict: str,
    explanation: str = "",
    cannot_judge_reason: str | None = None,
    reviewer_id: str,
    shown_context_hash: str,
    active_review_ms: int = 0,
    idempotency_key: str,
    owner: str | None = None,
) -> HumanJudgment:
    req = get_request(session, request_id, for_update=True)
    project = session.get(Project, req.project_id)
    assert project is not None
    existing = session.scalar(
        select(HumanJudgment).where(
            HumanJudgment.project_id == project.id, HumanJudgment.idempotency_key == idempotency_key
        )
    )
    if existing is not None:
        return existing  # idempotent replay
    if req.state == ReviewRequestState.JUDGED:
        raise ReviewError("request already judged; submit a correction instead")
    if req.state not in (ReviewRequestState.OPEN, ReviewRequestState.LEASED):
        raise ReviewError(f"request is {req.state}")
    if req.state == ReviewRequestState.LEASED and owner is not None and req.lease_owner not in (None, owner):
        if req.lease_expiry is not None and req.lease_expiry > utcnow():
            raise LeaseConflict("request leased by another reviewer")
    if verdict not in {v.value for v in HumanVerdict}:
        raise ReviewError(f"invalid verdict {verdict!r}")
    if verdict == HumanVerdict.CANNOT_JUDGE:
        if cannot_judge_reason not in {r.value for r in CannotJudgeReason}:
            raise ReviewError("CANNOT_JUDGE requires a category: MISSING_CONTEXT, AMBIGUOUS_POLICY, OUT_OF_SCOPE, OTHER")
    else:
        cannot_judge_reason = None
    trace = session.get(TraceSnapshot, req.trace_id)
    assert trace is not None
    if shown_context_hash != trace.content_hash:
        raise StaleSnapshot("the displayed snapshot does not match the stored trace; reload before judging")
    judgment = HumanJudgment(
        project_id=project.id,
        trace_id=trace.id,
        review_request_id=req.id,
        purpose=req.purpose,
        policy_epoch=project.policy_epoch,
        verdict=verdict,
        explanation=(explanation or "")[:20000],
        cannot_judge_reason=cannot_judge_reason,
        reviewer_id=reviewer_id,
        shown_context_hash=shown_context_hash,
        active_review_ms=max(0, int(active_review_ms)),
        idempotency_key=idempotency_key,
    )
    prior = active_judgment_for(session, trace.id, project.policy_epoch)
    session.add(judgment)
    session.flush()
    if prior is not None:
        judgment.supersedes_id = prior.id
        prior.superseded_by_id = judgment.id
    req.state = ReviewRequestState.JUDGED
    req.judgment_id = judgment.id
    req.lease_owner = None
    req.lease_expiry = None
    session.flush()
    return judgment


def correct_judgment(
    session: Session,
    judgment_id: str,
    *,
    verdict: str,
    explanation: str = "",
    cannot_judge_reason: str | None = None,
    reviewer_id: str,
    idempotency_key: str,
) -> HumanJudgment:
    """Append a superseding judgment; the original stays in history."""
    prior = session.get(HumanJudgment, judgment_id, with_for_update=True)
    if prior is None:
        raise ReviewError(f"judgment {judgment_id} not found")
    # Idempotent replay first (as in submit_judgment): a retried correction must return the judgment it
    # already created instead of failing because that very correction superseded ``prior``.
    existing = session.scalar(
        select(HumanJudgment).where(
            HumanJudgment.project_id == prior.project_id, HumanJudgment.idempotency_key == idempotency_key
        )
    )
    if existing is not None:
        return existing
    if prior.superseded_by_id is not None:
        raise ReviewError("judgment already superseded; correct the latest judgment")
    if verdict not in {v.value for v in HumanVerdict}:
        raise ReviewError(f"invalid verdict {verdict!r}")
    if verdict == HumanVerdict.CANNOT_JUDGE and cannot_judge_reason not in {r.value for r in CannotJudgeReason}:
        raise ReviewError("CANNOT_JUDGE requires a category")
    if verdict != HumanVerdict.CANNOT_JUDGE:
        cannot_judge_reason = None
    trace = session.get(TraceSnapshot, prior.trace_id)
    assert trace is not None
    new = HumanJudgment(
        project_id=prior.project_id,
        trace_id=prior.trace_id,
        review_request_id=prior.review_request_id,
        purpose=prior.purpose,
        policy_epoch=prior.policy_epoch,
        verdict=verdict,
        explanation=(explanation or "")[:20000],
        cannot_judge_reason=cannot_judge_reason,
        reviewer_id=reviewer_id,
        shown_context_hash=trace.content_hash,
        active_review_ms=0,
        supersedes_id=prior.id,
        idempotency_key=idempotency_key,
    )
    session.add(new)
    session.flush()
    prior.superseded_by_id = new.id
    session.flush()
    return new


def reveal_allowed(req: ReviewRequest) -> bool:
    """TRAIN selection reasons/predictions may be shown after judging; DEV/AUDIT stay blind."""
    return req.state == ReviewRequestState.JUDGED and req.purpose == ReviewPurpose.TRAIN


def count_states(session: Session, project_id: str) -> dict[str, int]:
    rows = session.execute(
        select(ReviewRequest.state, func.count()).where(ReviewRequest.project_id == project_id).group_by(ReviewRequest.state)
    )
    return {state: n for state, n in rows}
