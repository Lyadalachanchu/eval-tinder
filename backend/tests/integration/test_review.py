"""Human review: batches, exposure history, leases, blind idempotent judgments, eligibility.

Expert verdicts in these tests follow the fixture's truthful-reporting policy
(``tests.cases``) unless a test deliberately submits a wrong label to correct it.
"""

from __future__ import annotations

import json
import math
from datetime import timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from eval_tinder.api.app import create_app
from eval_tinder.db.enums import (
    ExposureKind,
    ExposureStatus,
    HumanVerdict,
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
from eval_tinder.domain.partitions import DEFAULT_SPLIT, assign_partition
from eval_tinder.ids import new_id, utcnow
from eval_tinder.services import review
from eval_tinder.services.imports import import_jsonl_sync
from eval_tinder.services.projects import bump_policy_epoch, create_project
from eval_tinder.services.review import (
    BatchSpec,
    LeaseConflict,
    ReviewError,
    StaleSnapshot,
    active_judgment_for,
    active_judgments,
    claim,
    correct_judgment,
    create_requests,
    create_review_batch,
    dev_topup_target,
    eligible_traces,
    open_request_trace_ids,
    quarantine_group,
    record_exposure,
    resolved_label_counts,
    reveal_allowed,
    sealed_group_ids,
    skip,
    submit_judgment,
    varied_seed_sample,
)
from tests.cases import DEV_CASES, TRAIN_CASES, DevCase

SEED = 20260910
CASES: list[DevCase] = TRAIN_CASES + DEV_CASES
REVIEWER = "expert"


# ---------------------------------------------------------------- fixture builders


def group_ids_for(
    partition: str, n: int, *, seed: int = SEED, split: dict[str, float] = DEFAULT_SPLIT
) -> list[str]:
    """Deterministically find ``n`` group ids that the seeded hash maps to ``partition``."""
    found: list[str] = []
    i = 0
    while len(found) < n:
        candidate = f"grp-{partition.lower()}-{i}"
        if assign_partition(candidate, seed, split) == partition:
            found.append(candidate)
        i += 1
    return found


def record(external_id: str, group_id: str, case: DevCase, **overrides: Any) -> dict:
    rec: dict[str, Any] = {
        "external_id": external_id,
        "group_id": group_id,
        "timestamp": "2026-08-12T10:30:00Z",
        "input": case.input,
        "context": {"subscription_id": f"s-{external_id}"},
        "tool_calls": case.tool_calls(),
        "output": case.output,
        "metadata": {"task_type": "cancellation", "language": "en"},
        "source_type": "PRODUCTION",
    }
    rec.update(overrides)
    return rec


def cancellation_records(group_ids: list[str], *, per_group: int = 1) -> list[dict]:
    recs = []
    for gi, gid in enumerate(group_ids):
        for k in range(per_group):
            recs.append(record(f"{gid}-r{k}", gid, CASES[(gi * per_group + k) % len(CASES)]))
    return recs


def jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


def import_groups(session, project: Project, *, train=0, dev=0, audit=0, per_group=1) -> dict[str, list[str]]:
    groups = {
        "TRAIN": group_ids_for("TRAIN", train),
        "DEV": group_ids_for("DEV", dev),
        "AUDIT_RESERVE": group_ids_for("AUDIT_RESERVE", audit),
    }
    recs = cancellation_records(
        groups["TRAIN"] + groups["DEV"] + groups["AUDIT_RESERVE"], per_group=per_group
    )
    batch = import_jsonl_sync(session, project, jsonl(recs))
    assert batch.counts["inserted"] == len(recs) and batch.line_errors == []
    return groups


def traces_in(session, project: Project, group_id: str) -> list[TraceSnapshot]:
    return list(
        session.scalars(
            select(TraceSnapshot)
            .where(TraceSnapshot.project_id == project.id, TraceSnapshot.group_id == group_id)
            .order_by(TraceSnapshot.external_id)
        )
    )


def truthful_label(trace: TraceSnapshot) -> str:
    """The fixture's expert label for this trace (truthful status reporting, not task completion)."""
    status = None
    for call in trace.tool_calls or []:
        status = (call.get("result") or {}).get("status")
    for c in CASES:
        if c.input == trace.input and c.output == trace.output and c.status == status:
            return c.label
    raise LookupError(f"no fixture case for {trace.external_id}")


def assignment_of(session, project: Project, group_id: str) -> PartitionAssignment:
    return session.scalar(
        select(PartitionAssignment).where(
            PartitionAssignment.project_id == project.id, PartitionAssignment.group_id == group_id
        )
    )


def request_for(session, project: Project, trace: TraceSnapshot, *, purpose: str = "TRAIN") -> ReviewRequest:
    category = {"TRAIN": SelectionCategory.SEED, "DEV": SelectionCategory.DEV_RANDOM}[purpose]
    (req,) = create_requests(session, project, [trace], purpose=purpose, category=category)
    return req


def label(
    session,
    project: Project,
    trace: TraceSnapshot,
    verdict: str | None = None,
    *,
    purpose: str = "TRAIN",
    key: str | None = None,
    explanation: str = "",
    reason: str | None = None,
) -> tuple[ReviewRequest, HumanJudgment]:
    """Open a request for ``trace`` and judge it; defaults to the truthful fixture label."""
    req = request_for(session, project, trace, purpose=purpose)
    judgment = submit_judgment(
        session,
        req.id,
        verdict=verdict or truthful_label(trace),
        explanation=explanation,
        cannot_judge_reason=reason,
        reviewer_id=REVIEWER,
        shown_context_hash=trace.content_hash,
        idempotency_key=key or new_id(),
    )
    return req, judgment


def judgment_count(session, project: Project) -> int:
    return session.scalar(
        select(func.count()).select_from(HumanJudgment).where(HumanJudgment.project_id == project.id)
    )


def new_project(session, *, seed: int = SEED, name: str = "review") -> Project:
    return create_project(session, name=name, partition_seed=seed)


# ---------------------------------------------------------------- batches


def test_seed_batch_draws_one_train_trace_per_group_and_marks_groups_inspected(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=5, dev=3, audit=2, per_group=2)
    requests = create_review_batch(
        db_session, project, BatchSpec(purpose="TRAIN", kind="SEED", size=10, seed=7)
    )

    picked_groups = [r.trace.group_id for r in requests]
    assert len(requests) == 5  # one per TRAIN group even though 10 TRAIN traces exist and size is 10
    assert len(set(picked_groups)) == 5
    assert set(picked_groups) == set(groups["TRAIN"])
    assert all(r.purpose == ReviewPurpose.TRAIN and r.state == ReviewRequestState.OPEN for r in requests)
    assert all(r.selection_category == SelectionCategory.SEED for r in requests)
    assert len({r.batch_id for r in requests}) == 1
    assert all(r.expected_reading_length > 0 for r in requests)

    batch_id = requests[0].batch_id
    events = list(db_session.scalars(select(ExposureEvent).where(ExposureEvent.project_id == project.id)))
    assert {(e.group_id, e.kind, e.reference_id) for e in events} == {
        (g, ExposureKind.TRAIN_REVIEW, batch_id) for g in groups["TRAIN"]
    }
    for g in groups["TRAIN"]:
        assert assignment_of(db_session, project, g).exposure_status == ExposureStatus.INSPECTED
    for g in groups["DEV"] + groups["AUDIT_RESERVE"]:
        assert assignment_of(db_session, project, g).exposure_status == ExposureStatus.UNTOUCHED


def test_seed_batch_respects_size_and_is_deterministic_for_a_seed(db_session):
    project = new_project(db_session)
    import_groups(db_session, project, train=6, per_group=2)
    pool = eligible_traces(db_session, project, "TRAIN")
    first = varied_seed_sample(pool, 4, 11)
    again = varied_seed_sample(pool, 4, 11)
    assert [t.id for t in first] == [t.id for t in again]
    assert len(first) == 4 and len({t.group_id for t in first}) == 4
    requests = create_review_batch(
        db_session, project, BatchSpec(purpose="TRAIN", kind="SEED", size=4, seed=11)
    )
    assert [r.trace_id for r in requests] == [t.id for t in first]


def test_dev_random_batch_draws_only_dev_groups(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=4, dev=3, audit=2, per_group=2)
    requests = create_review_batch(
        db_session, project, BatchSpec(purpose="DEV", kind="DEV_RANDOM", size=10, seed=3)
    )
    assert len(requests) == 3
    assert {r.trace.group_id for r in requests} == set(groups["DEV"])
    assert all(r.purpose == ReviewPurpose.DEV for r in requests)
    assert all(r.selection_category == SelectionCategory.DEV_RANDOM for r in requests)
    for g in groups["DEV"]:
        assert assignment_of(db_session, project, g).exposure_status == ExposureStatus.INSPECTED
    kinds = set(db_session.scalars(select(ExposureEvent.kind).where(ExposureEvent.project_id == project.id)))
    assert kinds == {ExposureKind.DEV_REVIEW}
    with pytest.raises(ReviewError):
        create_review_batch(
            db_session, project, BatchSpec(purpose="TRAIN", kind="DEV_RANDOM", size=1, seed=1)
        )
    with pytest.raises(ReviewError):
        create_review_batch(db_session, project, BatchSpec(purpose="DEV", kind="SEED", size=1, seed=1))


def test_audit_reserve_traces_are_never_offered_for_review(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=3, dev=2, audit=4, per_group=2)
    audit = set(groups["AUDIT_RESERVE"])

    with pytest.raises(ReviewError):
        create_review_batch(db_session, project, BatchSpec(purpose="AUDIT", kind="RANDOM", size=1, seed=1))
    assert not {t.group_id for t in eligible_traces(db_session, project, "TRAIN")} & audit
    assert not {t.group_id for t in eligible_traces(db_session, project, "DEV")} & audit
    offered = []
    offered += create_review_batch(
        db_session, project, BatchSpec(purpose="TRAIN", kind="SEED", size=50, seed=1)
    )
    offered += create_review_batch(
        db_session, project, BatchSpec(purpose="DEV", kind="DEV_RANDOM", size=50, seed=1)
    )
    offered += create_review_batch(
        db_session, project, BatchSpec(purpose="TRAIN", kind="RANDOM", size=50, seed=2)
    )
    offered += create_review_batch(
        db_session, project, BatchSpec(purpose="DEV", kind="RANDOM", size=50, seed=2)
    )
    assert not {r.trace.group_id for r in offered} & audit
    assert all(r.purpose != ReviewPurpose.AUDIT for r in offered)
    for g in audit:
        assert assignment_of(db_session, project, g).exposure_status == ExposureStatus.UNTOUCHED
    touched = set(
        db_session.scalars(select(ExposureEvent.group_id).where(ExposureEvent.project_id == project.id))
    )
    assert not touched & audit


# ---------------------------------------------------------------- leases


def test_claim_rejects_another_owner_while_lease_is_unexpired(db_session):
    project = new_project(db_session)
    (g,) = import_groups(db_session, project, train=1)["TRAIN"]
    req = request_for(db_session, project, traces_in(db_session, project, g)[0])
    leased = claim(db_session, req.id, owner="alice", lease_seconds=600)
    assert leased.state == ReviewRequestState.LEASED and leased.lease_owner == "alice"
    assert leased.lease_expiry > utcnow()
    with pytest.raises(LeaseConflict):
        claim(db_session, req.id, owner="bob", lease_seconds=600)
    assert req.lease_owner == "alice" and req.state == ReviewRequestState.LEASED


def test_claim_reclaims_after_lease_expiry(db_session):
    project = new_project(db_session)
    (g,) = import_groups(db_session, project, train=1)["TRAIN"]
    req = request_for(db_session, project, traces_in(db_session, project, g)[0])
    claim(db_session, req.id, owner="alice", lease_seconds=600)
    req.lease_expiry = utcnow() - timedelta(seconds=1)
    db_session.flush()
    reclaimed = claim(db_session, req.id, owner="bob", lease_seconds=600)
    assert reclaimed.lease_owner == "bob" and reclaimed.state == ReviewRequestState.LEASED
    assert reclaimed.lease_expiry > utcnow()


def test_claim_by_same_owner_renews_the_lease(db_session):
    project = new_project(db_session)
    (g,) = import_groups(db_session, project, train=1)["TRAIN"]
    req = request_for(db_session, project, traces_in(db_session, project, g)[0])
    first = claim(db_session, req.id, owner="alice", lease_seconds=10).lease_expiry
    second = claim(db_session, req.id, owner="alice", lease_seconds=600).lease_expiry
    assert second > first
    assert req.lease_owner == "alice" and req.state == ReviewRequestState.LEASED


def test_claim_rejects_judged_and_skipped_requests(db_session):
    project = new_project(db_session)
    (g,) = import_groups(db_session, project, train=1, per_group=2)["TRAIN"]
    t1, t2 = traces_in(db_session, project, g)
    judged_req, _ = label(db_session, project, t1)
    with pytest.raises(LeaseConflict):
        claim(db_session, judged_req.id, owner="alice", lease_seconds=60)
    skipped_req = request_for(db_session, project, t2)
    skip(db_session, skipped_req.id, owner="alice")
    with pytest.raises(LeaseConflict):
        claim(db_session, skipped_req.id, owner="alice", lease_seconds=60)


# ---------------------------------------------------------------- submissions


def test_submit_judgment_is_idempotent_on_idempotency_key(db_session):
    project = new_project(db_session)
    (g,) = import_groups(db_session, project, train=1)["TRAIN"]
    trace = traces_in(db_session, project, g)[0]
    req = request_for(db_session, project, trace)
    verdict = truthful_label(trace)
    kwargs = dict(
        reviewer_id=REVIEWER, shown_context_hash=trace.content_hash, idempotency_key="browser-retry-1"
    )
    first = submit_judgment(db_session, req.id, verdict=verdict, explanation="first", **kwargs)
    replay = submit_judgment(db_session, req.id, verdict=verdict, explanation="retry", **kwargs)
    assert replay.id == first.id
    assert replay.explanation == "first"
    assert judgment_count(db_session, project) == 1
    assert req.state == ReviewRequestState.JUDGED and req.judgment_id == first.id
    assert req.lease_owner is None and req.lease_expiry is None
    # A different key on an already judged request is refused: corrections are explicit appends.
    with pytest.raises(ReviewError):
        submit_judgment(db_session, req.id, verdict=verdict, **{**kwargs, "idempotency_key": "another"})
    assert judgment_count(db_session, project) == 1


def test_submit_judgment_requires_the_displayed_snapshot_hash(db_session):
    project = new_project(db_session)
    (g,) = import_groups(db_session, project, train=1)["TRAIN"]
    trace = traces_in(db_session, project, g)[0]
    req = request_for(db_session, project, trace)
    with pytest.raises(StaleSnapshot):
        submit_judgment(
            db_session,
            req.id,
            verdict=truthful_label(trace),
            reviewer_id=REVIEWER,
            shown_context_hash="0" * 64,
            idempotency_key="stale-1",
        )
    assert judgment_count(db_session, project) == 0
    assert req.state == ReviewRequestState.OPEN and req.judgment_id is None
    ok = submit_judgment(
        db_session,
        req.id,
        verdict=truthful_label(trace),
        reviewer_id=REVIEWER,
        shown_context_hash=trace.content_hash,
        idempotency_key="fresh-1",
    )
    assert ok.shown_context_hash == trace.content_hash


def test_cannot_judge_requires_a_category_and_is_stored_as_cannot_judge(db_session):
    project = new_project(db_session)
    (g,) = import_groups(db_session, project, train=1)["TRAIN"]
    trace = traces_in(db_session, project, g)[0]
    req = request_for(db_session, project, trace)
    common = dict(reviewer_id=REVIEWER, shown_context_hash=trace.content_hash)
    with pytest.raises(ReviewError):
        submit_judgment(db_session, req.id, verdict="CANNOT_JUDGE", idempotency_key="cj-0", **common)
    with pytest.raises(ReviewError):
        submit_judgment(
            db_session,
            req.id,
            verdict="CANNOT_JUDGE",
            cannot_judge_reason="BECAUSE",
            idempotency_key="cj-1",
            **common,
        )
    assert judgment_count(db_session, project) == 0
    judgment = submit_judgment(
        db_session,
        req.id,
        verdict="CANNOT_JUDGE",
        cannot_judge_reason="MISSING_CONTEXT",
        idempotency_key="cj-2",
        **common,
    )
    assert judgment.verdict == HumanVerdict.CANNOT_JUDGE
    assert judgment.verdict != HumanVerdict.FAIL
    assert judgment.cannot_judge_reason == "MISSING_CONTEXT"
    counts = resolved_label_counts(db_session, project)["TRAIN"]
    assert counts == {"PASS": 0, "FAIL": 0, "CANNOT_JUDGE": 1, "resolved": 0}
    # A determinate verdict never carries a cannot-judge category.
    (g2,) = group_ids_for("TRAIN", 2)[1:]
    import_jsonl_sync(db_session, project, jsonl(cancellation_records([g2])))
    other = traces_in(db_session, project, g2)[0]
    _, determinate = label(db_session, project, other, reason="OTHER")
    assert determinate.verdict in ("PASS", "FAIL") and determinate.cannot_judge_reason is None


def test_skip_creates_no_judgment_and_frees_the_request(db_session):
    project = new_project(db_session)
    (g,) = import_groups(db_session, project, train=1)["TRAIN"]
    trace = traces_in(db_session, project, g)[0]
    req = request_for(db_session, project, trace)
    claim(db_session, req.id, owner="alice", lease_seconds=600)
    with pytest.raises(LeaseConflict):
        skip(db_session, req.id, owner="bob")
    skipped = skip(db_session, req.id, owner="alice")
    assert skipped.state == ReviewRequestState.SKIPPED
    assert skipped.lease_owner is None and skipped.lease_expiry is None and skipped.judgment_id is None
    assert judgment_count(db_session, project) == 0
    assert trace.id not in open_request_trace_ids(db_session, project.id)
    assert trace.id in {t.id for t in eligible_traces(db_session, project, "TRAIN")}
    with pytest.raises(ReviewError):
        submit_judgment(
            db_session,
            req.id,
            verdict=truthful_label(trace),
            reviewer_id=REVIEWER,
            shown_context_hash=trace.content_hash,
            idempotency_key="after-skip",
        )


def test_second_judgment_for_same_trace_and_epoch_supersedes_the_first(db_session):
    project = new_project(db_session)
    (g,) = import_groups(db_session, project, train=1)["TRAIN"]
    trace = traces_in(db_session, project, g)[0]
    truth = truthful_label(trace)
    wrong = "FAIL" if truth == "PASS" else "PASS"
    _, first = label(db_session, project, trace, wrong, key="j-1")
    _, second = label(db_session, project, trace, truth, key="j-2")

    assert second.id != first.id
    assert second.supersedes_id == first.id
    assert first.superseded_by_id == second.id
    assert second.superseded_by_id is None
    assert first.verdict == wrong and second.verdict == truth  # the earlier row is preserved, not rewritten
    assert judgment_count(db_session, project) == 2
    assert active_judgment_for(db_session, trace.id, project.policy_epoch).id == second.id
    active = active_judgments(db_session, project)
    assert [(t.id, j.id) for t, j in active] == [(trace.id, second.id)]


def test_correct_judgment_appends_and_is_idempotent(db_session):
    project = new_project(db_session)
    (g,) = import_groups(db_session, project, train=1)["TRAIN"]
    trace = traces_in(db_session, project, g)[0]
    truth = truthful_label(trace)
    wrong = "FAIL" if truth == "PASS" else "PASS"
    _, original = label(db_session, project, trace, wrong, key="orig")

    corrected = correct_judgment(
        db_session,
        original.id,
        verdict=truth,
        explanation="corrected after re-reading the tool result",
        reviewer_id=REVIEWER,
        idempotency_key="fix-1",
    )
    assert corrected.id != original.id
    assert corrected.supersedes_id == original.id and original.superseded_by_id == corrected.id
    assert corrected.verdict == truth and original.verdict == wrong
    assert corrected.review_request_id == original.review_request_id
    assert corrected.policy_epoch == original.policy_epoch
    assert corrected.shown_context_hash == trace.content_hash
    assert judgment_count(db_session, project) == 2

    replay = correct_judgment(
        db_session,
        original.id,
        verdict=wrong,
        explanation="retry",
        reviewer_id=REVIEWER,
        idempotency_key="fix-1",
    )
    assert replay.id == corrected.id and replay.verdict == truth
    assert judgment_count(db_session, project) == 2
    with pytest.raises(ReviewError):  # only the latest judgment may be corrected
        correct_judgment(
            db_session, original.id, verdict=wrong, reviewer_id=REVIEWER, idempotency_key="fix-2"
        )
    with pytest.raises(ReviewError):  # corrections obey the CANNOT_JUDGE category rule too
        correct_judgment(
            db_session, corrected.id, verdict="CANNOT_JUDGE", reviewer_id=REVIEWER, idempotency_key="fix-3"
        )
    assert active_judgment_for(db_session, trace.id, project.policy_epoch).id == corrected.id
    assert judgment_count(db_session, project) == 2


def test_reveal_allowed_only_for_judged_train_requests(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=1, dev=1)
    train_trace = traces_in(db_session, project, groups["TRAIN"][0])[0]
    dev_trace = traces_in(db_session, project, groups["DEV"][0])[0]
    open_train = request_for(db_session, project, train_trace)
    assert reveal_allowed(open_train) is False
    claim(db_session, open_train.id, owner="alice", lease_seconds=60)
    assert reveal_allowed(open_train) is False
    submit_judgment(
        db_session,
        open_train.id,
        verdict=truthful_label(train_trace),
        reviewer_id=REVIEWER,
        shown_context_hash=train_trace.content_hash,
        idempotency_key="train-1",
        owner="alice",
    )
    assert reveal_allowed(open_train) is True
    judged_dev, _ = label(db_session, project, dev_trace, purpose="DEV")
    assert reveal_allowed(judged_dev) is False
    audit = ReviewRequest(purpose=ReviewPurpose.AUDIT, state=ReviewRequestState.JUDGED)
    assert reveal_allowed(audit) is False


# ---------------------------------------------------------------- eligibility


def test_eligible_traces_excludes_labeled_groups(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=3, per_group=2)
    labeled_group = groups["TRAIN"][0]
    judged, sibling = traces_in(db_session, project, labeled_group)
    label(db_session, project, judged)
    eligible = {t.id for t in eligible_traces(db_session, project, "TRAIN")}
    assert judged.id not in eligible and sibling.id not in eligible
    assert len(eligible) == 4
    # Opting out of the group exclusion (pool/bulk callers) brings the whole labeled group back in.
    relaxed = {t.id for t in eligible_traces(db_session, project, "TRAIN", exclude_labeled_groups=False)}
    assert {judged.id, sibling.id} <= relaxed and len(relaxed) == 6


def test_eligible_traces_excludes_open_and_leased_requests(db_session):
    project = new_project(db_session)
    (g,) = import_groups(db_session, project, train=1, per_group=2)["TRAIN"]
    t1, t2 = traces_in(db_session, project, g)
    req = request_for(db_session, project, t1)
    assert {t.id for t in eligible_traces(db_session, project, "TRAIN")} == {t2.id}
    claim(db_session, req.id, owner="alice", lease_seconds=60)
    assert {t.id for t in eligible_traces(db_session, project, "TRAIN")} == {t2.id}
    assert {t.id for t in eligible_traces(db_session, project, "TRAIN", exclude_open_requests=False)} == {
        t1.id,
        t2.id,
    }
    skip(db_session, req.id, owner="alice")
    assert {t.id for t in eligible_traces(db_session, project, "TRAIN")} == {t1.id, t2.id}


def test_eligible_traces_excludes_cannot_judge_traces(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=2, per_group=2)
    unresolved, sibling = traces_in(db_session, project, groups["TRAIN"][0])
    label(db_session, project, unresolved, "CANNOT_JUDGE", reason="MISSING_CONTEXT")
    assert unresolved.id in review.unresolved_context_trace_ids(db_session, project)
    default = {t.id for t in eligible_traces(db_session, project, "TRAIN")}
    assert unresolved.id not in default and sibling.id not in default
    relaxed = {t.id for t in eligible_traces(db_session, project, "TRAIN", exclude_labeled_groups=False)}
    assert unresolved.id not in relaxed  # never re-queued through ordinary review
    assert sibling.id in relaxed
    everything = {
        t.id
        for t in eligible_traces(
            db_session, project, "TRAIN", exclude_labeled_groups=False, exclude_unresolved=False
        )
    }
    assert unresolved.id in everything


def test_eligible_traces_excludes_sealed_and_quarantined_groups(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=3)
    quarantined, sealed, clean = groups["TRAIN"]
    quarantine_group(db_session, project.id, quarantined, reason="cross-partition duplicate discovered")
    record_exposure(db_session, project.id, sealed, ExposureKind.AUDIT_SEALED, None)
    assert assignment_of(db_session, project, quarantined).exposure_status == ExposureStatus.QUARANTINED
    assert assignment_of(db_session, project, sealed).exposure_status == ExposureStatus.SEALED
    assert sealed_group_ids(db_session, project.id) == {quarantined, sealed}
    eligible = eligible_traces(db_session, project, "TRAIN")
    assert {t.group_id for t in eligible} == {clean}
    batch = create_review_batch(db_session, project, BatchSpec(purpose="TRAIN", kind="SEED", size=10, seed=1))
    assert {r.trace.group_id for r in batch} == {clean}


# ---------------------------------------------------------------- counts and top-up


def test_resolved_label_counts_counts_only_active_pass_and_fail(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=4, dev=1, audit=1)
    t1, t2, t3, t4 = (traces_in(db_session, project, g)[0] for g in groups["TRAIN"])
    d1 = traces_in(db_session, project, groups["DEV"][0])[0]
    label(db_session, project, t1)
    label(db_session, project, t2)
    label(db_session, project, t3, "CANNOT_JUDGE", reason="AMBIGUOUS_POLICY")
    truth4 = truthful_label(t4)
    _, wrong = label(db_session, project, t4, "FAIL" if truth4 == "PASS" else "PASS")
    correct_judgment(db_session, wrong.id, verdict=truth4, reviewer_id=REVIEWER, idempotency_key="fix-4")
    label(db_session, project, d1, purpose="DEV")

    counts = resolved_label_counts(db_session, project)
    expected_train = {"PASS": 0, "FAIL": 0, "CANNOT_JUDGE": 1, "resolved": 3}
    for verdict in (
        truthful_label(t1),
        truthful_label(t2),
        truth4,
    ):  # the superseded wrong label is not counted
        expected_train[verdict] += 1
    assert counts["TRAIN"] == expected_train
    assert counts["DEV"] == {"PASS": 0, "FAIL": 0, "CANNOT_JUDGE": 0, "resolved": 1} | {truthful_label(d1): 1}
    assert counts["AUDIT_RESERVE"] == {"PASS": 0, "FAIL": 0, "CANNOT_JUDGE": 0, "resolved": 0}
    assert judgment_count(db_session, project) == 6


@pytest.mark.parametrize("resolved_train", [0, 1, 4, 31, 32, 33, 100, 159, 160, 161, 1000])
def test_dev_topup_target_formula(resolved_train):
    assert dev_topup_target(resolved_train) == min(40, max(8, math.ceil(resolved_train / 4)))


def test_dev_topup_target_honours_cap_and_floor_overrides():
    assert dev_topup_target(0, cap=10, floor=2) == 2
    assert dev_topup_target(1000, cap=10, floor=2) == 10


# ---------------------------------------------------------------- policy epochs


def test_labels_after_policy_epoch_bump_do_not_mix_with_previous_epoch(db_session):
    project = new_project(db_session)
    (g,) = import_groups(db_session, project, train=1)["TRAIN"]
    trace = traces_in(db_session, project, g)[0]
    truth = truthful_label(trace)
    _, epoch1 = label(db_session, project, trace, truth, key="e1")
    assert epoch1.policy_epoch == 1

    bump_policy_epoch(db_session, project, reason="expert now grades truthful status reporting")
    assert project.policy_epoch == 2
    assert active_judgments(db_session, project) == []
    assert resolved_label_counts(db_session, project)["TRAIN"]["resolved"] == 0
    # Old-epoch labels are revalidated, so the trace is offered again at the new epoch.
    assert trace.id in {t.id for t in eligible_traces(db_session, project, "TRAIN")}
    (req,) = create_review_batch(db_session, project, BatchSpec(purpose="TRAIN", kind="SEED", size=5, seed=1))
    assert req.trace_id == trace.id
    epoch2 = submit_judgment(
        db_session,
        req.id,
        verdict=truth,
        reviewer_id=REVIEWER,
        shown_context_hash=trace.content_hash,
        idempotency_key="e2",
    )
    assert epoch2.policy_epoch == 2
    assert epoch2.supersedes_id is None and epoch1.superseded_by_id is None  # no cross-epoch supersession
    assert active_judgment_for(db_session, trace.id, 1).id == epoch1.id
    assert active_judgment_for(db_session, trace.id, 2).id == epoch2.id
    assert [j.id for _, j in active_judgments(db_session, project)] == [epoch2.id]
    assert resolved_label_counts(db_session, project)["TRAIN"]["resolved"] == 1
    # The exposure history survives the epoch change: the group is still INSPECTED, never untouched again.
    assert assignment_of(db_session, project, g).exposure_status == ExposureStatus.INSPECTED


# ---------------------------------------------------------------- HTTP surface


def test_api_review_flow_is_blind_stale_safe_and_idempotent(db_session, settings):
    project = new_project(db_session, name="api-review")
    groups = import_groups(db_session, project, train=3, dev=1, audit=1)
    db_session.commit()
    client = TestClient(create_app())

    created = client.post(
        f"/projects/{project.id}/review-batches",
        json={"purpose": "TRAIN", "kind": "SEED", "size": 2, "seed": 5, "idempotency_key": "batch-1"},
    )
    assert created.status_code == 201, created.text
    requests = created.json()
    assert len(requests) == 2 and all(r["selection_reason"] is None for r in requests)
    replay = client.post(
        f"/projects/{project.id}/review-batches",
        json={"purpose": "TRAIN", "kind": "SEED", "size": 2, "seed": 5, "idempotency_key": "batch-1"},
    )
    assert replay.status_code == 201 and {r["id"] for r in replay.json()} == {r["id"] for r in requests}

    nxt = client.get(f"/projects/{project.id}/next-review")
    assert nxt.status_code == 200 and nxt.json() is not None
    case = nxt.json()
    assert case["request"]["state"] == "LEASED" and case["request"]["purpose"] == "TRAIN"
    assert case["trace"]["group_id"] in groups["TRAIN"]
    assert case["predictions"] is None  # blind before judging
    trace = db_session.get(TraceSnapshot, case["trace"]["id"])
    body = {
        "verdict": truthful_label(trace),
        "explanation": "matches the recorded tool status",
        "shown_context_hash": "0" * 64,
        "idempotency_key": "judge-1",
    }
    stale = client.post(f"/review-requests/{case['request']['id']}/judgments", json=body)
    assert stale.status_code == 409, stale.text
    body["shown_context_hash"] = case["shown_context_hash"]
    ok = client.post(f"/review-requests/{case['request']['id']}/judgments", json=body)
    assert ok.status_code == 201, ok.text
    again = client.post(f"/review-requests/{case['request']['id']}/judgments", json=body)
    assert again.status_code == 201 and again.json()["id"] == ok.json()["id"]
    assert ok.json()["verdict"] == truthful_label(trace) and ok.json()["reviewer_id"] == settings.reviewer_id
    db_session.expire_all()
    assert judgment_count(db_session, project) == 1

    judged = client.get(f"/review-requests/{case['request']['id']}").json()
    assert judged["request"]["state"] == "JUDGED" and judged["predictions"] == []  # revealed (none exist yet)
    listed = client.get(f"/projects/{project.id}/review-requests").json()
    assert all(r["purpose"] != "AUDIT" for r in listed)
    audit_listing = client.get(f"/projects/{project.id}/judgments", params={"partition": "AUDIT_RESERVE"})
    assert audit_listing.status_code == 400
