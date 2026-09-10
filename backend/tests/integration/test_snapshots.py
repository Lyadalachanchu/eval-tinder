"""Frozen dataset snapshots: membership rules, determinism, disjointness, immutability after freezing.

Expert verdicts follow the fixture's truthful-reporting policy (``tests.cases``)
unless a test deliberately submits a wrong label in order to correct it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from sqlalchemy import func, select

from eval_tinder.db.enums import ExposureKind, ExposureStatus, HumanVerdict, Partition, SelectionCategory
from eval_tinder.db.models import (
    DatasetSnapshot,
    HumanJudgment,
    PartitionAssignment,
    Project,
    ReviewRequest,
    TraceSnapshot,
)
from eval_tinder.domain.partitions import DEFAULT_SPLIT, assign_partition
from eval_tinder.domain.rendering import render_trace
from eval_tinder.ids import new_id, sha256_hex
from eval_tinder.services.imports import import_jsonl_sync
from eval_tinder.services.projects import create_project
from eval_tinder.services.review import (
    correct_judgment,
    create_requests,
    quarantine_group,
    record_exposure,
    submit_judgment,
)
from eval_tinder.services.snapshots import (
    SnapshotError,
    assert_disjoint,
    freeze_snapshot,
    get_snapshot,
    snapshot_cases,
    snapshot_rows,
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
    # The context must make each record's content unique without echoing any bookkeeping id, so the
    # rendered-case tests can assert that external/group ids never reach the model through the renderer.
    rec: dict[str, Any] = {
        "external_id": external_id,
        "group_id": group_id,
        "timestamp": "2026-08-12T10:30:00Z",
        "input": case.input,
        "context": {"subscription_id": f"s-{sha256_hex(external_id)[:12]}"},
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


def latest_trace(session, project: Project, group_id: str) -> TraceSnapshot:
    return session.scalar(
        select(TraceSnapshot)
        .where(
            TraceSnapshot.project_id == project.id,
            TraceSnapshot.group_id == group_id,
            TraceSnapshot.is_latest.is_(True),
        )
        .order_by(TraceSnapshot.external_id)
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


def flip(verdict: str) -> str:
    return "FAIL" if verdict == "PASS" else "PASS"


def label(
    session,
    project: Project,
    trace: TraceSnapshot,
    verdict: str | None = None,
    *,
    purpose: str = "TRAIN",
    explanation: str = "",
    reason: str | None = None,
) -> HumanJudgment:
    category = {"TRAIN": SelectionCategory.SEED, "DEV": SelectionCategory.DEV_RANDOM}[purpose]
    (req,) = create_requests(session, project, [trace], purpose=purpose, category=category)
    return submit_judgment(
        session,
        req.id,
        verdict=verdict or truthful_label(trace),
        explanation=explanation,
        cannot_judge_reason=reason,
        reviewer_id=REVIEWER,
        shown_context_hash=trace.content_hash,
        idempotency_key=new_id(),
    )


def correct(session, judgment: HumanJudgment, verdict: str, explanation: str = "corrected") -> HumanJudgment:
    return correct_judgment(
        session,
        judgment.id,
        verdict=verdict,
        explanation=explanation,
        reviewer_id=REVIEWER,
        idempotency_key=new_id(),
    )


def snapshot_count(session, project: Project) -> int:
    return session.scalar(
        select(func.count()).select_from(DatasetSnapshot).where(DatasetSnapshot.project_id == project.id)
    )


def new_project(session, *, seed: int = SEED, name: str = "snapshots") -> Project:
    return create_project(session, name=name, partition_seed=seed)


# ---------------------------------------------------------------- membership


def test_freeze_train_snapshot_includes_only_active_pass_fail_of_latest_traces(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=6, dev=2)
    t_pass_or_fail, t_other, t_cannot, t_corrected, t_revised, t_unlabeled = (
        latest_trace(db_session, project, g) for g in groups["TRAIN"]
    )
    d1 = latest_trace(db_session, project, groups["DEV"][0])

    j1 = label(db_session, project, t_pass_or_fail, explanation="truthful")
    j2 = label(db_session, project, t_other)
    label(db_session, project, t_cannot, "CANNOT_JUDGE", reason="MISSING_CONTEXT")
    wrong = label(db_session, project, t_corrected, flip(truthful_label(t_corrected)))
    fixed = correct(db_session, wrong, truthful_label(t_corrected))
    label(db_session, project, t_revised)
    label(db_session, project, d1, purpose="DEV")
    # A new revision of the judged trace arrives: the judged row is no longer the latest snapshot.
    revised = record(
        t_revised.external_id, t_revised.group_id, CASES[0], output="Your plan is now cancelled."
    )
    import_jsonl_sync(db_session, project, jsonl([revised]))
    db_session.refresh(t_revised)
    assert t_revised.is_latest is False

    snapshot = freeze_snapshot(db_session, project, Partition.TRAIN)
    assert snapshot.partition == "TRAIN" and snapshot.policy_epoch == project.policy_epoch
    expected = sorted([(t_pass_or_fail.id, j1.id), (t_other.id, j2.id), (t_corrected.id, fixed.id)])
    assert snapshot.ordered_trace_ids == [tid for tid, _ in expected]
    assert snapshot.ordered_judgment_ids == [jid for _, jid in expected]
    assert wrong.id not in snapshot.ordered_judgment_ids  # superseded
    assert t_cannot.id not in snapshot.ordered_trace_ids  # CANNOT_JUDGE is not a binary target
    assert t_revised.id not in snapshot.ordered_trace_ids  # judged row is no longer the latest revision
    assert t_unlabeled.id not in snapshot.ordered_trace_ids
    assert d1.id not in snapshot.ordered_trace_ids  # other partition
    assert all(
        j.verdict in (HumanVerdict.PASS, HumanVerdict.FAIL) for _, j in snapshot_rows(db_session, snapshot)
    )
    assert len(snapshot.content_hash) == 64


def test_freeze_snapshot_excludes_sealed_and_quarantined_groups(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=3)
    quarantined, sealed, clean = groups["TRAIN"]
    judgments = {g: label(db_session, project, latest_trace(db_session, project, g)) for g in groups["TRAIN"]}
    quarantine_group(db_session, project.id, quarantined, reason="cross-partition duplicate")
    # Sealing normally targets untouched audit material; a labeled group is already INSPECTED, so the
    # SEALED status is set directly here to prove the snapshot honours it regardless of how it arose.
    assignment = db_session.scalar(
        select(PartitionAssignment).where(
            PartitionAssignment.project_id == project.id, PartitionAssignment.group_id == sealed
        )
    )
    assignment.exposure_status = ExposureStatus.SEALED
    record_exposure(db_session, project.id, sealed, ExposureKind.AUDIT_SEALED, None)
    db_session.flush()

    snapshot = freeze_snapshot(db_session, project, Partition.TRAIN)
    assert snapshot.ordered_trace_ids == [judgments[clean].trace_id]
    assert snapshot.ordered_judgment_ids == [judgments[clean].id]


def test_freeze_snapshot_is_deterministic_and_hash_tracks_label_changes(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=4)
    traces = [latest_trace(db_session, project, g) for g in groups["TRAIN"]]
    judgments = [label(db_session, project, t, explanation=f"reason {i}") for i, t in enumerate(traces)]

    first = freeze_snapshot(db_session, project, Partition.TRAIN)
    second = freeze_snapshot(db_session, project, Partition.TRAIN)
    assert first.id != second.id
    assert first.ordered_trace_ids == second.ordered_trace_ids == sorted(t.id for t in traces)
    assert first.ordered_judgment_ids == second.ordered_judgment_ids
    assert first.content_hash == second.content_hash

    target = judgments[0]
    corrected = correct(db_session, target, flip(target.verdict))
    third = freeze_snapshot(db_session, project, Partition.TRAIN)
    assert third.ordered_trace_ids == first.ordered_trace_ids  # same cases, ordered the same way
    assert third.ordered_judgment_ids != first.ordered_judgment_ids
    assert corrected.id in third.ordered_judgment_ids and target.id not in third.ordered_judgment_ids
    assert third.content_hash != first.content_hash
    # Restoring the original verdict is a new judgment row, so the hash still differs from the first freeze.
    restored = correct(db_session, corrected, target.verdict)
    fourth = freeze_snapshot(db_session, project, Partition.TRAIN)
    assert restored.id in fourth.ordered_judgment_ids
    assert fourth.content_hash not in {first.content_hash, third.content_hash}


def test_assert_disjoint_raises_on_overlap(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=2, dev=2)
    for g in groups["TRAIN"]:
        label(db_session, project, latest_trace(db_session, project, g))
    for g in groups["DEV"]:
        label(db_session, project, latest_trace(db_session, project, g), purpose="DEV")
    train = freeze_snapshot(db_session, project, Partition.TRAIN)
    dev = freeze_snapshot(db_session, project, Partition.DEV)
    assert len(train.ordered_trace_ids) == 2 and len(dev.ordered_trace_ids) == 2
    assert_disjoint(train, dev)  # group-level partitions keep TRAIN and DEV apart
    overlapping = DatasetSnapshot(
        project_id=project.id,
        partition="DEV",
        policy_epoch=1,
        ordered_trace_ids=[dev.ordered_trace_ids[0], train.ordered_trace_ids[1]],
        ordered_judgment_ids=[dev.ordered_judgment_ids[0], train.ordered_judgment_ids[1]],
        content_hash="x",
    )
    with pytest.raises(SnapshotError, match="share 1 trace"):
        assert_disjoint(train, overlapping)


def test_freezing_audit_reserve_raises(db_session):
    project = new_project(db_session)
    import_groups(db_session, project, train=1, audit=2)
    with pytest.raises(SnapshotError):
        freeze_snapshot(db_session, project, Partition.AUDIT_RESERVE)
    with pytest.raises(SnapshotError):
        freeze_snapshot(db_session, project, "AUDIT_RESERVE")
    assert snapshot_count(db_session, project) == 0


def test_labels_submitted_after_freezing_do_not_change_the_snapshot(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=3)
    t1, t2, t_late = (latest_trace(db_session, project, g) for g in groups["TRAIN"])
    j1 = label(db_session, project, t1)
    j2 = label(db_session, project, t2)
    frozen = freeze_snapshot(db_session, project, Partition.TRAIN)
    frozen_ids = (list(frozen.ordered_trace_ids), list(frozen.ordered_judgment_ids), frozen.content_hash)
    frozen_verdicts = [(t.id, j.id, j.verdict) for t, j in snapshot_rows(db_session, frozen)]
    db_session.flush()

    # Labels collected "during the job": a brand-new label and a correction of a frozen one.
    label(db_session, project, t_late)
    corrected = correct(db_session, j1, flip(j1.verdict))
    db_session.expire_all()

    reloaded = get_snapshot(db_session, frozen.id)
    assert (reloaded.ordered_trace_ids, reloaded.ordered_judgment_ids, reloaded.content_hash) == frozen_ids
    assert t_late.id not in reloaded.ordered_trace_ids
    assert corrected.id not in reloaded.ordered_judgment_ids
    # The frozen rows still resolve to the judgments as they were at freeze time (the original row is kept).
    assert [(t.id, j.id, j.verdict) for t, j in snapshot_rows(db_session, reloaded)] == frozen_verdicts
    assert [j.superseded_by_id for _, j in snapshot_rows(db_session, reloaded) if j.id == j1.id] == [
        corrected.id
    ]
    assert j2.id in reloaded.ordered_judgment_ids
    # They belong to the next snapshot.
    following = freeze_snapshot(db_session, project, Partition.TRAIN)
    assert t_late.id in following.ordered_trace_ids
    assert corrected.id in following.ordered_judgment_ids and j1.id not in following.ordered_judgment_ids
    assert following.content_hash != reloaded.content_hash


def test_snapshot_cases_render_with_partition_tag_in_index_order(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=3, dev=2)
    for g in groups["TRAIN"]:
        label(db_session, project, latest_trace(db_session, project, g), explanation=f"why {g}")
    for g in groups["DEV"]:
        label(
            db_session, project, latest_trace(db_session, project, g), purpose="DEV", explanation=f"why {g}"
        )

    for partition in (Partition.TRAIN, Partition.DEV):
        snapshot = freeze_snapshot(db_session, project, partition)
        cases = snapshot_cases(db_session, snapshot)
        rows = snapshot_rows(db_session, snapshot)
        assert len(cases) == len(snapshot.ordered_trace_ids) == (3 if partition == Partition.TRAIN else 2)
        assert [c.index for c in cases] == list(range(len(cases)))
        assert {c.partition for c in cases} == {partition.value}
        for c, (trace, judgment) in zip(cases, rows, strict=True):
            assert trace.id == snapshot.ordered_trace_ids[c.index]
            assert c.case.text == render_trace(trace).text
            assert c.case.text_hash == render_trace(trace).text_hash
            assert c.label == judgment.verdict == truthful_label(trace)
            assert c.explanation == judgment.explanation == f"why {trace.group_id}"
            # The rendered case carries no bookkeeping ids or labels.
            for forbidden in (trace.id, trace.external_id, trace.group_id, judgment.id, judgment.explanation):
                assert forbidden not in c.case.text


def test_snapshot_rows_keep_frozen_order_and_pairing(db_session):
    project = new_project(db_session)
    groups = import_groups(db_session, project, train=4)
    for g in groups["TRAIN"]:
        label(db_session, project, latest_trace(db_session, project, g))
    snapshot = freeze_snapshot(db_session, project, Partition.TRAIN)
    rows = snapshot_rows(db_session, snapshot)
    assert [t.id for t, _ in rows] == snapshot.ordered_trace_ids
    assert [j.id for _, j in rows] == snapshot.ordered_judgment_ids
    assert all(j.trace_id == t.id for t, j in rows)
    assert [t.id for t, _ in rows] == sorted(t.id for t, _ in rows)
    open_requests = db_session.scalar(
        select(func.count())
        .select_from(ReviewRequest)
        .where(ReviewRequest.project_id == project.id, ReviewRequest.state != "JUDGED")
    )
    assert open_requests == 0
