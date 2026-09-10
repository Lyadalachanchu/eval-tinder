"""Frozen dataset snapshots for optimization.

A snapshot pins the exact traces and judgments (by id) that a run may use. New
labels collected mid-run belong to the next snapshot. Only determinate PASS/FAIL
labels from non-sealed, non-quarantined groups enter a snapshot.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.db.enums import HumanVerdict, Partition
from eval_tinder.db.models import DatasetSnapshot, HumanJudgment, PartitionAssignment, Project, TraceSnapshot
from eval_tinder.domain.rendering import render_trace
from eval_tinder.gepa.examples import LabeledCase
from eval_tinder.ids import hash_value
from eval_tinder.services.review import active_judgments


class SnapshotError(ValueError):
    pass


def freeze_snapshot(session: Session, project: Project, partition: str) -> DatasetSnapshot:
    if partition == Partition.AUDIT_RESERVE:
        raise SnapshotError("AUDIT_RESERVE is never frozen into an optimization snapshot")
    pairs = active_judgments(session, project, partition)
    sealed = set(
        session.scalars(
            select(PartitionAssignment.group_id).where(
                PartitionAssignment.project_id == project.id,
                PartitionAssignment.exposure_status.in_(["SEALED", "QUARANTINED"]),
            )
        )
    )
    rows = [
        (t, j)
        for t, j in pairs
        if j.verdict in (HumanVerdict.PASS, HumanVerdict.FAIL) and t.is_latest and t.group_id not in sealed
    ]
    rows.sort(key=lambda tj: tj[0].id)
    trace_ids = [t.id for t, _ in rows]
    judgment_ids = [j.id for _, j in rows]
    content = [(t.id, t.content_hash, j.id, j.verdict, j.explanation) for t, j in rows]
    snapshot = DatasetSnapshot(
        project_id=project.id,
        partition=partition,
        policy_epoch=project.policy_epoch,
        ordered_trace_ids=trace_ids,
        ordered_judgment_ids=judgment_ids,
        content_hash=hash_value({"partition": partition, "epoch": project.policy_epoch, "rows": content}),
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def get_snapshot(session: Session, snapshot_id: str) -> DatasetSnapshot:
    snap = session.get(DatasetSnapshot, snapshot_id)
    if snap is None:
        raise SnapshotError(f"snapshot {snapshot_id} not found")
    return snap


def snapshot_rows(session: Session, snapshot: DatasetSnapshot) -> list[tuple[TraceSnapshot, HumanJudgment]]:
    traces = {t.id: t for t in session.scalars(select(TraceSnapshot).where(TraceSnapshot.id.in_(snapshot.ordered_trace_ids)))}
    judgments = {
        j.id: j for j in session.scalars(select(HumanJudgment).where(HumanJudgment.id.in_(snapshot.ordered_judgment_ids)))
    }
    rows = []
    for tid, jid in zip(snapshot.ordered_trace_ids, snapshot.ordered_judgment_ids, strict=True):
        rows.append((traces[tid], judgments[jid]))
    return rows


def snapshot_cases(session: Session, snapshot: DatasetSnapshot) -> list[LabeledCase]:
    cases = []
    for i, (t, j) in enumerate(snapshot_rows(session, snapshot)):
        cases.append(
            LabeledCase(index=i, case=render_trace(t), label=j.verdict, explanation=j.explanation, partition=snapshot.partition)
        )
    return cases


def assert_disjoint(a: DatasetSnapshot, b: DatasetSnapshot) -> None:
    overlap = set(a.ordered_trace_ids) & set(b.ordered_trace_ids)
    if overlap:
        raise SnapshotError(f"snapshots share {len(overlap)} trace(s)")
