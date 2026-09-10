"""Trace browsing with provisional predictions. Audit-reserve material is never listed here."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from eval_tinder.api.deps import get_db, require_auth
from eval_tinder.api.routers.review import trace_view
from eval_tinder.api.schemas import TracePage
from eval_tinder.db.models import GradingRun, HumanJudgment, PartitionAssignment, TraceSnapshot
from eval_tinder.services import projects as project_service

router = APIRouter(tags=["traces"], dependencies=[Depends(require_auth)])

BROWSABLE = ("TRAIN", "DEV")


@router.get("/projects/{project_id}/traces", response_model=TracePage)
def list_traces(
    project_id: str,
    partition: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    project = project_service.get_project(db, project_id)
    if partition is not None and partition not in BROWSABLE:
        raise ValueError("only TRAIN and DEV traces are browsable; audit material is sealed")
    stmt = (
        select(TraceSnapshot, PartitionAssignment.partition)
        .join(
            PartitionAssignment,
            (PartitionAssignment.project_id == TraceSnapshot.project_id)
            & (PartitionAssignment.group_id == TraceSnapshot.group_id),
        )
        .where(TraceSnapshot.project_id == project.id, TraceSnapshot.is_latest.is_(True))
        .where(PartitionAssignment.partition.in_(BROWSABLE))
        .where(PartitionAssignment.exposure_status.notin_(["SEALED", "QUARANTINED"]))
    )
    if partition:
        stmt = stmt.where(PartitionAssignment.partition == partition)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.execute(stmt.order_by(TraceSnapshot.group_id, TraceSnapshot.external_id).offset(offset).limit(limit)).all()
    trace_ids = [t.id for t, _ in rows]
    judgments = {
        j.trace_id: j
        for j in db.scalars(
            select(HumanJudgment).where(
                HumanJudgment.trace_id.in_(trace_ids), HumanJudgment.policy_epoch == project.policy_epoch,
                HumanJudgment.superseded_by_id.is_(None),
            )
        )
    }
    shadow_runs: dict[str, GradingRun] = {}
    if project.active_shadow_grader_id and trace_ids:
        for r in db.scalars(
            select(GradingRun).where(GradingRun.grader_id == project.active_shadow_grader_id, GradingRun.trace_id.in_(trace_ids))
            .order_by(GradingRun.created_at)
        ):
            shadow_runs[r.trace_id] = r
    items = []
    for t, part in rows:
        j = judgments.get(t.id)
        r = shadow_runs.get(t.id)
        items.append(
            {
                "trace": trace_view(db, t).model_dump(),
                "partition": part,
                "human_judgment": None if j is None or j.purpose == "AUDIT" else {
                    "verdict": j.verdict, "kind": "HUMAN", "cannot_judge_reason": j.cannot_judge_reason,
                    "explanation": j.explanation, "judgment_id": j.id,
                },
                "shadow_prediction": None if r is None else {
                    "verdict": r.verdict, "status": r.status, "kind": "MACHINE", "provisional": True,
                    "grader_id": r.grader_id, "explanation": r.explanation, "evidence": r.evidence,
                },
            }
        )
    return TracePage(items=items, total=total, limit=limit, offset=offset)
