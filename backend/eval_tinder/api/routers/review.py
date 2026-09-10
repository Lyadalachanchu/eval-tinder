from __future__ import annotations

import random

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.api.deps import get_db, require_auth, settings_dep
from eval_tinder.api.schemas import (
    ClaimRequest,
    CorrectionCreate,
    JudgmentCreate,
    JudgmentOut,
    ReviewBatchCreate,
    ReviewCase,
    ReviewRequestOut,
    TraceView,
)
from eval_tinder.config import Settings
from eval_tinder.db.enums import GradingPurpose, ReviewRequestState
from eval_tinder.db.models import GradingRun, HumanJudgment, PartitionAssignment, ReviewRequest, TraceSnapshot
from eval_tinder.services import projects as project_service
from eval_tinder.services import review as review_service
from eval_tinder.services.projects import project_config

router = APIRouter(tags=["review"], dependencies=[Depends(require_auth)])


def request_out(r: ReviewRequest, *, reveal: bool = False) -> ReviewRequestOut:
    return ReviewRequestOut(
        id=r.id, project_id=r.project_id, trace_id=r.trace_id, purpose=r.purpose, state=r.state,
        selection_category=r.selection_category if (reveal or r.selection_category in ("SEED", "DEV_RANDOM", "AUDIT")) else "HIDDEN",
        selection_reason=(r.selection_reason or {}) if reveal else None,
        expected_reading_length=r.expected_reading_length, lease_owner=r.lease_owner, lease_expiry=r.lease_expiry,
        judgment_id=r.judgment_id, batch_id=r.batch_id, created_at=r.created_at,
    )


def trace_view(db: Session, t: TraceSnapshot) -> TraceView:
    partition = db.scalar(
        select(PartitionAssignment.partition).where(
            PartitionAssignment.project_id == t.project_id, PartitionAssignment.group_id == t.group_id
        )
    )
    return TraceView(
        id=t.id, external_id=t.external_id, group_id=t.group_id, revision=t.revision, timestamp=t.timestamp,
        input=t.input, context=t.context, tool_calls=t.tool_calls, output=t.output, metadata=t.metadata_ or {},
        source_type=t.source_type, content_hash=t.content_hash, partition=partition,
    )


def judgment_out(j: HumanJudgment) -> JudgmentOut:
    return JudgmentOut(
        id=j.id, trace_id=j.trace_id, review_request_id=j.review_request_id, purpose=j.purpose,
        policy_epoch=j.policy_epoch, verdict=j.verdict, explanation=j.explanation,
        cannot_judge_reason=j.cannot_judge_reason, reviewer_id=j.reviewer_id, active_review_ms=j.active_review_ms,
        supersedes_id=j.supersedes_id, superseded_by_id=j.superseded_by_id, created_at=j.created_at,
    )


@router.post("/projects/{project_id}/review-batches", response_model=list[ReviewRequestOut], status_code=201)
def create_batch(project_id: str, body: ReviewBatchCreate, db: Session = Depends(get_db)):
    project = project_service.get_project(db, project_id)
    cfg = project_config(project)
    if body.kind == "ACTIVE":
        raise review_service.ReviewError("active TRAIN batches are created through POST /projects/{id}/selection-rounds")
    if body.idempotency_key:
        existing = list(db.scalars(select(ReviewRequest).where(
            ReviewRequest.project_id == project.id, ReviewRequest.batch_id == body.idempotency_key)))
        if existing:
            return [request_out(r) for r in existing]
    size = body.size or (cfg.bootstrap_train_labels if body.purpose == "TRAIN" else cfg.bootstrap_dev_labels)
    seed = body.seed if body.seed is not None else random.SystemRandom().randrange(1, 2**31 - 1)
    spec = review_service.BatchSpec(purpose=body.purpose, kind=body.kind, size=size, seed=seed)
    pool = review_service.eligible_traces(db, project, review_service.PURPOSE_TO_PARTITION[body.purpose])
    if body.kind == "SEED":
        picks = review_service.varied_seed_sample(pool, size, seed)
        category = "SEED"
    else:
        picks = review_service.random_sample(pool, size, seed)
        category = "DEV_RANDOM" if body.kind == "DEV_RANDOM" else "RANDOM"
    if body.kind == "DEV_RANDOM" and body.purpose != "DEV":
        raise review_service.ReviewError("DEV_RANDOM batches require purpose DEV")
    if body.kind == "SEED" and body.purpose != "TRAIN":
        raise review_service.ReviewError("SEED batches require purpose TRAIN")
    requests = review_service.create_requests(
        db, project, picks, purpose=spec.purpose, category=category, batch_id=body.idempotency_key
    )
    return [request_out(r) for r in requests]


@router.get("/projects/{project_id}/review-requests", response_model=list[ReviewRequestOut])
def list_requests(
    project_id: str,
    state: str | None = Query(default=None),
    purpose: str | None = Query(default=None),
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_db),
):
    project_service.get_project(db, project_id)
    stmt = select(ReviewRequest).where(ReviewRequest.project_id == project_id)
    if state:
        stmt = stmt.where(ReviewRequest.state == state)
    if purpose:
        stmt = stmt.where(ReviewRequest.purpose == purpose)
    stmt = stmt.where(ReviewRequest.purpose != "AUDIT") if purpose != "AUDIT" else stmt
    rows = db.scalars(stmt.order_by(ReviewRequest.created_at).limit(limit))
    return [request_out(r, reveal=review_service.reveal_allowed(r)) for r in rows]


@router.get("/review-requests/{request_id}", response_model=ReviewCase)
def get_request(request_id: str, db: Session = Depends(get_db)):
    r = review_service.get_request(db, request_id)
    t = db.get(TraceSnapshot, r.trace_id)
    reveal = review_service.reveal_allowed(r)
    predictions = None
    if reveal:
        runs = db.scalars(
            select(GradingRun).where(GradingRun.trace_id == r.trace_id, GradingRun.purpose.in_([GradingPurpose.POOL, GradingPurpose.PROBE, GradingPurpose.BULK]))
            .order_by(GradingRun.created_at.desc()).limit(20)
        )
        predictions = [
            {"grader_id": g.grader_id, "verdict": g.verdict, "status": g.status, "explanation": g.explanation,
             "evidence": g.evidence, "kind": "MACHINE", "provisional": True}
            for g in runs
        ]
    return ReviewCase(request=request_out(r, reveal=reveal), trace=trace_view(db, t), shown_context_hash=t.content_hash,
                      predictions=predictions)


@router.post("/review-requests/{request_id}/claim", response_model=ReviewRequestOut)
def claim(request_id: str, body: ClaimRequest | None = None, db: Session = Depends(get_db),
          settings: Settings = Depends(settings_dep), user: str = Depends(require_auth)):
    lease = (body.lease_seconds if body and body.lease_seconds else settings.lease_seconds)
    r = review_service.claim(db, request_id, owner=user, lease_seconds=lease)
    return request_out(r)


@router.post("/review-requests/{request_id}/release", response_model=ReviewRequestOut)
def release(request_id: str, db: Session = Depends(get_db), user: str = Depends(require_auth)):
    return request_out(review_service.release(db, request_id, owner=user))


@router.post("/review-requests/{request_id}/skip", response_model=ReviewRequestOut)
def skip(request_id: str, db: Session = Depends(get_db), user: str = Depends(require_auth)):
    return request_out(review_service.skip(db, request_id, owner=user))


@router.post("/review-requests/{request_id}/judgments", response_model=JudgmentOut, status_code=201)
def submit(request_id: str, body: JudgmentCreate, db: Session = Depends(get_db), user: str = Depends(require_auth)):
    j = review_service.submit_judgment(
        db, request_id, verdict=body.verdict, explanation=body.explanation, cannot_judge_reason=body.cannot_judge_reason,
        reviewer_id=user, shown_context_hash=body.shown_context_hash, active_review_ms=body.active_review_ms,
        idempotency_key=body.idempotency_key, owner=user,
    )
    return judgment_out(j)


@router.post("/judgments/{judgment_id}/corrections", response_model=JudgmentOut, status_code=201)
def correct(judgment_id: str, body: CorrectionCreate, db: Session = Depends(get_db), user: str = Depends(require_auth)):
    j = review_service.correct_judgment(
        db, judgment_id, verdict=body.verdict, explanation=body.explanation, cannot_judge_reason=body.cannot_judge_reason,
        reviewer_id=user, idempotency_key=body.idempotency_key,
    )
    return judgment_out(j)


@router.get("/projects/{project_id}/judgments", response_model=list[JudgmentOut])
def list_judgments(project_id: str, partition: str | None = None, db: Session = Depends(get_db)):
    project = project_service.get_project(db, project_id)
    if partition == "AUDIT_RESERVE":
        raise review_service.ReviewError("audit judgments are only available through the audit report")
    pairs = review_service.active_judgments(db, project, partition)
    return [judgment_out(j) for _, j in pairs if j.purpose != "AUDIT"]


@router.get("/projects/{project_id}/next-review", response_model=ReviewCase | None)
def next_review(project_id: str, purpose: str | None = None, db: Session = Depends(get_db),
                settings: Settings = Depends(settings_dep), user: str = Depends(require_auth)):
    """Claim and return the next open request (seed/DEV/active), never audit material."""
    project = project_service.get_project(db, project_id)
    stmt = select(ReviewRequest).where(
        ReviewRequest.project_id == project.id,
        ReviewRequest.state.in_([ReviewRequestState.OPEN, ReviewRequestState.LEASED]),
        ReviewRequest.purpose != "AUDIT",
    )
    if purpose:
        stmt = stmt.where(ReviewRequest.purpose == purpose)
    for r in db.scalars(stmt.order_by(ReviewRequest.created_at)):
        try:
            claimed = review_service.claim(db, r.id, owner=user, lease_seconds=settings.lease_seconds)
        except review_service.LeaseConflict:
            continue
        t = db.get(TraceSnapshot, claimed.trace_id)
        return ReviewCase(request=request_out(claimed), trace=trace_view(db, t), shown_context_hash=t.content_hash)
    return None
