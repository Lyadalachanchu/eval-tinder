"""Bulk prediction jobs and provisional prediction listings. Audit material is never reachable here."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from eval_tinder.api.deps import get_db, require_auth, settings_dep
from eval_tinder.api.routers.jobs import job_out
from eval_tinder.api.schemas import JobOut
from eval_tinder.api.schemas_export import GradingJobCreate, PredictionPage
from eval_tinder.config import Settings
from eval_tinder.services import bulk_grading
from eval_tinder.services import projects as project_service

router = APIRouter(tags=["grading"], dependencies=[Depends(require_auth)])


@router.post("/projects/{project_id}/grading-jobs", response_model=JobOut, status_code=202)
def create_grading_job(
    project_id: str,
    body: GradingJobCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    project = project_service.get_project(db, project_id)
    if body.partition is not None and body.partition not in bulk_grading.BULK_PARTITIONS:
        raise bulk_grading.BulkGradingError(
            "bulk grading is limited to TRAIN and DEV; AUDIT_RESERVE material is sealed until an audit releases it"
        )
    job = bulk_grading.enqueue_bulk_grading(
        db,
        project,
        grader_id=body.grader_id,
        partition=body.partition,
        trace_ids=body.trace_ids,
        idempotency_key=body.idempotency_key,
        settings=settings,
    )
    return job_out(job)


@router.get("/projects/{project_id}/predictions", response_model=PredictionPage)
def list_predictions(
    project_id: str,
    grader_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    project = project_service.get_project(db, project_id)
    gid = grader_id or project.active_shadow_grader_id
    if not gid:
        raise bulk_grading.BulkGradingError("grader_id is required (the project has no active shadow grader)")
    items = bulk_grading.predictions_for(db, project, grader_id=gid)
    return PredictionPage(items=items[offset : offset + limit], total=len(items), limit=limit, offset=offset,
                          grader_id=gid)
