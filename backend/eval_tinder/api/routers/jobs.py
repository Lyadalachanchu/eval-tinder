from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.api.deps import get_db, require_auth
from eval_tinder.api.schemas import JobOut
from eval_tinder.db.models import Job
from eval_tinder.services import jobs as job_service
from eval_tinder.services.projects import NotFound

router = APIRouter(tags=["jobs"], dependencies=[Depends(require_auth)])


def job_out(j: Job) -> JobOut:
    return JobOut(
        id=j.id, project_id=j.project_id, kind=j.kind, state=j.state, progress=j.progress or {}, result=j.result or {},
        attempts=j.attempts, error=j.error, cancel_requested=j.cancel_requested, created_at=j.created_at,
        started_at=j.started_at, finished_at=j.finished_at,
    )


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)):
    j = db.get(Job, job_id)
    if j is None:
        raise NotFound(f"job {job_id} not found")
    return job_out(j)


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, db: Session = Depends(get_db)):
    return job_out(job_service.request_cancel(db, job_id))


@router.get("/projects/{project_id}/jobs", response_model=list[JobOut])
def list_jobs(project_id: str, db: Session = Depends(get_db)):
    rows = db.scalars(select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc()).limit(200))
    return [job_out(j) for j in rows]
