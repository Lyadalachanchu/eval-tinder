"""Exports: versioned project bundles (zip via a job) and portable grader bundles (JSON)."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.api.deps import get_db, require_auth, settings_dep
from eval_tinder.api.schemas_export import ExportCreate, ExportOut
from eval_tinder.config import Settings
from eval_tinder.db.models import ExportBundle, Job
from eval_tinder.services import exports as export_service
from eval_tinder.services import projects as project_service
from eval_tinder.services.projects import NotFound

router = APIRouter(tags=["exports"], dependencies=[Depends(require_auth)])


def export_out(db: Session, bundle: ExportBundle) -> ExportOut:
    job = db.get(Job, bundle.job_id) if bundle.job_id else None
    ready = export_service.export_ready(bundle, job)
    return ExportOut(
        id=bundle.id,
        project_id=bundle.project_id,
        kind=bundle.kind,
        state=job.state if job else "UNKNOWN",
        job_id=bundle.job_id,
        manifest=bundle.manifest or {},
        download_url=f"/exports/{bundle.id}/download" if ready else None,
        error=job.error if job else None,
        created_at=bundle.created_at,
    )


def _get_bundle(db: Session, export_id: str) -> ExportBundle:
    bundle = db.get(ExportBundle, export_id)
    if bundle is None:
        raise NotFound(f"export {export_id} not found")
    return bundle


@router.post("/projects/{project_id}/exports", response_model=ExportOut, status_code=202)
def create_export(
    project_id: str,
    body: ExportCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    project = project_service.get_project(db, project_id)
    bundle, _job = export_service.create_export(
        db, project, kind=body.kind, grader_id=body.grader_id, idempotency_key=body.idempotency_key,
        settings=settings,
    )
    return export_out(db, bundle)


@router.get("/projects/{project_id}/exports", response_model=list[ExportOut])
def list_exports(project_id: str, db: Session = Depends(get_db)):
    project_service.get_project(db, project_id)
    rows = db.scalars(
        select(ExportBundle).where(ExportBundle.project_id == project_id).order_by(ExportBundle.created_at.desc())
    )
    return [export_out(db, b) for b in rows]


@router.get("/exports/{export_id}", response_model=ExportOut)
def get_export(export_id: str, db: Session = Depends(get_db)):
    return export_out(db, _get_bundle(db, export_id))


@router.get("/exports/{export_id}/download")
def download_export(export_id: str, db: Session = Depends(get_db)):
    bundle = _get_bundle(db, export_id)
    job = db.get(Job, bundle.job_id) if bundle.job_id else None
    if not export_service.export_ready(bundle, job):
        state = job.state if job else "UNKNOWN"
        raise NotFound(f"export {export_id} is not ready (job state {state})")
    return FileResponse(
        Path(bundle.path), media_type="application/zip", filename=f"eval-tinder-{bundle.kind.lower()}-{bundle.id}.zip"
    )


@router.get("/graders/{grader_id}/bundle")
def get_grader_bundle(grader_id: str, db: Session = Depends(get_db)):
    grader = project_service.get_grader(db, grader_id)
    return export_service.grader_bundle(db, grader)
