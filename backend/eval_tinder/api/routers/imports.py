from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.api.deps import get_db, require_auth, settings_dep
from eval_tinder.api.schemas import ImportOut
from eval_tinder.config import Settings
from eval_tinder.db.models import ImportBatch
from eval_tinder.services import imports as import_service
from eval_tinder.services import projects as project_service

router = APIRouter(tags=["imports"], dependencies=[Depends(require_auth)])


def import_out(b: ImportBatch) -> ImportOut:
    return ImportOut(
        id=b.id, project_id=b.project_id, filename=b.filename, state=b.state, counts=b.counts or {},
        line_errors=b.line_errors or [], job_id=b.job_id, created_at=b.created_at,
    )


@router.post("/projects/{project_id}/imports", response_model=ImportOut, status_code=202)
async def create_import(
    project_id: str,
    file: UploadFile = File(...),
    idempotency_key: str = Form(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    project = project_service.get_project(db, project_id)
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"upload exceeds {settings.max_upload_bytes} bytes")
    batch, _job = import_service.enqueue_import(
        db, project, data, filename=file.filename or "upload.jsonl", idempotency_key=idempotency_key, settings=settings
    )
    return import_out(batch)


@router.get("/projects/{project_id}/imports", response_model=list[ImportOut])
def list_imports(project_id: str, db: Session = Depends(get_db)):
    project_service.get_project(db, project_id)
    rows = db.scalars(select(ImportBatch).where(ImportBatch.project_id == project_id).order_by(ImportBatch.created_at.desc()))
    return [import_out(b) for b in rows]


@router.get("/imports/{import_id}", response_model=ImportOut)
def get_import(import_id: str, db: Session = Depends(get_db)):
    b = db.get(ImportBatch, import_id)
    if b is None:
        raise project_service.NotFound(f"import {import_id} not found")
    return import_out(b)
