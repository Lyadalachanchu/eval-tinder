from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.api.deps import get_db, require_auth
from eval_tinder.db.models import SelectionRound
from eval_tinder.services import projects as project_service
from eval_tinder.services import selection as selection_service

router = APIRouter(tags=["selection"], dependencies=[Depends(require_auth)])


class SelectionRoundCreate(BaseModel):
    seed: int | None = None
    idempotency_key: str


@router.post("/projects/{project_id}/selection-rounds", status_code=202)
def create_round(project_id: str, body: SelectionRoundCreate, db: Session = Depends(get_db)):
    project = project_service.get_project(db, project_id)
    rnd, job = selection_service.create_selection_round(db, project, seed=body.seed, idempotency_key=body.idempotency_key)
    view = selection_service.round_view(rnd, db)
    view["job_id"] = job.id
    return view


@router.get("/projects/{project_id}/selection-rounds")
def list_rounds(project_id: str, db: Session = Depends(get_db)):
    project_service.get_project(db, project_id)
    rows = db.scalars(select(SelectionRound).where(SelectionRound.project_id == project_id).order_by(SelectionRound.created_at.desc()))
    return [selection_service.round_view(r, db) for r in rows]


@router.get("/selection-rounds/{round_id}")
def get_round(round_id: str, db: Session = Depends(get_db)):
    rnd = db.get(SelectionRound, round_id)
    if rnd is None:
        raise project_service.NotFound(f"selection round {round_id} not found")
    return selection_service.round_view(rnd, db)
