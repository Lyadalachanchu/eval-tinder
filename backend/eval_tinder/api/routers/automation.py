"""Automation policy endpoints: explicit gated enable/disable bound to one audited pipeline."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from eval_tinder.api.deps import get_db, require_auth
from eval_tinder.api.schemas_audit import PolicyOut, PolicySet
from eval_tinder.services import automation as automation_service
from eval_tinder.services import projects as project_service

router = APIRouter(tags=["automation"], dependencies=[Depends(require_auth)])


@router.post("/projects/{project_id}/automation-policy", response_model=PolicyOut)
def set_policy(project_id: str, body: PolicySet, db: Session = Depends(get_db), user: str = Depends(require_auth)):
    project = project_service.get_project(db, project_id)
    automation_service.set_policy(
        db,
        project,
        audit_id=body.audit_id,
        enable=body.enable,
        user=user,
        reason=body.reason,
        permitted_verdicts=list(body.permitted_verdicts) if body.permitted_verdicts is not None else None,
        supported_scope=body.supported_scope,
    )
    return PolicyOut(**automation_service.policy_status(db, project))


@router.get("/projects/{project_id}/automation-policy", response_model=PolicyOut)
def get_policy(project_id: str, db: Session = Depends(get_db)):
    project = project_service.get_project(db, project_id)
    return PolicyOut(**automation_service.policy_status(db, project))
