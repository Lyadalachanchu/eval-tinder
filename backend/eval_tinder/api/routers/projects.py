from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from eval_tinder.api.deps import get_db, require_auth
from eval_tinder.api.schemas import ProjectCreate, ProjectDashboard, ProjectOut
from eval_tinder.db.models import AutomationPolicy, GraderVersion, OptimizationRun, PartitionAssignment, Project
from eval_tinder.services import optimization as opt_service
from eval_tinder.services import projects as project_service
from eval_tinder.services.review import count_states, resolved_label_counts

router = APIRouter(prefix="/projects", tags=["projects"], dependencies=[Depends(require_auth)])


def project_out(p: Project) -> ProjectOut:
    return ProjectOut(
        id=p.id, name=p.name, description=p.description, policy_epoch=p.policy_epoch, policy_notes=p.policy_notes,
        configuration=p.configuration or {}, active_shadow_grader_id=p.active_shadow_grader_id,
        automation_policy_id=p.automation_policy_id, created_at=p.created_at,
    )


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    if body.idempotency_key:
        existing = db.scalar(select(Project).where(Project.configuration["idempotency_key"].as_string() == body.idempotency_key))
        if existing is not None:
            return project_out(existing)
    config = dict(body.configuration)
    if body.idempotency_key:
        config["idempotency_key"] = body.idempotency_key
    project = project_service.create_project(
        db, name=body.name, description=body.description, partition_seed=body.partition_seed, configuration=config
    )
    return project_out(project)


@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    return [project_out(p) for p in db.scalars(select(Project).order_by(Project.created_at))]


@router.get("/{project_id}", response_model=ProjectDashboard)
def get_project(project_id: str, db: Session = Depends(get_db)):
    p = project_service.get_project(db, project_id)
    partitions = {
        row[0]: row[1]
        for row in db.execute(
            select(PartitionAssignment.partition, func.count())
            .where(PartitionAssignment.project_id == p.id)
            .group_by(PartitionAssignment.partition)
        )
    }
    shadow = None
    if p.active_shadow_grader_id:
        g = db.get(GraderVersion, p.active_shadow_grader_id)
        if g is not None:
            shadow = {"id": g.id, "label": g.label, "manifest_hash": g.manifest_hash, "status": "PROVISIONAL (shadow)"}
    automation = None
    if p.automation_policy_id:
        a = db.get(AutomationPolicy, p.automation_policy_id)
        if a is not None:
            automation = {"id": a.id, "state": a.state, "pipeline_hash": a.pipeline_hash, "audit_id": a.audit_id}
    return ProjectDashboard(
        project=project_out(p),
        partitions={k: partitions.get(k, 0) for k in ("TRAIN", "DEV", "AUDIT_RESERVE")},
        labels=resolved_label_counts(db, p),
        review_states=count_states(db, p.id),
        readiness=opt_service.run_readiness(db, p),
        graders=db.scalar(select(func.count()).select_from(GraderVersion).where(GraderVersion.project_id == p.id)) or 0,
        runs=db.scalar(select(func.count()).select_from(OptimizationRun).where(OptimizationRun.project_id == p.id)) or 0,
        shadow_grader=shadow,
        automation=automation,
    )


@router.post("/{project_id}/policy-epoch", response_model=ProjectOut)
def bump_epoch(project_id: str, body: dict, db: Session = Depends(get_db)):
    p = project_service.get_project(db, project_id)
    reason = str(body.get("reason") or "").strip()
    if not reason:
        raise ValueError("a reason is required to start a new policy epoch")
    project_service.bump_policy_epoch(db, p, reason=reason)
    return project_out(p)
