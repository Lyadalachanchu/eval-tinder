from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.api.deps import get_db, require_auth
from eval_tinder.api.routers.optimization import eval_out, unified_diff
from eval_tinder.api.schemas import GraderOut
from eval_tinder.db.models import CandidateEvaluation, GraderVersion, Project
from eval_tinder.domain.manifest import GraderManifest, pipeline_hash
from eval_tinder.services import projects as project_service

router = APIRouter(tags=["graders"], dependencies=[Depends(require_auth)])


def grader_out(db: Session, g: GraderVersion) -> GraderOut:
    parent_diff = None
    if g.parent_ids:
        parent = db.get(GraderVersion, g.parent_ids[0])
        if parent is not None:
            parent_diff = unified_diff(parent.instruction_text, g.instruction_text, fromfile="parent", tofile="this")
    project = db.get(Project, g.project_id)
    evals = [eval_out(ev) for ev in db.scalars(select(CandidateEvaluation).where(CandidateEvaluation.grader_id == g.id))]
    return GraderOut(
        id=g.id, project_id=g.project_id, label=g.label, origin=g.origin, parent_ids=g.parent_ids or [],
        optimization_run_id=g.optimization_run_id, candidate_index=g.candidate_index, instruction_text=g.instruction_text,
        immutable_policy_context=g.immutable_policy_context, model_config=g.model_config_, renderer_version=g.renderer_version,
        parser_version=g.parser_version, policy_epoch=g.policy_epoch, manifest=g.manifest, manifest_hash=g.manifest_hash,
        pipeline_hash=pipeline_hash(GraderManifest.from_dict(g.manifest)), created_at=g.created_at,
        diff_from_parent=parent_diff, evaluations=evals,
        is_active_shadow=bool(project and project.active_shadow_grader_id == g.id),
    )


@router.get("/graders/{grader_id}", response_model=GraderOut)
def get_grader(grader_id: str, db: Session = Depends(get_db)):
    return grader_out(db, project_service.get_grader(db, grader_id))


@router.get("/projects/{project_id}/graders", response_model=list[GraderOut])
def list_graders(project_id: str, db: Session = Depends(get_db)):
    project_service.get_project(db, project_id)
    return [grader_out(db, g) for g in project_service.list_graders(db, project_id)]
