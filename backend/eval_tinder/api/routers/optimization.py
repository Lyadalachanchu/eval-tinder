from __future__ import annotations

import difflib

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.api.deps import get_db, require_auth
from eval_tinder.api.schemas import CandidateOut, OptimizationRunCreate, OptimizationRunOut, ShadowSelect
from eval_tinder.db.models import CandidateEvaluation, DatasetSnapshot, GraderVersion, OptimizationRun
from eval_tinder.services import optimization as opt_service
from eval_tinder.services import projects as project_service
from eval_tinder.services.jobs import request_cancel

router = APIRouter(tags=["optimization"], dependencies=[Depends(require_auth)])


def unified_diff(a: str, b: str, *, fromfile: str = "seed", tofile: str = "candidate") -> str:
    return "".join(
        difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True), fromfile=fromfile, tofile=tofile)
    )


def eval_out(ev: CandidateEvaluation | None) -> dict | None:
    if ev is None:
        return None
    return {
        "id": ev.id,
        "dev_snapshot_id": ev.dev_snapshot_id,
        "complete": ev.complete,
        "source": ev.source,
        "aggregate_metrics": ev.aggregate_metrics,
        "per_case_scores": ev.per_case_scores,
        "verdicts": ev.verdicts,
        "kind": "DEVELOPMENT_AGREEMENT",
    }


def run_out(db: Session, run: OptimizationRun, *, with_candidates: bool = True) -> OptimizationRunOut:
    train = db.get(DatasetSnapshot, run.train_snapshot_id)
    dev = db.get(DatasetSnapshot, run.dev_snapshot_id)
    seed = db.get(GraderVersion, run.seed_grader_id)
    candidates: list[CandidateOut] = []
    if with_candidates and seed is not None:
        members = set((run.result_summary or {}).get("member_indices") or [])
        ids = (run.result_summary or {}).get("candidate_grader_ids") or {}
        evals = {
            ev.grader_id: ev
            for ev in db.scalars(select(CandidateEvaluation).where(CandidateEvaluation.run_id == run.id))
        }
        seen = set()
        for idx_str, gid in sorted(ids.items(), key=lambda kv: int(kv[0])):
            g = db.get(GraderVersion, gid)
            if g is None or gid in seen:
                continue
            seen.add(gid)
            candidates.append(
                CandidateOut(
                    grader_id=g.id, candidate_index=int(idx_str), label=g.label, parent_ids=g.parent_ids or [],
                    instruction_text=g.instruction_text, manifest_hash=g.manifest_hash, evaluation=eval_out(evals.get(g.id)),
                    is_seed=(g.id == seed.id), is_member=int(idx_str) in members,
                    diff_from_seed=unified_diff(seed.instruction_text, g.instruction_text),
                )
            )
    return OptimizationRunOut(
        id=run.id, project_id=run.project_id, state=run.state, seed_grader_id=run.seed_grader_id, seed_choice=run.seed_choice,
        train_snapshot_id=run.train_snapshot_id, dev_snapshot_id=run.dev_snapshot_id,
        train_size=len(train.ordered_trace_ids) if train else 0, dev_size=len(dev.ordered_trace_ids) if dev else 0,
        policy_epoch=run.policy_epoch, metric_version=run.metric_version, config=run.config or {}, budgets=run.budgets or {},
        usage=run.usage or {}, result_summary=run.result_summary or {}, job_id=run.job_id, error=run.error,
        created_at=run.created_at, finished_at=run.finished_at, candidates=candidates,
    )


@router.post("/projects/{project_id}/optimization-runs", response_model=OptimizationRunOut, status_code=202)
def create_run(project_id: str, body: OptimizationRunCreate, db: Session = Depends(get_db)):
    project = project_service.get_project(db, project_id)
    request = opt_service.RunRequest(
        max_metric_calls=body.max_metric_calls, reflection_minibatch_size=body.reflection_minibatch_size,
        num_threads=body.num_threads, seed=body.seed, seed_grader_id=body.seed_grader_id, label=body.label,
        max_provider_calls=body.max_provider_calls, max_total_tokens=body.max_total_tokens,
        evaluate_all_candidates=body.evaluate_all_candidates,
    )
    run, _job = opt_service.create_run(db, project, request, idempotency_key=body.idempotency_key)
    return run_out(db, run, with_candidates=False)


@router.get("/projects/{project_id}/optimization-runs", response_model=list[OptimizationRunOut])
def list_runs(project_id: str, db: Session = Depends(get_db)):
    project_service.get_project(db, project_id)
    rows = db.scalars(select(OptimizationRun).where(OptimizationRun.project_id == project_id).order_by(OptimizationRun.created_at.desc()))
    return [run_out(db, r, with_candidates=False) for r in rows]


@router.get("/optimization-runs/{run_id}", response_model=OptimizationRunOut)
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(OptimizationRun, run_id)
    if run is None:
        raise project_service.NotFound(f"run {run_id} not found")
    return run_out(db, run)


@router.post("/optimization-runs/{run_id}/cancel", response_model=OptimizationRunOut)
def cancel_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(OptimizationRun, run_id)
    if run is None:
        raise project_service.NotFound(f"run {run_id} not found")
    if run.job_id:
        job = request_cancel(db, run.job_id)
        if job.state == "CANCELLED" and run.state == "QUEUED":
            run.state = "CANCELLED"
    return run_out(db, run, with_candidates=False)


@router.post("/projects/{project_id}/shadow-grader")
def select_shadow(project_id: str, body: ShadowSelect, db: Session = Depends(get_db), user: str = Depends(require_auth)):
    project = project_service.get_project(db, project_id)
    if body.grader_id is None:
        opt_service.clear_shadow(db, project, reason=body.reason, user=user)
    else:
        opt_service.select_shadow(db, project, body.grader_id, reason=body.reason, user=user)
    return {
        "project_id": project.id,
        "active_shadow_grader_id": project.active_shadow_grader_id,
        "status": "PROVISIONAL",
        "note": "A shadow grader produces provisional predictions only; it never enables verified automation.",
        "history": (project.configuration or {}).get("shadow_history", []),
    }
