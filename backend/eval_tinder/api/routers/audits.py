"""Audit endpoints: lock, report, blind review, corrections, spending, release.

Locked sample ids are never returned; a reviewer sees one blind case at a time and
machine predictions on audit material are never exposed through review endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from eval_tinder.api.deps import get_db, require_auth, settings_dep
from eval_tinder.api.routers.review import judgment_out, request_out, trace_view
from eval_tinder.api.schemas import CorrectionCreate, JudgmentOut, ReviewCase
from eval_tinder.api.schemas_audit import AuditCreate, AuditOut, AuditSpend
from eval_tinder.config import Settings
from eval_tinder.db.models import AuditRun, TraceSnapshot
from eval_tinder.services import audits as audit_service
from eval_tinder.services import projects as project_service

router = APIRouter(tags=["audits"], dependencies=[Depends(require_auth)])


def audit_out(db: Session, a: AuditRun) -> AuditOut:
    judged, unresolved = audit_service.judged_count(db, a)
    plan = a.sampling_plan or {}
    return AuditOut(
        id=a.id,
        project_id=a.project_id,
        grader_id=a.grader_id,
        pipeline_hash=a.pipeline_hash,
        policy_epoch=a.policy_epoch,
        state=a.state,
        population_definition=a.population_definition or {},
        sampling_plan={k: v for k, v in plan.items() if k != "idempotency_key"},
        risk_targets=a.risk_targets or {},
        planned_n=int(plan.get("planned_n") or len(a.locked_sample_ids or [])),
        locked_count=len(a.locked_sample_ids or []),
        judged_count=judged,
        unresolved_count=unresolved,
        report=a.report,
        report_version=a.report_version,
        report_history_versions=audit_service.report_history_versions(a),
        correction_history=list(a.correction_history or []),
        grading_job_id=a.grading_job_id,
        created_at=a.created_at,
        completed_at=a.completed_at,
    )


@router.post("/projects/{project_id}/audits", response_model=AuditOut, status_code=201)
def create_audit(project_id: str, body: AuditCreate, db: Session = Depends(get_db)):
    project = project_service.get_project(db, project_id)
    audit, _job = audit_service.lock_audit(
        db,
        project,
        grader_id=body.grader_id,
        planned_n=body.planned_n,
        seed=body.seed,
        population=body.population,
        sampling_plan=body.sampling_plan,
        risk_targets=body.risk_targets,
        idempotency_key=body.idempotency_key,
    )
    return audit_out(db, audit)


@router.get("/projects/{project_id}/audits", response_model=list[AuditOut])
def list_audits(project_id: str, db: Session = Depends(get_db)):
    project_service.get_project(db, project_id)
    return [audit_out(db, a) for a in audit_service.list_audits(db, project_id)]


@router.get("/audits/{audit_id}", response_model=AuditOut)
def get_audit(audit_id: str, db: Session = Depends(get_db)):
    """Stored state and report; nothing is recomputed here."""
    return audit_out(db, audit_service.get_audit(db, audit_id))


@router.post("/audits/{audit_id}/recompute", response_model=AuditOut)
def recompute_audit(audit_id: str, db: Session = Depends(get_db)):
    audit = audit_service.get_audit(db, audit_id)
    audit_service.persist_report(db, audit)
    return audit_out(db, audit)


@router.get("/audits/{audit_id}/next-review", response_model=ReviewCase | None)
def next_audit_review(
    audit_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
    user: str = Depends(require_auth),
):
    """Claim the next locked case. Always blind: no predictions, no selection reason."""
    audit = audit_service.get_audit(db, audit_id)
    req = audit_service.next_audit_review(db, audit, owner=user, lease_seconds=settings.lease_seconds)
    if req is None:
        return None
    trace = db.get(TraceSnapshot, req.trace_id)
    assert trace is not None
    return ReviewCase(
        request=request_out(req, reveal=False),
        trace=trace_view(db, trace),
        shown_context_hash=trace.content_hash,
        predictions=None,
    )


@router.post("/audits/{audit_id}/judgments/{judgment_id}/corrections", response_model=JudgmentOut, status_code=201)
def correct_audit_judgment(
    audit_id: str,
    judgment_id: str,
    body: CorrectionCreate,
    db: Session = Depends(get_db),
    user: str = Depends(require_auth),
):
    audit = audit_service.get_audit(db, audit_id)
    judgment = audit_service.correct_audit_judgment(
        db,
        audit,
        judgment_id,
        verdict=body.verdict,
        explanation=body.explanation,
        cannot_judge_reason=body.cannot_judge_reason,
        reviewer_id=user,
        idempotency_key=body.idempotency_key,
    )
    return judgment_out(judgment)


@router.post("/audits/{audit_id}/spend", response_model=AuditOut)
def spend_audit(audit_id: str, body: AuditSpend, db: Session = Depends(get_db)):
    audit = audit_service.get_audit(db, audit_id)
    audit_service.mark_spent(db, audit, reason=body.reason)
    return audit_out(db, audit)


@router.post("/audits/{audit_id}/release", response_model=AuditOut)
def release_audit(audit_id: str, db: Session = Depends(get_db)):
    audit = audit_service.get_audit(db, audit_id)
    audit_service.release_audit_groups(db, audit)
    return audit_out(db, audit)
