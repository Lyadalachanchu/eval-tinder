"""Explicit, gated automation enablement bound to one frozen pipeline.

Rules (plan section 11, "Explicit automation enablement")
- Default: no policy / DISABLED. Nothing enables automation automatically; a
  shadow choice never does.
- Enabling requires a COMPLETE audit in this project whose stored report passed
  the predeclared gate, permitted verdicts within the audit's declared ones, and
  a supported scope that only narrows the audited population and window.
- The policy binds the audit's exact pipeline hash, grader, risk targets and
  gate result. A changed active pipeline, a new policy epoch, or a corrected
  audit judgment invalidates it; re-enabling needs a fresh audit and an explicit
  decision.
- Every decision appends to the policy history; failures are recorded as a
  DISABLED policy listing the failed checks rather than raised, except for a
  missing audit, which is a caller error.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.db.enums import AuditState, AutomationState
from eval_tinder.db.models import AuditRun, AutomationPolicy, GraderVersion, Project
from eval_tinder.domain.manifest import GraderManifest, pipeline_hash
from eval_tinder.ids import utcnow
from eval_tinder.services.audits import parse_iso_datetime

SCOPE_KEYS = {"source_type", "task_types", "time_window"}


class AutomationError(ValueError):
    pass


def current_policy(session: Session, project: Project) -> AutomationPolicy | None:
    if project.automation_policy_id:
        policy = session.get(AutomationPolicy, project.automation_policy_id)
        if policy is not None and policy.project_id == project.id:
            return policy
    return session.scalar(
        select(AutomationPolicy)
        .where(AutomationPolicy.project_id == project.id)
        .order_by(AutomationPolicy.created_at.desc())
    )


def _default_scope(audit: AuditRun) -> dict[str, Any]:
    population = audit.population_definition or {}
    window = population.get("time_window") or {}
    return {
        "source_type": "PRODUCTION",
        "task_types": population.get("task_types"),
        "time_window": {"start": window.get("start"), "end": window.get("end")},
        "weighting": population.get("weighting", "group-weighted"),
        "note": "Evidence covers only the audited population and window; a new scope needs a new audit.",
    }


def _scope_problems(requested: Mapping[str, Any], audit: AuditRun) -> tuple[list[str], dict[str, Any]]:
    """Check that ``requested`` narrows the audited population. Returns (problems, normalized scope)."""
    problems: list[str] = []
    if not isinstance(requested, Mapping):
        return ["supported_scope must be an object"], {}
    unknown = sorted(set(requested) - SCOPE_KEYS)
    if unknown:
        problems.append(f"unknown supported_scope keys {unknown}; allowed: {sorted(SCOPE_KEYS)}")
    population = audit.population_definition or {}
    source = requested.get("source_type", "PRODUCTION")
    if source != "PRODUCTION":
        problems.append(f"supported_scope.source_type must be PRODUCTION, got {source!r}")
    audited_types = population.get("task_types")
    requested_types = requested.get("task_types", audited_types)
    if requested_types is not None:
        if not isinstance(requested_types, (list, tuple)) or not requested_types:
            problems.append("supported_scope.task_types must be null or a non-empty list")
            requested_types = None
        else:
            requested_types = sorted({str(t) for t in requested_types})
            if audited_types is not None and not set(requested_types) <= set(audited_types):
                extra = sorted(set(requested_types) - set(audited_types))
                problems.append(f"supported_scope.task_types {extra} were not part of the audited population")
    elif audited_types is not None:
        problems.append(f"the audit covered task_types {audited_types}; a broader scope needs a new audit")
    audited_window = population.get("time_window") or {}
    a_start = parse_iso_datetime(audited_window.get("start"), "audit time_window.start")
    a_end = parse_iso_datetime(audited_window.get("end"), "audit time_window.end")
    window = requested.get("time_window", {"start": audited_window.get("start"), "end": audited_window.get("end")})
    r_start: datetime | None = None
    r_end: datetime | None = None
    if window is None:
        window = {"start": None, "end": None}
    if not isinstance(window, Mapping) or set(window) - {"start", "end"}:
        problems.append("supported_scope.time_window must be an object with optional start/end")
        window = {"start": None, "end": None}
    else:
        try:
            r_start = parse_iso_datetime(window.get("start"), "supported_scope.time_window.start")
            r_end = parse_iso_datetime(window.get("end"), "supported_scope.time_window.end")
        except ValueError as e:
            problems.append(str(e))
    if a_start is not None and (r_start is None or r_start < a_start):
        problems.append("supported_scope.time_window.start must not precede the audited window start")
    if a_end is not None and (r_end is None or r_end > a_end):
        problems.append("supported_scope.time_window.end must not exceed the audited window end")
    if r_start is not None and r_end is not None and r_start > r_end:
        problems.append("supported_scope.time_window.start must not be after end")
    scope = {
        "source_type": "PRODUCTION",
        "task_types": requested_types,
        "time_window": {
            "start": r_start.isoformat() if r_start else None,
            "end": r_end.isoformat() if r_end else None,
        },
        "weighting": population.get("weighting", "group-weighted"),
        "note": "Evidence covers only the audited population and window; a new scope needs a new audit.",
    }
    return problems, scope


def _history_entry(**fields: Any) -> dict[str, Any]:
    return {"at": utcnow().isoformat(), **fields}


def set_policy(
    session: Session,
    project: Project,
    *,
    audit_id: str,
    enable: bool,
    user: str,
    reason: str,
    permitted_verdicts: list[str] | None = None,
    supported_scope: Mapping[str, Any] | None = None,
) -> AutomationPolicy:
    """Explicitly enable or disable automation for the pipeline an audit certified."""
    audit = session.get(AuditRun, audit_id)
    if audit is None or audit.project_id != project.id:
        raise AutomationError(f"audit {audit_id} not found in project {project.id}")
    if not (reason or "").strip():
        raise AutomationError("a reason is required for every automation decision")
    policy = current_policy(session, project)
    if policy is None:
        policy = AutomationPolicy(
            project_id=project.id,
            pipeline_hash=audit.pipeline_hash,
            grader_id=audit.grader_id,
            state=AutomationState.DISABLED.value,
            history=[],
        )
        session.add(policy)
        session.flush()
    history = list(policy.history or [])
    audit_targets = dict(audit.risk_targets or {})
    audited_verdicts = list(audit_targets.get("permitted_verdicts") or [])
    verdicts = list(permitted_verdicts) if permitted_verdicts is not None else audited_verdicts
    report = audit.report or {}
    audit_gate = report.get("gate") if isinstance(report, Mapping) else None

    if not enable:
        policy.state = AutomationState.DISABLED.value
        policy.audit_id = audit.id
        policy.pipeline_hash = audit.pipeline_hash
        policy.grader_id = audit.grader_id
        policy.risk_targets = audit_targets
        policy.permitted_verdicts = verdicts
        policy.supported_scope = _default_scope(audit)
        policy.gate_result = {"passed": False, "checks": [], "failed": [], "audit_gate": audit_gate,
                              "note": "explicitly disabled"}
        policy.enabled_by = None
        policy.reason = reason
        history.append(_history_entry(action="disable", requested_enable=False, resulting_state=policy.state,
                                      user=user, reason=reason, audit_id=audit.id, pipeline_hash=audit.pipeline_hash))
        policy.history = history
        project.automation_policy_id = policy.id
        session.flush()
        return policy

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("audit_complete", audit.state == AuditState.COMPLETE.value, f"audit state is {audit.state}")
    check("report_present", bool(report), "stored report present" if report else "the audit has no stored report")
    check(
        "audit_gate_passed",
        bool(audit_gate and audit_gate.get("passed")),
        f"audit gate failed checks {audit_gate.get('failed')}" if audit_gate and not audit_gate.get("passed")
        else ("audit gate passed" if audit_gate else "no gate result"),
    )
    check(
        "policy_epoch_current",
        project.policy_epoch == audit.policy_epoch,
        f"project policy epoch {project.policy_epoch} vs audited epoch {audit.policy_epoch}",
    )
    verdict_ok = bool(verdicts) and set(verdicts) <= set(audited_verdicts) and set(verdicts) <= {"PASS", "FAIL"}
    check(
        "permitted_verdicts_within_audit",
        verdict_ok,
        f"requested {verdicts} within audited {audited_verdicts}" if verdict_ok
        else f"requested {verdicts} are not a non-empty subset of the audited permitted verdicts {audited_verdicts}",
    )
    if supported_scope is None:
        scope = _default_scope(audit)
        scope_problems: list[str] = []
    else:
        scope_problems, scope = _scope_problems(supported_scope, audit)
    check(
        "scope_within_audit",
        not scope_problems,
        "; ".join(scope_problems) if scope_problems else "scope equals or narrows the audited population",
    )
    # The stored report must still describe the currently active judgments: a correction made through
    # any path (not only the audit endpoint) changes the judgment set, so recompute and compare.
    fresh_ok, fresh_detail = _report_is_fresh(session, audit)
    check("report_fresh", fresh_ok, fresh_detail)
    # Enablement binds to the deployed pipeline: the active shadow grader must be the audited one.
    shadow_ok, shadow_detail = _active_shadow_matches(session, project, audit)
    check("active_shadow_is_audited_pipeline", shadow_ok, shadow_detail)
    # No sample shopping: a released audit of the same pipeline and epoch whose gate failed is contrary evidence.
    contrary = _contrary_released_audits(session, project, audit)
    check(
        "no_contrary_released_audit",
        not contrary,
        f"released audit(s) of this pipeline failed their gate: {contrary}" if contrary
        else "no released audit of this pipeline and epoch failed its gate",
    )
    passed = all(c["passed"] for c in checks)
    failed = [c["name"] for c in checks if not c["passed"]]

    policy.audit_id = audit.id
    policy.pipeline_hash = audit.pipeline_hash
    policy.grader_id = audit.grader_id
    policy.risk_targets = audit_targets
    policy.permitted_verdicts = [v for v in verdicts if v in ("PASS", "FAIL")] or audited_verdicts
    policy.supported_scope = scope or _default_scope(audit)
    policy.gate_result = {"passed": passed, "checks": checks, "failed": failed, "audit_gate": audit_gate}
    policy.reason = reason
    if passed:
        policy.state = AutomationState.ENABLED.value
        policy.enabled_by = user
    else:
        policy.state = AutomationState.DISABLED.value
        policy.enabled_by = None
    history.append(
        _history_entry(
            action="enable", requested_enable=True, resulting_state=policy.state, user=user, reason=reason,
            audit_id=audit.id, pipeline_hash=audit.pipeline_hash, failed_checks=failed,
        )
    )
    policy.history = history
    project.automation_policy_id = policy.id
    session.flush()
    return policy


def _report_is_fresh(session: Session, audit: AuditRun) -> tuple[bool, str]:
    from eval_tinder.services import audits as audit_service

    stored = audit.report if isinstance(audit.report, Mapping) else None
    if not stored:
        return False, "the audit has no stored report"
    try:
        current = audit_service.compute_report(session, audit)
    except Exception as e:  # noqa: BLE001
        return False, f"could not recompute the report: {e}"
    stored_ids = list(stored.get("judgment_ids") or [])
    current_ids = list(current.get("judgment_ids") or [])
    if stored_ids != current_ids:
        return False, "the stored report no longer matches the active audit judgments; recompute the audit"
    stored_gate = bool((stored.get("gate") or {}).get("passed"))
    current_gate = bool((current.get("gate") or {}).get("passed"))
    if stored_gate != current_gate:
        return False, "the stored gate result differs from a fresh recomputation; recompute the audit"
    return True, "stored report matches the active judgments"


def _active_shadow_matches(session: Session, project: Project, audit: AuditRun) -> tuple[bool, str]:
    if not project.active_shadow_grader_id:
        return False, "no active shadow grader; select the audited grader before enabling"
    shadow = session.get(GraderVersion, project.active_shadow_grader_id)
    if shadow is None:
        return False, "active shadow grader not found"
    shadow_hash = pipeline_hash(GraderManifest.from_dict(shadow.manifest))
    if shadow_hash != audit.pipeline_hash:
        return False, "the active shadow grader is not the audited pipeline"
    return True, "the active shadow grader is the audited pipeline"


def _contrary_released_audits(session: Session, project: Project, audit: AuditRun) -> list[str]:
    rows = session.scalars(
        select(AuditRun).where(
            AuditRun.project_id == project.id,
            AuditRun.pipeline_hash == audit.pipeline_hash,
            AuditRun.policy_epoch == audit.policy_epoch,
            AuditRun.id != audit.id,
            AuditRun.state.in_([AuditState.COMPLETE.value, AuditState.SPENT.value]),
        )
    )
    mine = audit.risk_targets or {}
    contrary = []
    for other in rows:
        report = other.report if isinstance(other.report, Mapping) else {}
        gate = report.get("gate") or {}
        if not report or gate.get("passed"):
            continue
        # A failure under a STRICTER target is not evidence against this (looser) gate; a failure under
        # the same or a looser target is contrary evidence (re-sampling until a pass is sample shopping).
        theirs = other.risk_targets or {}
        stricter = _targets_stricter(theirs, mine)
        if not stricter:
            contrary.append(other.id)
    return contrary


def _targets_stricter(theirs: Mapping[str, Any], mine: Mapping[str, Any]) -> bool:
    """True when ``theirs`` demands more than ``mine`` on at least one gated quantity and less on none."""

    def _num(d: Mapping[str, Any], key: str) -> float | None:
        v = d.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    tighter = False
    for key, lower_is_stricter in (("max_error_rate", True), ("gate_false_pass_rate", True), ("min_coverage", False),
                                   ("confidence", False)):
        a, b = _num(theirs, key), _num(mine, key)
        if a is None or b is None:
            continue
        if lower_is_stricter:
            if a > b:
                return False
            if a < b:
                tighter = True
        else:
            if a < b:
                return False
            if a > b:
                tighter = True
    return tighter


def invalidate_for_pipeline_failure(session: Session, project: Project, pipeline_hash_value: str, reason: str) -> AutomationPolicy | None:
    """A released audit whose gate failed revokes an ENABLED policy for that pipeline hash."""
    policy = current_policy(session, project)
    if policy is None or policy.state != AutomationState.ENABLED.value or policy.pipeline_hash != pipeline_hash_value:
        return None
    return _invalidate(session, policy, reason=reason, action="invalidate_contrary_audit")


def _invalidate(session: Session, policy: AutomationPolicy, *, reason: str, action: str) -> AutomationPolicy:
    history = list(policy.history or [])
    history.append(_history_entry(action=action, previous_state=policy.state, resulting_state="INVALIDATED",
                                  reason=reason, audit_id=policy.audit_id, pipeline_hash=policy.pipeline_hash))
    policy.history = history
    policy.state = AutomationState.INVALIDATED.value
    policy.reason = reason
    policy.gate_result = {**(policy.gate_result or {}), "invalidated": True, "invalidation_reason": reason}
    session.flush()
    return policy


def invalidate_if_pipeline_changed(session: Session, project: Project) -> AutomationPolicy | None:
    """An ENABLED policy survives only while the active pipeline and policy epoch match the audited ones."""
    policy = current_policy(session, project)
    if policy is None or policy.state != AutomationState.ENABLED.value:
        return policy
    reasons: list[str] = []
    shadow = session.get(GraderVersion, project.active_shadow_grader_id) if project.active_shadow_grader_id else None
    if shadow is None:
        reasons.append("no active shadow grader: the audited pipeline is not the active one")
    else:
        active_hash = pipeline_hash(GraderManifest.from_dict(shadow.manifest))
        if active_hash != policy.pipeline_hash:
            reasons.append(
                f"active shadow grader {shadow.id} has pipeline hash {active_hash[:12]}..., "
                f"enablement is bound to {policy.pipeline_hash[:12]}..."
            )
    audit = session.get(AuditRun, policy.audit_id) if policy.audit_id else None
    if audit is not None and project.policy_epoch != audit.policy_epoch:
        reasons.append(f"policy epoch changed from {audit.policy_epoch} to {project.policy_epoch}")
    if not reasons:
        return policy
    return _invalidate(session, policy, reason="; ".join(reasons), action="invalidate_pipeline_changed")


def invalidate_for_audit(session: Session, audit: AuditRun, reason: str) -> list[AutomationPolicy]:
    """Every ENABLED policy depending on ``audit`` becomes INVALIDATED (e.g. after a judgment correction)."""
    changed = []
    for policy in session.scalars(select(AutomationPolicy).where(AutomationPolicy.audit_id == audit.id)):
        if policy.state == AutomationState.ENABLED.value:
            changed.append(_invalidate(session, policy, reason=reason, action="invalidate_audit"))
        else:
            history = list(policy.history or [])
            history.append(_history_entry(action="audit_changed", previous_state=policy.state,
                                          resulting_state=policy.state, reason=reason, audit_id=audit.id))
            policy.history = history
            session.flush()
    return changed


def policy_status(session: Session, project: Project) -> dict[str, Any]:
    policy = current_policy(session, project)
    if policy is None:
        return {
            "id": None,
            "project_id": project.id,
            "state": AutomationState.DISABLED.value,
            "reason": "no policy",
            "pipeline_hash": None,
            "grader_id": None,
            "audit_id": None,
            "risk_targets": {},
            "permitted_verdicts": [],
            "supported_scope": {},
            "gate_result": {},
            "enabled_by": None,
            "history": [],
            "updated_at": None,
            "note": "Automation is disabled by default; enabling requires a passing independent audit.",
        }
    return {
        "id": policy.id,
        "project_id": policy.project_id,
        "state": policy.state,
        "reason": policy.reason,
        "pipeline_hash": policy.pipeline_hash,
        "grader_id": policy.grader_id,
        "audit_id": policy.audit_id,
        "risk_targets": policy.risk_targets or {},
        "permitted_verdicts": list(policy.permitted_verdicts or []),
        "supported_scope": policy.supported_scope or {},
        "gate_result": policy.gate_result or {},
        "enabled_by": policy.enabled_by,
        "history": list(policy.history or []),
        "updated_at": policy.updated_at,
        "note": "Enablement applies only to the exact frozen pipeline hash within the supported scope.",
    }
