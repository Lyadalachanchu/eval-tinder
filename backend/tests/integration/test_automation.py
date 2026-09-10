"""Integration tests for explicit, gated automation enablement and its invalidation paths."""
from __future__ import annotations

import pytest
from sqlalchemy import select

from eval_tinder.db.enums import AuditState, AutomationState, OptimizationState
from eval_tinder.db.models import AuditRun, AutomationPolicy, HumanJudgment, ReviewRequest, TraceSnapshot
from eval_tinder.domain.manifest import GraderManifest, pipeline_hash
from eval_tinder.services import audits, automation, optimization, review
from eval_tinder.services.automation import AutomationError
from eval_tinder.services.projects import bump_policy_epoch
from tests.integration.test_audits import (
    RISK_TARGETS,
    build_project,
    judge_all,
    lock,
    run_grading,
    truth_labeler,
)


def complete_audit(db_session, session_factory, settings, fx, **lock_kwargs) -> AuditRun:
    audit, _ = lock(db_session, fx, **lock_kwargs)
    run_grading(db_session, session_factory, settings)
    audit = db_session.get(AuditRun, audit.id)
    judge_all(db_session, audit, truth_labeler(fx))
    audits.persist_report(db_session, audit)
    return audit


def enable(db_session, fx, audit, **kwargs) -> AutomationPolicy:
    return automation.set_policy(
        db_session, fx.project, audit_id=audit.id, enable=True, user="expert",
        reason=kwargs.pop("reason", "audit evidence supports automation"), **kwargs,
    )


def test_default_is_disabled_and_enable_binds_the_audited_pipeline(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    assert automation.policy_status(db_session, fx.project) == {
        **automation.policy_status(db_session, fx.project), "state": "DISABLED", "reason": "no policy", "audit_id": None,
    }
    assert automation.invalidate_if_pipeline_changed(db_session, fx.project) is None
    optimization.select_shadow(db_session, fx.project, fx.truthful_grader.id, reason="release candidate", user="expert")
    audit = complete_audit(db_session, session_factory, settings, fx)
    assert audit.state == AuditState.COMPLETE and audit.report["gate"]["passed"] is True

    policy = enable(db_session, fx, audit)
    assert policy.state == AutomationState.ENABLED
    assert policy.pipeline_hash == audit.pipeline_hash == pipeline_hash(GraderManifest.from_dict(fx.truthful_grader.manifest))
    assert policy.grader_id == fx.truthful_grader.id and policy.audit_id == audit.id
    assert policy.risk_targets == audit.risk_targets
    assert policy.permitted_verdicts == ["PASS", "FAIL"]
    assert policy.supported_scope["task_types"] is None and policy.supported_scope["source_type"] == "PRODUCTION"
    assert policy.supported_scope["time_window"] == {
        "start": audit.population_definition["time_window"]["start"], "end": audit.population_definition["time_window"]["end"]}
    assert policy.gate_result["passed"] is True and policy.gate_result["audit_gate"]["passed"] is True
    assert policy.gate_result["failed"] == []
    assert policy.enabled_by == "expert" and policy.reason == "audit evidence supports automation"
    assert fx.project.automation_policy_id == policy.id
    assert [h["action"] for h in policy.history] == ["enable"]
    status = automation.policy_status(db_session, fx.project)
    assert status["state"] == "ENABLED" and status["pipeline_hash"] == audit.pipeline_hash and status["id"] == policy.id
    assert automation.invalidate_if_pipeline_changed(db_session, fx.project).state == AutomationState.ENABLED

    # Narrowing is allowed; explicit disable records a reason and keeps history.
    window = audit.population_definition["time_window"]
    narrowed = enable(db_session, fx, audit, permitted_verdicts=["FAIL"],
                      supported_scope={"task_types": ["cancellation"], "time_window": {"start": window["start"], "end": window["end"]}})
    assert narrowed.id == policy.id and narrowed.state == AutomationState.ENABLED
    assert narrowed.permitted_verdicts == ["FAIL"] and narrowed.supported_scope["task_types"] == ["cancellation"]
    disabled = automation.set_policy(db_session, fx.project, audit_id=audit.id, enable=False, user="expert", reason="pause")
    assert disabled.state == AutomationState.DISABLED and disabled.reason == "pause" and disabled.enabled_by is None
    assert [h["action"] for h in disabled.history] == ["enable", "enable", "disable"]
    assert db_session.scalar(select(AutomationPolicy.id).where(AutomationPolicy.project_id == fx.project.id)) == policy.id


def test_enable_refused_when_gate_scope_or_verdicts_do_not_hold(db_session, session_factory, settings):
    fx = build_project(db_session, settings, reserve=60)
    optimization.select_shadow(db_session, fx.project, fx.truthful_grader.id, reason="rc", user="expert")
    with pytest.raises(AutomationError, match="not found"):
        automation.set_policy(db_session, fx.project, audit_id="missing", enable=True, user="expert", reason="x")

    # Zero errors in 20 cases bounds the error rate at ~0.139, above a predeclared 0.1 target.
    strict = complete_audit(db_session, session_factory, settings, fx, risk_targets={**RISK_TARGETS, "max_error_rate": 0.1})
    assert strict.report["gate"]["passed"] is False
    policy = enable(db_session, fx, strict)
    assert policy.state == AutomationState.DISABLED and policy.enabled_by is None
    assert policy.gate_result["passed"] is False and policy.gate_result["failed"] == ["audit_gate_passed"]
    failing = next(c for c in policy.gate_result["checks"] if c["name"] == "audit_gate_passed")
    assert failing["passed"] is False and "automatic_error_rate_bound" in failing["detail"]
    assert policy.gate_result["audit_gate"]["failed"] == ["automatic_error_rate_bound"]
    assert fx.project.automation_policy_id == policy.id
    assert automation.policy_status(db_session, fx.project)["state"] == "DISABLED"

    # An audit still in review has no report: enablement is refused, never raised.
    pending, _ = lock(db_session, fx, seed=21, key="pending", population={"task_types": ["cancellation"]})
    policy = enable(db_session, fx, pending)
    assert policy.state == AutomationState.DISABLED
    assert {"audit_complete", "report_present", "audit_gate_passed"} <= set(policy.gate_result["failed"])

    # A passing audit with a declared task-type scope: broader scope or verdicts are refused.
    good = complete_audit(db_session, session_factory, settings, fx, seed=33, key="good",
                          population={"task_types": ["cancellation"]},
                          risk_targets={**RISK_TARGETS, "permitted_verdicts": ["FAIL"]})
    assert good.report["gate"]["passed"] is True
    policy = enable(db_session, fx, good, supported_scope={"task_types": ["cancellation", "refund"]})
    assert policy.state == AutomationState.DISABLED and policy.gate_result["failed"] == ["scope_within_audit"]
    assert "refund" in next(c["detail"] for c in policy.gate_result["checks"] if c["name"] == "scope_within_audit")
    policy = enable(db_session, fx, good, supported_scope={"time_window": {"start": "2020-01-01T00:00:00Z", "end": None}})
    assert policy.state == AutomationState.DISABLED and policy.gate_result["failed"] == ["scope_within_audit"]
    policy = enable(db_session, fx, good, permitted_verdicts=["PASS"])
    assert policy.state == AutomationState.DISABLED and policy.gate_result["failed"] == ["permitted_verdicts_within_audit"]
    policy = enable(db_session, fx, good)
    assert policy.state == AutomationState.ENABLED and policy.permitted_verdicts == ["FAIL"]
    assert policy.supported_scope["task_types"] == ["cancellation"]
    assert [h["resulting_state"] for h in policy.history][-4:] == ["DISABLED", "DISABLED", "DISABLED", "ENABLED"]


def test_lock_refuses_missing_risk_targets_without_inventing_defaults(db_session, settings):
    fx = build_project(db_session, settings, reserve=5)
    with pytest.raises(ValueError) as excinfo:
        audits.lock_audit(
            db_session, fx.project, grader_id=fx.truthful_grader.id, planned_n=2, seed=1, population={},
            sampling_plan={"unit": "group", "method": "uniform_random", "independence_assumption_documented": True,
                           "independence_note": "documented"},
            risk_targets={"permitted_verdicts": ["PASS", "FAIL"]}, idempotency_key="k",
        )
    assert "max_error_rate" in str(excinfo.value) and "min_coverage" in str(excinfo.value)
    assert "confidence" in str(excinfo.value) and "no default" in str(excinfo.value)
    assert db_session.scalar(select(AuditRun)) is None


def test_selecting_a_different_shadow_invalidates_enablement(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    optimization.select_shadow(db_session, fx.project, fx.truthful_grader.id, reason="rc", user="expert")
    audit = complete_audit(db_session, session_factory, settings, fx)
    policy = enable(db_session, fx, audit)
    assert policy.state == AutomationState.ENABLED

    optimization.select_shadow(db_session, fx.project, fx.seed_grader.id, reason="try the seed", user="expert")
    db_session.flush()
    policy = automation.current_policy(db_session, fx.project)
    assert policy.state == AutomationState.INVALIDATED  # through the select_shadow hook
    assert "pipeline hash" in policy.reason and policy.history[-1]["action"] == "invalidate_pipeline_changed"
    assert policy.gate_result["invalidated"] is True
    # Coming back to the audited grader does not silently re-enable: a new explicit decision is required.
    optimization.select_shadow(db_session, fx.project, fx.truthful_grader.id, reason="back", user="expert")
    assert automation.current_policy(db_session, fx.project).state == AutomationState.INVALIDATED
    assert automation.policy_status(db_session, fx.project)["state"] == "INVALIDATED"
    policy = enable(db_session, fx, audit, reason="re-enable after review")
    assert policy.state == AutomationState.ENABLED

    # Without any active shadow grader the audited pipeline is not the active one.
    optimization.clear_shadow(db_session, fx.project, reason="none", user="expert")
    assert automation.invalidate_if_pipeline_changed(db_session, fx.project).state == AutomationState.INVALIDATED
    assert "no active shadow grader" in automation.current_policy(db_session, fx.project).reason


def test_policy_epoch_change_invalidates_and_blocks_enablement(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    optimization.select_shadow(db_session, fx.project, fx.truthful_grader.id, reason="rc", user="expert")
    audit = complete_audit(db_session, session_factory, settings, fx)
    assert enable(db_session, fx, audit).state == AutomationState.ENABLED
    bump_policy_epoch(db_session, fx.project, reason="the expert now also requires an apology")
    policy = automation.invalidate_if_pipeline_changed(db_session, fx.project)
    assert policy.state == AutomationState.INVALIDATED and "policy epoch changed from 1 to 2" in policy.reason
    policy = enable(db_session, fx, audit)
    assert policy.state == AutomationState.DISABLED and "policy_epoch_current" in policy.gate_result["failed"]


def test_correcting_an_audit_judgment_invalidates_audit_and_policy(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    optimization.select_shadow(db_session, fx.project, fx.truthful_grader.id, reason="rc", user="expert")
    audit = complete_audit(db_session, session_factory, settings, fx)
    policy = enable(db_session, fx, audit)
    assert policy.state == AutomationState.ENABLED
    original = audit.report
    judgment = db_session.scalar(select(HumanJudgment).where(
        HumanJudgment.trace_id.in_(audit.locked_sample_ids), HumanJudgment.verdict == "FAIL"))
    new = audits.correct_audit_judgment(db_session, audit, judgment.id, verdict="PASS", explanation="misread the status",
                                        cannot_judge_reason=None, reviewer_id="expert", idempotency_key="c1")
    assert audit.state == AuditState.INVALIDATED
    assert policy.state == AutomationState.INVALIDATED and policy.history[-1]["action"] == "invalidate_audit"
    assert judgment.id in policy.reason
    assert audit.report == original and audit.report_history == []
    assert audit.correction_history[0]["new_judgment_id"] == new.id
    audits.persist_report(db_session, audit)
    assert audit.state == AuditState.COMPLETE and audit.report_version == 2
    assert audit.report_history[0]["report"] == original
    assert policy.state == AutomationState.INVALIDATED  # recomputing evidence never re-enables by itself
    assert automation.policy_status(db_session, fx.project)["state"] == "INVALIDATED"


def test_spent_audit_cannot_enable(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    optimization.select_shadow(db_session, fx.project, fx.truthful_grader.id, reason="rc", user="expert")
    audit = complete_audit(db_session, session_factory, settings, fx)
    audits.mark_spent(db_session, audit, reason="results were used to revise the grader")
    assert audit.report["gate"]["failed"] == ["audit_not_spent"]
    policy = enable(db_session, fx, audit)
    assert policy.state == AutomationState.DISABLED
    assert {"audit_complete", "audit_gate_passed"} <= set(policy.gate_result["failed"])
    assert next(c["detail"] for c in policy.gate_result["checks"] if c["name"] == "audit_complete") == "audit state is SPENT"


def _label_batch(db_session, project, purpose, kind, size, truth, key):
    spec = review.BatchSpec(purpose=purpose, kind=kind, size=size, seed=1)
    for req in review.create_review_batch(db_session, project, spec):
        trace = db_session.get(TraceSnapshot, req.trace_id)
        review.claim(db_session, req.id, owner="expert", lease_seconds=60)
        review.submit_judgment(db_session, req.id, verdict=truth[trace.external_id], reviewer_id="expert",
                               shown_context_hash=trace.content_hash, idempotency_key=f"{key}-{req.id}", owner="expert")


def test_create_run_marks_completed_audits_spent(db_session, session_factory, settings, monkeypatch):
    fx = build_project(db_session, settings)
    completed = complete_audit(db_session, session_factory, settings, fx)
    ongoing, _ = lock(db_session, fx, seed=4, key="ongoing")
    calls = []
    original = audits.mark_completed_audits_spent

    def spy(session, project, *, reason):
        calls.append((project.id, reason))
        return original(session, project, reason=reason)

    monkeypatch.setattr(audits, "mark_completed_audits_spent", spy)
    _label_batch(db_session, fx.project, "TRAIN", "SEED", 3, fx.truth, "t")
    _label_batch(db_session, fx.project, "DEV", "DEV_RANDOM", 2, fx.truth, "d")
    run, job = optimization.create_run(db_session, fx.project, optimization.RunRequest(max_metric_calls=20),
                                       idempotency_key="run-1", settings=settings)
    assert run.state == OptimizationState.QUEUED and job.kind == "OPTIMIZATION"
    assert len(calls) == 1 and calls[0][0] == fx.project.id and "optimization run" in calls[0][1]
    assert completed.state == AuditState.SPENT and ongoing.state == AuditState.IN_REVIEW
    assert completed.report_history[0]["event"] == "spent"
    # Audit requests and judgments never leaked into the frozen snapshots.
    from eval_tinder.db.models import DatasetSnapshot

    for snapshot_id in (run.train_snapshot_id, run.dev_snapshot_id):
        snapshot = db_session.get(DatasetSnapshot, snapshot_id)
        assert not set(snapshot.ordered_trace_ids) & set(completed.locked_sample_ids)
    # The idempotent replay of the same run request does not spend anything again.
    optimization.create_run(db_session, fx.project, optimization.RunRequest(), idempotency_key="run-1", settings=settings)
    assert len(calls) == 1
    assert db_session.scalar(select(ReviewRequest).where(ReviewRequest.audit_run_id == ongoing.id)) is not None
