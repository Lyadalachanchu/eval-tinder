"""Regression tests for the findings of the adversarial verification pass."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from eval_tinder.api.app import create_app
from eval_tinder.db.enums import AutomationState, GradingPurpose, JobState, OptimizationState, ReviewPurpose
from eval_tinder.db.models import AuditRun, ExposureEvent, PartitionAssignment, ReviewRequest, SelectionRound
from eval_tinder.demo import seed_demo
from eval_tinder.gepa.service import CancellationToken
from eval_tinder.llm.budget import BudgetGuard
from eval_tinder.services import automation, audits, jobs as job_service, optimization, review
from eval_tinder.services.grading import GraderRuntime, SealedMaterial, grade_many, grade_trace
from eval_tinder.services.optimization import RunRequest
from eval_tinder.services.projects import bump_policy_epoch, seed_grader_for
from eval_tinder.services.review import create_requests, eligible_traces, quarantine_group
from eval_tinder.services.selection import round_view
from tests.integration.test_audits import RISK_TARGETS, build_project, judge_all, lock, run_grading, truth_labeler
from tests.integration.test_automation import complete_audit, enable
from tests.integration.test_optimization import TRUTHFUL_INSTRUCTIONS, labeled_project, run_through_worker

# ----------------------------------------------------------------- optimization run states


def test_cancelled_run_is_persisted_as_cancelled_not_budget_exhausted(db_session, session_factory, settings, monkeypatch):
    fx = labeled_project(db_session, settings)
    token = CancellationToken()
    token.cancel()
    monkeypatch.setattr(optimization, "CancellationToken", lambda: token)
    run, job, _ = run_through_worker(
        db_session, session_factory, settings, monkeypatch, fx, candidates=[TRUTHFUL_INSTRUCTIONS, TRUTHFUL_INSTRUCTIONS + " v2"]
    )
    assert run.state == OptimizationState.CANCELLED
    assert job.state == JobState.CANCELLED
    assert run.result_summary["partial"] is True


def test_gepa_best_candidate_beyond_shortlist_cap_is_still_evaluated(db_session, session_factory, settings, monkeypatch):
    fx = labeled_project(db_session, settings)
    seed_text = seed_grader_for(db_session, fx.project).instruction_text
    weak = [f"{seed_text}\nVariant {i}: keep explanations brief." for i in range(12)]
    run, job, _ = run_through_worker(db_session, session_factory, settings, monkeypatch, fx, candidates=weak + [TRUTHFUL_INSTRUCTIONS])
    summary = run.result_summary
    assert summary["gepa_best_index"] == 13
    assert 13 in summary["evaluated_indices"], "GEPA's own best candidate must be evaluated on the app path"
    assert summary["recommended_grader_id"] is not None and run.state == OptimizationState.SUCCEEDED
    assert summary["unevaluated_indices"], "the shortlist cap still applies to the remaining members"


def test_budget_exhaustion_during_dev_evaluation_is_an_explicit_partial_state(db_session, session_factory, settings, monkeypatch):
    fx = labeled_project(db_session, settings)
    # The fake optimizer spends 2 candidates x DEV calls; the seed re-evaluation fits, the candidate's does not.
    run, job, _ = run_through_worker(
        db_session, session_factory, settings, monkeypatch, fx, candidates=[TRUTHFUL_INSTRUCTIONS],
        request=RunRequest(max_provider_calls=12),
    )
    assert run.state == OptimizationState.BUDGET_EXHAUSTED and job.state == JobState.BUDGET_EXHAUSTED
    assert run.result_summary["partial"] is True and run.result_summary["evaluation_budget_exhausted"] is True
    assert run.result_summary["recommended_grader_id"] is None


# ----------------------------------------------------------------- blind review surfaces


def test_selection_round_view_hides_categories_until_judged(db_session, settings):
    result = seed_demo(db_session, simulate_expert=False, settings=settings)
    from eval_tinder.db.models import Project

    project = db_session.get(Project, result["project_id"])
    traces = eligible_traces(db_session, project, "TRAIN")[:2]
    rnd = SelectionRound(project_id=project.id, state="COMPLETE", strategy_version="select-v1", seed=1)
    db_session.add(rnd)
    db_session.flush()
    reqs = create_requests(
        db_session, project, traces, purpose=ReviewPurpose.TRAIN, category="DISAGREEMENT", batch_id=rnd.id,
        selection_round_id=rnd.id, reasons={t.id: {"score": 0.5} for t in traces},
    )
    rnd.selected_requests = [{"request_id": r.id, "trace_id": r.trace_id, "category": "DISAGREEMENT"} for r in reqs]
    db_session.flush()
    view = round_view(rnd, db_session)
    assert {e["category"] for e in view["selected_requests"]} == {"HIDDEN"}
    assert view["category_counts"] == {"DISAGREEMENT": 2}
    review.submit_judgment(
        db_session, reqs[0].id, verdict="PASS", reviewer_id="expert", shown_context_hash=traces[0].content_hash,
        idempotency_key="j1",
    )
    view = round_view(rnd, db_session)
    by_id = {e["request_id"]: e["category"] for e in view["selected_requests"]}
    assert by_id[reqs[0].id] == "DISAGREEMENT" and by_id[reqs[1].id] == "HIDDEN"


def test_traces_listing_shows_only_bulk_predictions_and_never_for_open_requests(db_session, settings):
    from eval_tinder.db.models import Project

    result = seed_demo(db_session, simulate_expert=False, settings=settings)
    project = db_session.get(Project, result["project_id"])
    grader = seed_grader_for(db_session, project)
    optimization.select_shadow(db_session, project, grader.id, reason="test", user="t")
    pooled, bulk_open, bulk_free = eligible_traces(db_session, project, "TRAIN")[:3]
    runtime = GraderRuntime.build(project, grader, settings=settings)
    grade_trace(db_session, project, runtime, pooled, purpose=GradingPurpose.POOL, settings=settings)
    grade_trace(db_session, project, runtime, bulk_open, purpose=GradingPurpose.BULK, settings=settings)
    grade_trace(db_session, project, runtime, bulk_free, purpose=GradingPurpose.BULK, settings=settings)
    create_requests(db_session, project, [bulk_open], purpose=ReviewPurpose.TRAIN, category="RANDOM")
    db_session.commit()
    client = TestClient(create_app())
    items = client.get(f"/projects/{project.id}/traces", params={"limit": 200}).json()["items"]
    shown = {i["trace"]["id"]: i["shadow_prediction"] for i in items}
    assert shown[pooled.id] is None, "committee votes are not shadow predictions"
    assert shown[bulk_open.id] is None, "a case awaiting blind review shows no machine verdict"
    assert shown[bulk_free.id] is not None and shown[bulk_free.id]["kind"] == "MACHINE"


# ----------------------------------------------------------------- idempotency scoping


def test_idempotency_keys_are_scoped_per_project_and_kind(db_session, settings):
    from eval_tinder.db.models import Project

    a = db_session.get(Project, seed_demo(db_session, name="a", settings=settings)["project_id"])
    b = db_session.get(Project, seed_demo(db_session, name="b", partition_seed=5, settings=settings)["project_id"])
    job_service.enqueue(db_session, kind="IMPORT", payload={}, idempotency_key="shared", project_id=a.id)
    with pytest.raises(job_service.JobError, match="another project"):
        job_service.find_existing(db_session, "shared", project_id=b.id, kind="IMPORT")
    with pytest.raises(job_service.JobError):
        job_service.find_existing(db_session, "shared", project_id=a.id, kind="OPTIMIZATION")
    assert job_service.find_existing(db_session, "shared", project_id=a.id, kind="IMPORT") is not None


# ----------------------------------------------------------------- automation and audits


def test_enablement_requires_audited_shadow_fresh_report_and_no_contrary_audit(db_session, session_factory, settings):
    fx = build_project(db_session, settings, reserve=60)
    # 1. The active shadow must be the audited pipeline.
    optimization.select_shadow(db_session, fx.project, fx.seed_grader.id, reason="rc", user="expert")
    audit = complete_audit(db_session, session_factory, settings, fx, key="a1", seed=11)
    assert audit.report["gate"]["passed"] is True
    policy = enable(db_session, fx, audit)
    assert policy.state == AutomationState.DISABLED and "active_shadow_is_audited_pipeline" in policy.gate_result["failed"]
    optimization.select_shadow(db_session, fx.project, fx.truthful_grader.id, reason="rc", user="expert")
    policy = enable(db_session, fx, audit)
    assert policy.state == AutomationState.ENABLED
    # 2. A generic correction (bypassing the audit endpoint) makes the stored report stale.
    judged = db_session.scalars(
        select(ReviewRequest).where(ReviewRequest.audit_run_id == audit.id, ReviewRequest.state == "JUDGED")
    ).first()
    review.correct_judgment(
        db_session, judged.judgment_id, verdict="FAIL", explanation="changed my mind", reviewer_id="expert",
        idempotency_key="corr-1",
    )
    policy = enable(db_session, fx, audit)
    assert policy.state == AutomationState.DISABLED and "report_fresh" in policy.gate_result["failed"]
    # 3. A policy-epoch bump invalidates whatever was enabled.
    optimization.select_shadow(db_session, fx.project, fx.truthful_grader.id, reason="rc", user="expert")
    audits.persist_report(db_session, audit)  # recompute after the correction (gate may now fail; that is fine)
    bump_policy_epoch(db_session, fx.project, reason="new standard")
    status = automation.policy_status(db_session, fx.project)
    assert status["state"] in ("INVALIDATED", "DISABLED")


def test_contrary_released_audit_of_same_pipeline_blocks_enablement_unless_stricter(db_session, settings):
    fx = build_project(db_session, settings, reserve=10)
    base = dict(RISK_TARGETS)
    good = AuditRun(
        project_id=fx.project.id, grader_id=fx.truthful_grader.id, pipeline_hash="h", policy_epoch=fx.project.policy_epoch,
        risk_targets=base, state="COMPLETE", report={"gate": {"passed": True}}, locked_sample_ids=[],
    )
    failed_same = AuditRun(
        project_id=fx.project.id, grader_id=fx.truthful_grader.id, pipeline_hash="h", policy_epoch=fx.project.policy_epoch,
        risk_targets=base, state="COMPLETE", report={"gate": {"passed": False}}, locked_sample_ids=[],
    )
    db_session.add_all([good, failed_same])
    db_session.flush()
    assert automation._contrary_released_audits(db_session, fx.project, good) == [failed_same.id]
    failed_same.risk_targets = {**base, "max_error_rate": base["max_error_rate"] / 2}  # stricter: not contrary
    db_session.flush()
    assert automation._contrary_released_audits(db_session, fx.project, good) == []
    assert automation._targets_stricter({"max_error_rate": 0.1}, {"max_error_rate": 0.2}) is True
    assert automation._targets_stricter({"max_error_rate": 0.2}, {"max_error_rate": 0.2}) is False
    assert automation._targets_stricter({"max_error_rate": 0.3}, {"max_error_rate": 0.2}) is False


def test_lock_audit_rejects_idempotency_replay_with_a_different_body(db_session, settings):
    fx = build_project(db_session, settings, reserve=30)
    audit, job = lock(db_session, fx, planned_n=5, key="same-key")
    again, job2 = lock(db_session, fx, planned_n=5, key="same-key")
    assert again.id == audit.id and job2.id == job.id
    with pytest.raises(audits.AuditError, match="already used"):
        lock(db_session, fx, planned_n=6, key="same-key")


def test_audit_report_withholds_case_details_until_complete(db_session, session_factory, settings):
    fx = build_project(db_session, settings, reserve=30)
    audit, _ = lock(db_session, fx, planned_n=5, key="w1")
    run_grading(db_session, session_factory, settings)
    audit = db_session.get(AuditRun, audit.id)
    judge_all(db_session, audit, truth_labeler(fx), limit=2)
    report = audits.compute_report(db_session, audit)
    assert report["case_details_withheld"] is True
    assert all("trace_id" not in e for e in report["human_unresolved"])
    assert all("trace_id" not in e for e in report["operational_failures"])
    assert len(report["judgment_ids"]) == 2
    judge_all(db_session, audit, truth_labeler(fx))
    report = audits.compute_report(db_session, audit)
    assert report["complete"] is True and report["case_details_withheld"] is False and len(report["judgment_ids"]) == 5


def test_quarantined_groups_are_refused_for_every_purpose_including_audit(db_session, settings):
    from eval_tinder.db.models import Project

    project = db_session.get(Project, seed_demo(db_session, simulate_expert=False, settings=settings)["project_id"])
    trace = eligible_traces(db_session, project, "TRAIN")[0]
    quarantine_group(db_session, project.id, trace.group_id, reason="cross-partition duplicate")
    runtime = GraderRuntime.build(project, seed_grader_for(db_session, project), settings=settings)
    for purpose in (GradingPurpose.PROBE, GradingPurpose.BULK, GradingPurpose.AUDIT):
        with pytest.raises(SealedMaterial, match="QUARANTINED"):
            grade_trace(db_session, project, runtime, trace, purpose=purpose, settings=settings)


def test_refused_calls_record_no_exposure(db_session, settings):
    from eval_tinder.db.models import Project

    project = db_session.get(Project, seed_demo(db_session, simulate_expert=False, settings=settings)["project_id"])
    untouched = {
        a.group_id
        for a in db_session.scalars(select(PartitionAssignment).where(PartitionAssignment.project_id == project.id))
        if a.exposure_status == "UNTOUCHED"
    }
    traces, seen_groups = [], set()
    for t in eligible_traces(db_session, project, "TRAIN"):
        if t.group_id in untouched and t.group_id not in seen_groups:
            traces.append(t)
            seen_groups.add(t.group_id)
        if len(traces) == 3:
            break
    assert len(traces) == 3
    guard = BudgetGuard(max_calls=1, max_total_tokens=10**9, max_tokens_per_call=100)
    runtime = GraderRuntime.build(project, seed_grader_for(db_session, project), settings=settings, budget=guard)
    runs = grade_many(db_session, project, runtime, traces, purpose=GradingPurpose.PROBE, settings=settings)
    assert [r.status for r in runs] == ["OK", "BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED"]
    events = list(db_session.scalars(select(ExposureEvent).where(ExposureEvent.project_id == project.id, ExposureEvent.kind == "PROBE")))
    assert {e.group_id for e in events} == {traces[0].group_id}
    statuses = {
        a.group_id: a.exposure_status
        for a in db_session.scalars(select(PartitionAssignment).where(PartitionAssignment.project_id == project.id))
    }
    assert statuses[traces[1].group_id] == "UNTOUCHED" and statuses[traces[2].group_id] == "UNTOUCHED"


def test_review_request_listing_never_enumerates_audit_requests(db_session, session_factory, settings):
    fx = build_project(db_session, settings, reserve=30)
    lock(db_session, fx, planned_n=5, key="l1")
    db_session.commit()
    client = TestClient(create_app())
    assert client.get(f"/projects/{fx.project.id}/review-requests", params={"purpose": "AUDIT"}).status_code == 400
    rows = client.get(f"/projects/{fx.project.id}/review-requests").json()
    assert all(r["purpose"] != "AUDIT" for r in rows)
    assert client.get("/review-requests/does-not-exist").status_code == 404
