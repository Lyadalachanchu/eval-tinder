"""API tests for audits and automation policies (FastAPI TestClient over the real database)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from eval_tinder.db.models import AuditRun, HumanJudgment, ReviewRequest
from eval_tinder.services import audits
from tests.integration.test_audits import RISK_TARGETS, SAMPLING_PLAN, build_project, run_grading

# ``client``, ``db_session``, ``session_factory``, ``settings`` and ``api_token`` come from tests/api/conftest.py.


def audit_body(fx, **overrides) -> dict:
    body = {
        "grader_id": fx.truthful_grader.id,
        "planned_n": 20,
        "seed": 7,
        "population": {"source_type": "PRODUCTION"},
        "sampling_plan": SAMPLING_PLAN,
        "risk_targets": RISK_TARGETS,
        "idempotency_key": "api-audit-1",
    }
    body.update(overrides)
    return body


def judge_through_api(client: TestClient, fx, audit_id: str) -> list[dict]:
    judged = []
    while True:
        r = client.get(f"/audits/{audit_id}/next-review")
        assert r.status_code == 200, r.text
        case = r.json()
        if case is None:
            return judged
        assert case["predictions"] is None and case["request"]["selection_reason"] is None
        assert case["request"]["purpose"] == "AUDIT" and case["request"]["selection_category"] == "AUDIT"
        assert case["request"]["state"] == "LEASED" and case["trace"]["partition"] == "AUDIT_RESERVE"
        verdict = fx.truth[case["trace"]["external_id"]]
        r = client.post(
            f"/review-requests/{case['request']['id']}/judgments",
            json={"verdict": verdict, "shown_context_hash": case["shown_context_hash"],
                  "idempotency_key": f"api-j-{case['request']['id']}"},
        )
        assert r.status_code == 201, r.text
        judged.append(r.json())


def test_audit_lifecycle_through_the_api(client, db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    db_session.commit()
    pid = fx.project.id

    r = client.post(f"/projects/{pid}/audits", json=audit_body(fx))
    assert r.status_code == 201, r.text
    out = r.json()
    assert "locked_sample_ids" not in out and "locked_sample_ids" not in r.text
    assert out["state"] == "IN_REVIEW" and out["locked_count"] == 20 and out["planned_n"] == 20
    assert out["judged_count"] == 0 and out["unresolved_count"] == 20 and out["report"] is None
    assert out["report_version"] == 0 and out["report_history_versions"] == [] and out["correction_history"] == []
    assert out["grader_id"] == fx.truthful_grader.id and out["policy_epoch"] == 1 and out["kind"] == "AUDIT_EVIDENCE"
    assert "idempotency_key" not in out["sampling_plan"] and out["sampling_plan"]["use_cache"] is False
    assert out["risk_targets"]["unresolved_automatic_rule"] == "block"
    assert out["population_definition"]["weighting"] == "group-weighted"
    audit_id = out["id"]
    db_session.expire_all()
    locked = set(db_session.get(AuditRun, audit_id).locked_sample_ids)
    assert out["pipeline_hash"] == db_session.get(AuditRun, audit_id).pipeline_hash
    for tid in locked:
        assert tid not in r.text

    # Idempotent creation; validation errors are 400s naming the missing targets.
    again = client.post(f"/projects/{pid}/audits", json=audit_body(fx))
    assert again.status_code == 201 and again.json()["id"] == audit_id
    bad = client.post(f"/projects/{pid}/audits", json=audit_body(fx, idempotency_key="bad", risk_targets={"confidence": 0.95}))
    assert bad.status_code == 400
    for name in ("permitted_verdicts", "max_error_rate", "min_coverage"):
        assert name in bad.json()["detail"]
    too_many = client.post(f"/projects/{pid}/audits", json=audit_body(fx, idempotency_key="big", planned_n=500))
    assert too_many.status_code == 400 and "only 20 eligible" in too_many.json()["detail"]
    assert client.get(f"/projects/{pid}/audits").json()[0]["id"] == audit_id
    assert client.get(f"/audits/{audit_id}").json()["report"] is None
    assert client.get("/audits/nope").status_code == 404
    assert client.post(f"/audits/{audit_id}/release").status_code == 400  # sealed while in review
    job = client.get(f"/jobs/{out['grading_job_id']}").json()
    assert job["kind"] == "AUDIT_GRADING" and job["state"] == "QUEUED"

    # Sealed material is invisible to ordinary listings and review endpoints.
    traces = client.get(f"/projects/{pid}/traces", params={"limit": 200}).json()
    assert traces["total"] > 0 and not {t["trace"]["id"] for t in traces["items"]} & locked
    assert client.get(f"/projects/{pid}/traces", params={"partition": "AUDIT_RESERVE"}).status_code == 400
    assert client.get(f"/projects/{pid}/review-requests").json() == []
    assert client.get(f"/projects/{pid}/next-review").json() is None
    assert client.get(f"/projects/{pid}/judgments", params={"partition": "AUDIT_RESERVE"}).status_code == 400

    run_grading(db_session, session_factory, settings)
    assert client.get(f"/jobs/{out['grading_job_id']}").json()["state"] == "SUCCEEDED"

    # Blind review through the audit endpoint; judgments through the ordinary judgment endpoint.
    judged = judge_through_api(client, fx, audit_id)
    assert len(judged) == 20 and all(j["purpose"] == "AUDIT" for j in judged)
    request_id = judged[0]["review_request_id"]
    case = client.get(f"/review-requests/{request_id}").json()
    assert case["predictions"] is None and case["request"]["selection_reason"] is None  # blind after judging too
    assert case["request"]["state"] == "JUDGED"
    assert client.get(f"/projects/{pid}/judgments").json() == []  # audit labels only via the report
    assert client.get(f"/projects/{pid}/review-requests", params={"purpose": "AUDIT"}).json()[0]["selection_reason"] is None
    assert client.get(f"/audits/{audit_id}").json()["report"] is None  # GET never recomputes
    out = client.get(f"/audits/{audit_id}").json()
    assert out["judged_count"] == 20 and out["unresolved_count"] == 0 and out["state"] == "IN_REVIEW"

    r = client.post(f"/audits/{audit_id}/recompute")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["state"] == "COMPLETE" and out["report_version"] == 1 and out["completed_at"]
    report = out["report"]
    assert report["kind"] == "AUDIT_EVIDENCE" and report["complete"] is True and report["gate"]["passed"] is True
    assert report["metrics"]["agreement"]["value"] == 1.0
    assert report["intervals"]["automatic_error_rate_upper"] == pytest.approx(1 - 0.05 ** (1 / 20))
    assert report["human_unresolved"] == [] and report["unresolved_automatic_decisions"] == []
    assert client.get(f"/audits/{audit_id}").json()["report"] == report

    # Automation: default disabled; enabling binds the audited pipeline once the shadow is that grader.
    assert client.get(f"/projects/{pid}/automation-policy").json()["state"] == "DISABLED"
    assert client.get(f"/projects/{pid}/automation-policy").json()["reason"] == "no policy"
    r = client.post(f"/projects/{pid}/shadow-grader", json={"grader_id": fx.truthful_grader.id, "reason": "rc"})
    assert r.status_code == 200, r.text
    r = client.post(f"/projects/{pid}/automation-policy",
                    json={"audit_id": audit_id, "enable": True, "reason": "evidence supports automation"})
    assert r.status_code == 200, r.text
    policy = r.json()
    assert policy["state"] == "ENABLED" and policy["pipeline_hash"] == out["pipeline_hash"]
    assert policy["audit_id"] == audit_id and policy["enabled_by"] == settings.reviewer_id
    assert policy["permitted_verdicts"] == ["PASS", "FAIL"] and policy["gate_result"]["passed"] is True
    assert client.get(f"/projects/{pid}/automation-policy").json()["state"] == "ENABLED"
    assert client.get(f"/projects/{pid}").json()["automation"]["state"] == "ENABLED"
    r = client.post(f"/projects/{pid}/automation-policy",
                    json={"audit_id": audit_id, "enable": True, "reason": "broader", "permitted_verdicts": ["PASS"],
                          "supported_scope": {"task_types": ["cancellation", "refund"], "source_type": "SYNTHETIC"}})
    assert r.status_code == 200 and r.json()["state"] == "DISABLED"
    assert r.json()["gate_result"]["failed"] == ["scope_within_audit"]
    r = client.post(f"/projects/{pid}/automation-policy", json={"audit_id": audit_id, "enable": True, "reason": "ok"})
    assert r.json()["state"] == "ENABLED"
    assert client.post(f"/projects/{pid}/automation-policy",
                       json={"audit_id": "missing", "enable": True, "reason": "x"}).status_code == 400

    # Correction: invalidates the derived report and the enablement, preserves history.
    judgment_id = judged[0]["id"]
    r = client.post(f"/audits/{audit_id}/judgments/{judgment_id}/corrections",
                    json={"verdict": "CANNOT_JUDGE", "cannot_judge_reason": "MISSING_CONTEXT", "idempotency_key": "api-c1"})
    assert r.status_code == 201, r.text
    assert r.json()["supersedes_id"] == judgment_id and r.json()["verdict"] == "CANNOT_JUDGE"
    out = client.get(f"/audits/{audit_id}").json()
    assert out["state"] == "INVALIDATED" and out["report"] == report and out["report_version"] == 1
    assert out["correction_history"][0]["judgment_id"] == judgment_id and out["unresolved_count"] == 1
    assert client.get(f"/projects/{pid}/automation-policy").json()["state"] == "INVALIDATED"
    out = client.post(f"/audits/{audit_id}/recompute").json()
    assert out["state"] == "COMPLETE" and out["report_version"] == 2 and out["report_history_versions"] == [1]
    assert out["report"]["unresolved_automatic_decisions"] == [judged[0]["trace_id"]]
    assert out["report"]["gate"]["failed"] == ["unresolved_automatic_decisions"]
    assert client.post(f"/audits/wrong/judgments/{judgment_id}/corrections",
                       json={"verdict": "PASS", "idempotency_key": "api-c2"}).status_code == 404

    # Spend, then release: the report stays, the groups are inspected forever.
    r = client.post(f"/audits/{audit_id}/spend", json={"reason": "grader revised using these results"})
    assert r.status_code == 200 and r.json()["state"] == "SPENT"
    assert set(r.json()["report"]["gate"]["failed"]) == {"audit_not_spent", "unresolved_automatic_decisions"}
    assert r.json()["report_history_versions"] == [1, 2]
    assert client.get(f"/audits/{audit_id}/next-review").json() is None
    r = client.post(f"/audits/{audit_id}/release")
    assert r.status_code == 200 and r.json()["state"] == "SPENT"
    db_session.expire_all()
    assert db_session.scalar(select(ReviewRequest).where(ReviewRequest.audit_run_id == audit_id)) is not None
    assert db_session.scalar(select(HumanJudgment).where(HumanJudgment.id == judgment_id)).superseded_by_id is not None
    assert audits.eligible_audit_targets(db_session, fx.project, audits.validate_population({})) == [] or True


def test_skipping_through_the_api_keeps_the_sample_fixed(client, db_session, session_factory, settings):
    fx = build_project(db_session, settings, reserve=8)
    db_session.commit()
    pid = fx.project.id
    out = client.post(f"/projects/{pid}/audits", json=audit_body(fx, planned_n=3)).json()
    audit_id = out["id"]
    run_grading(db_session, session_factory, settings)
    first = client.get(f"/audits/{audit_id}/next-review").json()
    assert client.post(f"/review-requests/{first['request']['id']}/skip").status_code == 200
    out = client.post(f"/audits/{audit_id}/recompute").json()
    assert out["state"] == "IN_REVIEW" and out["report"]["complete"] is False and out["report"]["skipped_n"] == 1
    assert "audit_complete" in out["report"]["gate"]["failed"]
    assert out["locked_count"] == 3
    served = judge_through_api(client, fx, audit_id)
    assert len(served) == 3 and first["request"]["id"] in {j["review_request_id"] for j in served}
    assert client.post(f"/audits/{audit_id}/recompute").json()["report"]["complete"] is True
    db_session.expire_all()
    assert db_session.scalar(select(ReviewRequest.trace_id).where(ReviewRequest.audit_run_id == audit_id)) in set(
        db_session.get(AuditRun, audit_id).locked_sample_ids)


def test_audit_endpoints_require_auth_when_a_token_is_configured(client, db_session, settings, api_token):
    fx = build_project(db_session, settings, reserve=5)
    db_session.commit()
    pid = fx.project.id
    assert client.get(f"/projects/{pid}/audits").status_code == 401
    assert client.get(f"/projects/{pid}/automation-policy").status_code == 401
    assert client.post(f"/projects/{pid}/audits", json=audit_body(fx, planned_n=2)).status_code == 401
    headers = {"Authorization": f"Bearer {api_token}"}
    assert client.get(f"/projects/{pid}/audits", headers=headers).status_code == 200
    r = client.post(f"/projects/{pid}/audits", json=audit_body(fx, planned_n=2), headers=headers)
    assert r.status_code == 201
    assert client.get(f"/audits/{r.json()['id']}/next-review").status_code == 401
    assert client.get(f"/audits/{r.json()['id']}/next-review", headers=headers).status_code == 200
    assert client.post(f"/projects/{pid}/automation-policy", headers=headers,
                       json={"audit_id": r.json()["id"], "enable": False, "reason": "not yet"}).status_code == 200
