"""API tests for selection rounds (M3): queueing, idempotency, blind review requests, cancellation, readiness.

The project data (imports, truthful-reporting labels, one fake optimization run) is built through the services by
``tests.integration.test_selection.build_fixture``; everything a user would do afterwards goes through the HTTP API
and the real worker registry.
"""
from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select

from eval_tinder.db.enums import ReviewRequestState
from eval_tinder.db.models import ReviewRequest, SelectionRound
from eval_tinder.worker.main import drain
from tests.integration.test_selection import (
    ROUND_SEED,
    TEMPLATES,
    Fixture,
    build_fixture,
    by_category,
    picks_of,
    random_draws,
)


def start_round(client: TestClient, project_id: str, *, key: str, seed: int = ROUND_SEED) -> dict[str, Any]:
    resp = client.post(f"/projects/{project_id}/selection-rounds", json={"seed": seed, "idempotency_key": key})
    assert resp.status_code == 202, resp.text
    return resp.json()


def get_round(client: TestClient, round_id: str) -> dict[str, Any]:
    resp = client.get(f"/selection-rounds/{round_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def get_request(client: TestClient, request_id: str) -> dict[str, Any]:
    resp = client.get(f"/review-requests/{request_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def judge(client: TestClient, request_id: str, *, verdict: str, explanation: str = "") -> dict[str, Any]:
    case = get_request(client, request_id)
    resp = client.post(
        f"/review-requests/{request_id}/judgments",
        json={"verdict": verdict, "explanation": explanation, "shown_context_hash": case["shown_context_hash"],
              "active_review_ms": 900, "idempotency_key": f"api-judge:{request_id}"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def demo_request(db_session, fx: Fixture, round_id: str) -> ReviewRequest:
    return db_session.scalars(
        select(ReviewRequest).where(ReviewRequest.selection_round_id == round_id,
                                    ReviewRequest.trace_id == fx.trace_ids[fx.demo])
    ).one()


# ---------------------------------------------------------------- lifecycle


def test_selection_round_lifecycle_through_the_api(client, worker, db_session, session_factory, settings):
    fx = build_fixture(db_session, settings, session_factory, name="api-selection")
    pid = fx.project_id

    queued = start_round(client, pid, key="api-round-1")
    assert queued["state"] == "QUEUED" and queued["project_id"] == pid and queued["seed"] == ROUND_SEED
    assert queued["job_id"] and queued["committee_ids"] == [] and queued["selected_requests"] == []
    assert queued["strategy_version"] == "select-v1" and queued["error"] is None
    replay = start_round(client, pid, key="api-round-1", seed=99)
    assert replay["id"] == queued["id"] and replay["job_id"] == queued["job_id"] and replay["seed"] == ROUND_SEED
    refused = client.post(f"/projects/{pid}/selection-rounds", json={"seed": 1, "idempotency_key": "api-round-2"})
    assert refused.status_code == 400 and f"{queued['id']} is already QUEUED" in refused.json()["detail"]
    job = client.get(f"/jobs/{queued['job_id']}").json()
    assert job["kind"] == "SELECTION" and job["state"] == "QUEUED" and job["project_id"] == pid
    assert client.get("/selection-rounds/does-not-exist").status_code == 404
    assert client.get("/projects/does-not-exist/selection-rounds").status_code == 404
    assert client.post("/projects/does-not-exist/selection-rounds", json={"idempotency_key": "x"}).status_code == 404

    assert drain(worker) == 1

    done = get_round(client, queued["id"])
    assert done["state"] == "COMPLETE" and done["error"] is None and done["job_id"] == queued["job_id"]
    assert len(done["selected_requests"]) == 10 and done["batch_id"] == done["id"]
    assert set(done["committee_ids"]) == {fx.graders["seed"], fx.graders["truthful"]}
    report = done["committee_report"]
    assert report["members"] == done["committee_ids"] and report["dev_snapshot_id"] == fx.dev_snapshot_id
    assert report["reason"] == "no_behavioral_diversity" and report["diversity_claimed"] is False
    assert set(report["shortlist"]["shortlisted"]) == set(fx.graders.values()) and report["shortlist"]["exclusions"] == []
    assert report["notes"] == [] and report["usage"]["exhausted"] is False and report["usage"]["calls"] > 0
    assert done["probe_size"] == done["pool_size"] == 14
    assert done["context_repair"] == [fx.trace_ids[fx.unclear]]
    assert "not a calibrated error probability" in done["note"]
    categories = {e["category"] for e in done["selected_requests"]}
    assert categories <= {"DISAGREEMENT", "COVERAGE", "RANDOM"} and "DISAGREEMENT" in categories
    assert all(e["expected_reading_length"] > 0 and e["request_id"] and e["trace_id"] for e in done["selected_requests"])
    demo_entry = next(e for e in done["selected_requests"] if e["trace_id"] == fx.trace_ids[fx.demo])
    assert demo_entry["category"] == "DISAGREEMENT"
    job = client.get(f"/jobs/{queued['job_id']}").json()
    assert job["state"] == "SUCCEEDED" and job["result"]["partial"] is False
    assert job["result"]["round_id"] == done["id"] and job["result"]["batch_size"] == 10
    listed = client.get(f"/projects/{pid}/selection-rounds").json()
    assert [r["id"] for r in listed] == [done["id"]] and listed[0]["state"] == "COMPLETE"
    # The queue is free again once the round completed.
    assert start_round(client, pid, key="api-round-2")["id"] != done["id"]


# ---------------------------------------------------------------- blind review, reveal, readiness


def test_selected_requests_are_blind_until_judged_through_the_api(client, worker, db_session, session_factory,
                                                                  settings):
    fx = build_fixture(db_session, settings, session_factory, name="api-blind")
    pid = fx.project_id
    queued = start_round(client, pid, key="api-blind-round")
    assert drain(worker) == 1
    rnd = get_round(client, queued["id"])
    demo = demo_request(db_session, fx, rnd["id"])
    seed_id, truthful_id = fx.graders["seed"], fx.graders["truthful"]

    before = get_request(client, demo.id)
    assert before["request"]["selection_category"] == "HIDDEN"
    assert before["request"]["selection_reason"] is None and before["predictions"] is None
    assert before["request"]["state"] == "OPEN" and before["request"]["purpose"] == "TRAIN"
    assert before["request"]["batch_id"] == rnd["id"]
    assert before["trace"]["external_id"] == fx.demo and before["trace"]["partition"] == "TRAIN"
    assert before["trace"]["output"] == TEMPLATES["queued_processing"].output
    assert before["shown_context_hash"] == before["trace"]["content_hash"]
    open_requests = client.get(f"/projects/{pid}/review-requests", params={"state": "OPEN"}).json()
    by_id = {r["id"]: r for r in open_requests}
    assert len(open_requests) == 11  # the batch plus the fixture's still-open seed request
    assert all(by_id[e["request_id"]]["selection_category"] == "HIDDEN" for e in rnd["selected_requests"])
    assert all(by_id[e["request_id"]]["selection_reason"] is None for e in rnd["selected_requests"])
    assert [r["selection_category"] for r in open_requests if r["batch_id"] != rnd["id"]] == ["SEED"]

    judgment = judge(client, demo.id, verdict="PASS", explanation="The answer truthfully says it is still processing.")
    assert judgment["verdict"] == "PASS" and judgment["purpose"] == "TRAIN" and judgment["trace_id"] == demo.trace_id

    after = get_request(client, demo.id)
    assert after["request"]["state"] == "JUDGED" and after["request"]["judgment_id"] == judgment["id"]
    assert after["request"]["selection_category"] == "DISAGREEMENT"
    reason = after["request"]["selection_reason"]
    assert reason["category"] == "DISAGREEMENT" and reason["rank"] == 1 and reason["score"] == 0.5
    assert reason["committee_size"] == 2 and reason["committee_votes_hidden_until_judged"] is True
    assert reason["votes"]["counts"] == {"PASS": 1, "FAIL": 1, "REVIEW": 0} and reason["votes"]["valid_count"] == 2
    predictions = after["predictions"]
    assert predictions and all(p["kind"] == "MACHINE" and p["provisional"] is True for p in predictions)
    assert all(p["status"] == "OK" and p["explanation"] and p["verdict"] in ("PASS", "FAIL") for p in predictions)
    votes = {(p["grader_id"], p["verdict"]) for p in predictions}
    assert votes == {(seed_id, "FAIL"), (truthful_id, "PASS"), (fx.graders["variant"], "FAIL")}
    assert len(predictions) == 5  # three probe verdicts plus the two (cached) committee pool votes
    assert {p["grader_id"] for p in predictions} == set(rnd["committee_report"]["shortlist"]["shortlisted"])
    # Judging one request reveals nothing about the others.
    other = next(e["request_id"] for e in rnd["selected_requests"] if e["request_id"] != demo.id)
    assert get_request(client, other)["request"]["selection_category"] == "HIDDEN"

    # Judging the whole batch (ten TRAIN labels) makes the project ready for the next optimization round.
    picks = picks_of(db_session, db_session.get(SelectionRound, rnd["id"]), fx)
    assert len(picks) == 10 and len(random_draws(picks)) == 2 and len(by_category(picks)["COVERAGE"]) == 2
    for p in picks:
        if p.request.id == demo.id:
            continue
        t = TEMPLATES[p.template]
        judge(client, p.request.id, verdict=t.label, explanation=t.explanation)
    dashboard = client.get(f"/projects/{pid}").json()
    readiness = dashboard["readiness"]
    assert readiness["new_train_labels_since_last_run"] == 10 and readiness["ready_to_optimize_again"] is True
    assert readiness["last_run_id"] == fx.run_id and dashboard["labels"]["TRAIN"]["resolved"] == 6 + 10
    assert dashboard["review_states"].get(ReviewRequestState.JUDGED) == 6 + 4 + 10
    listed = client.get(f"/projects/{pid}/review-requests", params={"purpose": "TRAIN", "state": "JUDGED"}).json()
    revealed = [r for r in listed if r["batch_id"] == rnd["id"]]
    assert len(revealed) == 10 and all(r["selection_category"] != "HIDDEN" and r["selection_reason"] for r in revealed)
    assert {r["selection_category"] for r in revealed} <= {"DISAGREEMENT", "COVERAGE", "RANDOM"}


# ---------------------------------------------------------------- cancellation


def test_cancelling_a_queued_selection_job_through_the_api(client, worker, db_session, session_factory, settings):
    fx = build_fixture(db_session, settings, session_factory, name="api-cancel", optimize=False)
    pid = fx.project_id
    queued = start_round(client, pid, key="api-cancel-round")
    cancelled = client.post(f"/jobs/{queued['job_id']}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["state"] == "CANCELLED"
    assert drain(worker) == 0
    rnd = get_round(client, queued["id"])
    assert rnd["state"] == "QUEUED" and rnd["selected_requests"] == [] and rnd["batch_id"] is None
    assert rnd["pool_size"] == 0 and rnd["probe_size"] == 0
    open_requests = client.get(f"/projects/{pid}/review-requests", params={"state": "OPEN"}).json()
    assert [r["selection_category"] for r in open_requests] == ["SEED"]  # only the fixture's own open request
    # Bug fix: the cancelled round no longer blocks the project; it is finalized when the next round is created.
    replacement = start_round(client, pid, key="api-after-cancel")
    assert replacement["id"] != queued["id"] and replacement["state"] == "QUEUED"
    stale = get_round(client, queued["id"])
    assert stale["state"] == "FAILED" and "CANCELLED" in stale["error"] and stale["selected_requests"] == []
    assert drain(worker) == 1
    assert get_round(client, replacement["id"])["state"] == "COMPLETE"
    assert [r["state"] for r in client.get(f"/projects/{pid}/selection-rounds").json()] == ["COMPLETE", "FAILED"]
