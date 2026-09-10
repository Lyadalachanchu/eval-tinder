"""M2 through the HTTP API: a budgeted, idempotent optimization run executed by the real worker and the
real ``dspy.GEPA`` integration (deterministic scripted fake models), candidate inspection with diffs and
development agreement, grader manifests without credentials, explicit shadow selection, and cancellation.

With the fixture's truthful-reporting labels the scripted reflection model proposes a truthful-reporting
rule, so the run is expected to recommend a candidate. That verifies the application's mechanics and
product rules; it is not evidence that optimization learns anything about real data.
"""
from __future__ import annotations

import json
import re

from eval_tinder.grader.signature import DEFAULT_SEED_INSTRUCTIONS
from eval_tinder.worker.main import drain
from tests.api.conftest import REVIEWER, seed_labeled_project
from tests.cases import DEV_CANARY, DEV_CASES, TRAIN_CANARY, TRAIN_CASES

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SEED_DEV_AGREEMENT = 0.5  # the completion-grading seed agrees with the truthful labels on d1 and d2 only


def test_optimization_round_candidates_graders_and_shadow_selection(client, worker):
    seeded = seed_labeled_project(client, worker)
    pid = seeded.id
    dash = client.get(f"/projects/{pid}").json()
    assert dash["labels"]["TRAIN"]["resolved"] == len(TRAIN_CASES) and dash["labels"]["DEV"]["resolved"] == len(DEV_CASES)
    assert dash["readiness"]["bootstrap_ready"] is True and dash["readiness"]["active_run"] is None
    assert dash["review_states"] == {"JUDGED": len(TRAIN_CASES) + len(DEV_CASES)}
    assert all(j["reviewer_id"] == REVIEWER for j in seeded.train_judgments + seeded.dev_judgments)
    seed_grader = next(g for g in client.get(f"/projects/{pid}/graders").json() if g["origin"] == "SEED")

    # ---- create: 202, frozen sizes, QUEUED; idempotent on the key; a second concurrent run is refused
    created = client.post(f"/projects/{pid}/optimization-runs", json={"idempotency_key": "opt-1", "max_metric_calls": 40})
    assert created.status_code == 202, created.text
    run = created.json()
    assert run["state"] == "QUEUED" and run["job_id"] and run["candidates"] == []
    assert run["train_size"] == len(TRAIN_CASES) and run["dev_size"] == len(DEV_CASES)
    assert run["train_snapshot_id"] != run["dev_snapshot_id"] and run["policy_epoch"] == 1
    assert run["seed_grader_id"] == seed_grader["id"] and run["seed_choice"] == "generic_seed"
    assert run["metric_version"] == "agreement-v1"
    assert run["config"]["max_metric_calls"] == 40 and run["budgets"]["max_metric_calls"] == 40
    assert run["config"]["preflight"]["train_size"] == len(TRAIN_CASES)
    assert run["budgets"]["estimate"]["cost_usd"] is None, "no pricing table configured -> no invented cost"
    assert run["result_summary"] == {} and run["usage"] == {} and run["finished_at"] is None
    assert client.get(f"/jobs/{run['job_id']}").json()["state"] == "QUEUED"

    replay = client.post(f"/projects/{pid}/optimization-runs", json={"idempotency_key": "opt-1", "max_metric_calls": 99})
    assert replay.status_code == 202 and replay.json()["id"] == run["id"]
    assert replay.json()["config"]["max_metric_calls"] == 40
    refused = client.post(f"/projects/{pid}/optimization-runs", json={"idempotency_key": "opt-other", "max_metric_calls": 40})
    assert refused.status_code == 400, refused.text
    assert f"run {run['id']} is already QUEUED" in refused.json()["detail"]
    assert [r["id"] for r in client.get(f"/projects/{pid}/optimization-runs").json()] == [run["id"]]
    assert client.get(f"/projects/{pid}").json()["readiness"]["active_run"] == run["id"]
    assert client.post(f"/projects/{pid}/optimization-runs", json={"max_metric_calls": 40}).status_code == 422

    # ---- execute through the real worker, handler, and dspy.GEPA adapter
    assert drain(worker) == 1
    job = client.get(f"/jobs/{run['job_id']}").json()
    assert job["state"] == "SUCCEEDED" and job["error"] is None, job
    got = client.get(f"/optimization-runs/{run['id']}")
    assert got.status_code == 200, got.text
    run = got.json()
    assert run["state"] == "SUCCEEDED" and run["finished_at"] and run["error"] is None
    summary = run["result_summary"]
    assert summary["improved"] is True and summary["partial"] is False
    assert summary["seed_agreement"] == SEED_DEV_AGREEMENT and summary["best_agreement"] == 1.0
    assert summary["comparison"]["recommend"] is True
    recommended_id = summary["recommended_grader_id"]
    assert recommended_id and recommended_id != seed_grader["id"]
    assert run["usage"]["by_role"]["grading"]["calls"] > 0 and run["usage"]["by_role"]["reflection"]["calls"] > 0
    assert "not evidence of production accuracy" in summary["note"]
    assert client.get(f"/projects/{pid}").json()["readiness"]["active_run"] is None
    assert client.get(f"/projects/{pid}").json()["runs"] == 1

    # ---- candidates: instruction text, diff from the seed, DEV agreement, seed and member flags
    candidates = run["candidates"]
    assert len(candidates) >= 2, "the seed plus at least one proposed candidate"
    assert [c["candidate_index"] for c in candidates] == sorted(c["candidate_index"] for c in candidates)
    assert len({c["grader_id"] for c in candidates}) == len(candidates)
    (seed_candidate,) = [c for c in candidates if c["is_seed"]]
    assert seed_candidate["candidate_index"] == 0 and seed_candidate["grader_id"] == seed_grader["id"]
    assert seed_candidate["instruction_text"] == DEFAULT_SEED_INSTRUCTIONS and seed_candidate["diff_from_seed"] == ""
    assert seed_candidate["parent_ids"] == [] and isinstance(seed_candidate["is_member"], bool)
    assert seed_candidate["evaluation"]["aggregate_metrics"]["agreement"] == SEED_DEV_AGREEMENT
    assert seed_candidate["evaluation"]["kind"] == "DEVELOPMENT_AGREEMENT"
    (best,) = [c for c in candidates if c["grader_id"] == recommended_id]
    assert best["is_seed"] is False and best["is_member"] is True and best["candidate_index"] >= 1
    assert best["parent_ids"] and all(HEX64.match(c["manifest_hash"]) for c in candidates)
    assert "truthful" in best["instruction_text"].lower()
    evaluation = best["evaluation"]
    assert evaluation["complete"] is True and evaluation["dev_snapshot_id"] == run["dev_snapshot_id"]
    assert evaluation["aggregate_metrics"]["agreement"] == 1.0 and evaluation["aggregate_metrics"]["false_passes"] == 0
    assert evaluation["aggregate_metrics"]["coverage"] == 1.0
    assert evaluation["aggregate_metrics"]["human_classes"] == {"PASS": 3, "FAIL": 1}
    assert set(evaluation["per_case_scores"]["app"]) == set(evaluation["verdicts"])
    assert all(v["status"] == "OK" for v in evaluation["verdicts"].values())
    diff = best["diff_from_seed"]
    assert diff.startswith("--- seed\n+++ candidate\n") and "@@" in diff
    assert any(line.startswith("+") and "truthful" in line.lower() for line in diff.splitlines())
    for c in candidates:
        assert isinstance(c["instruction_text"], str) and c["instruction_text"]
        assert c["evaluation"] is None or c["evaluation"]["kind"] == "DEVELOPMENT_AGREEMENT"
        for leaked in (TRAIN_CANARY, DEV_CANARY, REVIEWER):
            assert leaked not in c["instruction_text"], "labels, explanations, and bookkeeping never enter a grader"

    # ---- graders: manifest, hashes, lineage diff, evaluations; never a credential
    grader = client.get(f"/graders/{recommended_id}")
    assert grader.status_code == 200, grader.text
    g = grader.json()
    assert g["id"] == recommended_id and g["project_id"] == pid and g["origin"] == "GEPA"
    assert g["optimization_run_id"] == run["id"] and g["candidate_index"] == best["candidate_index"]
    assert g["instruction_text"] == best["instruction_text"] and g["parent_ids"] == best["parent_ids"]
    assert g["manifest"]["instruction_text"] == best["instruction_text"]
    assert g["manifest"]["policy_epoch"] == 1 and g["manifest"]["predictor_name"] == "judge"
    assert g["manifest"]["renderer_version"] == g["renderer_version"] and g["manifest"]["parser_version"] == g["parser_version"]
    assert g["manifest_hash"] == best["manifest_hash"] and HEX64.match(g["manifest_hash"])
    assert HEX64.match(g["pipeline_hash"]) and g["pipeline_hash"] != g["manifest_hash"]
    assert g["model_config"]["provider"] == "fake" and g["manifest"]["model_config"] == g["model_config"]
    assert g["diff_from_parent"] is not None and g["diff_from_parent"].startswith("--- parent\n+++ this\n")
    assert g["is_active_shadow"] is False and g["policy_epoch"] == 1
    assert [e["dev_snapshot_id"] for e in g["evaluations"]] == [run["dev_snapshot_id"]]
    assert g["evaluations"][0]["aggregate_metrics"]["agreement"] == 1.0
    lowered = json.dumps(g).lower()
    assert "sk-" not in lowered and "api_key" not in lowered and "api-key" not in lowered
    for leaked in (TRAIN_CANARY, DEV_CANARY, REVIEWER):
        assert leaked not in json.dumps(g)
    seed_view = client.get(f"/graders/{seed_grader['id']}").json()
    assert seed_view["diff_from_parent"] is None and seed_view["parent_ids"] == []
    assert seed_view["instruction_text"] == DEFAULT_SEED_INSTRUCTIONS
    assert seed_view["evaluations"][0]["aggregate_metrics"]["agreement"] == SEED_DEV_AGREEMENT
    assert "sk-" not in json.dumps(seed_view).lower() and "api_key" not in json.dumps(seed_view).lower()
    listed = client.get(f"/projects/{pid}/graders").json()
    assert len(listed) == client.get(f"/projects/{pid}").json()["graders"] == len(candidates)
    assert client.get("/graders/missing").status_code == 404

    # ---- explicit shadow selection: provisional, recorded with a reason, never automation
    chosen = client.post(f"/projects/{pid}/shadow-grader", json={"grader_id": recommended_id, "reason": "best DEV agreement"})
    assert chosen.status_code == 200, chosen.text
    assert chosen.json()["active_shadow_grader_id"] == recommended_id and chosen.json()["status"] == "PROVISIONAL"
    assert chosen.json()["history"][-1]["grader_id"] == recommended_id
    assert chosen.json()["history"][-1]["reason"] == "best DEV agreement" and chosen.json()["history"][-1]["previous"] is None
    dash = client.get(f"/projects/{pid}").json()
    assert dash["shadow_grader"]["id"] == recommended_id and "PROVISIONAL" in dash["shadow_grader"]["status"]
    assert dash["shadow_grader"]["manifest_hash"] == g["manifest_hash"]
    assert dash["project"]["active_shadow_grader_id"] == recommended_id
    assert dash["automation"] is None and dash["project"]["automation_policy_id"] is None
    assert dash["project"]["configuration"]["automation_enabled"] is False
    assert client.get(f"/graders/{recommended_id}").json()["is_active_shadow"] is True
    assert client.post(f"/projects/{pid}/shadow-grader", json={"grader_id": "missing", "reason": "x"}).status_code == 404
    assert client.post(f"/projects/{pid}/shadow-grader", json={"grader_id": recommended_id, "reason": ""}).status_code == 422

    # ---- traces: human labels stay HUMAN; a shadow prediction appears only where one was actually made
    traces = client.get(f"/projects/{pid}/traces").json()
    assert traces["total"] == len(TRAIN_CASES) + len(DEV_CASES)
    for item in traces["items"]:
        assert item["human_judgment"]["kind"] == "HUMAN" and item["human_judgment"]["verdict"] in {"PASS", "FAIL"}
        prediction = item["shadow_prediction"]
        if item["partition"] == "TRAIN":
            assert prediction is None, "no TRAIN trace has been graded by the shadow grader yet"
        elif prediction is not None:  # DEV traces were graded during the run's DEV evaluation
            assert prediction["kind"] == "MACHINE" and prediction["provisional"] is True
            assert prediction["grader_id"] == recommended_id and prediction["verdict"] in {"PASS", "FAIL", "REVIEW"}

    cleared = client.post(f"/projects/{pid}/shadow-grader", json={"grader_id": None, "reason": "back to blind review"})
    assert cleared.status_code == 200 and cleared.json()["active_shadow_grader_id"] is None
    assert cleared.json()["history"][-1]["previous"] == recommended_id
    dash = client.get(f"/projects/{pid}").json()
    assert dash["shadow_grader"] is None and dash["project"]["active_shadow_grader_id"] is None
    assert client.get(f"/graders/{recommended_id}").json()["is_active_shadow"] is False
    assert all(i["shadow_prediction"] is None for i in client.get(f"/projects/{pid}/traces").json()["items"])


def test_cancelling_a_queued_run_through_the_api(client, worker):
    pid = seed_labeled_project(client, worker, name="m2-cancel").id
    created = client.post(f"/projects/{pid}/optimization-runs", json={"idempotency_key": "cancel-1", "max_metric_calls": 40})
    assert created.status_code == 202 and created.json()["state"] == "QUEUED"
    run = created.json()

    cancelled = client.post(f"/optimization-runs/{run['id']}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["state"] == "CANCELLED" and cancelled.json()["finished_at"]
    assert "cancel" in (cancelled.json()["error"] or "").lower()
    job = client.get(f"/jobs/{run['job_id']}").json()
    assert job["state"] == "CANCELLED" and job["finished_at"]
    assert drain(worker) == 0, "a cancelled queued job never executes"
    after = client.get(f"/optimization-runs/{run['id']}").json()
    assert after["state"] == "CANCELLED" and after["candidates"] == [] and after["result_summary"] == {}
    dash = client.get(f"/projects/{pid}").json()
    assert dash["readiness"]["active_run"] is None and dash["runs"] == 1 and dash["graders"] == 1
    assert client.post(f"/optimization-runs/{run['id']}/cancel").json()["state"] == "CANCELLED", "idempotent"
    assert client.post("/optimization-runs/missing/cancel").status_code == 404

    # Cancellation unblocks the project: a new run can be queued and cancelled again.
    again = client.post(f"/projects/{pid}/optimization-runs", json={"idempotency_key": "cancel-2", "max_metric_calls": 40})
    assert again.status_code == 202 and again.json()["id"] != run["id"] and again.json()["state"] == "QUEUED"
    assert client.post(f"/optimization-runs/{again.json()['id']}/cancel").json()["state"] == "CANCELLED"
    assert drain(worker) == 0
    states = [r["state"] for r in client.get(f"/projects/{pid}/optimization-runs").json()]
    assert states == ["CANCELLED", "CANCELLED"]
