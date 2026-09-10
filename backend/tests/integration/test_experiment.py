"""M5: demo seeding and the random-vs-committee learning experiment (fake provider, deterministic)."""
from __future__ import annotations

import json

import pytest

from eval_tinder.db.enums import HumanVerdict
from eval_tinder.db.models import HumanJudgment, Project, ReviewRequest
from eval_tinder.demo import DEMO_TRUTH, SIMULATED_REVIEWER, load_truth, seed_demo
from eval_tinder.experiments.selection_experiment import run_experiment
from eval_tinder.services.review import resolved_label_counts


def test_seed_demo_creates_project_import_and_batches(db_session, settings):
    result = seed_demo(db_session, simulate_expert=False, settings=settings)
    project = db_session.get(Project, result["project_id"])
    assert project is not None and "SYNTHETIC" in project.description
    assert result["import_counts"]["inserted"] > 50
    assert result["seed_train_requests"] == 12 and result["dev_requests"] == 8
    assert db_session.query(HumanJudgment).count() == 0
    counts = resolved_label_counts(db_session, project)
    assert counts["TRAIN"]["resolved"] == 0


def test_simulated_expert_labels_are_marked_simulated(db_session, settings):
    result = seed_demo(db_session, simulate_expert=True, settings=settings)
    judgments = db_session.query(HumanJudgment).all()
    assert result["simulated_judgments"] == len(judgments) > 0
    assert all(j.reviewer_id == SIMULATED_REVIEWER and j.explanation.startswith("[SIMULATED]") for j in judgments)
    assert all(j.verdict in {v.value for v in HumanVerdict} for j in judgments)
    truth = load_truth(DEMO_TRUTH)
    for j in judgments:
        req = db_session.get(ReviewRequest, j.review_request_id)
        assert req.state == "JUDGED"
    assert any(j.verdict == "PASS" for j in judgments) and any(j.verdict == "FAIL" for j in judgments)
    assert len(truth) >= 60


@pytest.mark.timeout(600)
def test_experiment_runs_both_strategies_and_reports_measured_outcome(session_factory, settings, db_session):
    result = run_experiment(session_factory, labeling_budget=30, batch_size=5, max_metric_calls=40, seed=5, settings=settings)
    assert set(result["strategies"]) == {"random", "committee"}
    for strategy, d in result["strategies"].items():
        assert d["labels_used"] <= 30
        assert d["rounds"] >= 1
        assert d["benchmark"]["benchmark_size"] > 0
        assert d["benchmark"]["note"].startswith("Experiment benchmark")
        assert d["trajectory"][0]["round"] == 0
        assert d["simulated_expert_minutes"] > 0
        assert d["actual_expert_minutes"] is None
    comparison = result["comparison"]
    assert comparison["agreement"]["outcome"] in {"committee_better", "tie", "random_better", "not_estimable"}
    json.dumps(result, default=str)  # serializable report
    # projects are isolated per strategy
    ids = {d["project_id"] for d in result["strategies"].values()}
    assert len(ids) == 2
