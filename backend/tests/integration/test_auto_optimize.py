"""Opt-in automatic optimization rounds: off by default, budget-respecting, never overlapping."""
from __future__ import annotations

from eval_tinder.db.enums import ReviewPurpose
from eval_tinder.db.models import OptimizationRun, Project, ReviewRequest
from eval_tinder.demo import SimulatedExpert, load_truth, seed_demo
from eval_tinder.services.optimization import maybe_auto_optimize, run_readiness
from eval_tinder.services.review import BatchSpec, create_review_batch


def _labeled_project(db_session, settings, *, automatic: bool) -> Project:
    result = seed_demo(
        db_session, simulate_expert=True, settings=settings,
        configuration={
            "automatic_optimization": automatic, "new_train_labels_per_round": 3, "gepa_max_metric_calls": 40,
            # the simulated expert may answer CANNOT_JUDGE for a few seed cases; keep bootstrap thresholds reachable
            "bootstrap_train_labels": 6, "bootstrap_dev_labels": 4,
        },
    )
    project = db_session.get(Project, result["project_id"])
    # bootstrap thresholds also size the seed batches; a CANNOT_JUDGE answer would leave the
    # project one label short, so label a few extra random TRAIN cases as well
    expert = SimulatedExpert(load_truth())
    create_review_batch(db_session, project, BatchSpec(purpose=ReviewPurpose.TRAIN, kind="RANDOM", size=4, seed=3))
    expert.answer_open_requests(db_session, project, purpose=ReviewPurpose.TRAIN)
    return project


def test_default_is_off(db_session, settings):
    project = _labeled_project(db_session, settings, automatic=False)
    assert maybe_auto_optimize(db_session, project, settings=settings) is None
    assert db_session.query(OptimizationRun).count() == 0


def test_opt_in_enqueues_one_run_and_never_overlaps(db_session, settings):
    project = _labeled_project(db_session, settings, automatic=True)
    run = maybe_auto_optimize(db_session, project, settings=settings)
    assert run is not None and run.state == "QUEUED" and run.config["label"] == "automatic"
    # a second call while the run is queued does nothing (no race to promote)
    assert maybe_auto_optimize(db_session, project, settings=settings) is None
    assert db_session.query(OptimizationRun).count() == 1
    # once the run finished, more labels are needed before the next automatic round
    run.state = "NO_IMPROVEMENT"
    db_session.flush()
    assert maybe_auto_optimize(db_session, project, settings=settings) is None
    expert = SimulatedExpert(load_truth())
    # CANNOT_JUDGE answers are not resolved labels, so label a comfortably larger batch
    create_review_batch(db_session, project, BatchSpec(purpose=ReviewPurpose.TRAIN, kind="RANDOM", size=8, seed=9))
    expert.answer_open_requests(db_session, project, purpose=ReviewPurpose.TRAIN)
    readiness = run_readiness(db_session, project)
    assert readiness["new_train_labels_since_last_run"] >= 3 and readiness["ready_to_optimize_again"]
    second = maybe_auto_optimize(db_session, project, settings=settings)
    assert second is not None and second.id != run.id
    assert db_session.query(ReviewRequest).filter_by(purpose="AUDIT").count() == 0
