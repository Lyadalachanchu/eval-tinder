"""Optimization service: frozen snapshots, budgeted jobs, candidate persistence, DEV comparison, recommendation.

Most tests drive ``FakeOptimizerService`` (fixed candidate texts, real metric and grading path) through the
real worker handler; one test runs the real ``dspy.GEPA`` integration end to end. Neither proves that
optimization learns anything about real data: they verify the application's mechanics and product rules.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from eval_tinder.db.enums import (
    GraderOrigin,
    GradingPurpose,
    GradingStatus,
    JobKind,
    JobState,
    OptimizationState,
    Partition,
    ReviewPurpose,
)
from eval_tinder.db.models import CandidateEvaluation, DatasetSnapshot, GraderVersion, GradingRun, Job, OptimizationRun
from eval_tinder.domain.manifest import METRIC_VERSION, GraderManifest
from eval_tinder.domain.metrics import NOT_ESTIMABLE
from eval_tinder.gepa.service import FakeOptimizerService, GepaOptimizerService
from eval_tinder.grader.signature import DEFAULT_SEED_INSTRUCTIONS
from eval_tinder.llm.fakes import (
    TRUTHFUL_KEYWORDS,
    ScriptedGradingLM,
    ScriptedReflectionLM,
    keyword_switch_policy,
    truthful_reflection_proposer,
)
from eval_tinder.services import optimization as opt
from eval_tinder.services.optimization import (
    OptimizationError,
    RunRequest,
    cancel_run,
    clear_shadow,
    compare_evaluations,
    create_run,
    evaluate_on_dev,
    execute_run,
    run_readiness,
    select_shadow,
)
from eval_tinder.services.projects import create_grader_version
from eval_tinder.services.snapshots import freeze_snapshot
from eval_tinder.worker.handlers import build_handlers
from eval_tinder.worker.main import Worker, drain
from tests.cases import DEV_CANARY, DEV_CASES, TRAIN_CANARY, TRAIN_CASES
from tests.integration.test_grading import (
    TRUTHFUL_INSTRUCTIONS,
    import_cases,
    label_cases,
    labeled_project,
    make_project,
    variants,
)

NON_IMPROVING_INSTRUCTIONS = DEFAULT_SEED_INSTRUCTIONS + "\nKeep explanations brief and cite the tool status."
assert not any(k in NON_IMPROVING_INSTRUCTIONS.lower() for k in TRUTHFUL_KEYWORDS)
assert any(k in TRUTHFUL_INSTRUCTIONS.lower() for k in TRUTHFUL_KEYWORDS)

# Under the fixture's truthful-reporting labels the seed (completion behaviour) agrees on d1 and d2 only.
SEED_DEV_AGREEMENT = 0.5
DEV_IDS_BY_KEY = {"d1": 1.0, "d2": 1.0, "d3": 0.0, "d4": 0.0}


# ----------------------------------------------------------------- helpers


def count(session, model) -> int:
    return session.scalar(select(func.count()).select_from(model))


def worker_with_fake(settings, session_factory, monkeypatch, fake: FakeOptimizerService) -> Worker:
    """The real handler registry; only the optimizer implementation is swapped for the deterministic fake."""
    monkeypatch.setattr(opt, "GepaOptimizerService", lambda: fake)
    return Worker(build_handlers(), settings=settings, worker_id="opt-worker", session_factory=session_factory)


def run_through_worker(db_session, session_factory, settings, monkeypatch, fx, *, candidates, request=None, key="run-1"):
    run, job = create_run(db_session, fx.project, request or RunRequest(), idempotency_key=key, settings=settings)
    db_session.commit()
    fake = FakeOptimizerService(candidates)
    assert drain(worker_with_fake(settings, session_factory, monkeypatch, fake)) == 1
    db_session.expire_all()
    return db_session.get(OptimizationRun, run.id), db_session.get(Job, job.id), fake


def evaluation_for(session, grader_id: str, dev_snapshot_id: str) -> CandidateEvaluation:
    return session.scalars(
        select(CandidateEvaluation).where(
            CandidateEvaluation.grader_id == grader_id, CandidateEvaluation.dev_snapshot_id == dev_snapshot_id
        )
    ).one()


def run_candidates(session, run_id: str) -> list[GraderVersion]:
    return list(
        session.scalars(
            select(GraderVersion).where(GraderVersion.optimization_run_id == run_id).order_by(GraderVersion.candidate_index)
        )
    )


# ----------------------------------------------------------------- create_run


def test_create_run_freezes_snapshots_and_enqueues_idempotently(db_session, settings):
    fx = labeled_project(db_session, settings)
    run, job = create_run(db_session, fx.project, RunRequest(label="first round"), idempotency_key="k1", settings=settings)

    train = db_session.get(DatasetSnapshot, run.train_snapshot_id)
    dev = db_session.get(DatasetSnapshot, run.dev_snapshot_id)
    assert sorted(train.ordered_trace_ids) == sorted(t.id for t in fx.train.values())
    assert sorted(train.ordered_judgment_ids) == sorted(j.id for j in fx.train_labels.values())
    assert sorted(dev.ordered_trace_ids) == sorted(t.id for t in fx.dev.values())
    assert sorted(dev.ordered_judgment_ids) == sorted(j.id for j in fx.dev_labels.values())
    assert (train.partition, dev.partition, train.policy_epoch) == (Partition.TRAIN, Partition.DEV, 1)
    assert set(train.ordered_trace_ids).isdisjoint(dev.ordered_trace_ids)

    assert run.seed_choice == "generic_seed" and run.seed_grader_id == fx.seed.id
    assert run.state == OptimizationState.QUEUED and run.metric_version == METRIC_VERSION and run.policy_epoch == 1
    assert run.config["preflight"] == {
        "train_size": 6, "dev_size": 4, "min_useful_metric_calls": 2 * 4 + 2 * 3, "estimated_full_evals": 30.0,
    }
    assert run.config["label"] == "first round" and run.config["max_metric_calls"] == 300
    assert run.budgets["max_metric_calls"] == 300
    assert run.budgets["max_provider_calls"] == settings.max_provider_calls_per_job
    assert run.budgets["max_total_tokens"] == settings.max_total_tokens_per_job
    assert run.budgets["max_tokens_per_call"] == settings.max_tokens_per_call
    estimate = run.budgets["estimate"]
    assert estimate["grading_calls"] > 300 and estimate["reflection_calls"] >= 1
    assert "cost_usd" in estimate and estimate["cost_usd"] is None, "no pricing table configured -> no invented cost"
    assert "Estimate only" in estimate["note"]
    assert run.artifact_path and run.artifact_path.endswith(run.id)

    assert job.kind == JobKind.OPTIMIZATION and job.state == JobState.QUEUED and job.max_attempts == 1
    assert job.payload == {"run_id": run.id, "project_id": fx.project.id} and job.payload_ref == run.id
    assert job.idempotency_key == "k1" and run.job_id == job.id

    run_again, job_again = create_run(db_session, fx.project, RunRequest(label="retry"), idempotency_key="k1", settings=settings)
    assert (run_again.id, job_again.id) == (run.id, job.id)
    assert run_again.config["label"] == "first round"
    assert count(db_session, OptimizationRun) == 1 and count(db_session, Job) == 1


def test_second_create_run_while_one_is_queued_is_refused(db_session, settings):
    fx = labeled_project(db_session, settings)
    run, _ = create_run(db_session, fx.project, RunRequest(), idempotency_key="k1", settings=settings)
    with pytest.raises(OptimizationError, match=f"run {run.id} is already QUEUED"):
        create_run(db_session, fx.project, RunRequest(), idempotency_key="k2", settings=settings)
    assert count(db_session, OptimizationRun) == 1 and count(db_session, Job) == 1


def test_create_run_without_dev_labels_fails_preflight_and_creates_nothing(db_session, settings):
    fx = labeled_project(db_session, settings, dev=[])
    import_cases(db_session, fx.project, DEV_CASES, Partition.DEV)  # DEV traces exist but carry no label
    with pytest.raises(OptimizationError, match="DEV"):
        create_run(db_session, fx.project, RunRequest(), idempotency_key="k1", settings=settings)
    assert count(db_session, OptimizationRun) == 0 and count(db_session, Job) == 0


# ----------------------------------------------------------------- execution through the worker


def test_worker_run_persists_candidates_evaluations_and_recommendation(db_session, session_factory, settings, monkeypatch):
    fx = labeled_project(db_session, settings)
    run, job = create_run(db_session, fx.project, RunRequest(), idempotency_key="run-1", settings=settings)
    db_session.commit()
    # A label submitted after the run was created belongs to the next snapshot, never to this run.
    late_case = variants(TRAIN_CASES[:1], 1, "late")
    late_trace = import_cases(db_session, fx.project, late_case, Partition.TRAIN)[late_case[0].key]
    label_cases(db_session, fx.project, {late_case[0].key: late_trace}, late_case, ReviewPurpose.TRAIN)
    db_session.commit()
    seed_before = (fx.seed.instruction_text, fx.seed.manifest_hash, json.dumps(fx.seed.manifest, sort_keys=True))

    fake = FakeOptimizerService([TRUTHFUL_INSTRUCTIONS])
    assert drain(worker_with_fake(settings, session_factory, monkeypatch, fake)) == 1
    db_session.expire_all()
    run, job = db_session.get(OptimizationRun, run.id), db_session.get(Job, job.id)

    assert fake.runs == [{"seed": DEFAULT_SEED_INSTRUCTIONS, "train": 6, "dev": 4, "run_dir": run.artifact_path}]
    assert job.state == JobState.SUCCEEDED and job.result["state"] == OptimizationState.SUCCEEDED
    assert run.state == OptimizationState.SUCCEEDED and run.error is None and run.finished_at is not None
    train = db_session.get(DatasetSnapshot, run.train_snapshot_id)
    assert late_trace.id not in train.ordered_trace_ids and len(train.ordered_trace_ids) == 6
    assert (fx.seed.instruction_text, fx.seed.manifest_hash, json.dumps(fx.seed.manifest, sort_keys=True)) == seed_before
    assert fx.seed.instruction_text == DEFAULT_SEED_INSTRUCTIONS

    candidates = run_candidates(db_session, run.id)
    assert len(candidates) == 1, "the seed is not re-created; every non-seed candidate becomes a GraderVersion"
    cand = candidates[0]
    assert cand.origin == GraderOrigin.GEPA and cand.candidate_index == 1 and cand.parent_ids == [fx.seed.id]
    assert cand.instruction_text == TRUTHFUL_INSTRUCTIONS and cand.project_id == fx.project.id
    manifest = GraderManifest.from_dict(cand.manifest)
    assert manifest.manifest_hash == cand.manifest_hash and manifest.instruction_text == TRUTHFUL_INSTRUCTIONS
    assert cand.manifest["model_config"] == fx.seed.manifest["model_config"]
    assert cand.immutable_policy_context == fx.seed.immutable_policy_context and cand.policy_epoch == 1
    assert cand.manifest_hash != fx.seed.manifest_hash

    summary = run.result_summary
    assert summary["candidate_grader_ids"] == {"0": fx.seed.id, "1": cand.id}
    assert summary["n_candidates"] == 2 and summary["member_indices"] == [0, 1] and summary["evaluated_indices"] == [0, 1]
    assert summary["seed_agreement"] == SEED_DEV_AGREEMENT and summary["best_agreement"] == 1.0
    assert summary["best_index"] == 1 and summary["recommended_grader_id"] == cand.id and summary["improved"] is True
    assert summary["partial"] is False and summary["partial_reason"] is None
    comparison = summary["comparison"]
    assert comparison["recommend"] is True and comparison["reason"] == "meets the conservative rule"
    assert "ties keep the incumbent" in comparison["rule"]
    assert comparison["incumbent"]["grader_id"] == fx.seed.id and comparison["candidate"]["grader_id"] == cand.id
    assert comparison["incumbent"]["false_passes"] == 0 and comparison["candidate"]["false_passes"] == 0
    assert "not evidence of production accuracy" in summary["note"]
    assert run.usage["calls"] > 0 and run.usage["by_role"]["grading"]["calls"] == run.usage["calls"]

    dev_ids = {t.id for t in fx.dev.values()}
    for grader, expected in ((fx.seed, SEED_DEV_AGREEMENT), (cand, 1.0)):
        ev = evaluation_for(db_session, grader.id, run.dev_snapshot_id)
        assert ev.run_id == run.id and ev.complete is True and ev.source == "GEPA"
        assert set(ev.per_case_scores) == {"gepa", "app"}
        assert set(ev.per_case_scores["app"]) == dev_ids and set(ev.per_case_scores["gepa"]) == dev_ids
        assert set(ev.per_case_scores["app"].values()) <= {0.0, 1.0}
        assert all(v is not None for v in ev.per_case_scores["gepa"].values())
        assert set(ev.verdicts) == dev_ids
        for tid, v in ev.verdicts.items():
            assert set(v) == {"verdict", "status", "grading_run_id"} and v["status"] == GradingStatus.OK
            gr = db_session.get(GradingRun, v["grading_run_id"])
            assert gr.trace_id == tid and gr.grader_id == grader.id and gr.purpose == GradingPurpose.DEV_EVALUATION
            assert gr.job_id == job.id and gr.verdict == v["verdict"]
        agg = ev.aggregate_metrics
        assert {"agreement", "false_passes", "coverage", "baselines", "insufficient_class_coverage", "complete"} <= set(agg)
        assert agg["complete"] is True and agg["insufficient_class_coverage"] is False
        assert agg["agreement"] == expected and agg["false_passes"] == 0 and agg["coverage"] == 1.0
        assert agg["human_classes"] == {"PASS": 3, "FAIL": 1} and agg["dev_size"] == 4
        assert set(agg["baselines"]) == {"always_pass", "always_fail"}
        assert agg["baselines"]["always_pass"]["agreement"]["value"] == 0.75
        assert agg["gepa_val_score"] == expected
    seed_ev = evaluation_for(db_session, fx.seed.id, run.dev_snapshot_id)
    assert {k: seed_ev.per_case_scores["app"][fx.dev[k].id] for k in DEV_IDS_BY_KEY} == DEV_IDS_BY_KEY

    # The next run seeds from the recommended candidate and freezes the label that arrived late.
    run2, _ = create_run(db_session, fx.project, RunRequest(), idempotency_key="run-2", settings=settings)
    assert run2.seed_choice == f"previous_best_supported:{run.id}" and run2.seed_grader_id == cand.id
    assert late_trace.id in db_session.get(DatasetSnapshot, run2.train_snapshot_id).ordered_trace_ids


def test_no_improvement_run_keeps_the_seed_as_incumbent(db_session, session_factory, settings, monkeypatch):
    fx = labeled_project(db_session, settings)
    run, job, _ = run_through_worker(
        db_session, session_factory, settings, monkeypatch, fx, candidates=[NON_IMPROVING_INSTRUCTIONS]
    )
    assert run.state == OptimizationState.NO_IMPROVEMENT and job.state == JobState.SUCCEEDED
    summary = run.result_summary
    assert summary["recommended_grader_id"] is None and summary["improved"] is False
    assert summary["best_index"] == 0 and summary["best_agreement"] == SEED_DEV_AGREEMENT
    assert summary["comparison"] is None
    assert fx.project.active_shadow_grader_id is None

    (cand,) = run_candidates(db_session, run.id)
    assert cand.origin == GraderOrigin.GEPA and cand.instruction_text == NON_IMPROVING_INSTRUCTIONS
    seed_ev = evaluation_for(db_session, fx.seed.id, run.dev_snapshot_id)
    cand_ev = evaluation_for(db_session, cand.id, run.dev_snapshot_id)
    assert cand_ev.complete and cand_ev.aggregate_metrics["agreement"] == SEED_DEV_AGREEMENT
    tie = compare_evaluations(seed_ev, cand_ev)
    assert tie["recommend"] is False and "ties keep the incumbent" in tie["reason"]


def test_budget_exhausted_run_is_partial_without_invented_agreement(db_session, session_factory, settings, monkeypatch):
    fx = labeled_project(db_session, settings)
    run, job, fake = run_through_worker(
        db_session, session_factory, settings, monkeypatch, fx, candidates=[TRUTHFUL_INSTRUCTIONS],
        request=RunRequest(max_provider_calls=2),
    )
    assert run.budgets["max_provider_calls"] == 2
    assert run.state == OptimizationState.BUDGET_EXHAUSTED
    assert job.state == JobState.BUDGET_EXHAUSTED and job.result == {"partial": True}
    assert "budget exhausted" in (job.error or "").lower() and "budget exhausted" in (run.error or "").lower()
    assert run.usage["exhausted"] is True and run.usage["calls"] <= 2

    summary = run.result_summary
    assert summary["partial"] is True and "budget" in summary["partial_reason"].lower()
    assert summary["best_agreement"] is None and summary["best_index"] is None
    assert summary["recommended_grader_id"] is None and summary["improved"] is False
    assert summary["seed_agreement"] in (None, NOT_ESTIMABLE)
    assert run_candidates(db_session, run.id) == [], "the fake never reached the candidate; nothing is invented"

    evaluations = list(db_session.scalars(select(CandidateEvaluation).where(CandidateEvaluation.run_id == run.id)))
    assert evaluations and all(ev.complete is False for ev in evaluations)
    seed_ev = evaluation_for(db_session, fx.seed.id, run.dev_snapshot_id)
    assert seed_ev.aggregate_metrics["complete"] is False and seed_ev.aggregate_metrics["agreement"] == NOT_ESTIMABLE
    assert all(v is None for v in seed_ev.per_case_scores["app"].values())
    dev_runs = list(db_session.scalars(select(GradingRun).where(GradingRun.job_id == job.id)))
    assert dev_runs and all(r.status == GradingStatus.BUDGET_EXHAUSTED and r.verdict == "REVIEW" for r in dev_runs)

    # The evaluation can be finished later within a fresh budget, and the project is not blocked.
    dev_snapshot = db_session.get(DatasetSnapshot, run.dev_snapshot_id)
    resumed = evaluate_on_dev(db_session, fx.project, fx.seed, dev_snapshot, budget=None, job_id=None, settings=settings)
    assert resumed.id == seed_ev.id and resumed.complete is True
    assert resumed.aggregate_metrics["agreement"] == SEED_DEV_AGREEMENT
    assert all(v["status"] == GradingStatus.OK for v in resumed.verdicts.values())
    next_run, _ = create_run(db_session, fx.project, RunRequest(), idempotency_key="run-2", settings=settings)
    assert next_run.state == OptimizationState.QUEUED and next_run.seed_grader_id == fx.seed.id


def test_cancelling_a_queued_run_prevents_execution(db_session, session_factory, settings, monkeypatch):
    fx = labeled_project(db_session, settings)
    run, job = create_run(db_session, fx.project, RunRequest(), idempotency_key="run-1", settings=settings)
    cancel_run(db_session, run)
    assert run.state == OptimizationState.CANCELLED and job.state == JobState.CANCELLED
    assert run.finished_at is not None and "cancelled" in (run.error or "")
    db_session.commit()

    fake = FakeOptimizerService([TRUTHFUL_INSTRUCTIONS])
    assert drain(worker_with_fake(settings, session_factory, monkeypatch, fake)) == 0
    assert execute_run(session_factory, run.id, optimizer=fake, settings=settings)["note"] == "already finished"
    assert fake.runs == []
    db_session.expire_all()
    assert db_session.get(OptimizationRun, run.id).state == OptimizationState.CANCELLED
    assert count(db_session, GradingRun) == 0 and run_candidates(db_session, run.id) == []
    # cancelling twice is harmless, and the project is free for a new run
    assert cancel_run(db_session, run).state == OptimizationState.CANCELLED
    run2, _ = create_run(db_session, fx.project, RunRequest(), idempotency_key="run-2", settings=settings)
    assert run2.state == OptimizationState.QUEUED and run2.id != run.id


# ----------------------------------------------------------------- DEV evaluation and comparison


def test_evaluate_on_dev_reuses_complete_evaluations_and_separates_snapshots(db_session, session_factory, settings, monkeypatch):
    fx = labeled_project(db_session, settings)
    run, _, _ = run_through_worker(db_session, session_factory, settings, monkeypatch, fx, candidates=[TRUTHFUL_INSTRUCTIONS])
    (cand,) = run_candidates(db_session, run.id)
    dev_snapshot = db_session.get(DatasetSnapshot, run.dev_snapshot_id)
    existing = evaluation_for(db_session, cand.id, dev_snapshot.id)
    before = count(db_session, GradingRun)

    again = evaluate_on_dev(db_session, fx.project, cand, dev_snapshot, budget=None, job_id=None, settings=settings)
    assert again.id == existing.id and again.complete is True
    assert count(db_session, GradingRun) == before, "a complete evaluation is returned as-is"

    new_snapshot = freeze_snapshot(db_session, fx.project, Partition.DEV)
    assert new_snapshot.id != dev_snapshot.id
    fresh = evaluate_on_dev(db_session, fx.project, fx.seed, new_snapshot, budget=None, job_id=None, settings=settings)
    old = evaluation_for(db_session, fx.seed.id, dev_snapshot.id)
    assert fresh.id != old.id and fresh.dev_snapshot_id == new_snapshot.id and fresh.run_id is None
    assert fresh.source == "APP" and fresh.complete and fresh.aggregate_metrics["agreement"] == SEED_DEV_AGREEMENT
    assert set(fresh.per_case_scores) == {"app"}, "no GEPA subscores exist for a snapshot GEPA never saw"
    assert count(db_session, CandidateEvaluation) == 3

    with pytest.raises(OptimizationError, match="different DEV snapshots"):
        compare_evaluations(old, fresh)
    with pytest.raises(OptimizationError, match="different DEV snapshots"):
        compare_evaluations(fresh, existing)


def test_comparison_refuses_a_candidate_that_adds_false_passes(db_session, settings):
    fx = labeled_project(db_session, settings)
    dev_snapshot = freeze_snapshot(db_session, fx.project, Partition.DEV)
    always_pass_grader = create_grader_version(
        db_session, fx.project, instruction_text="Always answer PASS.", origin=GraderOrigin.IMPORTED, settings=settings,
    )
    always_pass_lm = ScriptedGradingLM(lambda s, c: {"verdict": "PASS", "evidence_json": "[]", "explanation": "ok"})
    seed_ev = evaluate_on_dev(
        db_session, fx.project, fx.seed, dev_snapshot, budget=None, job_id=None,
        lm=ScriptedGradingLM(keyword_switch_policy), settings=settings,
    )
    cand_ev = evaluate_on_dev(
        db_session, fx.project, always_pass_grader, dev_snapshot, budget=None, job_id=None, lm=always_pass_lm,
        settings=settings,
    )
    assert seed_ev.aggregate_metrics["agreement"] == SEED_DEV_AGREEMENT and seed_ev.aggregate_metrics["false_passes"] == 0
    assert cand_ev.aggregate_metrics["agreement"] == 0.75 and cand_ev.aggregate_metrics["false_passes"] == 1
    assert cand_ev.aggregate_metrics["failure_recall"] == 0.0

    result = compare_evaluations(seed_ev, cand_ev)
    assert result["recommend"] is False and "adds false passes" in result["reason"]
    assert result["candidate"]["agreement"] > result["incumbent"]["agreement"]
    assert result["insufficient_class_coverage"] is False and result["dev_snapshot_id"] == dev_snapshot.id
    assert compare_evaluations(seed_ev, seed_ev)["recommend"] is False, "a tie keeps the incumbent"


def test_comparison_flags_insufficient_class_coverage_when_dev_has_one_class(db_session, settings):
    all_pass = [c for c in DEV_CASES if c.label == "PASS"]
    assert len(all_pass) == 3
    fx = labeled_project(db_session, settings, dev=all_pass)
    dev_snapshot = freeze_snapshot(db_session, fx.project, Partition.DEV)
    truthful = create_grader_version(
        db_session, fx.project, instruction_text=TRUTHFUL_INSTRUCTIONS, origin=GraderOrigin.IMPORTED, settings=settings,
    )
    seed_ev = evaluate_on_dev(db_session, fx.project, fx.seed, dev_snapshot, budget=None, job_id=None, settings=settings)
    cand_ev = evaluate_on_dev(db_session, fx.project, truthful, dev_snapshot, budget=None, job_id=None, settings=settings)
    assert seed_ev.aggregate_metrics["insufficient_class_coverage"] is True
    assert seed_ev.aggregate_metrics["human_classes"] == {"PASS": 3, "FAIL": 0}
    assert seed_ev.aggregate_metrics["failure_recall"] == NOT_ESTIMABLE
    assert cand_ev.aggregate_metrics["agreement"] == 1.0

    result = compare_evaluations(seed_ev, cand_ev)
    assert result["insufficient_class_coverage"] is True
    assert "insufficient class coverage" in result["reason"] and "failure detection cannot be inferred" in result["reason"]


# ----------------------------------------------------------------- shadow selection and readiness


def test_select_shadow_records_history_and_never_touches_automation(db_session, settings):
    fx = labeled_project(db_session, settings)
    fx.project.automation_policy_id = "policy-frozen-1"
    db_session.flush()
    first = create_grader_version(
        db_session, fx.project, instruction_text=TRUTHFUL_INSTRUCTIONS, origin=GraderOrigin.IMPORTED, settings=settings,
    )
    second = create_grader_version(
        db_session, fx.project, instruction_text=NON_IMPROVING_INSTRUCTIONS, origin=GraderOrigin.IMPORTED, settings=settings,
    )

    select_shadow(db_session, fx.project, first.id, reason="best DEV agreement in run 1", user="expert-1")
    assert fx.project.active_shadow_grader_id == first.id
    history = fx.project.configuration["shadow_history"]
    assert len(history) == 1
    assert {k: history[0][k] for k in ("grader_id", "reason", "user", "previous")} == {
        "grader_id": first.id, "reason": "best DEV agreement in run 1", "user": "expert-1", "previous": None,
    }
    assert history[0]["at"]

    select_shadow(db_session, fx.project, second.id, reason="inspecting an alternative", user="expert-2")
    assert fx.project.active_shadow_grader_id == second.id
    history = fx.project.configuration["shadow_history"]
    assert [h["grader_id"] for h in history] == [first.id, second.id] and history[1]["previous"] == first.id

    clear_shadow(db_session, fx.project, reason="back to provisional", user="expert-1")
    assert fx.project.active_shadow_grader_id is None
    assert [h["grader_id"] for h in fx.project.configuration["shadow_history"]] == [first.id, second.id, None]

    other = make_project(db_session, settings)
    foreign = create_grader_version(db_session, other, instruction_text="x", origin=GraderOrigin.IMPORTED, settings=settings)
    with pytest.raises(OptimizationError, match="another project"):
        select_shadow(db_session, fx.project, foreign.id, reason="oops", user="expert-1")

    assert fx.project.automation_policy_id == "policy-frozen-1"
    assert fx.project.configuration["automation_enabled"] is False
    assert fx.project.configuration["bootstrap_train_labels"] == settings.defaults.bootstrap_train_labels


def test_run_readiness_tracks_bootstrap_and_next_round_counts(db_session, settings):
    fx = labeled_project(
        db_session, settings, train=TRAIN_CASES[:5],
        bootstrap_train_labels=6, bootstrap_dev_labels=4, new_train_labels_per_round=2,
    )
    ready = run_readiness(db_session, fx.project)
    assert ready["bootstrap_ready"] is False and ready["resolved_train"] == 5 and ready["resolved_dev"] == 4
    assert (ready["bootstrap_train_labels"], ready["bootstrap_dev_labels"]) == (6, 4)
    assert ready["ready_to_optimize_again"] is False and ready["last_run_id"] is None and ready["active_run"] is None
    assert "not sample-size guarantees" in ready["note"]

    sixth = import_cases(db_session, fx.project, TRAIN_CASES[5:], Partition.TRAIN)
    label_cases(db_session, fx.project, sixth, TRAIN_CASES[5:], ReviewPurpose.TRAIN)
    ready = run_readiness(db_session, fx.project)
    assert ready["bootstrap_ready"] is True and ready["resolved_train"] == 6
    assert ready["dev_topup_target"] == 4

    run, _ = create_run(db_session, fx.project, RunRequest(), idempotency_key="run-1", settings=settings)
    ready = run_readiness(db_session, fx.project)
    assert ready["last_run_id"] == run.id and ready["active_run"] == run.id
    assert ready["new_train_labels_since_last_run"] == 0 and ready["ready_to_optimize_again"] is False

    extra = variants(TRAIN_CASES[:1], 1, "x")
    label_cases(db_session, fx.project, import_cases(db_session, fx.project, extra, Partition.TRAIN), extra, ReviewPurpose.TRAIN)
    ready = run_readiness(db_session, fx.project)
    assert ready["new_train_labels_since_last_run"] == 1 and ready["ready_to_optimize_again"] is False

    more = variants(TRAIN_CASES[1:2], 1, "y")
    label_cases(db_session, fx.project, import_cases(db_session, fx.project, more, Partition.TRAIN), more, ReviewPurpose.TRAIN)
    ready = run_readiness(db_session, fx.project)
    assert ready["resolved_train"] == 8 and ready["new_train_labels_since_last_run"] == 2
    assert ready["ready_to_optimize_again"] is True
    assert ready["automatic_optimization"] is False


# ----------------------------------------------------------------- the real optimizer through execute_run


def test_real_gepa_run_recommends_a_truthful_grader(db_session, session_factory, settings):
    fx = labeled_project(db_session, settings)
    run, job = create_run(
        db_session, fx.project, RunRequest(max_metric_calls=60, num_threads=2), idempotency_key="real-1", settings=settings,
    )
    db_session.commit()
    grading_lm = ScriptedGradingLM(keyword_switch_policy)
    reflection_lm = ScriptedReflectionLM(truthful_reflection_proposer)

    result = execute_run(
        session_factory, run.id, optimizer=GepaOptimizerService(), settings=settings,
        grading_lm=grading_lm, reflection_lm=reflection_lm,
    )
    db_session.expire_all()
    run = db_session.get(OptimizationRun, run.id)
    assert result["state"] == OptimizationState.SUCCEEDED and run.state == OptimizationState.SUCCEEDED
    assert run.config["max_metric_calls"] == 60 and run.result_summary["partial"] is False

    summary = run.result_summary
    assert summary["seed_agreement"] == SEED_DEV_AGREEMENT and summary["best_agreement"] == 1.0
    assert summary["gepa_best_index"] is not None and summary["improved"] is True
    recommended = db_session.get(GraderVersion, summary["recommended_grader_id"])
    assert recommended is not None and "truthful" in recommended.instruction_text.lower()
    assert recommended.origin == GraderOrigin.GEPA and recommended.optimization_run_id == run.id
    assert recommended.parent_ids, "GEPA candidates record lineage"
    for parent_id in recommended.parent_ids:
        parent = db_session.get(GraderVersion, parent_id)
        assert parent is not None and parent.project_id == fx.project.id
    assert summary["candidate_grader_ids"]["0"] == fx.seed.id
    for grader in run_candidates(db_session, run.id):
        assert grader.origin == GraderOrigin.GEPA and grader.parent_ids and grader.candidate_index >= 1
        assert GraderManifest.from_dict(grader.manifest).manifest_hash == grader.manifest_hash

    ev = evaluation_for(db_session, recommended.id, run.dev_snapshot_id)
    assert ev.complete and ev.aggregate_metrics["agreement"] == 1.0 and ev.aggregate_metrics["false_passes"] == 0
    assert set(ev.per_case_scores["app"].values()) == {1.0}
    seed_ev = evaluation_for(db_session, fx.seed.id, run.dev_snapshot_id)
    assert seed_ev.complete and seed_ev.aggregate_metrics["agreement"] == SEED_DEV_AGREEMENT
    assert summary["comparison"]["recommend"] is True

    assert run.usage["by_role"]["grading"]["calls"] > 0 and run.usage["by_role"]["reflection"]["calls"] > 0
    assert run.usage["exhausted"] is False
    reflection_text = reflection_lm.all_prompt_text()
    assert TRAIN_CANARY in reflection_text and DEV_CANARY not in reflection_text
    grading_text = grading_lm.all_prompt_text()
    assert TRAIN_CANARY not in grading_text and DEV_CANARY not in grading_text
    assert json.loads((Path(run.artifact_path) / "outcome.json").read_text())["best_index"] == summary["gepa_best_index"]
    assert db_session.get(Job, job.id).state == JobState.QUEUED, "execute_run was driven directly, not by the worker"
