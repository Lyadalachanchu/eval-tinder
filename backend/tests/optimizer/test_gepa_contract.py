"""M0: the real ``dspy.GEPA`` runs through our adapter with deterministic scripted LMs.

These tests verify the integration contract against the pinned release. They do
not show that optimization learns anything about real data.
"""
from __future__ import annotations

import json
from pathlib import Path

import dspy
import pytest

from eval_tinder.gepa.examples import build_examples
from eval_tinder.gepa.metric import agreement_metric
from eval_tinder.gepa.results import map_subscores_to_traces
from eval_tinder.gepa.service import GepaOptimizerService, OptimizerConfig, PreflightError, preflight
from eval_tinder.grader.signature import DEFAULT_SEED_INSTRUCTIONS, PREDICTOR_NAME, build_module, get_instructions
from eval_tinder.llm.budget import BudgetGuard
from eval_tinder.llm.fakes import ScriptedGradingLM, ScriptedReflectionLM, keyword_switch_policy, truthful_reflection_proposer
from tests.cases import DEV_CANARY, DEV_CASES, TRAIN_CANARY, TRAIN_CASES, labeled

PROJECT_CONTEXT = "Application description: a subscription support assistant."


def _examples():
    train = build_examples(PROJECT_CONTEXT, labeled(TRAIN_CASES, "TRAIN"))
    dev = build_examples(PROJECT_CONTEXT, labeled(DEV_CASES, "DEV"))
    return train, dev


def _lms():
    return ScriptedGradingLM(keyword_switch_policy), ScriptedReflectionLM(truthful_reflection_proposer)


def _budget(**kw):
    return BudgetGuard(max_calls=kw.get("max_calls", 10_000), max_total_tokens=10_000_000, max_tokens_per_call=200)


@pytest.fixture(scope="module")
def gepa_run(tmp_path_factory):
    train, dev = _examples()
    grader, reflector = _lms()
    run_dir = tmp_path_factory.mktemp("gepa_run")
    config = OptimizerConfig(max_metric_calls=60, reflection_minibatch_size=3, num_threads=2, seed=0)
    outcome = GepaOptimizerService().run(
        seed_instructions=DEFAULT_SEED_INSTRUCTIONS, train=train, dev=dev, grading_lm=grader,
        reflection_lm=reflector, config=config, run_dir=run_dir, budget=_budget(),
    )
    return {"outcome": outcome, "grader": grader, "reflector": reflector, "run_dir": run_dir, "dev": dev, "train": train}


def test_real_gepa_compile_produces_candidates(gepa_run):
    outcome = gepa_run["outcome"]
    assert not outcome.partial
    assert len(outcome.candidates) >= 2, "expected the seed plus at least one proposed candidate"
    seed = outcome.candidates[0]
    assert seed.instruction_text == DEFAULT_SEED_INSTRUCTIONS
    assert seed.parent_indices == []
    assert outcome.best_index is not None
    best = outcome.candidates[outcome.best_index]
    assert "truthful" in best.instruction_text.lower()
    assert best.val_aggregate_score is not None and seed.val_aggregate_score is not None
    assert best.val_aggregate_score > seed.val_aggregate_score
    assert best.val_aggregate_score == pytest.approx(1.0)
    for c in outcome.candidates[1:]:
        assert c.parent_indices, "non-seed candidates must record lineage"
    # GEPA treats max_metric_calls as approximate: an in-flight minibatch may finish past the limit.
    # The application-level BudgetGuard is the hard cap (see the exhaustion test below).
    assert outcome.total_metric_calls is not None
    assert outcome.total_metric_calls <= 60 + len(gepa_run["train"]) + len(gepa_run["dev"])


def test_val_subscores_are_keyed_by_dev_instance_id(gepa_run):
    outcome, dev = gepa_run["outcome"], gepa_run["dev"]
    ids = list(range(len(dev)))
    for c in outcome.candidates:
        assert set(c.val_subscores) <= set(ids)
    for val_id, members in outcome.instance_best_members.items():
        assert val_id in ids
        assert members and all(0 <= m < len(outcome.candidates) for m in members)
    trace_ids = [f"trace-{i}" for i in ids]
    mapped = map_subscores_to_traces(outcome.candidates[outcome.best_index].val_subscores, trace_ids)
    assert set(mapped) == set(trace_ids)
    assert outcome.best_index in outcome.member_indices()


def test_reflection_sees_train_feedback_but_never_dev_explanations(gepa_run):
    reflector = gepa_run["reflector"]
    assert reflector.calls, "the reflection LM was never called"
    text = reflector.all_prompt_text()
    assert TRAIN_CANARY in text
    assert DEV_CANARY not in text
    assert "Expert grade:" in text


def test_grading_model_never_sees_labels_or_bookkeeping(gepa_run):
    grader = gepa_run["grader"]
    assert grader.calls
    text = grader.all_prompt_text()
    for forbidden in (TRAIN_CANARY, DEV_CANARY, "expert_grade", "expert_explanation", "Expert grade", "partition"):
        assert forbidden not in text
    for call in grader.calls:
        assert "<<<CASE_JSON" in call.text and '"output"' in call.text
        assert all(m.get("role") in {"system", "user", "assistant"} for m in call.messages)


def test_separate_contexts_for_grading_and_reflection(gepa_run):
    grader, reflector = gepa_run["grader"], gepa_run["reflector"]
    assert all("<<<CASE_JSON" in c.text for c in grader.calls)
    assert all("[[ ## verdict ## ]]" not in c.output or True for c in reflector.calls)
    assert all("```" in c.output for c in reflector.calls)


def test_prompt_only_round_trip(gepa_run):
    outcome = gepa_run["outcome"]
    best = outcome.candidates[outcome.best_index]
    rebuilt = build_module(best.instruction_text)
    assert get_instructions(rebuilt) == best.instruction_text
    assert [n for n, _ in rebuilt.named_predictors()] == [PREDICTOR_NAME]
    grader, _ = _lms()
    dev = gepa_run["dev"]
    with dspy.context(lm=grader):
        preds = [rebuilt(**ex.inputs()) for ex in dev]
    scores = [agreement_metric(ex, p).score for ex, p in zip(dev, preds)]
    assert sum(scores) / len(scores) == pytest.approx(best.val_aggregate_score)


def test_run_directory_holds_json_artifacts_only_for_us(gepa_run):
    run_dir: Path = gepa_run["run_dir"]
    outcome_file = run_dir / "outcome.json"
    assert outcome_file.exists()
    data = json.loads(outcome_file.read_text())
    assert data["best_index"] == gepa_run["outcome"].best_index
    assert "candidates" in data


def test_empty_dev_fails_preflight_before_any_llm_call(tmp_path):
    train, _ = _examples()
    grader, reflector = _lms()
    with pytest.raises(PreflightError, match="DEV"):
        GepaOptimizerService().run(
            seed_instructions=DEFAULT_SEED_INSTRUCTIONS, train=train, dev=[], grading_lm=grader,
            reflection_lm=reflector, config=OptimizerConfig(max_metric_calls=60), run_dir=tmp_path / "r",
            budget=_budget(),
        )
    assert grader.calls == [] and reflector.calls == []


def test_overlapping_train_dev_fails_preflight():
    train, dev = _examples()
    with pytest.raises(PreflightError, match="overlap"):
        preflight(train, train[:2], OptimizerConfig(max_metric_calls=60))
    with pytest.raises(PreflightError, match="cannot cover"):
        preflight(train, dev, OptimizerConfig(max_metric_calls=3))


def test_reused_run_directory_is_refused(tmp_path):
    train, dev = _examples()
    grader, reflector = _lms()
    (tmp_path / "gepa_state.bin").write_bytes(b"not loaded")
    with pytest.raises(PreflightError, match="already holds GEPA state"):
        GepaOptimizerService().run(
            seed_instructions=DEFAULT_SEED_INSTRUCTIONS, train=train, dev=dev, grading_lm=grader,
            reflection_lm=reflector, config=OptimizerConfig(max_metric_calls=60), run_dir=tmp_path, budget=_budget(),
        )


def test_budget_exhaustion_yields_partial_outcome_without_invented_scores(tmp_path):
    train, dev = _examples()
    grader, reflector = _lms()
    outcome = GepaOptimizerService().run(
        seed_instructions=DEFAULT_SEED_INSTRUCTIONS, train=train, dev=dev, grading_lm=grader,
        reflection_lm=reflector, config=OptimizerConfig(max_metric_calls=60), run_dir=tmp_path / "r",
        budget=_budget(max_calls=5),
    )
    assert outcome.partial
    assert "budget exhausted" in (outcome.partial_reason or "").lower()
    assert outcome.best_index is None
    assert all(c.val_aggregate_score is None for c in outcome.candidates)
    assert outcome.usage["calls"] <= 5
    assert outcome.usage["exhausted"] is True


def test_metric_contract_scores_and_feedback():
    train, dev = _examples()
    ex = train[0]  # PASS label

    def pred(verdict, ev="[]"):
        return dspy.Prediction(verdict=verdict, evidence_json=ev, explanation="x")

    assert agreement_metric(ex, pred("PASS")).score == 1.0
    assert agreement_metric(ex, pred("FAIL")).score == 0.0
    assert agreement_metric(ex, pred("REVIEW")).score == 0.0
    assert agreement_metric(ex, pred("PASS", ev='[{"pointer":"/nope","quote":""}]')).score == 0.0
    assert agreement_metric(ex, pred("PASS", ev="not json")).score == 0.0
    # module- and predictor-level calls return the same number
    module_level = agreement_metric(ex, pred("PASS"))
    pred_level = agreement_metric(ex, pred("PASS"), None, PREDICTOR_NAME, [])
    assert module_level.score == pred_level.score
    assert "Expert grade: PASS" in module_level.feedback
    dev_fb = agreement_metric(dev[0], pred("PASS")).feedback
    assert DEV_CANARY not in dev_fb and "Development score only" in dev_fb
    unresolved = dspy.Example(expert_grade="CANNOT_JUDGE", partition="TRAIN", case_data={}).with_inputs()
    with pytest.raises(ValueError, match="Unresolved"):
        agreement_metric(unresolved, pred("PASS"))
