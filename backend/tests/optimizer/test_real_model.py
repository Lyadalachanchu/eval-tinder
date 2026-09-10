"""Opt-in tests against a real provider.

REAL_MODEL=1  -> one grading smoke call with the configured GRADER_MODEL.
REAL_GEPA=1   -> a small real dspy.GEPA run with real grading and reflection models.

They report what actually happened (improvement, tie, or regression) and never
require that GEPA beats the seed. Credentials come from the environment only.
"""
from __future__ import annotations

import json
import os

import pytest

from eval_tinder.config import get_settings, reset_settings_cache
from eval_tinder.domain.manifest import ModelConfig
from eval_tinder.domain.rendering import render_project_context
from eval_tinder.gepa.examples import build_examples
from eval_tinder.gepa.service import GepaOptimizerService, OptimizerConfig
from eval_tinder.grader.runtime import grade
from eval_tinder.grader.signature import DEFAULT_SEED_INSTRUCTIONS, build_module
from eval_tinder.llm.budget import BudgetGuard
from eval_tinder.llm.factory import build_grading_lm, build_reflection_lm
from eval_tinder.llm.recording import RecordingLM
from tests.cases import DEV_CANARY, DEV_CASES, TRAIN_CANARY, TRAIN_CASES, labeled

PROJECT_CONTEXT = render_project_context("A subscription support assistant that can cancel subscriptions.")


def _real_settings():
    reset_settings_cache()
    settings = get_settings()
    if settings.llm_provider != "litellm" or not settings.grader_model or not settings.reflection_model:
        pytest.skip("set LLM_PROVIDER=litellm, GRADER_MODEL and REFLECTION_MODEL (and provider credentials)")
    return settings


@pytest.mark.real_model
def test_real_grader_smoke():
    settings = _real_settings()
    lm = build_grading_lm(
        ModelConfig(provider="litellm", model=settings.grader_model, temperature=settings.grader_temperature,
                    max_tokens=settings.grader_max_tokens),
        settings=settings,
    )
    module = build_module(DEFAULT_SEED_INSTRUCTIONS)
    case = TRAIN_CASES[1].render()  # accepted status, answer claims completion
    result = grade(module, lm, case=case, project_context=PROJECT_CONTEXT, max_case_chars=settings.max_case_chars)
    print("\nREAL MODEL RESULT:", json.dumps(result.as_dict(), indent=1))
    assert result.verdict in {"PASS", "FAIL", "REVIEW"}
    if result.status != "OK":
        assert result.verdict == "REVIEW"
    assert result.usage.get("calls", 0) >= 1


@pytest.mark.real_gepa
def test_real_gepa_round_reports_actual_outcome(tmp_path):
    settings = _real_settings()
    grading = RecordingLM(
        build_grading_lm(
            ModelConfig(provider="litellm", model=settings.grader_model, temperature=settings.grader_temperature,
                        max_tokens=settings.grader_max_tokens),
            settings=settings,
        )
    )
    reflection = RecordingLM(build_reflection_lm(settings=settings))
    train = build_examples(PROJECT_CONTEXT, labeled(TRAIN_CASES, "TRAIN"))
    dev = build_examples(PROJECT_CONTEXT, labeled(DEV_CASES, "DEV"))
    budget = BudgetGuard(max_calls=int(os.environ.get("REAL_GEPA_MAX_CALLS", "120")), max_total_tokens=5_000_000,
                         max_tokens_per_call=settings.max_tokens_per_call)
    config = OptimizerConfig(max_metric_calls=int(os.environ.get("REAL_GEPA_METRIC_CALLS", "40")),
                             reflection_minibatch_size=3, num_threads=2, seed=0)
    outcome = GepaOptimizerService().run(
        seed_instructions=DEFAULT_SEED_INSTRUCTIONS, train=train, dev=dev, grading_lm=grading, reflection_lm=reflection,
        config=config, run_dir=tmp_path / "real_gepa", budget=budget,
    )
    seed = outcome.candidates[0]
    best = outcome.candidates[outcome.best_index] if outcome.best_index is not None else None
    report = {
        "partial": outcome.partial,
        "partial_reason": outcome.partial_reason,
        "n_candidates": len(outcome.candidates),
        "seed_dev_agreement": seed.val_aggregate_score,
        "best_index": outcome.best_index,
        "best_dev_agreement": best.val_aggregate_score if best else None,
        "total_metric_calls": outcome.total_metric_calls,
        "usage": outcome.usage,
        "best_instruction": best.instruction_text if best else None,
    }
    print("\nREAL GEPA OUTCOME:", json.dumps(report, indent=1))
    # Contract, not learning: structure is intact and leakage rules hold with a real provider.
    assert seed.instruction_text == DEFAULT_SEED_INSTRUCTIONS
    for c in outcome.candidates:
        assert set(c.val_subscores) <= set(range(len(dev)))
    grading_text = grading.all_prompt_text()
    for forbidden in (TRAIN_CANARY, DEV_CANARY, "expert_grade", "Expert grade"):
        assert forbidden not in grading_text
    if reflection.calls:
        reflection_text = reflection.all_prompt_text()
        assert DEV_CANARY not in reflection_text
    if best is not None and seed.val_aggregate_score is not None and best.val_aggregate_score is not None:
        delta = best.val_aggregate_score - seed.val_aggregate_score
        print(f"REAL GEPA DEV agreement delta: {delta:+.3f} (improvement, tie, or regression is reported as measured)")
