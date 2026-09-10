"""Agreement metric with the five-argument GEPA calling convention.

score = 1 if the canonical predicted grade equals the expert grade, else 0.
REVIEW on a determinate human label scores 0 (no partial credit for abstaining).
TRAIN examples receive reflective feedback; DEV examples only a score.
"""
from __future__ import annotations

from typing import Any

import dspy

from eval_tinder.domain.manifest import METRIC_VERSION
from eval_tinder.grader.runtime import canonicalize

__all__ = ["METRIC_VERSION", "agreement_metric", "score_only"]


def score_only(gold: dspy.Example, pred: Any) -> tuple[float, Any]:
    if gold.expert_grade not in {"PASS", "FAIL"}:
        raise ValueError("Unresolved human judgments cannot enter binary optimization")
    canon = canonicalize(pred, getattr(gold, "case_data", None))
    score = float(canon.status == "OK" and canon.verdict in {"PASS", "FAIL"} and canon.verdict == gold.expert_grade)
    return score, canon


def agreement_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score, canon = score_only(gold, pred)
    partition = getattr(gold, "partition", None)
    if partition == "TRAIN":
        shown = canon.verdict if canon.status == "OK" else f"{canon.verdict} (operational status {canon.status}: {canon.error})"
        feedback = (
            f"Predicted grade: {shown}. Expert grade: {gold.expert_grade}. "
            f"Expert explanation: {gold.expert_explanation or 'Not supplied'}. "
            "Infer a reusable distinction consistent with these judgments; "
            "do not memorize case identifiers or add unrelated requirements."
        )
    else:
        feedback = "Development score only; not available for reflective feedback."
    return dspy.Prediction(score=score, feedback=feedback)
