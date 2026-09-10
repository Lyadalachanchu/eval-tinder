"""Deterministic orchestration boundary around the single predictor.

``grade(...)`` -> status + effective verdict (PASS/FAIL/REVIEW) + validated evidence
+ brief explanation. Malformed output, invalid evidence, provider failures, and
budget exhaustion never become PASS: they are recorded as operational statuses
with an effective verdict of REVIEW. The same ``canonicalize`` function is used by
the GEPA metric, DEV evaluation, probes, bulk grading, and audits.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import dspy
from dspy.clients.base_lm import BaseLM

from eval_tinder.db.enums import GradingStatus, MachineVerdict
from eval_tinder.domain.rendering import CaseDocument
from eval_tinder.grader.evidence import EvidenceError, parse_evidence_json, validate_evidence
from eval_tinder.grader.signature import GraderModule, build_module
from eval_tinder.llm.budget import BudgetExhausted

VALID_VERDICTS = {v.value for v in MachineVerdict}


@dataclass
class CanonicalResult:
    status: str
    verdict: str  # effective verdict; non-OK statuses are REVIEW
    raw_verdict: str | None
    evidence: list[dict[str, Any]]
    explanation: str
    error: str | None = None

    @property
    def is_vote(self) -> bool:
        return self.status == GradingStatus.OK


def canonicalize(pred: Any, case_data: Any) -> CanonicalResult:
    """Validate a raw prediction (dspy.Prediction or dict) against the case document."""
    if pred is None:
        return CanonicalResult(GradingStatus.MALFORMED_OUTPUT, "REVIEW", None, [], "", "no prediction")
    get = (lambda k: pred.get(k)) if isinstance(pred, dict) else (lambda k: getattr(pred, k, None))
    raw_verdict = get("verdict")
    verdict = str(raw_verdict).strip().upper() if raw_verdict is not None else None
    explanation = str(get("explanation") or "").strip()[:2000]
    if verdict not in VALID_VERDICTS:
        return CanonicalResult(
            GradingStatus.MALFORMED_OUTPUT, "REVIEW", verdict, [], explanation, f"invalid verdict {raw_verdict!r}"
        )
    try:
        items = parse_evidence_json(get("evidence_json") if get("evidence_json") is not None else "[]")
        evidence = validate_evidence(items, case_data)
    except EvidenceError as e:
        return CanonicalResult(GradingStatus.INVALID_EVIDENCE, "REVIEW", verdict, [], explanation, str(e))
    return CanonicalResult(GradingStatus.OK, verdict, verdict, evidence, explanation, None)


@dataclass
class GradeResult:
    status: str
    verdict: str
    raw_verdict: str | None
    evidence: list[dict[str, Any]]
    explanation: str
    error: str | None
    attempts: int
    latency_ms: int
    usage: dict[str, Any] = field(default_factory=dict)
    prompt_hash: str = ""

    @property
    def is_vote(self) -> bool:
        return self.status == GradingStatus.OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "verdict": self.verdict,
            "raw_verdict": self.raw_verdict,
            "evidence": self.evidence,
            "explanation": self.explanation,
            "error": self.error,
            "attempts": self.attempts,
            "latency_ms": self.latency_ms,
            "usage": self.usage,
        }


RETRYABLE_STATUSES = {GradingStatus.MALFORMED_OUTPUT, GradingStatus.INVALID_EVIDENCE, GradingStatus.PROVIDER_ERROR}


def _usage_from_lm(lm: BaseLM, before: int) -> dict[str, Any]:
    calls = lm.history[before:] if hasattr(lm, "history") else []
    pt = ct = 0
    for h in calls:
        u = h.get("usage") or {}
        pt += int(u.get("prompt_tokens") or 0)
        ct += int(u.get("completion_tokens") or 0)
    return {"calls": len(calls), "prompt_tokens": pt, "completion_tokens": ct}


def grade(
    module: GraderModule,
    lm: BaseLM,
    *,
    case: CaseDocument,
    project_context: str,
    max_case_chars: int,
    max_attempts: int = 2,
) -> GradeResult:
    started = time.perf_counter()
    if case.char_count > max_case_chars:
        return GradeResult(
            GradingStatus.CONTEXT_TOO_LARGE, "REVIEW", None, [], "", f"rendered case has {case.char_count} chars "
            f"(> {max_case_chars})", 0, 0
        )
    history_before = len(getattr(lm, "history", []))
    last: CanonicalResult | None = None
    attempts = 0
    for attempts in range(1, max_attempts + 1):
        try:
            with dspy.context(lm=lm):
                pred = module(project_context=project_context, case=case.text)
        except BudgetExhausted as e:
            last = CanonicalResult(GradingStatus.BUDGET_EXHAUSTED, "REVIEW", None, [], "", str(e))
            break
        except Exception as e:  # provider error or adapter parse failure after dspy's own retries
            name = type(e).__name__
            status = GradingStatus.MALFORMED_OUTPUT if "Parse" in name or "Adapter" in name else GradingStatus.PROVIDER_ERROR
            last = CanonicalResult(status, "REVIEW", None, [], "", f"{name}: {str(e)[:500]}")
            continue
        last = canonicalize(pred, case.data)
        if last.status == GradingStatus.OK:
            break
    assert last is not None
    return GradeResult(
        status=last.status,
        verdict=last.verdict,
        raw_verdict=last.raw_verdict,
        evidence=last.evidence,
        explanation=last.explanation,
        error=last.error,
        attempts=attempts,
        latency_ms=int((time.perf_counter() - started) * 1000),
        usage=_usage_from_lm(lm, history_before),
    )


def module_for_instructions(instruction_text: str) -> GraderModule:
    return build_module(instruction_text)
