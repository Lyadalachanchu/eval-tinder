"""``GepaOptimizerService`` wraps ``dspy.GEPA``; no framework types leak out.

Preflight rules
- TRAIN and DEV must be explicit, nonempty, and disjoint (by rendered-case hash).
- Only PASS/FAIL labels enter optimization.
- Exactly one budget mode: ``max_metric_calls``.
- A fresh, unique run directory (never resume a directory that holds GEPA state).
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import dspy
from dspy.clients.base_lm import BaseLM

from eval_tinder.gepa.metric import agreement_metric
from eval_tinder.gepa.results import CandidateRecord, OptimizationOutcome, candidates_from_json, extract_outcome
from eval_tinder.grader.signature import build_module
from eval_tinder.llm.budget import BudgetExhausted, BudgetGuard
from eval_tinder.llm.factory import MeteredLM


class PreflightError(ValueError):
    pass


@dataclass
class OptimizerConfig:
    max_metric_calls: int = 300
    reflection_minibatch_size: int = 3
    num_threads: int = 2
    seed: int = 0
    use_merge: bool = True
    candidate_selection_strategy: str = "pareto"
    skip_perfect_score: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def preflight(train: list[dspy.Example], dev: list[dspy.Example], config: OptimizerConfig) -> dict[str, Any]:
    if not train:
        raise PreflightError("TRAIN snapshot is empty")
    if not dev:
        raise PreflightError("DEV snapshot is empty; GEPA would silently use TRAIN as validation")
    train_hashes = {e.case_hash for e in train}
    dev_hashes = {e.case_hash for e in dev}
    overlap = train_hashes & dev_hashes
    if overlap:
        raise PreflightError(f"TRAIN and DEV overlap on {len(overlap)} rendered case(s)")
    for e in train + dev:
        if e.expert_grade not in {"PASS", "FAIL"}:
            raise PreflightError(f"label {e.expert_grade!r} cannot enter binary optimization")
    for e in train:
        if e.partition != "TRAIN":
            raise PreflightError("train example carries a non-TRAIN partition tag")
    for e in dev:
        if e.partition != "DEV":
            raise PreflightError("dev example carries a non-DEV partition tag")
    min_useful = 2 * len(dev) + 2 * config.reflection_minibatch_size
    if config.max_metric_calls < min_useful:
        raise PreflightError(
            f"max_metric_calls={config.max_metric_calls} cannot cover seed validation plus one proposal "
            f"cycle (needs at least {min_useful} for {len(dev)} DEV cases)"
        )
    return {
        "train_size": len(train),
        "dev_size": len(dev),
        "min_useful_metric_calls": min_useful,
        "estimated_full_evals": round(config.max_metric_calls / (len(train) + len(dev)), 2),
    }


class CancellationToken:
    """Cooperative cancellation. GEPA checks it between iterations (graceful stop)."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class _Stopper:
    """GEPA ``StopperProtocol``: stop when cancelled or when the provider budget is exhausted."""

    def __init__(self, token: CancellationToken | None, budget: BudgetGuard):
        self.token = token
        self.budget = budget
        self.fired: str | None = None

    def __call__(self, state: Any) -> bool:
        if self.token is not None and self.token.cancelled:
            self.fired = "cancelled"
            return True
        if self.budget.exhausted:
            self.fired = "budget_exhausted"
            return True
        return False


class OptimizerService(Protocol):
    def run(
        self,
        *,
        seed_instructions: str,
        train: list[dspy.Example],
        dev: list[dspy.Example],
        grading_lm: BaseLM,
        reflection_lm: BaseLM,
        config: OptimizerConfig,
        run_dir: Path,
        budget: BudgetGuard,
        cancellation: CancellationToken | None = None,
    ) -> OptimizationOutcome: ...


class GepaOptimizerService:
    """The real integration: one ``dspy.GEPA`` compile per run."""

    def run(
        self,
        *,
        seed_instructions: str,
        train: list[dspy.Example],
        dev: list[dspy.Example],
        grading_lm: BaseLM,
        reflection_lm: BaseLM,
        config: OptimizerConfig,
        run_dir: Path,
        budget: BudgetGuard,
        cancellation: CancellationToken | None = None,
    ) -> OptimizationOutcome:
        preflight(train, dev, config)
        run_dir = Path(run_dir)
        if (run_dir / "gepa_state.bin").exists():
            raise PreflightError(f"run directory {run_dir} already holds GEPA state; use a fresh directory")
        run_dir.mkdir(parents=True, exist_ok=True)

        metered_grader = MeteredLM(grading_lm, budget, role="grading")
        metered_reflection = MeteredLM(reflection_lm, budget, role="reflection")

        optimizer = dspy.GEPA(
            metric=agreement_metric,
            reflection_lm=metered_reflection,
            candidate_selection_strategy=config.candidate_selection_strategy,
            max_metric_calls=config.max_metric_calls,
            reflection_minibatch_size=config.reflection_minibatch_size,
            use_merge=config.use_merge,
            skip_perfect_score=config.skip_perfect_score,
            track_stats=True,
            track_best_outputs=False,
            num_threads=config.num_threads,
            seed=config.seed,
            log_dir=str(run_dir),
            warn_on_score_mismatch=False,
            gepa_kwargs={"stop_callbacks": [stopper := _Stopper(cancellation, budget)]},
        )
        seed_program = build_module(seed_instructions)
        try:
            with dspy.context(lm=metered_grader):
                optimized = optimizer.compile(student=seed_program, trainset=train, valset=dev)
        except BudgetExhausted as e:
            # Aborted mid-evaluation: only instruction texts are trustworthy; scores are unknown.
            return self._salvage(run_dir, seed_instructions, budget, reason=str(e))
        outcome = extract_outcome(optimized.detailed_results, usage=budget.snapshot())
        if stopper.fired == "cancelled":
            outcome.partial = True
            outcome.partial_reason = "cancelled before the metric-call budget was spent"
        _write_json(run_dir / "outcome.json", _outcome_json(outcome))
        return outcome

    @staticmethod
    def _salvage(run_dir: Path, seed_instructions: str, budget: BudgetGuard, *, reason: str) -> OptimizationOutcome:
        texts = [seed_instructions]
        cand_file = run_dir / "candidates.json"
        if cand_file.exists():
            try:
                texts = candidates_from_json(json.loads(cand_file.read_text())) or texts
            except (json.JSONDecodeError, OSError):
                pass
        records = [
            CandidateRecord(index=i, instruction_text=t, parent_indices=[] if i == 0 else [0],
                            val_aggregate_score=None, val_subscores={})
            for i, t in enumerate(texts)
        ]
        return OptimizationOutcome(
            candidates=records,
            best_index=None,
            instance_best_members={},
            total_metric_calls=None,
            num_full_val_evals=None,
            log_dir=str(run_dir),
            seed=None,
            usage=budget.snapshot(),
            partial=True,
            partial_reason=reason,
        )


class FakeOptimizerService:
    """Deterministic stand-in for application tests.

    It performs no reflection. It evaluates a fixed list of candidate instruction
    texts (seed first) on DEV with the real metric and grading path, so the
    application's persistence, selection, and comparison logic runs on genuine
    per-case scores. It proves nothing about learning.
    """

    def __init__(self, candidate_texts: list[str] | None = None, *, fail_with: Exception | None = None):
        self.candidate_texts = candidate_texts or []
        self.fail_with = fail_with
        self.runs: list[dict[str, Any]] = []

    def run(
        self,
        *,
        seed_instructions: str,
        train: list[dspy.Example],
        dev: list[dspy.Example],
        grading_lm: BaseLM,
        reflection_lm: BaseLM,
        config: OptimizerConfig,
        run_dir: Path,
        budget: BudgetGuard,
        cancellation: CancellationToken | None = None,
    ) -> OptimizationOutcome:
        preflight(train, dev, config)
        self.runs.append({"seed": seed_instructions, "train": len(train), "dev": len(dev), "run_dir": str(run_dir)})
        if self.fail_with is not None:
            raise self.fail_with
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        texts = [seed_instructions] + [t for t in self.candidate_texts if t != seed_instructions]
        metered = MeteredLM(grading_lm, budget, role="grading")
        records: list[CandidateRecord] = []
        try:
            for i, text in enumerate(texts):
                if cancellation is not None and cancellation.cancelled and i > 0:
                    break
                module = build_module(text)
                subs: dict[int, float] = {}
                with dspy.context(lm=metered):
                    for j, ex in enumerate(dev):
                        pred = module(**ex.inputs())
                        subs[j] = float(agreement_metric(ex, pred).score)
                records.append(
                    CandidateRecord(
                        index=i,
                        instruction_text=text,
                        parent_indices=[] if i == 0 else [0],
                        val_aggregate_score=sum(subs.values()) / len(subs),
                        val_subscores=subs,
                        discovery_eval_count=(i + 1) * len(dev),
                    )
                )
        except BudgetExhausted as e:
            return OptimizationOutcome(
                candidates=records or [CandidateRecord(0, seed_instructions, [], None, {})],
                best_index=None, instance_best_members={}, total_metric_calls=None, num_full_val_evals=None,
                log_dir=str(run_dir), seed=config.seed, usage=budget.snapshot(), partial=True, partial_reason=str(e),
            )
        best = max(range(len(records)), key=lambda i: (records[i].val_aggregate_score or 0.0, -i))
        members: dict[int, list[int]] = {}
        for j in range(len(dev)):
            top = max(r.val_subscores[j] for r in records)
            members[j] = [r.index for r in records if r.val_subscores[j] == top]
        outcome = OptimizationOutcome(
            candidates=records, best_index=best, instance_best_members=members,
            total_metric_calls=len(records) * len(dev), num_full_val_evals=len(records),
            log_dir=str(run_dir), seed=config.seed, usage=budget.snapshot(),
        )
        _write_json(run_dir / "outcome.json", _outcome_json(outcome))
        return outcome


def _outcome_json(outcome: OptimizationOutcome) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "index": c.index,
                "instruction_text": c.instruction_text,
                "parent_indices": c.parent_indices,
                "val_aggregate_score": c.val_aggregate_score,
                "val_subscores": {str(k): v for k, v in c.val_subscores.items()},
                "discovery_eval_count": c.discovery_eval_count,
            }
            for c in outcome.candidates
        ],
        "best_index": outcome.best_index,
        "instance_best_members": {str(k): v for k, v in outcome.instance_best_members.items()},
        "total_metric_calls": outcome.total_metric_calls,
        "num_full_val_evals": outcome.num_full_val_evals,
        "seed": outcome.seed,
        "usage": outcome.usage,
        "partial": outcome.partial,
        "partial_reason": outcome.partial_reason,
    }


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    os.replace(tmp, path)
