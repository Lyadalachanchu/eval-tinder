"""Framework-free representation of a GEPA run's outcome and candidate extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eval_tinder.grader.signature import PREDICTOR_NAME, get_instructions

STOP_CANCELLED = "cancelled"
STOP_BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class CandidateRecord:
    index: int
    instruction_text: str
    parent_indices: list[int]
    val_aggregate_score: float | None
    val_subscores: dict[int, float]  # keyed by validation instance id (index into frozen DEV list)
    discovery_eval_count: int | None = None


@dataclass
class OptimizationOutcome:
    candidates: list[CandidateRecord]
    best_index: int | None
    instance_best_members: dict[int, list[int]]  # val instance id -> candidate indices
    total_metric_calls: int | None
    num_full_val_evals: int | None
    log_dir: str | None
    seed: int | None
    usage: dict[str, Any] = field(default_factory=dict)
    partial: bool = False
    partial_reason: str | None = None
    # Structured stop reason for a partial outcome: STOP_CANCELLED or STOP_BUDGET_EXHAUSTED (None when the
    # optimizer ran to completion). Callers must branch on this, never on the free-text ``partial_reason``:
    # a cancellation message may legitimately mention the budget ("cancelled before the budget was spent").
    stop_reason: str | None = None

    def member_indices(self) -> list[int]:
        members: set[int] = set()
        if self.best_index is not None:
            members.add(self.best_index)
        for indices in self.instance_best_members.values():
            members.update(indices)
        return sorted(members)


def extract_outcome(detailed_results: Any, *, usage: dict[str, Any] | None = None) -> OptimizationOutcome:
    """Convert ``DspyGEPAResult`` into plain records. Missing subscores stay missing (never zero)."""
    candidates: list[CandidateRecord] = []
    n = len(detailed_results.candidates)
    parents = list(getattr(detailed_results, "parents", []) or [])
    aggregates = list(getattr(detailed_results, "val_aggregate_scores", []) or [])
    subscores = list(getattr(detailed_results, "val_subscores", []) or [])
    discovery = list(getattr(detailed_results, "discovery_eval_counts", []) or [])
    for i, program in enumerate(detailed_results.candidates):
        text = get_instructions(program)
        par = parents[i] if i < len(parents) else []
        par = [int(p) for p in (par or []) if p is not None]
        agg = float(aggregates[i]) if i < len(aggregates) and aggregates[i] is not None else None
        subs = {int(k): float(v) for k, v in (subscores[i] if i < len(subscores) else {}).items()}
        candidates.append(
            CandidateRecord(
                index=i,
                instruction_text=text,
                parent_indices=par,
                val_aggregate_score=agg,
                val_subscores=subs,
                discovery_eval_count=int(discovery[i]) if i < len(discovery) else None,
            )
        )
    best = int(detailed_results.best_idx) if n else None
    members = {
        int(k): sorted(int(i) for i in v)
        for k, v in (getattr(detailed_results, "per_val_instance_best_candidates", {}) or {}).items()
    }
    return OptimizationOutcome(
        candidates=candidates,
        best_index=best,
        instance_best_members=members,
        total_metric_calls=getattr(detailed_results, "total_metric_calls", None),
        num_full_val_evals=getattr(detailed_results, "num_full_val_evals", None),
        log_dir=getattr(detailed_results, "log_dir", None),
        seed=getattr(detailed_results, "seed", None),
        usage=usage or {},
    )


def map_subscores_to_traces(subscores: dict[int, float], dev_trace_ids: list[str]) -> dict[str, float | None]:
    """Instance id -> trace id mapping against the frozen DEV snapshot. Missing = None."""
    out: dict[str, float | None] = {tid: None for tid in dev_trace_ids}
    for idx, score in subscores.items():
        if 0 <= idx < len(dev_trace_ids):
            out[dev_trace_ids[idx]] = score
        else:
            raise ValueError(f"validation instance id {idx} is outside the frozen DEV snapshot of size {len(dev_trace_ids)}")
    return out


def candidates_from_json(candidates_json: list[dict[str, str]]) -> list[str]:
    """Instruction texts from GEPA's ``candidates.json`` (used to salvage a budget-exhausted run)."""
    texts = []
    for cand in candidates_json:
        if PREDICTOR_NAME in cand:
            texts.append(cand[PREDICTOR_NAME])
    return texts
