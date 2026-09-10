"""Application-level provider budgets.

A GEPA metric-call budget is not a token or dollar budget. ``BudgetGuard`` caps
provider calls and tokens per job; on exhaustion no new calls are submitted and
the job finishes in an explicit BUDGET_EXHAUSTED state (in-flight calls are
accounted for, results are never invented).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


class BudgetExhausted(BaseException):
    """Raised when the application budget is spent.

    Deliberately a ``BaseException``: DSPy's evaluator and parallelizer catch
    ``Exception`` per example and would otherwise convert exhaustion into silent
    zero scores. Exhaustion must abort the optimization run instead.
    """


@dataclass
class UsageTotals:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_role: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "by_role": {k: dict(v) for k, v in self.by_role.items()},
        }


class BudgetGuard:
    def __init__(self, *, max_calls: int, max_total_tokens: int, max_tokens_per_call: int):
        self.max_calls = max_calls
        self.max_total_tokens = max_total_tokens
        self.max_tokens_per_call = max_tokens_per_call
        self._lock = threading.Lock()
        self._in_flight = 0
        self.totals = UsageTotals()
        self.exhausted = False

    def reserve(self, role: str) -> None:
        with self._lock:
            projected_calls = self.totals.calls + self._in_flight + 1
            projected_tokens = (
                self.totals.prompt_tokens
                + self.totals.completion_tokens
                + (self._in_flight + 1) * self.max_tokens_per_call
            )
            if self.exhausted or projected_calls > self.max_calls or projected_tokens > self.max_total_tokens:
                self.exhausted = True
                raise BudgetExhausted(
                    f"Provider budget exhausted before {role} call: "
                    f"calls={self.totals.calls}+{self._in_flight} in flight (max {self.max_calls}), "
                    f"tokens={self.totals.prompt_tokens + self.totals.completion_tokens} (max {self.max_total_tokens})"
                )
            self._in_flight += 1

    def release(self, role: str, usage: dict[str, Any] | None, *, failed: bool = False) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self.totals.calls += 1
            pt = int((usage or {}).get("prompt_tokens") or 0)
            ct = int((usage or {}).get("completion_tokens") or 0)
            self.totals.prompt_tokens += pt
            self.totals.completion_tokens += ct
            bucket = self.totals.by_role.setdefault(role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "failed": 0})
            bucket["calls"] += 1
            bucket["prompt_tokens"] += pt
            bucket["completion_tokens"] += ct
            if failed:
                bucket["failed"] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            d = self.totals.as_dict()
            d["exhausted"] = self.exhausted
            d["limits"] = {
                "max_calls": self.max_calls,
                "max_total_tokens": self.max_total_tokens,
                "max_tokens_per_call": self.max_tokens_per_call,
            }
            return d


def estimate_cost_usd(model: str, usage: dict[str, Any], pricing_table: dict[str, dict[str, float]]) -> float | None:
    """Return a cost only when the configured pricing table covers the model."""
    price = pricing_table.get(model)
    if not price:
        return None
    pt = float(usage.get("prompt_tokens") or 0)
    ct = float(usage.get("completion_tokens") or 0)
    return pt / 1000.0 * price.get("input_per_1k", 0.0) + ct / 1000.0 * price.get("output_per_1k", 0.0)
