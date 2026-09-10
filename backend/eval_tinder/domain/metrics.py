"""Agreement metrics between human verdicts and machine verdicts.

Every metric here is computed against *human* labels. Machine predictions are
never treated as labels: a human ``CANNOT_JUDGE`` case stays unresolved (it is
counted, but never scored as agreement or disagreement), and a machine
``REVIEW`` or operational failure is never an "agreement".

Conventions
-----------
* Confusion cells are named ``<human>_<machine>`` (human first).
* A machine call whose status is not ``"OK"`` is an *operational failure*. It
  casts no vote, so it is treated as ``REVIEW`` for the confusion cells and
  coverage denominators, and additionally reported on its own.
* Any ratio with a zero denominator is ``NOT_ESTIMABLE`` (a string sentinel),
  never ``0`` and never ``None``, so the UI can distinguish "no failures were
  observed" from "there was nothing to observe".

This module has no database, model, or network dependencies. The verdict
strings mirror ``eval_tinder.db.enums`` (``HumanVerdict``, ``MachineVerdict``,
``GradingStatus.OK``).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from typing import Any

from scipy.stats import binomtest

NOT_ESTIMABLE = "NOT_ESTIMABLE"

HUMAN_VERDICTS: frozenset[str] = frozenset({"PASS", "FAIL", "CANNOT_JUDGE"})
MACHINE_VERDICTS: frozenset[str] = frozenset({"PASS", "FAIL", "REVIEW"})
OK_STATUS = "OK"

MetricEntry = dict[str, Any]


def ratio(numerator: int, denominator: int) -> float | str:
    """``numerator / denominator``, or ``NOT_ESTIMABLE`` when ``denominator == 0``.

    Guarantees: never returns ``0.0`` for an empty denominator, never divides by
    zero, and raises ``ValueError`` for negative counts (which indicate a bug in
    the caller rather than an empty sample).
    """
    if numerator < 0 or denominator < 0:
        raise ValueError(f"counts must be non-negative, got numerator={numerator} denominator={denominator}")
    if denominator == 0:
        return NOT_ESTIMABLE
    return numerator / denominator


@dataclass(frozen=True)
class ConfusionTable:
    """Counts of (human verdict, machine verdict) pairs. Human first, machine second.

    ``pass_review`` and ``fail_review`` include both genuine machine ``REVIEW``
    votes and operational failures (non-OK status); ``operational_failures``
    reports the latter separately across *all* cases, including human
    ``CANNOT_JUDGE`` cases. ``unresolved_machine_binary`` counts OK machine
    PASS/FAIL votes on human ``CANNOT_JUDGE`` cases; it only feeds coverage over
    all cases and is never scored against a label.

    Construction validates that every count is a non-negative ``int`` and that
    the derived invariants hold (see ``__post_init__``).
    """

    pass_pass: int = 0
    pass_fail: int = 0
    pass_review: int = 0
    fail_pass: int = 0
    fail_fail: int = 0
    fail_review: int = 0
    operational_failures: int = 0
    human_unresolved: int = 0
    unresolved_machine_binary: int = 0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{f.name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{f.name} must be >= 0, got {value}")
        if self.unresolved_machine_binary > self.human_unresolved:
            raise ValueError(
                "unresolved_machine_binary cannot exceed human_unresolved "
                f"({self.unresolved_machine_binary} > {self.human_unresolved})"
            )
        non_votes = self.pass_review + self.fail_review + (self.human_unresolved - self.unresolved_machine_binary)
        if self.operational_failures > non_votes:
            raise ValueError(
                f"operational_failures ({self.operational_failures}) cannot exceed the number of non-vote "
                f"cases ({non_votes}); operational failures are counted as REVIEW"
            )

    @property
    def human_pass(self) -> int:
        return self.pass_pass + self.pass_fail + self.pass_review

    @property
    def human_fail(self) -> int:
        return self.fail_pass + self.fail_fail + self.fail_review

    @property
    def human_determinate(self) -> int:
        """Cases with a human PASS or FAIL label."""
        return self.human_pass + self.human_fail

    @property
    def total_cases(self) -> int:
        """All cases: human-determinate plus human CANNOT_JUDGE."""
        return self.human_determinate + self.human_unresolved

    @property
    def machine_binary_on_determinate(self) -> int:
        """OK machine PASS/FAIL votes on human-determinate cases."""
        return self.pass_pass + self.pass_fail + self.fail_pass + self.fail_fail

    @property
    def machine_binary_all(self) -> int:
        """OK machine PASS/FAIL votes on all cases."""
        return self.machine_binary_on_determinate + self.unresolved_machine_binary

    def as_table(self) -> dict[str, dict[str, int]]:
        """The 2x3 table as ``{human: {machine: count}}``."""
        return {
            "PASS": {"PASS": self.pass_pass, "FAIL": self.pass_fail, "REVIEW": self.pass_review},
            "FAIL": {"PASS": self.fail_pass, "FAIL": self.fail_fail, "REVIEW": self.fail_review},
        }


def build_confusion(rows: Iterable[tuple[str, str, str]]) -> ConfusionTable:
    """Tally ``(human_verdict, machine_verdict, machine_status)`` rows into a ``ConfusionTable``.

    ``human_verdict`` must be PASS, FAIL, or CANNOT_JUDGE; ``machine_verdict``
    must be PASS, FAIL, or REVIEW (``StrEnum`` members are accepted). A
    ``machine_status`` of ``"OK"`` is a valid vote; any other status is an
    operational failure, which is counted as REVIEW *and* in
    ``operational_failures``. Unknown verdicts or malformed rows raise
    ``ValueError``; nothing is skipped silently. Row order is irrelevant.
    """
    counts = dict.fromkeys(
        (
            "pass_pass", "pass_fail", "pass_review", "fail_pass", "fail_fail", "fail_review",
            "operational_failures", "human_unresolved", "unresolved_machine_binary",
        ),
        0,
    )
    for index, row in enumerate(rows):
        try:
            human_raw, machine_raw, status_raw = row
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {index} must be a (human_verdict, machine_verdict, machine_status) triple") from exc
        human = _as_str(human_raw, f"row {index} human_verdict")
        machine = _as_str(machine_raw, f"row {index} machine_verdict")
        status = _as_str(status_raw, f"row {index} machine_status")
        if human not in HUMAN_VERDICTS:
            raise ValueError(f"row {index}: unknown human verdict {human!r}; expected one of {sorted(HUMAN_VERDICTS)}")
        if machine not in MACHINE_VERDICTS:
            raise ValueError(
                f"row {index}: unknown machine verdict {machine!r}; expected one of {sorted(MACHINE_VERDICTS)}"
            )
        operational_failure = status != OK_STATUS
        if operational_failure:
            counts["operational_failures"] += 1
            machine = "REVIEW"
        if human == "CANNOT_JUDGE":
            counts["human_unresolved"] += 1
            if machine != "REVIEW":
                counts["unresolved_machine_binary"] += 1
            continue
        counts[f"{human.lower()}_{machine.lower()}"] += 1
    return ConfusionTable(**counts)


def _as_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string, got {type(value).__name__}")
    return str(value)  # normalizes StrEnum members to their plain value


def _entry(numerator: int, denominator: int, definition: str) -> MetricEntry:
    return {
        "value": ratio(numerator, denominator),
        "numerator": numerator,
        "denominator": denominator,
        "definition": definition,
    }


def compute_metrics(table: ConfusionTable) -> dict[str, dict]:
    """All agreement/coverage metrics for ``table``, plus the raw counts.

    Every metric entry has ``value`` (``float`` or ``NOT_ESTIMABLE``),
    ``numerator``, ``denominator`` and ``definition``. Values are ratios in
    ``[0, 1]``; a zero denominator always yields ``NOT_ESTIMABLE``. Human
    ``CANNOT_JUDGE`` cases appear only in the ``total_cases`` denominators and
    in ``counts``; they are never scored as agreement or error. The
    ``"counts"`` entry carries ``human_pass``, ``human_fail``,
    ``human_unresolved``, ``operational_failures``, ``total_cases`` and the 2x3
    ``table``.
    """
    if not isinstance(table, ConfusionTable):
        raise ValueError(f"expected a ConfusionTable, got {type(table).__name__}")
    metrics: dict[str, dict] = {
        "agreement": _entry(
            table.pass_pass + table.fail_fail,
            table.human_determinate,
            "Cases where the machine verdict equals the human PASS/FAIL label, over human-determinate cases. "
            "Machine REVIEW and operational failures count as non-agreement.",
        ),
        "automatic_coverage_all": _entry(
            table.machine_binary_all,
            table.total_cases,
            "Cases the machine decided automatically (OK status with PASS or FAIL), over all cases "
            "including human CANNOT_JUDGE.",
        ),
        "automatic_coverage_determinate": _entry(
            table.machine_binary_on_determinate,
            table.human_determinate,
            "Cases the machine decided automatically (OK status with PASS or FAIL), over human-determinate cases.",
        ),
        "automatic_error_rate": _entry(
            table.pass_fail + table.fail_pass,
            table.machine_binary_on_determinate,
            "Automatic decisions that contradict the human label, over automatic decisions on "
            "human-determinate cases. REVIEW and operational failures are neither errors nor decisions.",
        ),
        "false_pass_rate_among_accepted": _entry(
            table.fail_pass,
            table.pass_pass + table.fail_pass,
            "Human FAIL cases the machine accepted (PASS), over all human-determinate cases the machine "
            "accepted. Measures how often an automatic PASS hides a real failure.",
        ),
        "failure_recall": _entry(
            table.fail_fail,
            table.human_fail,
            "Human FAIL cases the machine also marked FAIL, over all human FAIL cases.",
        ),
        "human_unresolved_rate": _entry(
            table.human_unresolved,
            table.total_cases,
            "Cases the human marked CANNOT_JUDGE, over all cases.",
        ),
        "operational_failure_rate": _entry(
            table.operational_failures,
            table.total_cases,
            "Machine calls that did not produce a valid vote (non-OK status), over all cases.",
        ),
        "counts": {
            "human_pass": table.human_pass,
            "human_fail": table.human_fail,
            "human_unresolved": table.human_unresolved,
            "operational_failures": table.operational_failures,
            "total_cases": table.total_cases,
            "table": table.as_table(),
        },
    }
    return metrics


BASELINE_METRIC_KEYS: tuple[str, ...] = ("agreement", "failure_recall", "false_pass_rate_among_accepted")


def baselines(table: ConfusionTable) -> dict:
    """Metrics a trivial grader would score on the same human labels.

    ``always_pass`` answers PASS on every human-determinate case and
    ``always_fail`` answers FAIL on every one; both are computed from
    ``table``'s human counts only (the real machine verdicts are ignored).
    Each carries ``agreement``, ``failure_recall`` and
    ``false_pass_rate_among_accepted`` in the same entry shape as
    ``compute_metrics`` so the UI can show, e.g., that 95% agreement on a
    95%-PASS dataset comes with zero failure recall. Zero denominators yield
    ``NOT_ESTIMABLE`` exactly as for a real grader.
    """
    if not isinstance(table, ConfusionTable):
        raise ValueError(f"expected a ConfusionTable, got {type(table).__name__}")
    always_pass = ConfusionTable(
        pass_pass=table.human_pass, fail_pass=table.human_fail, human_unresolved=table.human_unresolved
    )
    always_fail = ConfusionTable(
        pass_fail=table.human_pass, fail_fail=table.human_fail, human_unresolved=table.human_unresolved
    )
    return {
        "always_pass": _baseline_entry(
            always_pass, "A grader that answers PASS on every case, scored against the same human labels."
        ),
        "always_fail": _baseline_entry(
            always_fail, "A grader that answers FAIL on every case, scored against the same human labels."
        ),
    }


def _baseline_entry(table: ConfusionTable, definition: str) -> dict:
    metrics = compute_metrics(table)
    entry: dict[str, Any] = {key: metrics[key] for key in BASELINE_METRIC_KEYS}
    entry["definition"] = definition
    return entry


def error_upper_bound(k: int, n: int, confidence: float = 0.95) -> float | None:
    """One-sided exact (Clopper-Pearson) upper confidence bound on a binomial rate.

    Given ``k`` observed errors in ``n`` independent uniformly sampled cases,
    returns the largest true error rate consistent with the data at
    ``confidence``, i.e. ``binomtest(k, n, alternative="less")
    .proportion_ci(confidence_level=confidence, method="exact").high``. For
    ``k == 0`` this equals ``1 - (1 - confidence) ** (1 / n)``; for ``k == n``
    it is ``1.0``. Returns ``None`` when ``n == 0`` (nothing observed). Raises
    ``ValueError`` when ``n < 0``, ``k`` is outside ``[0, n]``, or
    ``confidence`` is outside the open interval ``(0, 1)``.

    The bound is only meaningful under the sampling design checked by
    ``sampling_design_supported``; callers must not compute it otherwise.
    """
    if isinstance(k, bool) or isinstance(n, bool) or not isinstance(k, int) or not isinstance(n, int):
        raise ValueError(f"k and n must be ints, got k={k!r} n={n!r}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k} n={n}")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        raise ValueError(f"confidence must be a finite number in (0, 1), got {confidence!r}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in the open interval (0, 1), got {confidence}")
    if n == 0:
        return None
    interval = binomtest(k, n, alternative="less").proportion_ci(confidence_level=confidence, method="exact")
    return float(interval.high)


def bonferroni_confidence(joint_confidence: float, n_bounds: int) -> float:
    """Per-bound confidence level so that ``n_bounds`` bounds hold jointly at ``joint_confidence``.

    Returns ``1 - (1 - joint_confidence) / n_bounds``, which is in
    ``[joint_confidence, 1)``. Raises ``ValueError`` unless
    ``0 < joint_confidence < 1`` and ``n_bounds`` is a positive ``int``.
    """
    if (
        isinstance(joint_confidence, bool)
        or not isinstance(joint_confidence, (int, float))
        or not math.isfinite(joint_confidence)
        or not 0 < joint_confidence < 1
    ):
        raise ValueError(f"joint_confidence must be in the open interval (0, 1), got {joint_confidence!r}")
    if isinstance(n_bounds, bool) or not isinstance(n_bounds, int) or n_bounds < 1:
        raise ValueError(f"n_bounds must be a positive int, got {n_bounds!r}")
    return 1 - (1 - joint_confidence) / n_bounds


_SUPPORTED_DESIGN_REASON = (
    "uniform random sample of fresh groups with a documented independence assumption"
)


def sampling_design_supported(design: Mapping[str, Any]) -> tuple[bool, str]:
    """Whether confidence bounds may be computed for a sample drawn under ``design``.

    Supported only when ``design["unit"] == "group"``,
    ``design["method"] == "uniform_random"``,
    ``design["independence_assumption_documented"] is True`` and
    ``design["fresh_groups"] is True``. Returns ``(True, reason)`` in that case
    and ``(False, reason)`` otherwise, where ``reason`` names every unmet
    requirement. Never raises for a missing key; a non-mapping ``design`` is
    simply unsupported. Intervals must only be computed when this returns
    ``True``.
    """
    if not isinstance(design, Mapping):
        return False, f"sampling design must be a mapping, got {type(design).__name__}"
    problems: list[str] = []
    unit = design.get("unit")
    if unit != "group":
        problems.append(f"unit must be 'group' (got {unit!r}): bounds assume independent groups, not traces")
    method = design.get("method")
    if method != "uniform_random":
        problems.append(f"method must be 'uniform_random' (got {method!r}): only uniform sampling is supported")
    if design.get("independence_assumption_documented") is not True:
        problems.append("independence_assumption_documented must be True")
    if design.get("fresh_groups") is not True:
        problems.append("fresh_groups must be True: groups previously exposed to the grader are not a fresh sample")
    if problems:
        return False, "unsupported sampling design: " + "; ".join(problems)
    return True, f"supported: {_SUPPORTED_DESIGN_REASON}"
