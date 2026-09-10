"""Batch selection: which pooled traces humans should judge next.

Two pure steps live here:

1. ``build_pool`` draws a stratified random *pool* of candidate traces (at
   most one per group) from everything that could be reviewed.
2. ``select_batch`` picks one review *batch* from that pool using three
   categories: ``DISAGREEMENT`` (the committee split on the case),
   ``COVERAGE`` (strata with few human labels) and ``RANDOM`` (an unbiased
   sample drawn independently of any machine score).

Disagreement scores summarise *machine* votes. They decide what a human sees
next; they are never labels, and every reason emitted here says so
(``committee_votes_hidden_until_judged``). Cases whose every valid vote was
``REVIEW`` are routed to ``context_repair`` instead of being selected.

Every function is deterministic given ``seed`` (``random.Random(seed)``, never
the global generator), free of I/O, and never raises for a pool that is merely
too small: it selects what it can and records the shortfall. A share whose
denominator is zero is logged as ``NOT_ESTIMABLE``, never ``0``.
"""
from __future__ import annotations

import math
import random
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from eval_tinder.domain.disagreement import NOT_ESTIMABLE

STRATEGY_VERSION = "select-v1"

UNSTRATIFIED = "_unstratified"

# Category names mirror ``eval_tinder.db.enums.SelectionCategory`` as plain strings.
CATEGORY_DISAGREEMENT = "DISAGREEMENT"
CATEGORY_COVERAGE = "COVERAGE"
CATEGORY_RANDOM = "RANDOM"
CATEGORIES: tuple[str, ...] = (CATEGORY_DISAGREEMENT, CATEGORY_COVERAGE, CATEGORY_RANDOM)

# Quota keys (lower-case) and the category each quota fills, in processing order of the fallback.
QUOTA_DISAGREEMENT = "disagreement"
QUOTA_COVERAGE = "coverage"
QUOTA_RANDOM = "random"
QUOTA_KEYS: tuple[str, ...] = (QUOTA_DISAGREEMENT, QUOTA_COVERAGE, QUOTA_RANDOM)
_QUOTA_CATEGORY = {
    QUOTA_DISAGREEMENT: CATEGORY_DISAGREEMENT,
    QUOTA_COVERAGE: CATEGORY_COVERAGE,
    QUOTA_RANDOM: CATEGORY_RANDOM,
}

DEFAULT_QUOTAS: Mapping[str, int] = MappingProxyType(
    {QUOTA_DISAGREEMENT: 6, QUOTA_COVERAGE: 2, QUOTA_RANDOM: 2}
)

# Log reasons.
REASON_NO_DISAGREEMENT_SCORES = "no_disagreement_scores"
REASON_CANDIDATES_EXHAUSTED = "candidates_exhausted"
REASON_QUOTA_FILLED = "quota_filled"


def stratum_key(strata: Mapping[str, str]) -> str:
    """Stable string key for a strata mapping.

    Guarantees: the key depends only on the ``(name, value)`` pairs, never on
    insertion order (``"language=en|task_type=cancellation"`` for
    ``{"task_type": "cancellation", "language": "en"}``); an empty mapping
    yields ``UNSTRATIFIED`` (``"_unstratified"``). Raises ``ValueError`` for a
    non-mapping or for non-string names or values.
    """
    if not isinstance(strata, Mapping):
        raise ValueError(f"strata must be a mapping of name -> value, got {type(strata).__name__}")
    if not strata:
        return UNSTRATIFIED
    for name, value in strata.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError(f"strata names and values must be strings, got {name!r}={value!r}")
    return "|".join(f"{name}={strata[name]}" for name in sorted(strata))


@dataclass(frozen=True)
class PoolCase:
    """One reviewable trace as the selector sees it.

    ``strata`` are descriptive facets (task type, tool outcome, language, ...)
    and may be empty. ``reading_length`` is the rendered length in characters
    used only for tie-breaking (shorter cases first). Machine verdicts are
    deliberately not part of a pool case.
    """

    trace_id: str
    group_id: str
    strata: Mapping[str, str]
    reading_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id:
            raise ValueError("trace_id must be a non-empty string")
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id must be a non-empty string")
        stratum_key(self.strata)  # validates the mapping
        if isinstance(self.reading_length, bool) or not isinstance(self.reading_length, int):
            raise ValueError(f"reading_length must be an int, got {type(self.reading_length).__name__}")
        if self.reading_length < 0:
            raise ValueError(f"reading_length must be >= 0, got {self.reading_length}")

    @property
    def stratum(self) -> str:
        """``stratum_key(self.strata)``."""
        return stratum_key(self.strata)


@dataclass
class SelectedCase:
    """One batch member. ``rank`` is 1-based within ``category``."""

    trace_id: str
    group_id: str
    category: str
    score: float | None
    rank: int
    reason: dict[str, Any]


@dataclass
class BatchSelection:
    """Result of ``select_batch``.

    ``selected`` lists DISAGREEMENT picks, then COVERAGE picks, then RANDOM
    picks (uniform draws first, then fallback fills). ``context_repair`` holds
    the pool traces routed away because every valid committee vote was
    ``REVIEW``. ``exhausted`` maps a quota key to the number of its slots the
    category could not fill from its own candidates; such slots are filled
    from random eligible cases when any remain. ``log`` is an ordered list of
    JSON-serialisable dicts recording every phase.
    """

    selected: list[SelectedCase]
    context_repair: list[str]
    exhausted: dict[str, int]
    log: list[dict[str, Any]]
    seed: int
    strategy_version: str = STRATEGY_VERSION

    def by_category(self) -> dict[str, list[SelectedCase]]:
        """Selected cases grouped by category, every category present (possibly empty)."""
        grouped: dict[str, list[SelectedCase]] = {category: [] for category in CATEGORIES}
        for case in self.selected:
            grouped[case.category].append(case)
        return grouped


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {type(seed).__name__}")
    return seed


def _validate_size(size: int) -> int:
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError(f"size must be an int, got {type(size).__name__}")
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}")
    return size


def _validate_cases(cases: Sequence[PoolCase], *, what: str) -> None:
    for case in cases:
        if not isinstance(case, PoolCase):
            raise ValueError(f"{what} must contain PoolCase instances, got {type(case).__name__}")


def build_pool(cases: Sequence[PoolCase], *, size: int = 200, seed: int) -> list[PoolCase]:
    """Draw a stratified random pool of at most ``size`` cases.

    Cases are grouped by ``stratum_key``; each group is shuffled with
    ``random.Random(seed)`` (groups visited in sorted key order) and the pool is
    filled round-robin across the sorted stratum keys until ``size`` is reached
    or every group is spent.

    Guarantees: ``len(pool) <= size``; at most one case per ``group_id`` and
    per ``trace_id`` (the first seen after shuffling wins); every stratum with
    cases contributes before any stratum contributes twice (so a stratum with
    ``k`` usable cases contributes ``min(k, ceil-share)``); the result is a
    pure function of ``(cases, size, seed)``, independent of process state.
    Raises ``ValueError`` for a negative or non-``int`` ``size``, a non-``int``
    ``seed``, or a non-``PoolCase`` entry.
    """
    size = _validate_size(size)
    seed = _validate_seed(seed)
    _validate_cases(cases, what="cases")
    rng = random.Random(seed)
    groups: dict[str, list[PoolCase]] = {}
    for case in cases:
        groups.setdefault(case.stratum, []).append(case)
    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])
    positions = {key: 0 for key in keys}
    pool: list[PoolCase] = []
    used_groups: set[str] = set()
    used_traces: set[str] = set()
    while len(pool) < size:
        progressed = False
        for key in keys:
            if len(pool) >= size:
                break
            bucket = groups[key]
            index = positions[key]
            while index < len(bucket) and (
                bucket[index].group_id in used_groups or bucket[index].trace_id in used_traces
            ):
                index += 1
            if index < len(bucket):
                case = bucket[index]
                pool.append(case)
                used_groups.add(case.group_id)
                used_traces.add(case.trace_id)
                index += 1
                progressed = True
            positions[key] = index
        if not progressed:
            break
    return pool


def _validate_quotas(quotas: Mapping[str, int]) -> dict[str, int]:
    """Return ``{quota_key: int >= 0}`` for every key in ``QUOTA_KEYS`` (missing => 0)."""
    if not isinstance(quotas, Mapping):
        raise ValueError(f"quotas must be a mapping, got {type(quotas).__name__}")
    unknown = sorted(set(quotas) - set(QUOTA_KEYS))
    if unknown:
        raise ValueError(f"unknown quota keys {unknown}; expected a subset of {list(QUOTA_KEYS)}")
    normalized: dict[str, int] = {}
    for key in QUOTA_KEYS:
        value = quotas.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"quotas[{key!r}] must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"quotas[{key!r}] must be >= 0, got {value}")
        normalized[key] = value
    return normalized


def _validate_scores(disagreement: Mapping[str, float | None]) -> dict[str, float | None]:
    """Return a copy with every score either ``None`` or a finite float."""
    if not isinstance(disagreement, Mapping):
        raise ValueError(f"disagreement must be a mapping, got {type(disagreement).__name__}")
    scores: dict[str, float | None] = {}
    for trace_id, raw in disagreement.items():
        if raw is None:
            scores[trace_id] = None
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"disagreement[{trace_id!r}] must be a number or None, got {type(raw).__name__}")
        score = float(raw)
        if not math.isfinite(score):
            raise ValueError(f"disagreement[{trace_id!r}] must be finite, got {score}")
        scores[trace_id] = score
    return scores


def _validate_counts(labeled_strata_counts: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(labeled_strata_counts, Mapping):
        raise ValueError(
            f"labeled_strata_counts must be a mapping, got {type(labeled_strata_counts).__name__}"
        )
    counts: dict[str, int] = {}
    for key, raw in labeled_strata_counts.items():
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"labeled_strata_counts[{key!r}] must be an int, got {type(raw).__name__}")
        if raw < 0:
            raise ValueError(f"labeled_strata_counts[{key!r}] must be >= 0, got {raw}")
        counts[key] = raw
    return counts


@dataclass
class _Batch:
    """Mutable bookkeeping shared by the selection phases."""

    picks: dict[str, list[SelectedCase]] = field(
        default_factory=lambda: {category: [] for category in CATEGORIES}
    )
    used_groups: set[str] = field(default_factory=set)
    used_traces: set[str] = field(default_factory=set)

    def is_free(self, case: PoolCase) -> bool:
        return case.trace_id not in self.used_traces and case.group_id not in self.used_groups

    def take(
        self, case: PoolCase, category: str, score: float | None, reason: dict[str, Any]
    ) -> SelectedCase:
        picks = self.picks[category]
        selected = SelectedCase(
            trace_id=case.trace_id,
            group_id=case.group_id,
            category=category,
            score=score,
            rank=len(picks) + 1,
            reason=reason,
        )
        picks.append(selected)
        self.used_groups.add(case.group_id)
        self.used_traces.add(case.trace_id)
        return selected


def _share(numerator: int, denominator: int) -> float | str:
    return NOT_ESTIMABLE if denominator == 0 else numerator / denominator


def _pick_random(batch: _Batch, order: Sequence[PoolCase], quota: int) -> int:
    """Walk the pre-shuffled ``order`` and take up to ``quota`` free cases as RANDOM picks."""
    taken = 0
    for case in order:
        if taken >= quota:
            break
        if not batch.is_free(case):
            continue
        batch.take(
            case,
            CATEGORY_RANDOM,
            None,
            {
                "draw": taken + 1,
                "independent_of_score": True,
                "stratum": case.stratum,
                "committee_votes_hidden_until_judged": True,
            },
        )
        taken += 1
    return taken


def _pick_disagreement(
    batch: _Batch,
    eligible: Sequence[PoolCase],
    scores: Mapping[str, float | None],
    tiebreak: Mapping[str, float],
    quota: int,
) -> dict[str, Any]:
    """Fill DISAGREEMENT slots by score, preferring unseen strata; returns the log entry."""
    candidates = [
        case
        for case in eligible
        if batch.is_free(case) and scores.get(case.trace_id) is not None and scores[case.trace_id] > 0
    ]
    candidates.sort(key=lambda case: (-scores[case.trace_id], case.reading_length, tiebreak[case.trace_id]))
    scored = sum(1 for case in eligible if scores.get(case.trace_id) is not None)
    entry: dict[str, Any] = {
        "event": "disagreement",
        "quota": quota,
        "candidates": len(candidates),
        "scored": scored,
        "scored_share": _share(scored, len(eligible)),
        "filled": 0,
        "deferred_allowed": 0,
    }
    if quota == 0:
        entry["reason"] = REASON_QUOTA_FILLED
        return entry
    if not candidates:
        entry["reason"] = REASON_NO_DISAGREEMENT_SCORES
        entry["note"] = "no committee scores above 0 (no committee, all votes agree, or all cases errored)"
        return entry

    def take(case: PoolCase, deferred: bool) -> None:
        batch.take(
            case,
            CATEGORY_DISAGREEMENT,
            scores[case.trace_id],
            {
                "score": scores[case.trace_id],
                "stratum": case.stratum,
                "stratum_already_selected": deferred,
                "committee_votes_hidden_until_judged": True,
            },
        )
        entry["filled"] += 1
        if deferred:
            entry["deferred_allowed"] += 1

    used_strata: set[str] = set()
    deferred: list[PoolCase] = []
    for case in candidates:
        if entry["filled"] >= quota:
            break
        if not batch.is_free(case):
            continue
        key = case.stratum
        if key in used_strata:
            deferred.append(case)
            continue
        take(case, deferred=False)
        used_strata.add(key)
    for case in deferred:
        if entry["filled"] >= quota:
            break
        if not batch.is_free(case):
            continue
        take(case, deferred=True)
    entry["reason"] = REASON_QUOTA_FILLED if entry["filled"] >= quota else REASON_CANDIDATES_EXHAUSTED
    return entry


def _pick_coverage(
    batch: _Batch,
    eligible: Sequence[PoolCase],
    counts: Mapping[str, int],
    tiebreak: Mapping[str, float],
    quota: int,
) -> dict[str, Any]:
    """Fill COVERAGE slots from the least-labeled strata; returns the log entry."""
    effective: dict[str, int] = dict(counts)
    candidates = [case for case in eligible if batch.is_free(case)]
    entry: dict[str, Any] = {
        "event": "coverage",
        "quota": quota,
        "candidates": len(candidates),
        "strata_labeled_counts": {case.stratum: effective.get(case.stratum, 0) for case in candidates},
        "filled": 0,
    }
    while entry["filled"] < quota and candidates:
        candidates.sort(
            key=lambda case: (effective.get(case.stratum, 0), case.reading_length, tiebreak[case.trace_id])
        )
        case = candidates[0]
        key = case.stratum
        labeled = effective.get(key, 0)
        batch.take(
            case,
            CATEGORY_COVERAGE,
            None,
            {
                "stratum": key,
                "labeled_count": labeled,
                "committee_votes_hidden_until_judged": True,
            },
        )
        entry["filled"] += 1
        effective[key] = labeled + 1
        candidates = [other for other in candidates if batch.is_free(other)]
    entry["reason"] = REASON_QUOTA_FILLED if entry["filled"] >= quota else REASON_CANDIDATES_EXHAUSTED
    return entry


def select_batch(
    pool: Sequence[PoolCase],
    *,
    disagreement: Mapping[str, float | None],
    all_review: Collection[str],
    labeled_strata_counts: Mapping[str, int],
    quotas: Mapping[str, int] = DEFAULT_QUOTAS,
    seed: int,
    excluded_trace_ids: Collection[str] = (),
) -> BatchSelection:
    """Select one review batch from ``pool``.

    Inputs: ``disagreement`` maps trace id -> committee disagreement score
    (``None`` = not estimable; ids absent from the pool are ignored);
    ``all_review`` holds trace ids whose every valid committee vote was
    ``REVIEW``; ``labeled_strata_counts`` maps ``stratum_key`` -> number of
    human labels already collected (missing => 0); ``quotas`` maps
    ``"disagreement"`` / ``"coverage"`` / ``"random"`` -> slot count (missing
    => 0); ``excluded_trace_ids`` are never selected nor routed.

    Rules, in order:

    1. Eligible = pool minus ``excluded_trace_ids`` minus ``all_review``; the
       ``all_review`` ids still in play are returned as ``context_repair``.
    2. RANDOM slots are drawn first, uniformly from eligible with
       ``random.Random(seed)``; the same ``(pool, excluded, all_review, seed)``
       yields the same random picks whatever ``disagreement`` says.
    3. DISAGREEMENT slots take cases with a score ``> 0`` in descending score
       order, ties broken by ``reading_length`` ascending then a per-case draw
       from ``random.Random(seed + 1)``. A case whose stratum already appears
       among the disagreement picks is deferred until every other stratum has
       been used, then allowed.
    4. COVERAGE slots take the free case from the least-labeled stratum (each
       coverage pick counts as one more label for its stratum for the rest of
       the round), same tie-breaks.
    5. No ``group_id`` or ``trace_id`` appears twice in the batch.
    6. Slots a category cannot fill are filled from the remaining eligible
       cases in random order (category ``RANDOM``, reason ``fallback_for``)
       and counted in ``exhausted``.
    7. With no positive score at all (no committee, unanimous votes), the
       whole disagreement quota falls back and the log says so.

    Guarantees: never raises for a pool smaller than the quotas (it selects
    as many as possible and logs ``unfilled``); ``rank`` is 1-based within
    each category; the result is a pure function of the arguments. Raises
    ``ValueError`` for malformed inputs: unknown or negative quotas, a
    non-``int`` seed, a non-finite score, a negative labeled count, or a
    duplicate trace id in the pool.
    """
    normalized_quotas = _validate_quotas(quotas)
    seed = _validate_seed(seed)
    scores = _validate_scores(disagreement)
    counts = _validate_counts(labeled_strata_counts)
    _validate_cases(pool, what="pool")
    seen: set[str] = set()
    for case in pool:
        if case.trace_id in seen:
            raise ValueError(f"duplicate trace_id {case.trace_id!r} in pool")
        seen.add(case.trace_id)
    excluded = set(excluded_trace_ids)
    review = set(all_review)

    context_repair = [
        case.trace_id for case in pool if case.trace_id not in excluded and case.trace_id in review
    ]
    eligible = [case for case in pool if case.trace_id not in excluded and case.trace_id not in review]
    total_quota = sum(normalized_quotas.values())
    log: list[dict[str, Any]] = [
        {
            "event": "start",
            "strategy_version": STRATEGY_VERSION,
            "seed": seed,
            "quotas": dict(normalized_quotas),
            "total_quota": total_quota,
            "pool_size": len(pool),
            "excluded_in_pool": sum(1 for case in pool if case.trace_id in excluded),
            "context_repair": len(context_repair),
            "eligible": len(eligible),
            "eligible_share": _share(len(eligible), len(pool)),
        }
    ]
    batch = _Batch()

    # (2) Random draws come first and depend only on the eligible list and the seed.
    order = list(eligible)
    random.Random(seed).shuffle(order)
    random_filled = _pick_random(batch, order, normalized_quotas[QUOTA_RANDOM])
    log.append(
        {
            "event": "random",
            "quota": normalized_quotas[QUOTA_RANDOM],
            "candidates": len(eligible),
            "filled": random_filled,
            "reason": (
                REASON_QUOTA_FILLED
                if random_filled >= normalized_quotas[QUOTA_RANDOM]
                else REASON_CANDIDATES_EXHAUSTED
            ),
        }
    )

    # Per-case tie-break draws, assigned in pool order so they never depend on scores.
    tie_rng = random.Random(seed + 1)
    tiebreak = {case.trace_id: tie_rng.random() for case in eligible}

    # (3) and (4).
    log.append(_pick_disagreement(batch, eligible, scores, tiebreak, normalized_quotas[QUOTA_DISAGREEMENT]))
    log.append(_pick_coverage(batch, eligible, counts, tiebreak, normalized_quotas[QUOTA_COVERAGE]))

    # (6) and (7): fall back to the random order for every slot a category left open.
    exhausted: dict[str, int] = {}
    for quota_key in QUOTA_KEYS:
        shortfall = normalized_quotas[quota_key] - len(batch.picks[_QUOTA_CATEGORY[quota_key]])
        if shortfall > 0:
            exhausted[quota_key] = shortfall
    fallback_filled: dict[str, int] = {}
    remaining = iter(order)
    for quota_key in (QUOTA_DISAGREEMENT, QUOTA_COVERAGE):
        for _ in range(exhausted.get(quota_key, 0)):
            case = next((candidate for candidate in remaining if batch.is_free(candidate)), None)
            if case is None:
                break
            batch.take(
                case,
                CATEGORY_RANDOM,
                None,
                {
                    "fallback_for": quota_key,
                    "stratum": case.stratum,
                    "committee_votes_hidden_until_judged": True,
                },
            )
            fallback_filled[quota_key] = fallback_filled.get(quota_key, 0) + 1
    if exhausted:
        log.append(
            {
                "event": "fallback",
                "exhausted": dict(exhausted),
                "filled_from_random": dict(fallback_filled),
                "note": (
                    "disagreement quota filled from random eligible cases: no committee scores above 0"
                    if exhausted.get(QUOTA_DISAGREEMENT) == normalized_quotas[QUOTA_DISAGREEMENT]
                    and normalized_quotas[QUOTA_DISAGREEMENT] > 0
                    else "unfilled category slots were filled from random eligible cases"
                ),
            }
        )

    selected: list[SelectedCase] = []
    for category in CATEGORIES:
        selected.extend(batch.picks[category])
    log.append(
        {
            "event": "done",
            "seed": seed,
            "selected": len(selected),
            "counts": {category: len(batch.picks[category]) for category in CATEGORIES},
            "context_repair": len(context_repair),
            "unfilled": total_quota - len(selected),
            "fill_share": _share(len(selected), total_quota),
        }
    )
    return BatchSelection(
        selected=selected,
        context_repair=context_repair,
        exhausted=exhausted,
        log=log,
        seed=seed,
        strategy_version=STRATEGY_VERSION,
    )
