"""Shortlisting grader candidates and forming a behaviorally diverse committee.

Inputs are small summaries of grader versions (``CandidateSummary``) and their
*probe predictions* (machine verdicts on a shared probe set). Machine verdicts
are only ever compared with each other here, never with human labels: the
committee is chosen for behavioral diversity among candidates that already
cleared a dev-agreement quality floor.

Every function is pure, deterministic, and free of I/O. Distances with too few
shared cases are ``None`` in code and ``NOT_ESTIMABLE`` in the log, never ``0``.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from eval_tinder.domain.disagreement import NOT_ESTIMABLE, is_vote

logger = logging.getLogger(__name__)

# Tolerance when comparing a candidate's agreement with ``best - quality_gap`` so that a
# candidate sitting exactly on the floor is never dropped by floating-point rounding.
FLOOR_TOLERANCE = 1e-9

# Shortlist exclusion reasons.
REASON_UNSUPPORTED_PROGRAM = "unsupported_program"
REASON_INCOMPLETE_DEV_EVALUATION = "incomplete_dev_evaluation"
REASON_DUPLICATE_MANIFEST = "duplicate_manifest"
REASON_BELOW_QUALITY_FLOOR = "below_quality_floor"
REASON_SHORTLIST_CAP = "shortlist_cap"

# Committee log actions and reasons.
ACTION_SEED = "seed"
ACTION_ADDED = "added"
ACTION_SKIPPED = "skipped"
ACTION_STOPPED = "stopped"
REASON_EMPTY_SHORTLIST = "empty_shortlist"
REASON_NO_PROBE_PREDICTIONS = "no_probe_predictions"
REASON_DUPLICATE_GRADER = "duplicate_grader_id"
REASON_INSUFFICIENT_COVERAGE = "insufficient_shared_coverage"
REASON_NO_DIVERSITY = "no_behavioral_diversity"
REASON_CANDIDATES_EXHAUSTED = "candidates_exhausted"
REASON_COMMITTEE_FULL = "committee_full"
REASON_MAX_DISTANCE = "max_min_distance"
REASON_FIRST_ELIGIBLE = "first_eligible_in_shortlist_order"


@dataclass(frozen=True)
class CandidateSummary:
    """What committee logic knows about one grader version.

    ``dev_agreement`` is agreement with *human* dev labels measured elsewhere;
    ``None`` means it was never measured. Predictions and labels are never
    carried here.
    """

    grader_id: str
    manifest_hash: str
    dev_agreement: float | None
    dev_complete: bool
    prompt_length: int
    usable: bool = True
    note: str = ""


@dataclass
class ShortlistResult:
    """Outcome of ``shortlist_candidates``. Every excluded candidate appears once in ``exclusions``."""

    shortlisted: list[CandidateSummary]
    exclusions: list[dict[str, Any]]
    quality_floor: float | None
    best_agreement: float | None


@dataclass
class CommitteeResult:
    """Outcome of ``form_committee``. ``members`` are grader ids in the order they were chosen."""

    members: list[str]
    log: list[dict[str, Any]]
    diversity_claimed: bool
    reason: str


def _order_key(c: CandidateSummary) -> tuple[int, float, int, str, str]:
    """Total order used everywhere: higher agreement, shorter prompt, manifest hash, grader id.

    Candidates without an agreement sort last. Because the order is total over
    (``manifest_hash``, ``grader_id``), results never depend on input order.
    """
    missing = c.dev_agreement is None
    agreement = 0.0 if missing else float(c.dev_agreement)  # type: ignore[arg-type]
    return (int(missing), -agreement, c.prompt_length, c.manifest_hash, c.grader_id)


def _exclusion(c: CandidateSummary, reason: str) -> dict[str, Any]:
    return {"grader_id": c.grader_id, "reason": reason}


def shortlist_candidates(
    candidates: Sequence[CandidateSummary], *, quality_gap: float = 0.10, max_shortlist: int = 12
) -> ShortlistResult:
    """Reduce grader candidates to a quality-floored, de-duplicated, capped shortlist.

    Steps, in order (each candidate is excluded at most once, with the first applicable reason):
    1. ``usable=False`` -> ``unsupported_program``;
    2. ``dev_complete=False`` or ``dev_agreement=None`` -> ``incomplete_dev_evaluation``;
    3. exact ``manifest_hash`` duplicates -> ``duplicate_manifest``, keeping the first in the
       deterministic order (``-dev_agreement, prompt_length, manifest_hash, grader_id``);
    4. ``best_agreement`` is the maximum remaining agreement and ``quality_floor = best - quality_gap``;
       candidates with ``dev_agreement < quality_floor`` (beyond ``FLOOR_TOLERANCE``) -> ``below_quality_floor``;
    5. the survivors are sorted by (``-dev_agreement, prompt_length, manifest_hash``) and capped to
       ``max_shortlist``; the extras -> ``shortlist_cap``.

    Guarantees: the output is identical for any permutation of ``candidates``; ``quality_floor`` and
    ``best_agreement`` are ``None`` iff nothing survives steps 1-3; the floor is logged at INFO level.
    Raises ``ValueError`` for a negative ``quality_gap`` or a ``max_shortlist`` below 1.
    """
    if quality_gap < 0:
        raise ValueError(f"quality_gap must be non-negative, got {quality_gap!r}")
    if max_shortlist < 1:
        raise ValueError(f"max_shortlist must be at least 1, got {max_shortlist!r}")

    exclusions: list[dict[str, Any]] = []
    pool = sorted(candidates, key=_order_key)

    kept: list[CandidateSummary] = []
    for c in pool:
        if c.usable:
            kept.append(c)
        else:
            exclusions.append(_exclusion(c, REASON_UNSUPPORTED_PROGRAM))
    pool, kept = kept, []

    for c in pool:
        if c.dev_complete and c.dev_agreement is not None:
            kept.append(c)
        else:
            exclusions.append(_exclusion(c, REASON_INCOMPLETE_DEV_EVALUATION))
    pool, kept = kept, []

    seen_hashes: set[str] = set()
    for c in pool:
        if c.manifest_hash in seen_hashes:
            exclusions.append(_exclusion(c, REASON_DUPLICATE_MANIFEST))
        else:
            seen_hashes.add(c.manifest_hash)
            kept.append(c)
    pool, kept = kept, []

    best_agreement: float | None = None
    quality_floor: float | None = None
    if pool:
        best_agreement = max(float(c.dev_agreement) for c in pool)  # type: ignore[arg-type]
        quality_floor = best_agreement - quality_gap
        for c in pool:
            if float(c.dev_agreement) >= quality_floor - FLOOR_TOLERANCE:  # type: ignore[arg-type]
                kept.append(c)
            else:
                exclusions.append(_exclusion(c, REASON_BELOW_QUALITY_FLOOR))
        pool, kept = kept, []

    pool.sort(key=_order_key)
    shortlisted = pool[:max_shortlist]
    exclusions.extend(_exclusion(c, REASON_SHORTLIST_CAP) for c in pool[max_shortlist:])

    logger.info(
        "shortlist: candidates=%d best_agreement=%s quality_gap=%.4f quality_floor=%s shortlisted=%d excluded=%d",
        len(candidates),
        NOT_ESTIMABLE if best_agreement is None else f"{best_agreement:.4f}",
        quality_gap,
        NOT_ESTIMABLE if quality_floor is None else f"{quality_floor:.4f}",
        len(shortlisted),
        len(exclusions),
    )
    return ShortlistResult(
        shortlisted=shortlisted, exclusions=exclusions, quality_floor=quality_floor, best_agreement=best_agreement
    )


def prediction_distance(
    a: Mapping[str, str | None], b: Mapping[str, str | None], *, min_shared: int = 5
) -> tuple[float | None, int]:
    """Fraction of shared probe cases on which two graders' verdicts differ.

    A case is *shared* when both mappings hold a valid verdict for it (``None`` and any
    non-verdict value mean "no vote" and are not shared). Returns ``(distance, shared_count)``,
    where ``distance`` is ``None`` when ``shared_count < min_shared`` or ``shared_count == 0``
    (never ``0`` for an empty denominator). The result is symmetric in ``a`` and ``b`` and
    independent of key order.
    """
    shared = [k for k in a if k in b and is_vote(a[k]) and is_vote(b[k])]
    n = len(shared)
    if n == 0 or n < min_shared:
        return None, n
    differing = sum(1 for k in shared if a[k] != b[k])
    return differing / n, n


def _log_entry(
    round_no: int,
    action: str,
    reason: str,
    grader_id: str | None,
    *,
    min_distance: float | None = None,
    min_shared: int | None = None,
) -> dict[str, Any]:
    return {
        "round": round_no,
        "action": action,
        "reason": reason,
        "grader_id": grader_id,
        "min_distance": NOT_ESTIMABLE if min_distance is None else min_distance,
        "min_shared": min_shared,
    }


def _has_predictions(preds: Mapping[str, str | None] | None) -> bool:
    return bool(preds) and any(is_vote(v) for v in preds.values())  # type: ignore[union-attr]


def _selection_key(min_distance: float, c: CandidateSummary) -> tuple[float, float, int, str, str]:
    """Larger min distance, then higher agreement, shorter prompt, manifest hash, grader id (all ascending keys)."""
    agreement = float("-inf") if c.dev_agreement is None else float(c.dev_agreement)
    return (-min_distance, -agreement, c.prompt_length, c.manifest_hash, c.grader_id)


def form_committee(
    shortlisted: Sequence[CandidateSummary],
    probe_predictions: Mapping[str, Mapping[str, str | None]],
    *,
    size: int = 4,
    min_shared: int = 5,
) -> CommitteeResult:
    """Greedily assemble a committee of behaviorally different graders from a shortlist.

    The seed is the first candidate in ``shortlisted`` order that has at least one valid probe
    verdict (candidates without any are skipped and logged). Each round adds the remaining
    candidate whose *minimum* ``prediction_distance`` to the current members is largest, breaking
    ties by higher ``dev_agreement``, shorter ``prompt_length``, then ``manifest_hash`` ascending.

    Stopping rules (``reason`` names the one that fired):
    - ``committee_full`` when ``size`` members are chosen;
    - ``no_behavioral_diversity`` when the best available minimum distance is not ``> 0``;
    - ``candidates_exhausted`` when no candidate remains;
    - ``insufficient_shared_coverage`` when every remaining candidate lacks ``min_shared`` shared
      valid verdicts with some member (such candidates are skipped, with a logged reason, and are
      never added: coverage cannot improve as members are added);
    - ``empty_shortlist`` / ``no_probe_predictions`` when no seed exists.

    Guarantees: members are always drawn from ``shortlisted`` (never fabricated) and are unique;
    ``diversity_claimed`` is ``True`` only when at least two members were added on the basis of a
    positive measured distance with adequate coverage (a seed alone, or a seed plus one, never claims
    diversity); identical inputs give identical output. Raises ``ValueError`` when ``size < 1``.
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size!r}")

    log: list[dict[str, Any]] = []
    if not shortlisted:
        log.append(_log_entry(0, ACTION_STOPPED, REASON_EMPTY_SHORTLIST, None))
        return CommitteeResult(members=[], log=log, diversity_claimed=False, reason=REASON_EMPTY_SHORTLIST)

    eligible: list[CandidateSummary] = []
    seen_ids: set[str] = set()
    for c in shortlisted:
        if c.grader_id in seen_ids:
            log.append(_log_entry(0, ACTION_SKIPPED, REASON_DUPLICATE_GRADER, c.grader_id))
            continue
        seen_ids.add(c.grader_id)
        if not _has_predictions(probe_predictions.get(c.grader_id)):
            log.append(_log_entry(0, ACTION_SKIPPED, REASON_NO_PROBE_PREDICTIONS, c.grader_id))
            continue
        eligible.append(c)
    if not eligible:
        log.append(_log_entry(0, ACTION_STOPPED, REASON_NO_PROBE_PREDICTIONS, None))
        return CommitteeResult(members=[], log=log, diversity_claimed=False, reason=REASON_NO_PROBE_PREDICTIONS)

    seed, pool = eligible[0], eligible[1:]
    members: list[CandidateSummary] = [seed]
    log.append(_log_entry(0, ACTION_SEED, REASON_FIRST_ELIGIBLE, seed.grader_id))
    diverse_additions = 0
    reason = REASON_COMMITTEE_FULL
    round_no = 0

    while len(members) < size:
        round_no += 1
        if not pool:
            reason = REASON_CANDIDATES_EXHAUSTED
            log.append(_log_entry(round_no, ACTION_STOPPED, reason, None))
            break

        scored: list[tuple[float, int, CandidateSummary]] = []
        for c in pool:
            distances: list[float | None] = []
            shared_counts: list[int] = []
            for m in members:
                d, n = prediction_distance(
                    probe_predictions[c.grader_id], probe_predictions[m.grader_id], min_shared=min_shared
                )
                distances.append(d)
                shared_counts.append(n)
            if any(d is None for d in distances):
                log.append(
                    _log_entry(
                        round_no, ACTION_SKIPPED, REASON_INSUFFICIENT_COVERAGE, c.grader_id,
                        min_distance=None, min_shared=min(shared_counts),
                    )
                )
                continue
            scored.append((min(d for d in distances if d is not None), min(shared_counts), c))

        if not scored:
            reason = REASON_INSUFFICIENT_COVERAGE
            log.append(_log_entry(round_no, ACTION_STOPPED, reason, None))
            break

        best_distance, best_shared, best = min(scored, key=lambda t: _selection_key(t[0], t[2]))
        if not best_distance > 0:
            reason = REASON_NO_DIVERSITY
            log.append(
                _log_entry(
                    round_no, ACTION_STOPPED, reason, best.grader_id,
                    min_distance=best_distance, min_shared=best_shared,
                )
            )
            break

        members.append(best)
        diverse_additions += 1
        log.append(
            _log_entry(
                round_no, ACTION_ADDED, REASON_MAX_DISTANCE, best.grader_id,
                min_distance=best_distance, min_shared=best_shared,
            )
        )
        pool = [c for _, _, c in scored if c is not best]
    else:
        log.append(_log_entry(round_no + 1, ACTION_STOPPED, REASON_COMMITTEE_FULL, None))

    diversity_claimed = diverse_additions >= 2
    logger.info(
        "committee: members=%s diverse_additions=%d diversity_claimed=%s reason=%s",
        [m.grader_id for m in members], diverse_additions, diversity_claimed, reason,
    )
    return CommitteeResult(
        members=[m.grader_id for m in members], log=log, diversity_claimed=diversity_claimed, reason=reason
    )
