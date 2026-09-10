"""Disagreement statistics over committee votes.

A *vote* is a machine verdict (``PASS``, ``FAIL`` or ``REVIEW``) produced by one
committee member for one case. ``None`` marks an operational error (the grading
call produced no verdict) and is never counted as a vote. Machine votes are
never human labels; nothing here compares them with human verdicts.

Every function is pure and deterministic. A quantity whose denominator would be
zero is reported as ``None`` in code (and as ``NOT_ESTIMABLE`` in logs), never
as ``0``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

VALID_VOTES = ("PASS", "FAIL", "REVIEW")
NOT_ESTIMABLE = "NOT_ESTIMABLE"


def is_vote(value: Any) -> bool:
    """True iff ``value`` is a valid verdict string (``None`` and anything else are not votes)."""
    return isinstance(value, str) and value in VALID_VOTES


def _split_votes(votes: Sequence[str | None]) -> tuple[list[str], int]:
    """Return ``(valid_votes, error_count)``.

    ``None`` entries count as errors. Any other value outside ``VALID_VOTES``
    raises ``ValueError`` so malformed data never silently changes a statistic.
    """
    valid: list[str] = []
    errors = 0
    for vote in votes:
        if vote is None:
            errors += 1
        elif is_vote(vote):
            valid.append(str(vote))
        else:
            raise ValueError(f"invalid vote {vote!r}; expected one of {VALID_VOTES} or None")
    return valid, errors


def gini_disagreement(votes: Sequence[str | None], *, min_valid: int = 2) -> float | None:
    """Gini impurity of the valid votes: ``1 - sum_v p_v ** 2``.

    Guarantees:
    - ``None`` entries are operational errors: ignored, never counted as votes.
    - Any other value outside ``VALID_VOTES`` raises ``ValueError``.
    - Returns ``None`` (not estimable) when fewer than ``min_valid`` valid votes remain,
      and always when no valid vote remains, whatever ``min_valid`` is.
    - Otherwise returns a float in ``[0, 1 - 1/len(VALID_VOTES)]``; identical votes give exactly ``0.0``.
    """
    valid, _ = _split_votes(votes)
    n = len(valid)
    if n == 0 or n < min_valid:
        return None
    return 1.0 - sum((valid.count(v) / n) ** 2 for v in VALID_VOTES)


def vote_summary(votes: Sequence[str | None]) -> dict[str, Any]:
    """Summarise one case's votes.

    Returns a dict with keys:
    - ``counts``: ``{verdict: count}`` for every verdict in ``VALID_VOTES`` (``0`` when absent);
    - ``valid_count``: number of valid votes;
    - ``error_count``: number of ``None`` entries (operational errors, not votes);
    - ``all_review``: ``valid_count >= 1`` and every valid vote is ``REVIEW``;
    - ``unanimous``: ``valid_count >= 1`` and every valid vote is the same verdict.

    Raises ``ValueError`` on any non-``None`` value outside ``VALID_VOTES``.
    """
    valid, errors = _split_votes(votes)
    counts = {v: valid.count(v) for v in VALID_VOTES}
    distinct = set(valid)
    return {
        "counts": counts,
        "valid_count": len(valid),
        "error_count": errors,
        "all_review": distinct == {"REVIEW"},
        "unanimous": len(distinct) == 1,
    }
