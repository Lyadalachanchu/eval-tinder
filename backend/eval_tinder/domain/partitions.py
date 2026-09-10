"""Deterministic group-level partition assignment.

Partitions are assigned per *group* (never per trace) so related traces can
never straddle TRAIN and the held-out partitions. Assignment is a pure
function of ``(seed, group_id)``: importing the same group again, on any
machine, in any order, alongside any other groups, yields the same partition.

This module has no database or model dependencies. The partition names here
mirror ``eval_tinder.db.enums.Partition`` as plain strings.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from eval_tinder.ids import sha256_hex

PARTITIONS: tuple[str, ...] = ("TRAIN", "DEV", "AUDIT_RESERVE")

DEFAULT_SPLIT: dict[str, float] = {"TRAIN": 0.70, "DEV": 0.15, "AUDIT_RESERVE": 0.15}

_SUM_TOLERANCE = 1e-9
_HEX_DIGITS = 16  # 64 bits of the digest => u has 2**64 distinct values in [0, 1)


def validate_split(split: Mapping[str, float]) -> dict[str, float]:
    """Return a normalized copy of ``split`` keyed in ``PARTITIONS`` order.

    Guarantees on return: the keys are exactly ``set(PARTITIONS)``, every share
    is a finite ``float >= 0``, and the shares sum to ``1.0`` within ``1e-9``.
    Raises ``ValueError`` (never silently repairs) for anything else.
    """
    if not isinstance(split, Mapping):
        raise ValueError(f"split must be a mapping of partition -> share, got {type(split).__name__}")
    keys = set(split)
    expected = set(PARTITIONS)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"split keys must be exactly {sorted(expected)}; missing={missing} unexpected={extra}")
    normalized: dict[str, float] = {}
    for name in PARTITIONS:
        raw = split[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"split[{name!r}] must be a number, got {type(raw).__name__}")
        share = float(raw)
        if not math.isfinite(share):
            raise ValueError(f"split[{name!r}] must be finite, got {share}")
        if share < 0:
            raise ValueError(f"split[{name!r}] must be >= 0, got {share}")
        normalized[name] = share
    total = math.fsum(normalized.values())
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise ValueError(f"split shares must sum to 1.0 (within {_SUM_TOLERANCE}), got {total!r}")
    return normalized


def _unit_interval(seed: int, group_id: str) -> float:
    """Map ``(seed, group_id)`` to a float in ``[0, 1)`` via the first 64 bits of SHA-256."""
    digest = sha256_hex(f"{seed}:{group_id}")
    return int(digest[:_HEX_DIGITS], 16) / 2**64


def assign_partition(group_id: str, seed: int, split: Mapping[str, float]) -> str:
    """Assign ``group_id`` to one of ``PARTITIONS`` deterministically.

    ``u = sha256(f"{seed}:{group_id}")[:16 hex] / 2**64`` lies in ``[0, 1)`` and
    is compared against the cumulative shares in ``PARTITIONS`` order
    (TRAIN, then DEV, then AUDIT_RESERVE). A partition with share ``0`` is never
    returned. The result depends only on ``seed``, ``group_id`` and ``split``,
    never on other groups, call order, or process state.

    Raises ``ValueError`` for an invalid split, a non-``int`` seed, or an empty
    or non-string ``group_id``.
    """
    shares = validate_split(split)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {type(seed).__name__}")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("group_id must be a non-empty string")
    u = _unit_interval(seed, group_id)
    cumulative = 0.0
    for name in PARTITIONS:
        cumulative += shares[name]
        if u < cumulative and shares[name] > 0:
            return name
    # Only reachable when floating-point accumulation leaves the final cumulative
    # threshold a hair below 1.0 and u lands in that gap: fall back to the last
    # partition that actually has a share.
    for name in reversed(PARTITIONS):
        if shares[name] > 0:
            return name
    raise AssertionError("unreachable: validate_split guarantees a positive share")  # pragma: no cover


def expected_counts(group_count: int, split: Mapping[str, float]) -> dict[str, int]:
    """Expected number of groups per partition for ``group_count`` groups.

    Uses largest-remainder rounding so the counts are non-negative integers
    that always sum to exactly ``group_count``; ties go to earlier partitions in
    ``PARTITIONS`` order. This is the expectation of the hash-based assignment,
    not the realized outcome for any particular set of group ids.

    Raises ``ValueError`` for a negative or non-``int`` ``group_count`` or an
    invalid split.
    """
    shares = validate_split(split)
    if isinstance(group_count, bool) or not isinstance(group_count, int):
        raise ValueError(f"group_count must be an int, got {type(group_count).__name__}")
    if group_count < 0:
        raise ValueError(f"group_count must be >= 0, got {group_count}")
    exact = {name: group_count * shares[name] for name in PARTITIONS}
    counts = {name: int(math.floor(exact[name])) for name in PARTITIONS}
    remainder = group_count - sum(counts.values())
    # Stable sort: descending fractional part, ties broken by PARTITIONS order.
    by_fraction = sorted(PARTITIONS, key=lambda name: -(exact[name] - counts[name]))
    for name in by_fraction[:remainder]:
        counts[name] += 1
    return counts


def small_import_warning(
    group_count: int, split: Mapping[str, float], *, min_groups: int = 8
) -> str | None:
    """Human-readable warning when DEV or AUDIT_RESERVE would receive too few groups.

    Returns ``None`` when both held-out partitions are expected to receive at
    least ``min_groups`` groups, otherwise a message naming each short
    partition with its expected count and explaining that independent
    evaluation may lack enough material. Never raises for small inputs; raises
    ``ValueError`` only for invalid arguments.
    """
    if isinstance(min_groups, bool) or not isinstance(min_groups, int) or min_groups < 0:
        raise ValueError(f"min_groups must be a non-negative int, got {min_groups!r}")
    counts = expected_counts(group_count, split)
    short = [name for name in ("DEV", "AUDIT_RESERVE") if counts[name] < min_groups]
    if not short:
        return None
    detail = ", ".join(f"{name} ~{counts[name]} group(s)" for name in short)
    return (
        f"Small import: with {group_count} group(s) and the configured split, the held-out partitions "
        f"are expected to receive fewer than {min_groups} groups each ({detail}). "
        "Independent evaluation may lack enough material: DEV agreement estimates will be noisy and the "
        "AUDIT_RESERVE may be too small to bound the error rate. Consider importing more groups before "
        "relying on the evaluation numbers."
    )
