"""Unit tests for deterministic group-level partition assignment."""
from __future__ import annotations

import hashlib
import math
import random

import pytest

from eval_tinder.db.enums import Partition
from eval_tinder.domain.partitions import (
    DEFAULT_SPLIT,
    PARTITIONS,
    assign_partition,
    expected_counts,
    small_import_warning,
    validate_split,
)

SPLIT = {"TRAIN": 0.7, "DEV": 0.15, "AUDIT_RESERVE": 0.15}


def _group_ids(n: int, prefix: str = "grp") -> list[str]:
    return [f"{prefix}-{i:06d}" for i in range(n)]


def _reference_assignment(group_id: str, seed: int, split: dict[str, float]) -> str:
    """The formula from the specification, written independently of the implementation."""
    u = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16], 16) / 2**64
    cumulative = 0.0
    for name in ("TRAIN", "DEV", "AUDIT_RESERVE"):
        cumulative += split[name]
        if u < cumulative:
            return name
    return "AUDIT_RESERVE"


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------


def test_partitions_match_database_enum():
    assert PARTITIONS == ("TRAIN", "DEV", "AUDIT_RESERVE")
    assert tuple(p.value for p in Partition) == PARTITIONS


def test_default_split_is_valid():
    assert validate_split(DEFAULT_SPLIT) == DEFAULT_SPLIT


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_assignment_is_deterministic_across_calls():
    ids = _group_ids(500)
    first = [assign_partition(g, 42, SPLIT) for g in ids]
    second = [assign_partition(g, 42, SPLIT) for g in ids]
    assert first == second


def test_assignment_is_independent_of_order_and_neighbours():
    ids = _group_ids(1000)
    forward = {g: assign_partition(g, 7, SPLIT) for g in ids}
    shuffled = list(ids)
    random.Random(0).shuffle(shuffled)
    backward = {g: assign_partition(g, 7, SPLIT) for g in shuffled}
    subset_only = {g: assign_partition(g, 7, SPLIT) for g in ids[::7]}
    assert forward == backward
    assert all(forward[g] == subset_only[g] for g in subset_only)


def test_assignment_is_independent_of_split_object_identity():
    a = {"TRAIN": 0.7, "DEV": 0.15, "AUDIT_RESERVE": 0.15}
    b = {"AUDIT_RESERVE": 0.15, "TRAIN": 0.7, "DEV": 0.15}  # same shares, different key order
    for g in _group_ids(200):
        assert assign_partition(g, 3, a) == assign_partition(g, 3, b)


def test_assignment_matches_specified_hash_formula():
    for seed in (0, 1, 123456789, -5):
        for g in _group_ids(300, prefix=f"s{seed}"):
            assert assign_partition(g, seed, SPLIT) == _reference_assignment(g, seed, SPLIT)


def test_different_seeds_produce_different_assignments():
    ids = _group_ids(2000)
    with_seed_1 = [assign_partition(g, 1, SPLIT) for g in ids]
    with_seed_2 = [assign_partition(g, 2, SPLIT) for g in ids]
    differing = sum(a != b for a, b in zip(with_seed_1, with_seed_2, strict=True))
    # Two independent draws from a 70/15/15 split disagree ~48% of the time.
    assert differing > 600


def test_similar_group_ids_are_not_correlated():
    ids = [f"g{i}" for i in range(2000)]
    counts = {p: 0 for p in PARTITIONS}
    for g in ids:
        counts[assign_partition(g, 9, SPLIT)] += 1
    for name in PARTITIONS:
        assert abs(counts[name] / len(ids) - SPLIT[name]) < 0.05


# --------------------------------------------------------------------------
# Distribution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "split",
    [
        {"TRAIN": 0.7, "DEV": 0.15, "AUDIT_RESERVE": 0.15},
        {"TRAIN": 0.5, "DEV": 0.3, "AUDIT_RESERVE": 0.2},
        {"TRAIN": 0.8, "DEV": 0.2, "AUDIT_RESERVE": 0.0},
    ],
)
def test_distribution_over_20000_groups_within_two_points(split):
    rng = random.Random(2024)
    ids = [f"group-{rng.getrandbits(64):016x}" for _ in range(20_000)]
    counts = {p: 0 for p in PARTITIONS}
    for g in ids:
        counts[assign_partition(g, 11, split)] += 1
    assert sum(counts.values()) == 20_000
    for name in PARTITIONS:
        observed = counts[name] / 20_000
        assert abs(observed - split[name]) <= 0.02, f"{name}: observed {observed:.4f}, expected {split[name]}"


def test_zero_share_partition_is_never_assigned():
    split = {"TRAIN": 1.0, "DEV": 0.0, "AUDIT_RESERVE": 0.0}
    assert {assign_partition(g, 5, split) for g in _group_ids(2000)} == {"TRAIN"}
    split = {"TRAIN": 0.0, "DEV": 0.0, "AUDIT_RESERVE": 1.0}
    assert {assign_partition(g, 5, split) for g in _group_ids(2000)} == {"AUDIT_RESERVE"}
    split = {"TRAIN": 0.0, "DEV": 1.0, "AUDIT_RESERVE": 0.0}
    assert {assign_partition(g, 5, split) for g in _group_ids(2000)} == {"DEV"}


def test_all_partitions_reachable_with_positive_shares():
    seen = {assign_partition(g, 1, SPLIT) for g in _group_ids(1000)}
    assert seen == set(PARTITIONS)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_split",
    [
        {"TRAIN": 0.7, "DEV": 0.3},  # missing AUDIT_RESERVE
        {"TRAIN": 0.7, "DEV": 0.15, "AUDIT_RESERVE": 0.15, "TEST": 0.0},  # unexpected key
        {"TRAIN": 0.9, "DEV": 0.15, "AUDIT_RESERVE": 0.15},  # sums to 1.2
        {"TRAIN": 0.5, "DEV": 0.25, "AUDIT_RESERVE": 0.2},  # sums to 0.95
        {"TRAIN": 1.2, "DEV": -0.1, "AUDIT_RESERVE": -0.1},  # negative shares
        {"TRAIN": 0.7, "DEV": 0.15, "AUDIT_RESERVE": 0.15 + 1e-6},  # outside tolerance
        {"TRAIN": "0.7", "DEV": 0.15, "AUDIT_RESERVE": 0.15},  # non-numeric
        {"TRAIN": True, "DEV": 0.0, "AUDIT_RESERVE": 0.0},  # bool is not a share
        {"TRAIN": math.nan, "DEV": 0.15, "AUDIT_RESERVE": 0.15},
        {"TRAIN": math.inf, "DEV": 0.15, "AUDIT_RESERVE": 0.15},
        {},
        [("TRAIN", 0.7), ("DEV", 0.15), ("AUDIT_RESERVE", 0.15)],  # not a mapping
        None,
    ],
)
def test_invalid_splits_raise(bad_split):
    with pytest.raises(ValueError):
        validate_split(bad_split)
    with pytest.raises(ValueError):
        assign_partition("g-1", 0, bad_split)
    with pytest.raises(ValueError):
        expected_counts(100, bad_split)


def test_validate_split_accepts_rounding_noise_and_normalizes():
    noisy = {"TRAIN": 0.1 + 0.2, "DEV": 0.7 - 0.3, "AUDIT_RESERVE": 0.3}  # floating-point sum is not exactly 1
    normalized = validate_split(noisy)
    assert list(normalized) == list(PARTITIONS)
    assert all(isinstance(v, float) for v in normalized.values())
    assert math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-9)


def test_validate_split_accepts_integer_shares():
    assert validate_split({"TRAIN": 1, "DEV": 0, "AUDIT_RESERVE": 0}) == {"TRAIN": 1.0, "DEV": 0.0, "AUDIT_RESERVE": 0.0}


def test_validate_split_returns_a_copy():
    original = dict(SPLIT)
    normalized = validate_split(original)
    normalized["TRAIN"] = 0.0
    assert original == SPLIT


@pytest.mark.parametrize("seed", ["42", 4.2, None, True])
def test_assign_partition_rejects_non_int_seed(seed):
    with pytest.raises(ValueError):
        assign_partition("g-1", seed, SPLIT)


@pytest.mark.parametrize("group_id", ["", None, 12, b"g-1"])
def test_assign_partition_rejects_bad_group_id(group_id):
    with pytest.raises(ValueError):
        assign_partition(group_id, 0, SPLIT)


# --------------------------------------------------------------------------
# Expected counts
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group_count", "split", "expected"),
    [
        (0, SPLIT, {"TRAIN": 0, "DEV": 0, "AUDIT_RESERVE": 0}),
        (1, SPLIT, {"TRAIN": 1, "DEV": 0, "AUDIT_RESERVE": 0}),
        (10, SPLIT, {"TRAIN": 7, "DEV": 2, "AUDIT_RESERVE": 1}),  # tie on .5 goes to the earlier partition
        (20, SPLIT, {"TRAIN": 14, "DEV": 3, "AUDIT_RESERVE": 3}),
        (500, SPLIT, {"TRAIN": 350, "DEV": 75, "AUDIT_RESERVE": 75}),
        (3, {"TRAIN": 1 / 3, "DEV": 1 / 3, "AUDIT_RESERVE": 1 / 3}, {"TRAIN": 1, "DEV": 1, "AUDIT_RESERVE": 1}),
        (7, {"TRAIN": 1.0, "DEV": 0.0, "AUDIT_RESERVE": 0.0}, {"TRAIN": 7, "DEV": 0, "AUDIT_RESERVE": 0}),
    ],
)
def test_expected_counts(group_count, split, expected):
    assert expected_counts(group_count, split) == expected


@pytest.mark.parametrize("n", [0, 1, 2, 3, 9, 10, 11, 99, 100, 101, 12345])
def test_expected_counts_always_sum_to_group_count(n):
    rng = random.Random(n)
    raw = [rng.random() + 0.01 for _ in PARTITIONS]
    total = sum(raw)
    split = {name: value / total for name, value in zip(PARTITIONS, raw, strict=True)}
    counts = expected_counts(n, split)
    assert sum(counts.values()) == n
    assert all(v >= 0 for v in counts.values())
    for name in PARTITIONS:
        assert abs(counts[name] - n * split[name]) < 1.0


@pytest.mark.parametrize("bad", [-1, 1.5, "10", None, True])
def test_expected_counts_rejects_bad_group_count(bad):
    with pytest.raises(ValueError):
        expected_counts(bad, SPLIT)


# --------------------------------------------------------------------------
# Small-import warning
# --------------------------------------------------------------------------


def test_warning_present_for_ten_groups():
    message = small_import_warning(10, SPLIT)
    assert message is not None
    assert "10 group" in message
    assert "DEV" in message and "AUDIT_RESERVE" in message
    assert "independent evaluation" in message.lower()


def test_warning_absent_for_five_hundred_groups():
    assert small_import_warning(500, SPLIT) is None


def test_warning_boundary_uses_min_groups_inclusive():
    # 52 -> 36.4 / 7.8 / 7.8: floors 36, 7, 7; the remainder of 2 lifts DEV and AUDIT_RESERVE (largest
    # fractional parts) to 8 each.
    assert expected_counts(52, SPLIT) == {"TRAIN": 36, "DEV": 8, "AUDIT_RESERVE": 8}
    assert small_import_warning(52, SPLIT) is None
    # 51 -> 35.7 / 7.65 / 7.65: floors 35, 7, 7; the remainder of 2 goes to TRAIN (0.7) and then DEV
    # (0.65, earlier in PARTITIONS order than AUDIT_RESERVE's tie), so AUDIT_RESERVE stays at 7.
    assert expected_counts(51, SPLIT) == {"TRAIN": 36, "DEV": 8, "AUDIT_RESERVE": 7}
    message = small_import_warning(51, SPLIT)
    assert message is not None
    assert "AUDIT_RESERVE ~7" in message
    assert "DEV ~" not in message


def test_warning_names_only_short_partitions():
    split = {"TRAIN": 0.5, "DEV": 0.4, "AUDIT_RESERVE": 0.1}
    message = small_import_warning(40, split)  # DEV ~16, AUDIT_RESERVE ~4
    assert message is not None
    assert "AUDIT_RESERVE ~4" in message
    assert "DEV ~" not in message


def test_warning_respects_min_groups_override():
    assert small_import_warning(10, SPLIT, min_groups=1) is None
    assert small_import_warning(500, SPLIT, min_groups=100) is not None
    assert small_import_warning(0, SPLIT, min_groups=0) is None


def test_warning_for_zero_groups():
    message = small_import_warning(0, SPLIT)
    assert message is not None
    assert "0 group" in message


def test_warning_when_split_starves_a_held_out_partition():
    split = {"TRAIN": 1.0, "DEV": 0.0, "AUDIT_RESERVE": 0.0}
    message = small_import_warning(10_000, split)
    assert message is not None
    assert "DEV ~0" in message and "AUDIT_RESERVE ~0" in message


@pytest.mark.parametrize("bad", [-1, 1.5, "8", None, True])
def test_warning_rejects_bad_min_groups(bad):
    with pytest.raises(ValueError):
        small_import_warning(10, SPLIT, min_groups=bad)
