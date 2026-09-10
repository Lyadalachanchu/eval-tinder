"""Unit tests for shortlisting candidates and forming a diverse committee."""
from __future__ import annotations

import logging
import random

import pytest

from eval_tinder.domain.committee import (
    ACTION_ADDED,
    ACTION_SEED,
    ACTION_SKIPPED,
    ACTION_STOPPED,
    REASON_BELOW_QUALITY_FLOOR,
    REASON_CANDIDATES_EXHAUSTED,
    REASON_COMMITTEE_FULL,
    REASON_DUPLICATE_GRADER,
    REASON_DUPLICATE_MANIFEST,
    REASON_EMPTY_SHORTLIST,
    REASON_INCOMPLETE_DEV_EVALUATION,
    REASON_INSUFFICIENT_COVERAGE,
    REASON_NO_DIVERSITY,
    REASON_NO_PROBE_PREDICTIONS,
    REASON_SHORTLIST_CAP,
    REASON_UNSUPPORTED_PROGRAM,
    CandidateSummary,
    CommitteeResult,
    ShortlistResult,
    form_committee,
    prediction_distance,
    shortlist_candidates,
)
from eval_tinder.domain.disagreement import NOT_ESTIMABLE

_LETTERS = {"P": "PASS", "F": "FAIL", "R": "REVIEW", ".": None}


def preds(pattern: str) -> dict[str, str | None]:
    """``"PPF.R"`` -> ``{"c0": "PASS", "c1": "PASS", "c2": "FAIL", "c3": None, "c4": "REVIEW"}``."""
    return {f"c{i}": _LETTERS[ch] for i, ch in enumerate(pattern)}


def cand(
    grader_id: str,
    agreement: float | None = 0.9,
    *,
    manifest_hash: str | None = None,
    prompt_length: int = 100,
    dev_complete: bool = True,
    usable: bool = True,
) -> CandidateSummary:
    return CandidateSummary(
        grader_id=grader_id,
        manifest_hash=manifest_hash or f"hash-{grader_id}",
        dev_agreement=agreement,
        dev_complete=dev_complete,
        prompt_length=prompt_length,
        usable=usable,
    )


def reasons(result: ShortlistResult) -> dict[str, str]:
    return {e["grader_id"]: e["reason"] for e in result.exclusions}


def ids(result: ShortlistResult) -> list[str]:
    return [c.grader_id for c in result.shortlisted]


# ---------------------------------------------------------------------------
# shortlist_candidates
# ---------------------------------------------------------------------------


class TestShortlist:
    def test_exact_duplicate_manifests_are_excluded_keeping_first_in_order(self):
        a = cand("a", 0.90, manifest_hash="same")
        b = cand("b", 0.90, manifest_hash="same")
        c = cand("c", 0.85, manifest_hash="other")
        result = shortlist_candidates([b, a, c])
        assert ids(result) == ["a", "c"]
        assert reasons(result) == {"b": REASON_DUPLICATE_MANIFEST}
        assert result.best_agreement == pytest.approx(0.90)

    def test_duplicate_kept_is_first_in_deterministic_order_not_input_order(self):
        low = cand("low", 0.80, manifest_hash="same")
        high = cand("high", 0.90, manifest_hash="same")
        result = shortlist_candidates([low, high])
        assert ids(result) == ["high"]
        assert reasons(result) == {"low": REASON_DUPLICATE_MANIFEST}

    def test_unusable_candidate_does_not_consume_its_manifest_hash(self):
        broken = cand("broken", 0.95, manifest_hash="same", usable=False)
        fine = cand("fine", 0.90, manifest_hash="same")
        result = shortlist_candidates([broken, fine])
        assert ids(result) == ["fine"]
        assert reasons(result) == {"broken": REASON_UNSUPPORTED_PROGRAM}

    def test_weak_specialist_below_floor_is_excluded(self):
        best = cand("best", 0.90)
        near = cand("near", 0.82)
        specialist = cand("specialist", 0.70)  # 0.2 below, behaviorally different (see committee test)
        result = shortlist_candidates([specialist, near, best], quality_gap=0.10)
        assert ids(result) == ["best", "near"]
        assert reasons(result) == {"specialist": REASON_BELOW_QUALITY_FLOOR}
        assert result.best_agreement == pytest.approx(0.90)
        assert result.quality_floor == pytest.approx(0.80)

    def test_candidate_exactly_on_floor_is_kept(self):
        result = shortlist_candidates([cand("best", 0.95), cand("edge", 0.85)], quality_gap=0.10)
        assert ids(result) == ["best", "edge"]
        assert result.exclusions == []

    def test_incomplete_dev_evaluation_is_excluded(self):
        result = shortlist_candidates(
            [cand("ok", 0.9), cand("no-agreement", None), cand("partial", 0.99, dev_complete=False)]
        )
        assert ids(result) == ["ok"]
        assert reasons(result) == {
            "no-agreement": REASON_INCOMPLETE_DEV_EVALUATION,
            "partial": REASON_INCOMPLETE_DEV_EVALUATION,
        }
        # The incomplete 0.99 candidate never sets the floor.
        assert result.best_agreement == pytest.approx(0.9)

    def test_unusable_is_excluded_with_reason_before_other_checks(self):
        result = shortlist_candidates([cand("u", None, usable=False), cand("ok", 0.9)])
        assert reasons(result) == {"u": REASON_UNSUPPORTED_PROGRAM}
        assert ids(result) == ["ok"]

    def test_sorted_by_agreement_then_prompt_length_then_hash(self):
        cands = [
            cand("long", 0.90, prompt_length=300, manifest_hash="a"),
            cand("short", 0.90, prompt_length=100, manifest_hash="z"),
            cand("z-hash", 0.90, prompt_length=100, manifest_hash="zz"),
            cand("top", 0.95, prompt_length=900, manifest_hash="m"),
        ]
        result = shortlist_candidates(cands)
        assert ids(result) == ["top", "short", "z-hash", "long"]

    def test_cap_excludes_extras_with_reason(self):
        cands = [cand(f"g{i}", 0.90 - i * 0.01) for i in range(6)]
        result = shortlist_candidates(cands, max_shortlist=4, quality_gap=0.5)
        assert ids(result) == ["g0", "g1", "g2", "g3"]
        assert reasons(result) == {"g4": REASON_SHORTLIST_CAP, "g5": REASON_SHORTLIST_CAP}

    def test_every_candidate_is_shortlisted_or_excluded_exactly_once(self):
        cands = [
            cand("a", 0.9),
            cand("b", 0.9, manifest_hash="hash-a"),
            cand("c", 0.5),
            cand("d", None),
            cand("e", 0.9, usable=False),
        ]
        result = shortlist_candidates(cands)
        seen = ids(result) + [e["grader_id"] for e in result.exclusions]
        assert sorted(seen) == sorted(c.grader_id for c in cands)
        assert all(set(e) == {"grader_id", "reason"} for e in result.exclusions)

    def test_floor_is_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger="eval_tinder.domain.committee"):
            result = shortlist_candidates([cand("a", 0.9), cand("b", 0.6)], quality_gap=0.1)
        assert result.quality_floor == pytest.approx(0.8)
        messages = [r.getMessage() for r in caplog.records]
        assert any("quality_floor=0.8000" in m and "best_agreement=0.9000" in m for m in messages)

    def test_empty_input(self, caplog):
        with caplog.at_level(logging.INFO, logger="eval_tinder.domain.committee"):
            result = shortlist_candidates([])
        assert result.shortlisted == []
        assert result.exclusions == []
        assert result.quality_floor is None
        assert result.best_agreement is None
        assert any(f"quality_floor={NOT_ESTIMABLE}" in r.getMessage() for r in caplog.records)

    def test_nothing_survives_gives_none_floor(self):
        result = shortlist_candidates([cand("x", None), cand("y", 0.9, usable=False)])
        assert result.shortlisted == []
        assert result.quality_floor is None
        assert result.best_agreement is None

    def test_output_independent_of_input_order(self):
        cands = [
            cand("a", 0.90, prompt_length=120),
            cand("b", 0.90, prompt_length=100),
            cand("c", 0.88, manifest_hash="hash-a"),
            cand("d", 0.70),
            cand("e", None),
            cand("f", 0.89, usable=False),
        ] + [cand(f"g{i}", 0.85) for i in range(12)]
        baseline = shortlist_candidates(cands, max_shortlist=10)
        rng = random.Random(3)
        for _ in range(10):
            shuffled = list(cands)
            rng.shuffle(shuffled)
            assert shortlist_candidates(shuffled, max_shortlist=10) == baseline

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            shortlist_candidates([cand("a")], quality_gap=-0.1)
        with pytest.raises(ValueError):
            shortlist_candidates([cand("a")], max_shortlist=0)


# ---------------------------------------------------------------------------
# prediction_distance
# ---------------------------------------------------------------------------


class TestPredictionDistance:
    def test_fraction_differing_over_shared(self):
        d, n = prediction_distance(preds("PPPPFFFFRR"), preds("PPPPFFFFPP"))
        assert n == 10
        assert d == pytest.approx(0.2)

    def test_none_entries_are_not_shared(self):
        d, n = prediction_distance(preds("PPPPP.FFFF"), preds("PPPPPF.FFF"))
        assert n == 8
        assert d == 0.0

    def test_keys_missing_on_one_side_are_not_shared(self):
        a = preds("PPPPPP")
        b = {k: v for k, v in preds("FFFFFF").items() if k != "c5"}
        d, n = prediction_distance(a, b)
        assert n == 5
        assert d == 1.0

    def test_invalid_strings_are_not_votes(self):
        a = preds("PPPPP")
        b = dict(preds("PPPPP"), c0="oops")
        assert prediction_distance(a, b) == (None, 4)
        assert prediction_distance(a, b, min_shared=4) == (0.0, 4)

    def test_below_min_shared_is_not_estimable(self):
        assert prediction_distance(preds("PPF"), preds("FFP")) == (None, 3)
        assert prediction_distance(preds("PPF"), preds("FFP"), min_shared=3) == (1.0, 3)

    def test_zero_shared_is_never_zero_distance(self):
        assert prediction_distance({}, {}, min_shared=0) == (None, 0)
        assert prediction_distance(preds("..."), preds("PPP"), min_shared=0) == (None, 0)

    def test_symmetric(self):
        a, b = preds("PPFFRRPPFF"), preds("PFPFRPRFPF")
        assert prediction_distance(a, b) == prediction_distance(b, a)


# ---------------------------------------------------------------------------
# form_committee
# ---------------------------------------------------------------------------


def added(result: CommitteeResult) -> list[str]:
    return [e["grader_id"] for e in result.log if e["action"] == ACTION_ADDED]


class TestFormCommittee:
    def test_empty_shortlist(self):
        result = form_committee([], {})
        assert result.members == []
        assert result.diversity_claimed is False
        assert result.reason == REASON_EMPTY_SHORTLIST
        assert result.log[-1]["action"] == ACTION_STOPPED

    def test_identical_predictions_yield_single_member_and_no_diversity(self):
        cands = [cand("a", 0.9), cand("b", 0.89), cand("c", 0.88)]
        probes = {c.grader_id: preds("PPFFRPPFFR") for c in cands}
        result = form_committee(cands, probes)
        assert result.members == ["a"]
        assert result.diversity_claimed is False
        assert result.reason == REASON_NO_DIVERSITY
        stop = result.log[-1]
        assert stop["action"] == ACTION_STOPPED
        assert stop["min_distance"] == 0.0
        assert stop["grader_id"] == "b"  # the best zero-distance candidate is named

    def test_inadequate_shared_coverage_prevents_adding(self):
        seed = cand("seed", 0.9)
        sparse = cand("sparse", 0.89)
        probes = {"seed": preds("PPPPPPPPPP"), "sparse": preds("FFF.......")}
        result = form_committee([seed, sparse], probes, min_shared=5)
        assert result.members == ["seed"]
        assert result.diversity_claimed is False
        assert result.reason == REASON_INSUFFICIENT_COVERAGE
        skipped = [e for e in result.log if e["action"] == ACTION_SKIPPED]
        assert len(skipped) == 1
        assert skipped[0]["grader_id"] == "sparse"
        assert skipped[0]["reason"] == REASON_INSUFFICIENT_COVERAGE
        assert skipped[0]["min_distance"] == NOT_ESTIMABLE
        assert skipped[0]["min_shared"] == 3

    def test_coverage_is_checked_against_every_member(self):
        # "c" shares 6 cases with the seed (estimable) but none with "b", so once "b" is a member
        # its minimum distance is not estimable and it cannot join.
        probes = {
            "a": preds("PPPPPPPPPPPP"),
            "b": preds("......FFFFFF"),
            "c": preds("PFPFPF......"),
        }
        assert prediction_distance(probes["c"], probes["a"]) == (0.5, 6)
        assert prediction_distance(probes["c"], probes["b"]) == (None, 0)
        cands = [cand("a", 0.9), cand("b", 0.89), cand("c", 0.88)]
        result = form_committee(cands, probes, min_shared=5)
        assert result.members == ["a", "b"]  # b: distance 1.0 beats c: distance 0.5 in round 1
        skipped = [e for e in result.log if e["action"] == ACTION_SKIPPED]
        assert [(e["grader_id"], e["round"], e["min_shared"]) for e in skipped] == [("c", 2, 0)]
        assert result.reason == REASON_INSUFFICIENT_COVERAGE

    def test_lower_min_shared_allows_sparse_candidate(self):
        probes = {"seed": preds("PPPPPPPPPP"), "sparse": preds("FFF.......")}
        result = form_committee([cand("seed"), cand("sparse", 0.8)], probes, min_shared=3)
        assert result.members == ["seed", "sparse"]

    def test_greedy_picks_most_different_first(self):
        probes = {
            "a": preds("PPPPPPPPPP"),
            "b": preds("FFFPPPPPPP"),  # 0.3 from a, 0.5 from c
            "c": preds("FFFFFFFFPP"),  # 0.8 from a
        }
        cands = [cand("a", 0.90), cand("b", 0.89), cand("c", 0.80)]
        result = form_committee(cands, probes, size=4)
        assert result.members == ["a", "c", "b"]
        assert added(result) == ["c", "b"]
        add_c, add_b = [e for e in result.log if e["action"] == ACTION_ADDED]
        assert add_c["min_distance"] == pytest.approx(0.8)
        assert add_b["min_distance"] == pytest.approx(0.3)
        assert result.reason == REASON_CANDIDATES_EXHAUSTED
        assert result.diversity_claimed is True

    def test_stops_when_size_reached(self):
        probes = {
            "a": preds("PPPPPPPPPP"),
            "b": preds("FFFFFFFFFF"),
            "c": preds("RRRRRRRRRR"),
            "d": preds("PPPPPFFFFF"),
        }
        cands = [cand(g, 0.9 - i * 0.01) for i, g in enumerate("abcd")]
        result = form_committee(cands, probes, size=3)
        assert len(result.members) == 3
        assert result.members[0] == "a"
        assert result.reason == REASON_COMMITTEE_FULL
        assert result.diversity_claimed is True
        assert result.log[-1]["reason"] == REASON_COMMITTEE_FULL

    def test_size_one_is_seed_only(self):
        probes = {"a": preds("PPPPPPPPPP"), "b": preds("FFFFFFFFFF")}
        result = form_committee([cand("a"), cand("b", 0.8)], probes, size=1)
        assert result.members == ["a"]
        assert result.reason == REASON_COMMITTEE_FULL
        assert result.diversity_claimed is False

    def test_seed_plus_one_does_not_claim_diversity(self):
        probes = {"a": preds("PPPPPPPPPP"), "b": preds("FFFFFFFFFF")}
        result = form_committee([cand("a"), cand("b", 0.8)], probes, size=4)
        assert result.members == ["a", "b"]
        assert result.diversity_claimed is False
        assert result.reason == REASON_CANDIDATES_EXHAUSTED

    def test_diversity_not_claimed_when_second_addition_has_zero_distance(self):
        probes = {"a": preds("PPPPPPPPPP"), "b": preds("FFFFFFFFFF"), "c": preds("FFFFFFFFFF")}
        result = form_committee([cand("a"), cand("b", 0.85), cand("c", 0.85)], probes)
        assert result.members == ["a", "b"]
        assert result.diversity_claimed is False
        assert result.reason == REASON_NO_DIVERSITY

    def test_tie_broken_by_higher_dev_agreement(self):
        probes = {"a": preds("PPPPPPPPPP"), "b": preds("FFFFFPPPPP"), "c": preds("PPPPPFFFFF")}
        cands = [cand("a", 0.90), cand("b", 0.80, prompt_length=50), cand("c", 0.85, prompt_length=500)]
        result = form_committee(cands, probes)
        assert result.members[:2] == ["a", "c"]

    def test_tie_broken_by_shorter_prompt_then_hash(self):
        probes = {"a": preds("PPPPPPPPPP"), "b": preds("FFFFFPPPPP"), "c": preds("PPPPPFFFFF")}
        # Same distance to the seed and same agreement: shorter prompt wins.
        cands = [cand("a", 0.90), cand("b", 0.85, prompt_length=500), cand("c", 0.85, prompt_length=50)]
        assert form_committee(cands, probes).members[:2] == ["a", "c"]
        # Same prompt length as well: manifest hash ascending wins.
        cands = [
            cand("a", 0.90),
            cand("b", 0.85, prompt_length=50, manifest_hash="zzz"),
            cand("c", 0.85, prompt_length=50, manifest_hash="aaa"),
        ]
        assert form_committee(cands, probes).members[:2] == ["a", "c"]
        cands = [
            cand("a", 0.90),
            cand("b", 0.85, prompt_length=50, manifest_hash="aaa"),
            cand("c", 0.85, prompt_length=50, manifest_hash="zzz"),
        ]
        assert form_committee(cands, probes).members[:2] == ["a", "b"]

    def test_same_inputs_same_output(self):
        rng = random.Random(42)
        graders = [f"g{i}" for i in range(8)]
        probes = {g: preds("".join(rng.choice("PFR.") for _ in range(20))) for g in graders}
        cands = [cand(g, round(0.9 - i * 0.005, 3), prompt_length=100 + i) for i, g in enumerate(graders)]
        first = form_committee(cands, probes)
        for _ in range(5):
            # Mapping key order must not matter either.
            keys = list(probes)
            rng.shuffle(keys)
            reordered = {k: dict(reversed(list(probes[k].items()))) for k in keys}
            assert form_committee(cands, reordered) == first
        assert first.members[0] == "g0"
        assert all(m in graders for m in first.members)
        assert len(set(first.members)) == len(first.members)

    def test_seed_must_have_probe_predictions(self):
        probes = {"b": preds("PPPPPPPPPP"), "c": preds("FFFFFFFFFF"), "d": preds("..........")}
        cands = [cand("a", 0.95), cand("d", 0.92), cand("b", 0.9), cand("c", 0.8)]
        result = form_committee(cands, probes)
        assert result.members == ["b", "c"]
        skipped = [(e["grader_id"], e["reason"]) for e in result.log if e["action"] == ACTION_SKIPPED]
        assert skipped == [("a", REASON_NO_PROBE_PREDICTIONS), ("d", REASON_NO_PROBE_PREDICTIONS)]
        seed = next(e for e in result.log if e["action"] == ACTION_SEED)
        assert seed["grader_id"] == "b"

    def test_no_candidate_has_predictions(self):
        result = form_committee([cand("a"), cand("b")], {})
        assert result.members == []
        assert result.reason == REASON_NO_PROBE_PREDICTIONS
        assert result.diversity_claimed is False

    def test_duplicate_grader_ids_are_never_added_twice(self):
        probes = {"a": preds("PPPPPPPPPP"), "b": preds("FFFFFFFFFF")}
        result = form_committee([cand("a"), cand("a"), cand("b", 0.8)], probes)
        assert result.members == ["a", "b"]
        assert any(e["reason"] == REASON_DUPLICATE_GRADER for e in result.log)

    def test_weak_specialist_excluded_even_though_behaviorally_different(self):
        best = cand("best", 0.90)
        near = cand("near", 0.85)
        specialist = cand("specialist", 0.70)
        probes = {
            "best": preds("PPPPPPPPPP"),
            "near": preds("PPPPPPPPPF"),
            "specialist": preds("FFFFFFFFFF"),
        }
        shortlist = shortlist_candidates([specialist, near, best], quality_gap=0.10)
        assert reasons(shortlist) == {"specialist": REASON_BELOW_QUALITY_FLOOR}
        result = form_committee(shortlist.shortlisted, probes)
        assert "specialist" not in result.members
        assert result.members == ["best", "near"]

    def test_members_are_never_fabricated(self):
        probes = {"a": preds("PPPPPPPPPP"), "ghost": preds("FFFFFFFFFF")}
        result = form_committee([cand("a")], probes, size=4)
        assert result.members == ["a"]
        assert result.reason == REASON_CANDIDATES_EXHAUSTED

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            form_committee([cand("a")], {"a": preds("PPPPP")}, size=0)

    def test_log_entries_have_uniform_schema(self):
        probes = {"a": preds("PPPPPPPPPP"), "b": preds("FFFFFFFFFF"), "c": preds("FF........")}
        result = form_committee([cand("a"), cand("b", 0.85), cand("c", 0.85)], probes)
        for entry in result.log:
            assert set(entry) == {"round", "action", "reason", "grader_id", "min_distance", "min_shared"}
