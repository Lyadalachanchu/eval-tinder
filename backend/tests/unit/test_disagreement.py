"""Unit tests for committee vote disagreement statistics."""
from __future__ import annotations

import random

import pytest

from eval_tinder.db.enums import MachineVerdict
from eval_tinder.domain.disagreement import NOT_ESTIMABLE, VALID_VOTES, gini_disagreement, is_vote, vote_summary


def test_valid_votes_match_machine_verdicts():
    assert set(VALID_VOTES) == {v.value for v in MachineVerdict}
    assert NOT_ESTIMABLE == "NOT_ESTIMABLE"


class TestGiniDisagreement:
    def test_two_pass_one_fail_is_four_ninths(self):
        assert gini_disagreement(["PASS", "PASS", "FAIL"]) == pytest.approx(4 / 9, abs=1e-12)

    def test_identical_votes_is_exactly_zero(self):
        assert gini_disagreement(["FAIL", "FAIL", "FAIL"]) == 0.0
        assert gini_disagreement(["PASS", "PASS"]) == 0.0

    def test_errors_are_ignored_not_counted_as_votes(self):
        assert gini_disagreement(["PASS", None, "FAIL"]) == pytest.approx(0.5, abs=1e-12)
        assert gini_disagreement(["PASS", None, None, "FAIL"]) == pytest.approx(0.5, abs=1e-12)

    def test_single_valid_voter_is_not_estimable(self):
        assert gini_disagreement(["PASS"]) is None
        assert gini_disagreement(["PASS", None, None]) is None

    def test_no_valid_votes_is_none_even_with_min_valid_zero(self):
        assert gini_disagreement([]) is None
        assert gini_disagreement([None, None], min_valid=0) is None

    def test_min_valid_threshold(self):
        votes = ["PASS", "FAIL", "REVIEW"]
        assert gini_disagreement(votes, min_valid=3) == pytest.approx(2 / 3, abs=1e-12)
        assert gini_disagreement(votes, min_valid=4) is None

    def test_all_review_is_zero(self):
        assert gini_disagreement(["REVIEW", "REVIEW", "REVIEW"]) == 0.0

    def test_three_way_split_is_maximum(self):
        assert gini_disagreement(["PASS", "FAIL", "REVIEW"]) == pytest.approx(2 / 3, abs=1e-12)

    @pytest.mark.parametrize("bad", ["pass", "MAYBE", "", 1, True, "CANNOT_JUDGE"])
    def test_invalid_vote_raises(self, bad):
        with pytest.raises(ValueError):
            gini_disagreement(["PASS", bad, "FAIL"])

    def test_str_enum_votes_are_accepted(self):
        votes = [MachineVerdict.PASS, MachineVerdict.PASS, MachineVerdict.FAIL]
        assert gini_disagreement(votes) == pytest.approx(4 / 9, abs=1e-12)

    def test_order_invariant_and_deterministic(self):
        rng = random.Random(7)
        votes = ["PASS"] * 3 + ["FAIL"] * 2 + ["REVIEW"] + [None] * 2
        expected = gini_disagreement(votes)
        for _ in range(20):
            shuffled = list(votes)
            rng.shuffle(shuffled)
            assert gini_disagreement(shuffled) == expected

    def test_bounded_in_unit_interval(self):
        rng = random.Random(11)
        for _ in range(200):
            n = rng.randint(2, 9)
            votes = [rng.choice([*VALID_VOTES, None]) for _ in range(n)]
            g = gini_disagreement(votes)
            if g is not None:
                assert 0.0 <= g <= 2 / 3 + 1e-12


class TestVoteSummary:
    def test_counts_and_flags(self):
        s = vote_summary(["PASS", "PASS", "FAIL", None])
        assert s["counts"] == {"PASS": 2, "FAIL": 1, "REVIEW": 0}
        assert s["valid_count"] == 3
        assert s["error_count"] == 1
        assert s["all_review"] is False
        assert s["unanimous"] is False

    def test_all_review(self):
        s = vote_summary(["REVIEW", "REVIEW", None])
        assert s["all_review"] is True
        assert s["unanimous"] is True
        assert s["counts"] == {"PASS": 0, "FAIL": 0, "REVIEW": 2}
        assert gini_disagreement(["REVIEW", "REVIEW", None]) == 0.0

    def test_unanimous_pass_is_not_all_review(self):
        s = vote_summary(["PASS", "PASS"])
        assert s["unanimous"] is True
        assert s["all_review"] is False

    def test_no_valid_votes(self):
        s = vote_summary([None, None])
        assert s["valid_count"] == 0
        assert s["error_count"] == 2
        assert s["all_review"] is False
        assert s["unanimous"] is False
        assert s["counts"] == {"PASS": 0, "FAIL": 0, "REVIEW": 0}

    def test_empty(self):
        s = vote_summary([])
        assert s == {
            "counts": {"PASS": 0, "FAIL": 0, "REVIEW": 0},
            "valid_count": 0,
            "error_count": 0,
            "all_review": False,
            "unanimous": False,
        }

    def test_invalid_vote_raises(self):
        with pytest.raises(ValueError):
            vote_summary(["PASS", "maybe"])


def test_is_vote():
    assert is_vote("PASS") and is_vote("FAIL") and is_vote("REVIEW")
    assert is_vote(MachineVerdict.REVIEW)
    assert not is_vote(None)
    assert not is_vote("pass")
    assert not is_vote("CANNOT_JUDGE")
    assert not is_vote(0)
