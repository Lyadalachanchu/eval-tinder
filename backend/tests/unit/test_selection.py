"""Unit tests for pool building and batch selection."""
from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest

from eval_tinder.db.enums import SelectionCategory
from eval_tinder.domain.disagreement import NOT_ESTIMABLE
from eval_tinder.domain.selection import (
    CATEGORIES,
    CATEGORY_COVERAGE,
    CATEGORY_DISAGREEMENT,
    CATEGORY_RANDOM,
    DEFAULT_QUOTAS,
    REASON_CANDIDATES_EXHAUSTED,
    REASON_NO_DISAGREEMENT_SCORES,
    REASON_QUOTA_FILLED,
    STRATEGY_VERSION,
    UNSTRATIFIED,
    BatchSelection,
    PoolCase,
    SelectedCase,
    build_pool,
    select_batch,
    stratum_key,
)

TASKS = ("cancellation", "refund", "billing")
RESULTS = ("accepted", "completed", "failed")


def case(
    trace: str, group: str | None = None, strata: dict[str, str] | None = None, length: int = 100
) -> PoolCase:
    return PoolCase(
        trace_id=trace, group_id=group or f"g-{trace}", strata=strata or {}, reading_length=length
    )


def make_pool(n: int = 30) -> list[PoolCase]:
    """``n`` distinct-group cases cycling through 9 strata with varied reading lengths."""
    cases = []
    for i in range(n):
        strata = {"task_type": TASKS[i % 3], "tool_result": RESULTS[(i // 3) % 3], "language": "en"}
        cases.append(case(f"t{i:03d}", strata=strata, length=50 + (i * 37) % 200))
    return cases


def ids(selected: list[SelectedCase]) -> list[str]:
    return [s.trace_id for s in selected]


def picks(result: BatchSelection, category: str) -> list[SelectedCase]:
    return [s for s in result.selected if s.category == category]


def assert_well_formed(result: BatchSelection, pool: list[PoolCase]) -> None:
    """Invariants every selection must satisfy."""
    traces = ids(result.selected)
    assert len(traces) == len(set(traces)), "trace selected twice"
    groups = [s.group_id for s in result.selected]
    assert len(groups) == len(set(groups)), "group selected twice"
    assert not set(traces) & set(result.context_repair)
    by_trace = {c.trace_id: c for c in pool}
    for s in result.selected:
        assert s.category in CATEGORIES
        assert by_trace[s.trace_id].group_id == s.group_id
        assert s.reason["committee_votes_hidden_until_judged"] is True
    for category in CATEGORIES:
        ranks = [s.rank for s in result.selected if s.category == category]
        assert ranks == list(range(1, len(ranks) + 1)), f"{category} ranks must be 1..n"
    assert result.strategy_version == STRATEGY_VERSION
    json.dumps(result.log)  # the log must be JSON serialisable
    json.dumps([s.reason for s in result.selected])


class TestStratumKey:
    def test_empty_is_unstratified(self):
        assert stratum_key({}) == UNSTRATIFIED == "_unstratified"

    def test_sorted_and_formatted(self):
        key = stratum_key({"task_type": "cancellation", "tool_result": "accepted", "language": "en"})
        assert key == "language=en|task_type=cancellation|tool_result=accepted"

    def test_insertion_order_does_not_matter(self):
        assert stratum_key({"a": "1", "b": "2"}) == stratum_key({"b": "2", "a": "1"})

    @pytest.mark.parametrize("bad", [{"a": 1}, {1: "a"}, {"a": None}, ["a=b"], "a=b"])
    def test_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            stratum_key(bad)


class TestPoolCase:
    def test_stratum_property(self):
        assert case("t", strata={"language": "en"}).stratum == "language=en"
        assert case("t").stratum == UNSTRATIFIED

    def test_frozen(self):
        c = case("t")
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.trace_id = "u"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"trace_id": ""},
            {"group_id": ""},
            {"trace_id": 3},
            {"reading_length": -1},
            {"reading_length": True},
            {"reading_length": 1.5},
            {"strata": {"task_type": 3}},
        ],
    )
    def test_invalid_raises(self, kwargs):
        base = {"trace_id": "t", "group_id": "g", "strata": {}, "reading_length": 10}
        with pytest.raises(ValueError):
            PoolCase(**{**base, **kwargs})


class TestBuildPool:
    def test_caps_size(self):
        pool = build_pool(make_pool(50), size=10, seed=1)
        assert len(pool) == 10

    def test_size_zero_is_empty(self):
        assert build_pool(make_pool(5), size=0, seed=1) == []

    def test_size_larger_than_input_returns_everything_once(self):
        cases = make_pool(12)
        pool = build_pool(cases, size=200, seed=3)
        assert sorted(c.trace_id for c in pool) == sorted(c.trace_id for c in cases)

    def test_one_case_per_group(self):
        cases = [case("a1", group="g"), case("a2", group="g"), case("a3", group="g"), case("b1", group="h")]
        pool = build_pool(cases, size=10, seed=0)
        assert sorted(c.group_id for c in pool) == ["g", "h"]
        assert len(pool) == 2

    def test_one_case_per_group_across_strata(self):
        cases = [
            case("a1", group="g", strata={"task_type": "refund"}),
            case("a2", group="g", strata={"task_type": "billing"}),
        ]
        assert len(build_pool(cases, size=10, seed=0)) == 1

    def test_duplicate_trace_ids_kept_once(self):
        cases = [case("dup", group="g1"), case("dup", group="g2"), case("other")]
        pool = build_pool(cases, size=10, seed=0)
        assert sorted(c.trace_id for c in pool) == ["dup", "other"]

    def test_deterministic_for_seed(self):
        cases = make_pool(60)
        assert build_pool(cases, size=20, seed=42) == build_pool(cases, size=20, seed=42)

    def test_seed_changes_draw(self):
        cases = make_pool(60)
        a = build_pool(cases, size=20, seed=1)
        b = build_pool(cases, size=20, seed=2)
        assert [c.trace_id for c in a] != [c.trace_id for c in b]

    def test_round_robin_across_strata(self):
        a = [case(f"a{i}", strata={"task_type": "a"}) for i in range(10)]
        b = [case(f"b{i}", strata={"task_type": "b"}) for i in range(2)]
        pool = build_pool(a + b, size=4, seed=5)
        assert Counter(c.stratum for c in pool) == {"task_type=a": 2, "task_type=b": 2}
        # Strata alternate in sorted key order.
        assert [c.stratum for c in pool] == ["task_type=a", "task_type=b", "task_type=a", "task_type=b"]

    def test_exhausted_stratum_yields_to_others(self):
        a = [case(f"a{i}", strata={"task_type": "a"}) for i in range(10)]
        b = [case(f"b{i}", strata={"task_type": "b"}) for i in range(2)]
        pool = build_pool(a + b, size=6, seed=5)
        assert Counter(c.stratum for c in pool) == {"task_type=a": 4, "task_type=b": 2}

    def test_unstratified_cases_form_their_own_stratum(self):
        plain = [case(f"p{i}") for i in range(5)]
        strat = [case(f"s{i}", strata={"language": "en"}) for i in range(5)]
        pool = build_pool(plain + strat, size=4, seed=9)
        assert Counter(c.stratum for c in pool) == {UNSTRATIFIED: 2, "language=en": 2}

    def test_within_stratum_order_is_shuffled(self):
        cases = [case(f"c{i:02d}") for i in range(30)]
        pool = build_pool(cases, size=30, seed=11)
        assert [c.trace_id for c in pool] != [c.trace_id for c in cases]
        assert sorted(c.trace_id for c in pool) == sorted(c.trace_id for c in cases)

    def test_group_collision_skips_to_next_after_shuffle(self):
        # Two strata share a group; whichever stratum wins the group, the other keeps drawing.
        a = [case("a0", group="shared", strata={"t": "a"}), case("a1", group="ga1", strata={"t": "a"})]
        b = [case("b0", group="shared", strata={"t": "b"}), case("b1", group="gb1", strata={"t": "b"})]
        pool = build_pool(a + b, size=10, seed=0)
        assert len(pool) == 3
        assert len({c.group_id for c in pool}) == 3

    @pytest.mark.parametrize("bad_size", [-1, True, 1.0, "3"])
    def test_invalid_size_raises(self, bad_size):
        with pytest.raises(ValueError):
            build_pool(make_pool(3), size=bad_size, seed=0)

    @pytest.mark.parametrize("bad_seed", [True, 1.5, "0", None])
    def test_invalid_seed_raises(self, bad_seed):
        with pytest.raises(ValueError):
            build_pool(make_pool(3), size=3, seed=bad_seed)

    def test_non_pool_case_raises(self):
        with pytest.raises(ValueError):
            build_pool([case("a"), "not a case"], size=3, seed=0)  # type: ignore[list-item]


def scores_for(pool: list[PoolCase], top: str, top_score: float = 0.66) -> dict[str, float | None]:
    """A disagreement mapping whose maximum is ``top``; others get smaller distinct scores or None."""
    scores: dict[str, float | None] = {}
    for i, c in enumerate(pool):
        if c.trace_id == top:
            scores[c.trace_id] = top_score
        elif i % 5 == 0:
            scores[c.trace_id] = None
        elif i % 7 == 0:
            scores[c.trace_id] = 0.0
        else:
            scores[c.trace_id] = round(0.05 + (i * 0.013) % 0.5, 4)
    return scores


class TestSelectBatchRandom:
    def test_random_picks_identical_under_different_disagreement_mappings(self):
        pool = make_pool(40)
        a = select_batch(pool, disagreement=scores_for(pool, "t003"), all_review=(),
                         labeled_strata_counts={}, seed=7)
        b = select_batch(pool, disagreement={c.trace_id: 0.5 for c in pool}, all_review=(),
                         labeled_strata_counts={}, seed=7)
        c = select_batch(pool, disagreement={}, all_review=(), labeled_strata_counts={}, seed=7)
        random_a = [(s.trace_id, s.rank) for s in picks(a, CATEGORY_RANDOM) if "draw" in s.reason]
        random_b = [(s.trace_id, s.rank) for s in picks(b, CATEGORY_RANDOM) if "draw" in s.reason]
        random_c = [(s.trace_id, s.rank) for s in picks(c, CATEGORY_RANDOM) if "draw" in s.reason]
        assert len(random_a) == DEFAULT_QUOTAS["random"]
        assert random_a == random_b == random_c
        for result in (a, b, c):
            assert_well_formed(result, pool)

    def test_random_picks_carry_no_score(self):
        pool = make_pool(20)
        result = select_batch(pool, disagreement={c.trace_id: 0.4 for c in pool}, all_review=(),
                              labeled_strata_counts={}, seed=1)
        for s in picks(result, CATEGORY_RANDOM):
            assert s.score is None
            assert "score" not in s.reason
            assert s.reason["independent_of_score"] is True

    def test_random_picks_change_with_seed(self):
        pool = make_pool(40)
        a = select_batch(pool, disagreement={}, all_review=(), labeled_strata_counts={}, seed=1)
        b = select_batch(pool, disagreement={}, all_review=(), labeled_strata_counts={}, seed=2)
        assert ids(picks(a, CATEGORY_RANDOM)) != ids(picks(b, CATEGORY_RANDOM))

    def test_random_picks_come_from_eligible_only(self):
        pool = make_pool(12)
        excluded = {"t000", "t001", "t002", "t003"}
        review = {"t004", "t005", "t006", "t007"}
        result = select_batch(pool, disagreement={}, all_review=review, labeled_strata_counts={},
                              quotas={"random": 4}, seed=3, excluded_trace_ids=excluded)
        assert set(ids(result.selected)) == {"t008", "t009", "t010", "t011"}
        assert all(s.category == CATEGORY_RANDOM for s in result.selected)


class TestSelectBatchDisagreement:
    def test_highest_disagreement_is_rank_one(self):
        pool = make_pool(30)
        result = select_batch(pool, disagreement=scores_for(pool, "t013"), all_review=(),
                              labeled_strata_counts={},
                              quotas={"disagreement": 6, "coverage": 2, "random": 0}, seed=5)
        assert_well_formed(result, pool)
        top = result.selected[0]
        assert top.trace_id == "t013"
        assert top.category == CATEGORY_DISAGREEMENT
        assert top.rank == 1
        assert top.score == pytest.approx(0.66)
        assert top.reason == {
            "score": pytest.approx(0.66),
            "stratum": "language=en|task_type=refund|tool_result=completed",
            "stratum_already_selected": False,
            "committee_votes_hidden_until_judged": True,
        }

    def test_highest_disagreement_is_rank_one_unless_randomly_drawn_first(self):
        pool = make_pool(30)
        for seed in range(12):
            result = select_batch(pool, disagreement=scores_for(pool, "t013"), all_review=(),
                                  labeled_strata_counts={}, seed=seed)
            assert_well_formed(result, pool)
            top = next(s for s in result.selected if s.trace_id == "t013")
            if top.category == CATEGORY_RANDOM:
                assert "draw" in top.reason  # a genuine uniform draw, not a fallback
            else:
                assert (top.category, top.rank) == (CATEGORY_DISAGREEMENT, 1)

    def test_scores_descend_within_distinct_strata(self):
        pool = [case(f"c{i}", strata={"task_type": f"s{i}"}) for i in range(6)]
        scores = {"c0": 0.1, "c1": 0.6, "c2": 0.3, "c3": 0.5, "c4": 0.2, "c5": 0.4}
        result = select_batch(pool, disagreement=scores, all_review=(), labeled_strata_counts={},
                              quotas={"disagreement": 4}, seed=0)
        assert ids(result.selected) == ["c1", "c3", "c5", "c2"]
        assert [s.rank for s in result.selected] == [1, 2, 3, 4]
        assert result.exhausted == {}

    def test_zero_and_none_scores_are_not_candidates(self):
        pool = [case("zero"), case("none"), case("missing"), case("pos")]
        result = select_batch(pool, disagreement={"zero": 0.0, "none": None, "pos": 0.01}, all_review=(),
                              labeled_strata_counts={}, quotas={"disagreement": 4}, seed=0)
        by_id = {s.trace_id: s for s in result.selected}
        assert by_id["pos"].category == CATEGORY_DISAGREEMENT
        for trace in ("zero", "none", "missing"):
            assert by_id[trace].category == CATEGORY_RANDOM
            assert by_id[trace].reason["fallback_for"] == "disagreement"
        assert result.exhausted == {"disagreement": 3}

    def test_distinct_strata_preferred_then_deferred_allowed(self):
        pool = [
            case("a9", strata={"t": "a"}), case("a8", strata={"t": "a"}), case("a7", strata={"t": "a"}),
            case("b5", strata={"t": "b"}), case("c4", strata={"t": "c"}),
        ]
        scores = {"a9": 0.9, "a8": 0.8, "a7": 0.7, "b5": 0.5, "c4": 0.4}
        three = select_batch(pool, disagreement=scores, all_review=(), labeled_strata_counts={},
                             quotas={"disagreement": 3}, seed=0)
        assert ids(three.selected) == ["a9", "b5", "c4"]
        assert all(s.reason["stratum_already_selected"] is False for s in three.selected)

        five = select_batch(pool, disagreement=scores, all_review=(), labeled_strata_counts={},
                            quotas={"disagreement": 5}, seed=0)
        assert ids(five.selected) == ["a9", "b5", "c4", "a8", "a7"]
        deferred_flags = [s.reason["stratum_already_selected"] for s in five.selected]
        assert deferred_flags == [False, False, False, True, True]
        entry = next(e for e in five.log if e["event"] == "disagreement")
        assert entry["deferred_allowed"] == 2
        assert entry["filled"] == 5
        assert entry["reason"] == REASON_QUOTA_FILLED

    def test_ties_broken_by_reading_length_ascending(self):
        pool = [case("long", length=900), case("short", length=10), case("mid", length=300)]
        result = select_batch(pool, disagreement={t: 0.5 for t in ("long", "short", "mid")}, all_review=(),
                              labeled_strata_counts={}, quotas={"disagreement": 3}, seed=0)
        assert ids(result.selected) == ["short", "mid", "long"]

    def test_full_ties_broken_deterministically_by_seed(self):
        pool = [case(f"c{i}") for i in range(20)]
        scores = {c.trace_id: 0.5 for c in pool}
        a = select_batch(pool, disagreement=scores, all_review=(), labeled_strata_counts={},
                         quotas={"disagreement": 5}, seed=3)
        b = select_batch(pool, disagreement=scores, all_review=(), labeled_strata_counts={},
                         quotas={"disagreement": 5}, seed=3)
        assert ids(a.selected) == ids(b.selected)
        others = {ids(select_batch(pool, disagreement=scores, all_review=(), labeled_strata_counts={},
                                   quotas={"disagreement": 5}, seed=s).selected)[0] for s in range(10)}
        assert len(others) > 1, "tie-break draw must depend on the seed"

    def test_disagreement_never_takes_a_random_pick_or_its_group(self):
        pool = [case("x1", group="g"), case("x2", group="g"), case("y")]
        result = select_batch(pool, disagreement={"x1": 0.6, "x2": 0.6, "y": 0.1}, all_review=(),
                              labeled_strata_counts={}, quotas={"disagreement": 2, "random": 1}, seed=0)
        assert_well_formed(result, pool)
        assert len(result.selected) == 2  # only two groups exist

    def test_all_agree_pool_has_no_disagreement_picks(self):
        pool = make_pool(30)
        scores = {c.trace_id: (0.0 if i % 2 else None) for i, c in enumerate(pool)}
        result = select_batch(pool, disagreement=scores, all_review=(), labeled_strata_counts={}, seed=4)
        assert_well_formed(result, pool)
        assert {s.category for s in result.selected} <= {CATEGORY_COVERAGE, CATEGORY_RANDOM}
        assert not picks(result, CATEGORY_DISAGREEMENT)
        assert result.exhausted["disagreement"] == 6
        assert len(result.selected) == 10
        assert len(picks(result, CATEGORY_COVERAGE)) == 2
        assert len(picks(result, CATEGORY_RANDOM)) == 8
        assert sum(1 for s in result.selected if s.reason.get("fallback_for") == "disagreement") == 6
        entry = next(e for e in result.log if e["event"] == "disagreement")
        assert entry["reason"] == REASON_NO_DISAGREEMENT_SCORES
        assert entry["candidates"] == 0
        fallback = next(e for e in result.log if e["event"] == "fallback")
        assert fallback["exhausted"] == {"disagreement": 6}
        assert "no committee scores" in fallback["note"]

    def test_empty_disagreement_mapping_is_no_committee(self):
        pool = make_pool(30)
        result = select_batch(pool, disagreement={}, all_review=(), labeled_strata_counts={}, seed=4)
        assert not picks(result, CATEGORY_DISAGREEMENT)
        assert result.exhausted == {"disagreement": 6}
        assert len(result.selected) == 10

    def test_ranks_continue_across_random_and_fallback(self):
        pool = make_pool(30)
        result = select_batch(pool, disagreement={}, all_review=(), labeled_strata_counts={}, seed=4)
        random_picks = picks(result, CATEGORY_RANDOM)
        assert [s.rank for s in random_picks] == list(range(1, 9))
        assert ["draw" in s.reason for s in random_picks] == [True, True] + [False] * 6


class TestSelectBatchCoverage:
    def test_least_labeled_stratum_first_and_missing_counts_are_zero(self):
        pool = [
            case("a", strata={"t": "a"}), case("b", strata={"t": "b"}), case("c", strata={"t": "c"}),
        ]
        result = select_batch(pool, disagreement={}, all_review=(),
                              labeled_strata_counts={"t=a": 5, "t=b": 1}, quotas={"coverage": 1}, seed=0)
        assert ids(result.selected) == ["c"]
        assert result.selected[0].category == CATEGORY_COVERAGE
        assert result.selected[0].reason == {
            "stratum": "t=c", "labeled_count": 0, "committee_votes_hidden_until_judged": True,
        }

    def test_coverage_spreads_across_equally_underlabeled_strata(self):
        pool = [case(f"a{i}", strata={"t": "a"}, length=10) for i in range(3)] + [
            case("b0", strata={"t": "b"}, length=500)
        ]
        result = select_batch(pool, disagreement={}, all_review=(),
                              labeled_strata_counts={"t=a": 0, "t=b": 0}, quotas={"coverage": 2}, seed=0)
        assert sorted(s.reason["stratum"] for s in result.selected) == ["t=a", "t=b"]
        # The first pick is the shortest case of the tied strata; the second goes to the other stratum.
        assert ids(result.selected)[0].startswith("a")
        assert ids(result.selected)[1] == "b0"

    def test_coverage_keeps_taking_from_the_least_labeled_stratum(self):
        pool = [case(f"a{i}", strata={"t": "a"}) for i in range(3)] + [case("b0", strata={"t": "b"})]
        result = select_batch(pool, disagreement={}, all_review=(), labeled_strata_counts={"t=b": 10},
                              quotas={"coverage": 2}, seed=0)
        assert all(s.reason["stratum"] == "t=a" for s in result.selected)
        assert [s.reason["labeled_count"] for s in result.selected] == [0, 1]

    def test_coverage_skips_already_selected_cases_and_groups(self):
        pool = [case("x1", group="g", strata={"t": "a"}), case("x2", group="g", strata={"t": "a"}),
                case("y", strata={"t": "b"})]
        result = select_batch(pool, disagreement={"x1": 0.5}, all_review=(),
                              labeled_strata_counts={"t=a": 0, "t=b": 3},
                              quotas={"disagreement": 1, "coverage": 1}, seed=0)
        assert ids(result.selected) == ["x1", "y"]
        assert [s.category for s in result.selected] == [CATEGORY_DISAGREEMENT, CATEGORY_COVERAGE]

    def test_coverage_tie_breaks_by_reading_length(self):
        pool = [case("long", strata={"t": "a"}, length=800), case("short", strata={"t": "a"}, length=20)]
        result = select_batch(pool, disagreement={}, all_review=(), labeled_strata_counts={},
                              quotas={"coverage": 1}, seed=0)
        assert ids(result.selected) == ["short"]

    def test_coverage_exhaustion_recorded(self):
        pool = [case("only")]
        result = select_batch(pool, disagreement={}, all_review=(), labeled_strata_counts={},
                              quotas={"coverage": 3}, seed=0)
        assert ids(result.selected) == ["only"]
        assert result.exhausted == {"coverage": 2}
        entry = next(e for e in result.log if e["event"] == "coverage")
        assert entry["reason"] == REASON_CANDIDATES_EXHAUSTED


class TestSelectBatchRouting:
    def test_all_review_goes_to_context_repair_and_is_never_selected(self):
        pool = make_pool(20)
        review = {"t001", "t005", "t017"}
        scores = {c.trace_id: 0.5 for c in pool}
        scores["t005"] = 0.99  # the highest score is all-REVIEW: it must still be routed away
        result = select_batch(pool, disagreement=scores, all_review=review, labeled_strata_counts={},
                              quotas={"disagreement": 20, "coverage": 0, "random": 0}, seed=1)
        assert result.context_repair == ["t001", "t005", "t017"]  # pool order
        assert not set(ids(result.selected)) & review
        assert len(result.selected) == 17
        assert result.log[0]["context_repair"] == 3
        assert result.log[0]["eligible"] == 17

    def test_all_review_ids_outside_the_pool_are_ignored(self):
        pool = make_pool(5)
        result = select_batch(pool, disagreement={}, all_review={"ghost"}, labeled_strata_counts={}, seed=1)
        assert result.context_repair == []

    def test_excluded_never_selected_nor_routed(self):
        pool = make_pool(20)
        excluded = {"t000", "t003", "t019"}
        scores = {c.trace_id: 0.5 for c in pool}
        scores["t003"] = 0.99
        result = select_batch(pool, disagreement=scores, all_review={"t003", "t019"},
                              labeled_strata_counts={},
                              quotas={"disagreement": 20, "coverage": 2, "random": 2}, seed=1,
                              excluded_trace_ids=excluded)
        assert not set(ids(result.selected)) & excluded
        assert result.context_repair == []
        assert len(result.selected) == 17
        assert result.log[0]["excluded_in_pool"] == 3

    def test_pool_smaller_than_total_quota_selects_everything_without_error(self):
        pool = make_pool(4)
        result = select_batch(pool, disagreement={"t000": 0.5}, all_review=(), labeled_strata_counts={},
                              seed=2)
        assert_well_formed(result, pool)
        assert sorted(ids(result.selected)) == ["t000", "t001", "t002", "t003"]
        # Every case was taken by its own category before the fallback ran, so the six missing
        # slots are pure shortfall: nothing was left for the random fallback to fill.
        assert sum(result.exhausted.values()) == 6
        assert not any("fallback_for" in s.reason for s in result.selected)
        done = result.log[-1]
        assert done["event"] == "done"
        assert done["unfilled"] == 6
        assert done["selected"] == 4

    def test_empty_pool(self):
        result = select_batch([], disagreement={}, all_review=(), labeled_strata_counts={}, seed=0)
        assert result.selected == []
        assert result.context_repair == []
        assert result.exhausted == {"disagreement": 6, "coverage": 2, "random": 2}
        assert result.log[0]["eligible_share"] == NOT_ESTIMABLE
        disagreement_entry = next(e for e in result.log if e["event"] == "disagreement")
        assert disagreement_entry["scored_share"] == NOT_ESTIMABLE
        assert result.log[-1]["unfilled"] == 10

    def test_all_zero_quotas_select_nothing(self):
        pool = make_pool(10)
        result = select_batch(pool, disagreement={"t000": 0.5}, all_review=(), labeled_strata_counts={},
                              quotas={"disagreement": 0, "coverage": 0, "random": 0}, seed=0)
        assert result.selected == []
        assert result.exhausted == {}
        assert result.log[-1]["fill_share"] == NOT_ESTIMABLE

    def test_missing_quota_keys_mean_zero(self):
        pool = make_pool(10)
        result = select_batch(pool, disagreement={"t000": 0.5}, all_review=(), labeled_strata_counts={},
                              quotas={"disagreement": 1}, seed=0)
        assert ids(result.selected) == ["t000"]
        assert result.log[0]["quotas"] == {"disagreement": 1, "coverage": 0, "random": 0}


class TestSelectBatchShape:
    def test_selected_order_and_categories(self):
        pool = make_pool(30)
        result = select_batch(pool, disagreement=scores_for(pool, "t004"), all_review=(),
                              labeled_strata_counts={"language=en|task_type=refund|tool_result=failed": 9},
                              seed=8)
        assert_well_formed(result, pool)
        categories = [s.category for s in result.selected]
        assert categories == [CATEGORY_DISAGREEMENT] * 6 + [CATEGORY_COVERAGE] * 2 + [CATEGORY_RANDOM] * 2
        assert result.by_category() == {
            CATEGORY_DISAGREEMENT: picks(result, CATEGORY_DISAGREEMENT),
            CATEGORY_COVERAGE: picks(result, CATEGORY_COVERAGE),
            CATEGORY_RANDOM: picks(result, CATEGORY_RANDOM),
        }

    def test_log_records_seed_quotas_pool_size_and_counts(self):
        pool = make_pool(25)
        result = select_batch(pool, disagreement=scores_for(pool, "t002"), all_review={"t009"},
                              labeled_strata_counts={}, seed=99, excluded_trace_ids={"t010"})
        start, done = result.log[0], result.log[-1]
        assert start["event"] == "start"
        assert start["seed"] == 99
        assert start["strategy_version"] == STRATEGY_VERSION
        assert start["quotas"] == dict(DEFAULT_QUOTAS)
        assert start["pool_size"] == 25
        assert start["eligible"] == 23
        assert done["counts"] == {c: len(picks(result, c)) for c in CATEGORIES}
        assert done["selected"] == 10
        events = [e["event"] for e in result.log]
        assert events[:4] == ["start", "random", "disagreement", "coverage"]
        assert events[-1] == "done"
        assert result.seed == 99
        assert result.strategy_version == STRATEGY_VERSION

    def test_result_is_deterministic(self):
        pool = make_pool(40)
        kwargs = dict(disagreement=scores_for(pool, "t021"), all_review={"t002"},
                      labeled_strata_counts={"language=en|task_type=billing|tool_result=accepted": 4},
                      seed=13, excluded_trace_ids={"t030"})
        assert select_batch(pool, **kwargs) == select_batch(pool, **kwargs)

    def test_disagreement_scores_for_unknown_traces_are_ignored(self):
        pool = make_pool(5)
        result = select_batch(pool, disagreement={"ghost": 0.9, "t001": 0.2}, all_review=(),
                              labeled_strata_counts={}, quotas={"disagreement": 1}, seed=0)
        assert ids(result.selected) == ["t001"]

    def test_categories_mirror_selection_category_enum(self):
        assert set(CATEGORIES) <= {c.value for c in SelectionCategory}
        assert CATEGORY_DISAGREEMENT == SelectionCategory.DISAGREEMENT
        assert CATEGORY_COVERAGE == SelectionCategory.COVERAGE
        assert CATEGORY_RANDOM == SelectionCategory.RANDOM

    def test_default_quotas_are_read_only(self):
        assert dict(DEFAULT_QUOTAS) == {"disagreement": 6, "coverage": 2, "random": 2}
        with pytest.raises(TypeError):
            DEFAULT_QUOTAS["random"] = 5  # type: ignore[index]

    def test_inputs_are_not_mutated(self):
        pool = make_pool(10)
        scores = {"t000": 0.5}
        counts = {"x": 1}
        quotas = {"disagreement": 1, "coverage": 1, "random": 1}
        select_batch(pool, disagreement=scores, all_review=(), labeled_strata_counts=counts, quotas=quotas,
                     seed=0)
        assert scores == {"t000": 0.5}
        assert counts == {"x": 1}
        assert quotas == {"disagreement": 1, "coverage": 1, "random": 1}


class TestSelectBatchValidation:
    def base(self, **overrides):
        kwargs = dict(disagreement={}, all_review=(), labeled_strata_counts={}, seed=0)
        kwargs.update(overrides)
        return kwargs

    @pytest.mark.parametrize(
        "quotas",
        [{"audit": 1}, {"random": -1}, {"random": True}, {"random": 1.0}, ["disagreement"]],
    )
    def test_invalid_quotas_raise(self, quotas):
        with pytest.raises(ValueError):
            select_batch(make_pool(3), **self.base(quotas=quotas))

    @pytest.mark.parametrize("seed", [True, 1.5, "1", None])
    def test_invalid_seed_raises(self, seed):
        with pytest.raises(ValueError):
            select_batch(make_pool(3), **self.base(seed=seed))

    @pytest.mark.parametrize("score", [float("nan"), float("inf"), "0.5", True])
    def test_invalid_score_raises(self, score):
        with pytest.raises(ValueError):
            select_batch(make_pool(3), **self.base(disagreement={"t000": score}))

    @pytest.mark.parametrize("count", [-1, 1.5, True, "3"])
    def test_invalid_labeled_count_raises(self, count):
        with pytest.raises(ValueError):
            select_batch(make_pool(3), **self.base(labeled_strata_counts={"x": count}))

    def test_duplicate_trace_in_pool_raises(self):
        with pytest.raises(ValueError, match="duplicate trace_id"):
            select_batch([case("dup", group="a"), case("dup", group="b")], **self.base())

    def test_non_pool_case_raises(self):
        with pytest.raises(ValueError):
            select_batch([case("a"), {"trace_id": "b"}], **self.base())  # type: ignore[list-item]

    def test_int_scores_are_accepted(self):
        result = select_batch([case("a")], **self.base(disagreement={"a": 1}, quotas={"disagreement": 1}))
        assert result.selected[0].score == 1.0
