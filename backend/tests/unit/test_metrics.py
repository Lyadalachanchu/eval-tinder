"""Unit tests for human/machine agreement metrics, baselines, and error bounds."""
from __future__ import annotations

import math

import pytest

from eval_tinder.db.enums import GradingStatus, HumanVerdict, MachineVerdict
from eval_tinder.domain.metrics import (
    BASELINE_METRIC_KEYS,
    HUMAN_VERDICTS,
    MACHINE_VERDICTS,
    NOT_ESTIMABLE,
    OK_STATUS,
    ConfusionTable,
    baselines,
    bonferroni_confidence,
    build_confusion,
    compute_metrics,
    error_upper_bound,
    ratio,
    sampling_design_supported,
)

METRIC_KEYS = (
    "agreement",
    "automatic_coverage_all",
    "automatic_coverage_determinate",
    "automatic_error_rate",
    "false_pass_rate_among_accepted",
    "failure_recall",
    "human_unresolved_rate",
    "operational_failure_rate",
)

SUPPORTED_DESIGN = {
    "unit": "group",
    "method": "uniform_random",
    "independence_assumption_documented": True,
    "fresh_groups": True,
}


def _rows(**cells: int) -> list[tuple[str, str, str]]:
    """Expand ``human_machine=count`` keyword cells into (human, machine, "OK") rows."""
    out: list[tuple[str, str, str]] = []
    for cell, count in cells.items():
        human, machine = cell.upper().split("_")
        out += [(human, machine, "OK")] * count
    return out


@pytest.fixture
def always_pass_on_19_1() -> ConfusionTable:
    """19 human PASS + 1 human FAIL; the machine says PASS on every case."""
    return build_confusion(_rows(pass_pass=19, fail_pass=1))


# --------------------------------------------------------------------------
# Constants and ratio
# --------------------------------------------------------------------------


def test_verdict_vocabularies_match_database_enums():
    assert HUMAN_VERDICTS == {v.value for v in HumanVerdict}
    assert MACHINE_VERDICTS == {v.value for v in MachineVerdict}
    assert OK_STATUS == GradingStatus.OK.value


def test_ratio_returns_sentinel_for_zero_denominator_not_zero():
    assert ratio(0, 0) == NOT_ESTIMABLE
    assert ratio(5, 0) == NOT_ESTIMABLE
    assert ratio(0, 0) != 0
    assert ratio(0, 0) is not None


def test_ratio_computes_fraction():
    assert ratio(1, 4) == 0.25
    assert ratio(0, 7) == 0.0
    assert ratio(3, 3) == 1.0


@pytest.mark.parametrize(("num", "den"), [(-1, 4), (1, -4), (-1, 0)])
def test_ratio_rejects_negative_counts(num, den):
    with pytest.raises(ValueError):
        ratio(num, den)


# --------------------------------------------------------------------------
# build_confusion
# --------------------------------------------------------------------------


def test_build_confusion_fills_every_cell():
    table = build_confusion(
        _rows(pass_pass=1, pass_fail=2, pass_review=3, fail_pass=4, fail_fail=5, fail_review=6)
    )
    assert table == ConfusionTable(
        pass_pass=1, pass_fail=2, pass_review=3, fail_pass=4, fail_fail=5, fail_review=6
    )
    assert table.human_pass == 6
    assert table.human_fail == 15
    assert table.human_determinate == 21
    assert table.total_cases == 21
    assert table.machine_binary_on_determinate == 12
    assert table.operational_failures == 0
    assert table.human_unresolved == 0


def test_build_confusion_empty_rows_gives_empty_table():
    table = build_confusion([])
    assert table == ConfusionTable()
    assert table.total_cases == 0


def test_build_confusion_is_order_independent():
    rows = _rows(pass_pass=3, fail_fail=2, pass_review=1) + [("CANNOT_JUDGE", "PASS", "OK")]
    assert build_confusion(rows) == build_confusion(list(reversed(rows)))
    assert build_confusion(iter(rows)) == build_confusion(tuple(rows))


def test_build_confusion_accepts_enum_members():
    table = build_confusion(
        [
            (HumanVerdict.PASS, MachineVerdict.PASS, GradingStatus.OK),
            (HumanVerdict.FAIL, MachineVerdict.REVIEW, GradingStatus.OK),
            (HumanVerdict.CANNOT_JUDGE, MachineVerdict.FAIL, GradingStatus.PROVIDER_ERROR),
        ]
    )
    assert table.pass_pass == 1
    assert table.fail_review == 1
    assert table.human_unresolved == 1
    assert table.operational_failures == 1
    assert table.unresolved_machine_binary == 0


@pytest.mark.parametrize(
    "row",
    [
        ("MAYBE", "PASS", "OK"),
        ("pass", "PASS", "OK"),  # case-sensitive: the vocabulary is exact
        ("PASS", "UNSURE", "OK"),
        ("PASS", "CANNOT_JUDGE", "OK"),  # a human-only verdict is not a machine verdict
        ("REVIEW", "PASS", "OK"),  # a machine-only verdict is not a human verdict
        ("PASS", None, "OK"),
        (None, "PASS", "OK"),
        ("PASS", "PASS", None),
        ("PASS", "PASS"),  # malformed row
        ("PASS", "PASS", "OK", "extra"),
        "PASS",
    ],
)
def test_build_confusion_rejects_unknown_values(row):
    with pytest.raises(ValueError):
        build_confusion([("PASS", "PASS", "OK"), row])


def test_build_confusion_never_uses_machine_verdict_as_human_label():
    # A machine PASS on a human CANNOT_JUDGE case must not become a "pass_pass" agreement.
    table = build_confusion([("CANNOT_JUDGE", "PASS", "OK"), ("CANNOT_JUDGE", "FAIL", "OK")])
    assert table.pass_pass == 0 and table.fail_fail == 0
    assert table.human_determinate == 0
    assert table.human_unresolved == 2
    assert table.unresolved_machine_binary == 2
    assert compute_metrics(table)["agreement"]["value"] == NOT_ESTIMABLE


@pytest.mark.parametrize("status", ["MALFORMED_OUTPUT", "PROVIDER_ERROR", "CONTEXT_TOO_LARGE", "ok", "", "OK "])
def test_non_ok_status_is_operational_failure_counted_as_review(status):
    table = build_confusion([("PASS", "PASS", status), ("FAIL", "FAIL", status), ("FAIL", "REVIEW", status)])
    assert table.pass_pass == 0 and table.fail_fail == 0
    assert table.pass_review == 1
    assert table.fail_review == 2
    assert table.operational_failures == 3
    assert table.machine_binary_on_determinate == 0


def test_operational_failure_on_unresolved_case_is_counted_but_not_a_vote():
    table = build_confusion([("CANNOT_JUDGE", "PASS", "PROVIDER_ERROR"), ("CANNOT_JUDGE", "REVIEW", "OK")])
    assert table.human_unresolved == 2
    assert table.unresolved_machine_binary == 0
    assert table.operational_failures == 1


# --------------------------------------------------------------------------
# ConfusionTable invariants
# --------------------------------------------------------------------------


def test_confusion_table_defaults_and_as_table():
    table = ConfusionTable(pass_pass=2, fail_review=1)
    assert table.as_table() == {
        "PASS": {"PASS": 2, "FAIL": 0, "REVIEW": 0},
        "FAIL": {"PASS": 0, "FAIL": 0, "REVIEW": 1},
    }
    assert table.machine_binary_all == 2


def test_confusion_table_is_frozen():
    table = ConfusionTable(pass_pass=1)
    with pytest.raises(AttributeError):
        table.pass_pass = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pass_pass": -1},
        {"human_unresolved": -3},
        {"pass_pass": 1.0},
        {"pass_pass": True},
        {"pass_pass": "1"},
        {"unresolved_machine_binary": 1},  # exceeds human_unresolved == 0
        {"operational_failures": 1},  # nothing to be a failure on
        {"pass_pass": 5, "operational_failures": 1},  # a failure cannot land in a PASS/FAIL cell
        {"human_unresolved": 2, "unresolved_machine_binary": 2, "operational_failures": 1},
    ],
)
def test_confusion_table_rejects_invalid_counts(kwargs):
    with pytest.raises(ValueError):
        ConfusionTable(**kwargs)


def test_confusion_table_accepts_consistent_operational_failures():
    table = ConfusionTable(pass_review=2, fail_review=1, human_unresolved=3, unresolved_machine_binary=1,
                           operational_failures=5)
    assert table.operational_failures == 5


# --------------------------------------------------------------------------
# compute_metrics
# --------------------------------------------------------------------------


def test_metric_entries_have_uniform_shape(always_pass_on_19_1):
    metrics = compute_metrics(always_pass_on_19_1)
    assert set(metrics) == set(METRIC_KEYS) | {"counts"}
    for key in METRIC_KEYS:
        entry = metrics[key]
        assert set(entry) == {"value", "numerator", "denominator", "definition"}
        assert isinstance(entry["numerator"], int) and isinstance(entry["denominator"], int)
        assert entry["numerator"] <= entry["denominator"] or entry["denominator"] == 0
        assert entry["value"] == ratio(entry["numerator"], entry["denominator"])
        assert isinstance(entry["definition"], str) and entry["definition"]
    counts = metrics["counts"]
    assert set(counts) == {"human_pass", "human_fail", "human_unresolved", "operational_failures",
                           "total_cases", "table"}


def test_always_pass_grader_scores_high_agreement_but_zero_failure_recall(always_pass_on_19_1):
    metrics = compute_metrics(always_pass_on_19_1)
    assert metrics["agreement"]["value"] == pytest.approx(0.95)
    assert metrics["agreement"] == {
        "value": 0.95, "numerator": 19, "denominator": 20, "definition": metrics["agreement"]["definition"],
    }
    assert metrics["failure_recall"]["value"] == 0.0
    assert metrics["failure_recall"]["numerator"] == 0
    assert metrics["failure_recall"]["denominator"] == 1
    assert metrics["false_pass_rate_among_accepted"]["value"] == pytest.approx(0.05)
    assert metrics["automatic_error_rate"]["value"] == pytest.approx(0.05)
    assert metrics["automatic_coverage_all"]["value"] == 1.0
    assert metrics["automatic_coverage_determinate"]["value"] == 1.0
    assert metrics["human_unresolved_rate"]["value"] == 0.0
    assert metrics["operational_failure_rate"]["value"] == 0.0
    assert metrics["counts"] == {
        "human_pass": 19, "human_fail": 1, "human_unresolved": 0, "operational_failures": 0,
        "total_cases": 20,
        "table": {"PASS": {"PASS": 19, "FAIL": 0, "REVIEW": 0}, "FAIL": {"PASS": 1, "FAIL": 0, "REVIEW": 0}},
    }


def test_perfect_grader_metrics():
    metrics = compute_metrics(build_confusion(_rows(pass_pass=8, fail_fail=2)))
    assert metrics["agreement"]["value"] == 1.0
    assert metrics["failure_recall"]["value"] == 1.0
    assert metrics["false_pass_rate_among_accepted"]["value"] == 0.0
    assert metrics["automatic_error_rate"]["value"] == 0.0


def test_mixed_table_metric_arithmetic():
    table = build_confusion(
        _rows(pass_pass=10, pass_fail=2, pass_review=3, fail_pass=1, fail_fail=4, fail_review=0)
        + [("CANNOT_JUDGE", "PASS", "OK"), ("CANNOT_JUDGE", "REVIEW", "OK")]
        + [("PASS", "PASS", "PROVIDER_ERROR"), ("CANNOT_JUDGE", "FAIL", "MALFORMED_OUTPUT")]
    )
    m = compute_metrics(table)
    # determinate = 10+2+3+1+4+1(op failure -> pass_review) = 21; unresolved = 3; total = 24
    assert m["counts"]["total_cases"] == 24
    assert m["agreement"]["numerator"] == 14 and m["agreement"]["denominator"] == 21
    assert m["automatic_coverage_determinate"]["numerator"] == 17
    assert m["automatic_coverage_determinate"]["denominator"] == 21
    assert m["automatic_coverage_all"]["numerator"] == 18  # 17 + one OK PASS on a CANNOT_JUDGE case
    assert m["automatic_coverage_all"]["denominator"] == 24
    assert m["automatic_error_rate"]["numerator"] == 3 and m["automatic_error_rate"]["denominator"] == 17
    assert m["false_pass_rate_among_accepted"]["numerator"] == 1
    assert m["false_pass_rate_among_accepted"]["denominator"] == 11
    assert m["failure_recall"]["numerator"] == 4 and m["failure_recall"]["denominator"] == 5
    assert m["human_unresolved_rate"]["numerator"] == 3 and m["human_unresolved_rate"]["denominator"] == 24
    assert m["operational_failure_rate"]["numerator"] == 2
    assert m["operational_failure_rate"]["denominator"] == 24


def test_every_metric_is_not_estimable_on_an_empty_table():
    metrics = compute_metrics(ConfusionTable())
    for key in METRIC_KEYS:
        assert metrics[key]["value"] == NOT_ESTIMABLE, key
        assert metrics[key]["denominator"] == 0
    assert metrics["counts"]["total_cases"] == 0


def test_no_human_fail_makes_failure_recall_not_estimable():
    metrics = compute_metrics(build_confusion(_rows(pass_pass=10)))
    assert metrics["failure_recall"]["value"] == NOT_ESTIMABLE
    assert metrics["failure_recall"]["denominator"] == 0
    assert metrics["agreement"]["value"] == 1.0


def test_no_machine_pass_makes_false_pass_rate_not_estimable():
    metrics = compute_metrics(build_confusion(_rows(pass_fail=3, fail_fail=2)))
    assert metrics["false_pass_rate_among_accepted"]["value"] == NOT_ESTIMABLE
    assert metrics["failure_recall"]["value"] == 1.0


def test_all_review_makes_error_rate_not_estimable_but_coverage_zero():
    metrics = compute_metrics(build_confusion(_rows(pass_review=4, fail_review=1)))
    assert metrics["automatic_error_rate"]["value"] == NOT_ESTIMABLE
    assert metrics["automatic_coverage_all"]["value"] == 0.0
    assert metrics["automatic_coverage_determinate"]["value"] == 0.0
    assert metrics["agreement"]["value"] == 0.0
    assert metrics["failure_recall"]["value"] == 0.0


def test_only_unresolved_cases_leave_labelled_metrics_not_estimable():
    metrics = compute_metrics(build_confusion([("CANNOT_JUDGE", "PASS", "OK")] * 3))
    for key in ("agreement", "automatic_coverage_determinate", "automatic_error_rate",
                "false_pass_rate_among_accepted", "failure_recall"):
        assert metrics[key]["value"] == NOT_ESTIMABLE, key
    assert metrics["human_unresolved_rate"]["value"] == 1.0
    assert metrics["automatic_coverage_all"]["value"] == 1.0
    assert metrics["operational_failure_rate"]["value"] == 0.0


def test_operational_failures_reduce_coverage_but_are_not_agreement_errors():
    clean = build_confusion(_rows(pass_pass=8, fail_fail=2))
    with_failures = build_confusion(
        _rows(pass_pass=8, fail_fail=2) + [("PASS", "PASS", "PROVIDER_ERROR"), ("FAIL", "FAIL", "MALFORMED_OUTPUT")]
    )
    clean_m = compute_metrics(clean)
    failed_m = compute_metrics(with_failures)

    # Coverage drops: the failed calls are in the denominator but cast no vote.
    assert clean_m["automatic_coverage_all"]["value"] == 1.0
    assert failed_m["automatic_coverage_all"]["numerator"] == 10
    assert failed_m["automatic_coverage_all"]["denominator"] == 12
    assert failed_m["automatic_coverage_determinate"]["value"] == pytest.approx(10 / 12)
    assert failed_m["operational_failure_rate"]["value"] == pytest.approx(2 / 12)

    # They are not errors: the error numerator and the binary-decision denominator are untouched.
    assert failed_m["automatic_error_rate"]["numerator"] == 0
    assert failed_m["automatic_error_rate"]["denominator"] == clean_m["automatic_error_rate"]["denominator"] == 10
    assert failed_m["automatic_error_rate"]["value"] == 0.0
    assert failed_m["false_pass_rate_among_accepted"]["value"] == 0.0
    assert with_failures.pass_fail == 0 and with_failures.fail_pass == 0

    # They are reported separately from genuine REVIEW votes, though tallied as REVIEW in the table.
    assert failed_m["counts"]["operational_failures"] == 2
    assert failed_m["counts"]["table"]["PASS"]["REVIEW"] == 1
    assert failed_m["counts"]["table"]["FAIL"]["REVIEW"] == 1
    # Agreement counts them as non-agreement (REVIEW), per its definition.
    assert failed_m["agreement"]["numerator"] == 10 and failed_m["agreement"]["denominator"] == 12


def test_unresolved_human_cases_remain_in_counts_but_not_in_labelled_denominators():
    table = build_confusion(
        _rows(pass_pass=6, fail_fail=2)
        + [("CANNOT_JUDGE", "PASS", "OK"), ("CANNOT_JUDGE", "REVIEW", "OK"), ("CANNOT_JUDGE", "FAIL", "OK")]
    )
    metrics = compute_metrics(table)
    assert metrics["counts"]["human_unresolved"] == 3
    assert metrics["counts"]["total_cases"] == 11
    assert metrics["human_unresolved_rate"]["value"] == pytest.approx(3 / 11)
    assert metrics["agreement"]["denominator"] == 8
    assert metrics["agreement"]["value"] == 1.0
    assert metrics["automatic_coverage_determinate"]["value"] == 1.0
    assert metrics["automatic_coverage_all"]["numerator"] == 10
    assert metrics["automatic_coverage_all"]["denominator"] == 11
    # The machine's PASS/FAIL on unresolved cases never becomes agreement or error.
    assert metrics["automatic_error_rate"]["denominator"] == 8
    assert metrics["failure_recall"]["denominator"] == 2


def test_compute_metrics_rejects_non_table():
    with pytest.raises(ValueError):
        compute_metrics({"pass_pass": 1})  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def test_baselines_on_95_percent_pass_dataset(always_pass_on_19_1):
    result = baselines(always_pass_on_19_1)
    assert set(result) == {"always_pass", "always_fail"}
    for name in result:
        assert set(result[name]) == set(BASELINE_METRIC_KEYS) | {"definition"}
        for key in BASELINE_METRIC_KEYS:
            assert set(result[name][key]) == {"value", "numerator", "denominator", "definition"}

    always_pass = result["always_pass"]
    assert always_pass["agreement"]["value"] == pytest.approx(0.95)
    assert always_pass["failure_recall"]["value"] == 0.0
    assert always_pass["false_pass_rate_among_accepted"]["value"] == pytest.approx(0.05)

    always_fail = result["always_fail"]
    assert always_fail["agreement"]["value"] == pytest.approx(0.05)
    assert always_fail["failure_recall"]["value"] == 1.0
    assert always_fail["false_pass_rate_among_accepted"]["value"] == NOT_ESTIMABLE  # it accepts nothing


def test_baselines_ignore_the_real_machine_verdicts():
    # Same human labels, very different machine behaviour => identical baselines.
    a = build_confusion(_rows(pass_pass=5, fail_fail=5))
    b = build_confusion(_rows(pass_review=3, pass_fail=2, fail_pass=5))
    c = build_confusion(_rows(pass_pass=4, fail_fail=3) + [("PASS", "PASS", "PROVIDER_ERROR")] * 1
                        + [("FAIL", "FAIL", "MALFORMED_OUTPUT")] * 2)
    assert baselines(a) == baselines(b) == baselines(c)
    assert baselines(a)["always_pass"]["agreement"]["value"] == 0.5
    assert baselines(a)["always_fail"]["agreement"]["value"] == 0.5


def test_baselines_with_no_human_fail():
    result = baselines(build_confusion(_rows(pass_pass=4)))
    assert result["always_pass"]["agreement"]["value"] == 1.0
    assert result["always_pass"]["failure_recall"]["value"] == NOT_ESTIMABLE
    assert result["always_pass"]["false_pass_rate_among_accepted"]["value"] == 0.0
    assert result["always_fail"]["agreement"]["value"] == 0.0
    assert result["always_fail"]["failure_recall"]["value"] == NOT_ESTIMABLE


def test_baselines_on_empty_and_unresolved_only_tables():
    for table in (ConfusionTable(), build_confusion([("CANNOT_JUDGE", "PASS", "OK")])):
        result = baselines(table)
        for name in ("always_pass", "always_fail"):
            for key in BASELINE_METRIC_KEYS:
                assert result[name][key]["value"] == NOT_ESTIMABLE


def test_baselines_rejects_non_table():
    with pytest.raises(ValueError):
        baselines(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# error_upper_bound
# --------------------------------------------------------------------------


def test_error_upper_bound_reference_values():
    assert error_upper_bound(0, 20) == pytest.approx(0.1391, abs=1e-3)
    assert error_upper_bound(0, 100) == pytest.approx(0.0295, abs=1e-3)
    assert error_upper_bound(0, 20, confidence=0.95) == error_upper_bound(0, 20)


@pytest.mark.parametrize("n", [5, 20, 100])
def test_error_upper_bound_zero_errors_matches_closed_form(n):
    assert error_upper_bound(0, n) == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-9)


@pytest.mark.parametrize("n", [1, 5, 20, 100])
def test_error_upper_bound_all_errors_is_one(n):
    assert error_upper_bound(n, n) == pytest.approx(1.0, abs=1e-12)


def test_error_upper_bound_returns_none_for_no_observations():
    assert error_upper_bound(0, 0) is None
    assert error_upper_bound(0, 0, confidence=0.5) is None


def test_error_upper_bound_returns_float_in_unit_interval():
    value = error_upper_bound(3, 50)
    assert isinstance(value, float)
    assert 3 / 50 < value < 1.0


def test_error_upper_bound_is_monotone_in_errors_and_confidence():
    bounds = [error_upper_bound(k, 30) for k in range(0, 31)]
    assert bounds == sorted(bounds)
    assert error_upper_bound(2, 40, confidence=0.99) > error_upper_bound(2, 40, confidence=0.95)
    assert error_upper_bound(2, 40, confidence=0.95) > error_upper_bound(2, 40, confidence=0.80)


def test_error_upper_bound_shrinks_with_sample_size():
    assert error_upper_bound(0, 200) < error_upper_bound(0, 100) < error_upper_bound(0, 20)


@pytest.mark.parametrize(
    ("k", "n", "confidence"),
    [
        (0, -1, 0.95),
        (-1, 10, 0.95),
        (11, 10, 0.95),
        (0, 10, 0.0),
        (0, 10, 1.0),
        (0, 10, 1.5),
        (0, 10, -0.5),
        (0, 10, math.nan),
        (0, 0, 0.0),  # confidence is validated even when n == 0
        (0.0, 10, 0.95),
        (0, 10.0, 0.95),
        (True, 10, 0.95),
        ("0", 10, 0.95),
    ],
)
def test_error_upper_bound_rejects_invalid_inputs(k, n, confidence):
    with pytest.raises(ValueError):
        error_upper_bound(k, n, confidence)


# --------------------------------------------------------------------------
# bonferroni_confidence
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("joint", "n_bounds", "expected"),
    [(0.95, 1, 0.95), (0.95, 2, 0.975), (0.9, 5, 0.98), (0.99, 4, 0.9975), (0.5, 10, 0.95)],
)
def test_bonferroni_confidence_values(joint, n_bounds, expected):
    assert bonferroni_confidence(joint, n_bounds) == pytest.approx(expected, abs=1e-12)


def test_bonferroni_confidence_never_below_joint_and_below_one():
    for n_bounds in (1, 2, 3, 10, 1000):
        per_bound = bonferroni_confidence(0.95, n_bounds)
        assert 0.95 <= per_bound < 1.0


def test_bonferroni_composes_with_error_upper_bound():
    per_bound = bonferroni_confidence(0.95, 2)
    assert error_upper_bound(0, 20, per_bound) == pytest.approx(1 - 0.025 ** (1 / 20), abs=1e-9)


@pytest.mark.parametrize(
    ("joint", "n_bounds"),
    [(0.0, 1), (1.0, 1), (1.2, 1), (-0.1, 1), (math.nan, 1), (0.95, 0), (0.95, -1), (0.95, 1.5),
     (0.95, True), ("0.95", 1), (None, 1)],
)
def test_bonferroni_confidence_rejects_invalid_inputs(joint, n_bounds):
    with pytest.raises(ValueError):
        bonferroni_confidence(joint, n_bounds)


# --------------------------------------------------------------------------
# sampling_design_supported
# --------------------------------------------------------------------------


def test_supported_sampling_design():
    ok, reason = sampling_design_supported(SUPPORTED_DESIGN)
    assert ok is True
    assert "supported" in reason.lower()


def test_supported_design_ignores_extra_keys():
    ok, _ = sampling_design_supported({**SUPPORTED_DESIGN, "sample_size": 40, "seed": 7})
    assert ok is True


@pytest.mark.parametrize(
    ("override", "expected_fragment"),
    [
        ({"unit": "trace"}, "unit"),
        ({"unit": None}, "unit"),
        ({"method": "stratified"}, "method"),
        ({"method": "disagreement"}, "method"),
        ({"independence_assumption_documented": False}, "independence_assumption_documented"),
        ({"independence_assumption_documented": "yes"}, "independence_assumption_documented"),
        ({"independence_assumption_documented": 1}, "independence_assumption_documented"),
        ({"fresh_groups": False}, "fresh_groups"),
        ({"fresh_groups": "true"}, "fresh_groups"),
    ],
)
def test_unsupported_sampling_designs_name_the_problem(override, expected_fragment):
    ok, reason = sampling_design_supported({**SUPPORTED_DESIGN, **override})
    assert ok is False
    assert expected_fragment in reason


@pytest.mark.parametrize("missing", list(SUPPORTED_DESIGN))
def test_missing_design_key_is_unsupported(missing):
    design = {k: v for k, v in SUPPORTED_DESIGN.items() if k != missing}
    ok, reason = sampling_design_supported(design)
    assert ok is False
    assert missing in reason


def test_all_problems_are_reported_together():
    ok, reason = sampling_design_supported({})
    assert ok is False
    for key in SUPPORTED_DESIGN:
        assert key in reason


@pytest.mark.parametrize("design", [None, "group", ["group", "uniform_random"], 42])
def test_non_mapping_design_is_unsupported_without_raising(design):
    ok, reason = sampling_design_supported(design)  # type: ignore[arg-type]
    assert ok is False
    assert reason
