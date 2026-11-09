"""Prevent conservative research rules from silently discarding useful tradeoffs."""

import pytest

from scripts.evaluation_tradeoffs_v44 import observed_frontier, positive_hit_overlap


def test_native_localization_gain_is_retained_despite_original_false_alarm_increase():
    models = {
        "current": {"useful": 0.20833333333333334, "original": 3, "tone": 7, "blur": 11},
        "adapted": {"useful": 2 / 3, "original": 6, "tone": 7, "blur": 5},
    }
    result = observed_frontier(models, maximize=("useful",), minimize=("original", "tone", "blur"))
    assert result["observed_frontier"] == ["adapted", "current"]
    assert result["models"]["adapted"]["retained_in_comparison"]
    assert result["automatic_rejection"] is result["serving_admitted"] is False


def test_abstention_does_not_hide_loss_of_useful_detections():
    result = observed_frontier(
        {"current": {"hits": 648, "false": 58, "unknown": 387}, "cautious": {"hits": 500, "false": 7, "unknown": 897}},
        maximize=("hits",), minimize=("false", "unknown"),
    )
    assert len(result["observed_frontier"]) == 2


def test_dominated_point_is_annotated_and_retained_for_complementarity_review():
    result = observed_frontier(
        {"a": {"hit": 20, "false": 2}, "b": {"hit": 15, "false": 3}},
        maximize=("hit",), minimize=("false",),
    )
    assert result["models"]["b"] == {"observed_dominated_by": ["a"], "retained_in_comparison": True}
    assert result["statistical_superiority_established"] is False


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True])
def test_unchecked_or_invalid_metric_never_becomes_zero(bad):
    with pytest.raises(ValueError, match="finite"):
        observed_frontier({"a": {"hit": bad}}, maximize=("hit",), minimize=())


def test_missing_axis_and_duplicate_axes_refuse_a_comparison():
    with pytest.raises(ValueError):
        observed_frontier({"a": {}}, maximize=("hit",), minimize=())
    with pytest.raises(ValueError):
        observed_frontier({"a": {"hit": 1}}, maximize=("hit",), minimize=("hit",))


def test_complementary_region_hits_are_not_claimed_as_combined_mask_accuracy():
    result = positive_hit_overlap(
        {"a": True, "b": False, "c": True, "d": False},
        {"a": True, "b": True, "c": False, "d": False},
    )
    assert result["both_useful"] == 1
    assert result["candidate_useful_baseline_missed"] == 1
    assert result["baseline_useful_candidate_missed"] == 1
    assert result["either_method_useful"] == 3
    assert result["combined_serving_accuracy_measured"] is False


def test_unknown_reference_is_neither_a_miss_nor_an_original_verdict():
    result = positive_hit_overlap({"a": None, "b": True}, {"a": True, "b": None})
    assert result["candidate_useful_baseline_missed"] == 0
    assert result["baseline_useful_candidate_missed"] == 0
    assert result["candidate_useful_baseline_unknown"] == 1
    assert result["baseline_useful_candidate_unknown"] == 1
    assert result["whole_file_originality_established"] is False


def test_incomplete_or_non_boolean_case_support_refuses_selection():
    with pytest.raises(ValueError):
        positive_hit_overlap({"a": True}, {"b": True})
    with pytest.raises(ValueError):
        positive_hit_overlap({"a": 1}, {"a": True})
