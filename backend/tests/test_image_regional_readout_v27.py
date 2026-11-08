"""Prospective group weighting, invalid-outcome and uncertainty checks."""

import copy
import math

import numpy as np
import pytest

from scripts import image_regional_readout_v27 as readout


def metric(positive, flagged):
    tp, fp, fn = (10, 0, 10) if positive and flagged else (0, 0, 20) if positive else (0, int(flagged), 0)
    return {
        "status": "ok",
        "metrics": {
            "pixels": 100,
            "intersection": tp,
            "false_positive_pixels": fp,
            "false_negative_pixels": fn,
            "predicted_positive_pixels": tp + fp,
            "reference_positive_pixels": tp + fn,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1,
            "recall": tp / (tp + fn) if tp + fn else None,
        },
    }


def rows(groups, role, *, policies=None, candidate_ai=True):
    names = policies or readout.policies()
    return [
        {
            "role": role,
            "format": "native",
            "source_id": i,
            "component": f"group{i:03d}",
            "kind": kind,
            "baseline": metric(kind == "ai", False),
            "candidate": {name: metric(kind == "ai", candidate_ai and kind == "ai") for name in names},
        }
        for i in range(groups)
        for kind in readout.KINDS
    ]


def test_equal_calibration_outcomes_use_fixed_conservative_tie_break():
    result = readout.calibrate(rows(2, "calibration"), groups=2)
    assert result["selected_policy"] == "p0.90-a0.01000"
    assert len(result["policy_results"]) == 28
    assert result["serving_admitted"] is result["probability_calibration_performed"] is False


def test_no_eligible_calibration_is_an_explicit_empty_selection():
    values = rows(2, "calibration")
    for row in values:
        if row["kind"] == "tone":
            row["candidate"] = {name: metric(False, True) for name in readout.policies()}
    result = readout.calibrate(values, groups=2)
    assert result["selected_policy"] is None
    assert not any(p["eligible"] for p in result["policy_results"])


@pytest.mark.parametrize("status", ["failed", "unavailable", "skipped", "inconclusive"])
def test_unchecked_results_cannot_become_clear_observations(status):
    values = rows(2, "calibration")
    values[0]["baseline"] = {"status": status}
    report = readout.coverage(values, readout.policies())
    assert report["outcomes"]["baseline"]["not_successfully_checked"] == 1
    assert report["outcomes"]["baseline"]["not_successfully_checked_fraction"] == 1 / 8
    assert report["whole_file_originality_established"] is False
    with pytest.raises(ValueError, match="cannot become"):
        readout.calibrate(values, groups=2)


def test_missing_policy_and_duplicate_views_are_not_dropped():
    values = rows(2, "calibration")
    del values[0]["candidate"][next(iter(readout.policies()))]
    with pytest.raises(ValueError, match="must be present"):
        readout.calibrate(values, groups=2)
    values = rows(2, "calibration")
    values.append(copy.deepcopy(values[0]))
    with pytest.raises(ValueError, match="unique source-bound views"):
        readout.calibrate(values, groups=2)


def test_one_photograph_cannot_appear_in_multiple_components():
    values = rows(2, "calibration")
    values[0]["component"] = "group001"
    with pytest.raises(ValueError, match="unique source-bound views"):
        readout.calibrate(values, groups=2)


def test_pixel_counts_must_reconstruct_iou_recall_and_negative_truth():
    values = rows(2, "calibration")
    name = next(iter(readout.policies()))
    values[0]["candidate"][name]["metrics"]["iou"] = 0.99
    with pytest.raises(ValueError, match="reproduce pixel counts"):
        readout.calibrate(values, groups=2)
    values = rows(2, "calibration")
    values[1]["baseline"] = metric(True, False)
    with pytest.raises(ValueError, match="all-zero AI truth"):
        readout.calibrate(values, groups=2)


def test_near_duplicates_do_not_multiply_independent_evidence():
    values = rows(2, "point", policies=("p0.90-a0.01000",))
    for row in values:
        if row["component"] == "group001":
            row["candidate"]["p0.90-a0.01000"] = metric(row["kind"] == "ai", False)
    duplicates = []
    for sid in (2, 3):
        for row in values:
            if row["source_id"] == 1:
                duplicate = copy.deepcopy(row)
                duplicate["source_id"] = sid
                duplicates.append(duplicate)
    values.extend(duplicates)
    result = readout.compare(values, "p0.90-a0.01000", role="point", groups=2)
    assert result["paired_useful_fraction_difference"] == 0.5
    assert result["candidate"]["groups"] == 2
    assert result == readout.compare(list(reversed(values)), "p0.90-a0.01000", role="point", groups=2)


def test_exact_binomial_bounds_match_analytic_and_published_reference():
    assert abs(readout.binomial_upper(0, 48) - (1 - 0.025 ** (1 / 48))) < 1e-14
    # Official SciPy proportion_ci example: exact two-sided 95% for 7/50.
    assert abs(readout.binomial_upper(7, 50) - 0.26739600249700846) < 1e-12
    assert abs(readout.binomial_lower(7, 50) - 0.05819170033997342) < 1e-12
    assert readout.binomial_upper(48, 48) == 1
    assert readout.binomial_lower(0, 48) == 0


@pytest.mark.parametrize("k,n", [(1, 0), (-1, 1), (True, 2), (3, 2)])
def test_invalid_binomial_denominators_fail(k, n):
    with pytest.raises(ValueError):
        readout.binomial_upper(k, n)


def test_small_clean_confirmation_does_not_hide_rare_false_alarm_uncertainty():
    policy = "p0.90-a0.01000"
    result = readout.compare(
        rows(48, "confirmation", policies=(policy,)), policy, role="confirmation", groups=48
    )
    assert result["paired_group_bootstrap_95pct_interval"] == [1, 1]
    assert result["false_group_rate_difference_95pct_upper_bound"] > 0.05
    assert result["stage_passed"] is False
    # A larger independent cohort can resolve uncertainty without changing gates.
    result = readout.compare(
        rows(96, "confirmation", policies=(policy,)), policy, role="confirmation", groups=96
    )
    assert result["false_group_rate_difference_95pct_upper_bound"] < 0.05
    assert result["stage_passed"] is True
    assert result["serving_admitted"] is False


def test_recovered_false_groups_are_credited_without_ignoring_new_ones():
    policy = "p0.90-a0.01000"
    values = rows(48, "confirmation", policies=(policy,))
    for row in values:
        if row["kind"] == "tone" and row["source_id"] < 24:
            row["baseline"] = metric(False, True)
        if row["kind"] == "tone" and row["source_id"] in (30, 31):
            row["candidate"][policy] = metric(False, True)
    result = readout.compare(values, policy, role="confirmation", groups=48)
    assert result["newly_false_groups"] == 2 and result["recovered_false_groups"] == 24
    assert result["stage_passed"] is True
    assert result["false_group_rate_difference_95pct_upper_bound"] < 0


def test_image_gate_is_strict_and_does_not_accept_malformed_fields():
    policy = {"pixel": 0.5, "area": 0.01}
    assert not readout.image_gate(np.full((10, 10), 0.5), policy)
    values = np.full((10, 10), 0.5)
    values[0, 0] = 0.51
    assert readout.image_gate(values, policy)
    values[0, 0] = math.nan
    with pytest.raises(ValueError):
        readout.image_gate(values, policy)
