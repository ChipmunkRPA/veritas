"""Exact ranking, genuine-only points and complete adverse-outcome accounting."""

import copy
from fractions import Fraction

import numpy as np
import pytest

from scripts import image_relayformer_v18 as method
from scripts.review_relayformer_v18 import independent_mask, separate_counts


def test_every_retained_float32_ranking_bit_survives():
    raw = np.zeros((20, 40), np.float32)
    raw[1:4, 1:4] = 0.7
    raw[8:11, 1:4] = np.nextafter(np.float32(0.7), np.float32(1))
    raw[14:17, 1:4] = 0.8
    mask, facts = method.native_mask(raw)
    independent, check = independent_mask(raw)
    assert np.array_equal(mask, independent) and facts == check
    assert mask[9, 2] and mask[15, 2] and not mask[2, 2]
    assert raw.dtype == np.float32 and facts["source_map_quantized"] is False


def test_ties_use_source_y_then_x_and_do_not_fill_component_boxes():
    raw = np.zeros((20, 40), np.float32)
    raw[1, 1:5] = 0.8
    raw[5, 1:5] = 0.8
    raw[10, 1:5] = 0.8
    a, _ = method.native_mask(raw)
    b, _ = independent_mask(raw)
    assert np.array_equal(a, b) and a[1, 1] and a[5, 1] and not a[10, 1]


@pytest.mark.parametrize("score", [0.5, 1.0, np.nextafter(np.float32(1), np.float32(0))])
def test_ieee_readout_matches_integer_scaled_float32(score):
    raw = np.full((37, 51), score, np.float32)
    a, facts = method.native_mask(raw)
    b, other = independent_mask(raw)
    assert np.array_equal(a, b) and facts == other


def rows():
    return [
        {
            "source_id": str(i),
            "role": "point",
            "kind": k,
            "label": 0,
            "status": "ok",
            "ungated": {"candidate": {"predicted_positive_pixels": i, "pixels": 1000}},
        }
        for i in range(24)
        for k in ("original", "ordinary")
    ]


def test_exact_source_maximum_points_keep_one_flag_and_ties():
    values = rows()
    policy = method.point(values, "candidate")
    assert policy["groups_flagged"] == 1
    assert Fraction(*policy["strict_source_second_maximum"]) == Fraction(22, 1000)
    assert not method.eligible({"predicted_positive_pixels": 22, "pixels": 1000}, policy)
    assert method.eligible({"predicted_positive_pixels": 23, "pixels": 1000}, policy)
    for row in values:
        row["ungated"]["candidate"]["predicted_positive_pixels"] = 0
    assert method.point(values, "candidate")["groups_flagged"] == 0


@pytest.mark.parametrize("fault", ["AI", "diagnostic", "missing_operation", "failed"])
def test_bad_point_inputs_are_not_silently_assessed(fault):
    values = rows()
    if fault == "AI":
        values[0]["label"] = 1
    elif fault == "diagnostic":
        values[0]["role"] = "diagnostic"
    elif fault == "failed":
        values[0]["status"] = "failed"
    else:
        values.pop()
    with pytest.raises(ValueError):
        method.point(values, "candidate")


def test_separate_contingency_counts_match_errors_and_empty_masks():
    truth = np.zeros((31, 43), bool)
    truth[2:7, 4:12] = True
    selected = np.zeros_like(truth)
    selected[4:10, 8:15] = True
    assert method.counts(selected, truth) == separate_counts(selected, truth)
    assert method.counts(selected & False, truth) == separate_counts(selected & False, truth)


def test_no_highlights_cannot_pass_a_localization_improvement_gate():
    outcomes = []
    for fmt in ("native", "jpeg75"):
        for i in range(48):
            for kind in ("ai", "original", "ordinary"):
                truth = np.zeros((10, 10), bool)
                if kind == "ai":
                    truth[2:5, 2:5] = True
                metric = method.counts(np.zeros_like(truth), truth)
                outcomes.append(
                    {
                        "id": f"{fmt}:{i}:{kind}",
                        "source_id": str(i),
                        "kind": kind,
                        "format": fmt,
                        "metrics": {"current": metric, "candidate": copy.deepcopy(metric)},
                    }
                )
    result = method.aggregate(outcomes)
    assert result["advance_to_separate_fresh_confirmation_arm"] is None
    assert result["serving_admitted"] is False
