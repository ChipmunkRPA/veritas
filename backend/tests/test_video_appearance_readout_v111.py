from copy import deepcopy
from fractions import Fraction

import pytest

from scripts.video_appearance_readout_v111 import (
    logit,
    merge,
    reference_probability,
    review_case,
    summarize,
    support,
)


def head():
    return {"status": "fitted", "center": [0.0], "scale": [1.0], "coefficients": [1.0], "intercept": 0.0}


def fixture():
    local = head()
    model = {"response": head(), "calibration": head()}
    unit = {
        "id": "unit",
        "location": {
            "start": 10**9,
            "end": 2 * 10**9,
            "extent": 3 * 10**9,
            "sampled_frames": [1],
            "context_frames": [0, 2],
        },
        "sampled_ai_frame_presence": 1,
        "context_contains_ai_frame": False,
    }
    plan = {
        "group_index": 0,
        "condition": "swap_a",
        "source_group": "authored",
        "source_role": "selection",
        "publisher_manipulation_label": 1,
        "planned_scope_label": 1,
        "known_insert_frames": [1, 2],
        "known_insert_seconds": ["1", "2"],
        "intended_changed_ai_frames": [1],
        "model_frame_indices": [0, 1, 2],
        "units": [unit],
    }
    bound = {
        "planned": plan,
        "input_sha256": "input",
        "expected_rgb_sha256": {str(i): "rgb" + str(i) for i in range(3)},
        "table": {"pts": [0, 1, 2], "time_base": "1"},
    }
    row = {k: deepcopy(v) for k, v in plan.items() if k not in {"model_frame_indices", "units"}}
    row.update(
        input_sha256="input",
        units=[{**deepcopy(unit), "status": "ok", "raw": 2.0}],
        frame_inputs=[
            {
                "index": i,
                "pts": i,
                "time_base": "1",
                "rgb_sha256": "rgb" + str(i),
                "status": "ok",
                "faces": 1,
                "bbox": [20, 20, 80, 80],
                "landmarks": [[30, 40]] * 5,
                "feature_sha256": "a" * 64,
                "aligned_crop_sha256": "b" * 64,
            }
            for i in range(3)
        ],
        raw_file_flag=True,
        candidate_file_reference_probability=reference_probability([2.0], local, model),
        candidate_file_flag=True,
    )
    return row, bound, local, model


def test_fixed_reference_map_uses_positive_local_maximum_and_withholds_missing():
    _, _, local, model = fixture()
    assert reference_probability([2.0, -1.0], local, model) == pytest.approx(0.8807970779778823)
    assert reference_probability([-2.0, -1.0], local, model) == 0.5
    assert reference_probability([None, 2.0], local, model) is None
    assert reference_probability([], local, model) is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, True])
def test_invalid_observation_cannot_supply_a_finite_logit(bad):
    with pytest.raises(ValueError):
        logit(head(), [bad])


def test_native_review_reproduces_predictions_and_strict_equality_points():
    row, bound, local, model = fixture()
    actual = review_case(row, bound, local, model, 0.5, 0.5)
    assert actual["raw_flag"] and actual["candidate_flag"]
    row["raw_file_flag"] = row["candidate_file_flag"] = False
    actual = review_case(row, bound, local, model, 2.0, row["candidate_file_reference_probability"])
    assert actual["raw_flag"] is False and actual["candidate_flag"] is False


@pytest.mark.parametrize("change", ["clock", "pixels", "truth", "point", "unit", "zero_missing"])
def test_wrong_source_support_or_missing_zero_is_rejected(change):
    row, bound, local, model = fixture()
    if change == "clock":
        row["frame_inputs"][0]["pts"] = 1
    elif change == "pixels":
        row["frame_inputs"][0]["rgb_sha256"] = "changed"
    elif change == "truth":
        row["planned_scope_label"] = 0
    elif change == "point":
        row["candidate_file_flag"] = False
    elif change == "unit":
        row["units"][0]["location"]["end"] += 1
    else:
        row["frame_inputs"][0]["status"] = "unassessed"
        row["units"][0].update(status="not_applicable", raw=0.0)
    with pytest.raises(ValueError):
        review_case(row, bound, local, model, 0.5, 0.5)


def test_missing_window_remains_inconclusive_in_both_arms():
    row, bound, local, model = fixture()
    row["frame_inputs"][0]["status"] = "unassessed"
    row["units"][0].update(status="not_applicable", raw=None)
    row.update(raw_file_flag=None, candidate_file_reference_probability=None, candidate_file_flag=None)
    actual = review_case(row, bound, local, model, 0.5, 0.5)
    assert actual["raw_flag"] is actual["candidate_flag"] is None
    summary = summarize([actual], "candidate")
    assert (
        summary["inconclusive"]
        == summary["inconclusive_ai"]
        == summary["missed_ai_including_inconclusive"]
        == 1
    )
    assert summary["ai_found"] == 0 and summary["source_class_balanced_reference_brier"] is None


def test_bounding_union_and_sparse_false_positions_have_distinct_denominators():
    units = [
        {"location": {"start": 10**9, "end": 3 * 10**9, "sampled_frames": [1, 2]}},
        {"location": {"start": 2 * 10**9, "end": 4 * 10**9, "sampled_frames": [2, 3]}},
    ]
    actual = support(units, [True, True], {2, 7}, (Fraction(2), Fraction(3)))
    assert actual["flagged_sample_positions"] == 3 and actual["flagged_ai_sample_positions"] == 1
    assert actual["false_flagged_sample_positions"] == 2 and actual["ai_source_frames_not_scheduled"] == 1
    assert actual["bounding_seconds"] == 3 and actual["bounding_false_seconds"] == 2
    assert actual["bounding_interval_union_iou"] == pytest.approx(1 / 3)
    assert merge([(Fraction(1), Fraction(2)), (Fraction(2), Fraction(3))]) == [(Fraction(1), Fraction(3))]
