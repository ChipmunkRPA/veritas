"""Landmark transfer preserves v112 support and treats unavailable pairs explicitly."""

from copy import deepcopy

import pytest

from scripts import video_appearance_plan_v107 as planning
from scripts import video_landmark_transfer_v255 as transfer


def test_packet_reuses_all_planned_triplet_frames():
    unit = planning.windows(175, "1/25", set())["units"][0]
    frames = {
        index: {"bbox": [10, 11, 22, 23], "landmarks": [[0, 0]] * 5}
        for index in unit["location"]["sampled_frames"] + unit["location"]["context_frames"]
    }
    result = transfer.packet(unit, frames)
    assert result["centers"]["during"] == unit["middle_centers"]
    assert [row["frame_index"] for row in result["frame_inputs"]] == sorted(frames)
    assert result["frame_inputs"][0]["bbox"] == {"x": 10, "y": 11, "w": 22, "h": 23}

    with pytest.raises(ValueError, match="window frame support"):
        transfer.packet({**unit, "middle_centers": unit["before_centers"]}, frames)


def observation(score, *, flag=False, location=0):
    return {
        "metrics": dict.fromkeys(transfer.METRICS, score) if score is not None else None,
        "reason": "v112_window_unassessed" if score is None else None,
        "current_appearance_measured": score is not None,
        "current_flag": flag,
        "location": location,
    }


def test_paired_readout_keeps_exact_clock_and_reports_missing_endpoints():
    windows = {
        (1, "swap_a", 0): observation(0.125, flag=True),
        (2, "swap_a", 0): observation(0.25, flag=True),
        (3, "swap_a", 0): observation(0.125),
        (1, "original", 0): observation(0.375),
        (2, "original", 0): observation(None),
        (3, "original", 0): observation(0.5),
    }
    result = transfer.paired(windows, "swap_a", "original")
    assert result["existing_flag_windows_planned_for_pairing"] == 2
    assert result["matched_geometry_windows"] == result["matched_source_groups"] == 1
    assert result["missing_endpoints"] == {"control_geometry_unavailable": 1}
    metric = result["swap_minus_control"][transfer.METRICS[0]]
    assert metric["negative"] == 1 and metric["distribution"]["median"] == -0.25

    changed = deepcopy(windows)
    changed[1, "original", 0]["location"] = 3
    with pytest.raises(ValueError, match="same scheduled window"):
        transfer.paired(changed, "swap_a", "original")


def test_distributions_give_each_source_group_equal_weight():
    windows = {
        (1, "original", 0): observation(0),
        (1, "original", 1): observation(0),
        (1, "original", 2): observation(0),
        (2, "original", 0): observation(1),
        (2, "original", 1): observation(None),
        (3, "original", 0): observation(None),
    }
    result = transfer.summarize(windows)
    assert result["groups_with_geometry"] == 2
    assert result["groups_with_geometry_for_all_windows"] == 1
    assert result["geometry_measured_windows"] == 4
    assert result["unavailable_windows_by_reason"] == {"v112_window_unassessed": 2}
    assert result["metrics_across_group_medians"][transfer.METRICS[0]]["median"] == 0.5
