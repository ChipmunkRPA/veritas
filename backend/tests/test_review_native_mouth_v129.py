import copy

import numpy as np
import pytest

from app.detectors.video.decode import FrameTable
from scripts import review_native_mouth_v129 as audit


def test_independent_descriptor_projection_and_three_heads_have_closed_form_values(monkeypatch):
    table = FrameTable("1/1000", (0, 30, 70, 110, 160), (), "?????", 96, 64, 96, 64)
    image = np.zeros((5, 768), np.float32)
    image[:, 0] = [1, 2, 0, 3, 4]
    audio = np.zeros((3, 3840))
    audio[:, 0] = 1
    row = {
        "status": "extracted",
        "table": table.to_json(),
        "speech": {"indices": [1, 2, 3]},
        "frames": [{"status": "ok", "frame_index": i, "feature_row": i} for i in range(5)],
    }
    monkeypatch.setattr(
        audit.original.fitted.reviewed, "array_file", lambda r, dtype: image if dtype == np.float32 else audio
    )
    vp = np.zeros((16, 2304))
    vp[0, 0] = vp[1, 768] = vp[2, 1536] = 1
    ap = np.zeros((16, 3840))
    ap[0, 0] = 1
    learned = {
        "reference": {
            "visual_projection": {"center": np.zeros(2304), "components": vp, "scale": np.ones(16)},
            "audio_projection": {"center": np.zeros(3840), "components": ap, "scale": np.ones(16)},
            "audio_to_visual": np.eye(16),
        },
        "heads": {},
    }
    for name, dim in (("visual_only", 16), ("concatenation", 32), ("interaction", 64)):
        learned["heads"][name] = {
            "mean": np.zeros(dim),
            "scale": np.ones(dim),
            "coef": np.ones(dim),
            "intercept": 0,
        }
    indices, raw = audit.reconstruct(row, learned)
    assert indices == [1, 2, 3]
    assert raw["visual_only"][1] == pytest.approx(3)
    assert raw["concatenation"][1] == pytest.approx(4)
    assert raw["interaction"][1] == pytest.approx(8)


def test_native_picture_cell_union_has_known_vfr_overlap_without_double_counting():
    table = FrameTable("1/1000", (0, 20, 100, 130, 200), (), "?????", 96, 64, 96, 64)
    selected = [{"support_frames_half_open": [0, 3]}, {"support_frames_half_open": [2, 5]}]
    result = audit.frame_union(table, 50, selected, [1, 4])
    assert result["flagged_duration_seconds"] == 0.25
    assert result["edit_overlap_seconds"] == 0.18
    assert result["outside_edit_flagged_seconds"] == 0.07
    assert result["temporal_union_iou"] == 0.72
    assert result["localized_iou_0_2"] is True


def test_separate_scalar_sigmoid_and_correlations_have_known_values():
    calibration = {"slope": 1.0, "centre": 0.0, "scale": 1.0, "intercept": 0.0}
    assert audit.probability(0, calibration) == 0.5
    assert audit.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == 1
    assert audit.pearson([1, 2, 3, 4], [8, 6, 4, 2]) == -1
    assert audit.pearson([1, 1, 1], [2, 3, 4]) == 0
    with pytest.raises(ValueError):
        audit.close([np.nan], [1])
    with pytest.raises(ValueError):
        audit.close([1], [2])


def test_full_actual_lag_profile_is_replayed_and_wrong_coefficient_or_winner_fails():
    track = {
        "frame_inputs": [{"tick": i, "mouth_value": float(i)} for i in range(25, 50)],
        "windows": [
            {
                "start_tick": 25,
                "end_tick": 50,
                "audio_curve": [{"tick": i, "rms": float(i)} for i in range(10, 65)],
                "lag_profile": [{"offset_ticks": i, "correlation": 1.0} for i in range(-15, 16)],
                "offset_ticks": -15,
                "peak_correlation": 1.0,
            }
        ],
    }
    current = {"details": {"av_input": {"status": "ready", "tracks": [track]}}}
    assert audit.lag_replay(current) == (1, 31)
    broken = copy.deepcopy(current)
    broken["details"]["av_input"]["tracks"][0]["windows"][0]["offset_ticks"] = 0
    with pytest.raises(ValueError, match="lag"):
        audit.lag_replay(broken)
    broken = copy.deepcopy(current)
    broken["details"]["av_input"]["tracks"][0]["windows"][0]["lag_profile"][0]["correlation"] = 0.3
    with pytest.raises(ValueError, match="arithmetic"):
        audit.lag_replay(broken)
    assert audit.lag_replay({"details": {}}) == (0, 0)


def test_unrelated_role_or_neural_command_is_not_dispatched():
    with pytest.raises(ValueError):
        audit.run("confidence_cal")
    with pytest.raises(ValueError):
        audit.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "extract"])
