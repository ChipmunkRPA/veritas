from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.video.face_sequence import CORE, sequence_windows
from scripts.compare_mouth_sequence_v4 import MOUTH, choose_point, region_windows, summary, temporal_iou


def test_generalized_statistic_exactly_replays_the_production_mask():
    rng = np.random.default_rng(21)
    features = {i: rng.normal(size=(64, 8)).astype(np.float32) for i in range(41)}
    tracks = {"track:0": [SimpleNamespace(frame_index=i) for i in features]}
    centers = set(range(1, 40, 4))
    table = SimpleNamespace(seconds=lambda i: i / 25)
    expected = sequence_windows(centers, tracks, features, table)
    assert expected
    assert region_windows(centers, tracks, features, table, core=CORE) == expected


def test_mouth_grid_is_distinct_and_contains_the_aligned_lower_face():
    assert MOUTH.sum() == 12 and not np.array_equal(MOUTH, CORE)
    assert np.all(MOUTH.reshape(8, 8)[5:8, 2:6])


def test_ai_responses_cannot_change_the_control_selected_point():
    rows = [
        {"label": 0, "condition": c, "mouth_windows": [{"response": 0.1 + i * 0.01}]}
        for i, c in enumerate(("original", "local-light", "local-resample", "constant-audio-delay"))
    ]
    point = choose_point(rows)
    assert point > 0.13
    assert (
        choose_point(rows + [{"label": 1, "condition": "ai-redraw", "mouth_windows": [{"response": 999.0}]}])
        == point
    )


def test_missing_control_condition_withholds_a_review_point():
    assert choose_point([{"label": 0, "condition": "original", "mouth_windows": [{"response": 0.1}]}]) is None


def test_unassessed_ai_stays_a_miss_in_the_complete_denominator():
    row = {
        "label": 1,
        "assessed": False,
        "flagged": False,
        "localized": False,
        "temporal_iou": 0.0,
        "spatial_iou": 0.0,
    }
    result = summary([row])
    assert result["planned"] == result["ai_edits"] == result["ai_without_flag"] == 1
    assert result["assessed"] == result["localized_ai"] == 0
    assert result["mean_ai_temporal_iou"] == result["mean_ai_spatial_iou"] == 0


def test_interval_overlap_is_not_just_any_flag_somewhere_in_the_clip():
    assert temporal_iou([0, 1], [2, 3]) == 0
    assert temporal_iou([1, 2], [1.5, 2.5]) == pytest.approx(1 / 3)
    assert temporal_iou(None, [1, 2]) == 0
