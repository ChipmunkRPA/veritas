from types import SimpleNamespace

import numpy as np
import pytest

from scripts import compare_mouth_artifact_v5 as trial


def inputs():
    centers = set(range(1, 61, 4))
    tracks = {"track:0": [SimpleNamespace(frame_index=i) for i in range(63)]}
    table = SimpleNamespace(seconds=lambda i: 7 + i / 25)
    return centers, tracks, table


def test_constant_forgery_bias_does_not_create_a_local_edit_range():
    centers, tracks, table = inputs()
    windows = trial.artifact_windows(centers, tracks, {i: 9.0 for i in range(63)}, table)
    assert windows and all(w["response"] == 0.0 for w in windows)


def test_local_forgery_response_points_to_source_clock_and_surrounding_frames():
    centers, tracks, table = inputs()
    logits = {i: 3.0 if 18 <= i <= 38 else 0.0 for i in range(63)}
    window = trial.artifact_windows(centers, tracks, logits, table)[0]
    assert window["response"] > 2.0
    assert window["start_s"] >= 7.6 and window["end_s"] <= 8.6
    assert len(window["centers"][:2]) == len(window["centers"][-2:]) == 2
    shifted = trial.artifact_windows(centers, tracks, {i: z + 10 for i, z in logits.items()}, table)[0]
    assert shifted["centers"] == window["centers"]
    assert shifted["response"] == pytest.approx(window["response"])


def test_missing_triplets_and_large_source_gaps_are_unassessed():
    centers, tracks, table = inputs()
    assert trial.artifact_windows(centers, tracks, {i: 2.0 for i in centers}, table) == []
    table.seconds = lambda i: i
    assert trial.artifact_windows(centers, tracks, {i: 2.0 for i in range(63)}, table) == []


def test_calibration_point_cannot_be_optimized_by_ai_responses():
    rows = [
        {"label": 0, "condition": c, "artifact_windows": [{"response": value}]}
        for c, value in zip(sorted(trial.CONTROLS), (0.1, 0.2, 0.3, 0.4), strict=True)
    ]
    point = trial.choose_point(rows)
    assert point > 0.4
    assert (
        trial.choose_point(
            rows + [{"label": 1, "condition": "ai-redraw", "artifact_windows": [{"response": 99999}]}]
        )
        == point
    )
    assert trial.choose_point(rows[:-1]) is None


def test_measurement_polygon_preserves_native_position_and_is_not_a_pixel_claim():
    window = {"centers": [1, 2, 3, 4, 5, 6, 7]}
    matrix = np.array([[2.0, 0.0, -10.0], [0.0, 2.0, -20.0]])
    bindings = [
        {
            "frame_index": 4,
            "alignment_matrix_hex": [[float(x).hex() for x in row] for row in matrix],
            "alignment_side": 40,
        }
    ]
    table = SimpleNamespace(pts=list(range(10)), seconds=lambda i: i / 10, width=80, height=60)
    result = trial.measured_region(window, bindings, table)
    assert result["frame_index"] == result["pts"] == 4 and result["time_s"] == 0.4
    assert np.array_equal(result["polygon"], [[5, 10], [25, 10], [25, 30], [5, 30]])
    assert "not causal" in result["interpretation"]


def test_unassessed_positive_is_a_miss_even_with_no_exceptions():
    rows = [{"id": "missing", "group": "actor", "condition": "ai-redraw", "label": 1, "artifact_windows": []}]
    outcomes = trial.previous.evaluate(rows, "artifact", 0.4, None)
    summary = trial.previous.summary(outcomes)
    assert summary["planned"] == summary["ai_edits"] == summary["ai_without_flag"] == 1
    assert summary["localized_ai"] == summary["assessed"] == 0
