"""Native support and explicit clock provenance for future candidate evaluation."""
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.video_demamba_clock_v13 import UnsupportedSource, crop_box, native_clock, preprocess, sample_plan


def test_two_fps_native_stride_and_center_window_are_explicit():
    result = sample_plan(24., 240)
    assert result["stride"] == 12
    assert result["selected_indices"] == [72, 84, 96, 108, 120, 132, 144, 156]
    assert result["window_start_in_samples"] == 6
    bounded = sample_plan(24., 2400)
    assert len(bounded["sampled_indices"]) == 32
    assert bounded["selected_indices"] == list(range(144, 240, 12))


@pytest.mark.parametrize("fps,total", [(0., 240), (float("nan"), 240), (float("inf"), 240),
                                        (True, 240), (24., 0), (24., 12), (24., True), (24., 240.)])
def test_bad_metadata_or_too_little_actual_support_abstains(fps, total):
    with pytest.raises(UnsupportedSource):
        sample_plan(fps, total)


def test_crop_geometry_and_color_order_are_not_hidden():
    assert crop_box(100, 200) == (20, 0, 180, 100)
    assert crop_box(200, 100) == (0, 20, 100, 180)
    # The publisher trims height for square frames; preserve that declared behavior.
    assert crop_box(100, 100) == (0, 10, 100, 90)
    bgr = np.full((100, 200, 3), [0, 127, 255], np.uint8)
    actual, trace = preprocess(bgr)
    expected = (np.array([255., 127., 0.], np.float32) - np.array([.485, .456, .406], np.float32) * 255.) / (np.array([.229, .224, .225], np.float32) * 255.)
    assert actual.shape == (3, 224, 224) and actual.flags.c_contiguous
    assert np.array_equal(actual[:, 0, 0], expected)
    assert trace["crop_box_xyxy"] == [20, 0, 180, 100]


@pytest.mark.parametrize("times,count", [(["0", "0"], 2), (["0", "NaN"], 2), (["0", "1"], 3), (["1", "0"], 2)])
def test_missing_duplicated_or_reversed_clock_cannot_be_fabricated(monkeypatch, times, count):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps({"frames": [{"best_effort_timestamp_time": t} for t in times]})))
    with pytest.raises(UnsupportedSource):
        native_clock("unused", count)


def test_native_offset_is_preserved(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps({"frames": [{"best_effort_timestamp_time": t} for t in ["5.25", "5.29"]]})))
    assert native_clock("unused", 2) == [5.25, 5.29]
