from fractions import Fraction

import numpy as np
import pytest

from scripts import video_chroma_boundary_transfer as candidate


def sample_sites(y):
    return np.column_stack((np.full(len(y), 50), y)), np.tile([1., 0.], (len(y), 1))


def measure(lab, y):
    sites, normals = sample_sites(y)
    return candidate.contrast(lab, sites, normals, 4, np.zeros(2))


def test_local_chroma_step_separates_from_smooth_color_ramp():
    lab = np.zeros((101, 101, 2), np.float32)
    lab[:, 51:, 0] = 24
    ys = list(range(44, 58, 2))
    assert measure(lab, ys) == pytest.approx(np.log((24 + 1.5) / 1.5))
    lab[:, :, 0] = np.arange(101, dtype=np.float32)
    assert measure(lab, ys) == 0.


def test_partial_arc_measures_when_four_sites_remain_visible():
    lab = np.zeros((101, 101, 2), np.float32)
    lab[:, 51:, 0] = 24
    assert measure(lab, [-2, -1, 0, 3, 5, 7, 9]) > 2.
    assert measure(lab, [-3, -2, -1, 0, 3, 5, 7]) is None


def test_inconsistent_chroma_step_directions_supply_no_support():
    lab = np.zeros((101, 101, 2), np.float32)
    for y, step in zip(range(44, 58, 2), [24, -24, 24, -24, 24, -24, 0], strict=True):
        lab[y, 51:, 0] = step
    assert measure(lab, list(range(44, 58, 2))) == 0.


def observations(values):
    times = [Fraction(i, 4) for i in range(len(values))]
    features = {i: {"face_xywh": [20, 30, 100, 90], "arcs": {"chin/1": {"score": score,
                    "native_polyline": [[70, 100], [72, 100]]}}} for i, score in enumerate(values) if score is not None}
    return times, features


def test_partial_time_quorum_preserves_exact_native_flanks_and_core():
    times, features = observations([0, 0, 0, 0, 3, None, 4, 0, 0, 0, 0])
    best = candidate.scan(times, list(range(len(times))), features)["best"]
    assert best["score"] == 3 and best["arc"] == "chin/1"
    assert (best["start"], best["end"]) == ("1", "7/4")
    assert [anchor["time"] for anchor in best["anchors"]] == ["3/4", "3/2", "7/4"]
    assert candidate.outcome({"best": best, "truth": ["1", "7/4"]}, 2.) == "localized"
    assert candidate.outcome({"best": best, "truth": ["7/4", "5/2"]}, 2.) == "wrong_interval"


def test_steady_chroma_edge_does_not_make_an_interval_flag():
    times, features = observations([3] * 11)
    best = candidate.scan(times, list(range(len(times))), features)["best"]
    assert best["score"] == 0.
    assert candidate.outcome({"best": best, "truth": None}, 0.) == "measured_no_flag"


def test_missing_face_coverage_is_unassessed():
    times, features = observations([0, None, None, None, 0, None, None, None, 0])
    result = candidate.scan(times, list(range(len(times))), features)
    assert result == {"best": None, "measured_windows": 0}
    assert candidate.outcome({**result, "truth": None}, 0.) == "unassessed"
