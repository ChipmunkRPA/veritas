from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mouth_context_v10 as context
from scripts import mouth_temporal_v11 as method
from scripts.prepare_mouth_temporal_v11 import clock_span, seed_for


def fixture():
    centers = set(range(2, 59, 4))
    tracks = {"face": [SimpleNamespace(frame_index=i) for i in range(60)]}
    table = SimpleNamespace(seconds=lambda i: i / 25)
    values = {i: float(np.sin(i / 4)) for i in range(60)}
    head = {
        "model_inputs": [{"frame_index": i, "margin": v} for i, v in values.items()],
        "context_windows_all": context.windows(centers, tracks, values, table),
    }
    measured = [{"frame_index": i, "values": [i / 60] * 9} for i in range(60)]
    return {"v8": head, "v9": deepcopy(head)}, measured


def test_complete_windows_and_feature_order():
    heads, measured = fixture()
    rows = method.window_features(heads, measured)
    assert len(rows) == len(heads["v8"]["context_windows_all"]) > 12
    assert len(method.FEATURE_NAMES) == len(set(method.FEATURE_NAMES)) == 49
    assert all(len(r["features"]) == 49 and len(r["features_float64_sha256"]) == 64 for r in rows)
    first = rows[0]
    mid = first["centers"][2:-2]
    assert first["features"][14] == pytest.approx(np.mean(mid) / 60)
    assert first["features"][-2:] == [first["end_s"] - first["start_s"], len(mid)]


def test_missing_nonhighlighted_window_or_measurement_is_refused():
    heads, measured = fixture()
    heads["v9"]["context_windows_all"].pop()
    with pytest.raises(ValueError, match="identical complete"):
        method.window_features(heads, measured)
    heads, measured = fixture()
    with pytest.raises(ValueError, match="Missing, duplicate"):
        method.window_features(heads, measured[1:])


def test_geometry_and_quality_are_native_measures():
    points = [
        [0.5, 0.6, 0],
        [0.5, 0.65, 0],
        [0.4, 0.625, 0],
        [0.6, 0.625, 0],
        [0.3, 0.3, 0],
        [0.7, 0.3, 0],
        [0.5, 0.45, 0],
    ]
    values = method.geometry_values(points, 200, 100)
    assert values == pytest.approx([0.125, 0.5, 0, 0.1875, 0, 0.4])
    chip = np.full((20, 30, 3), 128, np.uint8)
    assert method.quality_values(chip) == pytest.approx([128 / 255, 0, 0])
    points[0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        method.geometry_values(points, 200, 100)


def test_fit_common_clock_does_not_pad_or_drop_valid_frames():
    table = SimpleNamespace(
        pts=list(range(1, 72)), tb=Fraction(1, 25), time_base="1/25", seconds=lambda i: (i + 1) / 25
    )

    class Table:
        def __len__(self):
            return 71

    wrapped = Table()
    wrapped.__dict__.update(table.__dict__)
    audio = SimpleNamespace(
        mapping={"timeline": [{"start_sample": 0, "source_pts": 0}], "sample_rate": 16000},
        pcm=np.zeros(72 * 640, np.float32),
    )
    span = clock_span(wrapped, audio)
    assert span["frames"] == 71
    assert span["source_sample_range"] == [640, 72 * 640]
    audio.pcm = audio.pcm[:-1]
    span = clock_span(wrapped, audio)
    assert span["frames"] == 70
    assert span["excluded_end_picture_frames"] == 1


def test_learning_refuses_point_role_and_missing_generator():
    with pytest.raises(ValueError, match="sixteen"):
        method.train([{"role": "point"}])
    rows = [
        {"role": "fit", "group": str(i), "condition": "ai-redraw", "label": 1, "features": [0.0] * 49}
        for i in range(16)
    ]
    with pytest.raises(ValueError, match="Every AI fitting"):
        method.train(rows)


def test_seed_is_bound_to_source_and_frame():
    assert seed_for("a", 1) == seed_for("a", 1)
    assert 0 <= seed_for("a", 1) < 2**63
    assert len({seed_for("a", 1), seed_for("a", 2), seed_for("b", 1)}) == 3


def test_standalone_feature_arithmetic_matches_all_windows():
    from scripts.review_mouth_temporal_v11 import feature_vector

    heads, measured = fixture()
    rows = method.window_features(heads, measured)
    for row in rows:
        assert feature_vector(row, heads, measured) == pytest.approx(row["features"], abs=1e-12)


def test_pose_features_do_not_depend_on_picture_translation():
    points = np.array(
        [
            [0.5, 0.6, 0],
            [0.5, 0.65, 0],
            [0.4, 0.625, 0],
            [0.6, 0.625, 0],
            [0.3, 0.3, 0],
            [0.7, 0.3, 0],
            [0.5, 0.45, 0],
        ]
    )
    moved = points.copy()
    moved[:, :2] += [0.1, 0.05]
    assert method.geometry_values(points, 200, 100) == pytest.approx(method.geometry_values(moved, 200, 100))
