from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import video_face_confound_v222 as confound


def roles():
    result = {}
    points = [[26, 27], [38, 27], [32, 32], [27, 39], [37, 39]]
    for group, start in (("before", 0), ("during", 3), ("after", 6)):
        result[group] = [{"frame_index": i, "rgb": np.full((64, 64, 3), 100, np.uint8),
                          "bbox": {"x": 20, "y": 20, "w": 24, "h": 24}, "landmarks": points}
                         for i in range(start, start + 3)]
    return result


def test_model_and_app_settings_are_not_imported():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(Path(confound.__file__).resolve().parents[1])}
    proc = subprocess.run([sys.executable, "-B", "-c", "import sys; from scripts import video_face_confound_v222; "
                           "assert not ({'torch','onnxruntime','transformers','app.core.config'} & sys.modules.keys())"],
                          env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_neutral_and_coherent_whole_frame_luminance_shift():
    inputs = roles()
    for row in inputs["during"]:
        row["rgb"][:] = 120
    result = confound.measure(inputs)
    assert result["status"] == "measured"
    pair = result["comparisons"]["before_to_during"]
    assert pair["geometry_rms_eye_units"] == 0
    assert pair["face_shift"]["luma"] == pytest.approx(20)
    assert pair["nonface_shift"]["luma"] == pytest.approx(20)
    assert pair["face_minus_nonface_shift"]["luma"] == pytest.approx(0)
    assert pair["face_nonface_shift_cosine"] == pytest.approx(1)
    assert result["comparisons"]["before_to_after"]["face_nonface_shift_cosine"] is None


def test_face_only_chroma_and_similarity_normalized_geometry():
    inputs = roles()
    for row in inputs["during"]:
        row["rgb"][20:44, 20:44] = [120, 100, 80]
        row["landmarks"] = [[x + 1, y + 1] for x, y in row["landmarks"]]
    result = confound.measure(inputs)
    pair = result["comparisons"]["before_to_during"]
    assert pair["geometry_rms_eye_units"] == pytest.approx(0)
    assert pair["nonface_shift"] == {channel: 0 for channel in confound.CHANNELS}
    assert pair["face_minus_nonface_shift"]["red_chroma"] > 0
    assert pair["face_minus_nonface_shift"]["blue_chroma"] < 0
    for row in inputs["during"]:
        row["landmarks"] = copy.deepcopy(row["landmarks"])
        row["landmarks"][2][1] += 2
    assert confound.measure(inputs)["comparisons"]["before_to_during"]["geometry_rms_eye_units"] > 0


def test_proper_similarity_transform_preserves_shape():
    points = np.asarray(roles()["before"][0]["landmarks"], dtype=float)
    rotated = points @ np.array([[0., -1.], [1., 0.]]) * 2.5 + [200., 100.]
    assert np.allclose(confound.geometry(points), confound.geometry(rotated))


def test_unavailable_inputs_are_never_scored():
    inputs = roles()
    assert confound.measure({})["reason"] == "missing_role_context"
    short = roles()
    short["after"].pop()
    assert confound.measure(short)["reason"] == "insufficient_role_frames"
    overlap = roles()
    overlap["during"][0]["frame_index"] = 2
    assert confound.measure(overlap)["reason"] == "overlapping_or_unordered_role_frames"
    invalid = roles()
    invalid["before"][0]["rgb"] = np.full((64, 64, 3), np.nan)
    assert confound.measure(invalid)["reason"] == "rgb_out_of_range"
    invalid = roles()
    invalid["before"][0]["landmarks"] = [[30, 30]] * 5
    assert confound.measure(invalid)["reason"] == "degenerate_eye_landmarks"
    invalid = roles()
    invalid["before"][0]["bbox"] = {"x": -1, "y": 0, "w": 10, "h": 10}
    assert confound.measure(invalid)["reason"] == "bbox_out_of_range"
    invalid = roles()
    for group in invalid.values():
        for row in group:
            row["bbox"] = {"x": 0, "y": 0, "w": 64, "h": 64}
    assert confound.measure(invalid)["reason"] == "insufficient_nonface_context"
    assert confound.measure(inputs)["status"] == "measured"
