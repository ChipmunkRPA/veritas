"""Regression tests for the motion and fixed-scene comparisons."""

import math

import cv2
import numpy as np

from scripts.video_face_motion_context import compare


def fixture(*, move=True, roll=True):
    groups = {"before": [1, 2], "during": [3, 4, 5, 6, 7], "after": [8, 9]}
    frames = {}
    measurements = []
    background = np.random.default_rng(42).integers(0, 255, (180, 320, 3), dtype=np.uint8)
    for role, indices in groups.items():
        core = role == "during"
        x = 70 if core and move else 140
        angle = math.radians(18 if core and roll else 0)
        rotation = np.asarray(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
        points = np.asarray(((-10, -10), (10, -10), (0, 0), (-6, 12), (6, 12))) @ rotation.T + (x, 61)
        for i in indices:
            image = background.copy()
            image[34:90, x-19:x+19] = np.asarray((81, 128, 167), dtype=np.uint8)
            frames[i] = image
            measurements.append({"frame_index": i, "pts": i * 40, "decoded_rgb_sha256": "a" * 64,
                                 "landmarks": points.tolist(), "bbox": {"x": x-19, "y": 34, "w": 38, "h": 56}})
    return {"frame_inputs": measurements, "centers": groups}, frames


def test_stationary_scene_supports_a_returning_face_movement():
    measured = compare(*fixture())
    assert measured["status"] == "measured"
    assert measured["translation_context"] and measured["translation_roll_context"]
    assert measured["registered_frames"] == measured["scheduled_frames"] == 9


def test_roll_and_translation_are_measured_separately():
    roll_only = compare(*fixture(move=False))
    assert roll_only["status"] == "measured"
    assert roll_only["roll_support"] == 5 and not roll_only["translation_context"]
    lateral_only = compare(*fixture(roll=False))
    assert lateral_only["translation_context"] and not lateral_only["translation_roll_context"]


def test_featureless_background_and_global_change_do_not_qualify():
    record, frames = fixture()
    flat = {i: np.full_like(rgb, 128) for i, rgb in frames.items()}
    assert compare(record, flat)["status"] == "background_registration_unassessed"
    moved = dict(frames)
    for i in record["centers"]["during"]:
        moved[i] = cv2.warpAffine(moved[i], np.asarray(((1, 0, 6), (0, 1, 5)), dtype=np.float32), (320, 180))
    result = compare(record, moved)
    assert result["status"] in {"background_motion_measured", "background_registration_unassessed"}
    assert not result.get("translation_context")
