from fractions import Fraction

import numpy as np
import pytest

from app.detectors.image.faces import Face
from scripts import video_face_return_transfer as trial


def observations(*, cut=False, scene_available=True):
    times = [Fraction(i, 8) for i in range(20)]
    face = Face(20, 22, 100, 110, .95)
    scene_a = np.broadcast_to([1., 0.], (32, 2))
    scene_b = np.broadcast_to([0., 1.], (32, 2))
    mask = np.full(32, scene_available)
    frames = {}
    for i in range(len(times)):
        middle = 8 <= i < 12
        frames[i] = {"vector": np.asarray([0., 1.] if middle else [1., 0.]), "face": face,
                     "scene": (scene_b if cut and middle else scene_a, mask)}
    return times, frames


def test_distribution_margin_requires_multiple_supported_faces():
    left, core, right = list(range(4)), list(range(4, 8)), list(range(8, 12))
    faces = np.tile([1., 0.], (12, 1))
    faces[core] = [0., 1.]
    assert trial.distribution_margin(1 - faces @ faces.T, left, core, right) == 1.
    faces[5:8] = [1., 0.]
    assert trial.distribution_margin(1 - faces @ faces.T, left, core, right) <= 0.


def test_half_second_same_shot_return_has_exact_time_anchors():
    times, frames = observations()
    arms, coverage = trial.candidate_windows(times, list(range(len(times))), frames)
    result = arms["rank"]
    assert result["score"] == pytest.approx(1.)
    assert (result["start"], result["end"]) == ("1", "3/2")
    assert [anchor["time"] for anchor in result["anchors"]] == ["7/8", "5/4", "3/2"]
    assert coverage["guarded"] == 0


def test_scene_cut_rejects_exact_nonface_discontinuity():
    times, frames = observations(cut=True)
    arms, coverage = trial.candidate_windows(times, list(range(len(times))), frames)
    assert arms["rank_without_guard"]["score"] == 1.
    assert arms["rank"]["score"] < .5
    assert coverage["guarded"] > 0


def test_missing_nonface_tiles_are_unassessed_for_guarded_arm():
    times, frames = observations(scene_available=False)
    arms, coverage = trial.candidate_windows(times, list(range(len(times))), frames)
    assert arms["rank_without_guard"] is not None
    assert arms["rank"] is None
    assert coverage["scene_unavailable"] == coverage["windows"] > 0


def test_scene_tiles_ignore_broad_luminance_offset_and_flat_scene():
    rng = np.random.default_rng(45)
    pixels = rng.integers(15, 160, (64, 96, 3), dtype=np.uint8)
    first = trial.scene(pixels, None)
    second = trial.scene(pixels + 60, None)
    assert trial.scene_correlation(first, second) == pytest.approx(1.)
    assert trial.scene_correlation(first, trial.scene(np.full_like(pixels, 50), None)) is None


def test_point_and_localization_outcomes_are_distinct():
    finding = {"start": "1", "end": "3/2", "score": .7}
    assert trial.classify(finding, .7, ["1", "3/2"]) == "measured_no_flag"
    assert trial.classify(finding, .5, ["1", "3/2"]) == "localized"
    assert trial.classify(finding, .5, ["3/2", "2"]) == "wrong_interval"
    assert trial.classify(finding, .5, None) == "false_flag"
    assert trial.classify(None, .5, None) == "unassessed"
