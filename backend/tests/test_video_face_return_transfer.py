from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from app.detectors.image.faces import Face
from app.detectors.video import face_local_detail
from app.detectors.video.decode import DecodedFrame, DecodeError, FrameTable
from app.evidence import face_portrait_return_binding
from app.evidence.face_local_return_binding import PORTRAIT
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


def test_dense_face_review_checks_one_picture_cuts_and_native_timestamps(monkeypatch, tmp_path):
    table = FrameTable(time_base="1/25", pts=tuple(range(100)), key_frames=(0,), pict_types="I" * 100,
                       width=64, height=48, coded_width=64, coded_height=48, rotation=0, interpolated=())
    red = np.full((48, 64, 3), (255, 0, 0), dtype=np.uint8)
    blue = np.full_like(red, (0, 0, 255))

    def install(verified):
        def decode(path, table, indices, **kwargs):
            assert list(indices) == list(range(12, 47))
            for i in indices:
                # The first support picture begins a new shot. Only the later cuts cross the support.
                yield DecodedFrame(i, table.pts[i], blue if i == 21 else red, pts_verified=verified or i != 21)

        monkeypatch.setattr(face_local_detail, "decode_frames", decode)

    install(True)
    assert face_local_detail.shot_changes(tmp_path / "source.mp4", table, 20, 40) == (21, 22)
    install(False)
    with pytest.raises(DecodeError, match="source clock"):
        face_local_detail.shot_changes(tmp_path / "source.mp4", table, 20, 40)


def test_portrait_face_review_requires_exact_upright_video_presentation():
    table = FrameTable(time_base="1/30000", pts=tuple(1000 * i for i in range(120)), key_frames=(0,),
                       pict_types="I" * 120, width=360, height=640, coded_width=360, coded_height=640,
                       rotation=0, interpolated=())
    encoding = "h264", "yuv420p"
    assert face_portrait_return_binding.supported(table, encoding)
    assert not face_local_detail.supported(table, encoding)
    assert not face_portrait_return_binding.supported(replace(table, time_base="1/30001"), encoding)
    assert not face_portrait_return_binding.supported(replace(table, rotation=90), encoding)
    assert not face_portrait_return_binding.supported(table, ("hevc", "yuv420p"))
    hevc = ("hevc", "yuv420p", "hvc1", "Main")
    assert face_portrait_return_binding.hevc_supported(table, hevc)
    assert not face_portrait_return_binding.hevc_supported(table, ("hevc", "yuv420p", "hev1", "Main"))
    assert not face_portrait_return_binding.hevc_supported(table, ("hevc", "yuv420p10le", "hvc1", "Main 10"))
    assert not face_portrait_return_binding.hevc_supported(replace(table, pts=tuple(p + 1000 for p in table.pts)), hevc)
    assert not face_portrait_return_binding.hevc_supported(replace(table, rotation=90), hevc)
    assert not face_portrait_return_binding.hevc_supported(
        replace(table, width=720, height=1280, coded_width=720, coded_height=1280), hevc)
    before, during, after = PORTRAIT.support(20, 40)
    assert before == [10, 12, 14, 16, 18] and during == list(range(20, 40, 2)) and after == [40, 42, 44, 46, 48]
