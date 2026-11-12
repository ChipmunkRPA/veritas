# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.image.faces import Face
from app.detectors.video import face_return
from app.detectors.video.adapters import face_appearance
from app.detectors.video.adapters.face_sequence import SequenceInputs
from app.detectors.video.decode import FrameTable
from app.engine.detector import DetectorResult, UnitScore
from scripts import video_face_return_transfer as trial


def table_for(rate):
    frame_count = rate * 6
    return FrameTable(f"1/{rate * 100}", tuple(300 + i * 100 for i in range(frame_count)), (0,),
                      "I" * frame_count, 1920, 1080, 1920, 1080)


@pytest.mark.parametrize("rate", [25, 30])
@pytest.mark.parametrize("control", ["steady", "return", "scene_cut", "unmeasured_edge"])
def test_candidate_and_clock_match_frozen_rank_arm(rate, control):
    table = table_for(rate)
    times, indices = face_return.schedule(table)
    _, reference_indices, _ = trial.schedules(times)
    assert list(indices) == reference_indices
    rng = np.random.default_rng(82)
    original = rng.integers(0, 256, (96, 144, 3), dtype=np.uint8)
    changed = rng.integers(0, 256, original.shape, dtype=np.uint8)
    features = {}
    reference = {}
    face = Face(52, 31, 35, 35, .99)
    for index in indices:
        middle = Fraction(11, 4) <= times[index] < Fraction(25, 8)
        pixels = changed if control == "scene_cut" and middle else original
        measured = face if control != "unmeasured_edge" or index != indices[24] else None
        vector = None if measured is None else np.eye(2, 128)[int(control != "steady" and middle)]
        features[index] = face_return.Sample(measured, vector, face_return.scene(pixels, measured))
        reference[index] = {"face": measured, "vector": vector, "scene": trial.scene(pixels, measured)}
        assert np.array_equal(features[index].scene[0], reference[index]["scene"][0])
        assert np.array_equal(features[index].scene[1], reference[index]["scene"][1])
    candidates, _ = trial.candidate_windows(times, reference_indices, reference)
    expected = candidates["rank"]
    actual = face_return.highest(times, indices, features)
    assert (actual is None) == (expected is None)
    if actual is not None:
        assert actual.response.hex() == expected["score"].hex()
        assert actual.scene_correlations == tuple(expected["scene_correlations"])
        assert (str(times[actual.start]), str(times[actual.end])) == (expected["start"], expected["end"])
        assert list(actual.anchors().values()) == [item["index"] for item in expected["anchors"]]


def test_scope_and_exclusive_overlap_are_exact():
    table = table_for(25)
    assert face_return.schedule(table) is not None
    assert face_return.schedule(replace(table, rotation=90)) is None
    assert face_return.schedule(replace(table, interpolated=(1,))) is None
    assert face_return.schedule(replace(table, pts=(*table.pts[:-1], table.pts[-1] + 1))) is None
    assert face_return.schedule(table_for(24)) is None
    assert face_return.schedule(replace(table, width=1280, height=720)) == face_return.schedule(table)
    assert face_return.schedule(replace(table_for(30), width=1280, height=720)) is None
    assert face_return.schedule(replace(table, width=640, height=360)) == face_return.schedule(table)
    assert face_return.schedule(replace(table_for(30), width=640, height=360)) is None
    assert face_return.schedule(replace(table, width=854, height=480)) is None
    candidate = face_return.Range(10, 19, .2, (1., 1.), ((0, 1, 2), (10, 12, 15), (19, 22, 25)), (8, 10, 17, 19))
    assert face_return.distinct(candidate, table, None)
    assert not face_return.distinct(candidate, table, {"start_pts": table.pts[10], "end_pts": table.pts[28]})
    assert face_return.distinct(candidate, table, {"start_pts": table.pts[10], "end_pts": table.pts[29]})
    assert not face_return.distinct(replace(candidate, response=.1), table, None)
    assert not face_return.distinct(replace(candidate, end=23), table, None)


def test_optional_review_is_applicable_without_coarse_inputs(monkeypatch):
    monkeypatch.setattr(face_appearance.settings, "face_min_size", 48)
    ctx = SimpleNamespace(outputs=SimpleNamespace(face_review=SequenceInputs(table_for(25), None, (), ())))
    adapter = face_appearance.FaceAppearanceReviewDetector()
    monkeypatch.setattr(face_appearance.settings, "video_face_short_return_review", False)
    assert not adapter.applicable(ctx).ok
    monkeypatch.setattr(face_appearance.settings, "video_face_short_return_review", True)
    assert adapter.applicable(ctx).ok


def test_secondary_only_result_requires_a_measured_selected_range(monkeypatch):
    monkeypatch.setattr(face_appearance.settings, "video_face_short_return_review", True)
    result = DetectorResult.not_applicable("coarse review lacked context")
    monkeypatch.setattr(face_appearance, "_inspect_return", lambda *args: None)
    assert face_appearance._with_return(result, None, None, None, None) is result
    assert result.aggregate is None and result.status == "not_applicable"

    record = {"start_stream_s": 2.88, "end_stream_s": 3.24, "response": .35}
    monkeypatch.setattr(face_appearance, "_inspect_return", lambda details, *args: details.update(face_return=record))
    selected = face_appearance._with_return(result, None, None, None, None)
    assert selected.status == "ok" and selected.aggregate == UnitScore("global", .35, "score")
    assert selected.details["review_only"] and "2.880 to 3.240" in selected.observed


def test_secondary_uses_existing_aggregate_when_incumbent_is_present(monkeypatch):
    monkeypatch.setattr(face_appearance.settings, "video_face_short_return_review", True)
    current = DetectorResult(status="ok", aggregate=UnitScore("global", .15, "score"), observed="Incumbent range.",
                             details={"face_sequence": {"start_pts": 1, "end_pts": 2}})
    record = {"start_stream_s": 2.88, "end_stream_s": 3.24, "response": .35}
    monkeypatch.setattr(face_appearance, "_inspect_return", lambda details, *args: details.update(face_return=record))
    selected = face_appearance._with_return(current, None, None, None, None)
    assert selected is current and selected.aggregate.raw == .15
    assert selected.observed.startswith("Incumbent range.") and "2.880 to 3.240" in selected.observed
