import hashlib
from fractions import Fraction

import cv2
import numpy as np
import pytest

from app.detectors.image.faces import Face
from scripts import video_selected_face_boundary as candidate


def test_selection_replays_skipped_center_without_ranking():
    times = [Fraction(i, 32) for i in range(121)]
    rgb = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(len(times))]
    original = {"before": (8, 24), "during": (32, 40, 56, 64), "after": (72, 88)}
    row = {"original": {"start": "1", "end": "2"}, "selected_input_sha256": hashlib.sha256(
        "".join(rgb[i] for i in candidate.indices(original)).encode()).hexdigest()}
    assert candidate.exact_roles(times, row, rgb) == original
    row["selected_input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not uniquely identified"):
        candidate.exact_roles(times, row, rgb)


def test_boundary_gap_removes_gaps_measured_within_either_side():
    assert candidate.boundary_gap([1, 1, 1, 1]) == 0
    assert candidate.boundary_gap([.5, 1.5, 3.5, 7.5]) == 0
    assert candidate.boundary_gap([1, 1, 3, 3]) == pytest.approx(np.log(3.5 / 1.5))
    assert candidate.boundary_gap([3, 3, 1, 1]) == pytest.approx(-np.log(3.5 / 1.5))


def triplet():
    face = Face(75, 48, 100, 118, .98, [(106, 84), (145, 85), (126, 105), (111, 126), (144, 126)])
    rng = np.random.default_rng(543)
    picture = rng.integers(30, 222, (224, 248), dtype=np.uint8)
    picture = cv2.GaussianBlur(picture, (3, 3), .6)
    return [candidate.Picture(np.roll(picture, shift, axis=1), 0, 0, face) for shift in (-2, 0, 2)]


def test_native_local_tracking_explains_uniform_translation():
    rows, reason = candidate.center_measurement(triplet())
    assert reason is None
    assert sum(row["status"] == "measured" for row in rows.values()) >= 2
    assert all(abs(row["gap"]) < .05 for row in rows.values() if row["status"] == "measured")


def test_bad_geometry_and_absent_face_remain_unmeasured():
    frames = triplet()
    assert candidate.center_measurement([None, *frames[1:]]) == ({}, "single_face_missing")
    bad_face = Face(75, 48, 100, 118, .98, [(106, 84), (111, 84), (126, 105), (111, 126), (144, 126)])
    frames[1] = candidate.Picture(frames[1].gray, 0, 0, bad_face)
    assert candidate.center_measurement(frames) == ({}, "unstable_face_geometry")


def test_grid_concentration_does_not_count_as_local_evidence(monkeypatch):
    monkeypatch.setattr(candidate, "track", lambda _a, _b, points: (points, np.ones(len(points), bool)))
    monkeypatch.setattr(candidate, "residual", lambda *_args: (4., 2.))
    rows, reason = candidate.center_measurement(triplet())
    assert reason is None
    assert {row["status"] for row in rows.values()} == {"grid_concentrated"}


def observations(before, during, after, *, comparison=True):
    roles = {}
    for role, gaps in zip(candidate.ROLES, (before, during, after), strict=True):
        roles[role] = []
        for value in gaps:
            row = {"right_cheek": {"status": "measured", "gap": value}}
            if comparison:
                row["left_jaw"] = {"status": "measured", "gap": 0.}
            roles[role].append({"arcs": row})
    return roles


@pytest.mark.parametrize("sign", [1, -1])
def test_support_requires_persistence_beyond_both_flanks(sign):
    observation = observations([0, .02], [sign, sign, 0, sign, 0], [0, .06])
    status, arcs, supported = candidate.assess(observation)
    assert status == "supported" and supported == ["right_cheek"]
    assert arcs["right_cheek"]["support_direction"] == sign
    high_flank = observations([sign, sign], [sign, sign, sign], [0, 0])
    assert candidate.assess(high_flank)[0] == "measured_no_support"
    brief = observations([0, 0], [sign, 0, 0], [0, 0])
    assert candidate.assess(brief)[0] == "measured_no_support"


def test_insufficient_reliable_arcs_cannot_corroborate():
    one_arc = observations([0, 0], [1, 1, 1], [0, 0], comparison=False)
    status, _, supported = candidate.assess(one_arc)
    assert status == "unmeasured" and supported == []
