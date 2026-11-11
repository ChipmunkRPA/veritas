from fractions import Fraction

import numpy as np

from app.detectors.image.faces import Face
from scripts import video_selected_landmark_reliability as trial


def measurement(before, during, after):
    if before is None:
        return {"status": "unavailable", "reason": "inexact_role_coverage"}
    return {"status": "measured", "eye_box_width": {
        role: {"median": value} for role, value in zip(trial.ROLES, (before, during, after), strict=True)
    }}


def test_frozen_ratio_keeps_boundary_equality_and_reports_missing_geometry():
    assert trial.classify_measurement(measurement(.4, .3, .6))["status"] == "flag_retained"
    low = trial.classify_measurement(measurement(.4, .299, .6))
    assert low["status"] == "flag_withheld_low_support"
    missing = trial.classify_measurement(measurement(None, None, None))
    assert missing["status"] == "flag_retained_unavailable_geometry"
    assert missing["geometry_unavailable_reason"] == "inexact_role_coverage"


def test_research_selector_never_replaces_an_unsupported_first_window(monkeypatch):
    ids = [2, 7, 12, 17, 22, 27, 32]
    next_ids = [32, 37, 42, 47, 52, 57, 62]
    windows = [{"response": trial.POINT + .1, "centers": ids},
               {"response": trial.POINT + .05, "centers": next_ids}]
    monkeypatch.setattr(trial.common.face_appearance, "windows", lambda *_: windows)
    during = {i + k for i in ids[2:-2] for k in (-1, 0, 1)}
    inputs = {}
    for i in {j + k for j in ids + next_ids for k in (-1, 0, 1)}:
        d = 20 if i in during else 50
        inputs[i] = {"appearance": np.ones(2), "rgb_sha256": "a"*64,
                     "face": Face(0, 0, 100, 100, .9, [(20., 30.), (20.+d, 30.), (45., 55.), (30., 75.), (55., 75.)])}
    times = [Fraction(i, 25) for i in range(64)]
    actual = trial.selection(ids + next_ids, None, times, inputs)
    assert actual["status"] == "flag_withheld_low_support"
    assert actual["original"] == {"score": trial.POINT + .1, "start": str(times[12]), "end": str(times[22])}
    assert set(actual) == {"original", "status", "eye_width_medians", "selected_input_sha256", "geometry_unavailable_reason"}
