"""Regression coverage for the portable four-media review."""

from copy import deepcopy
from math import nextafter

import pytest

from scripts import replay_public_evaluations as replay
from scripts import review_writing_bidirectional_transfer as writing


def test_single_upload_readers_match_the_offline_linux_reference():
    reference = replay.load_receipt(replay.ROOT / replay.PANELS / "single-upload-cues-linux-reference-2026-11-13.json")
    actual = {**reference, "checks": {modality: replay.replay(modality, reference["suite"]) for modality in replay.CHECKS}}
    assert replay.compare(reference, actual)


def test_writing_replay_requires_exact_points_with_small_coefficient_drift(monkeypatch):
    data = writing.study.source.read(replay.ROOT / replay.PANELS / "writing-bidirectional-transfer-2026-11-13.json")
    calibrated = writing.study.calibrated

    def refit_with_platform_drift(*args):
        refit = calibrated(*args)
        coefficient = refit["exterior"]["parameters"][0]
        refit["exterior"]["parameters"][0] = nextafter(coefficient, 1.0)
        return refit

    monkeypatch.setattr(writing.study, "calibrated", refit_with_platform_drift)
    assert writing.review(data) == data["result"]

    changed_point = deepcopy(data)
    changed_point["points"]["exterior"]["point"] += 1e-12
    with pytest.raises(ValueError, match="calibration maxima, operating point or fit coverage differ"):
        writing.review(changed_point)

    changed_fit = deepcopy(data)
    changed_fit["points"]["exterior"]["parameters"][0] += 1e-12
    with pytest.raises(ValueError, match="Frozen coefficients differ beyond numerical tolerance"):
        writing.review(changed_fit)
