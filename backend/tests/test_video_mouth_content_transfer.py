from fractions import Fraction

import numpy as np

from scripts import review_video_mouth_content_transfer as audit
from scripts import video_mouth_content_transfer as experiment


def test_speech_spectral_shape_is_invariant_to_constant_gain():
    time = np.arange(5 * experiment.HOP) / experiment.RATE
    wave = (np.rint(1000 * np.sin(2 * np.pi * 532 * time) + 500 * np.sin(2 * np.pi * 1090 * time)) * 2).astype("<i2")
    shape, level = experiment.acoustic(wave, 5)
    louder_shape, louder_level = experiment.acoustic(wave * 2, 5)
    assert np.allclose(louder_level, level * 2)
    assert np.allclose(louder_shape, shape)


def test_paired_readout_requires_measured_samples_on_both_sides():
    values = np.ones(64)
    values[20:44] = 5
    available = np.ones(len(values), dtype=bool)
    measured = experiment.best_range(values, available)
    result = measured["best"]
    assert measured["compared"] > 0 and result["score"] == 4
    assert result["medians"] == [1, 5, 1]
    before, during, after = result["context_ticks"]
    lo, hi = result["extent_ticks"]
    assert before < lo <= during < hi <= after
    available[26:] = False
    assert experiment.best_range(values, available)["best"]["score"] <= 0
    assert experiment.best_range(values, np.zeros(64, dtype=bool)) == {"compared": 0, "best": None}


def test_primary_readout_and_independent_overlap_agree_at_half_open_boundary():
    for predicted in ([4, 8], [4, 10], [7, 11], [8, 12]):
        truth = [4, 12]
        assert audit.overlaps(predicted, truth) == (experiment.prior.overlap(predicted, truth) >= Fraction(1, 2))
