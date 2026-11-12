"""Authored controls for the model-free overlap evidence sample grid."""

import numpy as np

from scripts.audio_harmonic_overlap import RATE, SAMPLES, measure


def pitched(frequency):
    time = np.arange(SAMPLES) / RATE
    return sum(4000 / order**.65 * np.sin(2 * np.pi * order * frequency * time + order * .39)
               for order in range(1, 10))


def pcm(values):
    return np.rint(values).astype("<i2").tobytes()


def test_localizes_two_continuous_harmonic_patterns_at_their_sample_range():
    second = pitched(223)
    second[:RATE] = 0
    second[2 * RATE:] = 0
    (interval,) = measure(pcm(pitched(142) + second))
    assert abs(interval[0] - RATE) <= 320
    assert abs(interval[1] - 2 * RATE) <= 320
    assert (interval[0] - 160) % 320 == (interval[1] - 160) % 320 == 0


def test_single_pattern_and_broadband_noise_do_not_establish_a_pair():
    single = pitched(142)
    rng = np.random.default_rng(118)
    assert measure(pcm(single)) == []
    assert measure(pcm(single + rng.normal(0, 350, len(single)))) == []
    assert measure(pcm(rng.normal(0, 350, len(single)))) == []
