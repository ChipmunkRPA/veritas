import numpy as np
import pytest

from scripts.audio_frequency_separation import RATE, SAMPLES, apply_projection, separate, supplement


def test_frequency_projection_keeps_exact_clock_and_can_reduce_a_known_interferer():
    rng = np.random.default_rng(7147)
    clock = np.arange(SAMPLES) / RATE
    foreground_envelope = 0.08 + (0.5 + 0.5 * np.cos(2 * np.pi * 1.83 * clock)) ** 4
    noise_envelope = 0.08 + (0.5 + 0.5 * np.sin(2 * np.pi * 2.61 * clock)) ** 4
    center = 0.08 * rng.normal(size=SAMPLES) * foreground_envelope
    noise = 0.11 * rng.normal(size=SAMPLES) * noise_envelope
    foreground = np.column_stack((center, center))
    background = np.column_stack((noise, -0.21 * np.roll(noise, 3)))

    status, projection = separate(foreground + background)
    assert status == "available"
    assert projection is not None
    target = apply_projection(foreground, projection.coefficients)
    residual = apply_projection(background, projection.coefficients)
    assert projection.wave.shape == (SAMPLES,)
    np.testing.assert_allclose(projection.wave, target + residual, atol=1e-6, rtol=0)
    assert np.linalg.norm(target) / np.linalg.norm(residual) > (
        np.linalg.norm(center) / np.linalg.norm(background.mean(axis=1))
    )


def test_frequency_projection_rejects_dual_mono_and_incomplete_stereo():
    samples = np.random.default_rng(19).normal(0, 0.1, size=SAMPLES)
    dual_mono = np.column_stack((samples, samples))
    assert separate(dual_mono) == ("rank_deficient", None)
    assert separate(dual_mono[:-1]) == ("invalid_stereo", None)


@pytest.mark.parametrize("current", ["ai", "original", "unassessed"])
def test_supplement_keeps_existing_or_unassessed_direction(current):
    assert supplement(current, "ai") == current


@pytest.mark.parametrize("second", ["original", "inconclusive", "unassessed"])
def test_supplement_requires_an_ai_second_opinion(second):
    assert supplement("inconclusive", second) == "inconclusive"


def test_supplement_can_recover_an_inconclusive_direction():
    assert supplement("inconclusive", "ai") == "ai"
