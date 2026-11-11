from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from scripts import audio_stereo_reference_suppression as method


def _signal():
    rng = np.random.default_rng(20261111)
    low = butter(2, [300, 3500], fs=method.SAMPLE_RATE, btype="bandpass", output="sos")
    speech = sosfiltfilt(low, rng.normal(0, 0.016, method.SAMPLES))
    noise = sosfiltfilt(low, rng.normal(0, 0.067, method.SAMPLES))
    stereo_speech = np.column_stack((speech, speech))
    stereo_noise = np.column_stack((noise, 0.15 * noise))
    return stereo_speech, stereo_noise


@pytest.mark.parametrize("arm", method.ARMS)
def test_frozen_reference_preserves_the_common_signal_and_is_channel_swap_invariant(arm):
    speech, noise = _signal()
    mixture = speech + noise
    mono = mixture.mean(axis=1)
    weights = method.fit(mono, mixture, arm)
    output = method.apply(mono, mixture, weights)
    swapped = mixture[:, ::-1]
    np.testing.assert_array_equal(output, method.apply(mono, swapped, method.fit(mono, swapped, arm)))
    np.testing.assert_allclose(method.apply(speech.mean(axis=1), speech, weights), speech[:, 0], atol=1e-16)
    residual = method.apply(noise.mean(axis=1), noise, weights)
    assert np.linalg.norm(residual) < 0.5 * np.linalg.norm(noise.mean(axis=1))
    np.testing.assert_allclose(output, speech[:, 0] + residual, atol=1e-16)


def test_rejects_unsupported_input_and_clocks():
    speech, _ = _signal()
    mono = speech.mean(axis=1)
    with pytest.raises(ValueError, match="clock"):
        method.fit(mono[:-1], speech, "whole_clip")
    with pytest.raises(ValueError, match="Unknown"):
        method.fit(mono, speech, "unknown")
