import numpy as np

from scripts import audio_decoder_peak_transfer as peaks


def test_local_contrast_recovers_a_tone_pair_under_broadband_noise():
    rate = peaks.RATE
    seconds = np.arange(6 * rate) / rate
    noise = np.random.default_rng(87).normal(0, .04, len(seconds))
    tones = .01 * (np.sin(2 * np.pi * 1500 * seconds) + np.sin(2 * np.pi * 3000 * seconds))
    start, stop = 2 * rate, 2 * rate + peaks.WIDTH
    signal = noise.copy()
    signal[start:stop] += tones[start:stop]
    raw = np.rint(signal * 32768).astype("<i2").tobytes()
    readings = peaks.readouts(raw)
    assert readings == peaks.readouts(raw, project=True)
    score, location, frequencies = peaks.point_measure(readings, "paired_contrast")
    assert score > 7
    assert location == [start, stop]
    assert set(frequencies) == {1500, 3000}


def test_continuous_tones_keep_absolute_support_but_lose_local_contrast():
    rate = peaks.RATE
    seconds = np.arange(6 * rate) / rate
    noise = np.random.default_rng(81).normal(0, .04, len(seconds))
    tones = .01 * (np.sin(2 * np.pi * 1500 * seconds) + np.sin(2 * np.pi * 3000 * seconds))
    raw = np.rint((noise + tones) * 32768).astype("<i2").tobytes()
    readings = peaks.readouts(raw)
    assert readings == peaks.readouts(raw, project=True)
    absolute, _, _ = peaks.point_measure(readings, "absolute")
    contrast, _, _ = peaks.point_measure(readings, "paired_contrast")
    assert absolute > 14
    assert contrast < 4
