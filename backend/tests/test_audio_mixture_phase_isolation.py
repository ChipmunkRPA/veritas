import numpy as np

from scripts import audio_mixture_phase_isolation as experiment


def _tones():
    time = np.arange(experiment.SAMPLES, dtype=np.float32) / experiment.RATE
    return np.sin(2 * np.pi * 310 * time), np.sin(2 * np.pi * 1201 * time)


def test_stitch_uses_overlap_to_check_speaker_order_and_polarity():
    first, second = _tones()
    width = 4 * experiment.RATE
    windows = iter((np.stack((first[:width], second[:width])),
                    np.stack((-second[-width:], first[-width:]))))
    streams, correlations = experiment.select_streams(first + second, lambda _: next(windows))
    np.testing.assert_allclose(streams, [first, second], rtol=0, atol=2e-7)
    np.testing.assert_allclose(correlations, [1, -1], rtol=0, atol=2e-7)


def test_stitch_declines_when_one_stream_cannot_be_matched():
    first, second = _tones()
    width = 4 * experiment.RATE
    windows = iter((np.stack((first[:width], second[:width])),
                    np.stack((first[-width:], np.zeros(width, np.float32)))))
    streams, correlations = experiment.select_streams(first + second, lambda _: next(windows))
    assert streams == []
    assert min(map(abs, correlations)) == 0


def test_masks_preserve_original_clock_and_mix_even_where_estimates_are_silent():
    first, second = _tones()
    mixture = first + second
    estimates = [first.copy(), second.copy()]
    for view in estimates:
        view[32000:48000] = 0
    sources = experiment.mixture_phase(mixture, estimates)
    np.testing.assert_allclose(sources[0] + sources[1], mixture, rtol=0, atol=1e-6)
    np.testing.assert_allclose(sources[0][33000:47000], mixture[33000:47000] / 2, rtol=0, atol=1e-6)
    assert np.linalg.norm(sources[0][:31000] - first[:31000]) < 1
