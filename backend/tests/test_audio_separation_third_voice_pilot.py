import numpy as np
import pytest

from scripts.audio_separation_third_voice_pilot import matches, signal_quality


def test_separation_assignment_recovers_permuted_sources_at_native_alignment():
    times = np.arange(24000) / 8000
    sources = [np.sin(2 * np.pi * hz * times) for hz in (100, 211, 329)]
    mixture = sources[0] + sources[1] + .2 * sources[2]
    result, extra = matches(mixture, sources, [sources[2] * -.1, sources[0], sources[1] * .5])
    assert extra == 0
    assert all(row["si_sdr_db"] > 80 and row["gain_db"] > 3 for row in result)
    assert signal_quality(np.roll(sources[0], 20), sources[0]) < 0


def test_unmatched_output_is_not_credited_as_a_third_known_voice():
    times = np.arange(24000) / 8000
    a, b = [np.sin(2 * np.pi * hz * times) for hz in (100, 211)]
    mix = a + b
    _, extra = matches(mix, [a, b], [a, b, .12 * mix])
    assert extra == pytest.approx(.12)
    with pytest.raises(ValueError, match="measurable waveform"):
        signal_quality(a, np.zeros_like(a))
