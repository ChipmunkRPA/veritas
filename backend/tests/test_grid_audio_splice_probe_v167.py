import numpy as np
import pytest

from scripts.grid_audio_splice_probe_v167 import checked_history, pcm_digest, render, seam_jumps


@pytest.fixture
def source():
    rng = np.random.default_rng(29)
    return rng.integers(-30000, 30001, 47648, dtype=np.int16).astype(np.float32) / 32768


@pytest.fixture
def support():
    return {
        "status": "ready",
        "shift_samples": 5120,
        "whole_input_reencoding_required": True,
        "preserve_length_origins_and_outside_samples": True,
        "padding_or_wraparound": False,
        "output_samples_half_open": [17280, 30080],
        "source_samples_half_open": [12160, 24960],
    }


def test_hard_copy_is_an_earlier_source_copy_and_preserves_other_samples(source, support):
    hard = render(source, support, 0)
    assert np.array_equal(hard[:17280], source[:17280])
    assert np.array_equal(hard[17280:30080], source[12160:24960])
    assert np.array_equal(hard[30080:], source[30080:])
    assert pcm_digest(hard) != pcm_digest(source)


@pytest.mark.parametrize("width", [320, 800])
def test_fade_preserves_source_edge_jumps_and_exact_shifted_interior(source, support, width):
    original = source.copy()
    faded = render(source, support, width)
    hard = render(source, support, 0)
    start, end = support["output_samples_half_open"]
    assert np.array_equal(source, original)
    assert np.array_equal(faded[: start + 1], source[: start + 1])
    assert np.array_equal(faded[end - 1 :], source[end - 1 :])
    assert np.array_equal(faded[start + width : end - width], hard[start + width : end - width])
    assert seam_jumps(faded, support) == seam_jumps(source, support)
    assert pcm_digest(faded) not in {pcm_digest(source), pcm_digest(hard)}


def test_historical_parity_rejects_changed_window_score_or_direction():
    baseline = {"status": "ok", "raw": 0.25, "decision": "inconclusive", "windows": [[0, 47648]]}
    assert checked_history(baseline, baseline) == 0
    with pytest.raises(ValueError, match="availability, selected windows or direction"):
        checked_history(baseline | {"decision": "original"}, baseline)
    with pytest.raises(ValueError, match="raw score"):
        checked_history(baseline | {"raw": 0.251}, baseline)
    unavailable = {"status": "not_applicable", "raw": None, "decision": "unavailable", "windows": []}
    assert checked_history(unavailable, unavailable) is None


def test_crossfade_rounds_fractional_negative_sample_to_canonical_zero(support):
    source = np.zeros(47648, dtype="<f4")
    source[12161] = -1 / 32768
    faded = render(source, support, 320)
    assert faded[17281] == 0 and not np.signbit(faded[17281])
