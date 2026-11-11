"""Authored PCM controls for the lab-only exact repetition cue."""

import numpy as np
import pytest

from scripts import audio_exact_repetition as repetition


def waveform(count: int = 48000) -> np.ndarray:
    return np.random.default_rng(20261110).integers(-18000, 18000, size=count, dtype=np.dtype("<i2"))


def test_non_grid_aligned_delay_and_exact_unmodified_samples():
    values = waveform()
    first, second, length = 4113, 17423, 4803
    values[second : second + length] = values[first : first + length]
    raw = values.tobytes()
    result = repetition.find_repeated_spans(raw)

    assert result.status == "measured"
    assert result.spans == (repetition.RepeatedSpan((first, first + length), (second, second + length)),)
    span = result.spans[0]
    assert span.lag_samples == second - first
    assert raw[2 * span.first[0] : 2 * span.first[1]] == raw[2 * span.second[0] : 2 * span.second[1]]
    assert result.anchor_pair_checks < repetition.MAX_ANCHOR_PAIR_CHECKS
    assert result.sample_comparisons < repetition.MAX_SAMPLE_COMPARISONS


def test_overwritten_donor_only_supports_the_surviving_nonoverlapping_witness():
    values = waveform()
    first, second, copy_length = 12160, 17280, 12800
    values[second : second + copy_length] = values[first : first + copy_length].copy()
    result = repetition.find_repeated_spans(values.tobytes())

    assert result.status == "measured"
    assert result.spans == (repetition.RepeatedSpan((12160, 17280), (17280, 22400)),)
    assert result.spans[0].sample_count == 5120 < copy_length


def test_single_altered_sample_splits_a_run_below_the_minimum():
    values = waveform()
    values[20000:24001] = values[4100:8101]
    values[22000] ^= 1
    result = repetition.find_repeated_spans(values.tobytes())

    assert result.status == "measured"
    assert not result.spans
    assert result.sample_comparisons > 0


@pytest.mark.parametrize("level", [0, 9000])
def test_constant_or_silent_pcm_is_unavailable(level):
    values = np.full(48000, level, dtype="<i2")
    assert repetition.find_repeated_spans(values.tobytes()) == repetition.RepetitionResult(
        "unavailable", "insufficient_signal_variation"
    )


def test_highly_repetitive_signal_hits_the_bounded_pair_budget():
    values = np.resize(waveform(127), repetition.MAX_SAMPLES)
    result = repetition.find_repeated_spans(values.tobytes())

    assert result.status == "ambiguous"
    assert result.reason == "anchor_pair_budget"
    assert result.anchor_pair_checks == repetition.MAX_ANCHOR_PAIR_CHECKS
    assert result.sample_comparisons == 0
    assert not result.spans


def test_extension_comparison_budget_stops_without_returning_a_partial_witness(monkeypatch):
    monkeypatch.setattr(repetition, "MAX_SAMPLE_COMPARISONS", 10)
    values = waveform()
    values[20000:24000] = values[4100:8100]
    result = repetition.find_repeated_spans(values.tobytes())

    assert result.status == "ambiguous"
    assert result.reason == "sample_comparison_budget"
    assert result.sample_comparisons == 10
    assert not result.spans


@pytest.mark.parametrize("sample_count", [repetition.MIN_SAMPLES - 1, repetition.MAX_SAMPLES + 1])
def test_duration_outside_the_supported_range_is_unavailable(sample_count):
    assert repetition.find_repeated_spans(waveform(sample_count).tobytes()) == repetition.RepetitionResult(
        "unavailable", "duration_out_of_scope"
    )


def test_partial_pcm_sample_is_rejected():
    with pytest.raises(ValueError, match="complete signed 16-bit samples"):
        repetition.find_repeated_spans(b"\x00")
