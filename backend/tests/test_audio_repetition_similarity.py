"""Authored short-PCM tests for the lab similarity cue."""

import numpy as np
import pytest

from app.evidence import audio_waveform_similarity as similarity


def signal(seed: int, count: int = 48000) -> np.ndarray:
    return np.random.default_rng(seed).integers(-13000, 13000, size=count, dtype=np.dtype("<i2"))


@pytest.mark.parametrize("lag", [5123, 13781])
def test_sustained_similarity_recovers_non_aligned_lag_after_perturbation(lag):
    pcm = signal(170)
    first, second, count = 4193, 4193 + lag, 4803
    donor = pcm[first : first + count].astype(np.float64)
    perturbation = signal(171, count) / 8
    pcm[second : second + count] = np.rint(0.72 * donor + perturbation + 900)
    assert pcm[first : first + count].tobytes() != pcm[second : second + count].tobytes()

    result = similarity.find_similar_spans(pcm.tobytes())

    assert result.status == "measured"
    (span,) = result.spans
    assert span.lag_samples == lag
    assert span.sample_count >= 0.85 * count
    assert span.first[1] <= span.second[0]
    assert span.second == tuple(index + lag for index in span.first)
    for observed, expected in zip(span.first, (first, first + count), strict=True):
        assert abs(observed - expected) < similarity.WINDOW
    assert span.minimum_similarity >= similarity.MIN_WINDOW_SIMILARITY
    assert span.windows >= 12


def test_unrelated_waveform_has_no_supported_pair():
    result = similarity.find_similar_spans(signal(172, similarity.MAX_SAMPLES).tobytes())
    assert result.status == "measured"
    assert not result.spans
    assert result.lags_checked <= similarity.MAX_LAGS


def test_repeated_low_energy_pattern_cannot_support_a_pair():
    pcm = signal(173)
    quiet = np.random.default_rng(174).integers(-5, 6, size=5000, dtype=np.dtype("<i2"))
    # Even an audible DC offset is insufficient activity for a sustained match.
    pcm[6000:11000] = quiet + 3000
    pcm[20000:25000] = quiet + 3000
    result = similarity.find_similar_spans(pcm.tobytes())
    assert result.status == "measured"
    assert not result.spans


@pytest.mark.parametrize("kind", ["pulse", "tone"])
def test_periodic_pcm_is_ambiguous(kind):
    if kind == "pulse":
        pcm = np.resize(signal(175, 127), 48000)
    else:
        t = np.arange(48000) / similarity.SAMPLE_RATE
        pcm = np.rint(12000 * np.sin(2 * np.pi * 190 * t)).astype("<i2")
    result = similarity.find_similar_spans(pcm.tobytes())
    assert result == similarity.SimilarityResult("ambiguous", "periodic_signal")


@pytest.mark.parametrize("level", [0, -12000])
def test_constant_pcm_has_insufficient_activity(level):
    pcm = np.full(48000, level, dtype="<i2")
    assert similarity.find_similar_spans(pcm.tobytes()) == similarity.SimilarityResult(
        "unavailable", "insufficient_signal_variation"
    )


def test_candidate_cap_is_reported_and_limits_actual_lag_work(monkeypatch):
    monkeypatch.setattr(similarity, "MAX_LAGS", 1)
    pcm = signal(176)
    pcm[10000:16000] = pcm[2000:8000]
    pcm[25000:28200] = pcm[18000:21200]
    result = similarity.find_similar_spans(pcm.tobytes())

    assert result.status == "measured"
    assert result.candidate_cap_hit and result.lags_checked == 1
    assert [span.lag_samples for span in result.spans] == [8000]


def test_rejects_a_partial_sample_and_reports_unsupported_duration():
    with pytest.raises(ValueError, match="complete signed16 PCM bytes"):
        similarity.find_similar_spans(b"\x00")
    assert similarity.find_similar_spans(
        bytes(2 * (similarity.MIN_SAMPLES - 1))
    ) == similarity.SimilarityResult("unavailable", "duration_out_of_scope")
