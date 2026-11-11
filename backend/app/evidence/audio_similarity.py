"""Bounded repetition context from a pair of similar decoded audio ranges."""
from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.detectors.audio.decode import source_ranges, wav_bytes
from app.evidence import audio_similarity_binding as binding

_FILTERS = tuple(
    butter(binding.FILTER_ORDER, band, fs=binding.SAMPLE_RATE, btype="bandpass", output="sos")
    for band in binding.BANDS
)


@dataclass(frozen=True)
class SimilarSpan:
    first: tuple[int, int]
    second: tuple[int, int]


@dataclass(frozen=True)
class SimilarityResult:
    status: Literal["measured", "unavailable", "ambiguous"]
    reason: str | None = None
    spans: tuple[SimilarSpan, ...] = ()


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


def _lag_peaks(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nominate lags using the frozen linear autocorrelation used in the transfer test."""
    centered = features - features.mean()
    size = len(centered)
    fft_size = 1 << (2 * size - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    products = np.fft.irfft(spectrum * spectrum.conjugate(), n=fft_size)
    total, energy = _prefix(centered), _prefix(centered * centered)
    sample_count = size + 1
    lags = np.arange(binding.MIN_REPEAT_SAMPLES, sample_count - binding.MIN_REPEAT_SAMPLES + 1)
    overlap = size - lags
    first_sum, second_sum = total[overlap], total[size] - total[lags]
    first_energy = np.maximum(energy[overlap] - first_sum**2 / overlap, 0)
    second_energy = np.maximum(energy[size] - energy[lags] - second_sum**2 / overlap, 0)
    covariance = products[lags] - first_sum * second_sum / overlap
    scale = np.sqrt(first_energy * second_energy)
    active = np.minimum(first_energy, second_energy) >= overlap * binding.MIN_LAG_RMS**2
    scores = np.divide(covariance, scale, out=np.zeros_like(scale), where=active)
    scores = np.clip(scores, -1, 1)
    previous = np.concatenate(([-np.inf], scores[:-1]))
    following = np.concatenate((scores[1:], [-np.inf]))
    peaks = (scores >= previous) & (scores > following) & (scores >= binding.MIN_LAG_PEAK)
    return lags[peaks], scores[peaks]


def _window_stats(total: np.ndarray, energy: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sums = total[starts + binding.WINDOW] - total[starts]
    centered_energy = energy[starts + binding.WINDOW] - energy[starts] - sums * sums / binding.WINDOW
    return sums, np.maximum(centered_energy, 0)


def find_similar_spans(pcm_s16le: bytes) -> SimilarityResult:
    """Search at most 24 candidate lags in continuous 2.5–4 s of mono PCM16.

    Each verified window must match at least two active frequency bands. The
    paired ranges describe acoustic similarity and supply no origin decision.
    """
    if not isinstance(pcm_s16le, bytes) or len(pcm_s16le) % 2:
        raise ValueError("Expected complete signed 16-bit samples")
    sample_count = len(pcm_s16le) // 2
    if not binding.MIN_SAMPLES <= sample_count <= binding.MAX_SAMPLES:
        return SimilarityResult("unavailable", "duration_out_of_scope")
    raw = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float64)
    features = [sosfiltfilt(filt, raw) for filt in _FILTERS]
    active = sum(np.sqrt(np.mean((band - band.mean()) ** 2)) >= binding.MIN_BAND_RMS for band in features)
    if active < binding.MIN_SUPPORTING_BANDS:
        return SimilarityResult("unavailable", "insufficient_active_bands")

    nominations = []
    for band in features:
        peaks, scores = _lag_peaks(band[:-1])
        eligible = np.lexsort((peaks, -scores))[:binding.LAGS_PER_BAND]
        nominations.extend((float(scores[index]), int(peaks[index])) for index in eligible)
    lags = []
    for _, lag in sorted(nominations, key=lambda item: (-item[0], item[1])):
        if all(abs(lag - previous) > binding.LAG_MERGE_SAMPLES for previous in lags):
            lags.append(lag)

    stats = [(band, _prefix(band), _prefix(band * band)) for band in features]
    window, hop = binding.WINDOW, binding.HOP
    spans = []
    for lag in lags:
        starts = np.arange(0, sample_count - lag - window + 1, hop)
        support = np.zeros(len(starts), dtype=np.uint8)
        for band, total, energy in stats:
            first_sum, first_energy = _window_stats(total, energy, starts)
            second_sum, second_energy = _window_stats(total, energy, starts + lag)
            products = _prefix(band[:-lag] * band[lag:])
            covariance = products[starts + window] - products[starts] - first_sum * second_sum / window
            denominator = np.sqrt(first_energy * second_energy)
            correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
            support += (correlation >= binding.MIN_CORRELATION) & (
                np.minimum(first_energy, second_energy) >= window * binding.MIN_BAND_RMS**2
            )
        transitions = np.flatnonzero(np.diff(np.r_[False, support >= binding.MIN_SUPPORTING_BANDS, False]))
        for start, end in zip(transitions[::2], transitions[1::2], strict=True):
            windows = min(int(end - start), 1 + (lag - window) // hop)
            count = window + (windows - 1) * hop
            if count >= binding.MIN_REPEAT_SAMPLES:
                low = int(starts[start])
                spans.append(SimilarSpan((low, low + count), (low + lag, low + lag + count)))
                if len(spans) > binding.MAX_PAIRS:
                    return SimilarityResult("ambiguous", "span_budget")
    return SimilarityResult("measured", spans=tuple(spans))


def build(pcm, mapping, source, make_asset):
    """Freeze the two actual source clips and bind each to its own sample hash."""
    samples = len(pcm)
    if (mapping["source_sha256"] != source["sha256"] or mapping["pcm_sha256"] != source["pcm_sha256"]
            or mapping["samples"] != samples or mapping["sample_rate"] != binding.SAMPLE_RATE
            or source["timeline"] != mapping["timeline"] or source["timeline_time_base"] != mapping["timeline_time_base"]):
        raise ValueError("Repeated-audio similarity input differs from the frozen source")
    eligible = binding.MIN_SAMPLES <= samples <= binding.MAX_SAMPLES
    context = {
        "schema": binding.SCHEMA,
        "evidence_type": "context",
        "affects_overall_confidence": False,
        "interpretation": binding.INTERPRETATION,
        "source_sha256": source["sha256"],
        "pcm_sha256": source["pcm_sha256"],
        "sample_rate": binding.SAMPLE_RATE,
        "input_samples": samples,
        "method": deepcopy(binding.METHOD),
        "searched_interval": None,
        "status": "unavailable",
        "reason": "source_clock_discontinuous" if eligible else "duration_out_of_scope",
        "pairs": [],
    }
    if eligible and len(mapping["timeline"]) == 1:
        pcm = np.asarray(pcm, dtype="<f4")
        if pcm.ndim != 1 or hashlib.sha256(pcm.tobytes()).hexdigest() != source["pcm_sha256"]:
            raise ValueError("Repeated-audio similarity samples differ from the frozen source")
        scaled = pcm * 32768.0
        if (not np.isfinite(scaled).all() or scaled.min() < -32768 or scaled.max() > 32767
                or not np.array_equal(scaled, np.rint(scaled))):
            raise ValueError("Expected the exact decoded signed 16-bit input")
        payload = scaled.astype("<i2").tobytes()
        result = find_similar_spans(payload)
        context.update(searched_interval={"start_sample": 0, "end_sample": samples}, status=result.status, reason=result.reason)
        for index, span in enumerate(result.spans, 1):
            count = span.first[1] - span.first[0]
            pair = {"id": f"RS{index:02d}", "sample_count": count, "lag_samples": span.second[0] - span.first[0],
                    "support_windows": 1 + (count - binding.WINDOW) // binding.HOP, "artifacts": {}}
            for role, (start, end) in (("first", span.first), ("second", span.second)):
                pair[role] = {"start_sample": start, "end_sample": end,
                              "source_timestamp_ranges": source_ranges(mapping, start, end),
                              "pcm_s16le_sha256": hashlib.sha256(payload[2 * start:2 * end]).hexdigest()}
                pair["artifacts"][f"{role}_clip"] = make_asset(wav_bytes(pcm[start:end]), "audio/wav", samples=count,
                                                              sample_rate=binding.SAMPLE_RATE, channels=1)
            context["pairs"].append(pair)
    binding.validate(context, source)
    return context


def verify(context, *, pcm, mapping, source, read_asset):
    """Replay the measured ranges and compare both WAVs to the frozen PCM."""
    binding.validate(context, source)

    def checked_asset(data, media_type, **metadata):
        sha = hashlib.sha256(data).hexdigest()
        if read_asset(sha) != data:
            raise ValueError("Repeated-audio similarity playback clip differs from its source samples")
        return {"sha256": sha, "media_type": media_type, **metadata}

    expected = build(pcm, mapping, source, checked_asset)
    recorded = deepcopy(context)
    for pair in recorded["pairs"]:
        for clip in pair["artifacts"].values():
            clip.pop("url", None)
    if expected != recorded:
        raise ValueError("Repeated-audio similarity differs from the measured source input")
