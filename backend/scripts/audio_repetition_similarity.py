"""Lab-only repeated-waveform similarity on native short PCM sample clocks.

The bounded search returns inspection ranges. Similarity does not establish
an edit, a full boundary, or the origin of speech.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np

SAMPLE_RATE = 16000
MIN_SAMPLES = 40000
MAX_SAMPLES = 64000
MIN_SUPPORT = 2400
WINDOW = 640
HOP = 160
PREEMPHASIS = 0.95
MIN_ACTIVITY_STD = 32
MIN_FEATURE_STD = 8
MIN_LAG_PEAK = 0.035
MIN_WINDOW_SIMILARITY = 0.82
LAG_PEAK_SEPARATION = 24
MAX_LAGS = 12
MAX_SPANS = 8
PERIODIC_PEAK = 0.94
RIVAL_SIMILARITY_DELTA = 0.04


@dataclass(frozen=True)
class SimilaritySpan:
    first: tuple[int, int]
    second: tuple[int, int]
    minimum_similarity: float
    median_similarity: float
    windows: int

    @property
    def lag_samples(self) -> int:
        return self.second[0] - self.first[0]

    @property
    def sample_count(self) -> int:
        return self.first[1] - self.first[0]


@dataclass(frozen=True)
class SimilarityResult:
    status: Literal["measured", "unavailable", "ambiguous"]
    reason: str | None = None
    spans: tuple[SimilaritySpan, ...] = ()
    lags_checked: int = 0
    candidate_cap_hit: bool = False


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


def _lag_peaks(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-centered linear autocorrelation on each active overlapping pair."""
    centered = features - features.mean()
    size = len(centered)
    fft_size = 1 << (2 * size - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    products = np.fft.irfft(spectrum * spectrum.conjugate(), n=fft_size)
    total, energy = _prefix(centered), _prefix(centered * centered)
    sample_count = size + 1
    lags = np.arange(MIN_SUPPORT, sample_count - MIN_SUPPORT + 1)
    overlap = size - lags
    first_sum, second_sum = total[overlap], total[size] - total[lags]
    first_energy = np.maximum(energy[overlap] - first_sum**2 / overlap, 0)
    second_energy = np.maximum(energy[size] - energy[lags] - second_sum**2 / overlap, 0)
    covariance = products[lags] - first_sum * second_sum / overlap
    scale = np.sqrt(first_energy * second_energy)
    active = np.minimum(first_energy, second_energy) >= overlap * MIN_FEATURE_STD**2
    scores = np.divide(covariance, scale, out=np.zeros_like(scale), where=active)
    scores = np.clip(scores, -1, 1)
    previous = np.concatenate(([-np.inf], scores[:-1]))
    following = np.concatenate((scores[1:], [-np.inf]))
    peaks = (scores >= previous) & (scores > following) & (scores >= MIN_LAG_PEAK)
    return lags[peaks], scores[peaks]


def _select_lags(peaks: np.ndarray, scores: np.ndarray) -> tuple[list[int], bool]:
    selected = []
    for index in np.lexsort((peaks, -scores)):
        lag = int(peaks[index])
        if any(abs(lag - chosen) < LAG_PEAK_SEPARATION for chosen in selected):
            continue
        if len(selected) == MAX_LAGS:
            return selected, True
        selected.append(lag)
    return selected, False


def _window_stats(
    total: np.ndarray, energy: np.ndarray, starts: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    sums = total[starts + width] - total[starts]
    centered_energy = energy[starts + width] - energy[starts] - sums * sums / width
    return sums, np.maximum(centered_energy, 0)


def _lag_spans(raw: np.ndarray, features: np.ndarray, lag: int) -> list[SimilaritySpan]:
    starts = np.arange(0, len(raw) - lag - WINDOW + 1, HOP)
    shifted = starts + lag
    raw_total, raw_energy = _prefix(raw), _prefix(raw * raw)
    _, active_first = _window_stats(raw_total, raw_energy, starts, WINDOW)
    _, active_second = _window_stats(raw_total, raw_energy, shifted, WINDOW)

    # Each feature uses two consecutive PCM samples inside the reported range.
    width = WINDOW - 1
    feature_total, feature_energy = _prefix(features), _prefix(features * features)
    first_sum, first_energy = _window_stats(feature_total, feature_energy, starts, width)
    second_sum, second_energy = _window_stats(feature_total, feature_energy, shifted, width)
    paired = _prefix(features[:-lag] * features[lag:])
    covariance = paired[starts + width] - paired[starts] - first_sum * second_sum / width
    scale = np.sqrt(first_energy * second_energy)
    similarity = np.divide(covariance, scale, out=np.zeros_like(scale), where=scale > 0)
    similarity = np.clip(similarity, -1, 1)
    supported = (
        (np.minimum(active_first, active_second) >= WINDOW * MIN_ACTIVITY_STD**2)
        & (np.minimum(first_energy, second_energy) >= width * MIN_FEATURE_STD**2)
        & (similarity >= MIN_WINDOW_SIMILARITY)
    )

    transitions = np.flatnonzero(np.diff(np.concatenate(([False], supported, [False]))))
    spans = []
    for first, stop in zip(transitions[::2], transitions[1::2], strict=True):
        count = min(int(stop - first), 1 + (lag - WINDOW) // HOP)
        support = WINDOW + (count - 1) * HOP
        if support < MIN_SUPPORT:
            continue
        a = int(starts[first])
        measured = similarity[first : first + count]
        spans.append(
            SimilaritySpan(
                (a, a + support),
                (a + lag, a + lag + support),
                float(measured.min()),
                float(np.median(measured)),
                count,
            )
        )
    return spans


def _competing_lags(spans: list[SimilaritySpan]) -> bool:
    def overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    return any(
        left.lag_samples != right.lag_samples
        and abs(left.median_similarity - right.median_similarity) <= RIVAL_SIMILARITY_DELTA
        and min(overlap(left.first, right.first), overlap(left.second, right.second))
        >= 0.75 * min(left.sample_count, right.sample_count)
        for left, right in combinations(spans, 2)
    )


def find_similar_spans(pcm_s16le: bytes) -> SimilarityResult:
    """Inspect up to twelve plausible lags in 2.5–4 s mono 16 kHz PCM16.

    A measured empty result covers only the nominated lags. The reported
    half-open intervals span consecutive active 40 ms comparisons at one
    integer lag, including at least 150 ms of nonoverlapping support.
    """
    if not isinstance(pcm_s16le, bytes) or len(pcm_s16le) % 2:
        raise ValueError("Expected complete signed16 PCM bytes")
    count = len(pcm_s16le) // 2
    if not MIN_SAMPLES <= count <= MAX_SAMPLES:
        return SimilarityResult("unavailable", "duration_out_of_scope")
    raw = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float64)
    features = raw[1:] - PREEMPHASIS * raw[:-1]
    if raw.std() < MIN_ACTIVITY_STD or features.std() < MIN_FEATURE_STD:
        return SimilarityResult("unavailable", "insufficient_signal_variation")

    peaks, scores = _lag_peaks(features)
    # Repeated strong peaks with substantial overlap make the delay ambiguous.
    if np.count_nonzero((peaks <= count // 2) & (scores >= PERIODIC_PEAK)) >= 3:
        return SimilarityResult("ambiguous", "periodic_signal")
    selected, capped = _select_lags(peaks, scores)
    spans = []
    for tested, lag in enumerate(selected, 1):
        spans.extend(_lag_spans(raw, features, lag))
        if len(spans) > MAX_SPANS:
            return SimilarityResult("ambiguous", "span_budget", lags_checked=tested, candidate_cap_hit=capped)
    if _competing_lags(spans):
        return SimilarityResult(
            "ambiguous", "competing_lags", lags_checked=len(selected), candidate_cap_hit=capped
        )
    spans.sort(key=lambda row: (-row.sample_count, -row.median_similarity, row.first, row.second))
    return SimilarityResult(
        "measured", spans=tuple(spans), lags_checked=len(selected), candidate_cap_hit=capped
    )
