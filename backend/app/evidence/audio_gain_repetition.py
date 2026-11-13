"""Find native PCM excerpts related by one fixed gain and quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.evidence import audio_exact_repetition as exact
from app.evidence.audio_gain_repetition_binding import GAINS

MAX_ANCHOR_HITS = 4096
MAX_CANDIDATES = 48
MIN_ACTIVE_ANCHORS = 6


@dataclass(frozen=True)
class GainRepeat:
    first: tuple[int, int]
    second: tuple[int, int]
    numerator: int
    denominator: int
    scaled_occurrence: Literal["first", "second"]

    @property
    def sample_count(self) -> int:
        return self.first[1] - self.first[0]


@dataclass(frozen=True)
class GainRepetitionResult:
    status: Literal["measured", "unavailable", "ambiguous"]
    reason: str | None = None
    spans: tuple[GainRepeat, ...] = ()


def _activity(values: np.ndarray, step: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(values, exact.ANCHOR_SAMPLES)[::step]
    return np.ptp(windows, axis=1) >= exact.MIN_ANCHOR_RANGE


def find_gain_repeats(pcm_s16le: bytes) -> GainRepetitionResult:
    """Search for a nonoverlapping repeat with an exact, unclipped gain relation."""
    if len(pcm_s16le) % 2:
        raise ValueError("Expected complete signed 16-bit samples")
    count = len(pcm_s16le) // 2
    if not exact.MIN_SAMPLES <= count <= exact.MAX_SAMPLES:
        return GainRepetitionResult("unavailable", "duration_out_of_scope")

    observed = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.int32)
    active = np.flatnonzero(_activity(observed, exact.ANCHOR_STEP)) * exact.ANCHOR_STEP
    if len(active) < MIN_ACTIVE_ANCHORS:
        return GainRepetitionResult("unavailable", "insufficient_signal_variation")

    anchors: dict[bytes, list[tuple[int, int]]] = {}
    transformed = []
    anchor_size = exact.ANCHOR_SAMPLES
    for index, (numerator, denominator) in enumerate(GAINS):
        scaled = np.rint(observed * numerator / denominator).astype(np.int32)
        transformed.append(scaled)
        payload = scaled.astype("<i2").tobytes()
        for start in active.tolist():
            window = scaled[start:start + anchor_size]
            low, high = int(window.min()), int(window.max())
            if -32768 <= low <= high <= 32767 and high - low >= exact.MIN_ANCHOR_RANGE:
                key = payload[2 * start:2 * (start + anchor_size)]
                anchors.setdefault(key, []).append((index, start))

    hits = 0
    candidate_hits: dict[tuple[int, int], int] = {}
    for start in range(count - anchor_size + 1):
        key = pcm_s16le[2 * start:2 * (start + anchor_size)]
        for index, anchor in anchors.get(key, ()):
            lag = start - anchor
            if abs(lag) < exact.MIN_REPEAT_SAMPLES:
                continue
            hits += 1
            if hits > MAX_ANCHOR_HITS:
                return GainRepetitionResult("ambiguous", "anchor_pair_budget")
            candidate = (index, lag)
            candidate_hits[candidate] = candidate_hits.get(candidate, 0) + 1

    candidates = sorted(pair for pair, support in candidate_hits.items() if support >= 2)
    if len(candidates) > MAX_CANDIDATES:
        return GainRepetitionResult("ambiguous", "candidate_budget")

    spans: dict[tuple[tuple[int, int], tuple[int, int]], GainRepeat] = {}
    for index, lag in candidates:
        delta = abs(lag)
        earlier, later = observed[:-delta], observed[delta:]
        scaled = transformed[index][:-delta] if lag > 0 else transformed[index][delta:]
        target = later if lag > 0 else earlier
        donor = earlier if lag > 0 else later
        # Observed samples are signed 16-bit. An out-of-range scaled value cannot match.
        boundaries = np.flatnonzero(np.diff(scaled == target, prepend=False, append=False))
        for start, stop in boundaries.reshape(-1, 2).tolist():
            stop = min(stop, start + delta)
            if stop - start < exact.MIN_REPEAT_SAMPLES:
                continue
            if np.count_nonzero(_activity(donor[start:stop], exact.ANCHOR_STEP)) < MIN_ACTIVE_ANCHORS:
                continue
            if np.count_nonzero(_activity(target[start:stop], exact.ANCHOR_STEP)) < MIN_ACTIVE_ANCHORS:
                continue
            if np.array_equal(donor[start:stop], target[start:stop]):
                continue
            first, second = (start, stop), (start + delta, stop + delta)
            numerator, denominator = GAINS[index]
            # Report one exact relationship per range. The sign and ratio do not establish edit order.
            spans.setdefault((first, second), GainRepeat(first, second, numerator, denominator, "second" if lag > 0 else "first"))
            if len(spans) > exact.MAX_SPANS:
                return GainRepetitionResult("ambiguous", "span_budget")

    return GainRepetitionResult("measured", spans=tuple(sorted(spans.values(), key=lambda span: (-span.sample_count, span.first, span.second))))
