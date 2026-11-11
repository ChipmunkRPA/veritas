"""Bounded exact PCM repetition evidence for short clips in the research lab.

Matching sample ranges describe repeated decoded audio. They carry no speech
origin decision or claim about the complete edited interval.
"""

from __future__ import annotations

import sys
from array import array
from dataclasses import dataclass
from itertools import groupby
from typing import Literal

SAMPLE_RATE = 16000
MIN_SAMPLES = 40000
MAX_SAMPLES = 64000
MIN_REPEAT_SAMPLES = 2400
ANCHOR_SAMPLES = 64
ANCHOR_STEP = 128
MIN_ANCHOR_RANGE = 32
MAX_ANCHOR_PAIR_CHECKS = 4096
MAX_SAMPLE_COMPARISONS = 4 * MAX_SAMPLES
MAX_SPANS = 8


@dataclass(frozen=True)
class RepeatedSpan:
    """Two half-open ranges of exactly equal samples in source order."""

    first: tuple[int, int]
    second: tuple[int, int]

    @property
    def sample_count(self) -> int:
        return self.first[1] - self.first[0]

    @property
    def lag_samples(self) -> int:
        return self.second[0] - self.first[0]


@dataclass(frozen=True)
class RepetitionResult:
    status: Literal["measured", "unavailable", "ambiguous"]
    reason: str | None = None
    spans: tuple[RepeatedSpan, ...] = ()
    anchor_pair_checks: int = 0
    sample_comparisons: int = 0


def _extend(samples: array, start: int, lag: int, budget: int) -> tuple[tuple[int, int] | None, int]:
    left, right = start, start + ANCHOR_SAMPLES
    comparisons = 0
    while left > 0:
        if comparisons == budget:
            return None, comparisons
        comparisons += 1
        if samples[left - 1] != samples[left - 1 + lag]:
            break
        left -= 1
    while right + lag < len(samples):
        if comparisons == budget:
            return None, comparisons
        comparisons += 1
        if samples[right] != samples[right + lag]:
            break
        right += 1
    return (left, right), comparisons


def find_repeated_spans(pcm_s16le: bytes) -> RepetitionResult:
    """Search mono 16 kHz little-endian signed PCM for nonoverlapping repeats.

    Active anchors use a coarse peak-to-peak range and are indexed every 128
    samples. Each sample position can match an exact anchor, including delays
    off the anchor grid. Verified witnesses are at least 150 ms. A budget or
    result cap yields ambiguity without returning incomplete evidence.
    """
    if len(pcm_s16le) % 2:
        raise ValueError("Expected complete signed 16-bit samples")
    sample_count = len(pcm_s16le) // 2
    if not MIN_SAMPLES <= sample_count <= MAX_SAMPLES:
        return RepetitionResult("unavailable", "duration_out_of_scope")

    samples = array("h")
    samples.frombytes(pcm_s16le)
    if sys.byteorder == "big":
        samples.byteswap()

    def key(start: int) -> bytes:
        return pcm_s16le[2 * start : 2 * (start + ANCHOR_SAMPLES)]

    end = sample_count - ANCHOR_SAMPLES + 1
    anchors: dict[bytes, list[int]] = {}
    for start in range(0, end, ANCHOR_STEP):
        values = samples[start : start + ANCHOR_SAMPLES]
        if max(values) - min(values) >= MIN_ANCHOR_RANGE:
            anchors.setdefault(key(start), []).append(start)
    if not anchors:
        return RepetitionResult("unavailable", "insufficient_signal_variation")

    pair_checks = 0
    candidates = set()
    for start in range(end):
        for anchor in anchors.get(key(start), ()):
            if pair_checks == MAX_ANCHOR_PAIR_CHECKS:
                return RepetitionResult("ambiguous", "anchor_pair_budget", anchor_pair_checks=pair_checks)
            pair_checks += 1
            lag = abs(start - anchor)
            if lag >= MIN_REPEAT_SAMPLES:
                candidates.add((lag, min(start, anchor)))

    comparisons = 0
    spans = []
    for lag, pairs in groupby(sorted(candidates), key=lambda pair: pair[0]):
        covered_until = 0
        for _, start in pairs:
            if start < covered_until:
                continue
            interval, used = _extend(samples, start, lag, MAX_SAMPLE_COMPARISONS - comparisons)
            comparisons += used
            if interval is None:
                return RepetitionResult(
                    "ambiguous",
                    "sample_comparison_budget",
                    anchor_pair_checks=pair_checks,
                    sample_comparisons=comparisons,
                )
            left, covered_until = interval
            length = min(covered_until - left, lag)
            if length >= MIN_REPEAT_SAMPLES:
                spans.append(RepeatedSpan((left, left + length), (left + lag, left + lag + length)))
            if len(spans) > MAX_SPANS:
                return RepetitionResult(
                    "ambiguous", "span_budget", anchor_pair_checks=pair_checks, sample_comparisons=comparisons
                )

    spans.sort(key=lambda span: (-span.sample_count, span.first, span.second))
    return RepetitionResult(
        "measured", spans=tuple(spans), anchor_pair_checks=pair_checks, sample_comparisons=comparisons
    )
