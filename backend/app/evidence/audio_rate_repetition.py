# SPDX-License-Identifier: Apache-2.0
"""Locate quieter stereo sound repeated at a different sample playback rate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import butter, resample_poly, sosfiltfilt

from app.evidence.audio_channel_alignment import Alignment, projection
from app.evidence.audio_channel_alignment_binding import (
    MAX_PAIRS,
    MAX_SAMPLES,
    MIN_SAMPLES,
    NATIVE_FACTOR,
    NATIVE_RATE,
    RATE_DECIMATION,
    RATE_EDGE_MARGIN,
    RATE_GAP,
    RATE_MIN_SPAN,
    RATE_RATIOS,
    overlap,
    rate_sample,
)
from app.evidence.audio_native_alignment import estimate
from app.evidence.audio_similarity import _prefix

STEP = RATE_DECIMATION
RATE = NATIVE_RATE // STEP
BANDS = ((180, 460), (460, 940), (940, 1720))
FILTERS = tuple(butter(3, band, btype="bandpass", fs=RATE, output="sos") for band in BANDS)
ANCHOR = 400
ANCHOR_HOP = 160
WINDOW = 200
HOP = 60
MIN_RMS = 2.
ANCHOR_MATCH = .72
BAND_MATCH = .79


@dataclass(frozen=True)
class Match:
    first: tuple[int, int]
    second: tuple[int, int]
    faster: str
    ratio: tuple[int, int]


def _window_energy(values, count):
    prefix = _prefix(values)
    squares = _prefix(values * values)
    return np.maximum(0., squares[count:] - squares[:-count] - (prefix[count:] - prefix[:-count]) ** 2 / count)


def _nominate(original, stretched, num, den):
    """Find possible alignments with independent short anchors across the clip."""
    lags = set()
    count = len(stretched) - ANCHOR + 1
    nfft = next_fast_len(len(stretched) + ANCHOR - 1, real=True)
    transformed = rfft(stretched.astype(np.float32), nfft)
    energy = _window_energy(stretched, ANCHOR)
    faster_time = np.arange(count) * num / den
    samples = np.lib.stride_tricks.sliding_window_view(original, ANCHOR)[::ANCHOR_HOP]
    for base in range(0, len(samples), 16):
        anchors = samples[base:base + 16].astype(np.float32)
        anchors -= anchors.mean(axis=1, keepdims=True)
        power = np.einsum("ij,ij->i", anchors, anchors)
        dots = irfft(rfft(anchors[:, ::-1], nfft, axis=1) * transformed, nfft, axis=1)[:, ANCHOR - 1:len(stretched)]
        for index, (dot, anchor_energy) in enumerate(zip(dots, power, strict=True), base):
            if anchor_energy < ANCHOR * MIN_RMS**2:
                continue
            start = index * ANCHOR_HOP
            disjoint = ((faster_time >= start + ANCHOR + RATE_GAP / STEP)
                        | (faster_time + ANCHOR * num / den + RATE_GAP / STEP <= start))
            usable = disjoint & (energy >= ANCHOR * MIN_RMS**2)
            scores = np.divide(dot, np.sqrt(energy * anchor_energy), out=np.zeros(count), where=usable)
            best = int(scores.argmax())
            if scores[best] >= ANCHOR_MATCH:
                lags.add(best - start)
    return sorted(lags)


def _scores(first, second, starts, lag):
    scores = []
    for a, b in zip(first, second, strict=True):
        a = a[starts[0]: starts[-1] + WINDOW]
        b = b[starts[0] + lag: starts[-1] + lag + WINDOW]
        sa = _prefix(a)
        sb = _prefix(b)
        cross = _prefix(a * b)
        covariance = cross[WINDOW:] - cross[:-WINDOW] - (sa[WINDOW:] - sa[:-WINDOW]) * (sb[WINDOW:] - sb[:-WINDOW]) / WINDOW
        ea, eb = _window_energy(a, WINDOW), _window_energy(b, WINDOW)
        active = np.minimum(ea, eb) >= WINDOW * MIN_RMS**2
        scores.append(np.divide(covariance, np.sqrt(ea * eb), out=np.zeros(len(ea)), where=active)[::HOP])
    return np.sort(scores, axis=0)[-2]


def _matches(original, bands, ratio, low, high):
    num, den = ratio
    stretched = resample_poly(original, den, num)
    stretched_bands = [sosfiltfilt(filt, stretched) for filt in FILTERS]
    for lag in _nominate(np.sum(bands, axis=0), np.sum(stretched_bands, axis=0), num, den):
        starts = np.arange(max(0, -lag), min(len(original), len(stretched) - lag) - WINDOW + 1, HOP)
        if not len(starts):
            continue
        scores = _scores(bands, stretched_bands, starts, lag)
        edges = np.flatnonzero(np.diff(np.r_[False, scores >= BAND_MATCH, False]))
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            a, b = int(starts[start]), int(starts[stop - 1]) + WINDOW
            if STEP * (b - a) < RATE_MIN_SPAN:
                continue
            longer = (STEP * a, STEP * b)
            shorter = (rate_sample(a + lag, num, den), rate_sample(b + lag, num, den))
            first, second = sorted((longer, shorter))
            if low + RATE_EDGE_MARGIN <= first[0] and first[1] + RATE_GAP <= second[0] and second[1] <= high - RATE_EDGE_MARGIN:
                yield float(scores[start:stop].mean()), Match(first, second, "first" if first == shorter else "second", ratio)


def _overlaps(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def search(stereo: np.ndarray, fitted: Alignment | None = None):
    """Find rate-change matches and return their ranges on the original 48 kHz clock."""
    if (stereo.dtype != np.dtype("<i2") or stereo.ndim != 2 or stereo.shape[1] != 2
            or len(stereo) % NATIVE_FACTOR or not MIN_SAMPLES * NATIVE_FACTOR <= len(stereo) <= MAX_SAMPLES * NATIVE_FACTOR):
        raise ValueError("Unsupported 48 kHz continuous stereo input")
    fitted = fitted or estimate(stereo)
    if fitted is None:
        return None, "insufficient_alignment", ()
    low, high = overlap(len(stereo), fitted.right_offset_samples)
    values = resample_poly(projection(stereo, fitted).astype(np.float64), 1, STEP)
    bands = [sosfiltfilt(filt, values) for filt in FILTERS]
    if sum(band.std() >= MIN_RMS for band in bands) < 2:
        return fitted, "insufficient_active_bands", ()
    candidates = [item for ratio in RATE_RATIOS for item in _matches(values, bands, ratio, low, high)]
    pairs = []
    for _, pair in sorted(candidates, key=lambda record: record[0], reverse=True):
        if any(_overlaps(pair.first, old.first) and _overlaps(pair.second, old.second) for old in pairs):
            continue
        pairs.append(pair)
        if len(pairs) > MAX_PAIRS:
            return fitted, "span_budget", ()
    return fitted, None, tuple(sorted(pairs, key=lambda match: (match.first, match.second)))
