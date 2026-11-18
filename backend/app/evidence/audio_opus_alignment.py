# SPDX-License-Identifier: Apache-2.0
"""Locate repeated quieter sound in verified short stereo Opus recordings."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import sosfiltfilt

from app.evidence import audio_aac_alignment as repeat
from app.evidence import audio_channel_alignment, audio_native_alignment
from app.evidence.audio_channel_alignment_binding import AAC_EDGE_MARGIN, NATIVE_FACTOR, overlap
from app.evidence.audio_similarity_binding import HOP, MAX_PAIRS, MIN_REPEAT_SAMPLES, WINDOW

LOCAL_POINT = .62
MAX_FINE_LAGS = 12


def sparse_lags(bands):
    """Retain fine sample offsets backed by two frequency bands in the same local window."""
    values = np.array([band[::2] for band in bands])
    full = np.asarray(bands)
    width, step, distance = 960, 640, 1200
    valid = values.shape[1] - width + 1
    fft_size = next_fast_len(values.shape[1] + width - 1, real=True)
    spectrum = rfft(values, n=fft_size, axis=1)
    totals = np.column_stack((np.zeros(len(values)), np.cumsum(values, axis=1)))
    sums = totals[:, width:] - totals[:, :-width]
    powers = np.column_stack((np.zeros(len(values)), np.cumsum(values**2, axis=1)))
    energy = np.maximum(powers[:, width:] - powers[:, :-width] - sums**2 / width, 0)
    proposals = defaultdict(list)
    for start in range(0, valid - distance, step):
        right = start + distance
        template = values[:, start:start + width]
        centered = template - template.mean(axis=1, keepdims=True)
        products = irfft(spectrum * rfft(centered, n=fft_size, axis=1).conjugate(),
                         n=fft_size, axis=1)[:, right:valid]
        window_energy = energy[:, right:valid]
        anchor_energy = energy[:, start:start + 1]
        active = np.minimum(anchor_energy, window_energy) >= width * 4
        scores = np.divide(products, np.sqrt(anchor_energy * window_energy),
                           out=np.zeros_like(products), where=active)
        combined = np.partition(scores, -2, axis=0)[-2].copy()
        for _ in range(2):
            peak = int(np.argmax(combined))
            if combined[peak] < LOCAL_POINT:
                break
            proposals[peak + distance].append((start, float(combined[peak])))
            combined[max(0, peak - 127):peak + 128] = -1

    def ranked_lag(lag):
        return -len(proposals[lag]), -max(score for _, score in proposals[lag]), lag

    def agrees(first, lag):
        second = first + lag
        if second + 2 * width > full.shape[1]:
            return False
        left, right = (full[:, a:a + 2 * width] for a in (first, second))
        left = left - left.mean(axis=1, keepdims=True)
        right = right - right.mean(axis=1, keepdims=True)
        left_energy, right_energy = (np.einsum("ij,ij->i", side, side) for side in (left, right))
        strong = np.minimum(left_energy, right_energy) >= 2 * width * 4
        scores = np.einsum("ij,ij->i", left, right)
        return np.count_nonzero(strong & (scores >= LOCAL_POINT * np.sqrt(left_energy * right_energy))) >= 2

    fine = {
        lag
        for coarse in sorted(proposals, key=ranked_lag)[:12]
        for lag in range(2 * coarse - 2, 2 * coarse + 3)
        if lag >= 2 * distance and any(agrees(2 * start, lag) for start, _ in proposals[coarse])
    }
    return sorted(fine)



def search(stereo: np.ndarray):
    """Retain the whole-file result, then try independently supported local offsets."""
    fitted, reason, spans = repeat.search(stereo)
    if fitted is None or reason is not None or spans:
        return fitted, reason, spans
    projected = audio_channel_alignment.projection(stereo, fitted)
    values = audio_native_alignment.narrowband(projected)[::NATIVE_FACTOR]
    bands = [sosfiltfilt(filt, values) for filt in repeat.FILTERS]
    lags = sparse_lags(bands)
    if not lags or len(lags) > MAX_FINE_LAGS:
        return fitted, reason, spans
    stats = [(band, repeat._prefix(band), repeat._prefix(band * band)) for band in bands]
    low, high = overlap(len(stereo), fitted.right_offset_samples)
    candidates = []
    for lag in lags:
        starts = np.arange(0, len(values) - lag - WINDOW + 1, HOP)
        scores = repeat._window_scores(stats, starts, lag)
        edges = np.flatnonzero(np.diff(np.r_[False, scores >= repeat.MIN_CORRELATION, False]))
        for first, stop in zip(edges[::2], edges[1::2], strict=True):
            length = min(WINDOW + (int(stop - first) - 1) * HOP, lag)
            start = int(starts[first])
            pair = tuple((side * NATIVE_FACTOR, (side + length) * NATIVE_FACTOR) for side in (start, start + lag))
            if length >= MIN_REPEAT_SAMPLES and low + AAC_EDGE_MARGIN <= pair[0][0] and pair[1][1] <= high - AAC_EDGE_MARGIN:
                candidates.append((float(np.mean(scores[first:stop])), pair))
    located = []
    for _, pair in sorted(candidates, reverse=True):
        if any(all(abs(a[0] - b[0]) < repeat.NEARBY_PAIR for a, b in zip(pair, old, strict=True)) for old in located):
            continue
        located.append(pair)
        if len(located) > MAX_PAIRS:
            return fitted, "span_budget", ()
    return fitted, None, tuple(located)
