# SPDX-License-Identifier: Apache-2.0
"""Nominate short, adjacent reuses of actual waveform samples in one video."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.detectors.video import lipsync

SR, PICTURE_HOP = lipsync.SR, lipsync.HOP
WINDOW, STEP = 960, 160  # 60 ms per correlation, every 10 ms
MIN_WITNESS = 1920  # 120 ms
MIN_LAG, MAX_LAG = 1920, 5760  # 120–360 ms
NOMINATION_WINDOW, NOMINATION_HOP = 9600, 1920
MIN_PEAK, MAX_LOCAL_PEAKS = .12, 4
MATCH, ACTIVE_RMS = .78, .0006
ADJACENCY = 640
_FILTERS = tuple(butter(3, band, fs=SR, btype="bandpass", output="sos")
                 for band in ((180, 950), (950, 3500)))


def _prefix(array):
    return np.r_[0., np.cumsum(array, dtype=np.float64)]


def _lags(array):
    """Find local lag peaks, so a short repeat need not dominate the entire track."""
    width = min(NOMINATION_WINDOW, len(array))
    starts = sorted(set(range(0, len(array) - width + 1, NOMINATION_HOP)) | {len(array) - width})
    choices = set()
    lags = np.arange(MIN_LAG, MAX_LAG + 1)
    overlaps = width - lags
    for start in starts:
        local = array[start:start + width]
        padded = np.fft.rfft(local, n=1 << (2 * width - 1).bit_length())
        products = np.fft.irfft(padded * padded.conjugate())[lags]
        power = _prefix(local**2)
        left, right = power[overlaps], power[-1] - power[lags]
        scale = np.sqrt(np.maximum(0, left * right))
        active = np.minimum(left, right) >= overlaps * ACTIVE_RMS**2
        scores = np.divide(products, scale, out=np.zeros_like(scale), where=active)
        peaks = np.flatnonzero((scores >= MIN_PEAK) & (scores >= np.r_[-np.inf, scores[:-1]])
                               & (scores > np.r_[scores[1:], -np.inf]))
        ranks = sorted(peaks, key=lambda i: (-scores[i], lags[i]))
        chosen = []
        for index in ranks:
            lag = int(lags[index])
            if all(abs(lag - other) >= 32 for other in chosen):
                chosen.append(lag)
            if len(chosen) == MAX_LOCAL_PEAKS:
                break
        choices.update(chosen)
    return sorted(choices)


def _band_support(band, stats, lag, starts):
    power = stats
    left = power[starts + WINDOW] - power[starts]
    right = power[starts + lag + WINDOW] - power[starts + lag]
    cross = _prefix(band[:-lag] * band[lag:])
    value = cross[starts + WINDOW] - cross[starts]
    return ((value >= MATCH * np.sqrt(np.maximum(0, left * right)))
            & (np.minimum(left, right) >= WINDOW * ACTIVE_RMS**2))


def adjacent_repeats(wave):
    """Return adjacent or nearly adjacent pairs of distinct, two-band waveform support."""
    bands = [sosfiltfilt(filt, np.asarray(wave, dtype=np.float64)) for filt in _FILTERS]
    powers = [_prefix(band**2) for band in bands]
    lags = sorted(set().union(*map(_lags, bands)))
    proposals = set()
    for lag in lags:
        starts = np.arange(0, len(wave) - lag - WINDOW + 1, STEP)
        support = _band_support(bands[0], powers[0], lag, starts) & _band_support(bands[1], powers[1], lag, starts)
        for first, last in lipsync._runs(support):
            count = WINDOW + (last - first - 1) * STEP
            # If a periodic source extends into overlapping samples, the
            # endpoints cease to identify a unique pair of adjacent chunks.
            if not max(MIN_WITNESS, lag - ADJACENCY) <= count <= lag + STEP:
                continue
            low = int(starts[first])
            length = min(count, lag)
            proposals.add(((low, low + length), (low + lag, low + lag + length)))
    results = []
    for pair in sorted(proposals, key=lambda x: (x[0][0] - x[0][1], x)):
        if all(not _equivalent(pair, other) for other in results):
            results.append(pair)
    return results


def _equivalent(first, second):
    return all(max(abs(a - b) for a, b in zip(left, right, strict=True)) <= STEP
               for left, right in zip(first, second, strict=True))


def _patch_difference(first, second):
    a = first[4:20, 7:25].astype(np.float32)
    a -= a.mean()
    minimum = float("inf")
    for y in (3, 4, 5):
        for x in (6, 7, 8):
            b = second[y:y + 16, x:x + 18].astype(np.float32)
            b -= b.mean()
            minimum = min(minimum, float(np.mean(np.abs(a - b))))
    return minimum


def visual_support(pair, frames, patches):
    """Check that the same sound accompanies different observed moving mouths."""
    (start, end), (other, _) = pair
    delta = other - start
    ticks = np.arange(len(frames)) * PICTURE_HOP + PICTURE_HOP // 2
    expected = [(int(i), int((tick + delta) // PICTURE_HOP)) for i, tick in enumerate(ticks)
                if start <= tick < end and tick + delta < len(ticks) * PICTURE_HOP]
    observed = [(a, b) for a, b in expected if a in patches and b in patches]
    if len(observed) < 3 or 3 * len(observed) < 2 * len(expected):
        return None
    if any(len({frames[pair[index]] for pair in observed}) != len(observed) for index in (0, 1)):
        return None
    motion = [max(_patch_difference(patches[a[index]], patches[b[index]])
                  for a, b in zip(observed, observed[1:], strict=False)) for index in (0, 1)]
    if min(motion) < 2.0:
        return None
    differences = [_patch_difference(patches[a], patches[b]) for a, b in observed]
    if np.median(differences) < 5 or sum(diff >= 5 for diff in differences) < 3:
        return None
    return {"frame_pairs": [[frames[a], frames[b]] for a, b in observed],
            "mouth_difference_median": round(float(np.median(differences)), 6),
            "motion_maxima": [round(float(value), 6) for value in motion]}
