# SPDX-License-Identifier: Apache-2.0
"""Look for a short change in the delay between two recorded audio channels."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

RATE = 48000
WINDOW = 2880
HOP = 960
MAX_SHIFT = 36
FLANK = 8
INNER_MIN = 8
INNER_MAX = 51
MIN_RELATIVE_DELAY = 8
FILTER = butter(3, (250, 3600), fs=RATE, btype="bandpass", output="sos")


def windows(stereo):
    """Measure channel alignment in short observed windows on the source clock."""
    signal = sosfiltfilt(FILTER, stereo.astype(np.float64), axis=0)
    # Retain native delay resolution while limiting dot products to one third
    # of the sample positions. The bandpass stays below that new Nyquist rate.
    starts = np.arange(HOP, len(signal) - WINDOW - MAX_SHIFT, HOP)
    indices = starts[:, None] + np.arange(0, WINDOW, 3)
    left = signal[indices, 0]
    left_energy = np.einsum("ij,ij->i", left, left)
    shifts = np.arange(-MAX_SHIFT, MAX_SHIFT + 1)
    scores = []
    for shift in shifts:
        right = signal[indices + shift, 1]
        energy = np.einsum("ij,ij->i", right, right)
        supported = np.minimum(left_energy, energy) >= left.shape[1] * 24 ** 2
        scores.append(np.divide(np.einsum("ij,ij->i", left, right), np.sqrt(left_energy * energy),
                                out=np.zeros(len(starts)), where=supported))
    scores = np.asarray(scores).T
    best = np.argmax(scores, axis=1)
    peak = scores[np.arange(len(starts)), best]
    rival = np.where(np.abs(shifts[None, :] - shifts[best, None]) > 4, scores, -1.).max(axis=1)
    supported = (peak >= .84) & (peak - rival >= .013) & (np.abs(shifts[best]) < MAX_SHIFT)
    return starts + WINDOW // 2, shifts[best], supported


def _plateau(shifts, supported, low, high, minimum):
    indices = np.flatnonzero(supported[low:high]) + low
    if len(indices) < minimum:
        return None
    values = shifts[indices]
    center = int(np.rint(np.median(values)))
    agreeing = indices[np.abs(values - center) <= 2]
    if len(agreeing) < minimum:
        return None
    return center, int(agreeing[0]), int(agreeing[-1]), len(agreeing)


def locate(stereo):
    """Locate a delay step bounded by two stable portions with the same delay."""
    if stereo.ndim != 2 or stereo.shape[1] != 2 or not 120000 <= len(stereo) <= 192000:
        raise ValueError("Expected 2.5–4 seconds of continuous 48 kHz stereo")
    times, shifts, supported = windows(stereo)
    stable = [_plateau(shifts, supported, index, index + FLANK, 5) for index in range(len(times) - FLANK + 1)]
    assessed = False
    best = None
    for start in range(FLANK, len(times) - FLANK - INNER_MIN + 1):
        before = stable[start - FLANK]
        if before is None or before[2] < start - 3:
            continue
        for end in range(start + INNER_MIN, min(start + INNER_MAX, len(times) - FLANK) + 1):
            after = stable[end]
            if after is None or after[1] > end + 2 or abs(before[0] - after[0]) > 3:
                continue
            inside = _plateau(shifts, supported, start, end, max(6, (end - start) * 3 // 4))
            if inside is None or inside[1] > start + 2 or inside[2] < end - 3:
                continue
            assessed = True
            difference = min(abs(inside[0] - before[0]), abs(inside[0] - after[0]))
            if difference < MIN_RELATIVE_DELAY:
                continue
            low = int((times[before[2]] + times[inside[1]]) // 2)
            high = int((times[inside[2]] + times[after[1]]) // 2)
            score = difference * inside[3] / (end - start)
            if best is None or (score, high - low) > (best["score"], best["end_sample"] - best["start_sample"]):
                best = {"start_sample": low, "end_sample": high, "before_offset_samples": before[0],
                        "inside_offset_samples": inside[0], "after_offset_samples": after[0], "score": score,
                        "support": [before[3], inside[3], after[3]]}
    return assessed, best
