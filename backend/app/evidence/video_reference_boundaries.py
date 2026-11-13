# SPDX-License-Identifier: Apache-2.0
"""Refine a paired-picture range when both edges have consecutive native support."""

import numpy as np

BASIS = "paired_h264_local_boundaries/1"
RADIUS = 6


def refine(signal, bounds):
    """Return the measured half-open range or abstain when an edge is unclear."""
    first, stop = bounds
    signal = np.asarray(signal)
    before = signal[max(0, first - 10):max(3, first - 4)]
    after = signal[min(len(signal) - 3, stop + 4):stop + 10]
    quarter = (stop - first) // 4
    inside = signal[first + quarter:stop - quarter]
    context = signal[max(0, first - 10):stop + 10]
    if not np.isfinite(context).all():
        return None
    ordinary = max(float(np.median(before)), float(np.median(after)))
    margin = float(np.median(inside)) - ordinary
    if margin < 1.5:
        return None
    point = max(ordinary + .30 * margin, float(np.quantile(before, .8)) + .5,
                float(np.quantile(after, .8)) + .5)
    supported = signal >= point
    center = (first + stop) // 2
    if not supported[center]:
        return None
    left = right = center
    while left > max(1, first - RADIUS - 1) and supported[left - 1]:
        left -= 1
    while right < min(len(signal) - 2, stop + RADIUS) and supported[right]:
        right += 1
    if not (left >= 2 and right + 2 <= len(signal) and right - left >= 4
            and abs(left - first) <= RADIUS and abs(right - stop) <= RADIUS):
        return None
    if supported[left - 2:left].any() or supported[right:right + 2].any():
        return None
    sharpness = max(.75, .18 * margin)
    if min(signal[left] - signal[left - 1], signal[right - 1] - signal[right]) < sharpness:
        return None
    return [left, right]
