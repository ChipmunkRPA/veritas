# SPDX-License-Identifier: Apache-2.0
"""Find the boundaries of strong frame-by-frame support within a paired luma finding."""

import numpy as np

BASIS = "paired_h264_measured_picture_interval/1"


def refine(signal, proposal):
    """Return the measured support interval when both edges have a clear step."""
    start, stop = proposal
    values = np.asarray(signal)
    left = values[max(0, start - 10):max(3, start - 4)]
    right = values[min(len(values) - 3, stop + 4):stop + 10]
    inset = (stop - start) // 4
    if not np.isfinite(values[max(0, start - 10):stop + 10]).all():
        return None
    flank = max(float(np.median(left)), float(np.median(right)))
    margin = float(np.median(values[start + inset:stop - inset])) - flank
    if margin < 1.5:
        return None
    point = max(flank + .30 * margin, float(np.quantile(left, .8)) + .5, float(np.quantile(right, .8)) + .5)
    supported = values >= point
    center = (start + stop) // 2
    if not supported[center]:
        return None
    before = np.flatnonzero(~supported[max(0, start - 8):center]) + max(0, start - 8)
    after = np.flatnonzero(~supported[center:min(len(values), stop + 8)]) + center
    if not len(before) or not len(after):
        return None
    first, end = int(before[-1]) + 1, int(after[0])
    if (first < 2 or end + 2 > len(values) or end - first < 4 or abs(first - start) > 6 or abs(end - stop) > 6
            or supported[first - 2:first].any() or supported[end:end + 2].any()):
        return None
    step = min(values[first] - values[first - 1], values[end - 1] - values[end])
    return [first, end] if step >= max(.75, .50 * margin) else None
