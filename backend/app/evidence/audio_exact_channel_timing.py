"""Find a locally moved exact excerpt on one decoded audio channel."""

from __future__ import annotations

import numpy as np

from app.detectors.audio.decode import SAMPLE_RATE as RATE

ANCHOR = 60 * RATE // 1000
GRID = 20 * RATE // 1000
MIN_EXCERPT = 160 * RATE // 1000
MIN_SHIFT = 120 * RATE // 1000
MAX_SHIFT = 360 * RATE // 1000
MIN_LEVEL = .003 * 32768


def _unique_match(earlier, excerpt, position, size):
    lower = max(0, position - MAX_SHIFT)
    upper = min(size, position + MAX_SHIFT + ANCHOR)
    hit = earlier.find(excerpt, 2 * lower, 2 * upper)
    match = None
    while hit >= 0:
        if hit % 2 == 0:
            if match is not None:
                return None
            match = hit // 2
        hit = earlier.find(excerpt, hit + 1, 2 * upper)
    return match


def locate(earlier, analyzed):
    """Use the sample clock and exact waveform equality, with no inferred source labels."""
    if len(earlier) != len(analyzed):
        return None
    changed = np.flatnonzero(earlier != analyzed)
    if not changed.size:
        return None
    start, stop = int(changed[0]), int(changed[-1]) + 1
    size = len(earlier)
    if start < ANCHOR or stop > size - ANCHOR or not MIN_EXCERPT <= stop - start <= 2 * RATE:
        return None
    first = earlier.astype("<i2", copy=False).tobytes()
    second = analyzed.astype("<i2", copy=False).tobytes()
    candidates = {}
    for position in range(start, stop - ANCHOR + 1, GRID):
        if np.std(analyzed[position:position + ANCHOR], dtype=np.float64) < MIN_LEVEL:
            continue
        origin = _unique_match(first, second[2 * position:2 * (position + ANCHOR)], position, size)
        if origin is None or abs(position - origin) < MIN_SHIFT:
            continue
        candidates.setdefault(position - origin, []).append(position)

    matches = []
    for shift, positions in candidates.items():
        if positions[-1] - positions[0] < ANCHOR:
            continue
        best = None
        for position in positions:
            if best is not None and best[0] <= position < position + ANCHOR <= best[1]:
                continue
            left, right = position, position + ANCHOR
            while left > max(start, shift) and earlier[left - shift - 1] == analyzed[left - 1]:
                left -= 1
            while right < min(stop, size + shift) and earlier[right - shift] == analyzed[right]:
                right += 1
            if right - left >= MIN_EXCERPT and (best is None or right - left > best[1] - best[0]):
                best = left, right
        if best is not None:
            left, right = best
            supported = [p for p in positions if left <= p and p + ANCHOR <= right]
            if supported and supported[-1] - supported[0] >= ANCHOR:
                matches.append((shift, left, right))
    if len(matches) != 1:
        return None
    (shift, left, right), = matches
    return {"shift_samples": shift, "earlier_samples": [left - shift, right - shift],
            "submitted_samples": [left, right], "changed_samples": [start, stop]}
