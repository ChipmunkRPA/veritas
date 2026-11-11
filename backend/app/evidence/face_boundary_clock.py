# SPDX-License-Identifier: Apache-2.0
"""Exact native-frame selection for the 12 Hz face-boundary refinement."""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from fractions import Fraction
from itertools import pairwise
from math import ceil, floor

CADENCE = 12
COARSE_LIMIT = Fraction(1)
SCHEDULE_MARGIN = Fraction(5, 8)
MAX_FRAMES = ceil((COARSE_LIMIT + 2 * SCHEDULE_MARGIN) * CADENCE) + 3  # Grid endpoints and native rounding.


def native_schedule(pts: Sequence[int], time_base: Fraction, start_pts: int, end_pts: int) -> dict[int, Fraction]:
    """Select nearest native frames on the stream-origin grid, with earliest-index ties."""
    if (not pts or not 0 < (end_pts - start_pts) * time_base <= COARSE_LIMIT
            or any(a > b for a, b in pairwise(pts))):
        return {}
    origin = pts[0] * time_base
    low = start_pts * time_base - SCHEDULE_MARGIN
    high = end_pts * time_base + SCHEDULE_MARGIN
    first = max(0, floor((low - origin) * CADENCE) - 1)
    last = min(floor((pts[-1] * time_base - origin) * CADENCE), ceil((high - origin) * CADENCE) + 1)
    samples = {}
    for step in range(first, last + 1):
        target = (origin + Fraction(step, CADENCE)) / time_base
        right = bisect_left(pts, target)
        neighbors = []
        if right:
            neighbors.append(bisect_left(pts, pts[right - 1]))
        if right < len(pts):
            neighbors.append(right)
        i = min(neighbors, key=lambda index: (abs(pts[index] - target), index))
        instant = pts[i] * time_base
        if low <= instant <= high:
            samples[i] = instant
    return samples if len(samples) <= MAX_FRAMES else {}
