# SPDX-License-Identifier: MIT
"""Compare reversed source motion after a fixed crop of one occurrence."""
from __future__ import annotations

import math

import numpy as np

from app.detectors.video import picture_crop as coarse
from app.detectors.video import picture_crop_aligned as aligned
from app.evidence.picture_sequence import MIN_SPAN, REVERSE_CROP_PERCENTAGES


def _reverse_chains(linked, moving, minimum_steps):
    # Axis 1 counts backwards on the narrower occurrence. For an edge at
    # (left, left + lag), the narrow edge runs from right to right - 1.
    last = len(linked)
    for lag in range(1 - last, last):
        connected = linked.diagonal(lag)
        offsets = np.flatnonzero(connected & moving.diagonal(lag))
        if len(offsets) < coarse.MIN_EDGES:
            continue
        origin = max(0, -lag)
        left = offsets + origin
        right = last - left - lag
        for wider, disjoint in (("first", left + 1 < right - 1), ("second", right < left)):
            support = offsets[disjoint]
            gaps = np.diff(support) > coarse.MAX_EDGE_STEP
            losses = np.diff(np.cumsum(~connected)[support]) > 0
            for chain in np.split(support, np.flatnonzero(gaps | losses) + 1):
                if len(chain) < coarse.MIN_EDGES:
                    continue
                start, end = int(chain[0] + origin), int(chain[-1] + origin + 1)
                if end - start < minimum_steps or 5 * len(chain) < 3 * (end - start):
                    continue
                narrow = last - end - lag, last - start - lag
                ranges = ((start, end), narrow)
                first_support = tuple(int(edge + origin - start) for edge in chain)
                if wider == "second":
                    ranges = ranges[::-1]
                    first_support = tuple(end - start - 1 - offset for offset in reversed(first_support))
                yield ranges, wider, first_support


def find(frames, *, percentages=REVERSE_CROP_PERCENTAGES, landmarks=False):
    """Compare the three validated centered crops on the opening source clock."""
    minimum = math.ceil(25 * MIN_SPAN)
    if len(frames) < 2 * (minimum + 1):
        return None, None
    sums = aligned._integral(frames)
    w, h = coarse.WORK_WIDTH, coarse.WORK_HEIGHT
    unit = aligned.UNIT
    full = aligned._crop_vectors(sums, (0, 0, w * unit, h * unit))
    power = np.square(full).mean(axis=1)
    distinct = coarse._distance(full, full, power, power) >= coarse.MIN_DIRECT_RMS ** 2
    motion, rms = coarse._motion(full[::-1])
    centered = [aligned._crop_box(p / 100, w * (100 - p) / 200, h * (100 - p) / 200)
                for p in percentages]
    branches = [("centered", centered)]
    if landmarks:
        branches.append(("landmarks", aligned._landmark_boxes(frames, minimum + 1)))
    best, match, alignment = None, None, None
    for method, boxes in branches:
        for box in dict.fromkeys(boxes):
            crop = aligned._crop_vectors(sums, box)
            distances = coarse._distance(crop, full, np.square(crop).mean(axis=1), power)
            close = ((distances <= aligned.MAX_PICTURE_RMS ** 2) & distinct)[:, ::-1]
            linked = close[:-1, :-1] & close[1:, 1:]
            if np.count_nonzero(linked) < coarse.MIN_EDGES:
                continue
            crop_motion, crop_rms = coarse._motion(crop)
            weaker = np.minimum(crop_rms[:, None], rms[None, :])
            stronger = np.maximum(crop_rms[:, None], rms[None, :])
            correlation = crop_motion @ motion.T / full.shape[1]
            moving = ((weaker >= coarse.MIN_MOTION_RMS_25FPS) & (weaker >= coarse.MIN_MOTION_RATIO * stronger)
                      & (correlation >= coarse.MIN_MOTION_COSINE * weaker * stronger))
            for ranges, wider, support in _reverse_chains(linked, moving, minimum):
                first, second = (np.arange(start, stop + 1) for start, stop in ranges)
                errors = distances[first, second[::-1]] if wider == "first" else distances[second, first[::-1]]
                rank = len(first), len(support), -float(np.mean(errors)), -int(first[0]), -int(second[0])
                if best is None or rank > best:
                    best, match, alignment = rank, coarse.CropMatch(ranges, wider, box, support), method
    return match, alignment
