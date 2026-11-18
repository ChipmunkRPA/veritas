# SPDX-License-Identifier: Apache-2.0
"""Measure short pauses in detected speech, with and without noise reduction."""

from __future__ import annotations

import numpy as np

from app.evidence.video_silent_mouth import mouth_motion

BLOCK = 512
FLANK = 5
GAP = 1
WIDTHS = (7, 9, 11, 13, 16, 19)
ARMS = ("original", "enhanced", "both")
MIN_FLANK = .35


def _mean(prefix, starts, width):
    return (prefix[starts + width] - prefix[starts]) / width


def scan(original, enhanced, frames):
    """Compare the same original sample range in each of the two views."""
    values = np.column_stack((original, enhanced))
    if (len(original) != len(enhanced) or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))):
        raise ValueError("Speech activity must be bounded and share the same sample clock")
    prefix = np.concatenate((np.zeros((1, 2)), np.cumsum(values, axis=0)))
    picture = [frame for frame in frames if frame["gap_units"] is not None]
    context = FLANK + GAP
    readings = {arm: {"assessed": False, "reading": None} for arm in ARMS}
    for width in WIDTHS:
        starts = np.arange(context, len(values) - width - context + 1)
        before = _mean(prefix, starts - context, FLANK)
        after = _mean(prefix, starts + width + GAP, FLANK)
        center = _mean(prefix, starts, width)
        flank = np.minimum(before, after)
        eligible = flank >= MIN_FLANK
        margins = flank - center
        arm_eligibility = np.column_stack((eligible, eligible.all(axis=1)))
        arm_margins = np.column_stack((margins, margins.min(axis=1)))
        for index in np.flatnonzero(arm_eligibility.any(axis=1)):
            start = int(starts[index]) * BLOCK
            end = start + width * BLOCK
            if not (support := mouth_motion(picture, start, end)):
                continue
            motion, brackets = support
            for arm_index, arm in enumerate(ARMS):
                if not arm_eligibility[index, arm_index]:
                    continue
                result = readings[arm]
                result["assessed"] = True
                score = int(round(float(arm_margins[index, arm_index]) * 1_000_000))
                previous = result["reading"]
                if score <= 0 or (previous is not None and
                                  (-score, start, end) >= (-previous["margin_millionths"], *previous["span"])):
                    continue
                result["reading"] = {"span": [start, end], "margin_millionths": score,
                                     "picture_frames": brackets, **motion,
                                     "activity_millionths": (np.rint(
                                         1_000_000 * np.stack((before[index], center[index], after[index]))
                                     ).astype(int).tolist())}
    return readings
