# SPDX-License-Identifier: MIT
"""Compare near-silent soundtrack samples with successive visible mouth openings."""

from __future__ import annotations

import math
from itertools import groupby, pairwise

RATE = 16_000
HOP = 80
PEAK = 4
MIN_LENGTH, MAX_LENGTH = (2400, 12800)
FLANK = 1920
FLANK_RMS = 96
MAX_FRAME_GAP = 960
APERTURE_SWING = 160_000
APERTURE_STEP = 45_000
KEY = "silent_mouth"
SHOT_KEY = "shot_ranges"
TITLE = "Mouth movement with a near-silent soundtrack"
INTERPRETATION = ("The highlighted samples have very low amplitude. The frames show how the visible mouth "
                  "changes around the same source time.")
LIMITATION = ("The interval can occur during muting, audio dropouts or ordinary quiet motion. "
              "The observation carries zero weight in the AI assessment.")


def quiet_ranges(series, samples):
    """Measure the near-silent interior, with audible local sound at both ends."""
    if not isinstance(series, list) or len(series) != samples // HOP:
        raise ValueError("Sound levels differ from the presented audio extent")
    for row in series:
        if (not isinstance(row, list) or len(row) != 2 or any(type(n) is not int for n in row)
                or not 0 <= row[0] <= 32768 or not row[0]**2 <= row[1] <= HOP * row[0]**2):
            raise ValueError("Sound levels are outside the decoded sample range")
    flank = FLANK // HOP
    for silent, items in groupby(enumerate(series), key=lambda row: row[1][0] <= PEAK):
        if not silent:
            continue
        indices = [index for index, _ in items]
        start, end = indices[0], indices[-1] + 1
        if start < flank or end + flank > len(series) or not MIN_LENGTH <= (end - start) * HOP <= MAX_LENGTH:
            continue
        before = sum(row[1] for row in series[start-flank:start])
        during = sum(row[1] for row in series[start:end])
        after = sum(row[1] for row in series[end:end+flank])
        if min(before, after) < FLANK_RMS**2 * FLANK:
            continue
        yield {"span": [start * HOP, end * HOP], "peak_units": max(row[0] for row in series[start:end]),
               "before_rms_units": math.isqrt(before // FLANK), "after_rms_units": math.isqrt(after // FLANK),
               "during_rms_units": math.isqrt(during // ((end-start) * HOP))}


def mouth_motion(eligible, start, end):
    before = [p for p in eligible if start - MAX_FRAME_GAP <= p["sample"] < start]
    during = [p for p in eligible if start <= p["sample"] < end]
    after = [p for p in eligible if end <= p["sample"] < end + MAX_FRAME_GAP]
    if not (before and after and len(during) >= 4):
        return None
    involved = [before[-1], *during, after[0]]
    if any(b["sample"] - a["sample"] > MAX_FRAME_GAP for a, b in pairwise(involved)):
        return None
    gaps = [p["gap_units"] for p in during]
    swing = max(gaps) - min(gaps)
    changes = sum(abs(a - b) >= APERTURE_STEP for a, b in pairwise(gaps))
    if swing < APERTURE_SWING or changes < 2:
        return None
    middle = min(during, key=lambda p: (abs(p["sample"] * 2 - start - end), p["index"]))
    return ({"mouth_frames": len(during), "aperture_swing_units": swing, "mouth_steps": changes},
            {"before": before[-1]["index"], "during": middle["index"], "after": after[0]["index"]})


def record(series, frames, samples, origin, *, shot=None):
    eligible = [frame for frame in frames if frame["gap_units"] is not None]
    matches = []
    for event in quiet_ranges(series, samples):
        start, end = event["span"]
        if shot is not None and not shot[0] + FLANK <= start < end <= shot[1] - FLANK:
            continue
        support = mouth_motion(eligible, start, end)
        if support is None:
            continue
        motion, brackets = support
        measurement = {"status": "measured", **event, **motion}
        witness = {"audio_source_pts": [origin + start, origin + end], "picture_source_samples": [origin + start, origin + end],
                   "picture_frames": brackets, KEY: {"near_silent_peak_units": event["peak_units"],
                                                       "visible_mouth_frames": motion["mouth_frames"]}}
        matches.append((measurement, witness))
    return max(matches, key=lambda pair: (pair[0]["span"][1] - pair[0]["span"][0], -pair[0]["span"][0]), default=None)


def observation(witness):
    lo, hi = (sample / RATE for sample in witness["audio_source_pts"])
    frames = witness[KEY]["visible_mouth_frames"]
    return (f"The soundtrack was near-silent from {lo:.3f} to {hi:.3f} s while mouth opening changed across "
            f"{frames} source frames. The soundtrack had measurable sound immediately before and after this range.")


def measurements(measurement):
    return [{"label": label, "value": measurement[key], "unit": unit}
            for key, label, unit in (("mouth_frames", "Visible mouth frames in range", "frames"),
                                     ("peak_units", "Peak sample level in range", "of 32,768"),
                                     ("before_rms_units", "Sound level just before", "of 32,768 RMS"),
                                     ("after_rms_units", "Sound level just after", "of 32,768 RMS"))]
