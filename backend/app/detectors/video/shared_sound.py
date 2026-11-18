# SPDX-License-Identifier: Apache-2.0
"""Locate a temporary loss of shared stereo sound while channel differences persist."""

from __future__ import annotations

import numpy as np

from app.detectors.audio.stereo_ambience import BANDS, FRAME, HOP, _rolling, band_powers
from app.evidence.video_silent_mouth import mouth_motion

KEY = "shared_sound_drop"
FLANK = 11
GAP = 1
WIDTHS = (9, 12, 15, 19, 24, 31, 39)
ARMS = ("direct", "difference_adjusted")
MID_FLOOR = -55.
SIDE_FLOOR = -66.
MAX_SIDE_CHANGE = 5.
MIN_COMMON_DROP = 1.5


def levels(stereo):
    """Store milli-dB levels for four shared bands followed by four difference bands."""
    common, difference = band_powers(stereo)
    return np.rint(10000 * np.log10(np.maximum(np.column_stack((common, difference)), 1e-14))).astype(int).tolist()


def _room_adjusted(shared, difference):
    """Estimate a stable part of the common sound using the quieter moments of the same shot."""
    floor_ratio = np.quantile(shared - difference, .20, axis=0)
    observed = 10 ** (shared / 10)
    estimated = .90 * 10 ** ((difference + floor_ratio) / 10)
    # Keep a finite lower bound instead of treating the residual as a separated voice.
    remaining = np.maximum(np.maximum(observed - estimated, .035 * observed), 1e-14)
    return 10 * np.log10(remaining)


def validate_levels(values, samples):
    bands = 2 * len(BANDS)
    count = max(0, (samples - FRAME) // HOP + 1)
    if (not isinstance(values, list) or len(values) != count
            or any(not isinstance(row, list) or len(row) != bands
                   or any(type(value) is not int or not -140_000 <= value <= 0 for value in row) for row in values)):
        raise ValueError("The stereo levels differ from the presented sample extent or range")


def scan(values, frames, samples, *, bounds=None):
    """Compare audio-only and picture-supported responses on the same native interval grid."""
    if bounds is None:
        bounds = 0, samples
    lower, upper = bounds
    # Require each 40 ms observation to lie wholly inside the shot.
    first = (lower + HOP - 1) // HOP
    final = (upper - FRAME) // HOP + 1
    points = max(0, final - first)
    distance = FLANK + GAP
    empty = {name: None for name in ARMS}
    if points < WIDTHS[0] + 2 * distance:
        return False, empty, empty.copy()
    shared, side = np.split(np.asarray(values[first:final], dtype=np.float64) / 1000, 2, axis=1)
    adjusted = _room_adjusted(shared, side)
    data = np.column_stack((shared, side, adjusted))
    flanks = _rolling(data, FLANK, .5)
    peaks = {name: [] for name in ARMS}
    assessed = False
    for width in WIDTHS:
        index = np.arange(distance, len(data) - width - distance + 1)
        if not len(index):
            break
        left, right = flanks[index - distance], flanks[index + width + GAP]
        center = _rolling(data, width, .70)[index]
        a, b, c = (np.split(length, 3, axis=1) for length in (left, center, right))
        side_min = np.minimum(np.minimum(a[1], b[1]), c[1])
        side_max = np.maximum(np.maximum(a[1], b[1]), c[1])
        shared_flank = np.minimum(a[0], c[0])
        stable = (side_min >= SIDE_FLOOR) & (side_max - side_min <= MAX_SIDE_CHANGE) & (shared_flank >= MID_FLOOR)
        assessed |= bool((stable.sum(axis=1) >= 2).any())
        raw_drop = shared_flank - b[0]
        eligible = stable & (raw_drop >= MIN_COMMON_DROP)
        net_drop = np.minimum(a[2], c[2]) - b[2]
        # Penalize a change in room level even when it is within the stability guard.
        room_change = np.abs(b[1] - (a[1] + c[1]) / 2)
        for name, effect in zip(ARMS, (raw_drop, net_drop), strict=True):
            margin = np.sort(np.where(eligible, effect - .5 * room_change, -140.), axis=1)[:, -2]
            for i in np.flatnonzero(margin > 0):
                lo = int((first + index[i]) * HOP + (FRAME - HOP) // 2)
                hi = lo + width * HOP
                levels_triplet = [
                    [np.rint(group[band][i] * 1000).astype(int).tolist() for group in (a, b, c)]
                    for band in (0, 1)
                ]
                peaks[name].append({"span": [lo, hi], "margin_millidb": int(round(margin[i] * 1000)),
                                    "shared_millidb": levels_triplet[0], "difference_millidb": levels_triplet[1]})
    eligible_frames = [frame for frame in frames if frame["gap_units"] is not None and lower <= frame["sample"] < upper]
    audio, video = {}, {}
    for name, options in peaks.items():
        ranked = sorted(options, key=lambda row: (-row["margin_millidb"], row["span"][0] - row["span"][1], row["span"][0]))
        audio[name] = ranked[0] if ranked else None
        video[name] = None
        for row in ranked:
            if support := mouth_motion(eligible_frames, *row["span"]):
                motion, brackets = support
                video[name] = {**row, **motion, "picture_frames": brackets}
                break
    return assessed, audio, video
