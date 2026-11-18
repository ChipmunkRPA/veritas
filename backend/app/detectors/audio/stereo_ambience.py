# SPDX-License-Identifier: Apache-2.0
"""Compare brief changes in stereo difference with the sound on both sides."""

from __future__ import annotations

import numpy as np

RATE = 16_000
FRAME = 640
HOP = 320
FLANK = 10
GAP = 1
WIDTHS = (9, 12, 15, 19, 24, 31, 39, 49, 60)
BANDS = ((180, 580), (580, 1460), (1460, 3200), (3200, 6800))
SIDE_FLOOR = -78.0
MID_FLOOR = -49.0
MINIMUM_FRAMES = WIDTHS[0] + 2 * (FLANK + GAP)
KINDS = ("reduction", "spectrum")


def frames(stereo):
    """Common and difference energy per frequency band, in dBFS, on the aligned clock."""
    if stereo.ndim != 2 or stereo.shape[1] != 2 or len(stereo) < FRAME:
        raise ValueError("Expected two continuous decoded audio channels")
    signal = stereo.astype(np.float64) / 65536.0
    window = np.hanning(FRAME)
    indices = np.arange(0, len(signal) - FRAME + 1, HOP)
    pieces = signal[indices[:, None] + np.arange(FRAME)]
    common = pieces[:, :, 0] + pieces[:, :, 1]
    difference = pieces[:, :, 0] - pieces[:, :, 1]
    bins = np.fft.rfftfreq(FRAME, 1 / RATE)
    scale = 2.0 / (FRAME * np.dot(window, window))

    def energy(values):
        powers = np.abs(np.fft.rfft(values * window)) ** 2
        return scale * np.stack([powers[:, (bins >= lo) & (bins < hi)].sum(axis=1) for lo, hi in BANDS], axis=1)

    mid, side = energy(common), energy(difference)
    levels = np.column_stack((side, mid.sum(axis=1), side.sum(axis=1)))
    return 10 * np.log10(np.maximum(levels, 1e-14))


def _rolling(data, width, fraction):
    windows = np.lib.stride_tricks.sliding_window_view(data, width, axis=0)
    rank = int((width - 1) * fraction)
    return np.partition(windows, rank, axis=-1)[..., rank]


def _probabilities(levels):
    linear = 10 ** (levels / 10)
    return linear / linear.sum(axis=1, keepdims=True)


def _difference(first, second):
    return np.abs(first - second).sum(axis=1) / 2


def _proposal(start, width, before, during, after, score, bands):
    edge = FRAME // 2 - HOP // 2
    return {"start_sample": int(start * HOP + edge), "end_sample": int((start + width) * HOP + edge),
            "score": round(float(score), 4), "side_dbfs": [round(float(part[-1]), 3) for part in (before, during, after)],
            "common_dbfs": [round(float(part[-2]), 3) for part in (before, during, after)],
            "band_drop_db": [round(float(n), 3) for n in bands]}


def search(levels):
    """Find the best bounded reduction and change-of-spectrum proposals separately."""
    if len(levels) < MINIMUM_FRAMES:
        return False, {kind: None for kind in KINDS}
    surroundings = _rolling(levels, FLANK, .5)
    candidates = {kind: [] for kind in KINDS}
    assessed = False
    distance = FLANK + GAP
    for width in WIDTHS:
        indices = np.arange(distance, len(levels) - width - distance + 1)
        if not len(indices):
            break
        before, after = surroundings[indices - distance], surroundings[indices + width + GAP]
        inner = _rolling(levels, width, .5)[indices]
        upper = _rolling(levels, width, .75)[indices]
        flanks = np.minimum(before, after)
        reliable = (flanks[:, :len(BANDS)] >= SIDE_FLOOR) & (np.abs(before[:, :len(BANDS)] - after[:, :len(BANDS)]) <= 9)
        audible = (inner[:, -2] >= MID_FLOOR) & (before[:, -2] >= MID_FLOOR - 7) & (after[:, -2] >= MID_FLOOR - 7)
        available = (reliable.sum(axis=1) >= 2) & audible
        assessed |= bool(available.any())
        drops = flanks[:, :len(BANDS)] - upper[:, :len(BANDS)]
        strongest = np.sort(np.where(reliable, drops, -140.), axis=1)[:, -2]
        common_drop = np.maximum(0., flanks[:, -2] - inner[:, -2] - 3.)
        scores = strongest - common_drop
        left, center, right = (_probabilities(levels[:, :len(BANDS)]) for levels in (before, inner, after))
        drift = _difference(left, right)
        shape = np.minimum(_difference(left, center), _difference(right, center)) - drift
        for kind, values, eligible in (
            ("reduction", scores, available),
            ("spectrum", shape, available & (drift <= .23) & (inner[:, -1] >= SIDE_FLOOR + 3)),
        ):
            if eligible.any():
                best = int(np.argmax(np.where(eligible, values, -150.)))
                candidates[kind].append(_proposal(indices[best], width, before[best], inner[best], after[best],
                                                  values[best], drops[best]))
    result = {}
    for kind, options in candidates.items():
        if options:
            strongest = max(row["score"] for row in options)
            allowance = 1.5 if kind == "reduction" else .015
            options = [row for row in options if row["score"] >= strongest - allowance]
            result[kind] = max(options, key=lambda row: (row["end_sample"] - row["start_sample"], row["score"], -row["start_sample"]))
        else:
            result[kind] = None
    return assessed, result
