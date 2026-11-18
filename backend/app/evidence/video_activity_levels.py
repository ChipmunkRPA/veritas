# SPDX-License-Identifier: Apache-2.0
"""Exact source channel and difference levels for a reported speech decline."""

from __future__ import annotations

from math import log10

SQUARES = "stereo_level_squares"
KEY = "stereo_levels"
CHANNELS = ("Left channel", "Right channel", "Channel difference")
FACTORS = (1, 1, 4)


def squared_samples(stereo):
    import numpy as np

    from app.bench.video_speech_activity_gap import BLOCK

    source = np.asarray(stereo)
    if source.ndim != 2 or source.shape[1] != 2 or source.dtype.kind != "i" or source.dtype.itemsize != 2:
        raise ValueError("Stereo levels require the exact two-channel signed 16-bit input")
    length = len(source) // BLOCK * BLOCK
    left, right = source[:length].astype(np.int64).T
    traces = np.stack((left, right, left - right), axis=1)
    return (traces ** 2).reshape(-1, BLOCK, 3).sum(axis=1).tolist()


def validate(trace, samples):
    from app.bench.video_speech_activity_gap import BLOCK

    limits = (BLOCK * 32768**2, BLOCK * 32768**2, BLOCK * 65535**2)
    if (not isinstance(trace, list) or len(trace) != samples // BLOCK
            or any(not isinstance(row, list) or len(row) != len(limits)
                   or any(type(value) is not int or not 0 <= value <= limit
                          for value, limit in zip(row, limits, strict=True)) for row in trace)):
        raise ValueError("Stereo sound levels differ from their signed sample range")


def measure(trace, span):
    from app.bench.video_speech_activity_gap import BLOCK, FLANK, GAP

    start, end = (point // BLOCK for point in span)
    ranges = ((start - GAP - FLANK, start - GAP), (start, end), (end + GAP, end + GAP + FLANK))
    rows = []
    for a, b in ranges:
        if not 0 <= a < b <= len(trace):
            raise ValueError("Stereo level context falls outside presented samples")
        totals = [sum(block[index] for block in trace[a:b]) for index in range(3)]
        normal = (b - a) * BLOCK * 32768**2
        levels = [round(10 * log10(value / (normal * factor)), 2) if value else None
                  for value, factor in zip(totals, FACTORS, strict=True)]
        rows.append({"source_samples": [a * BLOCK, b * BLOCK], "rms_dbfs": levels})
    return rows


def measurements(rows):
    return [{"label": f"{name} sound level, before → during → after",
             "value": " → ".join("Silence" if levels[index] is None else format(levels[index], ".2f")
                                 for levels in (row["rms_dbfs"] for row in rows)),
             "unit": "dBFS RMS"} for index, name in enumerate(CHANNELS)]
