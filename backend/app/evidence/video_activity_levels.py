# SPDX-License-Identifier: Apache-2.0
"""Exact source channel and difference levels for a reported speech decline."""

from __future__ import annotations

from itertools import accumulate, islice
from math import isfinite, log10
from struct import iter_unpack

from app.evidence.video_silent_mouth import mouth_motion

SQUARES = "stereo_level_squares"
KEY = "stereo_levels"
CHANNELS = ("Left channel", "Right channel", "Channel difference")
FACTORS = (1, 1, 4)
BLOCK = 512
FLANK = 5
GAP = 1
WIDTHS = (7, 9, 11, 13, 16, 19)
MIN_FLANK = .35


def squared_samples(stereo):
    try:
        source = memoryview(stereo)
    except TypeError as error:
        raise ValueError("Stereo levels need a signed 16-bit sample buffer") from error
    if source.ndim != 2 or source.shape[1] != 2 or source.format not in ("h", "@h", "=h", "<h", ">h"):
        raise ValueError("Stereo levels require the exact two-channel signed 16-bit input")
    order = source.format[:-1] or "="
    exact = iter_unpack(order + "hh", source.tobytes())
    rows = []
    for _ in range(len(source) // BLOCK):
        left = right = difference = 0
        for a, b in islice(exact, BLOCK):
            left += a * a
            right += b * b
            difference += (a - b) ** 2
        rows.append([left, right, difference])
    return rows


def validate(trace, samples):
    limits = (BLOCK * 32768**2, BLOCK * 32768**2, BLOCK * 65535**2)
    if (not isinstance(trace, list) or len(trace) != samples // BLOCK
            or any(not isinstance(row, list) or len(row) != len(limits)
                   or any(type(value) is not int or not 0 <= value <= limit
                          for value, limit in zip(row, limits, strict=True)) for row in trace)):
        raise ValueError("Stereo sound levels differ from their signed sample range")


def measure(trace, span):
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


def scan_both(original, enhanced, frames):
    """Replay the two-reading speech decline from its bounded numeric witness."""
    signals = original, enhanced
    if len(original) != len(enhanced) or any(not isfinite(value) or not 0 <= value <= 1
                                            for signal in signals for value in signal):
        raise ValueError("Speech activity must be bounded and share the same sample clock")
    prefix = [list(accumulate(signal, initial=0.0)) for signal in signals]
    pictures = [frame for frame in frames if frame["gap_units"] is not None]
    context = FLANK + GAP
    result = {"assessed": False, "reading": None}
    for width in WIDTHS:
        for start in range(context, len(original) - width - context + 1):
            end = start + width
            before = [(p[start - GAP] - p[start - context]) / FLANK for p in prefix]
            during = [(p[end] - p[start]) / width for p in prefix]
            after = [(p[end + context] - p[end + GAP]) / FLANK for p in prefix]
            if min(*before, *after) < MIN_FLANK:
                continue
            span = [start * BLOCK, end * BLOCK]
            support = mouth_motion(pictures, *span)
            if support is None:
                continue
            motion, brackets = support
            result["assessed"] = True
            margin = round(1_000_000 * min(min(a, b) - mid for a, mid, b in zip(before, during, after, strict=True)))
            previous = result["reading"]
            if margin <= 0 or (previous is not None and
                               (-margin, *span) >= (-previous["margin_millionths"], *previous["span"])):
                continue
            result["reading"] = {
                "span": span, "margin_millionths": margin, "picture_frames": brackets, **motion,
                "activity_millionths": [[round(1_000_000 * value) for value in part] for part in (before, during, after)],
            }
    return result


def measurements(rows):
    return [{"label": f"{name} sound level, before → during → after",
             "value": " → ".join("Silence" if levels[index] is None else format(levels[index], ".2f")
                                 for levels in (row["rms_dbfs"] for row in rows)),
             "unit": "dBFS RMS"} for index, name in enumerate(CHANNELS)]
