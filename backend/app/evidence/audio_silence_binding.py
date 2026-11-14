"""Portable validation of digital-silence locations and measurement scope."""
from __future__ import annotations

import math

RATE = 16000
KEY = "digital_silence"
SCHEMA = "chipmunk.audio-digital-silence/1"
CONTEXT_SCHEMA = "chipmunk.acoustic-context/5"
VERSION_SUFFIX = "+silence"
TITLE = "Exact digital silence"
MINIMUM = RATE // 10
FLANK = RATE // 10
MIN_FLANK_DBFS = -50.0
MAXIMUM_EVENTS = 3
METHOD = {"minimum_zero_samples": MINIMUM, "flank_samples": FLANK, "minimum_flank_rms_dbfs": MIN_FLANK_DBFS,
          "maximum_events": MAXIMUM_EVENTS}
LIMITATION = ("This check locates exact silence in the decoded mono track. A mute, a noise gate, or ordinary "
              "editing can cause it. A speech pause with audible background sound has nonzero samples.")


def continuous_spans(timeline):
    spans = []
    for block in timeline:
        a, b = block["start_sample"], block["end_sample"]
        if spans and a == spans[-1][1] and block["source_pts"] == spans[-1][2]:
            spans[-1] = (spans[-1][0], b, block["source_pts"] + b - a)
        else:
            spans.append((a, b, block["source_pts"] + b - a))
    return [(a, b) for a, b, _ in spans if b - a >= MINIMUM + 2 * FLANK]


def validate(record, timeline, method):
    spans = continuous_spans(timeline)
    coverage = sum(b - a for a, b in spans)
    count = record["qualifying_intervals"]
    selected = record["selected"]
    if (record["schema"] != SCHEMA or record["affects_overall_confidence"] is not False or method != METHOD
            or type(record["checked_samples"]) is not int or record["checked_samples"] != coverage
            or type(count) is not int or not 0 <= count <= coverage // MINIMUM
            or not isinstance(selected, list) or len(selected) != min(count, MAXIMUM_EVENTS)):
        raise ValueError("Invalid digital-silence coverage")
    previous = None
    selected_ranges = {}
    for row in selected:
        a, b = row["start_sample"], row["end_sample"]
        levels = row["flank_rms_dbfs"]
        if (type(a) is not int or type(b) is not int or b - a < MINIMUM
                or not any(x <= a - FLANK < b + FLANK <= y for x, y in spans)
                or not isinstance(levels, list) or len(levels) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or not MIN_FLANK_DBFS <= value <= 0 for value in levels)
                or (previous is not None and previous >= (a - b, a))
                or any(a < d and b > c for c, d in selected_ranges)):
            raise ValueError("Invalid digital-silence interval or measured flanks")
        previous = (a - b, a)
        selected_ranges[a, b] = levels
    return selected_ranges
