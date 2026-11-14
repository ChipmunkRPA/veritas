# SPDX-License-Identifier: Apache-2.0
"""Bind a bounded sound measurement to the picture clock of a held mouth region."""

from __future__ import annotations

import math
from fractions import Fraction

SCHEMA = "chipmunk.video-mouth-hold-sound/1"
RATE = 16000
HOP = 320
WAV = "mouth_hold_audio"
WAVEFORM = "mouth_hold_waveform"
WAVEFORM_TITLE = "Sound during the mouth-region hold"
SOURCE_CLOCK = "source_pts"
SOURCE_WAVEFORM_TITLE = "Sound during the mouth-region hold (source time)"
MIN_BLOCKS = 6
HOLD_SCHEMAS = frozenset({"chipmunk.video-mouth-local-hold/1", "chipmunk.video-mouth-local-hold/2"})
LIMITATION = ("The mixed soundtrack can include a different speaker, ordinary dubbing or music. "
              "Strong background noise can mask voice-like periodicity. This observation carries zero AI confidence weight.")


def bounds(review, audio, time_base):
    if review.get("schema") not in HOLD_SCHEMAS or review.get("status") != "matched":
        return None
    samples = audio.get("samples")
    if audio.get("timeline_time_base") != f"1/{RATE}" or type(samples) is not int or samples < 1:
        return None
    left, right = review["match"]["frames"]
    clock = Fraction(time_base)
    first = math.ceil(review["pts"][left] * clock * RATE)
    last = math.floor(review["pts"][right] * clock * RATE)
    if first >= last:
        return None

    # Intersect with every decoded block. Only one continuous, unambiguous
    # source interval may supply sound, even when other parts of the file jump.
    cursor, expected_pts = 0, first
    start = stop = None
    for block in audio.get("timeline") or []:
        a, b, pts = (block.get(key) for key in ("start_sample", "end_sample", "source_pts"))
        if any(type(value) is not int for value in (a, b, pts)) or not cursor == a < b <= samples:
            return None
        cursor = b
        lo, hi = max(first, pts), min(last, pts + b - a)
        if lo >= hi:
            continue
        decoded_start = a + lo - pts
        if lo != expected_pts or (stop is not None and decoded_start != stop):
            return None
        if start is None:
            start = decoded_start
        stop, expected_pts = a + hi - pts, hi
    if cursor != samples or start is None or expected_pts != last:
        return None
    return start, stop, first, last


def block_starts(start: int, stop: int) -> range:
    offset = (start + HOP - 1) // HOP * HOP
    return range(offset, stop - 2 * HOP + 1, HOP)


def measurements(record):
    count, periodic = record["blocks"], record["periodic_blocks"]
    return [{"label": "Audio blocks with voice-like periodicity", "value": f"{periodic} of {count}",
             "unit": "40 ms blocks"}]


def observation(record):
    if record["sustained_periodicity"]:
        return "The soundtrack has sustained voice-like periodicity during the measured mouth-region hold."
    return "The soundtrack did not show sustained voice-like periodicity during the measured mouth-region hold."


def validate(review, packet, audio, time_base):
    record = review.get("sound_context")
    if record is None:
        if "mouth_hold_sound" in packet or {WAV, WAVEFORM} & packet["artifacts"].keys():
            raise ValueError("Mouth-hold sound has no committed measurement")
        return
    located = bounds(review, audio, time_base)
    if located is None:
        raise ValueError("Mouth-hold sound has no continuous source coverage")
    start, stop, first, last = located
    count = len(block_starts(start, stop))
    if (set(record) != {"schema", "samples", "source_pts", "time_base", "blocks", "periodic_blocks", "sustained_periodicity"}
            or record["schema"] != SCHEMA or record["samples"] != [start, stop] or record["source_pts"] != [first, last]
            or record["time_base"] != f"1/{RATE}" or type(record["blocks"]) is not int
            or record["blocks"] != count or count < MIN_BLOCKS
            or type(record["periodic_blocks"]) is not int or not 0 <= record["periodic_blocks"] <= count
            or record["sustained_periodicity"] is not (3 * record["periodic_blocks"] >= 2 * count)
            or packet["mouth_hold_sound"] != record
            or LIMITATION not in packet["limitations"]):
        raise ValueError("Mouth-hold sound differs from its source interval or recorded measurement")
    wav, waveform = (packet["artifacts"][key] for key in (WAV, WAVEFORM))
    if (wav.get("media_type") != "audio/wav" or wav.get("samples") != stop - start
            or wav.get("sample_rate") != RATE or wav.get("channels") != 1
            or waveform.get("media_type") != "image/png"
            or (waveform.get("width"), waveform.get("height")) != (960, 280)):
        raise ValueError("Mouth-hold sound assets differ from their source interval")
    if "time_axis" in waveform and waveform["time_axis"] != SOURCE_CLOCK:
        raise ValueError("Mouth-hold waveform uses an unknown time axis")
