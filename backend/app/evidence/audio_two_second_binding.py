"""Portable evidence rules for the two-second scan on longer speech."""

from __future__ import annotations

import math
import re
from collections import defaultdict

from app.evidence import audio_short_review_binding as short

KEY = "two_second_review"
SCHEMA = "chipmunk.extended-short-voice-review/1"
CONFIRMED_SCHEMA = "chipmunk.extended-short-voice-review/2"
DETECTOR_ID = "aud.extended_short_voice_review"
TITLE = "Two-second speech windows"
RATE = short.RATE
MINIMUM, MAXIMUM = 9 * RATE, 14 * RATE
LIMITATION = ("These are elevated responses in the complete two-second windows. Overlapping voices and ordinary editing "
              "can produce similar responses. The review carries zero weight in the overall AI assessment.")
REASONS = {
    "flagged": "At least one two-second window produced an AI-like speech response.",
    "no_flag": "No checked two-second window was flagged. Edits can remain undetected.",
    "unassessed": "None of the two-second windows contained enough detected speech to be checked.",
}
LIMITATIONS = {
    SCHEMA: LIMITATION,
    CONFIRMED_SCHEMA: ("Both windows contain some of the same audio. Overlapping voices and ordinary editing can "
                       "produce similar responses. The review carries zero weight in the overall AI assessment."),
}
REASONS_BY_SCHEMA = {
    SCHEMA: REASONS,
    CONFIRMED_SCHEMA: {
        **REASONS,
        "flagged": "An AI-like response repeated in windows that overlap by at least one second.",
        "no_flag": "No elevated window was confirmed by a second overlapping response. Edits can remain undetected.",
    },
}
TIMEOUT = "The two-second scan ran out of analysis time."
TIME_BUDGET = "The two-second scan did not have enough analysis time."
UNAVAILABLE = "The optional two-second scan could not complete."
SKIP_REASONS = frozenset({TIMEOUT, TIME_BUDGET, UNAVAILABLE})
SHA = re.compile(r"[0-9a-f]{64}")


def excerpt_starts(samples):
    if type(samples) is not int or not MINIMUM <= samples <= MAXIMUM:
        return []
    end = samples - short.SAMPLES
    return sorted({*range(0, end + 1, 2 * RATE), end})


def supported_source(source):
    from app.evidence.audio_long_review_binding import supported_source as supported_long_source

    return bool(excerpt_starts(source.get("samples"))) and supported_long_source(source)


def _absolute_range(excerpt, index):
    start, end = short.GRID[index]
    offset = excerpt["start_sample"]
    return offset + start, offset + end


def _cut(interval, excerpts):
    return {"start_sample": interval[0], "end_sample": interval[1], "supporting_excerpts": excerpts}


def regions(excerpts, *, schema=SCHEMA):
    selected = defaultdict(list)
    elevated = defaultdict(list)
    for index, excerpt in enumerate(excerpts):
        if (chosen := excerpt["selected_window"]) is not None:
            selected[_absolute_range(excerpt, chosen)].append(index)
        if schema == CONFIRMED_SCHEMA:
            for window in excerpt["elevated_windows"]:
                elevated[_absolute_range(excerpt, window)].append(index)
    if schema == SCHEMA:
        return [_cut(interval, support) for interval, support in sorted(selected.items())]
    confirmed, shown_pairs = [], set()
    for interval, support in sorted(selected.items()):
        start, _ = interval
        # Both windows span two seconds. A difference of at most one second implies the required overlap.
        nearby = (other for other in elevated if 0 < abs(other[0] - start) <= RATE)
        match = min(nearby, key=lambda other: (-abs(other[0] - start), other[0]), default=None)
        if match is not None:
            row = _cut(interval, support) | {"confirming_window": _cut(match, elevated[match])}
            pair = frozenset((start, match[0]))
            if pair in shown_pairs:
                row["same_pair_as"] = match[0]
            shown_pairs.add(pair)
            confirmed.append(row)
    return confirmed


def media_windows(review):
    for row in review.get("regions", ()):
        yield row
        if (confirming := row.get("confirming_window")) is not None:
            yield confirming


def status(excerpts, marked):
    if marked:
        return "flagged"
    return "no_flag" if any(count >= RATE for row in excerpts for count in row["speech_samples"]) else "unassessed"


def skipped(reason, *, schema):
    return {"reason": reason, **({"schema": schema} if schema == CONFIRMED_SCHEMA else {})}


def _require(valid, message):
    if not valid:
        raise ValueError(message)


def _validate_excerpt(row, start):
    counts = row["speech_samples"]
    _require(type(row["start_sample"]) is int and row["start_sample"] == start
             and SHA.fullmatch(row["pcm_sha256"])
             and isinstance(counts, list) and len(counts) == len(short.GRID)
             and all(type(count) is int and 0 <= count <= 2 * RATE for count in counts),
             "Invalid two-second speech excerpt or admission counts")


def summarize_readings(readings, samples, *, schema=SCHEMA):
    """Convert the exact admitted model readings into public window locations."""
    starts = excerpt_starts(samples)
    _require(bool(starts) and isinstance(readings, list) and len(readings) == len(starts), "Unexpected speech excerpt schedule")
    summary = []
    for row, start in zip(readings, starts, strict=True):
        _require(set(row) == {"start_sample", "pcm_sha256", "speech_samples", "responses"}, "Unexpected speech excerpt reading")
        _validate_excerpt(row, start)
        eligible = [pair for pair, count in zip(short.GRID, row["speech_samples"], strict=True) if count >= RATE]
        _require(isinstance(row["responses"], list) and len(row["responses"]) == len(eligible),
                 "Two-second response coverage differs from detected speech")
        for reading, pair in zip(row["responses"], eligible, strict=True):
            a, b, logit = reading["start_sample"], reading["end_sample"], reading["logit"]
            _require(set(reading) == {"start_sample", "end_sample", "logit"} and type(a) is type(b) is int
                     and (a, b) == pair and type(logit) in (int, float) and math.isfinite(logit), "Invalid two-second window response")
        chosen = short.selected_range(row["responses"])
        public = {key: row[key] for key in ("start_sample", "pcm_sha256", "speech_samples")}
        public["selected_window"] = short.GRID.index((chosen["start_sample"], chosen["end_sample"])) if chosen else None
        if schema == CONFIRMED_SCHEMA:
            public["elevated_windows"] = [short.GRID.index(pair) for reading, pair in zip(row["responses"], eligible, strict=True)
                                          if reading["logit"] > short.REVIEW_POINT]
        summary.append(public)
    return summary


def _validate_window(row, planned, source):
    from app.evidence.audio_long_review_binding import _validate_assets

    start, end, support = (row[key] for key in ("start_sample", "end_sample", "supporting_excerpts"))
    _require(set(row) == {*planned, "pcm_sha256", "source_timestamp_ranges", "artifacts"}
             and type(start) is type(end) is int and isinstance(support, list) and all(type(i) is int for i in support)
             and all(row[key] == planned[key] for key in ("start_sample", "end_sample", "supporting_excerpts"))
             and SHA.fullmatch(row["pcm_sha256"])
             and row["source_timestamp_ranges"] == [short.source_range(source, start, end)]
             and set(row["artifacts"]) == {"selected_clip", "waveform"}, "Two-second exhibit differs from its source interval")
    _validate_assets(row["artifacts"], end - start)
    if "same_pair_as" in planned:
        _require(type(row["same_pair_as"]) is int and row["same_pair_as"] == planned["same_pair_as"],
                 "Repeated presentation does not refer to the same two speech windows")
    if (confirming := planned.get("confirming_window")) is not None:
        _validate_window(row["confirming_window"], confirming, source)


def _validate_elevated(row):
    elevated, selected, counts = (row[key] for key in ("elevated_windows", "selected_window", "speech_samples"))
    _require(isinstance(elevated, list)
             and all(type(i) is int and 0 <= i < len(short.GRID) and counts[i] >= RATE for i in elevated)
             and elevated == sorted(set(elevated)) and (selected is None if not elevated else selected in elevated),
             "Two-second confirmation requires admitted elevated windows")


def validate(review, source):
    try:
        schema = review["schema"]
        _require(set(review) == {"schema", "title", "status", "reason", "limitation", "affects_overall_confidence", "excerpts", "regions"}
                 and schema in LIMITATIONS and review["title"] == TITLE and review["limitation"] == LIMITATIONS[schema]
                 and review["affects_overall_confidence"] is False and supported_source(source),
                 "Two-second evidence is outside its native source scope")
        excerpts, shown = review["excerpts"], review["regions"]
        _require(isinstance(excerpts, list) and isinstance(shown, list), "Invalid two-second evidence lists")
        if review["status"] == "skipped":
            _require(not excerpts and not shown and review["reason"] in SKIP_REASONS, "Skipped two-second scan carries measurements")
            return
        starts = excerpt_starts(source["samples"])
        _require(len(excerpts) == len(starts), "Incomplete set of speech excerpts")
        fields = {"start_sample", "pcm_sha256", "speech_samples", "selected_window"}
        if schema == CONFIRMED_SCHEMA:
            fields.add("elevated_windows")
        for row, start in zip(excerpts, starts, strict=True):
            _require(set(row) == fields, "Unexpected public speech excerpt fields")
            _validate_excerpt(row, start)
            index = row["selected_window"]
            _require(index is None or (type(index) is int and 0 <= index < len(short.GRID) and row["speech_samples"][index] >= RATE),
                     "Flagged window has insufficient speech or falls outside its excerpt")
            if schema == CONFIRMED_SCHEMA:
                _validate_elevated(row)
        expected = regions(excerpts, schema=schema)
        wanted = status(excerpts, expected)
        _require(review["status"] == wanted and review["reason"] == REASONS_BY_SCHEMA[schema][wanted]
                 and len(shown) == len(expected), "Public two-second status or ranges differ from the excerpt selections")
        for row, planned in zip(shown, expected, strict=True):
            _validate_window(row, planned, source)
    except (TypeError, KeyError, AttributeError) as error:
        raise ValueError("Incomplete two-second speech evidence") from error
