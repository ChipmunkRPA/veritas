"""Portable evidence rules for the two-second scan on longer speech."""

from __future__ import annotations

import math
import re
from collections import defaultdict

from app.evidence import audio_short_review_binding as short

KEY = "two_second_review"
SCHEMA = "chipmunk.extended-short-voice-review/1"
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


def regions(excerpts):
    selected = defaultdict(list)
    for index, excerpt in enumerate(excerpts):
        chosen = excerpt["selected_window"]
        if chosen is not None:
            start, end = short.GRID[chosen]
            offset = excerpt["start_sample"]
            selected[offset + start, offset + end].append(index)
    return [{"start_sample": start, "end_sample": end, "supporting_excerpts": support}
            for (start, end), support in sorted(selected.items())]


def status(excerpts):
    if any(row["selected_window"] is not None for row in excerpts):
        return "flagged"
    return "no_flag" if any(count >= RATE for row in excerpts for count in row["speech_samples"]) else "unassessed"


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


def summarize_readings(readings, samples):
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
        summary.append({key: row[key] for key in ("start_sample", "pcm_sha256", "speech_samples")} | {
            "selected_window": short.GRID.index((chosen["start_sample"], chosen["end_sample"])) if chosen else None})
    return summary


def validate(review, source):
    from app.evidence.audio_long_review_binding import _validate_assets

    try:
        _require(set(review) == {"schema", "title", "status", "reason", "limitation", "affects_overall_confidence", "excerpts", "regions"}
                 and review["schema"] == SCHEMA and review["title"] == TITLE and review["limitation"] == LIMITATION
                 and review["affects_overall_confidence"] is False and supported_source(source),
                 "Two-second evidence is outside its native source scope")
        excerpts, shown = review["excerpts"], review["regions"]
        _require(isinstance(excerpts, list) and isinstance(shown, list), "Invalid two-second evidence lists")
        if review["status"] == "skipped":
            _require(not excerpts and not shown and review["reason"] in SKIP_REASONS, "Skipped two-second scan carries measurements")
            return
        starts = excerpt_starts(source["samples"])
        _require(len(excerpts) == len(starts), "Incomplete set of speech excerpts")
        for row, start in zip(excerpts, starts, strict=True):
            _require(set(row) == {"start_sample", "pcm_sha256", "speech_samples", "selected_window"}, "Unexpected public speech excerpt fields")
            _validate_excerpt(row, start)
            index = row["selected_window"]
            _require(index is None or (type(index) is int and 0 <= index < len(short.GRID) and row["speech_samples"][index] >= RATE),
                     "Flagged window has insufficient speech or falls outside its excerpt")
        wanted, expected = status(excerpts), regions(excerpts)
        _require(review["status"] == wanted and review["reason"] == REASONS[wanted]
                 and len(shown) == len(expected), "Public two-second status or ranges differ from the excerpt selections")
        for row, planned in zip(shown, expected, strict=True):
            start, end, support = (row[key] for key in ("start_sample", "end_sample", "supporting_excerpts"))
            _require(set(row) == {*planned, "pcm_sha256", "source_timestamp_ranges", "artifacts"}
                     and type(start) is type(end) is int and isinstance(support, list) and all(type(i) is int for i in support)
                     and {key: row[key] for key in planned} == planned and SHA.fullmatch(row["pcm_sha256"])
                     and row["source_timestamp_ranges"] == [short.source_range(source, start, end)]
                     and set(row["artifacts"]) == {"selected_clip", "waveform"}, "Two-second exhibit differs from its source interval")
            _validate_assets(row["artifacts"], end - start)
    except (TypeError, KeyError, AttributeError) as error:
        raise ValueError("Incomplete two-second speech evidence") from error
