"""Portable selection and source bindings for adjacent brief speech windows."""

from __future__ import annotations

import math
import re

from app.evidence.audio_long_review_binding import _validate_assets
from app.evidence.audio_long_review_binding import supported_source as supported_native_source

KEY = "brief_voice_review"
SCHEMA = "chipmunk.brief-voice-review/1"
DETECTOR_ID = "aud.brief_voice_review"
TITLE = "Brief speech windows"
RATE, SAMPLES, WIDTH, STEP = 16000, 192000, 12000, 8000
SOURCE_CODECS = frozenset({"pcm_s16le", "flac"})
GRID = tuple(range(0, SAMPLES - WIDTH + 1, STEP))
POINT = 6.642364579351768
SHA = re.compile(r"[a-f0-9]{64}")
LIMITATION = ("Adjacent windows share some audio. Ordinary edits, overlapping voices and background noise can "
              "also produce elevated responses. The marked windows are listening ranges and carry zero weight "
              "in the overall AI assessment. The schedule ends at 11.750 s, leaving the final 250 ms outside this check.")
REASONS = {
    "flagged": "An elevated response repeated in adjacent 750 ms speech windows.",
    "no_flag": "No elevated response repeated in adjacent checked windows. Brief edits can remain undetected.",
    "unassessed": "No two adjacent windows contained enough detected speech to compare.",
}
TIMEOUT = "The brief speech check ran out of time."
TIME_BUDGET = "The brief speech check did not have enough analysis time."
UNAVAILABLE = "The brief speech check could not complete."
SKIP_REASONS = {TIMEOUT, TIME_BUDGET, UNAVAILABLE}


def supported_source(source: dict) -> bool:
    return (source.get("samples") == SAMPLES and source.get("source_codec") in SOURCE_CODECS
            and supported_native_source(source))


def selected_indices(checks: list[dict]) -> list[int]:
    return sorted({j for i, (first, second) in enumerate(zip(checks, checks[1:]))
                   if first["elevated"] is second["elevated"] is True for j in (i, i + 1)})


def skipped(reason: str) -> dict:
    return {"status": "skipped", "reason": reason}


def status(checks, selected):
    if selected:
        return "flagged"
    pair_checked = any(first["elevated"] is not None and second["elevated"] is not None
                       for first, second in zip(checks, checks[1:]))
    return "no_flag" if pair_checked else "unassessed"


def presentation(checks: list[dict], *, reason: str | None = None) -> dict:
    outcome = "skipped" if reason is not None else status(checks, selected_indices(checks))
    return {"schema": SCHEMA, "status": outcome, "reason": reason if reason is not None else REASONS[outcome],
            "limitation": LIMITATION, "affects_overall_confidence": False,
            "scheduled_windows": len(GRID), "speech_windows_checked": sum(row["elevated"] is not None for row in checks),
            "window_checks": checks}


def _validate_checks(checks):
    if not isinstance(checks, list) or len(checks) != len(GRID):
        raise ValueError("The brief speech review has incomplete window admissions")
    for row, start in zip(checks, GRID, strict=True):
        if (not isinstance(row, dict) or set(row) != {"start_sample", "speech_samples", "elevated"}
                or type(row["start_sample"]) is not int or row["start_sample"] != start
                or type(row["speech_samples"]) is not int or not 0 <= row["speech_samples"] <= WIDTH):
            raise ValueError("The brief speech review differs from its scheduled speech admissions")
        if ((row["elevated"] is not None and type(row["elevated"]) is not bool)
                or (row["elevated"] is not None) != (row["speech_samples"] * 2 >= WIDTH)):
            raise ValueError("The brief speech response differs from its recorded speech coverage")


def raw_checks(rows):
    if not isinstance(rows, list) or len(rows) != len(GRID):
        raise ValueError("Incomplete recorded brief speech readings")
    checks = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"start_sample", "speech_samples", "logit_hex", "pcm_sha256"}:
            raise ValueError("Unexpected brief speech reading")
        score = row["logit_hex"]
        elevated = None
        if isinstance(score, str):
            try:
                value = float.fromhex(score)
            except OverflowError as error:
                raise ValueError("Invalid exact brief speech response") from error
            if not math.isfinite(value) or value.hex() != score:
                raise ValueError("Invalid exact brief speech response")
            elevated = value > POINT
        elif score is not None:
            raise ValueError("Invalid brief speech response")
        if not isinstance(row["pcm_sha256"], str) or not SHA.fullmatch(row["pcm_sha256"]):
            raise ValueError("Invalid brief speech input digest")
        checks.append({"start_sample": row["start_sample"], "speech_samples": row["speech_samples"], "elevated": elevated})
    _validate_checks(checks)
    return checks


def window_fields(i):
    start, end = GRID[i], GRID[i] + WIDTH
    return {"id": f"B750-{i + 1:02}", "start_sample": start, "end_sample": end,
            "start_s": start / RATE, "end_s": end / RATE}


def validate(review, source):
    if not supported_source(source) or not isinstance(review, dict):
        raise ValueError("The brief speech review requires an exact twelve-second native mono PCM16 WAV or FLAC")
    reason = review.get("reason") if review.get("status") == "skipped" else None
    checks = review.get("window_checks")
    if review.get("status") == "skipped":
        if reason not in SKIP_REASONS or checks != []:
            raise ValueError("Invalid unavailable brief speech review")
    else:
        _validate_checks(checks)
    expected = presentation(checks, reason=reason)
    windows = review.get("windows")
    indices = selected_indices(checks)
    if (set(review) != {*expected, "windows"} or {key: review[key] for key in expected} != expected
            or review["affects_overall_confidence"] is not False or type(review["speech_windows_checked"]) is not int
            or not isinstance(windows, list)
            or len(windows) != len(indices)):
        raise ValueError("The brief speech review differs from its recorded selection")
    for row, i in zip(windows, indices, strict=True):
        expected_window = window_fields(i)
        start, end = expected_window["start_sample"], expected_window["end_sample"]
        timestamps = [{"start_sample": start, "end_sample": end, "source_start_pts": start, "source_end_pts": end,
                       "time_base": "1/16000"}]
        if (not isinstance(row, dict) or set(row) != {*expected_window, "pcm_sha256", "source_timestamp_ranges", "artifacts"}
                or any(row[key] != value for key, value in expected_window.items())
                or any(type(row[key]) is not int for key in ("start_sample", "end_sample"))
                or not isinstance(row["pcm_sha256"], str) or not SHA.fullmatch(row["pcm_sha256"])
                or row["source_timestamp_ranges"] != timestamps
                or any(type(row["source_timestamp_ranges"][0][key]) is not int for key in timestamps[0] if key != "time_base")):
            raise ValueError("The brief speech window differs from its exact source interval")
        if not isinstance(row["artifacts"], dict) or set(row["artifacts"]) != {"selected_clip", "waveform"}:
            raise ValueError("The brief speech waveform and listening clip are required")
        _validate_assets(row["artifacts"], WIDTH)
