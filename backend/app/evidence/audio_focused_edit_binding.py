"""Public source binding for the focused speech edit review."""

from __future__ import annotations

import re

from app.evidence.audio_short_review_binding import (
    LOSSLESS_FORMATS,
    LOSSLESS_RATE,
    RATE,
    SAMPLES,
    _continuous_clock,
    source_range,
    supported_resampled_lossless,
)

KEY = "focused_speech_edit_review"
SCHEMA = "chipmunk.focused-speech-edit-review/1"
RESAMPLED_SCHEMA = "chipmunk.focused-speech-edit-review/2"
DETECTOR_ID = "aud.focused_speech_edit"
TITLE = "Focused speech edit review"
LIMITATION = ("The marked window is a lead for closer listening. Ordinary editing and overlapping speech can affect "
              "this response. This check carries zero weight in the overall assessment.")
REASONS = {
    "flagged": "This window shows speech patterns associated with synthetic voice edits in our tests.",
    "no_flag": "No checked short window returned an elevated response. An edit can still be present.",
    "no_speech": "No short window contained enough detected speech for this check.",
    "time": "There was insufficient analysis time for the focused speech edit review.",
    "unavailable": "The focused speech edit review was unavailable for this analysis.",
}
STEP = RATE // 10
WIDTHS = (RATE // 2, 3 * RATE // 4, RATE)
WINDOWS = sum((SAMPLES - width) // STEP + 1 for width in WIDTHS)
SCORE_SCALE = 100_000_000
# Strict point frozen before held evaluation. The integer preserves all eight places in the stored run.
REVIEW_POINT = 219099886


def supported_source(source: dict) -> bool:
    return ((source.get("source_container"), source.get("source_codec")) in LOSSLESS_FORMATS
            and source.get("source_channels") == 1 and source.get("source_bit_depth") == 16
            and _continuous_clock(source))


def original_range(start: int, end: int) -> dict:
    ratio = LOSSLESS_RATE // RATE
    return {"start_pts": ratio * start, "end_pts": ratio * end, "time_base": "1/48000"}


def on_grid(start: int, end: int) -> bool:
    return (type(start) is int and type(end) is int and 0 <= start < end <= SAMPLES
            and start % STEP == 0 and end - start in WIDTHS)


def validate(review: dict, source: dict) -> None:
    fields = {"schema", "title", "status", "reason", "limitation", "affects_overall_confidence",
              "checked_windows", "interval", "start_s", "end_s", "source_timestamp_ranges", "artifacts"}
    if not isinstance(review, dict):
        raise ValueError("Missing focused speech review")
    schema = review.get("schema")
    resampled = schema == RESAMPLED_SCHEMA
    if resampled:
        fields.add("original_timestamp_range")
    eligible = supported_resampled_lossless(source) if resampled else supported_source(source)
    if (set(review) != fields or schema not in (SCHEMA, RESAMPLED_SCHEMA)
            or review["title"] != TITLE or review["limitation"] != LIMITATION
            or review["affects_overall_confidence"] is not False or not eligible):
        raise ValueError("Focused speech review does not match its supported source")
    status, reason, count = (review[key] for key in ("status", "reason", "checked_windows"))
    if type(status) is not str or status not in {"flagged", "no_flag", "skipped"} or type(reason) is not str:
        raise ValueError("Invalid focused speech review disposition")
    reasons = {REASONS[key] for key in ("no_speech", "time", "unavailable")} if status == "skipped" else {REASONS.get(status)}
    if (reason not in reasons
            or type(count) is not int or not 0 <= count <= WINDOWS or (count == 0) != (status == "skipped")):
        raise ValueError("Invalid focused speech review disposition")
    interval, assets = review["interval"], review["artifacts"]
    if status != "flagged":
        if (interval is not None or assets != {} or review["source_timestamp_ranges"] != []
                or review["start_s"] is not None or review["end_s"] is not None
                or review.get("original_timestamp_range") is not None):
            raise ValueError("An unflagged focused speech review includes a selected interval")
        return
    if not isinstance(interval, dict) or set(interval) != {"start_sample", "end_sample"}:
        raise ValueError("Missing focused speech interval")
    start, end = (interval[key] for key in ("start_sample", "end_sample"))
    if (not on_grid(start, end) or review["start_s"] != start / RATE or review["end_s"] != end / RATE
            or review["source_timestamp_ranges"] != [source_range(source, start, end)]
            or (resampled and review["original_timestamp_range"] != original_range(start, end))
            or not isinstance(assets, dict) or set(assets) != {"selected_clip", "waveform"}):
        raise ValueError("Focused speech media falls outside its measured source range")
    clip, waveform = assets["selected_clip"], assets["waveform"]
    if (any(not isinstance(asset, dict) or not re.fullmatch(r"[a-f0-9]{64}", str(asset.get("sha256", "")))
            for asset in assets.values())
            or (clip.get("media_type"), clip.get("samples"), clip.get("sample_rate"), clip.get("channels"))
            != ("audio/wav", end - start, RATE, 1)
            or (waveform.get("media_type"), waveform.get("width"), waveform.get("height")) != ("image/png", 960, 280)):
        raise ValueError("Invalid focused speech media")
