"""Shared source and public-packet checks for the optional short speech review."""

from __future__ import annotations

import re

RATE = 16000
SAMPLES = 6 * RATE
GRID = tuple((start, start + 2 * RATE) for start in range(0, 5 * RATE, RATE))
SCHEMA = "chipmunk.short-voice-review/1"
DETECTOR_ID = "aud.short_voice_review"
REVIEW_POINT = 6.822523597593649
MP3_REVIEW_POINT = 7.085599896496718
MP3_ORIGIN = 1105
TITLE = "Short speech-window review"
LIMITATION = ("This is a clue for the complete two-second window. It cannot establish AI generation or the boundaries "
              "of an edit. It carries no weight in the overall assessment.")
SCOPE_REASON = ("This check requires Full analysis of six continuous decoded seconds at 16 kHz. "
                "It supports 16-bit mono WAV and FLAC and a verified 48 kb/s mono MP3 profile.")


def source_metadata(mapping: dict) -> dict:
    metadata = {"source_codec": mapping.get("source_codec")}
    keys = {"flac": ("source_container", "source_bit_depth"), "mp3": ("source_container", "source_bit_rate")}
    metadata.update({key: mapping[key] for key in keys.get(metadata["source_codec"], ()) if key in mapping})
    return metadata


def source_metadata_matches(source: dict, mapping: dict) -> bool:
    return all(mapping.get(key) == value for key, value in source_metadata(source).items())


def supported_source(mapping: dict) -> bool:
    codec = mapping.get("source_codec")
    lossless = codec == "pcm_s16le" or (
        codec == "flac" and mapping.get("source_container") == "flac" and mapping.get("source_bit_depth") == 16
    )
    mp3 = codec == "mp3" and mapping.get("source_container") == "mp3" and mapping.get("source_bit_rate") == 48000
    return (
        (lossless or mp3) and mapping.get("samples") == SAMPLES
        and mapping.get("sample_rate") == mapping.get("source_sample_rate") == RATE
        and mapping.get("channels") == mapping.get("source_channels") == 1
        and len(mapping.get("audio_stream_indices", [0])) == 1
        and mapping.get("timeline_time_base") == "1/16000"
        and mapping.get("timeline") == [{"start_sample": 0, "end_sample": SAMPLES, "source_pts": MP3_ORIGIN if mp3 else 0}]
    )


def selected_range(responses: list[dict], codec: str | None = None) -> dict | None:
    best = max(responses, key=lambda item: (item["logit"], -item["start_sample"]), default=None)
    point = MP3_REVIEW_POINT if codec == "mp3" else REVIEW_POINT
    if best is None or best["logit"] <= point:
        return None
    return {key: best[key] for key in ("start_sample", "end_sample")}


def validate(review: dict, source: dict) -> None:
    try:
        _validate(review, source)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete short speech-window evidence") from error


def _validate(review: dict, source: dict) -> None:
    if (set(review) != {"schema", "title", "status", "reason", "limitation", "affects_overall_confidence", "checked_windows",
                        "interval", "start_s", "end_s", "source_timestamp_ranges", "artifacts"}
            or review["schema"] != SCHEMA or review["title"] != TITLE or review["limitation"] != LIMITATION
            or review["affects_overall_confidence"] is not False or review["status"] not in {"flagged", "no_flag", "skipped"}
            or not isinstance(review["reason"], str) or not review["reason"]):
        raise ValueError("Invalid short speech-window evidence status")
    windows = [(w["start_sample"], w["end_sample"]) for w in review["checked_windows"]]
    if (any(set(w) != {"start_sample", "end_sample"} for w in review["checked_windows"])
            or windows != sorted(set(windows)) or not set(windows) <= set(GRID)
            or any(type(a) is not int or type(b) is not int for a, b in windows)
            or (windows and not supported_source(source)) or (review["status"] != "skipped" and not windows)):
        raise ValueError("Invalid short speech-window sample coordinates")
    interval = review["interval"]
    assets = review["artifacts"]
    if review["status"] != "flagged":
        if interval is not None or assets or review["source_timestamp_ranges"] or review["start_s"] is not None or review["end_s"] is not None:
            raise ValueError("An unflagged short-window result includes a selected interval")
        return
    a, b = interval["start_sample"], interval["end_sample"]
    origin = source["timeline"][0]["source_pts"]
    ranges = [{"start_sample": a, "end_sample": b, "source_start_pts": origin + a,
               "source_end_pts": origin + b, "time_base": "1/16000"}]
    if (type(a) is not int or type(b) is not int or (a, b) not in windows
            or review["start_s"] != a / RATE or review["end_s"] != b / RATE or review["source_timestamp_ranges"] != ranges
            or set(assets) != {"selected_clip", "waveform"}):
        raise ValueError("Flagged short-window media is outside its checked source range")
    clip, waveform = assets["selected_clip"], assets["waveform"]
    if (clip["media_type"] != "audio/wav" or clip["samples"] != b - a or clip["sample_rate"] != RATE or clip["channels"] != 1
            or waveform["media_type"] != "image/png" or waveform["width"] != 960 or waveform["height"] != 280
            or any(not re.fullmatch("[a-f0-9]{64}", asset["sha256"]) for asset in assets.values())):
        raise ValueError("Invalid short speech-window media")
