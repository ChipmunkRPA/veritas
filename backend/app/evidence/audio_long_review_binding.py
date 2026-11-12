"""Portable source and decision rules for the bounded long speech review."""

from __future__ import annotations

import re

RATE = 16_000
SPAN = 3 * RATE
SCHEMA = "chipmunk.long-voice-review/1"
DETECTOR_ID = "aud.long_voice_review"
TITLE = "Long speech-window review"
LIMITATION = ("The responses apply to the three-second windows shown. Gaps between windows are outside this check. "
              "The review cannot establish how the speech was made or identify exact edit boundaries. "
              "It carries no weight in the overall assessment.")
DIRECTIONS = {"ai", "original", "inconclusive", "unassessed"}
LABELS = {"ai": "Leans AI-generated in this window", "original": "Leans original in this window",
          "inconclusive": "Inconclusive window", "unassessed": "No usable speech response"}
REASONS = {"corroborated": "At least two separate windows leaned AI-generated, and none leaned original.",
           "no_corroboration": "The sampled windows did not produce corroborating AI-like responses."}
SKIPPED_BUDGET = "The long-window review did not have enough analysis time."
SKIPPED_TIMEOUT = "The long-window review ran out of analysis time."
SKIPPED_UNAVAILABLE = "The optional long-window review could not complete."
SKIPPED_RECEIPT = "The optional long-window response could not be saved."
SKIP_REASONS = {SKIPPED_BUDGET, SKIPPED_TIMEOUT, SKIPPED_UNAVAILABLE, SKIPPED_RECEIPT}
SHA = re.compile(r"[0-9a-f]{64}")


def supported_source(source: dict) -> bool:
    samples = source.get("samples")
    return (type(samples) is int and 6 * RATE <= samples <= 14 * RATE
            and (source.get("source_codec"), source.get("source_container")) in (("flac", "flac"), ("pcm_s16le", "wav"))
            and source.get("source_bit_depth") == 16 and source.get("source_channels") == source.get("channels") == 1
            and source.get("source_sample_rate") == source.get("sample_rate") == RATE
            and len(source.get("audio_stream_indices", [0])) == 1 and source.get("timeline_time_base") == "1/16000"
            and source.get("timeline") == [{"start_sample": 0, "end_sample": samples, "source_pts": 0}])


def intervals(samples: int) -> list[tuple[int, int]]:
    if not 6 * RATE <= samples <= 14 * RATE:
        return []
    last = samples - SPAN
    starts = (0, last) if samples < 9 * RATE else (0, last // 2, last)
    return [(start, start + SPAN) for start in starts]


def status(windows: list[dict]) -> str:
    directions = [window["direction"] for window in windows]
    return "corroborated" if directions.count("ai") >= 2 and "original" not in directions else "no_corroboration"


def _require(valid, message):
    if not valid:
        raise ValueError(message)


def validate(review: dict, source: dict) -> None:
    try:
        _validate(review, source)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete long speech-window evidence") from error


def _validate(review, source):
    windows = review["windows"]
    _require(set(review) == {"schema", "title", "status", "reason", "limitation", "affects_overall_confidence", "windows"}
             and review["schema"] == SCHEMA and review["title"] == TITLE and review["limitation"] == LIMITATION
             and review["affects_overall_confidence"] is False and supported_source(source) and isinstance(windows, list),
             "Long speech-window evidence is outside its native source scope")
    if review["status"] == "skipped":
        _require(not windows and review["reason"] in SKIP_REASONS, "Skipped long speech review carries measurements")
        return
    expected = status(windows)
    _require(review["status"] == expected and review["reason"] == REASONS[expected]
             and [(window["start_sample"], window["end_sample"]) for window in windows] == intervals(source["samples"]),
             "Long speech-window status or sampled geometry differs from the fixed rule")
    for window in windows:
        a, b = window["start_sample"], window["end_sample"]
        ranges = [{"start_sample": a, "end_sample": b, "source_start_pts": a, "source_end_pts": b, "time_base": "1/16000"}]
        assets = window["artifacts"]
        media_expected = expected == "corroborated" and window["direction"] == "ai"
        _require(set(window) == {"start_sample", "end_sample", "pcm_sha256", "direction", "source_timestamp_ranges", "artifacts"}
                 and type(a) is type(b) is int and SHA.fullmatch(window["pcm_sha256"]) is not None
                 and window["direction"] in DIRECTIONS and window["source_timestamp_ranges"] == ranges
                 and set(assets) == ({"selected_clip", "waveform"} if media_expected else set()),
                 "Long speech-window media or clock differs from its measured range")
        if media_expected:
            clip, wave = assets["selected_clip"], assets["waveform"]
            _require(clip["media_type"] == "audio/wav" and clip["samples"] == SPAN and clip["sample_rate"] == RATE
                     and clip["channels"] == 1 and wave["media_type"] == "image/png" and (wave["width"], wave["height"]) == (960, 280)
                     and all(SHA.fullmatch(item["sha256"]) is not None for item in (clip, wave)),
                     "Invalid long speech-window assets")
