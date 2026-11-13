"""Portable source and decision rules for the bounded long speech review."""

from __future__ import annotations

import re

RATE = 16_000
SPAN = 3 * RATE
SCHEMA = "chipmunk.long-voice-review/1"
FOCUSED_SCHEMA = "chipmunk.long-voice-review/2"
FOCUSED_HOP = SPAN // 2
DETECTOR_ID = "aud.long_voice_review"
TITLE = "Long speech-window review"
LIMITATION = ("The responses apply to the three-second windows shown. Gaps between windows are outside this check. "
              "The review cannot establish how the speech was made or identify exact edit boundaries. "
              "It carries no weight in the overall assessment.")
FOCUSED_LIMITATION = ("A shorter range marks the overlap of two AI-leaning three-second responses. A full window can remain "
                      "when there is no shorter range wholly inside it. These ranges cannot establish how speech was made or "
                      "where an edit begins or ends. This review carries no weight in the overall assessment.")
DIRECTIONS = {"ai", "original", "inconclusive", "unassessed"}
LABELS = {"ai": "Leans AI-generated in this window", "original": "Leans original in this window",
          "inconclusive": "Inconclusive window", "unassessed": "No usable speech response"}
REASONS = {"corroborated": "At least two separate windows leaned AI-generated, and none leaned original.",
           "no_corroboration": "The sampled windows did not produce corroborating AI-like responses."}
FOCUSED_REASONS = {"corroborated": "The sampled windows support the speech ranges shown below.",
                   "no_corroboration": REASONS["no_corroboration"]}
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


def focused_scope(samples: int) -> bool:
    return 6 * RATE < samples <= 11 * RATE


def dense_intervals(samples: int) -> list[tuple[int, int]]:
    return [(start, start + SPAN) for start in range(0, samples - SPAN + 1, FOCUSED_HOP)] if focused_scope(samples) else []


def focused_intervals(samples: int) -> list[tuple[int, int]]:
    return sorted(set(intervals(samples)) | set(dense_intervals(samples))) if focused_scope(samples) else []


def focused_regions(samples: int, windows: list[dict]) -> list[dict]:
    index = {(window["start_sample"], window["end_sample"]): i for i, window in enumerate(windows)}
    dense = dense_intervals(samples)
    sparse = [index[span] for span in intervals(samples)]

    def region(start, end, basis, support):
        return {"start_sample": start, "end_sample": end, "basis": basis, "supporting_windows": support}

    overlap = [region(right[0], left[1], "overlap", [index[left], index[right]])
               for left, right in zip(dense, dense[1:])
               if windows[index[left]]["direction"] == windows[index[right]]["direction"] == "ai"]
    previous = [i for i in sparse if windows[i]["direction"] == "ai"]
    full = []
    if status([windows[i] for i in sparse]) == "corroborated":
        for i in previous:
            start, end = windows[i]["start_sample"], windows[i]["end_sample"]
            if not any(start <= item["start_sample"] < item["end_sample"] <= end for item in overlap):
                full.append(region(start, end, "sparse", previous))
    return sorted([*overlap, *full], key=lambda item: (item["start_sample"], item["end_sample"]))


def _require(valid, message):
    if not valid:
        raise ValueError(message)


def validate(review: dict, source: dict) -> None:
    try:
        _validate(review, source)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete long speech-window evidence") from error


def _validate(review, source):
    if review["schema"] == FOCUSED_SCHEMA:
        _validate_focused(review, source)
        return
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
            _validate_assets(assets, SPAN)


def _validate_assets(assets, samples):
    clip, wave = assets["selected_clip"], assets["waveform"]
    _require(clip["media_type"] == "audio/wav" and clip["samples"] == samples and clip["sample_rate"] == RATE
             and clip["channels"] == 1 and wave["media_type"] == "image/png" and (wave["width"], wave["height"]) == (960, 280)
             and all(SHA.fullmatch(item["sha256"]) is not None for item in (clip, wave)), "Invalid long speech-window assets")


def _source_ranges(start, end):
    return [{"start_sample": start, "end_sample": end, "source_start_pts": start, "source_end_pts": end, "time_base": "1/16000"}]


def _validate_focused(review, source):
    windows, regions = review["windows"], review["regions"]
    _require(set(review) == {"schema", "title", "status", "reason", "limitation", "affects_overall_confidence", "windows", "regions"}
             and review["title"] == TITLE and review["limitation"] == FOCUSED_LIMITATION and review["affects_overall_confidence"] is False
             and supported_source(source) and focused_scope(source["samples"]) and isinstance(windows, list) and isinstance(regions, list),
             "Focused speech review is outside its native source scope")
    if review["status"] == "skipped":
        _require(not windows and not regions and review["reason"] in SKIP_REASONS, "Skipped focused review carries measurements")
        return
    samples = source["samples"]
    _require([(window["start_sample"], window["end_sample"]) for window in windows] == focused_intervals(samples),
             "Focused speech-window schedule differs")
    for window in windows:
        start, end = window["start_sample"], window["end_sample"]
        _require(set(window) == {"start_sample", "end_sample", "direction", "pcm_sha256", "source_timestamp_ranges", "artifacts"}
                 and type(start) is type(end) is int and window["direction"] in DIRECTIONS
                 and SHA.fullmatch(window["pcm_sha256"]) and window["source_timestamp_ranges"] == _source_ranges(start, end)
                 and window["artifacts"] == {}, "Invalid focused speech-window reading")
    expected = focused_regions(samples, windows)
    state = "corroborated" if expected else "no_corroboration"
    _require(len(regions) == len(expected) and review["status"] == state and review["reason"] == FOCUSED_REASONS[state],
             "Focused speech regions differ from the supporting windows")
    for region, planned in zip(regions, expected, strict=True):
        start, end = region["start_sample"], region["end_sample"]
        _require(set(region) == {*planned, "pcm_sha256", "source_timestamp_ranges", "artifacts"}
                 and type(start) is type(end) is int and {key: region[key] for key in planned} == planned
                 and all(type(i) is int for i in region["supporting_windows"]) and SHA.fullmatch(region["pcm_sha256"])
                 and region["source_timestamp_ranges"] == _source_ranges(start, end)
                 and set(region["artifacts"]) == {"selected_clip", "waveform"}, "Focused region or source clock differs")
        _validate_assets(region["artifacts"], end - start)


def media_windows(review):
    return review.get("regions") if review.get("schema") == FOCUSED_SCHEMA else review.get("windows")
