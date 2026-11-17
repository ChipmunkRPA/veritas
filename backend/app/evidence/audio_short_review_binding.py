"""Shared source and public-packet checks for the optional short speech review."""

from __future__ import annotations

import re

RATE = 16000
SAMPLES = 6 * RATE
GRID = tuple((start, start + 2 * RATE) for start in range(0, 5 * RATE, RATE))
SCHEMA = "chipmunk.short-voice-review/1"
COVERAGE_SCHEMA = "chipmunk.short-voice-review/2"
NEARBY_SCHEMA = "chipmunk.short-voice-review/3"
DETECTOR_ID = "aud.short_voice_review"
NEARBY_ID = "aud.short_voice_nearby"
NEARBY_SHIFT = 1280
NEARBY_CODECS = {"pcm_s16le", "flac"}
IDENTICAL_CHANNEL_PCM = "identical_stereo_channel_pcm_sha256"
IDENTICAL_CHANNEL_DESCRIPTION = "Both recorded channels contain the same samples."
NEARBY_LABELS = {
    "elevated": "Elevated response",
    "below_review_point": "Below the review point",
    "below_speech_minimum": "Below the speech minimum",
}
NEARBY_SUMMARIES = {
    "repeated": "The elevated response repeated in every checked nearby window.",
    "varied": "At least one nearby window fell below the review point. Small timing changes affected this finding.",
    "incomplete": "Some nearby windows had too little detected speech to compare.",
}
REVIEW_POINT = 6.822523597593649
MP3_REVIEW_POINT = 7.085599896496718
MP3_ORIGIN = 1105
AAC_RATE, AAC_FRAME, AAC_REVIEW_POINT = 48_000, 1024, 7.049249651163794
LOSSLESS_RATE = 48_000
LOSSLESS_FORMATS = {("wav", "pcm_s16le"), ("flac", "flac")}
LOSSLESS_NATIVE_CLOCK = "source_native_clock_verified"
RESAMPLED_DESCRIPTION = "This check uses 16 kHz mono audio decoded from the 48 kHz source."
AAC_CONTAINER = "mov,mp4,m4a,3gp,3g2,mj2"
AAC_FIELDS = ("source_container", "source_profile", "source_bit_rate", "source_time_base", "source_start_pts",
              "source_duration_pts")
TITLE = "Short speech-window review"
LIMITATION = ("This is a clue for the complete two-second window. It cannot establish AI generation or the boundaries "
              "of an edit. It carries no weight in the overall assessment.")
SCOPE_REASON = ("This check requires Full analysis of six continuous decoded seconds at 16 kHz. "
                "It supports 16-bit WAV and FLAC with a single channel or two identical channels, "
                "and a verified 48 kb/s mono MP3 profile.")
AAC_SCOPE_REASON = ("This check requires a continuous six-second source in Full analysis. It supports 16-bit mono WAV and "
                    "FLAC or those lossless formats with two identical channels, the verified 48 kb/s mono MP3 profile, "
                    "and 48 kHz mono AAC-LC in M4A at the tested bit rate.")
NO_SPEECH_REASON = "No two-second interval contained enough detected speech."
AFTER_COVERAGE_SKIP = {
    "flagged": "The main voice check skipped its longer windows. A shorter window returned an elevated response. "
               "Brief ordinary speech and edits can also trigger this response.",
    "no_flag": "The main voice check skipped its longer windows. No checked two-second window was flagged. "
               "The recording could still contain an edit.",
    "skipped": "The main voice check skipped its longer windows. No two-second interval contained enough detected speech.",
}


def source_metadata(mapping: dict) -> dict:
    metadata = {"source_codec": mapping.get("source_codec")}
    keys = {"flac": ("source_container", "source_bit_depth"), "pcm_s16le": ("source_container", "source_bit_depth"),
            "mp3": ("source_container", "source_bit_rate"), "aac": AAC_FIELDS}
    metadata.update({key: mapping[key] for key in keys.get(metadata["source_codec"], ()) if key in mapping})
    if mapping.get("source_sample_rate") == LOSSLESS_RATE and metadata["source_codec"] in NEARBY_CODECS:
        metadata.update({key: mapping[key] for key in ("source_time_base", "source_start_pts", "source_duration_pts")
                         if key in mapping})
    if LOSSLESS_NATIVE_CLOCK in mapping:
        metadata[LOSSLESS_NATIVE_CLOCK] = mapping[LOSSLESS_NATIVE_CLOCK]
    if IDENTICAL_CHANNEL_PCM in mapping:
        metadata[IDENTICAL_CHANNEL_PCM] = mapping[IDENTICAL_CHANNEL_PCM]
    return metadata


def source_metadata_matches(source: dict, mapping: dict) -> bool:
    return all(mapping.get(key) == value for key, value in source_metadata(source).items())


def supported_aac_length(samples: int) -> bool:
    return SAMPLES <= samples <= SAMPLES + (AAC_FRAME + 2) // 3


def supported_aac(source: dict) -> bool:
    bit_rate, samples = source.get("source_bit_rate"), source.get("samples")
    return (
        source.get("source_codec") == "aac" and source.get("source_container") == AAC_CONTAINER
        and source.get("source_profile") == "LC" and type(bit_rate) is int and 64_000 <= bit_rate <= 67_000
        and source.get("source_time_base") == "1/48000" and source.get("source_start_pts") == 0
        and source.get("source_duration_pts") == 6 * AAC_RATE and type(samples) is int and supported_aac_length(samples)
        and source.get("sample_rate") == RATE and source.get("source_sample_rate") == AAC_RATE
        and source.get("channels") == source.get("source_channels") == 1
        and len(source.get("audio_stream_indices", [0])) == 1 and source.get("timeline_time_base") == "1/16000"
        and source.get("timeline") == [{"start_sample": 0, "end_sample": samples, "source_pts": 0}]
    )


def _continuous_clock(mapping: dict, origin: int = 0, *, source_rate: int = RATE) -> bool:
    return (
        mapping.get("samples") == SAMPLES and mapping.get("sample_rate") == RATE
        and mapping.get("source_sample_rate") == source_rate
        and mapping.get("channels") == 1 and len(mapping.get("audio_stream_indices", [0])) == 1
        and mapping.get("timeline_time_base") == "1/16000"
        and mapping.get("timeline") == [{"start_sample": 0, "end_sample": SAMPLES, "source_pts": origin}]
    )


def eligible_stereo_source(mapping: dict) -> bool:
    return (
        mapping.get("source_channels") == 2 and mapping.get("source_bit_depth") == 16
        and (mapping.get("source_container"), mapping.get("source_codec")) in LOSSLESS_FORMATS
        and _continuous_clock(mapping)
    )


def supported_resampled_lossless(mapping: dict) -> bool:
    container = mapping.get("source_container")
    origin = mapping.get("source_start_pts")
    duration = mapping.get("source_duration_pts")
    return (
        (container, mapping.get("source_codec")) in LOSSLESS_FORMATS
        and mapping.get("source_channels") == 1 and mapping.get("source_bit_depth") == 16
        and mapping.get("source_time_base") == "1/48000" and type(duration) is int and duration == 6 * LOSSLESS_RATE
        and mapping.get(LOSSLESS_NATIVE_CLOCK) is True
        and ((type(origin) is int and origin == 0) or (container == "wav" and origin is None))
        and _continuous_clock(mapping, source_rate=LOSSLESS_RATE)
    )


def scope_reason(*, aac: bool, resampled: bool) -> str:
    if not resampled:
        return AAC_SCOPE_REASON if aac else SCOPE_REASON
    carriers = " and a verified 48 kHz mono AAC-LC profile" if aac else ""
    return ("This check requires six continuous seconds in Full analysis. It supports 16-bit mono WAV and FLAC at "
            "16 or 48 kHz, lossless 16 kHz audio with two identical channels, a verified 48 kb/s mono MP3 profile"
            f"{carriers}.")


def supported_source(mapping: dict) -> bool:
    codec = mapping.get("source_codec")
    if codec == "aac":
        return supported_aac(mapping)
    if mapping.get("source_sample_rate") == LOSSLESS_RATE:
        return supported_resampled_lossless(mapping)
    if mapping.get("source_channels") == 2:
        pin = mapping.get(IDENTICAL_CHANNEL_PCM)
        return eligible_stereo_source(mapping) and isinstance(pin, str) and re.fullmatch("[a-f0-9]{64}", pin) is not None
    lossless = codec == "pcm_s16le" or (
        codec == "flac" and mapping.get("source_container") == "flac" and mapping.get("source_bit_depth") == 16
    )
    mp3 = codec == "mp3" and mapping.get("source_container") == "mp3" and mapping.get("source_bit_rate") == 48000
    return mapping.get("source_channels") == 1 and (lossless or mp3) and _continuous_clock(mapping, MP3_ORIGIN if mp3 else 0)


def selected_range(responses: list[dict], codec: str | None = None) -> dict | None:
    best = max(responses, key=lambda item: (item["logit"], -item["start_sample"]), default=None)
    point = {"mp3": MP3_REVIEW_POINT, "aac": AAC_REVIEW_POINT}.get(codec, REVIEW_POINT)
    if best is None or best["logit"] <= point:
        return None
    return {key: best[key] for key in ("start_sample", "end_sample")}


def source_range(source: dict, start: int, end: int) -> dict:
    origin = source["timeline"][0]["source_pts"]
    return {"start_sample": start, "end_sample": end, "source_start_pts": origin + start,
            "source_end_pts": origin + end, "time_base": "1/16000"}


def nearby_windows(start: int) -> list[tuple[int, int]]:
    starts = sorted({max(0, start - NEARBY_SHIFT), min(GRID[-1][0], start + NEARBY_SHIFT)} - {start})
    return [(point, point + 2 * RATE) for point in starts]


def speech_counts(spans, windows=GRID) -> list[int]:
    """Count the activity pass's disjoint spans on the source sample clock."""
    sample_spans = [(round(a * RATE), round(b * RATE)) for a, b in spans]
    return [sum(max(0, min(end, b) - max(start, a)) for a, b in sample_spans) for start, end in windows]


def nearby_summary(windows: list[dict]) -> str:
    outcomes = {row["status"] for row in windows}
    if "below_review_point" in outcomes:
        return "varied"
    return "incomplete" if "below_speech_minimum" in outcomes else "repeated"


def _validate_nearby(review: dict, source: dict, start: int) -> None:
    nearby = review["nearby_window_review"]
    windows = nearby["windows"]
    if (source["source_codec"] not in NEARBY_CODECS or set(nearby) != {"status", "windows"}
            or not isinstance(windows, list) or len(windows) != len(nearby_windows(start))):
        raise ValueError("Unexpected nearby speech-window review")
    for row, (a, b) in zip(windows, nearby_windows(start), strict=True):
        if (set(row) != {"start_sample", "end_sample", "speech_samples", "status", "source_timestamp_ranges", "pcm_sha256"}
                or (row["start_sample"], row["end_sample"]) != (a, b)
                or any(type(row[key]) is not int for key in ("start_sample", "end_sample", "speech_samples"))
                or not 0 <= row["speech_samples"] <= b - a or row["status"] not in NEARBY_LABELS
                or (row["speech_samples"] < RATE) != (row["status"] == "below_speech_minimum")
                or row["source_timestamp_ranges"] != [source_range(source, a, b)]
                or not re.fullmatch("[a-f0-9]{64}", row["pcm_sha256"])):
            raise ValueError("Nearby speech input differs from its source geometry")
    if nearby["status"] != nearby_summary(windows):
        raise ValueError("Nearby speech summary differs from its readings")


def window_coverage(speech_counts: list[int], source: dict) -> list[dict]:
    if not supported_source(source) or not isinstance(speech_counts, list) or len(speech_counts) != len(GRID):
        raise ValueError("Short speech-window coverage does not match a supported source")
    coverage = []
    for (start, end), speech in zip(GRID, speech_counts, strict=True):
        if type(speech) is not int or not 0 <= speech <= end - start:
            raise ValueError("Invalid short speech-window activity count")
        coverage.append({"start_sample": start, "end_sample": end, "speech_samples": speech,
                         "status": "checked" if 2 * speech >= end - start else "below_speech_minimum",
                         "source_timestamp_ranges": [source_range(source, start, end)]})
    return coverage


def validate(review: dict, source: dict) -> None:
    try:
        _validate(review, source)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete short speech-window evidence") from error


def _validate(review: dict, source: dict) -> None:
    fields = {"schema", "title", "status", "reason", "limitation", "affects_overall_confidence", "checked_windows",
              "interval", "start_s", "end_s", "source_timestamp_ranges", "artifacts"}
    nearby = review["schema"] == NEARBY_SCHEMA
    measured_coverage = review["schema"] in (COVERAGE_SCHEMA, NEARBY_SCHEMA)
    if measured_coverage:
        fields.add("window_coverage")
    if nearby:
        fields.add("nearby_window_review")
    if (set(review) != fields or review["schema"] not in {SCHEMA, COVERAGE_SCHEMA, NEARBY_SCHEMA}
            or review["title"] != TITLE or review["limitation"] != LIMITATION
            or review["affects_overall_confidence"] is not False or review["status"] not in {"flagged", "no_flag", "skipped"}
            or not isinstance(review["reason"], str) or not review["reason"]):
        raise ValueError("Invalid short speech-window evidence status")
    windows = [(w["start_sample"], w["end_sample"]) for w in review["checked_windows"]]
    if (any(set(w) != {"start_sample", "end_sample"} for w in review["checked_windows"])
            or windows != sorted(set(windows)) or not set(windows) <= set(GRID)
            or any(type(a) is not int or type(b) is not int for a, b in windows)
            or (windows and not supported_source(source)) or (review["status"] != "skipped" and not windows)):
        raise ValueError("Invalid short speech-window sample coordinates")
    if measured_coverage:
        coverage = review["window_coverage"]
        expected = window_coverage([row["speech_samples"] for row in coverage], source)
        admitted = [(row["start_sample"], row["end_sample"]) for row in expected if row["status"] == "checked"]
        if (coverage != expected or windows != admitted or (review["status"] == "skipped") != (not admitted)
                or (not admitted and review["reason"] not in {NO_SPEECH_REASON, AFTER_COVERAGE_SKIP["skipped"]})):
            raise ValueError("Short speech-window coverage disagrees with the analysis")
    interval = review["interval"]
    assets = review["artifacts"]
    if review["status"] != "flagged":
        if (nearby or interval is not None or assets or review["source_timestamp_ranges"]
                or review["start_s"] is not None or review["end_s"] is not None):
            raise ValueError("An unflagged short-window result includes a selected interval")
        return
    a, b = interval["start_sample"], interval["end_sample"]
    ranges = [source_range(source, a, b)]
    if (type(a) is not int or type(b) is not int or (a, b) not in windows
            or review["start_s"] != a / RATE or review["end_s"] != b / RATE or review["source_timestamp_ranges"] != ranges
            or set(assets) != {"selected_clip", "waveform"}):
        raise ValueError("Flagged short-window media is outside its checked source range")
    clip, waveform = assets["selected_clip"], assets["waveform"]
    if (clip["media_type"] != "audio/wav" or clip["samples"] != b - a or clip["sample_rate"] != RATE or clip["channels"] != 1
            or waveform["media_type"] != "image/png" or waveform["width"] != 960 or waveform["height"] != 280
            or any(not re.fullmatch("[a-f0-9]{64}", asset["sha256"]) for asset in assets.values())):
        raise ValueError("Invalid short speech-window media")
    if nearby:
        _validate_nearby(review, source, a)
