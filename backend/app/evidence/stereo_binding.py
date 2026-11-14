"""Standard-library validation of source-bound stereo channel evidence."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from app.evidence.audio_silence_binding import FLANK, MIN_FLANK_DBFS, MINIMUM, continuous_spans

SAMPLE_RATE = 16_000
SourceRanges = Callable[[int, int], list[dict[str, Any]]]

LEGACY_SCHEMA = "chipmunk.stereo-channel-context/1"
SCHEMA = "chipmunk.stereo-channel-context/2"
MAX_SECONDS = 30
QUIET_DBFS = -50
REVIEW_GAP_DB = 12
REVIEW_DOWNMIX_DB = 12
LEGACY_SILENCE_FLOOR_DBFS = -120
# A one-sample, one-LSB pulse in one second of signed 16-bit PCM is about -132 dBFS.
SILENCE_FLOOR_DBFS = -140
_HASH = re.compile(r"[a-f0-9]{64}")
CHANNEL_SILENCE = "channel_silence"
CHANNEL_SILENCE_SCHEMA = "chipmunk.stereo-channel-silence/1"
MAXIMUM_SILENCE = 2 * SAMPLE_RATE
MAXIMUM_SILENCE_EVENTS = 3
CHANNEL_SILENCE_METHOD = {
    "minimum_zero_samples": MINIMUM,
    "maximum_zero_samples": MAXIMUM_SILENCE,
    "flank_samples": FLANK,
    "minimum_rms_dbfs": MIN_FLANK_DBFS,
    "maximum_events": MAXIMUM_SILENCE_EVENTS,
}


def unsupported_status(channels: int, layout: str | None) -> str | None:
    if channels == 1:
        return "mono"
    if channels != 2 or layout not in (None, "unknown", "stereo"):
        return "unsupported"
    return None


def classify(left: float, right: float) -> tuple[str, float | None, str | None]:
    if max(left, right) < QUIET_DBFS:
        return "quiet", None, None
    gap = round(abs(left - right), 3)
    if gap >= REVIEW_GAP_DB:
        return "strong_imbalance", gap, "left" if left > right else "right"
    return "measured", gap, None


def downmix(left: float, right: float, mono: float) -> dict[str, Any]:
    louder = max(left, right)
    audible = louder >= QUIET_DBFS
    silent = mono == SILENCE_FLOOR_DBFS
    attenuation = round(louder - mono, 3) if audible and not silent else None
    strong = audible and (silent or (attenuation is not None and attenuation >= REVIEW_DOWNMIX_DB))
    return {
        "mono_rms_dbfs": mono,
        "attenuation_db": attenuation,
        "strong_attenuation": strong,
    }


def validate(context: dict[str, Any], source: dict[str, Any], *, ranges: SourceRanges) -> None:
    """Check the displayed measurements' binding, ranges, units and disposition."""
    try:
        _validate(context, source, ranges)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("Stereo evidence contains missing fields or invalid types") from error


def _validate(context: dict[str, Any], source: dict[str, Any], ranges: SourceRanges) -> None:
    schema = context.get("schema")
    if schema not in (LEGACY_SCHEMA, SCHEMA):
        raise ValueError("Stereo channel evidence uses an unsupported method")
    silence_floor = SILENCE_FLOOR_DBFS if schema == SCHEMA else LEGACY_SILENCE_FLOOR_DBFS
    expected = {
        "evidence_type": "context",
        "affects_overall_confidence": False,
        "source_sha256": source.get("source_sha256", source.get("sha256")),
        "source_stream_index": source["source_stream_index"],
        "source_channel_count": source["source_channels"],
        "sample_rate": SAMPLE_RATE,
        "window_seconds": 1,
        "max_seconds": MAX_SECONDS,
        "quiet_floor_dbfs": QUIET_DBFS,
        "strong_level_gap_db": REVIEW_GAP_DB,
        "silence_floor_dbfs": silence_floor,
    }
    if schema == SCHEMA:
        expected.update(mono_pcm_sha256=source["pcm_sha256"], strong_downmix_attenuation_db=REVIEW_DOWNMIX_DB)
    elif "mono_pcm_sha256" in context or "strong_downmix_attenuation_db" in context:
        raise ValueError("Legacy stereo context cannot contain downmix measurements")
    if (context.get("affects_overall_confidence") is not False
            or any(context.get(key) != value for key, value in expected.items())):
        raise ValueError("Stereo channel evidence differs from the recorded source or method")
    status = context.get("status")
    layout = context["source_channel_layout"]
    if layout is not None and not isinstance(layout, str):
        raise ValueError("Stereo channel evidence has an invalid channel layout")
    reason = unsupported_status(source["source_channels"], layout)
    count = min(source["samples"] // SAMPLE_RATE, MAX_SECONDS)
    windows = context["windows"]
    pcm_sha = context["stereo_pcm_sha256"]
    if reason or count == 0 or status == "unavailable":
        if (status != (reason or ("too_short" if count == 0 else "unavailable"))
                or windows != [] or pcm_sha is not None or CHANNEL_SILENCE in context):
            raise ValueError("Unassessed stereo evidence contains inconsistent measurements")
        return
    if not _HASH.fullmatch(str(pcm_sha)) or not isinstance(windows, list) or len(windows) != count:
        raise ValueError("Stereo evidence does not cover its declared PCM prefix")
    audible = False
    for index, window in enumerate(windows):
        start, stop = index * SAMPLE_RATE, (index + 1) * SAMPLE_RATE
        if (window["start_sample"] != start or window["end_sample"] != stop
                or window["source_timestamp_ranges"] != ranges(start, stop)):
            raise ValueError("Stereo evidence differs from its recorded source timestamps")
        left, right = window["left_rms_dbfs"], window["right_rms_dbfs"]
        if not all(_valid_level(level, silence_floor) for level in (left, right)):
            raise ValueError("Stereo evidence has invalid channel levels")
        window_status, gap, louder = classify(left, right)
        correlation = window["correlation"]
        if ((window["status"], window["level_gap_db"], window["louder_channel"]) != (window_status, gap, louder)
                or correlation is not None and (
                    window_status == "quiet" or not isinstance(correlation, (int, float))
                    or isinstance(correlation, bool) or not -1 <= correlation <= 1)):
            raise ValueError("Stereo evidence has inconsistent channel measurements")
        if schema == SCHEMA:
            measured = window["downmix"]
            mono = measured["mono_rms_dbfs"]
            if (not _valid_level(mono, silence_floor) or type(measured["strong_attenuation"]) is not bool
                    or measured != downmix(left, right, mono)):
                raise ValueError("Stereo evidence has inconsistent downmix measurements")
        elif "downmix" in window:
            raise ValueError("Legacy stereo context cannot contain downmix measurements")
        audible |= window_status != "quiet"
    if status != ("measured" if audible else "quiet"):
        raise ValueError("Stereo evidence summary differs from its measured windows")
    if CHANNEL_SILENCE in context:
        if schema != SCHEMA:
            raise ValueError("Legacy stereo context cannot contain exact channel silence")
        _validate_channel_silence(context[CHANNEL_SILENCE], source["timeline"], count * SAMPLE_RATE, ranges)


def _valid_level(value: Any, silence_floor: int) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and silence_floor <= value <= 0)


def channel_silence_spans(timeline, limit):
    return [(start, min(end, limit)) for start, end in continuous_spans(timeline)
            if min(end, limit) - start >= MINIMUM + 2 * FLANK]


def _validate_channel_silence(record, timeline, limit, ranges):
    spans = channel_silence_spans(timeline, limit)
    coverage = sum(end - start for start, end in spans)
    count = record["qualifying_intervals"]
    selected = record["selected"]
    if (record["schema"] != CHANNEL_SILENCE_SCHEMA or record["affects_overall_confidence"] is not False
            or record["method"] != CHANNEL_SILENCE_METHOD
            or type(record["checked_samples"]) is not int or record["checked_samples"] != coverage
            or type(count) is not int or not 0 <= count <= 2 * coverage // MINIMUM
            or not isinstance(selected, list) or len(selected) != min(count, MAXIMUM_SILENCE_EVENTS)):
        raise ValueError("Exact channel silence has invalid coverage")
    previous = None
    channel_ends = {}
    for row in selected:
        start, end, channel = row["start_sample"], row["end_sample"], row["silent_channel"]
        levels = row["flank_rms_dbfs"]
        if (type(start) is not int or type(end) is not int or channel not in ("left", "right")
                or not MINIMUM <= end - start <= MAXIMUM_SILENCE
                or not any(a <= start - FLANK and end + FLANK <= b for a, b in spans)
                or row["source_timestamp_ranges"] != ranges(start, end)
                or not isinstance(levels, list) or len(levels) != 2
                or not all(_valid_level(level, MIN_FLANK_DBFS) for level in [*levels, row["minimum_other_rms_dbfs"]])
                or previous is not None and previous >= (start, end, channel)
                or channel_ends.get(channel, 0) > start):
            raise ValueError("Exact channel silence differs from its measured scope or source timestamps")
        previous = (start, end, channel)
        channel_ends[channel] = end
