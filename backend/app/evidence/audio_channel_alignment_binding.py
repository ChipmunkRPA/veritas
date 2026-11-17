"""Validate stereo alignment evidence and reproduce its excerpts without DSP libraries."""

from __future__ import annotations

import sys
from array import array

from app.evidence.audio_repetition_binding import _sample_range, _timeline, _valid_hash
from app.evidence.audio_similarity_binding import (
    HOP,
    MAX_PAIRS,
    MAX_SAMPLES,
    MIN_REPEAT_SAMPLES,
    MIN_SAMPLES,
    SAMPLE_RATE,
)

SCHEMA = "chipmunk.aligned-stereo-repetition/1"
NATIVE_SCHEMA = "chipmunk.aligned-stereo-repetition/2"
AAC_SCHEMA = "chipmunk.aligned-stereo-repetition/3"
NATIVE_FACTOR = 3
NATIVE_RATE = SAMPLE_RATE * NATIVE_FACTOR
AAC_EDGE_MARGIN = 640
MAX_SHIFT = 12
GAIN_DENOMINATOR = 1024
ALIGNMENT_WINDOW = 1600
MIN_WINDOWS = 4
MIN_GAIN_CHANGE = 26
PROJECTION = "floor((gain_numerator * left[t] - 1024 * right[t + offset_samples]) / (gain_numerator + 1024))"
STATUSES = {
    None: "measured",
    "exact_channel_difference_found": "skipped",
    "channel_difference_sufficient": "skipped",
    "insufficient_alignment": "unavailable",
    "insufficient_active_bands": "unavailable",
    "span_budget": "ambiguous",
}


def native_scope(mapping):
    codec = mapping.get("source_codec")
    return (mapping.get("source_channels") == 2 and mapping.get("source_sample_rate") == NATIVE_RATE
            and MIN_SAMPLES <= mapping["samples"] <= MAX_SAMPLES and len(mapping["timeline"]) == 1
            and (codec in {"pcm_s16le", "pcm_s16be"} or (codec == "flac" and mapping.get("source_bit_depth") == 16)))



def aac_presented_samples(mapping):
    """Admit a single AAC-LC stream with a bounded, continuous 48 kHz clock."""
    from app.evidence.audio_short_review_binding import AAC_CONTAINER, AAC_FRAME

    if (mapping.get("source_codec"), mapping.get("source_profile"), mapping.get("source_container"),
            mapping.get("source_channels"), mapping.get("source_sample_rate"), mapping.get("source_time_base")) != (
            "aac", "LC", AAC_CONTAINER, 2, NATIVE_RATE, f"1/{NATIVE_RATE}") or "audio_stream_indices" in mapping:
        return None
    clock = mapping["timeline"]
    count, start, decoded = mapping.get("source_duration_pts"), mapping.get("source_start_pts"), mapping["samples"]
    if (type(count) is not int or not MIN_SAMPLES * NATIVE_FACTOR <= count <= MAX_SAMPLES * NATIVE_FACTOR
            or type(start) is not int or start < 0 or len(clock) != 1
            or start != NATIVE_FACTOR * clock[0]["source_pts"]
            or not -2 <= decoded * NATIVE_FACTOR - count <= AAC_FRAME + 2):
        return None
    return count


def overlap(count: int, shift: int) -> tuple[int, int]:
    return max(0, -shift), count - max(0, shift)


def excerpt_range(start: int, end: int, shift: int) -> tuple[int, int]:
    return start + min(0, shift), end + max(0, shift)


def project_excerpt(stereo: bytes, shift: int, gain: int) -> bytes:
    """Recompute the derived WAV payload from one original interleaved excerpt."""
    if len(stereo) % 4 or not 4 * abs(shift) < len(stereo):
        raise ValueError("Recorded stereo excerpt cannot cover its aligned pair of ranges")
    original = array("h")
    original.frombytes(stereo)
    if sys.byteorder == "big":
        original.byteswap()
    start, end = overlap(len(original) // 2, shift)
    left, right = original[2 * start:2 * end:2], original[2 * (start + shift) + 1:2 * (end + shift) + 1:2]
    values = array("h", ((gain * a - GAIN_DENOMINATOR * b) // (gain + GAIN_DENOMINATOR)
                         for a, b in zip(left, right, strict=True)))
    if sys.byteorder == "big":
        values.byteswap()
    return values.tobytes()


def _asset(asset, media_type, **metadata):
    if (asset["media_type"] != media_type or not _valid_hash(asset["sha256"])
            or any(asset[key] != value for key, value in metadata.items())):
        raise ValueError("Aligned stereo repetition has an invalid evidence asset")


def validate(record, source, *, exact_found):
    try:
        _validate(record, source, exact_found)
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError("Invalid aligned stereo repetition evidence") from error


def _validate(record, source, exact_found):
    schema = record["schema"]
    aac = schema == AAC_SCHEMA
    factor = NATIVE_FACTOR if schema in {NATIVE_SCHEMA, AAC_SCHEMA} else 1
    count = aac_presented_samples(source) if aac else source["samples"] * factor
    rate = SAMPLE_RATE * factor
    clock = _timeline(source, source["samples"])
    reason, pairs = record["reason"], record["pairs"]
    if (source["sample_rate"] != SAMPLE_RATE or count is None or not MIN_SAMPLES * factor <= count <= MAX_SAMPLES * factor
            or len(clock) != 1 or schema not in {SCHEMA, NATIVE_SCHEMA, AAC_SCHEMA} or record["source_sha256"] != source["sha256"]
            or record["sample_rate"] != rate or record["input_samples"] != count
            or record["projection"] != PROJECTION or reason not in STATUSES or record["status"] != STATUSES[reason]
            or not isinstance(pairs, list) or len(pairs) > MAX_PAIRS or (reason is not None and pairs)
            or exact_found != (reason == "exact_channel_difference_found")
            or (factor > 1 and record["status"] == "skipped") or (schema == NATIVE_SCHEMA and not native_scope(source))):
        raise ValueError("Aligned stereo repetition differs from its source or check status")
    alignment = record["alignment"]
    if reason in {"exact_channel_difference_found", "insufficient_alignment"}:
        if alignment is not None or record["searched_interval"] is not None or record["pcm_sha256"] is not None:
            raise ValueError("An unavailable stereo alignment claims measurements")
        return
    shift, gain = alignment["right_offset_samples"], alignment["right_gain_numerator"]
    support, usable = alignment["agreeing_windows"], alignment["usable_windows"]
    if (any(type(value) is not int for value in (shift, gain, support, usable)) or abs(shift) > MAX_SHIFT * factor
            or not GAIN_DENOMINATOR // 2 <= gain <= 2 * GAIN_DENOMINATOR or not MIN_WINDOWS <= support <= usable
            or usable > (count // factor - 1 - 2 * MAX_SHIFT) // ALIGNMENT_WINDOW or support * 3 < 2 * usable):
        raise ValueError("Aligned stereo repetition has inconsistent channel measurements")
    if factor == 1 and (shift == 0 and abs(gain - GAIN_DENOMINATOR) < MIN_GAIN_CHANGE) != (reason == "channel_difference_sufficient"):
        raise ValueError("An unnecessary stereo alignment could not have been searched")
    if reason == "channel_difference_sufficient":
        if record["searched_interval"] is not None or record["pcm_sha256"] is not None:
            raise ValueError("A skipped stereo alignment claims measurements")
        return
    low, high = overlap(count, shift)
    if record["searched_interval"] != {"start_sample": low, "end_sample": high} or not _valid_hash(record["pcm_sha256"]):
        raise ValueError("Aligned stereo search does not cover the real shared clock")
    extent = {"samples": count, "source_pts": clock[0]["source_pts"] * factor, "time_base": f"1/{rate}"}
    margin = AAC_EDGE_MARGIN if aac else 0
    seen = set()
    for index, pair in enumerate(pairs, 1):
        first, second = (_sample_range(pair[side], **extent) for side in ("first", "second"))
        length = first[1] - first[0]
        if (pair["id"] != f"AR{index:02d}" or (first, second) in seen or first[0] % (HOP * factor)
                or not low + margin <= first[0] < first[1] <= second[0] < second[1] <= high - margin
                or length != second[1] - second[0] or length < MIN_REPEAT_SAMPLES * factor
                or pair["sample_count"] != length or pair["lag_samples"] != second[0] - first[0]):
            raise ValueError("Aligned stereo repetition pair differs from the reported time range")
        seen.add((first, second))
        assets = pair["artifacts"]
        roles = {f"{side}_{suffix}" for side in ("first", "second") for suffix in ("clip", "recorded_clip", "waveform")}
        if set(assets) != roles:
            raise ValueError("Aligned stereo repetition is missing source excerpts or waveforms")
        for side, (start, end) in zip(("first", "second"), (first, second), strict=True):
            original = pair[side]["recorded_excerpt"]
            bounds = _sample_range(original, **extent)
            if (bounds != excerpt_range(start, end, shift) or not _valid_hash(original["pcm_s16le_sha256"])
                    or not _valid_hash(pair[side]["pcm_s16le_sha256"])):
                raise ValueError("Aligned stereo excerpt does not include both original channel ranges")
            _asset(assets[f"{side}_recorded_clip"], "audio/wav", samples=length + abs(shift), sample_rate=rate, channels=2)
            _asset(assets[f"{side}_clip"], "audio/wav", samples=length, sample_rate=rate, channels=1)
            _asset(assets[f"{side}_waveform"], "image/png", width=960, height=280)
