"""Validate exact repetition evidence for two separately recorded channels."""

from __future__ import annotations

from app.evidence import audio_channel_alignment_binding as alignment
from app.evidence import audio_gain_repetition_binding as gain
from app.evidence.audio_channel_difference import PROJECTION
from app.evidence.audio_exact_repetition import SAMPLE_RATE
from app.evidence.audio_repetition_binding import METHOD, _valid_hash
from app.evidence.audio_repetition_binding import validate as validate_mono

KEY = "stereo_repetition"
INPUT = "stereo_repetition_pcm_sha256"
SCHEMA = "chipmunk.stereo-exact-repetition/1"
DIFFERENCE_SCHEMA = "chipmunk.stereo-exact-repetition/2"
ALIGNED_SCHEMA = "chipmunk.stereo-exact-repetition/3"
NATIVE_ALIGNED_SCHEMA = "chipmunk.stereo-exact-repetition/4"
AAC_ALIGNED_SCHEMA = "chipmunk.stereo-exact-repetition/5"
DIFFERENCE_INPUT = "stereo_repetition_difference_version"
DIFFERENCE_VERSION = 1
ALIGNED_VERSION = 2
NATIVE_ALIGNED_VERSION = 3
AAC_ALIGNED_VERSION = 4
DIFFERENCE_KEY = "difference_measurement"
ALIGNED_KEY = "aligned_measurement"
DIFFERENCE_VERSIONS = {
    DIFFERENCE_SCHEMA: DIFFERENCE_VERSION, ALIGNED_SCHEMA: ALIGNED_VERSION, NATIVE_ALIGNED_SCHEMA: NATIVE_ALIGNED_VERSION,
    AAC_ALIGNED_SCHEMA: AAC_ALIGNED_VERSION,
}
NATIVE_VERSIONS = {NATIVE_ALIGNED_VERSION, AAC_ALIGNED_VERSION}
NATIVE_SCHEMAS = {NATIVE_ALIGNED_SCHEMA: alignment.NATIVE_SCHEMA, AAC_ALIGNED_SCHEMA: alignment.AAC_SCHEMA}
CHANNELS = ("left", "right")


def pairs(record):
    for row in record["channels"]:
        yield from row["measurement"]["pairs"]
        if context := row.get("gain_measurement"):
            yield from context["pairs"]
    if difference := record.get(DIFFERENCE_KEY):
        yield from difference["pairs"]
    if aligned := record.get(ALIGNED_KEY):
        yield from aligned["pairs"]


def _validate_measurement(measurement, source, prefix, *, difference=False):
    if not _valid_hash(measurement["pcm_sha256"]):
        raise ValueError("Stereo repetition lacks a sample commitment")
    normalized = []
    for index, pair in enumerate(measurement["pairs"], 1):
        if pair["id"] != f"{prefix}R{index:02d}":
            raise ValueError("Stereo repetition pair identifiers differ")
        assets = dict(pair["artifacts"])
        waveform = assets.pop("waveform")
        if (waveform["media_type"], waveform["width"], waveform["height"]) != ("image/png", 960, 280) or not _valid_hash(waveform["sha256"]):
            raise ValueError("Stereo repetition lacks its waveform")
        if difference:
            for side in ("first", "second"):
                original = assets.pop(f"{side}_recorded_clip")
                if (original["media_type"], original["samples"], original["sample_rate"], original["channels"]) != (
                    "audio/wav", pair["sample_count"], SAMPLE_RATE, 2
                ) or not _valid_hash(original["sha256"]):
                    raise ValueError("Channel difference lacks its recorded stereo excerpt")
        normalized.append({**pair, "id": f"R{index:02d}", "artifacts": assets})
    validate_mono({**measurement, "pairs": normalized}, {**source, "pcm_sha256": measurement["pcm_sha256"]})


def validate(record, source):
    try:
        schema = record["schema"]
        if schema in NATIVE_SCHEMAS:
            if (record["affects_overall_confidence"] is not False or record["source_sha256"] != source["sha256"]
                    or not _valid_hash(record["stereo_pcm_sha256"]) or record["channels"] != []
                    or DIFFERENCE_KEY in record or "difference_projection" in record
                    or record[ALIGNED_KEY]["schema"] != NATIVE_SCHEMAS[schema]):
                raise ValueError("Native stereo repetition does not match the source or its check")
            alignment.validate(record[ALIGNED_KEY], source, exact_found=False)
            return
        if (schema not in {SCHEMA, *DIFFERENCE_VERSIONS} or record["affects_overall_confidence"] is not False
                or record["source_sha256"] != source["sha256"] or source["source_channels"] != 2
                or source["source_sample_rate"] != SAMPLE_RATE or source["sample_rate"] != SAMPLE_RATE
                or not _valid_hash(record["stereo_pcm_sha256"]) or len(source["timeline"]) != 1
                or not METHOD["min_input_samples"] <= source["samples"] <= METHOD["max_input_samples"]
                or [row["channel"] for row in record["channels"]] != list(CHANNELS)):
            raise ValueError("Stereo repetition does not match the source or the supported input")
        for row in record["channels"]:
            channel = row["channel"]
            measurement = row["measurement"]
            _validate_measurement(measurement, source, channel[0].upper())
            if "gain_measurement" in row:
                gain.validate(row["gain_measurement"], source, channel, measurement["pcm_sha256"])
        if len({"gain_measurement" in row for row in record["channels"]}) != 1:
            raise ValueError("Gain repetition coverage differs between recorded channels")
        if schema in DIFFERENCE_VERSIONS:
            if record["difference_projection"] != PROJECTION:
                raise ValueError("Unsupported audio channel difference calculation")
            _validate_measurement(record[DIFFERENCE_KEY], source, "D", difference=True)
        elif DIFFERENCE_KEY in record or "difference_projection" in record:
            raise ValueError("Unexpected audio channel difference evidence")
        if schema == ALIGNED_SCHEMA:
            if record[ALIGNED_KEY]["schema"] != alignment.SCHEMA:
                raise ValueError("Aligned stereo evidence requires its historical sample clock")
            alignment.validate(record[ALIGNED_KEY], source, exact_found=bool(record[DIFFERENCE_KEY]["pairs"]))
        elif ALIGNED_KEY in record:
            raise ValueError("Unexpected aligned stereo repetition evidence")
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError("Invalid stereo repetition evidence") from error
