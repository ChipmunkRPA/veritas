"""Validate exact repetition evidence for two separately recorded channels."""

from __future__ import annotations

from app.evidence import audio_gain_repetition_binding as gain
from app.evidence.audio_exact_repetition import SAMPLE_RATE
from app.evidence.audio_repetition_binding import METHOD, _valid_hash
from app.evidence.audio_repetition_binding import validate as validate_mono

KEY = "stereo_repetition"
INPUT = "stereo_repetition_pcm_sha256"
SCHEMA = "chipmunk.stereo-exact-repetition/1"
CHANNELS = ("left", "right")


def pairs(record):
    for row in record["channels"]:
        yield from row["measurement"]["pairs"]
        if context := row.get("gain_measurement"):
            yield from context["pairs"]


def validate(record, source):
    try:
        if (record["schema"] != SCHEMA or record["affects_overall_confidence"] is not False
                or record["source_sha256"] != source["sha256"] or source["source_channels"] != 2
                or source["source_sample_rate"] != SAMPLE_RATE or source["sample_rate"] != SAMPLE_RATE
                or not _valid_hash(record["stereo_pcm_sha256"]) or len(source["timeline"]) != 1
                or not METHOD["min_input_samples"] <= source["samples"] <= METHOD["max_input_samples"]
                or [row["channel"] for row in record["channels"]] != list(CHANNELS)):
            raise ValueError("Stereo repetition does not match the source or the supported input")
        for row in record["channels"]:
            channel = row["channel"]
            measurement = row["measurement"]
            if not _valid_hash(measurement["pcm_sha256"]):
                raise ValueError("Stereo repetition lacks a channel sample commitment")
            normalized = []
            for index, pair in enumerate(measurement["pairs"], 1):
                artifacts = pair["artifacts"]
                waveform = artifacts["waveform"]
                if (waveform["media_type"], waveform["width"], waveform["height"]) != ("image/png", 960, 280) or not _valid_hash(waveform["sha256"]):
                    raise ValueError("Stereo repetition lacks its waveform")
                normalized.append({**pair, "id": f"R{index:02d}",
                                   "artifacts": {key: asset for key, asset in artifacts.items() if key != "waveform"}})
            if [pair["id"] for pair in measurement["pairs"]] != [
                f"{channel[0].upper()}R{i:02d}" for i in range(1, len(normalized) + 1)
            ]:
                raise ValueError("Stereo repetition pair identifiers differ")
            validate_mono({**measurement, "pairs": normalized}, {**source, "pcm_sha256": measurement["pcm_sha256"]})
            if "gain_measurement" in row:
                gain.validate(row["gain_measurement"], source, channel, measurement["pcm_sha256"])
        if len({"gain_measurement" in row for row in record["channels"]}) != 1:
            raise ValueError("Gain repetition coverage differs between recorded channels")
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError("Invalid stereo repetition evidence") from error
