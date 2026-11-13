"""Freeze the recorded window decisions and their exact source media."""

from __future__ import annotations

import hashlib
import json

from app.detectors.audio.decode import source_ranges, wav_bytes
from app.evidence import audio_long_review_binding as binding


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _receipt(record, store, source):
    pin = record.get("raw_output_sha256")
    if not pin:
        if record.get("status") != "skipped":
            raise ValueError("Unbound long speech-window response")
        return record
    raw = store.read_bytes(pin)
    if _digest(raw) != pin:
        raise ValueError("Long speech-window receipt hash changed")
    entry = json.loads(raw)
    details = entry["details"]
    if (entry["detector_id"] != binding.DETECTOR_ID or entry["status"] != "ok"
            or details["pcm_sha256"] != source["pcm_sha256"]):
        raise ValueError("Long speech-window receipt belongs to a different check or source")
    return details["long_voice_review"]


def _sample_witness(window, pcm, mapping):
    start, end = window["start_sample"], window["end_sample"]
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(pcm)
            or _digest(pcm[start:end].astype("<f4").tobytes()) != window["pcm_sha256"]):
        raise ValueError("Long speech-window PCM differs from the recorded source samples")
    return source_ranges(mapping, start, end)


def _waveform(window, pcm, waveform):
    return waveform(pcm, [], start=0, end=len(pcm), selected=(window["start_sample"], window["end_sample"]), scores=False)


def build(record, *, store, source, pcm, mapping, asset, waveform):
    raw = _receipt(record, store, source)
    focused = raw.get("schema") == binding.FOCUSED_SCHEMA
    fields = {"schema", "status", "reason", "windows", "regions"} if focused else {"schema", "status", "reason", "windows"}
    if (set(raw) != fields or not binding.supported_source(mapping)
            or not binding.supported_source(source)):
        raise ValueError("Long speech-window receipt is outside its native source profile")

    def artifacts(reading):
        start, end = reading["start_sample"], reading["end_sample"]
        return {"selected_clip": asset(wav_bytes(pcm[start:end]), "audio/wav", samples=end - start,
                                      sample_rate=binding.RATE, channels=1),
                "waveform": asset(_waveform(reading, pcm, waveform), "image/png", width=960, height=280)}

    windows = []
    for reading in raw["windows"]:
        if set(reading) != {"start_sample", "end_sample", "pcm_sha256", "direction"}:
            raise ValueError("Unexpected long speech-window receipt fields")
        witness = _sample_witness(reading, pcm, mapping)
        assets = artifacts(reading) if not focused and raw["status"] == "corroborated" and reading["direction"] == "ai" else {}
        windows.append({**reading, "source_timestamp_ranges": witness, "artifacts": assets})
    document = {**raw, "title": binding.TITLE, "limitation": binding.FOCUSED_LIMITATION if focused else binding.LIMITATION,
                "affects_overall_confidence": False, "windows": windows}
    if focused:
        regions = []
        for reading in raw["regions"]:
            if set(reading) != {"start_sample", "end_sample", "basis", "supporting_windows", "pcm_sha256"}:
                raise ValueError("Unexpected focused speech region fields")
            regions.append({**reading, "source_timestamp_ranges": _sample_witness(reading, pcm, mapping),
                            "artifacts": artifacts(reading)})
        document["regions"] = regions
    binding.validate(document, source)
    return document


def verify(document, record, *, store, source, pcm, mapping, read_asset, waveform):
    assets = {item["sha256"]: item for window in binding.media_windows(document) for item in window["artifacts"].values()}

    def checked(data, media_type, **metadata):
        pin = _digest(data)
        asset = assets[pin]
        if (read_asset(pin) != data or asset["media_type"] != media_type
                or any(asset.get(key) != value for key, value in metadata.items())):
            raise ValueError("Long speech-window asset differs from its recorded samples")
        return asset

    expected = build(record, store=store, source=source, pcm=pcm, mapping=mapping, asset=checked, waveform=waveform)
    if expected != document:
        raise ValueError("Long speech-window presentation differs from its frozen response")


def verify_source(document, *, pcm, mapping, waveform):
    """Compare exact sample inputs and waveform digests without scoring the audio."""
    binding.validate(document, mapping)
    for window in [*document["windows"], *document.get("regions", [])]:
        if _sample_witness(window, pcm, mapping) != window["source_timestamp_ranges"]:
            raise ValueError("Long speech-window source timestamps differ from the original clock")
        if (asset := window["artifacts"].get("waveform")) and _digest(_waveform(window, pcm, waveform)) != asset["sha256"]:
            raise ValueError("Long speech-window waveform differs from the original audio")
