"""Freeze the optional speech clue and its exact PCM and waveform assets."""

from __future__ import annotations

import hashlib
import json
import math

from app.detectors.audio.decode import source_ranges, wav_bytes
from app.evidence import audio_short_review_binding as binding


def _receipt(record, store, source):
    if not (pin := record.get("raw_output_sha256")):
        if record.get("status") != "skipped" or record.get("responses") or record.get("selected") is not None:
            raise ValueError("Unbound short speech-window response")
        return record
    raw = store.read_bytes(pin)
    if hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError("Short speech-window receipt hash changed")
    entry = json.loads(raw)
    details = entry.get("details", {})
    if (entry.get("detector_id") != binding.DETECTOR_ID or entry.get("status") != "ok"
            or details.get("pcm_sha256") != source["pcm_sha256"] or not binding.supported_source(source)):
        raise ValueError("Short speech-window receipt belongs to a different check or source")
    return details["short_voice_review"]


def build(record, *, store, source, pcm, mapping, asset, waveform):
    raw = _receipt(record, store, source)
    responses = raw["responses"]
    checked = [{key: row[key] for key in ("start_sample", "end_sample")} for row in responses]
    if raw["schema"] != binding.SCHEMA or any(type(row["logit"]) not in (int, float) or not math.isfinite(row["logit"]) for row in responses):
        raise ValueError("Short speech-window response is invalid")
    selected = binding.selected_range(responses, source.get("source_codec"))
    if raw["selected"] != selected or (selected is not None) != (raw["status"] == "flagged"):
        raise ValueError("Short speech-window flag differs from its frozen response")
    document = {
        "schema": binding.SCHEMA, "title": binding.TITLE, "status": raw["status"], "reason": raw["reason"],
        "limitation": binding.LIMITATION, "affects_overall_confidence": False, "checked_windows": checked,
        "interval": selected, "start_s": None, "end_s": None, "source_timestamp_ranges": [], "artifacts": {},
    }
    if selected:
        a, b = selected["start_sample"], selected["end_sample"]
        document.update(start_s=a / binding.RATE, end_s=b / binding.RATE, source_timestamp_ranges=source_ranges(mapping, a, b))
        document["artifacts"] = {
            "selected_clip": asset(wav_bytes(pcm[a:b]), "audio/wav", samples=b - a, sample_rate=binding.RATE, channels=1),
            "waveform": asset(waveform(pcm, [], start=0, end=len(pcm), selected=(a, b), scores=False),
                              "image/png", width=960, height=280),
        }
    binding.validate(document, source)
    return document


def verify(document, record, *, store, source, pcm, mapping, read_asset, waveform):
    def checked(data, media_type, **metadata):
        role = "selected_clip" if media_type == "audio/wav" else "waveform"
        asset = document["artifacts"][role]
        pin = hashlib.sha256(data).hexdigest()
        if (asset["sha256"] != pin or asset["media_type"] != media_type or read_asset(pin) != data
                or any(asset.get(key) != value for key, value in metadata.items())):
            raise ValueError("Short speech-window asset differs from its frozen samples")
        return asset

    expected = build(record, store=store, source=source, pcm=pcm, mapping=mapping, asset=checked, waveform=waveform)
    if expected != document:
        raise ValueError("Short speech-window presentation differs from its frozen response")
