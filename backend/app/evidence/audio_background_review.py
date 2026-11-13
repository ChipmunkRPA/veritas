"""Bind the processed response and its paired playback to the recorded source clock."""

import base64
import hashlib
import json
import math

import numpy as np

from app.detectors.audio import background_voice_review as detector
from app.detectors.audio.decode import source_ranges, wav_bytes
from app.evidence import audio_background_review_binding as binding


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _entry(store, pin):
    if not isinstance(pin, str) or not binding.SHA.fullmatch(pin):
        raise ValueError("Invalid background-reduced speech receipt pin")
    data = store.read_bytes(pin)
    if _sha(data) != pin:
        raise ValueError("Background-reduced speech receipt hash changed")
    return json.loads(data)


def _processed(record, store, source):
    if set(record) != {"raw_output_sha256", "primary_raw_output_sha256"}:
        raise ValueError("Unexpected background-reduced speech receipt references")
    entry = _entry(store, record["raw_output_sha256"])
    details = entry["details"]
    primary = _entry(store, record["primary_raw_output_sha256"])
    direction = detector.primary_direction(primary)
    source_sha = source["pcm_sha256"]
    if (entry["detector_id"] != binding.DETECTOR_ID or entry["status"] != "ok"
            or details["source_pcm_sha256"] != source_sha or direction not in {"inconclusive", "unassessed"}
            or direction != details["primary_direction"]
            or (direction == "inconclusive" and primary["details"]["audio_input"]["pcm_sha256"] != source_sha)):
        raise ValueError("Background-reduced speech receipt belongs to a different source or primary response")
    admitted, reading = details["admitted"], details["response_hex"]
    if type(admitted) is not bool or (admitted and not isinstance(reading, str)) or (not admitted and reading is not None):
        raise ValueError("Invalid background-reduced speech response")
    logit = float.fromhex(reading) if admitted else None
    if admitted and not math.isfinite(logit):
        raise ValueError("Invalid background-reduced speech response")
    status = detector.disposition(admitted, logit)
    if details["status"] != status:
        raise ValueError("Background-reduced speech disposition differs from its receipt")
    if status != "flagged":
        return status, None
    processed = base64.b64decode(details["processed_pcm"], validate=True)
    if len(processed) != binding.SPAN * 2 or _sha(processed) != details["processed_pcm_sha256"]:
        raise ValueError("Processed PCM differs from the measured interval")
    return status, processed


def _waveform(pcm, waveform):
    return waveform(pcm, [], start=0, end=binding.SPAN, selected=(0, binding.SPAN), scores=False)


def build(record, *, store, source, pcm, mapping, asset, waveform):
    if (not binding.supported_source(mapping) or _sha(pcm.astype("<f4").tobytes()) != source["pcm_sha256"]):
        raise ValueError("Background-reduced speech PCM differs from the source")
    if set(record) == {"status", "reason"} and record["status"] == "skipped":
        status, reason, processed = "skipped", record["reason"], None
    else:
        status, processed = _processed(record, store, source)
        reason = binding.REASONS[status]
    document = {"schema": binding.SCHEMA, "title": binding.TITLE, "status": status, "reason": reason,
                "limitation": binding.LIMITATION, "affects_overall_confidence": False, "interval": dict(binding.INTERVAL),
                "source_timestamp_ranges": source_ranges(mapping, 0, binding.SPAN), "artifacts": {}}
    if processed is not None:
        wave = np.frombuffer(processed, dtype="<i2").astype(np.float32) / 32768.0
        metadata = {"samples": binding.SPAN, "sample_rate": binding.RATE, "channels": 1}
        document["processed_pcm_sha256"] = _sha(processed)
        document["artifacts"] = {"selected_clip": asset(wav_bytes(pcm), "audio/wav", **metadata),
                                 "processed_clip": asset(wav_bytes(wave), "audio/wav", **metadata),
                                 "waveform": asset(_waveform(pcm, waveform), "image/png", width=960, height=280)}
    binding.validate(document, source)
    return document


def verify(document, record, *, store, source, pcm, mapping, read_asset, waveform):
    assets = {item["sha256"]: item for item in document["artifacts"].values()}

    def checked(data, media_type, **metadata):
        pin = _sha(data)
        asset = assets[pin]
        if (read_asset(pin) != data or asset["media_type"] != media_type
                or any(asset.get(key) != value for key, value in metadata.items())):
            raise ValueError("Background-reduced speech asset differs from its frozen input")
        return asset

    expected = build(record, store=store, source=source, pcm=pcm, mapping=mapping, asset=checked, waveform=waveform)
    if expected != document:
        raise ValueError("Background-reduced speech presentation differs from its receipt")
