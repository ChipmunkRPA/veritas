"""Freeze the optional speech clue and its exact PCM and waveform assets."""

from __future__ import annotations

import hashlib
import json
import math

from app.detectors.audio.decode import source_ranges, wav_bytes
from app.evidence import audio_short_review_binding as binding


def identical_stereo_proof(decoded) -> str | None:
    if not binding.eligible_stereo_source(decoded.mapping):
        return None
    import numpy as np

    channels = decoded.stereo_pcm
    if channels is None or channels.shape != (binding.SAMPLES, 2):
        return None
    left, right = channels.T
    if not np.array_equal(left, right) or not np.array_equal(left.astype(np.float32) / 32768, decoded.pcm):
        return None
    return hashlib.sha256(left.astype("<i2", copy=False).tobytes()).hexdigest()


def _receipt(record, store, source):
    if not (pin := record.get("raw_output_sha256")):
        if (record.get("schema") != binding.SCHEMA or record.get("status") != "skipped"
                or record.get("responses") or record.get("selected") is not None):
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


def _nearby(pin, *, store, source, pcm, selected):
    raw = store.read_bytes(pin)
    if hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError("Nearby speech-window receipt hash changed")
    entry = json.loads(raw)
    details = entry.get("details", {})
    if (entry.get("detector_id") != binding.NEARBY_ID or entry.get("status") != "ok"
            or details.get("pcm_sha256") != source["pcm_sha256"] or details.get("anchor") != selected
            or selected is None or source.get("source_codec") not in binding.NEARBY_CODECS):
        raise ValueError("Nearby speech-window receipt belongs to another check or source")
    windows = []
    for row in details["windows"]:
        a, b, speech, logit = (row[key] for key in ("start_sample", "end_sample", "speech_samples", "logit"))
        if ((logit is None) != (speech < binding.RATE) or
                (logit is not None and (type(logit) not in (int, float) or not math.isfinite(logit)))):
            raise ValueError("Invalid nearby speech-window response")
        status = "below_speech_minimum" if logit is None else "elevated" if logit > binding.REVIEW_POINT else "below_review_point"
        windows.append({"start_sample": a, "end_sample": b, "speech_samples": speech, "status": status,
                        "source_timestamp_ranges": [binding.source_range(source, a, b)],
                        "pcm_sha256": hashlib.sha256(pcm[a:b].astype("<f4").tobytes()).hexdigest()})
    return {"status": binding.nearby_summary(windows), "windows": windows}


def build(record, *, store, source, pcm, mapping, asset, waveform):
    raw = _receipt(record, store, source)
    responses = raw["responses"]
    checked = [{key: row[key] for key in ("start_sample", "end_sample")} for row in responses]
    if raw["schema"] not in {binding.SCHEMA, binding.COVERAGE_SCHEMA} or any(
            type(row["logit"]) not in (int, float) or not math.isfinite(row["logit"]) for row in responses):
        raise ValueError("Short speech-window response is invalid")
    selected = binding.selected_range(responses, source.get("source_codec"))
    if raw["selected"] != selected or (selected is not None) != (raw["status"] == "flagged"):
        raise ValueError("Short speech-window flag differs from its frozen response")
    document = {
        "schema": raw["schema"], "title": binding.TITLE, "status": raw["status"], "reason": raw["reason"],
        "limitation": binding.LIMITATION, "affects_overall_confidence": False, "checked_windows": checked,
        "interval": selected, "start_s": None, "end_s": None, "source_timestamp_ranges": [], "artifacts": {},
    }
    if raw["schema"] == binding.COVERAGE_SCHEMA:
        document["window_coverage"] = binding.window_coverage(raw["speech_counts"], source)
    if selected:
        a, b = selected["start_sample"], selected["end_sample"]
        document.update(start_s=a / binding.RATE, end_s=b / binding.RATE, source_timestamp_ranges=source_ranges(mapping, a, b))
        document["artifacts"] = {
            "selected_clip": asset(wav_bytes(pcm[a:b]), "audio/wav", samples=b - a, sample_rate=binding.RATE, channels=1),
            "waveform": asset(waveform(pcm, [], start=0, end=binding.SAMPLES, selected=(a, b), scores=False),
                              "image/png", width=960, height=280),
        }
    if pin := record.get("nearby_raw_output_sha256"):
        document["schema"] = binding.NEARBY_SCHEMA
        document["nearby_window_review"] = _nearby(pin, store=store, source=source, pcm=pcm, selected=selected)
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


def verify_nearby_source(review, pcm):
    for row in review["nearby_window_review"]["windows"]:
        actual = pcm[row["start_sample"]:row["end_sample"]]
        if hashlib.sha256(actual.astype("<f4").tobytes()).hexdigest() != row["pcm_sha256"]:
            raise ValueError("Nearby speech samples differ from the original audio")
