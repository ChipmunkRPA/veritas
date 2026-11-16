"""Bind the focused speech response and its audio assets to the saved source."""

from __future__ import annotations

import hashlib
import json

from app.detectors.audio.decode import source_ranges, wav_bytes
from app.evidence import audio_focused_edit_binding as binding


def build(record, *, store, source, pcm, mapping, asset, waveform):
    count, interval = 0, None
    if pin := record.get("raw_output_sha256"):
        raw = store.read_bytes(pin)
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError("Focused speech response hash changed")
        entry = json.loads(raw)
        details = entry.get("details", {})
        if (entry.get("detector_id") != binding.DETECTOR_ID or entry.get("status") != "ok"
                or details.get("pcm_sha256") != source["pcm_sha256"]):
            raise ValueError("Focused speech response belongs to a different check or source")
        response = details[binding.KEY]
        count, best = response["checked_windows"], response["best"]
        if type(count) is not int or (count == 0) != (best is None):
            raise ValueError("Focused speech response differs from its checked coverage")
        outcome = "no_speech" if best is None else "no_flag"
        if best is not None:
            a, b = best["start_sample"], best["end_sample"]
            if type(best["score_units"]) is not int or not binding.on_grid(a, b):
                raise ValueError("Focused speech response is outside its timing grid")
            if best["score_units"] > binding.REVIEW_POINT:
                outcome, interval = "flagged", {"start_sample": a, "end_sample": b}
    else:
        outcome = record["reason"]
        if outcome not in {"time", "unavailable"}:
            raise ValueError("Unexpected unmeasured focused speech response")
    status = outcome if outcome in {"flagged", "no_flag"} else "skipped"
    resampled = source.get("source_sample_rate") == binding.LOSSLESS_RATE
    document = {"schema": binding.RESAMPLED_SCHEMA if resampled else binding.SCHEMA,
                "title": binding.TITLE, "status": status,
                "reason": binding.REASONS[outcome], "limitation": binding.LIMITATION,
                "affects_overall_confidence": False, "checked_windows": count, "interval": interval,
                "start_s": None, "end_s": None, "source_timestamp_ranges": [], "artifacts": {}}
    if resampled:
        document["original_timestamp_range"] = binding.original_range(a, b) if interval else None
    if interval is not None:
        document.update(start_s=a / binding.RATE, end_s=b / binding.RATE, source_timestamp_ranges=source_ranges(mapping, a, b))
        document["artifacts"] = {
            "selected_clip": asset(wav_bytes(pcm[a:b]), "audio/wav", samples=b - a, sample_rate=binding.RATE, channels=1),
            "waveform": asset(waveform(pcm, [], start=0, end=binding.SAMPLES, selected=(a, b), scores=False),
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
            raise ValueError("Focused speech asset differs from its saved audio")
        return asset

    expected = build(record, store=store, source=source, pcm=pcm, mapping=mapping, asset=checked, waveform=waveform)
    if document != expected:
        raise ValueError("Focused speech presentation differs from its saved response")
