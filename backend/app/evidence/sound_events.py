"""Frozen mixed-track sound tags, separate from voice authenticity evidence."""
from __future__ import annotations

import hashlib
import json

from app.core.canonical import canonical_hash
from app.evidence.sound_event_binding import validate
from app.detectors.audio.decode import source_ranges, wav_bytes

SCHEMA = "chipmunk.sound-events/1"
LIMITATIONS = [
    "These are possible sound categories in a mixed recording. Voice, music and environmental sounds are not separated.",
    "Scores rank sound categories; they are not probabilities of AI generation, editing, or authenticity.",
    "Each response covers a complete window of up to ten seconds and does not locate an exact sound onset or edit boundary.",
]


def _read(store, sha):
    raw = store.read_bytes(sha)
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Sound-context raw output hash differs")
    return json.loads(raw)


def recorded_windows(store, record, binding):
    windows = []
    for ref in record.get("sound_checks", []):
        raw = _read(store, ref["raw_output_sha256"])
        if raw.get("detector_id") != ref["detector_id"] or raw.get("status") != "ok":
            raise ValueError("Sound context does not identify a completed check")
        details = raw.get("details", {})
        input_ = details.get("audio_input", {})
        if any(input_.get(key) != binding[key] for key in ("pcm_sha256", "samples", "sample_rate")):
            raise ValueError("Sound categories refer to different source samples")
        for row in details.get("sound_windows", []):
            windows.append({**row, "raw_output_sha256": ref["raw_output_sha256"]})
    return windows


def build(store, record, binding, source, pcm, mapping, make_asset, waveform):
    rows = recorded_windows(store, record, binding)
    if not record.get("sound_checks"):
        return None
    out = {"schema": SCHEMA, "status": "ready", "source": {k: source[k] for k in
           ("sha256", "pcm_sha256", "sample_rate", "samples")},
           "method": {"window_samples": 160000, "hop_samples": 160000,
                      "description": "Recorded sound-category responses for complete non-overlapping source windows."},
           "windows": [], "limitations": LIMITATIONS}
    for i, row in enumerate(rows, 1):
        a, b = row["start_sample"], row["end_sample"]
        out["windows"].append({"id": f"C{i:02d}", "start_sample": a, "end_sample": b,
            "start_s": a/16000, "end_s": b/16000,
            "labels": [{"label": v["label"], "score": v["score"]} for v in row["labels"]],
            "raw_output_sha256": row["raw_output_sha256"], "source_timestamp_ranges": source_ranges(mapping, a, b),
            "artifacts": {"selected_clip": make_asset(wav_bytes(pcm[a:b]), "audio/wav", samples=b-a, sample_rate=16000, channels=1),
                          "waveform": make_asset(waveform(pcm, [], start=a, end=b, selected=(a,b), scores=False),
                                                 "image/png", width=960, height=280)}})
    validate(out, source)
    return out



def verify(context, *, store, record, binding, source, pcm, mapping, waveform):
    validate(context, source)
    def asset(data, media_type, **metadata):
        return {"sha256": hashlib.sha256(data).hexdigest(), "media_type": media_type, **metadata}
    expected = build(store, record, binding, source, pcm, mapping, asset, waveform)
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k != "url"}
        return [strip(v) for v in value] if isinstance(value, list) else value
    if canonical_hash(strip(expected)) != canonical_hash(strip(context)):
        raise ValueError("Sound context differs from its frozen check or source samples")
    for window in context["windows"]:
        for item in window["artifacts"].values():
            if hashlib.sha256(store.read_bytes(item["sha256"])).hexdigest() != item["sha256"]:
                raise ValueError("Sound-context asset hash differs")
