"""Freeze the recorded window decisions and their exact source media."""

from __future__ import annotations

import hashlib
import json

from app.detectors.audio.decode import source_ranges, wav_bytes
from app.evidence import audio_long_review_binding as binding
from app.evidence import audio_two_second_binding as two_second
from app.evidence.audio_short_review_binding import SAMPLES as EXCERPT_SAMPLES


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _receipt(record, store, source, *, detector_id=binding.DETECTOR_ID, key="long_voice_review"):
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
    if (entry["detector_id"] != detector_id or entry["status"] != "ok"
            or details["pcm_sha256"] != source["pcm_sha256"]):
        raise ValueError("Long speech-window receipt belongs to a different check or source")
    return details[key]


def _sample_witness(window, pcm, mapping):
    start, end = window["start_sample"], window["end_sample"]
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(pcm)
            or _digest(pcm[start:end].astype("<f4").tobytes()) != window["pcm_sha256"]):
        raise ValueError("Long speech-window PCM differs from the recorded source samples")
    return source_ranges(mapping, start, end)


def _waveform(window, pcm, waveform):
    return waveform(pcm, [], start=0, end=len(pcm), selected=(window["start_sample"], window["end_sample"]), scores=False)


def _check_excerpt_inputs(excerpts, pcm):
    for row in excerpts:
        start = row["start_sample"]
        if _digest(pcm[start:start + EXCERPT_SAMPLES].astype("<f4").tobytes()) != row["pcm_sha256"]:
            raise ValueError("Six-second speech excerpt differs from the original samples")


def _two_second_review(record, *, store, source, pcm, mapping, artifacts):
    if not two_second.supported_source(source) or not two_second.supported_source(mapping):
        raise ValueError("Two-second scan is outside its source profile")
    public = {"schema": two_second.SCHEMA, "title": two_second.TITLE, "limitation": two_second.LIMITATION,
              "affects_overall_confidence": False}
    if set(record) == {"reason"} and record["reason"] in two_second.SKIP_REASONS:
        return {**public, "status": "skipped", "reason": record["reason"], "excerpts": [], "regions": []}
    raw = _receipt(record, store, source, detector_id=two_second.DETECTOR_ID, key=two_second.KEY)
    if set(raw) != {"schema", "excerpts"} or raw["schema"] != two_second.SCHEMA:
        raise ValueError("Malformed two-second scan receipt")
    excerpts = two_second.summarize_readings(raw["excerpts"], len(pcm))
    _check_excerpt_inputs(excerpts, pcm)
    regions = []
    for region in two_second.regions(excerpts):
        start, end = region["start_sample"], region["end_sample"]
        window = {**region, "pcm_sha256": _digest(pcm[start:end].astype("<f4").tobytes())}
        regions.append({**window, "source_timestamp_ranges": _sample_witness(window, pcm, mapping), "artifacts": artifacts(window)})
    status = two_second.status(excerpts)
    return {**public, "status": status, "reason": two_second.REASONS[status], "excerpts": excerpts, "regions": regions}


def build(record, *, store, source, pcm, mapping, asset, waveform, include_sample_coverage=True):
    extension = record.get(two_second.KEY)
    raw = _receipt({key: value for key, value in record.items() if key != two_second.KEY}, store, source)
    schema = raw.get("schema")
    focused = schema in binding.FOCUSED_SCHEMAS
    fields = {"schema", "status", "reason", "windows", "regions"} if focused else {"schema", "status", "reason", "windows"}
    if (set(raw) != fields or schema not in binding.LIMITATIONS or not binding.supported_review_source(mapping, schema)
            or not binding.supported_review_source(source, schema)):
        raise ValueError("Long speech-window receipt is outside its source profile")

    def artifacts(reading):
        start, end = reading["start_sample"], reading["end_sample"]
        return {"selected_clip": asset(wav_bytes(pcm[start:end]), "audio/wav", samples=end - start,
                                      sample_rate=binding.RATE, channels=1),
                "waveform": asset(_waveform(reading, pcm, waveform), "image/png", width=960, height=280)}

    windows = []
    for reading in raw["windows"]:
        binding.validate_window_fields(reading)
        witness = _sample_witness(reading, pcm, mapping)
        assets = artifacts(reading) if not focused and raw["status"] == "corroborated" and reading["direction"] == "ai" else {}
        windows.append({**reading, "source_timestamp_ranges": witness, "artifacts": assets})
    document = {**raw, "title": binding.TITLE, "limitation": binding.LIMITATIONS[schema],
                "affects_overall_confidence": False, "windows": windows}
    if focused:
        regions = []
        for reading in raw["regions"]:
            if set(reading) != {"start_sample", "end_sample", "basis", "supporting_windows", "pcm_sha256"}:
                raise ValueError("Unexpected focused speech region fields")
            regions.append({**reading, "source_timestamp_ranges": _sample_witness(reading, pcm, mapping),
                            "artifacts": artifacts(reading)})
        document["regions"] = regions
    if extension is not None:
        document[two_second.KEY] = _two_second_review(extension, store=store, source=source, pcm=pcm, mapping=mapping, artifacts=artifacts)
    binding.validate(document, source)
    if include_sample_coverage and windows:
        document[binding.COVERAGE] = binding.sample_coverage(len(pcm), windows)
    return document


def verify(document, record, *, store, source, pcm, mapping, read_asset, waveform):
    assets = {item["sha256"]: item for window in binding.all_media_windows(document) for item in window["artifacts"].values()}

    def checked(data, media_type, **metadata):
        pin = _digest(data)
        asset = assets[pin]
        if (read_asset(pin) != data or asset["media_type"] != media_type
                or any(asset.get(key) != value for key, value in metadata.items())):
            raise ValueError("Long speech-window asset differs from its recorded samples")
        return asset

    expected = build(record, store=store, source=source, pcm=pcm, mapping=mapping, asset=checked, waveform=waveform,
                     include_sample_coverage=binding.COVERAGE in document)
    if expected != document:
        raise ValueError("Long speech-window presentation differs from its frozen response")


def verify_source(document, *, pcm, mapping, waveform):
    """Compare exact sample inputs and waveform digests without scoring the audio."""
    binding.validate(document, mapping)
    extension = document.get(two_second.KEY, {})
    _check_excerpt_inputs(extension.get("excerpts", ()), pcm)
    for window in [*document["windows"], *document.get("regions", []), *extension.get("regions", ())]:
        if _sample_witness(window, pcm, mapping) != window["source_timestamp_ranges"]:
            raise ValueError("Long speech-window source timestamps differ from the original clock")
        if (asset := window["artifacts"].get("waveform")) and _digest(_waveform(window, pcm, waveform)) != asset["sha256"]:
            raise ValueError("Long speech-window waveform differs from the original audio")
