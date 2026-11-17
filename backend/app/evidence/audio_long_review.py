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
    skipped = set(record) in ({"reason"}, {"reason", "schema"}) and record["reason"] in two_second.SKIP_REASONS
    raw = None if skipped else _receipt(record, store, source, detector_id=two_second.DETECTOR_ID, key=two_second.KEY)
    schema = record.get("schema", two_second.SCHEMA) if skipped else raw.get("schema")
    if schema not in two_second.LIMITATIONS or (raw is not None and set(raw) != {"schema", "excerpts"}):
        raise ValueError("Malformed two-second scan receipt")
    if not two_second.supported_source(source, schema) or not two_second.supported_source(mapping, schema):
        raise ValueError("Two-second scan is outside its source profile")
    public = {"schema": schema, "title": two_second.TITLE, "limitation": two_second.LIMITATIONS[schema],
              "affects_overall_confidence": False}
    if skipped:
        return {**public, "status": "skipped", "reason": record["reason"], "excerpts": [], "regions": []}
    presented = schema == two_second.PRESENTED_SCHEMA
    extent = two_second.presented_samples(mapping) if presented else len(pcm)
    codec = mapping.get("source_codec") if presented else None
    excerpts = two_second.summarize_readings(raw["excerpts"], extent, schema=schema, codec=codec)
    _check_excerpt_inputs(excerpts, pcm)

    def freeze_window(region):
        start, end = region["start_sample"], region["end_sample"]
        window = {**region, "pcm_sha256": _digest(pcm[start:end].astype("<f4").tobytes())}
        if presented:
            window["native_source_range"] = two_second.native_range(start, end)
        return {**window, "source_timestamp_ranges": _sample_witness(window, pcm, mapping), "artifacts": artifacts(window)}

    regions = []
    for region in two_second.regions(excerpts, schema=schema):
        selected = freeze_window(region)
        if confirming := region.get("confirming_window"):
            selected["confirming_window"] = freeze_window(confirming)
        regions.append(selected)
    status = two_second.status(excerpts, regions)
    return {**public, "status": status, "reason": two_second.REASONS_BY_SCHEMA[schema][status], "excerpts": excerpts, "regions": regions}


def _artifacts(reading, pcm, waveform, asset):
    start, end = reading["start_sample"], reading["end_sample"]
    return {"selected_clip": asset(wav_bytes(pcm[start:end]), "audio/wav", samples=end - start, sample_rate=binding.RATE, channels=1),
            "waveform": asset(_waveform(reading, pcm, waveform), "image/png", width=960, height=280)}


def build_presented(record, *, store, source, pcm, mapping, asset, waveform):
    extent = two_second.presented_samples(mapping)
    review = _two_second_review(record, store=store, source=source, pcm=pcm, mapping=mapping,
                                artifacts=lambda row: _artifacts(row, pcm[:extent], waveform, asset))
    if review["schema"] != two_second.PRESENTED_SCHEMA:
        raise ValueError("Presented audio requires the native-clock two-second review")
    two_second.validate(review, source)
    return review


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
        return _artifacts(reading, pcm, waveform, asset)

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
    presented = document["schema"] == two_second.PRESENTED_SCHEMA
    windows = two_second.media_windows(document) if presented else binding.all_media_windows(document)
    assets = {item["sha256"]: item for window in windows for item in window["artifacts"].values()}

    def checked(data, media_type, **metadata):
        pin = _digest(data)
        asset = assets[pin]
        if (read_asset(pin) != data or asset["media_type"] != media_type
                or any(asset.get(key) != value for key, value in metadata.items())):
            raise ValueError("Long speech-window asset differs from its recorded samples")
        return asset

    if presented:
        expected = build_presented(record, store=store, source=source, pcm=pcm, mapping=mapping, asset=checked, waveform=waveform)
    else:
        expected = build(record, store=store, source=source, pcm=pcm, mapping=mapping, asset=checked, waveform=waveform,
                         include_sample_coverage=binding.COVERAGE in document)
    if expected != document:
        raise ValueError("Long speech-window presentation differs from its frozen response")


def verify_source(document, *, pcm, mapping, waveform):
    """Compare exact sample inputs and waveform digests without scoring the audio."""
    binding.validate(document, mapping)
    extension = document.get(two_second.KEY, {})
    _check_excerpt_inputs(extension.get("excerpts", ()), pcm)
    for window in [*document["windows"], *document.get("regions", []), *two_second.media_windows(extension)]:
        if _sample_witness(window, pcm, mapping) != window["source_timestamp_ranges"]:
            raise ValueError("Long speech-window source timestamps differ from the original clock")
        if (asset := window["artifacts"].get("waveform")) and _digest(_waveform(window, pcm, waveform)) != asset["sha256"]:
            raise ValueError("Long speech-window waveform differs from the original audio")


def verify_presented_source(document, *, pcm, mapping, waveform):
    two_second.validate(document, mapping)
    extent = two_second.presented_samples(mapping)
    _check_excerpt_inputs(document["excerpts"], pcm)
    for window in two_second.media_windows(document):
        if _sample_witness(window, pcm, mapping) != window["source_timestamp_ranges"] or (
                _digest(_waveform(window, pcm[:extent], waveform)) != window["artifacts"]["waveform"]["sha256"]):
            raise ValueError("Presented speech-window evidence differs from the original audio")
