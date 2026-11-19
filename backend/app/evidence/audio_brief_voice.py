"""Bind the brief speech selection to the recorded windows and original media."""

from __future__ import annotations

from app.detectors.audio.decode import wav_bytes
from app.evidence import audio_brief_voice_binding as binding
from app.evidence.audio_long_review import _artifacts, _digest, _receipt, _sample_witness, _waveform


def _pcm_digest(pcm, start):
    return _digest(pcm[start:start + binding.WIDTH].astype("<f4").tobytes())


def build(record, *, store, source, pcm, mapping, asset, waveform):
    if not binding.supported_source(mapping) or len(pcm) != binding.SAMPLES:
        raise ValueError("Brief speech-window input differs from the native PCM16 WAV or FLAC profile")
    if set(record) == {"status", "reason"} and record["status"] == "skipped":
        review = {**binding.presentation([], reason=record["reason"]), "windows": []}
    else:
        if set(record) != {"raw_output_sha256"}:
            raise ValueError("Missing frozen brief speech-window response")
        raw = _receipt(record, store, source, detector_id=binding.DETECTOR_ID, key=binding.KEY)
        if set(raw) != {"schema", "readings"} or raw["schema"] != binding.SCHEMA:
            raise ValueError("Malformed brief speech-window receipt")
        readings = raw["readings"]
        checks = binding.raw_checks(readings)
        if any(_pcm_digest(pcm, row["start_sample"]) != row["pcm_sha256"] for row in readings):
            raise ValueError("Brief speech-window readings belong to different source samples")
        windows = []
        for index in binding.selected_indices(checks):
            window = {**binding.window_fields(index), "pcm_sha256": readings[index]["pcm_sha256"]}
            windows.append({**window, "source_timestamp_ranges": _sample_witness(window, pcm, mapping),
                            "artifacts": _artifacts(window, pcm, waveform, asset)})
        review = {**binding.presentation(checks), "windows": windows}
    binding.validate(review, source)
    return review


def verify(review, record, *, store, source, pcm, mapping, read_asset, waveform):
    assets = {item["sha256"]: item for window in review["windows"] for item in window["artifacts"].values()}

    def checked(data, media_type, **metadata):
        pin = _digest(data)
        asset = assets[pin]
        if (read_asset(pin) != data or asset["media_type"] != media_type
                or any(asset.get(key) != value for key, value in metadata.items())):
            raise ValueError("Brief speech-window media differs from its recorded samples")
        return asset

    if build(record, store=store, source=source, pcm=pcm, mapping=mapping, asset=checked, waveform=waveform) != review:
        raise ValueError("Brief speech-window presentation differs from the frozen response")


def verify_source(review, *, pcm, mapping, waveform):
    """Rebuild the exact listening ranges and media without running inference."""
    binding.validate(review, mapping)
    for window in review["windows"]:
        start, end = window["start_sample"], window["end_sample"]
        if (_sample_witness(window, pcm, mapping) != window["source_timestamp_ranges"]
                or _digest(wav_bytes(pcm[start:end])) != window["artifacts"]["selected_clip"]["sha256"]
                or _digest(_waveform(window, pcm, waveform)) != window["artifacts"]["waveform"]["sha256"]):
            raise ValueError("Brief speech-window evidence differs from the original timeline or samples")
