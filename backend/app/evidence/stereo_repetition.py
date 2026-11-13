"""Source-locked exact repeat checks on individual stereo channels."""

from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np

from app.detectors.audio.decode import source_ranges
from app.evidence import audio_exact_repetition, audio_repetition
from app.evidence.stereo_repetition_binding import CHANNELS, KEY, SCHEMA, validate
from app.storage.artefacts import register_artefact
from app.storage.objects import ObjectIntegrityError


def native_pcm(decoded):
    mapping, channels = decoded.mapping, decoded.stereo_pcm
    if (channels is None or len(channels) != mapping["samples"] or mapping["source_channels"] != 2
            or mapping["source_sample_rate"] != audio_exact_repetition.SAMPLE_RATE or len(mapping["timeline"]) != 1
            or not audio_exact_repetition.MIN_SAMPLES <= len(channels) <= audio_exact_repetition.MAX_SAMPLES):
        return None
    codec = mapping.get("source_codec")
    if codec not in {"pcm_s16le", "pcm_s16be"} and (codec, mapping.get("source_bit_depth")) != ("flac", 16):
        return None
    return channels.astype("<i2", copy=False).tobytes()


def freeze(db, store, analysis_id, decoded):
    if (pcm := native_pcm(decoded)) is None:
        return None
    return register_artefact(db, store, pcm, media_type="application/octet-stream", role="stage_output",
                             created_by=SCHEMA, analysis_id=analysis_id)


def build(data, source, mapping, asset):
    from app.evidence.audio_packets import _chart

    if len(data) != 4 * source["samples"]:
        raise ValueError("Stereo samples do not cover the complete recorded clock")
    if source_ranges(mapping, 0, source["samples"]) != source_ranges(source, 0, source["samples"]):
        raise ValueError("Stereo repetition source clocks differ")
    channels = np.frombuffer(data, dtype="<i2").reshape(-1, 2)
    rows = []
    for channel, values in zip(CHANNELS, channels.T, strict=True):
        pcm = values.astype("<f4") / np.float32(32768)
        binding = {**source, "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest()}
        measurement = audio_repetition.build(pcm, mapping, binding, asset)
        for pair in measurement["pairs"]:
            pair["id"] = channel[0].upper() + pair["id"]
            first = pair["first"]
            excerpt = pcm[first["start_sample"]:first["end_sample"]]
            pair["artifacts"]["waveform"] = asset(
                _chart(excerpt, [], start=0, end=len(excerpt), selected=None, scores=False,
                       title=f"Recorded {channel} channel waveform | Time within excerpt"),
                "image/png", width=960, height=280,
            )
        rows.append({"channel": channel, "measurement": measurement})
    record = {"schema": SCHEMA, "source_sha256": source["sha256"], "affects_overall_confidence": False,
              "stereo_pcm_sha256": hashlib.sha256(data).hexdigest(), "channels": rows}
    validate(record, source)
    return record


def build_from_store(store, digest, source, mapping, asset):
    store.verify(digest)
    return build(store.read_bytes(digest), source, mapping, asset)


def verify(record, data, source, mapping, read_asset):
    def checked(data, media_type, **metadata):
        digest = hashlib.sha256(data).hexdigest()
        if read_asset(digest) != data:
            raise ValueError("Stereo repetition media differs from the recorded channel")
        return {"sha256": digest, "media_type": media_type, **metadata}

    validate(record, source)
    published = deepcopy(record)
    for row in published["channels"]:
        for pair in row["measurement"]["pairs"]:
            for asset in pair["artifacts"].values():
                asset.pop("url", None)
    if build(data, source, mapping, checked) != published:
        raise ValueError("Stereo repetition differs from the recorded channels")


def verify_frozen(store, document, mapping, digest, read_asset):
    record = document.get(KEY)
    if (digest is None) != (record is None):
        raise ObjectIntegrityError("Stereo repetition availability differs from the frozen audio input")
    if digest is not None:
        if record["stereo_pcm_sha256"] != digest:
            raise ObjectIntegrityError("Stereo repetition commits to different recorded channels")
        store.verify(digest)
        try:
            verify(record, store.read_bytes(digest), document["source"], mapping, read_asset)
        except (KeyError, ValueError) as error:
            raise ObjectIntegrityError("Stereo repetition differs from its frozen channels") from error
