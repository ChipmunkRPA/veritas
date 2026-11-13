"""Source-locked exact repeat checks on individual stereo channels."""

from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np

from app.detectors.audio.decode import source_ranges, wav_bytes
from app.evidence import audio_exact_repetition, audio_repetition
from app.evidence import audio_gain_repetition_binding as gain
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


def _build_gain(values, source, mapping, channel, pcm_sha256, asset):
    from app.evidence.audio_gain_repetition import find_gain_repeats
    from app.evidence.audio_packets import _chart

    payload = values.astype("<i2", copy=False).tobytes()
    result = find_gain_repeats(payload)
    pairs = []
    for index, span in enumerate(result.spans, 1):
        pair = {"id": f"{channel[0].upper()}G{index:02d}", "sample_count": span.sample_count,
                "lag_samples": span.second[0] - span.first[0], "gain_numerator": span.numerator,
                "gain_denominator": span.denominator, "scaled_occurrence": span.scaled_occurrence, "artifacts": {}}
        for side, (start, end) in (("first", span.first), ("second", span.second)):
            pair[side] = {"start_sample": start, "end_sample": end,
                          "source_timestamp_ranges": source_ranges(mapping, start, end),
                          "pcm_s16le_sha256": hashlib.sha256(payload[2 * start:2 * end]).hexdigest()}
            excerpt = values[start:end].astype("<f4") / np.float32(32768)
            pair["artifacts"][f"{side}_clip"] = asset(wav_bytes(excerpt), "audio/wav", samples=len(excerpt),
                                                      sample_rate=audio_exact_repetition.SAMPLE_RATE, channels=1)
            pair["artifacts"][f"{side}_waveform"] = asset(
                _chart(excerpt, [], start=0, end=len(excerpt), selected=None, scores=False,
                       title=f"Recorded {channel} channel waveform | {side.capitalize()} occurrence"),
                "image/png", width=960, height=280,
            )
        pairs.append(pair)
    return {"schema": gain.SCHEMA, "source_sha256": source["sha256"], "pcm_sha256": pcm_sha256,
            "sample_rate": audio_exact_repetition.SAMPLE_RATE, "input_samples": len(values),
            "status": result.status, "reason": result.reason, "pairs": pairs}


def build(data, source, mapping, asset, *, gain_version=None):
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
        row = {"channel": channel, "measurement": measurement}
        if gain_version == gain.GAIN_VERSION:
            row["gain_measurement"] = _build_gain(values, source, mapping, channel, binding["pcm_sha256"], asset)
        rows.append(row)
    record = {"schema": SCHEMA, "source_sha256": source["sha256"], "affects_overall_confidence": False,
              "stereo_pcm_sha256": hashlib.sha256(data).hexdigest(), "channels": rows}
    validate(record, source)
    return record


def build_from_store(store, digest, source, mapping, asset, *, gain_version=None):
    store.verify(digest)
    return build(store.read_bytes(digest), source, mapping, asset, gain_version=gain_version)


def verify(record, data, source, mapping, read_asset, *, gain_version=None):
    def checked(data, media_type, **metadata):
        digest = hashlib.sha256(data).hexdigest()
        if read_asset(digest) != data:
            raise ValueError("Stereo repetition media differs from the recorded channel")
        return {"sha256": digest, "media_type": media_type, **metadata}

    validate(record, source)
    published = deepcopy(record)
    from app.evidence.stereo_repetition_binding import pairs

    for pair in pairs(published):
        for asset in pair["artifacts"].values():
            asset.pop("url", None)
    if build(data, source, mapping, checked, gain_version=gain_version) != published:
        raise ValueError("Stereo repetition differs from the recorded channels")


def verify_frozen(store, document, mapping, digest, read_asset, *, gain_version=None):
    record = document.get(KEY)
    if (digest is None) != (record is None):
        raise ObjectIntegrityError("Stereo repetition availability differs from the frozen audio input")
    if digest is not None:
        if record["stereo_pcm_sha256"] != digest:
            raise ObjectIntegrityError("Stereo repetition commits to different recorded channels")
        store.verify(digest)
        try:
            verify(record, store.read_bytes(digest), document["source"], mapping, read_asset, gain_version=gain_version)
        except (KeyError, ValueError) as error:
            raise ObjectIntegrityError("Stereo repetition differs from its frozen channels") from error
