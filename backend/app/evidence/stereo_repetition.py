"""Source-locked repeat checks on recorded and derived stereo signals."""

from __future__ import annotations

import hashlib
import io
import wave
from copy import deepcopy

import numpy as np

from app.detectors.audio.decode import source_ranges, wav_bytes
from app.evidence import audio_channel_alignment_binding as alignment
from app.evidence import audio_channel_difference as difference
from app.evidence import audio_exact_repetition, audio_repetition
from app.evidence import audio_gain_repetition_binding as gain
from app.evidence.stereo_repetition_binding import (
    ALIGNED_KEY,
    ALIGNED_SCHEMA,
    ALIGNED_VERSION,
    CHANNELS,
    DIFFERENCE_KEY,
    DIFFERENCE_SCHEMA,
    DIFFERENCE_VERSION,
    KEY,
    SCHEMA,
    validate,
)
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


def _build_exact(values, source, mapping, asset, prefix, title):
    from app.evidence.audio_packets import _chart

    pcm = values.astype("<f4") / np.float32(32768)
    binding = {**source, "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest()}
    measurement = audio_repetition.build(pcm, mapping, binding, asset)
    for pair in measurement["pairs"]:
        pair["id"] = prefix + pair["id"]
        first = pair["first"]
        excerpt = pcm[first["start_sample"]:first["end_sample"]]
        pair["artifacts"]["waveform"] = asset(
            _chart(excerpt, [], start=0, end=len(excerpt), selected=None, scores=False,
                   title=f"{title} | Time within excerpt"),
            "image/png", width=960, height=280,
        )
    return measurement


def _stereo_wav(data):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((2, 2, audio_exact_repetition.SAMPLE_RATE, 0, "NONE", "not compressed"))
        audio.writeframes(data)
    return output.getvalue()


def _build_aligned(data, channels, source, mapping, asset, *, exact_found):
    from app.evidence import audio_channel_alignment
    from app.evidence.audio_packets import _chart

    status = {"schema": alignment.SCHEMA, "source_sha256": source["sha256"],
              "sample_rate": alignment.SAMPLE_RATE, "input_samples": len(channels), "projection": alignment.PROJECTION,
              "alignment": None, "pcm_sha256": None, "searched_interval": None, "pairs": []}
    if exact_found:
        return status | {"status": "skipped", "reason": "exact_channel_difference_found"}
    fitted, reason, spans = audio_channel_alignment.search(channels)
    status.update(status=alignment.STATUSES[reason], reason=reason, alignment=vars(fitted) if fitted else None)
    if fitted is None or status["status"] == "skipped":
        return status
    shift = fitted.right_offset_samples
    low, high = alignment.overlap(len(channels), shift)
    values = audio_channel_alignment.projection(channels, fitted)
    status.update(searched_interval={"start_sample": low, "end_sample": high},
                  pcm_sha256=hashlib.sha256(values.tobytes()).hexdigest())

    def span_record(start, end, payload):
        return {"start_sample": start, "end_sample": end, "source_timestamp_ranges": source_ranges(mapping, start, end),
                "pcm_s16le_sha256": hashlib.sha256(payload).hexdigest()}

    for index, span in enumerate(spans, 1):
        count = span.first[1] - span.first[0]
        pair = {"id": f"AR{index:02d}", "sample_count": count, "lag_samples": span.second[0] - span.first[0],
                "artifacts": {}}
        for side, (start, end) in (("first", span.first), ("second", span.second)):
            a, b = alignment.excerpt_range(start, end, shift)
            original = data[4 * a:4 * b]
            selected = values[start:end]
            pair[side] = span_record(start, end, selected.tobytes()) | {"recorded_excerpt": span_record(a, b, original)}
            pcm = selected.astype("<f4") / np.float32(32768)
            pair["artifacts"][f"{side}_recorded_clip"] = asset(
                _stereo_wav(original), "audio/wav", samples=b - a, sample_rate=alignment.SAMPLE_RATE, channels=2)
            pair["artifacts"][f"{side}_clip"] = asset(
                wav_bytes(pcm), "audio/wav", samples=count, sample_rate=alignment.SAMPLE_RATE, channels=1)
            pair["artifacts"][f"{side}_waveform"] = asset(
                _chart(pcm, [], start=0, end=count, selected=None, scores=False,
                       title=f"Sound after stereo alignment | {side.capitalize()} occurrence"),
                "image/png", width=960, height=280,
            )
        status["pairs"].append(pair)
    return status


def build(data, source, mapping, asset, *, gain_version=None, difference_version=None):
    if len(data) != 4 * source["samples"]:
        raise ValueError("Stereo samples do not cover the complete recorded clock")
    if source_ranges(mapping, 0, source["samples"]) != source_ranges(source, 0, source["samples"]):
        raise ValueError("Stereo repetition source clocks differ")
    if difference_version not in (None, DIFFERENCE_VERSION, ALIGNED_VERSION):
        raise ValueError("Unknown stereo repetition difference method")
    channels = np.frombuffer(data, dtype="<i2").reshape(-1, 2)
    rows = []
    for channel, values in zip(CHANNELS, channels.T, strict=True):
        measurement = _build_exact(values, source, mapping, asset, channel[0].upper(), f"Recorded {channel} channel waveform")
        row = {"channel": channel, "measurement": measurement}
        if gain_version == gain.GAIN_VERSION:
            row["gain_measurement"] = _build_gain(values, source, mapping, channel, measurement["pcm_sha256"], asset)
        rows.append(row)
    record = {"schema": SCHEMA, "source_sha256": source["sha256"], "affects_overall_confidence": False,
              "stereo_pcm_sha256": hashlib.sha256(data).hexdigest(), "channels": rows}
    if difference_version is not None:
        values = np.frombuffer(difference.project(data), dtype="<i2")
        measurement = _build_exact(values, source, mapping, asset, "D", "Channel difference waveform")
        for pair in measurement["pairs"]:
            for side in ("first", "second"):
                span = pair[side]
                original = _stereo_wav(data[4 * span["start_sample"]:4 * span["end_sample"]])
                pair["artifacts"][f"{side}_recorded_clip"] = asset(original, "audio/wav", samples=pair["sample_count"],
                                                                   sample_rate=audio_exact_repetition.SAMPLE_RATE, channels=2)
        record.update(schema=DIFFERENCE_SCHEMA, difference_projection=difference.PROJECTION)
        record[DIFFERENCE_KEY] = measurement
        if difference_version == ALIGNED_VERSION:
            record.update(schema=ALIGNED_SCHEMA)
            record[ALIGNED_KEY] = _build_aligned(data, channels, source, mapping, asset,
                                                 exact_found=bool(measurement["pairs"]))
    validate(record, source)
    return record


def build_from_store(store, digest, source, mapping, asset, *, gain_version=None, difference_version=None):
    store.verify(digest)
    return build(store.read_bytes(digest), source, mapping, asset, gain_version=gain_version,
                 difference_version=difference_version)


def verify(record, data, source, mapping, read_asset, *, gain_version=None, difference_version=None):
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
    if build(data, source, mapping, checked, gain_version=gain_version, difference_version=difference_version) != published:
        raise ValueError("Stereo repetition differs from the recorded channels")


def verify_frozen(store, document, mapping, digest, read_asset, *, gain_version=None, difference_version=None):
    record = document.get(KEY)
    if (digest is None) != (record is None):
        raise ObjectIntegrityError("Stereo repetition availability differs from the frozen audio input")
    if digest is not None:
        if record["stereo_pcm_sha256"] != digest:
            raise ObjectIntegrityError("Stereo repetition commits to different recorded channels")
        store.verify(digest)
        try:
            verify(record, store.read_bytes(digest), document["source"], mapping, read_asset, gain_version=gain_version,
                   difference_version=difference_version)
        except (KeyError, ValueError) as error:
            raise ObjectIntegrityError("Stereo repetition differs from its frozen channels") from error
