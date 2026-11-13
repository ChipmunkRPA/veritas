from base64 import b64encode
from copy import deepcopy
from hashlib import sha256
from math import inf, nextafter
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.canonical import canonical_json
from app.core.config import settings
from app.detectors.audio import background_voice_review as detector
from app.detectors.audio.decode import read_wav
from app.engine.boundary import Deadline
from app.engine.detector import Profile
from app.engine.stages import audio
from app.evidence import audio_background_review as evidence, audio_background_review_binding as binding, audio_packets


def source(pcm):
    return {"samples": len(pcm), "channels": 1, "sample_rate": 16_000, "source_channels": 1, "source_sample_rate": 16_000,
            "source_stream_index": 0, "source_container": "wav", "source_bit_depth": 16, "source_codec": "pcm_s16le",
            "timeline_time_base": "1/16000", "timeline": [{**binding.INTERVAL, "source_pts": 0}],
            "pcm_sha256": sha256(pcm.tobytes()).hexdigest()}


def primary(direction, pcm_sha256):
    if direction == "unassessed":
        return {"detector_id": detector.PRIMARY_ID, "status": "not_applicable", "reason": detector.NO_SPEECH}
    return {"detector_id": detector.PRIMARY_ID, "status": "ok",
            "unit_scores": [{"quality": binding.INTERVAL}],
            "details": {"audio_screening": {"scope_supported": True, "decision": direction},
                        "audio_input": {"pcm_sha256": pcm_sha256}}}


@pytest.mark.parametrize("override", [
    {"samples": 47_999}, {"samples": 48_001}, {"source_bit_depth": 24}, {"source_channels": 2},
    {"source_sample_rate": 48_000}, {"source_codec": "flac"}, {"audio_stream_indices": [0, 1]},
    {"timeline": [{**binding.INTERVAL, "source_pts": 1}]},
])
def test_declines_audio_outside_the_held_scope(override):
    mapping = source(np.zeros(48_000, dtype="<f4"))
    assert binding.supported_source(mapping)
    assert not binding.supported_source(mapping | override)


def test_scope_preserves_existing_direction_and_checks_provenance(monkeypatch):
    monkeypatch.setattr(detector, "configured", lambda: True)
    monkeypatch.setattr(settings, "audio_detector_mode", "candidate")
    ctx = SimpleNamespace(profile=Profile.FULL, deadline=Deadline(1), state={},
                          outputs={"decode": {"background_voice_review_source_supported": True}})
    assert audio._background_voice_review(ctx, None, [])["reason"] == binding.SKIPPED_UNAVAILABLE
    held = {"run_id": "saved", **primary("original", "a" * 64)}
    assert audio._background_voice_review(ctx, None, [held])["reason"] == binding.SKIPPED_DIRECTION
    admitted = {"run_id": "saved", **primary("unassessed", "")}
    assert audio._background_voice_review(ctx, None, [admitted])["reason"] == binding.SKIPPED_BUDGET
    monkeypatch.setattr(audio.provenance_gate, "stop", lambda _: True)
    assert audio._background_voice_review(ctx, None, [admitted]) is None
    ctx.profile = Profile.QUICK
    assert audio._background_voice_review(ctx, None, [admitted]) is None


@pytest.mark.parametrize(("direction", "admitted", "response", "expected"), [
    ("inconclusive", True, nextafter(detector.OPERATING_POINT, inf), "flagged"),
    ("unassessed", True, nextafter(detector.OPERATING_POINT, inf), "flagged"),
    ("inconclusive", True, detector.OPERATING_POINT, "no_flag"),
    ("unassessed", False, None, "unassessed"),
])
def test_replays_the_frozen_direction_and_processed_interval(direction, admitted, response, expected):
    pcm = np.zeros(48_000, dtype="<f4")
    mapping = source(pcm)
    processed = np.arange(len(pcm), dtype="<i2").tobytes()
    private = {"source_pcm_sha256": mapping["pcm_sha256"], "primary_direction": direction, "status": expected,
               "processed_pcm_sha256": sha256(processed).hexdigest(), "admitted": admitted,
               "response_hex": None if response is None else response.hex()}
    if expected == "flagged":
        private["processed_pcm"] = b64encode(processed).decode("ascii")
    entry = {"detector_id": binding.DETECTOR_ID, "status": "ok", "details": private}
    objects = {}

    def store_receipt(record):
        data = canonical_json(record)
        pin = sha256(data).hexdigest()
        objects[pin] = data
        return pin

    record = {"raw_output_sha256": store_receipt(entry),
              "primary_raw_output_sha256": store_receipt(primary(direction, mapping["pcm_sha256"]))}
    store = SimpleNamespace(read_bytes=objects.__getitem__)

    def asset(data, media_type, **metadata):
        pin = sha256(data).hexdigest()
        objects[pin] = data
        return {"sha256": pin, "media_type": media_type, **metadata}

    args = {"store": store, "source": mapping, "pcm": pcm, "mapping": mapping, "waveform": audio_packets._chart}
    review = evidence.build(record, **args, asset=asset)
    assert review["status"] == expected and review["affects_overall_confidence"] is False
    assert review["source_timestamp_ranges"] == binding.TIMESTAMPS
    evidence.verify(review, record, **args, read_asset=objects.__getitem__)
    if expected != "flagged":
        assert review["artifacts"] == {}
        return
    original = review["artifacts"]["selected_clip"]["sha256"]
    rendered = review["artifacts"]["processed_clip"]["sha256"]
    np.testing.assert_array_equal(read_wav(objects[original]), pcm)
    assert (read_wav(objects[rendered]) * 32768.0).astype("<i2").tobytes() == processed
    changed = deepcopy(review)
    changed["interval"]["end_sample"] -= 1
    with pytest.raises(ValueError, match="native source interval"):
        binding.validate(changed, mapping)
    objects[rendered] = objects[rendered][:-1] + bytes([objects[rendered][-1] ^ 1])
    with pytest.raises(ValueError, match="frozen input"):
        evidence.verify(review, record, **args, read_asset=objects.__getitem__)
    wrong_direction = record | {"primary_raw_output_sha256": store_receipt(primary("original", mapping["pcm_sha256"]))}
    with pytest.raises(ValueError, match="primary response"):
        evidence.build(wrong_direction, **args, asset=asset)
