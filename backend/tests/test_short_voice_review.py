from contextlib import nullcontext
from copy import deepcopy
from hashlib import sha256
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.canonical import canonical_json
from app.core.config import settings
from app.detectors.audio import short_voice_review
from app.detectors.audio.decode import read_wav
from app.detectors.video.adapters.common import AudioTrack
from app.engine import boundary
from app.engine.boundary import Cancelled, Deadline
from app.engine.detector import DetectorResult, Profile
from app.engine.stages import audio
from app.evidence import audio_packets, audio_short_review
from app.evidence import audio_short_review_binding as binding


def source(pcm):
    return {
        "samples": len(pcm), "sample_rate": 16000, "source_sample_rate": 16000, "channels": 1,
        "source_channels": 1, "source_codec": "pcm_s16le", "pcm_sha256": sha256(pcm.astype("<f4").tobytes()).hexdigest(),
        "timeline_time_base": "1/16000", "timeline": [{"start_sample": 0, "end_sample": len(pcm), "source_pts": 0}],
    }


def test_scope_excludes_lossy_stereo_multitrack_and_clock_changes():
    mapping = source(np.zeros(binding.SAMPLES, np.float32))
    assert binding.supported_source(mapping)
    assert not binding.supported_source(mapping | {"source_codec": "opus"})
    assert not binding.supported_source(mapping | {"source_channels": 2})
    assert not binding.supported_source(mapping | {"audio_stream_indices": [0, 1]})
    assert not binding.supported_source(mapping | {"timeline": [dict(mapping["timeline"][0], source_pts=1)]})


@pytest.mark.parametrize("optional", [DetectorResult.timeout("slow"), DetectorResult.failed("invalid")])
def test_optional_boundary_failure_does_not_change_completed_primary(monkeypatch, optional):
    pcm = np.zeros(binding.SAMPLES, np.float32)
    original = {
        "detectors": [{"detector_id": short_voice_review.PellaVoiceDetector.spec.id, "status": "ok", "unit_scores": [0.8, 0.7],
                       "details": {"audio_screening": {"decision": "inconclusive", "probability_ai": None}}}],
        "exhibits": ["primary-waveform"],
    }
    calls = []

    def primary(*args, **kwargs):
        calls.append("primary")
        return deepcopy(original)

    def separate(detector, ctx, **kwargs):
        calls.append("optional")
        assert detector.spec.id == binding.DETECTOR_ID and kwargs["timeout_s"] == short_voice_review.BUDGET_S
        return optional

    monkeypatch.setattr(settings, "audio_detector_mode", "candidate")
    monkeypatch.setattr(audio.common, "run_detectors", primary)
    monkeypatch.setattr(audio.common, "media", lambda ctx: None)
    monkeypatch.setattr(audio, "run_detector", separate)
    ctx = SimpleNamespace(
        profile=Profile.FULL, analysis_id="test", deadline=Deadline(100), check=lambda *_: None,
        outputs={"probe": {"facts": {"container": "wav"}}, "decode": {"short_voice_review_source_supported": True},
                 "condition": {"content": {"duration_s": 6, "speech_spans": [[0, 6]]}}},
        state={"audio": AudioTrack("analysed", pcm, len(pcm)), "run_store": object()},
        exhibits=SimpleNamespace(bind=lambda *_: None), log=SimpleNamespace(bind=lambda **_: None),
    )
    result = audio.detect(ctx)
    assert calls == ["primary", "optional"]
    assert result["detectors"] == original["detectors"] and result["exhibits"] == original["exhibits"]
    assert result["short_voice_review"]["status"] == "skipped"
    assert result["short_voice_review"]["selected"] is None
    ctx.deadline = Deadline(5)
    assert audio.detect(ctx)["short_voice_review"]["status"] == "skipped"
    assert calls == ["primary", "optional", "primary"]
    ctx.deadline = Deadline(100)

    def cancelled(*_):
        raise Cancelled("job cancelled")

    ctx.check = cancelled
    with pytest.raises(Cancelled):
        audio.detect(ctx)


def test_flagged_packet_anchors_exact_slice_without_public_raw_score():
    pcm = np.arange(binding.SAMPLES, dtype=np.int32).astype("<i2").astype(np.float32) / 32768
    mapping = source(pcm)
    p = iter([.1, .2, .3, .4, .9999])
    model = SimpleNamespace(predict=lambda chunks: [next(p)])
    review = short_voice_review.scan(pcm, [(0, 6)], model, Deadline(None))
    raw = canonical_json({"detector_id": binding.DETECTOR_ID, "status": "ok",
                          "details": {"pcm_sha256": mapping["pcm_sha256"], "short_voice_review": review}})
    pin = sha256(raw).hexdigest()
    assets = {}

    def asset(data, media_type, **meta):
        digest = sha256(data).hexdigest()
        assets[digest] = data
        return {"sha256": digest, "media_type": media_type, **meta}

    record = {"raw_output_sha256": pin}
    store = SimpleNamespace(read_bytes={pin: raw}.__getitem__)
    result = audio_short_review.build(record, store=store, source=mapping, pcm=pcm, mapping=mapping,
                                      asset=asset, waveform=audio_packets._chart)
    assert result["status"] == "flagged" and result["affects_overall_confidence"] is False
    assert result["interval"] == {"start_sample": 64000, "end_sample": 96000}
    assert result["start_s"] == 4.0 and result["end_s"] == 6.0
    assert b"logit" not in canonical_json(result)
    np.testing.assert_array_equal(read_wav(assets[result["artifacts"]["selected_clip"]["sha256"]]), pcm[64000:96000])
    audio_short_review.verify(result, record, store=store, source=mapping, pcm=pcm, mapping=mapping,
                              read_asset=assets.__getitem__, waveform=audio_packets._chart)
    changed = deepcopy(result)
    changed["source_timestamp_ranges"][0]["source_end_pts"] -= 1
    with pytest.raises(ValueError):
        binding.validate(changed, mapping)


def test_watchdog_discards_slow_optional_run_without_losing_primary(monkeypatch):
    pcm = np.zeros(binding.SAMPLES, np.float32)
    primary = [{"detector_id": short_voice_review.PellaVoiceDetector.spec.id, "status": "ok", "unit_scores": [0.2, 0.8],
                "details": {"audio_screening": {"decision": "inconclusive", "probability_ai": None}}}]
    release, entered = Event(), Event()
    records = []

    def blocked(self, ctx):
        entered.set()
        release.wait(5)
        return DetectorResult(status="ok")

    monkeypatch.setattr(settings, "audio_detector_mode", "candidate")
    monkeypatch.setattr(audio.common, "media", lambda ctx: None)
    monkeypatch.setattr(audio.common, "run_detectors", lambda *_args, **_kwargs: {"detectors": primary})
    monkeypatch.setattr(short_voice_review.ShortVoiceReview, "run", blocked)
    monkeypatch.setattr(short_voice_review, "BUDGET_S", 0.05)
    monkeypatch.setattr(boundary, "WATCHDOG_GRACE_S", 0.1)
    models = SimpleNamespace(activation_bytes_for_items=lambda _: 0,
                             lease=lambda *args, **kwargs: nullcontext(SimpleNamespace(device="cpu", dtype="float32")))
    deadline = Deadline(60)
    ctx = SimpleNamespace(
        profile=Profile.FULL, analysis_id="test", deadline=deadline, check=deadline.check,
        outputs={"probe": {"facts": {"container": "wav"}}, "decode": {"short_voice_review_source_supported": True},
                 "condition": {"content": {"duration_s": 6, "speech_spans": [[0, 6]]}}},
        state={"audio": AudioTrack("analysed", pcm, len(pcm)), "run_store": SimpleNamespace(persist=records.append),
               "models": models},
        exhibits=SimpleNamespace(bind=lambda *_: None), log=SimpleNamespace(bind=lambda **_: None),
    )
    try:
        output = audio.detect(ctx)
    finally:
        release.set()
    assert entered.is_set()
    assert output["detectors"] == primary
    assert output["short_voice_review"]["status"] == "skipped"
    assert [record.status for record in records] == ["timeout"]
    deadline.check("primary analysis can continue")
