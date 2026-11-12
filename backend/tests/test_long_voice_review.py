import shutil
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.canonical import canonical_json
from app.core.config import settings
from app.detectors.audio import long_voice_review
from app.detectors.audio.decode import decode_audio, read_wav, wav_bytes
from app.engine import runs
from app.engine.boundary import Deadline
from app.engine.detector import Profile
from app.engine.stages import audio
from app.evidence import audio_long_review, audio_packets
from app.evidence import audio_long_review_binding as binding
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, AnalysisStage, Base, DetectorRun
from app.storage.objects import ObjectStore


def native_source(samples):
    return {"samples": samples, "channels": 1, "sample_rate": binding.RATE, "source_channels": 1,
            "source_sample_rate": binding.RATE, "source_bit_depth": 16, "source_container": "flac", "source_codec": "flac",
            "timeline_time_base": "1/16000", "timeline": [{"start_sample": 0, "end_sample": samples, "source_pts": 0}]}


@pytest.mark.parametrize(("samples", "starts"), [
    (95_999, []), (96_000, [0, 48_000]), (143_999, [0, 95_999]), (144_000, [0, 48_000, 96_000]),
    (160_001, [0, 56_000, 112_001]), (224_000, [0, 88_000, 176_000]), (224_001, []),
])
def test_exact_disjoint_geometry(samples, starts):
    assert binding.intervals(samples) == [(start, start + 48_000) for start in starts]
    assert binding.supported_source(native_source(samples)) == bool(starts)


@pytest.mark.parametrize("change", [
    {"source_container": "ogg"}, {"source_codec": "mp3"}, {"source_bit_depth": 24},
    {"source_channels": 2}, {"source_sample_rate": 48_000}, {"audio_stream_indices": [0, 1]},
    {"timeline": [{"start_sample": 0, "end_sample": 144_000, "source_pts": 100}]},
    {"timeline": [{"start_sample": 0, "end_sample": 70_000, "source_pts": 0},
                  {"start_sample": 70_000, "end_sample": 144_000, "source_pts": 71_000}]},
])
def test_declines_other_native_profiles(change):
    assert not binding.supported_source(native_source(144_000) | change)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Native audio decoder required")
def test_native_pcm_wav_has_the_same_long_window_scope(tmp_path):
    pcm = np.arange(144_001, dtype=np.int32).astype("<i2").astype("<f4") / 32768
    body = wav_bytes(pcm)
    path = tmp_path / "source"
    path.write_bytes(body)
    decoded = decode_audio(path, source_sha256=sha256(body).hexdigest())
    np.testing.assert_array_equal(decoded.pcm, pcm)
    assert binding.supported_source(decoded.mapping)
    assert binding.intervals(len(decoded.pcm)) == [(0, 48_000), (48_000, 96_000), (96_001, 144_001)]
    assert not binding.supported_source(decoded.mapping | {"source_container": "matroska"})
    assert not binding.supported_source(decoded.mapping | {"source_codec": "pcm_s24le"})


@pytest.mark.parametrize(("directions", "status"), [
    (["ai", "unassessed", "ai"], "corroborated"),
    (["ai", "inconclusive", "ai"], "corroborated"),
    (["ai", "original", "ai"], "no_corroboration"),
    (["unassessed", "inconclusive", "ai"], "no_corroboration"),
])
def test_exact_waveform_audio_and_clock_replay(directions, status):
    pcm = np.arange(144_001, dtype=np.int32).astype("<i2").astype("<f4") / 32768
    source = native_source(len(pcm)) | {"pcm_sha256": sha256(pcm.tobytes()).hexdigest()}
    windows = [{"start_sample": a, "end_sample": b, "direction": d, "pcm_sha256": sha256(pcm[a:b].tobytes()).hexdigest()}
               for (a, b), d in zip(binding.intervals(len(pcm)), directions, strict=True)]
    receipt = {"schema": binding.SCHEMA, "status": status, "reason": binding.REASONS[status], "windows": windows}
    raw = canonical_json({"detector_id": binding.DETECTOR_ID, "status": "ok",
                          "details": {"pcm_sha256": source["pcm_sha256"], "long_voice_review": receipt}})
    pin = sha256(raw).hexdigest()
    store = SimpleNamespace(read_bytes={pin: raw}.__getitem__)
    record = {"raw_output_sha256": pin}
    assets = {}

    def asset(data, media_type, **metadata):
        digest = sha256(data).hexdigest()
        assets[digest] = data
        return {"sha256": digest, "media_type": media_type, **metadata}

    review = audio_long_review.build(record, store=store, source=source, pcm=pcm, mapping=source,
                                    asset=asset, waveform=audio_packets._chart)
    assert review["status"] == status and review["affects_overall_confidence"] is False
    for window in review["windows"]:
        a, b = window["start_sample"], window["end_sample"]
        assert window["source_timestamp_ranges"][0]["source_start_pts"] == a
        assert window["source_timestamp_ranges"][0]["source_end_pts"] == b
        if status == "corroborated" and window["direction"] == "ai":
            clip = window["artifacts"]["selected_clip"]
            np.testing.assert_array_equal(read_wav(assets[clip["sha256"]]), pcm[a:b])
        else:
            assert window["artifacts"] == {}
    audio_long_review.verify(review, record, store=store, source=source, pcm=pcm, mapping=source,
                             read_asset=assets.__getitem__, waveform=audio_packets._chart)
    audio_long_review.verify_source(review, pcm=pcm, mapping=source, waveform=audio_packets._chart)
    moved = deepcopy(review)
    moved["windows"][0]["source_timestamp_ranges"][0]["source_end_pts"] -= 1
    with pytest.raises(ValueError, match="clock"):
        audio_long_review.verify_source(moved, pcm=pcm, mapping=source, waveform=audio_packets._chart)
    changed_pcm = pcm.copy()
    changed_pcm[0] += 1 / 32768
    with pytest.raises(ValueError, match="PCM differs"):
        audio_long_review.verify_source(review, pcm=changed_pcm, mapping=source, waveform=audio_packets._chart)


def test_auxiliary_run_only_copies_the_detect_stage_pin(tmp_path):
    engine = create_engine("sqlite://")
    store = ObjectStore(tmp_path)
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(Analysis(id="input", kind="audio", filename="sample.flac", content_type="audio/flac",
                        size_bytes=1, sha256="a" * 64))
        db.add_all([
            DetectorRun(id=run_id, analysis_id="input", detector_id=detector, detector_version="1", status="ok")
            for run_id, detector in (("primary", "audio-test"), ("attempt-a", binding.DETECTOR_ID))
        ])
        db.flush()
        assert [row.id for row in runs.current(db, "input", store)] == ["primary"]
        assert [row.id for row in runs.current(db, "input", store, include_auxiliary=True)] == ["primary"]
        db.add(DetectorRun(id="attempt-z", analysis_id="input", detector_id=binding.DETECTOR_ID, detector_version="1", status="ok"))
        output = register_artefact(db, store, canonical_json({"detectors": [{"run_id": "primary"}],
                                    "long_voice_review": {"run_id": "attempt-a"}}),
                                    media_type="application/json", role="stage_output", created_by="detect", analysis_id="input")
        db.add(AnalysisStage(analysis_id="input", stage="detect", status="done", output_artefact_id=output.id))
        db.flush()
        assert runs.statuses(db, "input", store=store) == ["ok"]
        assert {row.id for row in runs.current(db, "input", store, include_auxiliary=True)} == {"primary", "attempt-a"}


def test_optional_review_respects_the_serving_model_and_provenance_gate(monkeypatch):
    monkeypatch.setattr(settings, "audio_detector_mode", "candidate")
    ctx = SimpleNamespace(profile=Profile.FULL, deadline=Deadline(5), state={},
                          outputs={"decode": {"long_voice_review_source_supported": True}})
    primary = [{"detector_id": long_voice_review.PellaVoiceDetector.spec.id, "status": "not_applicable"}]
    assert audio._long_voice_review(ctx, None, [])["reason"] == binding.SKIPPED_UNAVAILABLE
    assert audio._long_voice_review(ctx, None, primary)["reason"] == binding.SKIPPED_BUDGET
    monkeypatch.setattr(audio.provenance_gate, "stop", lambda _: True)
    assert audio._long_voice_review(ctx, None, primary) is None
