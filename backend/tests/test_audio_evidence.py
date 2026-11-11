from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import wave
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.audio.decode import SAMPLE_RATE, decode_audio, read_wav, source_ranges, wav_bytes
from app.detectors.video.adapters.audio import AudioClassifier, GustkingVoiceDetector
from app.detectors.video.adapters.common import AudioTrack
from app.engine.boundary import DetectorRunRecord
from app.engine.detector import DetectorResult, Profile, UnitScore
from app.engine.stages.common import DbRunStore
from app.evidence.audio_binding import source_clock_context, unscored_voice_intervals
from app.evidence.audio_packets import (
    PacketIntegrityError,
    build_packet_set,
    load_packet_set,
    packet_hash,
    persist_packet_set,
    validate_packet_set,
)
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Artefact, Base, DetectorRun
from app.storage.objects import ObjectStore


def _write_source(path, *, rate=48000, channels=2, seconds=8, silence=False):
    time = np.arange(rate * seconds) / rate
    samples = np.zeros_like(time) if silence else np.sin(2 * np.pi * 229 * time) * 0.4
    samples = (samples * 32768).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((np.column_stack([samples, samples // 2]) if channels == 2 else samples).tobytes())


@pytest.mark.parametrize("rate,channels", [(48000, 2), (44100, 1), (16000, 1)])
def test_decode_freezes_exact_input_and_full_sample_mapping(tmp_path, rate, channels):
    source = tmp_path / "sample.wav"
    _write_source(source, rate=rate, channels=channels, seconds=2)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    decoded = decode_audio(source, source_sha256=sha)
    assert len(decoded.pcm) == 2 * SAMPLE_RATE
    np.testing.assert_array_equal(read_wav(decoded.wav), decoded.pcm)
    assert decoded.mapping["source_sample_rate"] == rate
    assert decoded.mapping["source_channels"] == channels
    assert decoded.mapping["pcm_sha256"] == hashlib.sha256(decoded.pcm.astype("<f4").tobytes()).hexdigest()
    ranges = source_ranges(decoded.mapping, 0, len(decoded.pcm))
    assert ranges == [{"start_sample": 0, "end_sample": 32000, "source_start_pts": 0,
                       "source_end_pts": 32000, "time_base": "1/16000"}]


def test_nonzero_stream_start_is_preserved_before_wav_normalisation(tmp_path):
    source, shifted = tmp_path / "source.wav", tmp_path / "shifted.mka"
    _write_source(source, rate=16000, channels=1, seconds=2)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-c:a", "pcm_s16le", "-output_ts_offset", "2", str(shifted)],
                   check=True, capture_output=True)
    decoded = decode_audio(shifted, source_sha256=hashlib.sha256(shifted.read_bytes()).hexdigest())
    assert len(decoded.pcm) == 32000
    assert decoded.mapping["timeline"][0]["source_pts"] == 32000
    assert source_ranges(decoded.mapping, 8000, 24000)[0]["source_start_pts"] == 40000
    assert decoded.mapping["container_start_s"] == 2


def test_source_identity_and_first_stream_selection(tmp_path):
    silent, tone, dual = tmp_path / "silent.wav", tmp_path / "tone.wav", tmp_path / "dual.mka"
    _write_source(silent, silence=True, channels=1, seconds=2)
    _write_source(tone, channels=2, seconds=2)
    with pytest.raises(ValueError, match="recorded original"):
        decode_audio(silent, source_sha256="0" * 64)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(silent), "-i", str(tone), "-map", "0:a", "-map", "1:a",
                    "-c:a", "pcm_s16le", str(dual)], check=True, capture_output=True)
    decoded = decode_audio(dual, source_sha256=hashlib.sha256(dual.read_bytes()).hexdigest())
    assert decoded.mapping["source_stream_index"] == 0
    assert decoded.mapping["source_channels"] == 1
    assert not np.any(decoded.pcm)  # Automatic best-stream selection would choose the stereo tone.


def test_timestamp_discontinuity_remains_two_ranges():
    mapping = {"samples": 32000, "timeline_time_base": "1/16000", "timeline": [
        {"start_sample": 0, "end_sample": 16000, "source_pts": 0},
        {"start_sample": 16000, "end_sample": 32000, "source_pts": 48000}]}
    intervals = source_ranges(mapping, 8000, 24000)
    assert intervals == [
        {"start_sample": 8000, "end_sample": 16000, "source_start_pts": 8000, "source_end_pts": 16000, "time_base": "1/16000"},
        {"start_sample": 16000, "end_sample": 24000, "source_start_pts": 48000, "source_end_pts": 56000, "time_base": "1/16000"}]
    coverage = {"windows": [{"start_sample": 24000, "end_sample": 32000}, {"start_sample": 0, "end_sample": 8000},
                            {"start_sample": 2000, "end_sample": 5000}],
                "scored_intervals": [{"start_sample": 0, "end_sample": 8000}, {"start_sample": 24000, "end_sample": 32000}],
                "decoded_samples": 32000, "scored_samples": 16000, "unscored_samples": 16000}
    assert unscored_voice_intervals(coverage, {**mapping, "sample_rate": 16000}) == [
        {"start_sample": 8000, "end_sample": 24000, "source_timestamp_ranges": intervals}]
    mapping["timeline"][1]["start_sample"] = 15000
    with pytest.raises(ValueError, match="partition"):
        source_ranges(mapping, 0, 32000)


@pytest.fixture
def audio_fixture(tmp_path, request):
    source_path = tmp_path / "source.wav"
    if getattr(request, "param", None) == "clock_gap":
        _write_source(source_path, rate=16000, channels=1)
        shifted = tmp_path / "clock-gap.mka"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source_path),
                        "-af", r"asetnsamples=n=1600:p=0,asetpts=PTS+if(gte(N\,64000)\,16000\,0)",
                        "-c:a", "pcm_s16le", str(shifted)], check=True, capture_output=True, timeout=20)
        source_path = shifted
    else:
        _write_source(source_path)
    store = ObjectStore(tmp_path / "objects")
    source = store.put_file(source_path)
    decoded = decode_audio(source_path, source_sha256=source.sha256)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        content_type = "audio/x-matroska" if source_path.suffix == ".mka" else "audio/wav"
        analysis = Analysis(id="audio-evidence-test", kind="audio", filename=source_path.name, content_type=content_type,
                            size_bytes=source.size_bytes, sha256=source.sha256, status="done", revision=1)
        db.add(analysis)
        db.flush()
        wav = register_artefact(db, store, decoded.wav, media_type="audio/wav", role="working_copy", created_by="audio-input/1", analysis_id=analysis.id)
        mapping = register_artefact(db, store, canonical_json(decoded.mapping), media_type="application/json", role="stage_output", created_by="audio-input/1", analysis_id=analysis.id)
        input_ = {"version": "audio-input/1", "pcm_sha256": decoded.mapping["pcm_sha256"], "sample_rate": SAMPLE_RATE, "samples": len(decoded.pcm)}
        units = [{"unit_ref": f"window:{i}", "raw": score, "raw_kind": "probability",
                  "quality": {"start_sample": a * SAMPLE_RATE, "end_sample": b * SAMPLE_RATE, "start_s": a, "end_s": b}}
                 for i, (a, b, score) in enumerate(((0, 4, 0.2), (2, 6, 0.9), (4, 8, 0.3)))]
        entry = {"detector_id": "aud.gustking_xlsr", "status": "ok", "family": "voice", "exhibit_ids": ["ex_audio"],
                 "details": {"audio_input": input_}, "unit_scores": units}
        raw = register_artefact(db, store, canonical_json(entry), media_type="application/json", role="raw_output", created_by="voice-check", analysis_id=analysis.id)
        binding = {**input_, "wav_sha256": wav.sha256, "mapping_sha256": mapping.sha256}
        sid = "aud.gustking_xlsr:global"
        result = {"input": {"sha256": source.sha256}, "manifest": {"sampling": {"audio_evidence": {
            "input": binding, "checks": [{"detector_id": entry["detector_id"], "raw_output_sha256": raw.sha256}]}}},
            "signals": [{"id": sid, "detector_id": entry["detector_id"], "status": "ok", "display_id": "S01", "exhibit_ids": ["ex_audio"]}],
            "findings": [{"id": "F1", "evidence_ids": [sid]}]}
        analysis.result_hash = canonical_hash(result)
        store.put_bytes(canonical_json(result))
        yield db, store, analysis, result, decoded.pcm, entry


def _replace_entry(store, analysis, result, entry):
    result["manifest"]["sampling"]["audio_evidence"]["checks"][0]["raw_output_sha256"] = store.put_bytes(canonical_json(entry)).sha256
    analysis.result_hash = canonical_hash(result)


def test_packet_clips_equal_the_complete_scored_samples_and_context(audio_fixture):
    db, store, analysis, result, pcm, _ = audio_fixture
    before = deepcopy(result)
    doc = build_packet_set(db, analysis, store, result)
    assert doc["sha256"] == packet_hash(doc)
    assert doc["coverage"]["scored_samples"] == len(pcm)  # overlapping windows are counted only once
    assert doc["coverage"]["unscored_samples"] == 0
    assert doc["coverage"]["unscored_intervals"] == []
    assert doc["coverage"]["source_clock_context"] == {
        "affects_overall_confidence": False, "time_base": "1/16000", "minimum_jump_samples": 320,
        "checked_samples": len(pcm), "total_discontinuities": 0, "scored_windows_crossing": 0, "breaks": [],
    }
    assert len(doc["packets"]) == 1
    packet = doc["packets"][0]
    assert packet["interval"] == {"start_sample": 32000, "end_sample": 96000}
    assert packet["context_interval"] == {"start_sample": 0, "end_sample": 128000}
    assert packet["finding_ids"] == ["F1"]
    assert "gustking" not in json.dumps(doc)
    assert "boundary" in packet["interpretation"]
    for name, start, end in (("selected_clip", 32000, 96000), ("context_clip", 0, 128000)):
        np.testing.assert_array_equal(read_wav(store.read_bytes(packet["artifacts"][name]["sha256"])), pcm[start:end])
    for name in ("waveform", "scores"):
        assert Image.open(io.BytesIO(store.read_bytes(packet["artifacts"][name]["sha256"]))).size == (960, 280)
    assert result == before and analysis.result_hash == canonical_hash(before)


def test_audio_clock_context_keeps_large_backwards_and_forward_breaks_with_one_window_count():
    source = {"samples": 32000, "timeline_time_base": "1/16000", "timeline": [
        {"start_sample": 0, "end_sample": 8000, "source_pts": 0},
        {"start_sample": 8000, "end_sample": 16000, "source_pts": 8016},
        {"start_sample": 16000, "end_sample": 24000, "source_pts": 15200},
        {"start_sample": 24000, "end_sample": 32000, "source_pts": 26200},
    ]}
    coverage = {"windows": [{"start_sample": 0, "end_sample": 32000}, {"start_sample": 16000, "end_sample": 24000}]}
    measured = source_clock_context(coverage, source)
    assert measured["total_discontinuities"] == 2 and measured["scored_windows_crossing"] == 1
    assert [(row["boundary_sample"], row["jump_samples"], row["scored_windows_crossing"]) for row in measured["breaks"]] == [
        (16000, -816, 1), (24000, 3000, 1),
    ]


@pytest.mark.parametrize("audio_fixture", ["clock_gap"], indirect=True)
def test_source_clock_context_matches_a_real_gap_in_json_and_pdf(audio_fixture, tmp_path, monkeypatch):
    from test_video_nested_evidence import _check_public_zip

    from app.reports import engine, multimedia_evidence
    from app.reports.specimens import specimen

    db, store, analysis, result, pcm, _ = audio_fixture
    original = specimen("audio", "standard")
    result = {**original.result, **result}
    analysis.result_hash = store.put_bytes(canonical_json(result)).sha256
    document = build_packet_set(db, analysis, store, result)
    measured = document["coverage"]["source_clock_context"]
    assert measured == {"affects_overall_confidence": False, "time_base": "1/16000", "minimum_jump_samples": 320,
                        "checked_samples": len(pcm), "total_discontinuities": 1, "scored_windows_crossing": 1,
                        "breaks": [{"boundary_sample": 64000, "jump_samples": 16000, "scored_windows_crossing": 1,
                                    "before": {"start_sample": 0, "end_sample": 64000, "source_start_pts": 0, "source_end_pts": 64000},
                                    "after": {"start_sample": 64000, "end_sample": 128000,
                                              "source_start_pts": 80000, "source_end_pts": 144000}}]}
    assert document["coverage"]["scored_samples"] == len(pcm)
    assert len(document["coverage"]["windows"]) == 3
    forged = deepcopy(document)
    forged["coverage"]["source_clock_context"]["breaks"][0]["jump_samples"] += 1
    forged["sha256"] = packet_hash(forged)
    with pytest.raises(PacketIntegrityError, match="Audio source clock observations differ"):
        validate_packet_set(forged, analysis_id=analysis.id, revision=analysis.revision,
                            original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    legacy = deepcopy(document)
    del legacy["coverage"]["source_clock_context"]
    legacy["sha256"] = packet_hash(legacy)
    validate_packet_set(legacy, analysis_id=analysis.id, revision=analysis.revision,
                        original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)

    snapshot = original.model_copy(update={"result": result, "evidence": document,
        "analysis": original.analysis.model_copy(update={"id": analysis.id, "revision": analysis.revision}),
        "media": original.media.model_copy(update={"sha256": analysis.sha256, "size_bytes": analysis.size_bytes,
                                                    "filename": analysis.filename, "content_type": analysis.content_type}),
        "report": original.report.model_copy(update={"template_set": engine.TEMPLATE_SET})})
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    rendered = engine.render_pdf(snapshot)
    text = " ".join(engine.pdf_text(rendered.pdf).split())
    assert "Audio source clock breaks" in text and "At 4.000 s of analyzed audio" in text
    assert "On the original timeline, the audio spans 0.000 s to 4.000 s before the break and 5.000 s to 9.000 s after it." in text
    assert "1 recorded voice window crosses this break." in text and "/coverage/source_clock_context" in text
    assert "carries zero weight" in text
    _check_public_zip(SimpleNamespace(db=db, store=store, analysis=analysis, frozen=snapshot, document=document),
                      tmp_path, monkeypatch, rendered.pdf, source_audio=tmp_path / analysis.filename)


def test_current_audio_requires_source_bound_acoustics_but_legacy_remains_readable(audio_fixture):
    db, store, analysis, result, pcm, _ = audio_fixture
    doc = build_packet_set(db, analysis, store, result)
    assert doc['version']=='audio-evidence-3'
    assert doc['acoustic_context']['source']['pcm_sha256']==doc['source']['pcm_sha256']
    assert doc['acoustic_context']['summary']['measured_samples']==len(pcm)
    missing = deepcopy(doc)
    missing.pop('acoustic_context')
    missing['sha256']=packet_hash(missing)
    with pytest.raises(PacketIntegrityError, match='lacks frozen acoustic'):
        validate_packet_set(missing, analysis_id=analysis.id, revision=analysis.revision,
                            original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    missing['version']='audio-evidence-1'
    missing['sha256']=packet_hash(missing)
    validate_packet_set(missing, analysis_id=analysis.id, revision=analysis.revision,
                        original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    forged = deepcopy(doc)
    forged['acoustic_context']['source']['pcm_sha256']='f'*64
    forged['sha256']=packet_hash(forged)
    with pytest.raises(PacketIntegrityError, match='different source'):
        validate_packet_set(forged, analysis_id=analysis.id, revision=analysis.revision,
                            original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)


def test_unscored_spans_and_below_threshold_do_not_invent_regions(audio_fixture):
    db, store, analysis, result, pcm, entry = audio_fixture
    entry["unit_scores"] = [entry["unit_scores"][0]]
    _replace_entry(store, analysis, result, entry)
    doc = build_packet_set(db, analysis, store, result)
    assert doc["packets"] == []
    assert doc["coverage"]["scored_samples"] == 64000
    assert doc["coverage"]["unscored_samples"] == len(pcm) - 64000
    assert doc["coverage"]["unscored_intervals"] == [
        {"start_sample": 64000, "end_sample": 128000, "source_timestamp_ranges": [
            {"start_sample": 64000, "end_sample": 128000, "source_start_pts": 64000,
             "source_end_pts": 128000, "time_base": "1/16000"}]}
    ]
    forged = deepcopy(doc)
    forged["coverage"]["unscored_intervals"][0]["source_timestamp_ranges"][0]["source_start_pts"] += 1
    forged["sha256"] = packet_hash(forged)
    with pytest.raises(PacketIntegrityError, match="Unscored voice intervals differ"):
        validate_packet_set(forged, analysis_id=analysis.id, revision=analysis.revision,
                            original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    assert any("does not establish authenticity" in s for s in doc["limitations"])


def test_silence_decodes_but_no_successful_windows_means_no_claim(audio_fixture, tmp_path):
    silent = tmp_path / "silence.wav"
    _write_source(silent, silence=True, rate=16000, channels=1, seconds=2)
    decoded = decode_audio(silent, source_sha256=hashlib.sha256(silent.read_bytes()).hexdigest())
    assert not np.any(decoded.pcm)
    db, store, analysis, result, pcm, _ = audio_fixture
    result["manifest"]["sampling"]["audio_evidence"]["checks"] = []
    analysis.result_hash = canonical_hash(result)
    doc = build_packet_set(db, analysis, store, result)
    assert doc["packets"] == [] and doc["coverage"]["scored_samples"] == 0
    assert doc["coverage"]["unscored_samples"] == len(pcm)
    assert [(span["start_sample"], span["end_sample"]) for span in doc["coverage"]["unscored_intervals"]] == [(0, len(pcm))]


@pytest.mark.parametrize("mutation", ["input", "outside", "seconds", "failed", "family", "unlinked"])
def test_mismatched_input_or_window_records_are_refused(audio_fixture, mutation):
    db, store, analysis, result, _, entry = audio_fixture
    if mutation == "input":
        entry["details"]["audio_input"]["pcm_sha256"] = "0" * 64
    elif mutation == "outside":
        entry["unit_scores"][0]["quality"]["end_sample"] = 100000000
    elif mutation == "seconds":
        entry["unit_scores"][0]["quality"]["start_s"] = 1
    elif mutation == "failed":
        entry["status"] = "failed"
    elif mutation == "family":
        entry["family"] = "image"
    else:
        result["signals"] = []
    _replace_entry(store, analysis, result, entry)
    with pytest.raises(PacketIntegrityError):
        build_packet_set(db, analysis, store, result)


def test_real_run_store_output_uses_published_voice_family_without_inventing_metadata(audio_fixture):
    db, store, analysis, result, _, entry = audio_fixture
    # The production writer has a sparse _SpecView: unlike hand-built fixtures,
    # the immutable raw record intentionally does not contain a voice family.
    db.commit()
    runs = DbRunStore(lambda: Session(db.get_bind()), store)
    runs.persist(DetectorRunRecord(
        analysis_id=analysis.id, detector_id=entry["detector_id"], detector_version="test-1", status="ok", reason=None,
        device="cpu", dtype="float32", started_at="2026-10-31T12:00:00+00:00", duration_ms=1,
        result=DetectorResult(status="ok", details=entry["details"], exhibit_ids=entry["exhibit_ids"],
                              unit_scores=[UnitScore(**unit) for unit in entry["unit_scores"]])))
    run = db.scalar(select(DetectorRun).where(DetectorRun.analysis_id == analysis.id))
    raw = db.get(Artefact, run.raw_output_artefact_id)
    recorded = json.loads(store.read_bytes(raw.sha256))
    assert recorded["family"] == ""
    result["manifest"]["sampling"]["audio_evidence"]["checks"][0]["raw_output_sha256"] = raw.sha256
    result["signals"][0]["family"] = "voice"
    analysis.result_hash = canonical_hash(result)
    built = build_packet_set(db, analysis, store, result)
    assert len(built["packets"]) == 1
    assert built["packets"][0]["observations"][0]["raw_output_sha256"] == raw.sha256
    assert store.read_bytes(raw.sha256) == canonical_json(recorded)

    for family in ("image", None):
        result["signals"][0]["family"] = family
        analysis.result_hash = canonical_hash(result)
        with pytest.raises(PacketIntegrityError, match="no successful published signal"):
            build_packet_set(db, analysis, store, result)


def test_read_is_pure_and_persistence_is_idempotent(audio_fixture):
    db, store, analysis, result, _, _ = audio_fixture
    assert load_packet_set(db, analysis, store) is None
    doc = persist_packet_set(db, analysis, store, result=result)
    count = db.scalar(select(func.count()).select_from(Artefact))
    assert load_packet_set(db, analysis, store) == doc
    assert persist_packet_set(db, analysis, store, result=result) == doc
    assert db.scalar(select(func.count()).select_from(Artefact)) == count
    assert load_packet_set(db, analysis, store, set_id="unrelated") is None
    changed = deepcopy(doc)
    changed["packets"][0]["interval"]["end_sample"] += 1
    with pytest.raises(PacketIntegrityError):
        validate_packet_set(changed, analysis_id=analysis.id, revision=1, original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)


def test_older_result_is_not_silently_redecoded(audio_fixture):
    db, store, analysis, result, _, _ = audio_fixture
    result["manifest"]["sampling"] = {}
    analysis.result_hash = canonical_hash(result)
    before = db.scalar(select(func.count()).select_from(Artefact))
    assert persist_packet_set(db, analysis, store, result=result) is None
    assert db.scalar(select(func.count()).select_from(Artefact)) == before


def test_clip_wav_bytes_roundtrip_without_normalisation():
    pcm = np.array([-1, -0.5, 0, 1 / 32768, 32767 / 32768], dtype=np.float32)
    np.testing.assert_array_equal(read_wav(wav_bytes(pcm)), pcm)


def test_real_window_adapter_records_the_same_input_sent_to_inference(monkeypatch):
    pcm = np.arange(SAMPLE_RATE * 7, dtype=np.int32).astype(np.int16).astype(np.float32) / 32768
    seen = []

    def predict(chunks):
        seen.extend(chunks)
        return [0.1, 0.8, 0.2]

    monkeypatch.setattr(AudioClassifier, "from_parts", lambda *a, **kw: SimpleNamespace(predict=predict))
    leased = SimpleNamespace(obj=(None, None), revision="fixed-test-weights")
    ctx = SimpleNamespace(profile=Profile.FULL, device="cpu", lease=SimpleNamespace(get=lambda _: leased),
                          outputs=SimpleNamespace(audio=AudioTrack("analysed", pcm, len(pcm), max_windows=300)),
                          exhibits=SimpleNamespace(series=lambda *a, **kw: "audio-series"))
    result = GustkingVoiceDetector().run(ctx)
    assert result.details["audio_input"]["pcm_sha256"] == hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()
    for unit, actual_input in zip(result.unit_scores, seen, strict=True):
        start, end = int(unit.quality["start_sample"]), int(unit.quality["end_sample"])
        np.testing.assert_array_equal(actual_input, pcm[start:end])
        assert unit.quality["start_s"] == start / SAMPLE_RATE
        assert unit.quality["end_s"] == end / SAMPLE_RATE


pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="audio evidence needs local ffmpeg")
