from __future__ import annotations

import hashlib
import json
import subprocess
import wave
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api import v2_evidence_packets
from app.core.canonical import canonical_json
from app.core.config import settings
from app.detectors.audio import decode as audio_decode
from app.detectors.audio import stereo
from app.detectors.audio.decode import SAMPLE_RATE, decode_audio
from app.engine.stages import audio
from app.evidence import audio_packets
from app.reports import engine, multimedia_evidence, package, signing, verify
from app.reports import snapshot as report_snapshot
from app.reports.specimens import specimen
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Base, ReportRow
from app.storage.objects import ObjectIntegrityError, ObjectStore


def _source(path, *, seconds, rate=48000, channels=2, mode="left"):
    ticks = np.arange(round(rate * seconds))
    loud = (12000 * np.sin(2 * np.pi * 237 * ticks / rate)).astype("<i2")
    quiet = loud // 16
    if mode == "balanced":
        values = np.column_stack((loud, -loud))
    elif mode == "partial":
        values = np.column_stack((loud, (-0.8 * loud).astype("<i2")))
    elif mode == "in_phase":
        values = np.column_stack((loud, loud))
    elif mode == "quadrature":
        delayed = (12000 * np.cos(2 * np.pi * 237 * ticks / rate)).astype("<i2")
        values = np.column_stack((loud, delayed))
    elif mode == "first_left_then_balanced":
        right = np.where(ticks < rate, quiet, -loud)
        values = np.column_stack((loud, right))
    elif mode == "sparse":
        tiny = np.zeros_like(loud)
        tiny[100] = 1
        values = np.column_stack((tiny, tiny))
    elif mode == "quiet":
        values = np.column_stack((quiet // 16, quiet // 16))
    elif mode == "right":
        values = np.column_stack((quiet, loud))
    else:
        values = np.column_stack((loud, quiet))
    if channels == 1:
        values = loud
    elif channels > 2:
        values = np.tile(loud[:, None], (1, channels))
    with wave.open(str(path), "wb") as output:
        output.setparams((channels, 2, rate, 0, "NONE", ""))
        output.writeframes(values.astype("<i2").tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mode,louder,corr", [("left", "left", 1.0), ("right", "right", 1.0), ("balanced", None, -1.0)])
def test_stereo_context_preserves_mono_pcm_and_source_clock(tmp_path, mode, louder, corr):
    path = tmp_path / "source.wav"
    sha = _source(path, seconds=2, mode=mode)
    reference = decode_audio(path, source_sha256=sha)
    decoded = decode_audio(path, source_sha256=sha, inspect_stereo=True)
    context = decoded.mapping["stereo_context"]
    assert reference.wav == decoded.wav
    assert {k: v for k, v in decoded.mapping.items() if k != "stereo_context"} == reference.mapping
    assert context["status"] == "measured" and context["affects_overall_confidence"] is False
    assert context["source_sha256"] == sha and len(context["windows"]) == 2
    assert context["mono_pcm_sha256"] == reference.mapping["pcm_sha256"]
    stereo.validate(context, decoded.mapping)
    stereo.verify_mono_levels(context, reference.pcm)
    for index, row in enumerate(context["windows"]):
        assert row["start_sample"] == index * SAMPLE_RATE
        assert row["louder_channel"] == louder and row["correlation"] == corr
        expected_gap = pytest.approx(24.08, abs=0.01) if louder else 0
        assert row["level_gap_db"] == expected_gap
        downmix = row["downmix"]
        assert downmix["strong_attenuation"] is (mode == "balanced")
        if mode == "balanced":
            assert downmix["mono_rms_dbfs"] == context["silence_floor_dbfs"]
            assert downmix["attenuation_db"] is None
        assert row["source_timestamp_ranges"] == [{
            "start_sample": index * SAMPLE_RATE, "end_sample": (index + 1) * SAMPLE_RATE,
            "source_start_pts": index * SAMPLE_RATE, "source_end_pts": (index + 1) * SAMPLE_RATE,
            "time_base": "1/16000",
        }]


@pytest.mark.parametrize("mode,attenuation,marked", [
    ("in_phase", 0.0, False), ("quadrature", 3.01, False), ("partial", 19.998, True),
    ("quiet", None, False), ("sparse", None, False),
])
def test_downmix_distinguishes_phase_loss_from_controls(tmp_path, mode, attenuation, marked):
    path = tmp_path / "source.wav"
    sha = _source(path, seconds=1, mode=mode, rate=SAMPLE_RATE if mode == "sparse" else 48000)
    decoded = decode_audio(path, source_sha256=sha, inspect_stereo=True)
    context = decoded.mapping["stereo_context"]
    stereo.validate(context, decoded.mapping)
    window = context["windows"][0]
    assert window["downmix"]["strong_attenuation"] is marked
    if attenuation is None:
        assert window["downmix"]["attenuation_db"] is None
    else:
        assert window["downmix"]["attenuation_db"] == pytest.approx(attenuation, abs=0.015)
    if mode == "sparse":
        assert window["downmix"]["mono_rms_dbfs"] == pytest.approx(-132.35, abs=0.02)
        assert window["left_rms_dbfs"] == window["right_rms_dbfs"] == window["downmix"]["mono_rms_dbfs"]


def test_legacy_stereo_context_remains_readable_and_rejects_unversioned_downmix(tmp_path):
    path = tmp_path / "source.wav"
    sha = _source(path, seconds=2)
    decoded = decode_audio(path, source_sha256=sha, inspect_stereo=True)
    legacy = deepcopy(decoded.mapping["stereo_context"])
    legacy.update(schema=stereo.LEGACY_SCHEMA, silence_floor_dbfs=stereo.LEGACY_SILENCE_FLOOR_DBFS)
    del legacy["mono_pcm_sha256"], legacy["strong_downmix_attenuation_db"]
    for row in legacy["windows"]:
        del row["downmix"]
    stereo.validate(legacy, decoded.mapping)
    stereo.verify_mono_levels(legacy, decoded.pcm)
    legacy["windows"][0]["downmix"] = {"strong_attenuation": True}
    with pytest.raises(ValueError, match="Legacy stereo context"):
        stereo.validate(legacy, decoded.mapping)


@pytest.mark.parametrize("channels,seconds,mode,status", [
    (1, 2, "left", "mono"), (3, 2, "left", "unsupported"),
    (2, 0.5, "left", "too_short"), (2, 2, "quiet", "quiet"),
])
def test_stereo_context_reports_unassessed_input_explicitly(tmp_path, channels, seconds, mode, status):
    path = tmp_path / "source.wav"
    sha = _source(path, seconds=seconds, channels=channels, mode=mode)
    decoded = decode_audio(path, source_sha256=sha, inspect_stereo=True)
    context = decoded.mapping["stereo_context"]
    assert context["status"] == status
    if status == "quiet":
        assert len(context["windows"]) == 2
        assert all(w["status"] == "quiet" and w["louder_channel"] is None and w["level_gap_db"] is None
                   for w in context["windows"])
    else:
        assert context["windows"] == [] and context["stereo_pcm_sha256"] is None


def test_unavailable_stereo_sidecar_does_not_block_mono_analysis(tmp_path, monkeypatch):
    path = tmp_path / "source.wav"
    sha = _source(path, seconds=2)
    baseline = decode_audio(path, source_sha256=sha)
    original_timeline = audio_decode._named_timeline
    monkeypatch.setattr(audio_decode, "_named_timeline",
                        lambda log, name: "" if name == "mono" else original_timeline(log, name))
    decoded = decode_audio(path, source_sha256=sha, inspect_stereo=True)
    assert decoded.wav == baseline.wav
    assert decoded.mapping["timeline"] == baseline.mapping["timeline"]
    assert decoded.mapping["stereo_context"]["status"] == "unavailable"
    assert decoded.mapping["stereo_context"]["windows"] == []


def test_stereo_windows_use_real_source_pts_and_stop_after_30_seconds(tmp_path):
    path, shifted = tmp_path / "source.wav", tmp_path / "shifted.mka"
    _source(path, seconds=31, rate=SAMPLE_RATE)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-c:a", "pcm_s16le",
                    "-output_ts_offset", "2", str(shifted)], check=True, capture_output=True)
    sha = hashlib.sha256(shifted.read_bytes()).hexdigest()
    decoded = decode_audio(shifted, source_sha256=sha, inspect_stereo=True)
    windows = decoded.mapping["stereo_context"]["windows"]
    assert len(decoded.pcm) == 31 * SAMPLE_RATE and len(windows) == 30
    assert windows[0]["source_timestamp_ranges"][0]["source_start_pts"] == 2 * SAMPLE_RATE
    assert windows[-1]["end_sample"] == 30 * SAMPLE_RATE
    assert windows[-1]["source_timestamp_ranges"][-1]["source_end_pts"] == 32 * SAMPLE_RATE


def test_actual_audio_stage_packet_api_export_and_report_use_same_windows(tmp_path, monkeypatch):
    path = tmp_path / "source.wav"
    sha = _source(path, seconds=2, mode="first_left_then_balanced", rate=SAMPLE_RATE)
    store = ObjectStore(tmp_path / "objects")
    stored = store.put_file(path)
    database = create_engine("sqlite://")
    Base.metadata.create_all(database)
    snapshot = specimen("audio", "standard")
    with Session(database) as db:
        analysis = Analysis(id=snapshot.analysis.id, kind="audio", filename=path.name, content_type="audio/wav",
                            size_bytes=stored.size_bytes, sha256=sha, status="done", revision=snapshot.analysis.revision)
        db.add(analysis)
        db.commit()
        context = SimpleNamespace(outputs={"ingest": {"sha256": sha}}, store=store, state={},
                                  check=lambda *args: None, sessions=lambda: Session(database), analysis_id=analysis.id)
        evidence_input = audio.decode(context)["evidence_input"]
        result = deepcopy(snapshot.result)
        result["input"]["sha256"] = sha
        result["manifest"]["sampling"]["audio_evidence"] = {"input": evidence_input, "checks": []}
        analysis.result_hash = store.put_bytes(canonical_json(result)).sha256
        packet = audio_packets.build_packet_set(db, analysis, store, result)
        db.commit()
        mapping = json.loads(store.read_bytes(evidence_input["mapping_sha256"]))
        assert packet["stereo_context"] == mapping["stereo_context"]
        assert "model" not in json.dumps(packet["stereo_context"]).lower()

        monkeypatch.setattr(v2_evidence_packets, "get_session", lambda: Session(database))
        monkeypatch.setattr(v2_evidence_packets, "default_store", lambda: store)
        api = v2_evidence_packets.image_evidence(analysis.id)
        assert json.loads(api.body)["stereo_context"] == packet["stereo_context"]

        snapshot = snapshot.model_copy(update={"result": result, "evidence": packet,
            "media": snapshot.media.model_copy(update={"sha256": sha, "size_bytes": stored.size_bytes}),
            "report": snapshot.report.model_copy(update={"template_set": engine.TEMPLATE_SET})})
        monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
        assert multimedia_evidence.view(snapshot)["stereo_context"] == packet["stereo_context"]
        rendered = engine.render_pdf(snapshot)
        words = engine.pdf_text(rendered.pdf)
        gap = packet["stereo_context"]["windows"][0]["level_gap_db"]
        assert "Stereo channel comparison" in words and "Left louder" in words and str(gap) in words
        assert "/stereo_context" in words and "[16000, 32000)" in words
        assert "Strong level drop in the analyzed mono mix." in words
        assert "Mono is digitally silent" in words
        assert [w["downmix"]["strong_attenuation"] for w in packet["stereo_context"]["windows"]] == [False, True]

        altered_mono = deepcopy(packet["stereo_context"])
        altered_mono["windows"][1]["downmix"]["mono_rms_dbfs"] = -60.0
        with pytest.raises(ValueError, match="differs from the analyzed mono samples"):
            stereo.verify_mono_levels(altered_mono, context.state["audio"].pcm)


        key = signing.instance_key(tmp_path / "test-key.pem")
        pdf_artefact = register_artefact(db, store, rendered.pdf, media_type="application/pdf", role="report",
                                         created_by="report", analysis_id=analysis.id)
        snapshot_artefact = register_artefact(db, store, report_snapshot.dump(snapshot), media_type="application/json",
                                              role="stage_output", created_by="report-snapshot", analysis_id=analysis.id)
        report = ReportRow(id=snapshot.report.id, display_id=snapshot.report.display_id, revision=1,
                           analysis_id=analysis.id, analysis_revision=analysis.revision, profile="standard",
                           redacted=False, review_state="machine", issuable=False, template_set=engine.TEMPLATE_SET,
                           engine=rendered.engine_version, status="done", snapshot_hash=report_snapshot.snapshot_hash(snapshot),
                           snapshot_artefact_id=snapshot_artefact.id, artefact_id=pdf_artefact.id, sha256=pdf_artefact.sha256,
                           signature=key.sign(rendered.pdf), signing_key_id=key.key_id, requested_by="test",
                           requested_at=snapshot.report.requested_at, page_count=rendered.page_count)
        db.add(report)
        request = package.prepare(db, analysis, report, kind="draft", include_original=False, created_by="test",
                                  schema_id=package.PUBLIC_SCHEMA)
        monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
        built = package.build(db, request, store=store, key=key, exported_at=datetime.now(UTC))
        with zipfile.ZipFile(built.path) as export:
            assert json.loads(export.read("evidence.json")) == json.loads(api.body)
            assert export.read("report.pdf") == rendered.pdf
        verified = verify.verify_package(built.path, public_key_pem=key.public_pem())
        assert verified.ok, verified.as_dict()

        forged = deepcopy(packet)
        forged["stereo_context"]["windows"][0]["correlation"] = 0.5
        forged["sha256"] = audio_packets.packet_hash(forged)
        with pytest.raises(ObjectIntegrityError, match="differs from the frozen audio input mapping"):
            multimedia_evidence.view(snapshot.model_copy(update={"evidence": forged}))
