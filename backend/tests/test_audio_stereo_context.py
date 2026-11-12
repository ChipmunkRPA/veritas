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
from app.detectors.audio.decode import SAMPLE_RATE, decode_audio, read_wav, source_ranges, wav_bytes
from app.engine.stages import audio
from app.evidence import audio_packets, service, stereo_binding, stereo_listening
from app.evidence.audio_binding import stereo_attenuation_overlaps, unscored_voice_intervals
from app.evidence.stereo_listening_binding import select_windows
from app.reports import engine, multimedia_evidence, package, signing, verify, verify_audio_source
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
    legacy.update(schema=stereo_binding.LEGACY_SCHEMA, silence_floor_dbfs=stereo_binding.LEGACY_SILENCE_FLOOR_DBFS)
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


def test_listening_selects_at_most_three_windows_and_preserves_chronological_order():
    stereo = {"schema": stereo_binding.SCHEMA, "windows": [
        {"status": "strong_imbalance", "level_gap_db": gap,
         "downmix": {"strong_attenuation": strong, "attenuation_db": loss}}
        for gap, strong, loss in [(14, False, 4), (14, True, 13), (20, False, 5),
                                  (17, False, 2), (12, True, None)]
    ]}
    assert select_windows(stereo) == [1, 2, 4]


def test_actual_audio_stage_packet_api_export_and_report_use_same_windows(tmp_path, monkeypatch):
    first, second, path = (tmp_path / name for name in ("source.wav", "alternate.wav", "multitrack.mka"))
    _source(first, seconds=2, mode="first_left_then_balanced", rate=SAMPLE_RATE)
    _source(second, seconds=2, channels=1, rate=SAMPLE_RATE)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(first), "-i", str(second),
                    "-map", "0:a", "-map", "1:a", "-c:a", "copy", str(path)], check=True, timeout=30)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    store = ObjectStore(tmp_path / "objects")
    stored = store.put_file(path)
    database = create_engine("sqlite://")
    Base.metadata.create_all(database)
    snapshot = specimen("audio", "standard")
    with Session(database) as db:
        analysis = Analysis(id=snapshot.analysis.id, kind="audio", filename=path.name, content_type="audio/x-matroska",
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
        assert packet["source"]["audio_stream_indices"] == mapping["audio_stream_indices"] == [0, 1]
        assert packet["stereo_context"] == mapping["stereo_context"]
        assert "model" not in json.dumps(packet["stereo_context"]).lower()
        listening = packet["stereo_listening"]
        assert listening["affects_overall_confidence"] is False
        assert [row["stereo_window_index"] for row in listening["windows"]] == [0, 1]
        assets = listening["windows"][1]["artifacts"]
        left, right, mono = (read_wav(store.read_bytes(assets[name]["sha256"])) for name in ("left", "right", "mono"))
        assert np.any(left) and np.array_equal(left, -right) and not np.any(mono)
        assert np.array_equal(mono, context.state["audio"].pcm[SAMPLE_RATE:2*SAMPLE_RATE])
        assert {asset["sha256"] for asset in assets.values()} <= service.asset_refs(packet).keys()
        assert mapping["stereo_listening_pcm_sha256"] not in service.asset_refs(packet)
        decoded = decode_audio(path, source_sha256=sha, inspect_stereo=True)
        raw = stereo_listening.selected_pcm(packet["stereo_context"], decoded.stereo_pcm)
        assert stereo_listening.verify_payloads(listening, packet["stereo_context"], decoded.pcm, raw,
                                                read_waveform=store.read_bytes)
        (overlap,) = packet["coverage"]["stereo_attenuation_overlaps"]
        assert overlap == {
            "start_sample": 16000, "end_sample": 32000, "unscored_interval_index": 0, "stereo_window_index": 1,
            "source_timestamp_ranges": [{"start_sample": 16000, "end_sample": 32000,
                                         "source_start_pts": 16000, "source_end_pts": 32000, "time_base": "1/16000"}],
        }
        legacy = deepcopy(packet)
        del legacy["coverage"]["stereo_attenuation_overlaps"]
        audio_packets.validate_content(legacy)
        forged_overlap = deepcopy(packet)
        forged_overlap["coverage"]["stereo_attenuation_overlaps"][0]["stereo_window_index"] = 0
        with pytest.raises(audio_packets.PacketIntegrityError, match="Unscored stereo overlaps differ"):
            audio_packets.validate_content(forged_overlap)

        monkeypatch.setattr(v2_evidence_packets, "get_session", lambda: Session(database))
        monkeypatch.setattr(v2_evidence_packets, "default_store", lambda: store)
        api = v2_evidence_packets.image_evidence(analysis.id)
        assert json.loads(api.body)["stereo_context"] == packet["stereo_context"]
        assert json.loads(api.body)["stereo_listening"] == listening

        snapshot = snapshot.model_copy(update={"result": result, "evidence": packet,
            "media": snapshot.media.model_copy(update={"sha256": sha, "size_bytes": stored.size_bytes}),
            "report": snapshot.report.model_copy(update={"template_set": engine.TEMPLATE_SET})})
        monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
        assert multimedia_evidence.view(snapshot)["stereo_context"] == packet["stereo_context"]
        rendered = engine.render_pdf(snapshot)
        words = engine.pdf_text(rendered.pdf)
        assert "Audio track coverage" in words and "The second track was unassessed." in " ".join(words.split())
        assert "/source/audio_stream_indices" in words
        gap = packet["stereo_context"]["windows"][0]["level_gap_db"]
        assert "Stereo channel comparison" in words and "Left louder" in words and str(gap) in words
        assert "/stereo_context" in words and "[16000, 32000)" in words
        assert "Strong level drop in the analyzed mono mix." in words
        assert "Mono is digitally silent" in words
        assert "/stereo_listening/windows/1" in words
        assert "Unscored voice during a mono level drop" in words
        assert "Overlap 1.000–2.000 s of analyzed audio" in words and "/coverage/stereo_attenuation_overlaps" in words
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
        checked = verify.verify_package(built.path, public_key_pem=key.public_pem(), source_audio=path)
        assert checked.ok and "32,000 decoded mono samples" in checked.checks[-1].detail, checked.as_dict()
        assert "Recomputed stereo channel and mono-downmix evidence for 2 windows" in checked.checks[-1].detail
        forged_listening = deepcopy(packet)
        clips = forged_listening["stereo_listening"]["windows"][1]["artifacts"]
        clips["left"], clips["right"] = clips["right"], clips["left"]
        forged_listening["sha256"] = audio_packets.packet_hash(forged_listening)
        with pytest.raises(ObjectIntegrityError, match="differs from its committed channel samples"):
            multimedia_evidence.view(snapshot.model_copy(update={"evidence": forged_listening}))
        swapped_plot = deepcopy(packet)
        first, second = (row["artifacts"] for row in swapped_plot["stereo_listening"]["windows"])
        first["waveform"], second["waveform"] = second["waveform"], first["waveform"]
        swapped_plot["sha256"] = audio_packets.packet_hash(swapped_plot)
        with pytest.raises(ObjectIntegrityError, match="waveform differs from its committed channel samples"):
            multimedia_evidence.view(snapshot.model_copy(update={"evidence": swapped_plot}))
        wrong_source = tmp_path / "different.wav"
        _source(wrong_source, seconds=2, mode="in_phase", rate=SAMPLE_RATE)
        checked = verify.verify_package(built.path, public_key_pem=key.public_pem(), source_audio=wrong_source)
        assert checked.outcome == "not_verified"
        assert checked.checks[-1].name == "original audio samples" and "differs from the recorded original" in checked.checks[-1].detail

        forged = deepcopy(packet)
        forged["stereo_context"]["windows"][0]["correlation"] = 0.5
        forged["sha256"] = audio_packets.packet_hash(forged)
        with pytest.raises(ObjectIntegrityError, match="differs from the frozen audio input mapping"):
            multimedia_evidence.view(snapshot.model_copy(update={"evidence": forged}))
        stereo.validate(forged["stereo_context"], packet["source"])
        forged_path = tmp_path / "changed-stereo-evidence.zip"
        with zipfile.ZipFile(built.path) as export, zipfile.ZipFile(forged_path, "w") as modified:
            bundle = json.loads(export.read("bundle.json"))
            modified.writestr("evidence.json", canonical_json(forged))
        with zipfile.ZipFile(forged_path) as export, pytest.raises(ValueError, match="Stereo channel and mono-downmix"):
            verify_audio_source.check(export, bundle, path)


def test_source_audio_verifier_preserves_gap_inside_one_playback_interval(tmp_path):
    wav = tmp_path / "input.wav"
    _source(wav, seconds=2, mode="in_phase", rate=SAMPLE_RATE)
    original = tmp_path / "with-gap.mka"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav),
                    "-af", "asetpts=PTS+if(gte(T\\,1)\\,0.125/TB\\,0)",
                    "-c:a", "pcm_s16le", str(original)], check=True, timeout=20)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    decoded = decode_audio(original, source_sha256=digest)
    assert len(decoded.mapping["timeline"]) == 2
    start, end = 15000, 20000
    timestamps = source_ranges(decoded.mapping, start, end)
    assert len(timestamps) == 2 and timestamps[1]["source_start_pts"] - timestamps[0]["source_end_pts"] == 2000
    clip = wav_bytes(decoded.pcm[start:end])
    asset = {"sha256": hashlib.sha256(clip).hexdigest(), "media_type": "audio/wav"}
    bundle = {"schema": package.PUBLIC_SCHEMA, "analysis": {"kind": "audio"}, "source_commitment": {"sha256": digest},
              "artifacts": [asset]}
    evidence = {
        "source": {**decoded.mapping, "sha256": digest, "wav_sha256": hashlib.sha256(decoded.wav).hexdigest()},
        "coverage": {"decode_limit_reached": False}, "packets": [],
        "sound_events": {"windows": [{"id": "S01", "start_sample": start, "end_sample": end,
                                      "source_timestamp_ranges": timestamps, "artifacts": {"selected_clip": asset}}]},
    }
    archive_path = tmp_path / "audio.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with zipfile.ZipFile(archive_path) as archive:
        assert "Verified 1 published playback interval across 1 WAV asset" in verify_audio_source.check(archive, bundle, original)
    timestamps[1]["source_start_pts"] -= 2000
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with zipfile.ZipFile(archive_path) as archive, pytest.raises(ValueError, match="recorded source timestamps"):
        verify_audio_source.check(archive, bundle, original)


def test_voice_gap_intersection_preserves_discontinuous_source_clock_and_excludes_touching_ranges():
    source = {"samples": 32000, "sample_rate": 16000, "timeline_time_base": "1/16000", "timeline": [
        {"start_sample": 0, "end_sample": 5000, "source_pts": 80000},
        {"start_sample": 5000, "end_sample": 32000, "source_pts": 100000},
    ]}
    windows = [{"start_sample": 0, "end_sample": 3000}, {"start_sample": 7000, "end_sample": 16000}]
    coverage = {"decoded_samples": 32000, "scored_samples": 12000, "unscored_samples": 20000,
                "scored_intervals": windows, "windows": windows}
    gaps = unscored_voice_intervals(coverage, source)
    context = {"schema": stereo_binding.SCHEMA, "windows": [
        {"start_sample": 0, "end_sample": 16000, "downmix": {"strong_attenuation": True}},
        {"start_sample": 16000, "end_sample": 32000, "downmix": {"strong_attenuation": False}},
    ]}
    assert stereo_attenuation_overlaps(gaps, context) == [{
        "start_sample": 3000, "end_sample": 7000, "unscored_interval_index": 0, "stereo_window_index": 0,
        "source_timestamp_ranges": [
            {"start_sample": 3000, "end_sample": 5000, "source_start_pts": 83000, "source_end_pts": 85000, "time_base": "1/16000"},
            {"start_sample": 5000, "end_sample": 7000, "source_start_pts": 100000, "source_end_pts": 102000, "time_base": "1/16000"},
        ],
    }]
