"""Production repetition evidence using authored PCM and an encoded video."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import wave
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymupdf
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from test_public_evidence_package import rewrite

from app.api import v2_evidence_packets
from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.audio.decode import SAMPLE_RATE, decode_audio, read_wav
from app.detectors.video.decode import probe_frames
from app.engine.stages import audio, video
from app.evidence import (
    audio_exact_repetition,
    audio_repetition,
    service,
    video_audio,
    video_packets,
)
from app.evidence.audio_repetition_binding import validate
from app.reports import engine, multimedia_evidence, package, signing, verify
from app.reports import snapshot as report_snapshot
from app.reports.specimens import specimen
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, AnalysisStage, Base, ReportRow
from app.storage.objects import ObjectStore
from scripts import audio_exact_repetition as historical_repetition

FIRST, SECOND, LENGTH = 5120, 28160, 3200


def _authored():
    pcm = np.random.default_rng(113).integers(-16000, 16000, size=3 * SAMPLE_RATE, dtype="<i2")
    pcm[SECOND:SECOND + LENGTH] = pcm[FIRST:FIRST + LENGTH]
    assert pcm[SECOND - 1] != pcm[FIRST - 1]
    assert pcm[SECOND + LENGTH] != pcm[FIRST + LENGTH]
    return pcm


def _wav(path, pcm):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, SAMPLE_RATE, 0, "NONE", ""))
        output.writeframes(pcm.tobytes())
    return path


def _stage_record(db, store, analysis_id, name, data):
    asset = register_artefact(db, store, canonical_json(data), media_type="application/json", role="stage_output",
                              created_by=name, analysis_id=analysis_id)
    db.add(AnalysisStage(analysis_id=analysis_id, stage=name, status="done", output_artefact_id=asset.id))


def _signed_report_and_export(tmp_path, monkeypatch, db, store, analysis, snapshot, api, pair, source_pts, source_path):
    kind = analysis.kind
    clip = pair["artifacts"]["first_clip"]
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    rendered = engine.render_pdf(snapshot)
    text = " ".join(engine.pdf_text(rendered.pdf).split())
    assert "Exact repeated audio" in text and "0.200 s of identical samples" in text
    assert "0.320–0.520 s" in text and "1.760–1.960 s" in text
    assert f"Source PTS [{source_pts + FIRST}, {source_pts + FIRST + LENGTH})" in text
    assert f"Source PTS [{source_pts + SECOND}, {source_pts + SECOND + LENGTH})" in text
    compact = "".join(text.split())
    assert all(digest in compact for digest in (clip["sha256"], pair["pcm_s16le_sha256"], api["sha256"]))
    assert ("/audio/repetition_context" if kind == "video" else "/repetition_context") in text
    if kind == "video":
        assert "Video playback 0.720–0.920 s" in text and "Video playback 2.160–2.360 s" in text

    redacted = snapshot.model_copy(update={"report": snapshot.report.model_copy(update={"redacted": True})})
    html = engine.render_html(redacted, embed_fonts=False)
    assert clip["sha256"] in html and pair["pcm_s16le_sha256"] in html and "Exact repeated audio" in html
    assert "<audio" not in html and "data:audio/" not in html and "data:image/png;base64," not in html
    assert clip["url"] not in html
    with pymupdf.open(stream=rendered.pdf, filetype="pdf") as pdf:
        assert all(not name.endswith(".wav") for name in pdf.embfile_names())

    key = signing.instance_key(tmp_path / "test-key.pem")
    pdf_asset = register_artefact(db, store, rendered.pdf, media_type="application/pdf", role="report",
                                  created_by="report", analysis_id=analysis.id)
    snapshot_asset = register_artefact(db, store, report_snapshot.dump(snapshot), media_type="application/json",
                                       role="stage_output", created_by="report-snapshot", analysis_id=analysis.id)
    report = ReportRow(id=snapshot.report.id, display_id=snapshot.report.display_id, revision=1,
                       analysis_id=analysis.id, analysis_revision=analysis.revision, profile="standard",
                       redacted=False, review_state="machine", issuable=False, template_set=engine.TEMPLATE_SET,
                       engine=rendered.engine_version, status="done", snapshot_hash=report_snapshot.snapshot_hash(snapshot),
                       snapshot_artefact_id=snapshot_asset.id, artefact_id=pdf_asset.id, sha256=pdf_asset.sha256,
                       signature=key.sign(rendered.pdf), signing_key_id=key.key_id, requested_by="test",
                       requested_at=snapshot.report.requested_at, page_count=rendered.page_count)
    db.add(report)
    request = package.prepare(db, analysis, report, kind="draft", include_original=False, created_by="test",
                              schema_id=package.PUBLIC_SCHEMA)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    exported = package.build(db, request, store=store, key=key, exported_at=datetime.now(UTC))
    with zipfile.ZipFile(exported.path) as archive:
        asset_path = f"assets/{clip['sha256']}.wav"
        assert json.loads(archive.read("evidence.json")) == api
        assert archive.read("report.pdf") == rendered.pdf
        assert archive.namelist().count(asset_path) == 1
        assert archive.read(asset_path) == store.read_bytes(clip["sha256"])
        with archive.open(asset_path) as stream, wave.open(stream, "rb") as wav:
            assert hashlib.sha256(wav.readframes(wav.getnframes())).hexdigest() == pair["pcm_s16le_sha256"]

    trusted = tmp_path / "trusted.pem"
    trusted.write_bytes(key.public_pem())
    offline = subprocess.run([sys.executable, "-B", "-c", """
import json
import socket
import sys
from pathlib import Path
from app.reports.verify import verify_package
def unavailable(*args, **kwargs):
    raise RuntimeError("Offline verifier attempted network access")
socket.socket = unavailable
result = verify_package(Path(sys.argv[1]), public_key_pem=Path(sys.argv[2]).read_bytes())
assert not {"torch", "numpy", "app.core.config", "app.storage.db"} & sys.modules.keys()
print(json.dumps(result.as_dict()))
sys.exit(0 if result.ok else 1)
""", str(exported.path), str(trusted)], capture_output=True, text=True, timeout=20)
    assert offline.returncode == 0, offline.stdout + offline.stderr
    verified = verify.verify_package(exported.path, public_key_pem=key.public_pem(), source_audio=source_path)
    assert verified.ok, verified.as_dict()
    assert "Verified 4 published playback intervals across 3 WAV assets" in verified.checks[-1].detail

    def change_pair(files, field):
        evidence, bundle = json.loads(files["evidence.json"]), json.loads(files["bundle.json"])
        body = evidence if kind == "audio" else evidence["audio"]
        altered_pair = body["repetition_context"]["pairs"][0]
        if field == "source_time":
            altered_pair["second"]["source_timestamp_ranges"][0]["source_start_pts"] += 1
        elif field == "pcm":
            altered_pair["pcm_s16le_sha256"] = "f" * 64
        else:
            for side in ("first", "second"):
                interval = altered_pair[side]
                for key in ("start_sample", "end_sample"):
                    interval[key] += 1
                for key in ("start_sample", "end_sample", "source_start_pts", "source_end_pts"):
                    interval["source_timestamp_ranges"][0][key] += 1
        evidence["sha256"] = service.packet_hash(evidence)
        files["evidence.json"] = canonical_json(evidence)
        bundle["evidence"].update(sha256=hashlib.sha256(files["evidence.json"]).hexdigest(), set_sha256=evidence["sha256"])
        bundle["reports"][0]["evidence_set_sha256"] = evidence["sha256"]
        files["bundle.json"] = canonical_json(bundle)

    for field, expected in (("source_time", "recorded source timestamps"), ("pcm", "its committed samples")):
        altered = rewrite(exported.path, tmp_path / f"changed-{field}.zip", partial(change_pair, field=field), key)
        failed = verify.verify_package(altered, public_key_pem=key.public_pem())
        assert failed.outcome == "not_verified"
        assert any(expected in check.detail for check in failed.checks)
    shifted = rewrite(exported.path, tmp_path / "shifted-repetition.zip", partial(change_pair, field="coordinates"), key)
    assert verify.verify_package(shifted, public_key_pem=key.public_pem()).ok
    failed = verify.verify_package(shifted, public_key_pem=key.public_pem(), source_audio=source_path)
    assert failed.outcome == "not_verified" and "differs from the samples at its recorded source interval" in failed.checks[-1].detail


def test_serving_method_preserves_exact_historical_algorithm_bytes():
    # Old evaluation freezes pin the original script's byte hash. The runtime
    # image ships app/ and omits that script, so the promoted copy must agree.
    assert Path(audio_exact_repetition.__file__).read_bytes() == Path(historical_repetition.__file__).read_bytes()
    payload = _authored().tobytes()
    assert audio_exact_repetition.find_repeated_spans(payload).spans == (
        audio_exact_repetition.RepeatedSpan((FIRST, FIRST + LENGTH), (SECOND, SECOND + LENGTH)),
    )


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_production_packet_and_asset_read_use_the_exact_repeated_source_ranges(tmp_path, monkeypatch, kind):
    pcm = _authored()
    source_path = _wav(tmp_path / "source.wav", pcm)
    source_pts = 0
    if kind == "video":
        path = tmp_path / "offset.mkv"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        "color=size=96x64:rate=25:d=4", "-itsoffset", "0.4", "-i", str(source_path),
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "ffv1", "-c:a", "pcm_s16le",
                        "-output_ts_offset", "5", str(path)], check=True, timeout=20)
        source_path = path
        source_pts = 86400
    store = ObjectStore(tmp_path / "objects")
    source = store.put_file(source_path)
    database = create_engine("sqlite://")
    Base.metadata.create_all(database)

    def sessions():
        return Session(database)

    snapshot = specimen(kind, "standard")
    with sessions() as db:
        analysis = Analysis(id=snapshot.analysis.id, kind=kind, filename=source_path.name,
                            content_type="audio/wav" if kind == "audio" else "video/x-matroska",
                            size_bytes=source.size_bytes, sha256=source.sha256, status="done", revision=snapshot.analysis.revision)
        db.add(analysis)
        db.commit()
        ctx = SimpleNamespace(outputs={"ingest": {"sha256": source.sha256}}, store=store, check=lambda *args: None,
                              sessions=sessions, state={"media": SimpleNamespace(path=source_path)}, analysis_id=analysis.id)
        sampling = {}
        if kind == "audio":
            binding = audio.decode(ctx)["evidence_input"]
            sampling["audio_evidence"] = {"input": binding, "checks": []}
        else:
            frame_table = probe_frames(source_path).to_json()
            ctx.outputs["probe"] = {"frame_table": frame_table, "plan": {"legacy": {"frames": [0, 8, 16]}},
                                    "info": {"has_audio": True}}
            ctx.outputs["decode"] = video.decode(ctx)
            anchors = ctx.state["frame_store"].native_anchors()
            ctx.outputs["render"] = {"frame_anchor_version": video_packets.ANCHOR_VERSION, "frame_anchors": anchors}
            _stage_record(db, store, analysis.id, "probe", ctx.outputs["probe"])
            _stage_record(db, store, analysis.id, "render", ctx.outputs["render"])
            sampling.update(video_audio_evidence=video_audio.published_record(ctx),
                            frame_anchor_version=video_packets.ANCHOR_VERSION,
                            frame_table_sha256=canonical_hash(frame_table), frame_anchor_sha256=canonical_hash(anchors),
                            timeline_origin_s=5.0)
            binding = sampling["video_audio_evidence"]["input"]

        result = deepcopy(snapshot.result)
        result["input"]["sha256"] = source.sha256
        result["manifest"]["sampling"] = sampling
        result.update(signals=[], findings=[], exhibits=[])
        analysis.result_hash = store.put_bytes(canonical_json(result)).sha256
        document = service.persist_packet_set(db, analysis, store, result=result)
        db.commit()

        monkeypatch.setattr(v2_evidence_packets, "get_session", sessions)
        monkeypatch.setattr(v2_evidence_packets, "default_store", lambda: store)
        api = json.loads(v2_evidence_packets.image_evidence(analysis.id).body)
        assert api == document
        part = api if kind == "audio" else api["audio"]
        context = part["repetition_context"]
        assert context["status"] == "measured" and context["affects_overall_confidence"] is False
        assert context["searched_interval"] == {"start_sample": 0, "end_sample": len(pcm)}
        [pair] = context["pairs"]
        assert pair["id"] == "R01" and pair["sample_count"] == LENGTH and pair["lag_samples"] == SECOND - FIRST
        for name, start in (("first", FIRST), ("second", SECOND)):
            assert pair[name] == {"start_sample": start, "end_sample": start + LENGTH, "source_timestamp_ranges": [
                {"start_sample": start, "end_sample": start + LENGTH, "source_start_pts": source_pts + start,
                 "source_end_pts": source_pts + start + LENGTH, "time_base": "1/16000"},
            ]}
        clip = pair["artifacts"]["first_clip"]
        assert clip == pair["artifacts"]["second_clip"]
        assert service.asset_refs(api)[clip["sha256"]] == clip
        assert pair["pcm_s16le_sha256"] == hashlib.sha256(pcm[FIRST:FIRST + LENGTH].tobytes()).hexdigest()
        np.testing.assert_array_equal(read_wav(store.read_bytes(clip["sha256"])), pcm[FIRST:FIRST + LENGTH] / 32768)

        frozen = read_wav(store.read_bytes(binding["wav_sha256"]))
        mapping = json.loads(store.read_bytes(binding["mapping_sha256"]))
        audio_repetition.verify(context, pcm=frozen, mapping=mapping, source=part["source"], read_asset=store.read_bytes)
        with pytest.raises(ValueError, match="playback clip"):
            audio_repetition.verify(context, pcm=frozen, mapping=mapping, source=part["source"], read_asset=lambda _: b"bad")
        forged = deepcopy(context)
        forged["pairs"][0]["second"]["source_timestamp_ranges"][0]["source_start_pts"] += 1
        with pytest.raises(ValueError, match="source timestamps"):
            validate(forged, part["source"])

        # A stored analysis made before this cue has no field at all.
        historic = deepcopy(api)
        old_audio = historic if kind == "audio" else historic["audio"]
        del old_audio["repetition_context"]
        historic["sha256"] = service.packet_hash(historic)
        service.validate_packet_set(historic, analysis_id=analysis.id, revision=analysis.revision,
                                    original_result_hash=analysis.result_hash, source_sha256=source.sha256)

        snapshot = snapshot.model_copy(update={
            "result": result, "evidence": api,
            "media": snapshot.media.model_copy(update={"sha256": source.sha256, "size_bytes": source.size_bytes,
                                                       "filename": analysis.filename, "content_type": analysis.content_type}),
            "report": snapshot.report.model_copy(update={"template_set": engine.TEMPLATE_SET}),
        })
        _signed_report_and_export(tmp_path, monkeypatch, db, store, analysis, snapshot, api, pair, source_pts, source_path)


@pytest.mark.parametrize("mode,status,reason", [
    ("short", "unavailable", "duration_out_of_scope"),
    ("long", "unavailable", "duration_out_of_scope"),
    ("clock_gap", "unavailable", "source_clock_discontinuous"),
    ("quiet", "unavailable", "insufficient_signal_variation"),
    ("periodic", "ambiguous", "anchor_pair_budget"),
    ("ordinary", "measured", None),
])
def test_unsupported_or_ambiguous_inputs_have_no_claim(tmp_path, mode, status, reason):
    raw = _authored()
    if mode == "short":
        raw = raw[:SAMPLE_RATE]
    elif mode == "long":
        raw = np.tile(raw, 2)
    elif mode == "quiet":
        raw = np.zeros_like(raw)
    elif mode == "periodic":
        raw = np.resize(raw[:127], len(raw))
    elif mode == "ordinary":
        raw = np.random.default_rng(416).integers(-16000, 16000, size=len(raw), dtype="<i2")
    path = _wav(tmp_path / "sample.wav", raw)
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    decoded = decode_audio(path, source_sha256=source_sha)
    mapping = decoded.mapping
    if mode == "clock_gap":
        midpoint = len(raw) // 2
        mapping["timeline"] = [{"start_sample": 0, "end_sample": midpoint, "source_pts": 0},
                               {"start_sample": midpoint, "end_sample": len(raw), "source_pts": midpoint + 100}]
    source = {**mapping, "sha256": source_sha}
    context = audio_repetition.build(decoded.pcm, mapping, source, lambda *a, **k: pytest.fail("No clip expected"))
    assert (context["status"], context["reason"], context["pairs"]) == (status, reason, [])
    assert (context["searched_interval"] is None) == (mode in {"short", "long", "clock_gap"})
