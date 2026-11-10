"""Real source/evidence/rendering with authored detector scores, never accuracy."""

import copy
import hashlib
import json
import os
import zipfile
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from test_audio_evidence import _write_source

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.audio import combined, combined_adapter
from app.detectors.audio.decode import decode_audio, source_ranges
from app.engine.stages import audio_screening
from app.evidence.audio_packets import build_packet_set
from app.reports import engine, snapshot
from app.reports.specimens import specimen
from app.schemas.evidence import AnalysisResultV2
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Base
from app.storage.objects import ObjectStore


@pytest.fixture
def combined_export(tmp_path, monkeypatch, request):
    probability = getattr(request, "param", 0.55)
    direction = "ai" if probability > 0.5 else "original" if probability < 0.5 else "inconclusive"
    reason = {
        "ai": "Leaning towards AI-generated speech.",
        "original": "Leaning towards original speech.",
    }.get(direction, "No directional preference.")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    source_path = tmp_path / "AUTHORED-SOFTWARE-TEST-NO-MODEL-INFERENCE.wav"
    _write_source(source_path, rate=16000, channels=1, seconds=4)
    store = ObjectStore(tmp_path / "objects")
    source = store.put_file(source_path)
    decoded = decode_audio(source_path, source_sha256=source.sha256)
    sql = create_engine("sqlite://")
    Base.metadata.create_all(sql)
    with Session(sql) as db:
        analysis = Analysis(
            id="authored-combined-audio-export",
            kind="audio",
            filename=source_path.name,
            content_type="audio/wav",
            size_bytes=source.size_bytes,
            sha256=source.sha256,
            status="done",
            revision=1,
            result_schema_version=2,
        )
        db.add(analysis)
        db.flush()
        wav = register_artefact(
            db,
            store,
            decoded.wav,
            media_type="audio/wav",
            role="working_copy",
            created_by="audio-input/1",
            analysis_id=analysis.id,
        )
        mapping = register_artefact(
            db,
            store,
            canonical_json(decoded.mapping),
            media_type="application/json",
            role="stage_output",
            created_by="audio-input/1",
            analysis_id=analysis.id,
        )
        input_ = {
            "version": "audio-input/1",
            "pcm_sha256": decoded.mapping["pcm_sha256"],
            "sample_rate": 16000,
            "samples": 64000,
        }
        screen = {
            "status": "ok",
            "probability_ai": probability,
            "reference_estimate_ai": probability,
            "decision": direction,
            "scope_supported": True,
            "calibration_sha256": combined_adapter.BUNDLE_SHA256,
            "reason": reason,
            "limitation": combined.NOTE,
            "checked_windows": [{"start_sample": 0, "end_sample": 64000, "start_s": 0.0, "end_s": 4.0}],
            "checked_source_ranges": source_ranges(decoded.mapping, 0, 64000),
        }
        entry = {
            "detector_id": combined_adapter.DETECTOR_ID,
            "status": "ok",
            "family": "voice",
            "exhibit_ids": [],
            "details": {"audio_input": input_, "audio_screening": screen},
            "unit_scores": [
                {
                    "unit_ref": "window:0",
                    "raw": probability,
                    "raw_kind": "probability",
                    "quality": {"start_sample": 0, "end_sample": 64000, "start_s": 0.0, "end_s": 4.0},
                }
            ],
        }
        raw = register_artefact(
            db,
            store,
            canonical_json(entry),
            media_type="application/json",
            role="raw_output",
            created_by="authored-software-fixture",
            analysis_id=analysis.id,
        )
        sid = combined_adapter.DETECTOR_ID + ":global"
        signal = {
            "id": sid,
            "detector_id": combined_adapter.DETECTOR_ID,
            "detector_version": "1.0.0",
            "display_id": "S01",
            "name": "Combined speech inspection",
            "family": "voice",
            "category": "model",
            "modality": "audio",
            "status": "ok",
            "scope": "global",
            "scope_ref": "global",
            "raw_score": probability,
            "observed": f"Authored software-test reference estimate {probability:g}; no model inference.",
            "raw_kind": "probability",
            "model_revisions": {},
            "exhibit_ids": [],
            "details": entry["details"],
        }
        monkeypatch.setattr(audio_screening.video, "synthesis_signals", lambda *args: [copy.deepcopy(signal)])
        monkeypatch.setattr(audio_screening.video, "attach_temporal", lambda *args: None)
        cfg = combined_adapter.CalibrationReference()
        monkeypatch.setattr(
            cfg, "load", lambda: None
        )  # Scores are fixtures; no private model or calibration is evaluated.
        ctx = SimpleNamespace(
            outputs={"probe": {"facts": {"duration_s": 4.0}}},
            sessions=lambda: nullcontext(db),
            store=store,
            analysis_id=analysis.id,
        )
        fused = audio_screening.fuse(ctx, configuration=cfg)
        result = AnalysisResultV2.model_validate(
            {
                "analysis_id": analysis.id,
                "kind": "audio",
                "input": {
                    "sha256": source.sha256,
                    "size_bytes": source.size_bytes,
                    "mime_sniffed": "audio/wav",
                    "filename": source_path.name,
                },
                "manifest": {
                    "veritas_version": "2.1.1",
                    "pipeline_version": "authored-export-software-test",
                    "profile": "full",
                    "sampling": {
                        "audio_evidence": {
                            "input": {**input_, "wav_sha256": wav.sha256, "mapping_sha256": mapping.sha256},
                            "checks": [
                                {"detector_id": entry["detector_id"], "raw_output_sha256": raw.sha256}
                            ],
                        }
                    },
                },
                **{
                    key: fused[key]
                    for key in ("decision", "signals", "condition_profile", "calibration", "assessments")
                },
                "caveats": [
                    "AUTHORED SOFTWARE TEST. Detector scores are fixtures; no model inference or accuracy test was performed.",
                    *fused["caveats"],
                ],
                "findings": [
                    {
                        "id": "F1",
                        "family": "A1_voice",
                        "strength": "weak",
                        "direction": "for",
                        "review_priority": "info",
                        "rank": 1,
                        "title": "Speech screening estimate",
                        "statement": screen["reason"] + " " + combined.NOTE,
                        "evidence_ids": [sid],
                    }
                ],
            }
        ).model_dump(mode="json")
        analysis.result_hash = canonical_hash(result)
        store.put_bytes(canonical_json(result))
        evidence = build_packet_set(db, analysis, store, result)
        original = specimen("audio", "standard")
        authored_at = datetime(2026, 11, 9, tzinfo=UTC)
        frozen = original.model_copy(
            update={
                "result": result,
                "evidence": evidence,
                "case": None,
                "custody": [],
                "annotations": [],
                "actor_names": {},
                "model_cards": [],
                "revisions": [],
                "detector_names": {},
                "custody_verification": None,
                "report": original.report.model_copy(
                    update={
                        "id": "authored-combined-speech-software-report",
                        "display_id": "AUTHORED-SOFTWARE-TEST-055",
                        "revision": 1,
                        "requested_at": authored_at,
                        "requested_by": "software-qa",
                        "requested_by_name": "Chipmunk Verify",
                        "supersedes_report_id": None,
                        "revision_reason": None,
                    }
                ),
                "analysis": original.analysis.model_copy(
                    update={
                        "id": analysis.id,
                        "revision": 1,
                        "created_at": authored_at,
                        "finished_at": authored_at,
                        "supersedes_id": None,
                        "supersedes_revision": None,
                        "mode": None,
                        "coverage": snapshot.SnapshotCoverage(detectors_total=1, by_status={"ok": 1}),
                    }
                ),
                "review": original.review.model_copy(
                    update={"state": "machine", "issuable": False, "conclusion": None, "signoffs": []}
                ),
                "media": original.media.model_copy(
                    update={
                        "filename": source_path.name,
                        "size_bytes": source.size_bytes,
                        "sha256": source.sha256,
                        "md5": hashlib.md5(source_path.read_bytes(), usedforsecurity=False).hexdigest(),
                        "received_at": authored_at,
                        "date_acquired": authored_at,
                        "acquisition_method": "Authored tone for software export testing",
                        "custodian_hash_match": None,
                        "custodian_stated_hash": None,
                        "supplied_by": "Software QA",
                        "source": "Authored tone fixture; no neural inference",
                        "intake_notes": "Authored software export test only.",
                    }
                ),
            }
        )
        yield SimpleNamespace(
            db=db, store=store, analysis=analysis, result=result, frozen=frozen, evidence=evidence
        )


def test_json_export_keeps_the_new_reference_and_source_samples(combined_export, monkeypatch):
    from app.api import routes, v2_evidence

    f = combined_export
    monkeypatch.setattr(routes, "_get", lambda _: f.analysis)
    monkeypatch.setattr(routes, "_current_revision", lambda _: None)
    monkeypatch.setattr(routes, "_verified", lambda *args: None)
    monkeypatch.setattr(routes, "get_session", lambda: nullcontext(f.db))
    monkeypatch.setattr(v2_evidence, "_result", lambda *args: f.result)
    monkeypatch.setattr(routes.audit, "record", lambda *args, **kwargs: None)
    response = routes.export_json(
        f.analysis.id, SimpleNamespace(state=SimpleNamespace(principal=SimpleNamespace(id="software-qa")))
    )
    data = json.loads(response.body)
    assert (
        data["customer_assessment"]["direction"] == "ai"
        and data["customer_assessment"]["confidence_pct"] == 55.0
    )
    assert data["customer_assessment"]["finding_assessments"][0]["confidence_pct"] == 55.0
    text = json.dumps(data).lower()
    assert all(
        name not in text for name in ("wavlm", "pella", "microsoft", "coefficients", "variance_factor")
    )
    assert f.evidence["coverage"]["scored_samples"] == 64000
    assert f.evidence["packets"][0]["interval"] == {"start_sample": 0, "end_sample": 64000}
    assert f.evidence["packets"][0]["observations"][0]["unit"] == "calibrated AI reference estimate"
    output = os.environ.get("CHIPMUNK_AUDIO_EXPORT_QA_DIR")
    if output:
        folder = Path(output)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "api-export.json").write_bytes(response.body)


def test_real_pdf_contains_matching_reference_and_waveform(combined_export):
    f = combined_export
    before = snapshot.snapshot_hash(f.frozen)
    rendered = engine.render_pdf(f.frozen)
    text = engine.pdf_text(rendered.pdf)
    assert "55%" in text.replace(" ", "")
    assert "Leaning towards AI-generated speech" in text
    assert "calibrated AI reference estimate" in text
    assert "separately frozen global Platt mapping" in text
    assert "no calibration record is attached" not in text
    assert "clamped to 2–98" not in text
    assert "Platt calibration per input context" not in text
    assert "AUTHORED" in text and "SOFTWARE" in text
    assert "not prepared for court" in text.lower()
    assert "sales@chipmunkrpa.com" in text
    assert all(
        name not in text.lower()
        for name in ("wavlm", "pella", "microsoft", "examiner’s conclusion", "local examiner")
    )
    assert not rendered.refused
    assert snapshot.snapshot_hash(f.frozen) == before
    output = os.environ.get("CHIPMUNK_AUDIO_EXPORT_QA_DIR")
    if output:
        folder = Path(output)
        (folder / "output/pdf").mkdir(parents=True, exist_ok=True)
        (folder / "output/pdf/combined-speech-software-preview.pdf").write_bytes(rendered.pdf)
        (folder / "snapshot.json").write_bytes(snapshot.dump(f.frozen))
        (folder / "evidence.json").write_bytes(canonical_json(f.evidence))
        (folder / "result.json").write_bytes(canonical_json(f.result))
        from app.evidence.service import asset_refs

        for asset in asset_refs(f.evidence).values():
            digest = asset["sha256"]
            data = f.store.read_bytes(digest)
            (folder / "assets").mkdir(exist_ok=True)
            (folder / "assets" / digest).write_bytes(data)
        (folder / "receipt.json").write_text(
            json.dumps(
                {
                    "pages": rendered.page_count,
                    "pdf_sha256": hashlib.sha256(rendered.pdf).hexdigest(),
                    "source_samples": 64000,
                    "confidence_pct": 55.0,
                    "new_neural_calls": 0,
                    "authored_scores": True,
                },
                indent=2,
            )
            + "\n"
        )


@pytest.mark.parametrize("combined_export", [0.55, 0.45, 0.5], indirect=True)
def test_each_reference_direction_has_inspectable_source_support(combined_export):
    from app.evidence.audio_packets import verify_responses
    from app.evidence.video_audio import verify_source_content
    from app.reports import multimedia_evidence
    from app.reports.customer_view import customer_view

    f = combined_export
    packet = f.evidence["packets"][0]
    assert len(f.evidence["packets"]) == 1
    assert packet["interval"] == {"start_sample": 0, "end_sample": 64000}
    assert packet["observations"][0]["response_reference"]["direction"] == (
        "ai"
        if packet["observations"][0]["value"] > 0.5
        else "original"
        if packet["observations"][0]["value"] < 0.5
        else "inconclusive"
    )
    record = f.result["manifest"]["sampling"]["audio_evidence"]
    verify_responses(f.evidence, store=f.store, result=f.result, record=record, binding=record["input"])
    # The same stored audio content can be validated under a video's source
    # authority, including original-leaning and tied responses.
    content = {
        k: v
        for k, v in f.evidence.items()
        if k not in {"schema", "version", "id", "sha256", "analysis_id", "revision", "original_result_hash"}
    }
    channel = {**content, "status": "ready", "reason": None}
    result = copy.deepcopy(f.result)
    result["manifest"]["sampling"]["video_audio_evidence"] = {
        "input": record["input"],
        "voice_checks": record["checks"],
        "source_audio_present": True,
    }
    verify_source_content(channel, result=result, store=f.store, source_sha256=f.analysis.sha256)
    shown = multimedia_evidence.view(f.frozen, store=f.store)
    public = customer_view(f.frozen, {"header_left": "test", "evidence": shown}, lambda x: x)
    inference = next(p for p in public["evidence"]["packets"] if p["id"] == "A01")["customer_inference"]
    assert "Insufficient evidence" not in inference
    assert (
        ("Inconclusive speech tendency" in inference)
        if packet["observations"][0]["value"] == 0.5
        else ("55% reference confidence" in inference)
    )


@pytest.mark.parametrize(
    "change", ["unit", "probability", "direction", "prior", "pin", "scope", "clock", "gap", "missing", "bool"]
)
def test_score_meaning_tampering_is_rejected(combined_export, change):
    from app.evidence.audio_response_binding import validate

    doc = copy.deepcopy(combined_export.evidence)
    window = doc["coverage"]["windows"][0]
    reference = window["response_reference"]
    if change == "unit":
        doc["packets"][0]["observations"][0]["unit"] = "uncalibrated score"
    elif change == "probability":
        window["score"] = 0.1
    elif change == "direction":
        reference["direction"] = "original"
    elif change == "prior":
        reference["reference_prior_ai"] = True
    elif change == "pin":
        reference["calibration_sha256"] = "0" * 64
    elif change == "scope":
        window["end_sample"] = 32000
    elif change == "clock":
        doc["source"]["timeline_time_base"] = "1/8000"
    elif change == "gap":
        doc["source"]["timeline"][0]["end_sample"] = 32000
    elif change == "missing":
        del doc["coverage"]["response_semantics"]
    elif change == "bool":
        window["score"] = True
    with pytest.raises(ValueError):
        validate(doc, required=True)


def test_raw_run_reference_cannot_be_replaced_by_a_different_window(combined_export):
    from app.evidence.audio_packets import PacketIntegrityError, verify_responses

    f = combined_export
    doc = copy.deepcopy(f.evidence)
    doc["coverage"]["windows"][0]["raw_output_sha256"] = "0" * 64
    record = f.result["manifest"]["sampling"]["audio_evidence"]
    with pytest.raises(PacketIntegrityError):
        verify_responses(doc, store=f.store, result=f.result, record=record, binding=record["input"])


def test_signed_public_package_keeps_reference_meaning_without_private_parameters(combined_export, tmp_path):
    import pymupdf

    from app.reports import package, signing, verify
    from app.storage.db import ReportRow

    f = combined_export
    key = signing.instance_key(tmp_path / "authored-test-key.pem")
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((40, 60), "AUTHORED SOFTWARE TEST. No model inference.")
        pdf_bytes = pdf.tobytes()
    pdf_art = register_artefact(
        f.db,
        f.store,
        pdf_bytes,
        media_type="application/pdf",
        role="report",
        created_by="authored-software-test",
        analysis_id=f.analysis.id,
    )
    snap_art = register_artefact(
        f.db,
        f.store,
        snapshot.dump(f.frozen),
        media_type="application/json",
        role="stage_output",
        created_by="authored-software-test",
        analysis_id=f.analysis.id,
    )
    report = ReportRow(
        id=f.frozen.report.id,
        display_id=f.frozen.report.display_id,
        revision=1,
        analysis_id=f.analysis.id,
        analysis_revision=1,
        profile="standard",
        redacted=False,
        review_state="machine",
        issuable=False,
        template_set=engine.TEMPLATE_SET,
        engine="authored-software-test",
        status="done",
        snapshot_hash=snapshot.snapshot_hash(f.frozen),
        snapshot_artefact_id=snap_art.id,
        artefact_id=pdf_art.id,
        sha256=pdf_art.sha256,
        signature=key.sign(pdf_bytes),
        signing_key_id=key.key_id,
        requested_by="software-qa",
        requested_at=datetime(2026, 11, 9, tzinfo=UTC),
        page_count=1,
    )
    f.db.add(report)
    f.db.flush()
    row = package.prepare(
        f.db,
        f.analysis,
        report,
        kind="draft",
        include_original=False,
        created_by="software-qa",
        schema_id=package.PUBLIC_SCHEMA,
    )
    built = package.build(f.db, row, store=f.store, key=key, exported_at=datetime(2026, 11, 9, tzinfo=UTC))
    assert verify.verify_package(built.path, public_key_pem=key.public_pem()).ok
    with zipfile.ZipFile(built.path) as archive:
        evidence = json.loads(archive.read("evidence.json"))
        bundle = json.loads(archive.read("bundle.json"))
        public = json.loads(archive.read("result.json"))
        assert evidence["packets"][0]["observations"][0]["unit"] == "calibrated AI reference estimate"
        assert public["decision"]["probability_ai_calibrated"] == 0.55
        assert bundle["verification_scope"]["confidence_recalculated"] is False
        assert "coefficients" not in json.dumps(public).lower()
        assert all(name not in json.dumps(public).lower() for name in ("pella", "wavlm", "microsoft"))


def test_only_published_voice_checks_enter_audio_and_video_evidence(combined_export, monkeypatch):
    from app.engine.stages import audio, common
    from app.evidence import video_audio

    f = combined_export
    result = AnalysisResultV2.model_validate(f.result)
    primary = {
        "detector_id": combined_adapter.DETECTOR_ID,
        "status": "ok",
        "family": "voice",
        "run_id": "selected",
    }
    comparator = {**primary, "detector_id": "aud.pella_v2", "run_id": "comparator"}
    raw_sha = f.result["manifest"]["sampling"]["audio_evidence"]["checks"][0]["raw_output_sha256"]
    looked_up = []

    def get_run(_model, run_id):
        looked_up.append(run_id)
        assert run_id == "selected"
        return SimpleNamespace(analysis_id=f.analysis.id, raw_output_artefact_id="saved-run")

    ctx = SimpleNamespace(
        analysis_id=f.analysis.id,
        sessions=lambda: nullcontext(SimpleNamespace(get=get_run)),
        outputs={
            "decode": {
                "evidence_input": {"samples": 64000},
                "audio": {"status": "analysed", "samples": 64000},
            },
            "fuse": {"v1": {}},
        },
    )
    monkeypatch.setattr(common, "entries", lambda *args: [comparator, primary])
    monkeypatch.setattr(common, "build_result", lambda *args, **kwargs: result)
    monkeypatch.setattr(common, "publish", lambda _ctx, result, *args, **kwargs: result)
    monkeypatch.setattr(audio, "get_artefact", lambda *args, **kwargs: SimpleNamespace(sha256=raw_sha))
    monkeypatch.setattr(video_audio, "get_artefact", lambda *args, **kwargs: SimpleNamespace(sha256=raw_sha))
    published = audio.publish(ctx)
    standalone = published.manifest.sampling["audio_evidence"]["checks"]
    nested = video_audio.published_record(ctx, f.result["signals"])["voice_checks"]
    assert (
        standalone == nested == [{"detector_id": combined_adapter.DETECTOR_ID, "raw_output_sha256": raw_sha}]
    )
    assert looked_up == ["selected", "selected"]


@pytest.mark.parametrize(
    "enabled,samples,expected",
    [(True, 64000, ["new", "visual"]), (False, 64000, ["old", "visual"]), (True, 80000, ["old", "visual"])],
)
def test_video_speech_choice_uses_configuration_and_measured_scope(monkeypatch, enabled, samples, expected):
    from app.engine.stages.video import select_speech_signals

    monkeypatch.setattr(settings, "audio_detector_mode", "candidate")
    monkeypatch.setattr(settings, "audio_combined_enabled", enabled)
    ctx = SimpleNamespace(outputs={"decode": {"audio": {"samples": samples}}})
    signals = [
        {"id": "old", "family": "voice", "detector_id": "aud.pella_v2", "raw_score": 0.99},
        {"id": "new", "family": "voice", "detector_id": combined_adapter.DETECTOR_ID, "raw_score": 0.01},
        {"id": "visual", "family": "learned-face", "detector_id": "visual"},
    ]
    assert [s["id"] for s in select_speech_signals(ctx, signals)] == expected
    # An unavailable selected method must not silently fall back to the score
    # that happens to be present.
    if enabled and samples == 64000:
        assert [s["id"] for s in select_speech_signals(ctx, [signals[0], signals[2]])] == ["visual"]
