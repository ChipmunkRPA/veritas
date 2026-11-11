"""The customer interpretation is exported separately from immutable evidence."""
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api import routes
from app.engine.customer_assessment import customer_assessment


@pytest.mark.parametrize("schema_version", [1, 2])
@pytest.mark.parametrize("code,direction,confidence", [("inconclusive", "inconclusive", None), ("no_ai_indicators", "original", 88)])
def test_export_respects_the_recorded_decision(monkeypatch, schema_version, code, direction, confidence):
    stored = {"decision": {"code": code, "fusion_mode": "artefact", "probability_ai_calibrated": 0.12,
                           "ci90": [0.10, 0.17]}, "condition_profile": {"within_validated_envelope": True},
              "findings": [{"id": "F1", "family": "F4_localised_manipulation", "direction": "against",
                            "strength": "weak", "scope": {"kind": "bbox"}, "evidence_ids": ["missing-check"]}]}
    before = copy.deepcopy(stored)
    compatible = stored if schema_version == 1 else {"verdict": {"p_ai": 0.99}}
    analysis = SimpleNamespace(id="test", status="done", result_schema_version=schema_version, result=compatible,
                               to_summary=lambda: {"id": "test", "result_hash": "original-hash"})
    if schema_version == 2:
        from app.api import v2_evidence

        @contextmanager
        def sessions():
            yield SimpleNamespace(get=lambda *_: analysis)

        monkeypatch.setattr(routes, "get_session", sessions)
        monkeypatch.setattr(v2_evidence, "_result", lambda *_: stored)
    monkeypatch.setattr(routes, "_get", lambda _: analysis)
    monkeypatch.setattr(routes, "_current_revision", lambda _: None)
    monkeypatch.setattr(routes, "_verified", lambda *_: None)
    events = []
    monkeypatch.setattr(routes.audit, "record", lambda *args, **kwargs: events.append(kwargs))
    response = routes.export_json("test", SimpleNamespace(state=SimpleNamespace(principal=SimpleNamespace(id="local"))))
    data = json.loads(response.body)
    assert data["customer_assessment"] == customer_assessment({**stored, "status": "done"})
    assert data["customer_assessment"]["direction"] == direction
    assert data["customer_assessment"]["confidence_pct"] == confidence
    assert "threshold_pct" not in data["customer_assessment"]
    assert "threshold_pct" not in data["customer_assessment"]["service_policy"]
    finding = data["customer_assessment"]["finding_assessments"][0]
    assert finding["finding_id"] == "F1" and finding["scope_label"] == "Selected image region"
    assert finding["direction"] == "inconclusive" and finding["confidence_pct"] is None
    # The overall result cannot supply confidence to a source region with no available check.
    assert data["customer_assessment"]["service_policy"]["court_use"] == "not_prepared_for_court_use"
    assert data["customer_assessment"]["service_policy"]["separate_expert_engagement_required"] is True
    from app.api.public_projection import project

    assert data["result"] == project(compatible)
    assert data["result_hash"] == "original-hash"
    assert stored == before
    assert events[0]["after_hash"] == hashlib.sha256(response.body).hexdigest()


def headline_snapshot(name):
    from app.reports.specimens import specimen
    from app.schemas.evidence import SignalV2

    cases = json.loads((Path(__file__).parent / "fixtures/customer_assessment_cases.json").read_text())
    case = next(row for row in cases if row["name"] == name)
    source = case["input"]
    original = specimen(source.get("kind", "image"), "summary")
    result = copy.deepcopy(original.result)
    result["decision"].update(source["decision"])
    result["decision"]["probability_ai_uncalibrated"] = None
    result["condition_profile"] = source["condition_profile"]
    result["signals"] = [SignalV2(**{
        "detector_version": "1.0", "name": "Recorded content check", "family": "learned-whole-image", "category": "model",
        "detector_id": signal["id"], **signal,
    }).model_dump(mode="json") for signal in source.get("signals", [])]
    return original.model_copy(update={"result": result}), case["expected"]


@pytest.mark.parametrize("name", ["image inconclusive at 0.01 retains recorded outcome",
                                 "skipped audio checks retain visual-only scope",
                                 "failed contributing visual check prevents original tendency"])
def test_report_presentation_matches_api_headline_and_preserves_evidence(name):
    from app.reports import engine
    from app.reports.public_view import public_view
    from app.reports.snapshot import snapshot_hash
    from app.reports.view import page_view

    snapshot, expected = headline_snapshot(name)
    before = snapshot_hash(snapshot)
    displayed = public_view(snapshot, page_view(snapshot, images={}))
    assert displayed["headline"] == displayed["cover_headline"] == expected["headline"]
    assert displayed["customer_assessment"]["direction"] == expected["direction"]
    assert displayed["customer_assessment"]["confidence_pct"] == expected["confidence_pct"]
    html = engine.render_html(snapshot, embed_fonts=False)
    assert expected["headline"] in html
    if expected["direction"] == "inconclusive":
        assert "Leaning towards an original file" not in html
    else:
        assert "voice/audio checks were skipped" in html
        assert "checked separately" not in displayed["customer_assessment"]["scope_note"]
    assert snapshot_hash(snapshot) == before


def test_short_video_timing_coverage_survives_json_html_and_signed_zip(tmp_path, monkeypatch):
    import zipfile

    import pymupdf
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.core.canonical import canonical_hash, canonical_json
    from app.core.config import settings
    from app.detectors.video.decode import FrameTable
    from app.evidence.service import packet_hash
    from app.reports import engine, package, signing, snapshot, verify
    from app.storage.artefacts import register_artefact
    from app.storage.db import Analysis, Base, ReportRow
    from app.storage.objects import default_store

    frozen, expected = headline_snapshot("real video excludes checked voice and timing from visual percentage")
    result = frozen.result
    result["decision"].update(verbal=expected["headline"], reason="The recorded visual checks supply the estimate.",
                              band="likely_real")
    result["findings"] = []
    detector = "vid.lipsync.landmark_xcorr"
    result["signals"][-1].update(
        id=f"{detector}:track:0", detector_id=detector, scope="track", raw_score=0.0, raw_kind="score",
        metrics={"windows": 1, "peak_median": 0.8, "flagged": 0, "steps": 0, "collapses": 0},
        model_revisions={"private-timing-model": "private-checkpoint"},
    )
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    store = default_store()
    source = store.put_bytes(b"Authored video export fixture")
    table = FrameTable("1/25", tuple(range(74)), (0,), "I" + "P" * 73, 32, 24, 32, 24)
    frame_table = table.to_json()
    probe = store.put_bytes(canonical_json({"frame_table": frame_table}))
    render = store.put_bytes(canonical_json({"frame_anchors": []}))
    bindings = {"frame_table_sha256": canonical_hash(frame_table), "frame_anchor_sha256": canonical_hash([]),
                "timeline_origin_s": 0.0}
    result["manifest"]["sampling"].update(frame_anchor_version="stream-pts-rgb24/1", **bindings)
    result["input"].update(sha256=source.sha256, size_bytes=source.size_bytes)
    original_hash = canonical_hash(result)
    evidence = {
        "schema": "chipmunk.video-evidence/1", "version": "video-evidence-2", "status": "ready",
        "id": "timing-export-evidence", "analysis_id": frozen.analysis.id, "revision": frozen.analysis.revision,
        "original_result_hash": original_hash,
        "source": {"sha256": source.sha256, "time_base": table.time_base, "frame_count": len(table),
                   "width": table.width, "height": table.height, **bindings},
        "source_records": [{"stage": "probe", "sha256": probe.sha256}, {"stage": "render", "sha256": render.sha256}],
        "packets": [],
    }
    evidence["sha256"] = packet_hash(evidence)
    frozen = frozen.model_copy(update={
        "evidence": evidence,
        "media": frozen.media.model_copy(update={"sha256": source.sha256, "size_bytes": source.size_bytes}),
    })
    original_snapshot_hash = snapshot.snapshot_hash(frozen)
    database = create_engine("sqlite://")
    Base.metadata.create_all(database)
    key = signing.instance_key(tmp_path / "test-key.pem")
    expected_note = "Every measured face track had too few eligible speech windows to assess timing changes."

    with Session(database) as db:
        analysis = Analysis(id=frozen.analysis.id, kind="video", filename=frozen.media.filename, content_type="video/mp4",
                            size_bytes=source.size_bytes, sha256=source.sha256, status="done",
                            revision=frozen.analysis.revision, result_schema_version=2, result_hash=original_hash,
                            result_json=json.dumps({"verdict": {"p_ai": 0.99}}))
        db.add(analysis)
        db.flush()
        register_artefact(db, store, canonical_json(result), media_type="application/json", role="stage_output",
                          created_by="publish-result", analysis_id=analysis.id)
        register_artefact(db, store, canonical_json(evidence), media_type="application/json", role="exhibit_payload",
                          created_by="video-evidence-packets/2", analysis_id=analysis.id, artefact_id=evidence["id"])

        @contextmanager
        def sessions():
            yield db

        monkeypatch.setattr(routes, "get_session", sessions)
        monkeypatch.setattr(routes, "_verified", lambda *_: store.verify(source.sha256))
        monkeypatch.setattr(routes.audit, "record", lambda *args, **kwargs: None)
        request = SimpleNamespace(state=SimpleNamespace(principal=SimpleNamespace(id="local")))
        exported = json.loads(routes.export_json(analysis.id, request).body)
        assessment = exported["customer_assessment"]
        assert expected_note in assessment["scope_note"]
        assert assessment["direction"] == expected["direction"]
        assert assessment["confidence_pct"] == expected["confidence_pct"]
        assert exported["result_hash"] == original_hash and exported["evidence"] == evidence
        html = engine.render_html(frozen, embed_fonts=False)
        assert expected_note in html and assessment["headline"] in html

        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((50, 70), "Chipmunk Verify test report")
            pdf_bytes = pdf.tobytes()
        pdf_art = register_artefact(db, store, pdf_bytes, media_type="application/pdf", role="report",
                                    created_by="report-fixture", analysis_id=analysis.id)
        snap_art = register_artefact(db, store, snapshot.dump(frozen), media_type="application/json", role="stage_output",
                                     created_by="report-snapshot", analysis_id=analysis.id)
        report = ReportRow(
            id=frozen.report.id, display_id=frozen.report.display_id, revision=frozen.report.revision,
            analysis_id=analysis.id, analysis_revision=analysis.revision, profile=frozen.report.profile, redacted=False,
            review_state="machine", issuable=False, template_set=frozen.report.template_set, engine="test", status="done",
            snapshot_hash=original_snapshot_hash, snapshot_artefact_id=snap_art.id, artefact_id=pdf_art.id,
            sha256=pdf_art.sha256, signature=key.sign(pdf_bytes), signing_key_id=key.key_id,
            requested_by="test", requested_at=frozen.report.requested_at, page_count=1,
        )
        db.add(report)
        row = package.prepare(db, analysis, report, kind="draft", include_original=False, created_by="test",
                              schema_id=package.PUBLIC_SCHEMA)
        built = package.build(db, row, store=store, key=key, exported_at=frozen.report.requested_at)
        verified = verify.verify_package(built.path, public_key_pem=key.public_pem())
        assert verified.ok, verified.as_dict()
        with zipfile.ZipFile(built.path) as archive:
            public = json.loads(archive.read("result.json"))
            assert public["schema"] == "chipmunk.public-result/1"
            assert expected_note in public["note"]
            assert public["original_result_sha256"] == original_hash
            assert public["decision"]["probability_ai_calibrated"] == result["decision"]["probability_ai_calibrated"]
            assert archive.read("evidence.json") == canonical_json(evidence)
            assert archive.read("report.pdf") == pdf_bytes
            for private in ("private-timing-model", "private-checkpoint"):
                assert private not in json.dumps(public)
        assert snapshot.snapshot_hash(frozen) == original_snapshot_hash
        assert canonical_hash(result) == original_hash
        assert store.read_bytes(original_hash) == canonical_json(result)


@pytest.mark.parametrize("kind", ["image", "video", "audio", "document"])
@pytest.mark.parametrize("profile", ["summary", "standard", "forensic", "newsroom"])
def test_every_standard_report_discloses_screening_scope_and_private_engagement(kind, profile):
    from app.reports import engine
    from app.reports.specimens import specimen

    html = engine.render_html(specimen(kind, profile), embed_fonts=False)
    assert "60 % screening threshold" not in html
    assert "60% screening threshold" not in html
    assert "not prepared for court filing" in html
    assert "Do not rely on it as court evidence" in html
    assert "separate signed engagement" in html
    assert "sales@chipmunkrpa.com" in html
    assert "does not guarantee a conclusion or court admissibility" in html
    assert "Automated deepfake screening is uncertain" in html
    assert "80% customer-facing" not in html


@pytest.mark.parametrize('schema_version', [1, 2])
def test_retired_marker_export_preserves_original_decision_and_adds_current_interpretation(monkeypatch, schema_version):
    stored = {'kind': 'image', 'decision': {'code': 'ai_supported', 'fusion_mode': 'fallback',
              'probability_ai_uncalibrated': .85, 'driving_hypothesis': 'H3_provenance_integrity',
              'provenance_record': {'kind': 'self_declared_ai', 'marker': 'watermark', 'signal_ids': []}}}
    before = copy.deepcopy(stored)
    analysis = SimpleNamespace(id='historical-marker', status='done', result_schema_version=schema_version,
                               result=stored, to_summary=lambda: {'id': 'historical-marker', 'result_hash': 'saved-hash'})
    if schema_version == 2:
        from app.api import v2_evidence

        @contextmanager
        def sessions():
            yield SimpleNamespace(get=lambda *_: analysis)

        monkeypatch.setattr(routes, 'get_session', sessions)
        monkeypatch.setattr(v2_evidence, '_result', lambda *_: stored)
    monkeypatch.setattr(routes, '_get', lambda _: analysis)
    monkeypatch.setattr(routes, '_current_revision', lambda _: None)
    monkeypatch.setattr(routes, '_verified', lambda *_: None)
    monkeypatch.setattr(routes.audit, 'record', lambda *args, **kwargs: None)
    response = routes.export_json('historical-marker', SimpleNamespace(state=SimpleNamespace(principal=SimpleNamespace(id='local'))))
    data = json.loads(response.body)
    assert data['customer_assessment']['direction'] == 'inconclusive'
    assert data['customer_assessment']['confidence_pct'] is None
    assert 'retired rule' in data['customer_assessment']['meaning']
    assert data['result']['decision']['code'] == 'ai_supported' and data['result']['decision']['probability_ai_uncalibrated'] == .85
    assert stored == before
