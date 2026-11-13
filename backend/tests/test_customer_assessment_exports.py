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


def test_video_report_describes_limited_timing_coverage():
    from app.reports import engine
    from app.reports.public_view import public_view
    from app.reports.view import page_view

    snapshot, expected = headline_snapshot("real video excludes checked voice and timing from visual percentage")
    signal = next(row for row in snapshot.result["signals"] if row["family"] == "av-sync")
    signal["metrics"] = {"windows": 1, "longest_measured_run": 1, "steps": 0, "collapses": 0}
    view = public_view(snapshot, page_view(snapshot, images={}))
    assert view["customer_assessment"]["confidence_pct"] == expected["confidence_pct"]
    assert "too few measured speech windows" in view["customer_assessment"]["scope_note"]
    assert "too few measured speech windows" in engine.render_html(snapshot, embed_fonts=False)


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
