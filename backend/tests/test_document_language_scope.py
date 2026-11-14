from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from email.message import EmailMessage
from types import SimpleNamespace

import pymupdf
import pytest
from test_document_packets import packet_factory as packet_factory
from test_stripe_billing import billing as billing
from test_stripe_billing import event_for, signed

from app.api.public_projection import project
from app.billing import purchase_notice
from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.text import extract, language_scope, pipeline
from app.detectors.text.adapters import agreement
from app.detectors.text.adapters.learned import BinocularsDetector, FourClassTextDetector
from app.detectors.text.adapters.stylometry import StylometryDetector
from app.detectors.text.extract import from_string
from app.detectors.text.segment import Sentence, segment
from app.engine import pipelines
from app.engine.customer_assessment import customer_assessment
from app.engine.detector import Profile
from app.engine.stages import document
from app.evidence.document_packets import build_packet_set
from app.reports import engine, public_package
from app.reports import snapshot as reports_snapshot
from app.reports.specimens import specimen
from app.storage.db import Analysis

ENGLISH = (
    "The review team collected the field measurements before the public meeting. Several pages include questions that "
    "the original team can answer from the laboratory notebooks. We will update the report after completing the source review."
)
SPANISH = (
    "El laboratorio entregó los registros de la estación antes de la reunión del viernes. Durante la revisión, el equipo "
    "encontró varias observaciones que todavía deben compararse con las notas originales del archivo. El informe completo "
    "se publicará cuando termine esa comprobación."
)
FRENCH = (
    "Le comité a examiné l'ensemble des observations recueillies pendant les trois premières semaines du programme. "
    "Plusieurs dossiers contenaient des renseignements supplémentaires que les responsables pourront vérifier avant la "
    "publication du rapport. Une nouvelle réunion aura lieu dans la salle principale du bâtiment."
)
UNKNOWN = "123 456 789 012 345 678 901 234 567 890 112 223 334 445 556 667 778 889."


@pytest.mark.parametrize(("text", "expected"), [
    (SPANISH, "unsupported"), (FRENCH, "unsupported"), (ENGLISH, "english"),
    (f"{ENGLISH}\n\n{SPANISH}", "mixed"), (f"{FRENCH} {ENGLISH}", "mixed"), (UNKNOWN, "unknown"),
])
def test_actual_local_language_reading(text, expected):
    doc = from_string(text)
    assert language_scope.classify(doc.text, segment(doc)) == expected


def test_language_reads_are_bounded_across_large_documents(monkeypatch):
    calls = []

    def reading(sample):
        calls.append(sample)
        return {"code": "en", "confidence": 1}

    monkeypatch.setattr(language_scope.extract, "_ocr_language", reading)
    passages = [f"{i}. {ENGLISH}" for i in range(450)]
    text = "\n\n".join(passages)
    sentences = [Sentence(i, 0, len(passage), passage, i, 35) for i, passage in enumerate(passages)]
    assert language_scope.classify(text, sentences) == language_scope.ENGLISH
    assert len(calls) == language_scope.MAX_LOCAL_PASSAGES + 1
    assert calls[0] == text
    assert calls[1] == passages[0] and calls[-1] == passages[-1]
    assert all(len(passage) <= language_scope.MAX_PASSAGE_CHARS for passage in calls[1:])


def test_shared_language_identifier_is_safe_under_parallel_use(monkeypatch):
    monkeypatch.setattr(extract, "_LANGUAGE_IDENTIFIER", None)
    samples = [(SPANISH, "es"), (FRENCH, "fr"), (ENGLISH, "en")] * 8
    with ThreadPoolExecutor(max_workers=4) as pool:
        readings = list(pool.map(extract._ocr_language, (text for text, _ in samples)))
    assert [reading["code"] for reading in readings] == [code for _, code in samples]


def forbid(*_args, **_kwargs):
    pytest.fail("Writing inference must not load a model for this input")


@pytest.mark.parametrize(("text", "scope"), [
    (SPANISH, "unsupported"), (FRENCH, "unsupported"), (f"{ENGLISH} {SPANISH}", "mixed"), (UNKNOWN, "unknown"),
])
def test_direct_text_returns_unscored_sentences(monkeypatch, text, scope):
    monkeypatch.setattr(pipeline, "score_document", forbid)
    result = pipeline.analyze_text(text)
    headline, reason = language_scope.MESSAGES[scope]
    assert result["document"][language_scope.KEY] == scope
    assert result["verdict"]["summary"] == headline
    assert result["verdict"]["reasoning"] == [reason]
    assert result["verdict"]["probability_ai"] is None
    assert result["stats"]["scored_sentences"] == 0
    assert result["signals"] == result["artifacts"] == []
    assert result["sentences"] and all(row["band"] == "skipped" and row["p_ai"] is None for row in result["sentences"])


@pytest.mark.parametrize("ext", [".eml", ".pdf"])
def test_legacy_file_entry_point_skips_multilingual_writing(tmp_path, monkeypatch, ext):
    path = tmp_path / ("input" + ext)
    if ext == ".eml":
        message = EmailMessage()
        message.set_content(FRENCH)
        path.write_bytes(message.as_bytes())
    else:
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            assert page.insert_textbox(pymupdf.Rect(25, 30, 450, 550), FRENCH, fontsize=12) > 0
            pdf.save(path)
    monkeypatch.setattr(pipeline, "score_document", forbid)
    result = pipeline.analyze_document(path, ext, tmp_path)
    assert result["document"][language_scope.KEY] == "unsupported"
    assert result["verdict"]["probability_ai"] is None
    assert not (tmp_path / "sentence_scores.png").exists()


def test_english_control_reaches_existing_legacy_scoring(monkeypatch):
    called = []

    def score(doc, sentences):
        called.append(doc.text)
        return pipeline.DocumentScores(n_sentences=len(sentences)), {}

    monkeypatch.setattr(pipeline, "score_document", score)
    pipeline.analyze_text(ENGLISH)
    assert called == [ENGLISH]


def test_stage_guards_learned_checks_and_reports_unassessed_scope(packet_factory, monkeypatch):
    db, store, analysis, source_result, decoded, _ = packet_factory(".eml", scores=(), text=SPANISH)
    check_statuses = []
    monkeypatch.setattr(agreement, "enabled", lambda: True)
    monkeypatch.setattr(agreement, "record", forbid)

    def detect(_ctx, _modalities, *, outputs, **_unused):
        run_ctx = SimpleNamespace(outputs=outputs)
        for check in (FourClassTextDetector(), BinocularsDetector(), StylometryDetector()):
            applicability = check.applicable(run_ctx)
            assert not applicability.ok
            check_statuses.append(applicability.reason)
        return {"detectors": [], "artefacts": [], "exhibits": []}

    monkeypatch.setattr(document.common, "run_detectors", detect)
    ctx = SimpleNamespace(profile=Profile.FULL, state={}, check=lambda _: None,
                          outputs={"decode": decoded, "condition": {"profile": {"slice_id": "document/fixture"}}})
    ctx.outputs["detect"] = document.detect(ctx)
    fused = document.fuse(ctx)
    assert document.current_detect_output(ctx.outputs["detect"])
    assert not document.current_detect_output({"detectors": []})
    headline, reason = language_scope.MESSAGES["unsupported"]
    assert check_statuses == [reason] * 3
    assert fused["decision"]["verbal"] == headline
    assert fused["decision"]["probability_ai_uncalibrated"] is None
    assert fused["decision"]["probability_ai_calibrated"] is None
    assert fused["v1"]["stats"]["scored_sentences"] == 0
    assert fused["signals"] == fused["calibration"] == []
    snapshot = specimen("document", "standard")
    frozen = {**snapshot.result, **{k: v for k, v in fused.items() if k != "v1"},
              "findings": [], "exhibits": []}
    frozen["input"]["sha256"] = analysis.sha256
    frozen["manifest"]["sampling"].update(source_result["manifest"]["sampling"])
    snapshot.result.update(frozen)
    view = customer_assessment(snapshot.result)
    assert view["headline"] == headline and view["meaning"] == reason and view["confidence_pct"] is None
    assert view["scope_label"] == language_scope.ASSESSMENT_SCOPE
    assert "No passage received an authorship score" in view["scope_note"]
    assert project(snapshot.result)["decision"]["verbal"] == headline
    analysis.result_hash = canonical_hash(snapshot.result)
    store.put_bytes(canonical_json(snapshot.result))
    evidence = build_packet_set(db, analysis, store, snapshot.result)
    assert evidence["packets"] == []
    assert evidence["coverage"]["scored_windows"] == 0
    assert public_package.public_result(snapshot.result)["decision"]["reason"] == reason
    snapshot = snapshot.model_copy(update={"evidence_packets": evidence})
    report = engine.render_pdf(snapshot)
    assert not report.refused
    assert reason in " ".join(engine.pdf_text(report.pdf).split())


def test_new_paid_document_publishes_without_loading_writing_models(billing):
    purchase = billing.client.post(
        "/api/v2/billing/checkout", headers={**billing.headers, "Idempotency-Key": str(uuid.uuid4())},
        data={"kind": "document", "profile": "quick", "risk_accepted": "true",
              "risk_notice_version": purchase_notice.VERSION, "risk_notice_sha256": purchase_notice.SHA256},
        files={"file": ("article.txt", SPANISH.encode(), "text/plain")},
    )
    assert purchase.status_code == 201, purchase.text
    order = purchase.json()["order_id"]
    assert signed(billing, event_for(billing, order)).status_code == 200
    ident = billing.client.get(f"/api/v2/billing/orders/{order}", headers=billing.headers).json()["analysis_id"]
    models = SimpleNamespace(default_device="cpu", lease=forbid)
    outcome = pipelines.run_analysis(ident, profile="quick", sessions=billing.sessions, store=billing.store, models=models)
    assert outcome.status == "done", (outcome.error, outcome.stages)
    detect = outcome.outputs["detect"]
    assert detect[language_scope.KEY] == "unsupported"
    assert detect["detectors"] and all(r["status"] == "not_applicable" for r in detect["detectors"])

    with billing.sessions() as db:
        analysis = db.get(Analysis, ident)
        result = billing.store.read_bytes(analysis.result_hash)
        raw = document.AnalysisResultV2.model_validate_json(result).model_dump(mode="json")
        headline, reason = language_scope.MESSAGES["unsupported"]
        assert raw["decision"]["verbal"] == headline
        assert raw["decision"]["code"] == "not_assessable"
        assert analysis.probability_ai is None
        frozen = reports_snapshot.build(db, analysis, raw, report_id=str(uuid.uuid4()), revision=1, profile="standard",
            redacted=False, supersedes_report_id=None, revision_reason=None, requested_at=datetime.now(UTC),
            requested_by="authored-language-test", template_set=engine.TEMPLATE_SET,
            veritas_version=settings.version, store=billing.store)
        report = engine.render_pdf(frozen)
        assert not report.refused
        assert reason in " ".join(engine.pdf_text(report.pdf).split())
        assert public_package.public_result(raw)["decision"]["reason"] == reason
