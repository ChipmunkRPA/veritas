from __future__ import annotations

import io
import json
import uuid
from copy import deepcopy
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from test_document_packets import packet_factory as packet_factory
from test_stripe_billing import billing as billing
from test_stripe_billing import counts, event_for, signed

from app.api.public_projection import project
from app.billing import purchase_notice
from app.core.canonical import canonical_hash, canonical_json
from app.detectors.text import extract as extraction
from app.engine import pipelines
from app.engine.customer_assessment import customer_assessment
from app.engine.stages import document as stage
from app.evidence import document_word_visibility as visibility
from app.evidence.document_packets import build_packet_set, load_packet_set, packet_hash
from app.reports import engine, multimedia_evidence, public_package, public_verify, verify_document_source
from app.reports.specimens import specimen
from app.storage.db import Analysis
from app.storage.objects import ObjectIntegrityError


@pytest.fixture
def hidden_docx(tmp_path):
    path = tmp_path / "visibility.docx"
    doc = Document()
    paragraph = doc.add_paragraph("Visible 🧪 text. ")
    hyperlink = OxmlElement("w:hyperlink")
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    marker = OxmlElement("w:vanish")
    marker.set(qn("w:val"), "on")
    properties.append(marker)
    run.append(properties)
    hidden = OxmlElement("w:t")
    hidden.text = "Hidden link wording."
    run.append(hidden)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)
    paragraph.add_run(" This visible sentence has sufficient context.").font.hidden = False
    table_paragraph = doc.add_table(1, 1).cell(0, 0).paragraphs[0]
    table_paragraph.add_run("This table contains hidden writing.").font.hidden = True
    doc.add_paragraph("   ").runs[0].font.hidden = True
    doc.sections[0].header.paragraphs[0].add_run("Outside scope.").font.hidden = True
    doc.save(path)
    return path


def test_local_word_visibility_keeps_exact_coordinates_and_ignores_explicit_off(hidden_docx):
    doc = extraction.extract(hidden_docx, ".docx")
    record = doc.meta[visibility.KEY]
    assert record["checked_paragraphs"] == 2
    assert record["hidden_runs"] == 2
    prefix = len("Visible 🧪 text. ")
    assert record["locations"] == [
        {"body_path": [0], "body_paragraph": 1, "run": 1, "source_start": prefix, "source_end": prefix + len("Hidden link wording.")},
        {"body_path": [1, "row", 0, "cell", 0, 1], "body_paragraph": None, "run": 0, "source_start": 0,
         "source_end": len("This table contains hidden writing.")},
    ]
    assert "Hidden link wording." in doc.text
    assert record["authorship_confidence_contribution"] == 0
    decoded = {"meta": doc.meta, "word_tracked_changes": doc.word_review}
    assert stage.current_decode_output(decoded)
    old_meta = {key: value for key, value in doc.meta.items() if key != visibility.KEY}
    assert not stage.current_decode_output({**decoded, "meta": old_meta})
    for code in ("ai_supported", "no_ai_indicators"):
        decision = {"code": code, "verbal": "Fixture tendency", "reason": "Fixture",
                    "band": "uncertain", "confidence": "medium", "fusion_mode": "fallback",
                    "probability_ai_calibrated": .96 if code == "ai_supported" else .04}
        report = stage.enforce_extraction_scope({"decision": decision, "signals": [
            {"modality": "text", "scope": "global", "calibrated_probability": .96},
        ]}, doc)
        assessment = customer_assessment({**report, "kind": "document"})
        assert assessment["headline"] == visibility.HEADLINE
        assert assessment["direction"] == "unavailable" and assessment["confidence_pct"] is None
        assert "2 text runs" in assessment["meaning"]
        assert report["signals"][0]["calibrated_probability"] is None


def test_word_visibility_coordinates_replay_in_evidence_pdf_and_json(packet_factory, hidden_docx, monkeypatch, tmp_path):
    db, store, analysis, result, decoded, _ = packet_factory(".docx", scores=(), source_pdf=hidden_docx)
    record = decoded["meta"][visibility.KEY]
    commitment = canonical_hash(record)
    result["manifest"]["sampling"][visibility.COMMITMENT] = commitment
    analysis.result_hash = store.put_bytes(canonical_json(result)).sha256
    evidence = build_packet_set(db, analysis, store, result)
    assert load_packet_set(db, analysis, store) == evidence
    assert evidence["coverage"][visibility.KEY] == record
    assert evidence["source"][visibility.COMMITMENT] == commitment

    for change in ("omit", "relocate"):
        altered = deepcopy(evidence)
        if change == "omit":
            del altered["coverage"][visibility.KEY]
            del altered["source"][visibility.COMMITMENT]
        else:
            altered["coverage"][visibility.KEY]["locations"][0]["source_start"] -= 1
            altered["source"][visibility.COMMITMENT] = canonical_hash(altered["coverage"][visibility.KEY])
        altered["sha256"] = packet_hash(altered)
        with pytest.raises(ObjectIntegrityError, match="recorded source extraction"):
            multimedia_evidence.verify_assets(altered, store)

    snapshot = specimen("document", "standard")
    report_result = {**deepcopy(snapshot.result), "signals": result["signals"]}
    report_result["input"]["sha256"] = analysis.sha256
    report_result["manifest"]["sampling"] = result["manifest"]["sampling"]
    evidence = {**evidence, "analysis_id": snapshot.analysis.id, "revision": snapshot.analysis.revision,
                "original_result_hash": canonical_hash(report_result)}
    evidence["sha256"] = packet_hash(evidence)
    snapshot = snapshot.model_copy(update={"result": report_result, "evidence": evidence,
        "media": snapshot.media.model_copy(update={"sha256": analysis.sha256})})
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    exported = public_package._evidence(snapshot)
    assert exported["coverage"][visibility.KEY] == record
    assert public_package.public_result(report_result)[visibility.COMMITMENT] == commitment
    rendered = engine.render_pdf(snapshot)
    assert not rendered.refused
    pdf_text = " ".join(engine.pdf_text(rendered.pdf).split())
    assert "Word text visibility. Checked 2 extracted paragraphs and found 2 text runs marked hidden." in pdf_text
    assert "Table cell at row 1, column 1, run 1, characters" in pdf_text
    assert "Body path 1 / row / 0 / cell / 0 / 1" in pdf_text
    assert "Hidden link wording." not in pdf_text
    assert "PRIVATE MODEL" not in pdf_text

    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    archived = deepcopy(exported)
    del archived["coverage"][visibility.KEY]
    del archived["source"][visibility.COMMITMENT]
    for payload, expected in [(exported, True), (archived, False)]:
        zip_path = tmp_path / "word-evidence.zip"
        with ZipFile(zip_path, "w") as archive:
            archive.writestr("evidence.json", json.dumps(payload))
        with ZipFile(zip_path) as archive:
            checked = verify_document_source.check(archive, bundle, hidden_docx)
        assert ("Verified direct Word hidden formatting and retained run locations" in checked) is expected


def test_paid_document_pipeline_publishes_word_visibility(billing):
    word = Document()
    paragraph = word.add_paragraph("El informe explica las decisiones del equipo durante la reunión del martes. ")
    paragraph.add_run("Los participantes revisaron los materiales antes de continuar con el trabajo.").font.hidden = True
    content = io.BytesIO()
    word.save(content)
    purchase = billing.client.post(
        "/api/v2/billing/checkout", headers={**billing.headers, "Idempotency-Key": str(uuid.uuid4())},
        data={"kind": "document", "profile": "quick", "risk_accepted": "true",
              "risk_notice_version": purchase_notice.VERSION, "risk_notice_sha256": purchase_notice.SHA256},
        files={"file": ("visibility.docx", content.getvalue(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert purchase.status_code == 201, purchase.text
    assert counts(billing) == (0, 0, 0)
    order = purchase.json()["order_id"]
    assert signed(billing, event_for(billing, order)).status_code == 200
    ident = billing.client.get(f"/api/v2/billing/orders/{order}", headers=billing.headers).json()["analysis_id"]

    # Use unsupported writing language to test the model-free coverage path.
    def unexpected_lease(*args, **kwargs):
        pytest.fail("The unsupported writing language should not lease a model")

    models = SimpleNamespace(default_device="cpu", lease=unexpected_lease)
    outcome = pipelines.run_analysis(ident, profile="quick", sessions=billing.sessions, store=billing.store, models=models)
    assert outcome.status == "done", (outcome.error, outcome.stages)
    with billing.sessions() as db:
        analysis = db.get(Analysis, ident)
        result = json.loads(billing.store.read_bytes(analysis.result_hash))
        evidence = load_packet_set(db, analysis, billing.store)
        assert project(result)["decision"]["verbal"] == visibility.HEADLINE
        assert evidence["coverage"][visibility.KEY]["hidden_runs"] == 1
        assert result["manifest"]["sampling"][visibility.COMMITMENT] == canonical_hash(evidence["coverage"][visibility.KEY])
