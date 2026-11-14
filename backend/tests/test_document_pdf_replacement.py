from __future__ import annotations

import json
import uuid
from copy import deepcopy
from types import SimpleNamespace
from zipfile import ZipFile

import pymupdf
import pytest
from test_document_packets import packet_factory as packet_factory
from test_stripe_billing import billing as billing
from test_stripe_billing import counts, event_for, signed

from app.api.public_projection import project
from app.billing import purchase_notice
from app.core.canonical import canonical_hash
from app.detectors.text.extract import Document, extract
from app.engine import pipelines
from app.engine.customer_assessment import customer_assessment
from app.engine.stages.document import enforce_extraction_scope
from app.evidence import document_pdf_visibility as visibility
from app.evidence.document_packets import build_packet_set, load_packet_set, packet_hash
from app.reports import engine, multimedia_evidence, public_package, public_verify, verify_document_source
from app.reports.specimens import specimen
from app.storage.db import Analysis
from app.storage.objects import ObjectIntegrityError


def write_pdf(path, glyph_text, copy_text, *, rotation=0, ordinary_pages=0, context="English"):
    """Change only the copyable value. The fixture keeps the same visible page bytes when rendered."""
    with pymupdf.open() as pdf:
        for _ in range(ordinary_pages):
            pdf.new_page().insert_text((50, 50), "The first page contains a short description of the delivery.")
        page = pdf.new_page()
        line = ("El informe explica las decisiones del equipo durante la reunión del martes."
                if context == "Spanish" else "This notice records the amount assigned to the original delivery.")
        page.insert_text((40, 50), line)
        page.insert_text((70, 96), glyph_text)
        if copy_text is not None:
            xref = page.get_contents()[-1]
            stream = pdf.xref_stream(xref)
            value = (b"\xfe\xff" + copy_text.encode("utf-16-be")).hex().encode()
            pdf.update_stream(xref, b"/Span << /ActualText <" + value + b"> >> BDC\n" + stream + b"\nEMC")
        page.set_rotation(rotation)
        pdf.save(path)


@pytest.mark.parametrize(("glyphs", "stored", "detected"), [
    ("USD 100.00", None, False),
    ("USD 100.00", "USD 100.00", False),
    ("USD 100.00", "USD  100.00", False),
    ("USD 100.00", "USD 900.00", True),
    ("USD 100.00", "", True),
    ("USD 100.00", "USD 100.00 plus a fee", True),
    ("USD 100.00", "Alternate record." * 75, True),
])
def test_replacement_only_changes_the_copyable_value(tmp_path, glyphs, stored, detected):
    original = tmp_path / "original.pdf"
    variant = tmp_path / "variant.pdf"
    write_pdf(original, glyphs, None, rotation=90)
    write_pdf(variant, glyphs, stored, rotation=90)
    with pymupdf.open(original) as first, pymupdf.open(variant) as second:
        assert first[0].get_pixmap().samples == second[0].get_pixmap().samples
    report = extract(variant, ".pdf").meta[visibility.KEY]
    assert report["replacement_examined_pages"] == [1]
    assert not report["locations"]
    assert bool(report["replacement_differences"]) is detected
    if detected:
        row, = report["replacement_differences"]
        assert row["page"] == 1
        assert row["boxes"] or row["locations_incomplete"]
        assert all(x0 >= 69 and y0 == pytest.approx(84, abs=1) for x0, y0, _, _ in row["boxes"])


def test_significant_normalizes_ligatures_but_keeps_numeric_symbols():
    assert visibility._significant("ﬁnal café") == visibility._significant("final cafe\u0301")
    assert visibility._significant("100 m²") != visibility._significant("100 m2")


@pytest.mark.parametrize(("code", "score"), [("ai_supported", .96), ("no_ai_indicators", .04)])
def test_unchecked_or_different_text_withholds_either_writing_direction(tmp_path, monkeypatch, code, score):
    path = tmp_path / "layer.pdf"
    write_pdf(path, "USD 100.00", "USD 900.00")
    initial = {"decision": {"code": code, "verbal": "Fixture estimate", "reason": "Fixture", "band": "uncertain",
                            "confidence": "medium", "fusion_mode": "fallback", "probability_ai_calibrated": score}}
    differing = extract(path, ".pdf")
    found = customer_assessment({**enforce_extraction_scope(initial, differing), "kind": "document"})
    assert found["headline"] == visibility.REPLACEMENT_HEADLINE and found["confidence_pct"] is None

    monkeypatch.delattr(pymupdf, "TEXT_IGNORE_ACTUALTEXT")
    unavailable = extract(path, ".pdf")
    assert unavailable.meta[visibility.KEY]["replacement_examined_pages"] == []
    limited = customer_assessment({**enforce_extraction_scope(initial, unavailable), "kind": "document"})
    assert limited["headline"] == visibility.INCOMPLETE_HEADLINE and limited["confidence_pct"] is None

    ordinary_meta = {**differing.meta}
    del ordinary_meta[visibility.KEY]
    assert enforce_extraction_scope(initial, Document(differing.text, [], ordinary_meta)) == initial


def test_page_coordinates_export_and_replay(tmp_path, packet_factory, monkeypatch):
    path = tmp_path / "replacement.pdf"
    write_pdf(path, "USD 100.00", "USD 900.00", rotation=90, ordinary_pages=13)
    db, store, analysis, result, _, _ = packet_factory(".pdf", scores=(), source_pdf=path)
    evidence = build_packet_set(db, analysis, store, result)
    review = evidence["coverage"][visibility.KEY]
    assert review["replacement_examined_pages"] == list(range(1, 15))
    assert review["authorship_confidence_contribution"] == 0
    row, = review["replacement_differences"]
    assert row["page"] == 14 and not row["locations_incomplete"]
    assert len(row["boxes"]) == 1
    assert load_packet_set(db, analysis, store) == evidence

    forged = deepcopy(evidence)
    forged["coverage"][visibility.KEY]["replacement_differences"][0]["boxes"][0][0] += .1
    forged["sha256"] = packet_hash(forged)
    with pytest.raises(ObjectIntegrityError, match="recorded source extraction"):
        multimedia_evidence.verify_assets(forged, store)

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
    export = public_package._evidence(snapshot)
    assert export["coverage"][visibility.KEY] == review
    rendered = engine.render_pdf(snapshot)
    assert not rendered.refused
    pdf_text = " ".join(engine.pdf_text(rendered.pdf).split())
    assert "Page 14. Copyable text differs from glyph text." in pdf_text
    assert "unrotated PDF points" in pdf_text
    assert "zero AI-authorship weight" in pdf_text

    commitment = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
                  "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    zip_path = tmp_path / "evidence.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("evidence.json", json.dumps(export))
    with ZipFile(zip_path) as archive:
        assert "Verified the PDF copyable-text comparison and retained page locations." in verify_document_source.check(
            archive, commitment, path)


def test_paid_analysis_records_the_discrepancy_instead_of_a_writing_verdict(billing, tmp_path):
    path = tmp_path / "different-amount.pdf"
    write_pdf(path, "Importe de la entrega: 100 euros.", "Importe de la entrega: 900 euros.", context="Spanish")
    purchase = billing.client.post(
        "/api/v2/billing/checkout", headers={**billing.headers, "Idempotency-Key": str(uuid.uuid4())},
        data={"kind": "document", "profile": "quick", "risk_accepted": "true",
              "risk_notice_version": purchase_notice.VERSION, "risk_notice_sha256": purchase_notice.SHA256},
        files={"file": (path.name, path.read_bytes(), "application/pdf")},
    )
    assert purchase.status_code == 201, purchase.text
    assert counts(billing) == (0, 0, 0)
    order = purchase.json()["order_id"]
    assert signed(billing, event_for(billing, order)).status_code == 200
    ident = billing.client.get(f"/api/v2/billing/orders/{order}", headers=billing.headers).json()["analysis_id"]

    def unexpected_lease(*args, **kwargs):
        pytest.fail("The unsupported writing language should not lease a model")

    models = SimpleNamespace(default_device="cpu", lease=unexpected_lease)
    outcome = pipelines.run_analysis(ident, profile="quick", sessions=billing.sessions, store=billing.store, models=models)
    assert outcome.status == "done", (outcome.error, outcome.stages)
    with billing.sessions() as db:
        analysis = db.get(Analysis, ident)
        result = json.loads(billing.store.read_bytes(analysis.result_hash))
        evidence = load_packet_set(db, analysis, billing.store)
        assert project(result)["decision"]["verbal"] == visibility.REPLACEMENT_HEADLINE
        assert project(result)["decision"]["probability_ai_calibrated"] is None
        assert evidence["coverage"][visibility.KEY]["replacement_differences"][0]["page"] == 1
