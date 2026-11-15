from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace
from zipfile import ZipFile

import pymupdf
import pytest
from test_document_packets import packet_factory as packet_factory
from test_public_package_integration import multimedia_package as multimedia_package
from test_stripe_billing import billing as billing
from test_stripe_billing import counts, event_for, signed

from app.api import v2_evidence_packets
from app.api.public_projection import project
from app.billing import purchase_notice
from app.core.canonical import canonical_hash
from app.detectors.text.extract import extract
from app.detectors.text.pipeline import analyze_document_obj
from app.engine import pipelines
from app.engine.customer_assessment import customer_assessment
from app.evidence import document_pdf_visibility as visibility
from app.evidence.document_packets import build_packet_set, load_packet_set, packet_hash
from app.reports import engine, multimedia_evidence, package, public_package, verify
from app.reports.specimens import specimen
from app.storage.db import Analysis
from app.storage.objects import ObjectIntegrityError

PUBLIC_TEXT = "The public delivery record lists the inspection date and explains how the parcels were transferred safely."
COVERED_TEXT = "Private delivery account details"


def write_pdf(path, mode, *, separate_page=False, rotation=0, offset_crop=False):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        if separate_page:
            page.insert_text((55, 55), PUBLIC_TEXT)
            page = pdf.new_page()
        page.insert_text((55, 130), COVERED_TEXT)
        _, y0, _, y1 = page.get_texttrace()[0]["bbox"]
        x0, x1 = 55, 55 + pymupdf.get_text_length(COVERED_TEXT)
        if mode == "one-word":
            x0 += pymupdf.get_text_length("Private ")
            x1 = x0 + pymupdf.get_text_length("delivery")
        elif mode == "partial-glyphs":
            x1 = x0 + 7
        box = pymupdf.Rect(x0 - 2, y0 - 4, x1 + 2, y1 + 4)
        if mode == "applied":
            page.add_redact_annot(box, fill=(0, 0, 0))
            page.apply_redactions()
            page.insert_text((55, 170), PUBLIC_TEXT)
        elif mode == "review-annotation":
            page.add_rect_annot(box)
        else:
            page.draw_rect(box, color=None, fill=(1, 1, 1) if mode == "white" else (0, 0, 0),
                           fill_opacity=.7 if mode == "translucent" else 1, overlay=mode != "behind")
        if mode == "foreground":
            page.insert_text((55, 130), COVERED_TEXT, color=(1, 1, 1))
        if offset_crop:
            page.set_cropbox(pymupdf.Rect(20, 10, page.rect.width - 20, page.rect.height - 10))
        page.set_rotation(rotation)
        pdf.save(path)


@pytest.mark.parametrize(("mode", "rotation", "offset", "flag"), [
    ("black", 0, False, True),
    ("white", 90, True, True),
    ("one-word", 270, True, True),
    ("behind", 0, False, False),
    ("translucent", 0, False, False),
    ("foreground", 0, False, False),
    ("partial-glyphs", 0, False, False),
    ("review-annotation", 0, False, False),
    ("applied", 0, False, False),
])
def test_covered_words_are_distinguished_from_ordinary_page_marks(tmp_path, mode, rotation, offset, flag):
    path = tmp_path / "source.pdf"
    write_pdf(path, mode, separate_page=True, rotation=rotation, offset_crop=offset)
    doc = extract(path, ".pdf", allow_ocr=False)
    report = doc.meta[visibility.KEY]
    assert report["covered_examined_pages"] == [1, 2]
    assert report["authorship_confidence_contribution"] == 0
    assert not report["locations"]
    assert bool(report["covered_locations"]) is flag
    assert PUBLIC_TEXT in doc.text
    if flag:
        assert COVERED_TEXT not in doc.text
        assert [row["page"] for row in doc.source_map] == [1]
        row, = report["covered_locations"]
        region, = row["boxes"]
        assert row["page"] == 2 and row["covered_spans"] == 1
        assert region[1] == pytest.approx(121.4 - (10 if offset else 0), abs=.5)
        if mode == "one-word":
            assert 20 < region[2] - region[0] < 55
    elif mode != "applied":
        assert COVERED_TEXT in doc.text


def test_failed_inspection_is_not_counted_as_examined(tmp_path, monkeypatch):
    path = tmp_path / "source.pdf"
    write_pdf(path, "black")
    monkeypatch.setattr(visibility, "MAX_RECTANGLE_CANDIDATES", 0)
    report = extract(path, ".pdf", allow_ocr=False).meta[visibility.KEY]
    assert report["examined_pages"] == [1]
    assert report["covered_examined_pages"] == []
    assert report["covered_locations"] == []


def test_legacy_document_call_withholds_the_whole_document_assessment(covered_pdf):
    doc = extract(covered_pdf, ".pdf", allow_ocr=False)
    result = analyze_document_obj(doc, None)
    assert not result["signals"]
    assert result["verdict"]["probability_ai"] is None
    assert result["verdict"]["summary"] == visibility.COVERED_HEADLINE
    assert COVERED_TEXT not in result["text"]


@pytest.mark.parametrize("order", ["covered", "visible", "omitted"])
def test_reportlab_producer_and_legacy_extraction(tmp_path, order):
    from reportlab.pdfgen import canvas

    path = tmp_path / "other-producer.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.setFont("Helvetica", 12)
    pdf.drawString(55, 740, PUBLIC_TEXT)
    width = pdf.stringWidth(COVERED_TEXT, "Helvetica", 12)
    if order == "covered":
        pdf.drawString(55, 685, COVERED_TEXT)
    pdf.rect(51, 679, width + 8, 20, stroke=0, fill=1)
    if order == "visible":
        pdf.setFillColorRGB(1, 1, 1)
        pdf.drawString(55, 685, COVERED_TEXT)
    pdf.save()

    old = extract(path, ".pdf", allow_ocr=False, inspect_pdf_covers=False)
    new = extract(path, ".pdf", allow_ocr=False)
    assert old.meta[visibility.KEY]["schema"] == visibility.REPLACEMENT_SCHEMA
    assert (COVERED_TEXT in old.text) is (order != "omitted")
    assert (COVERED_TEXT in new.text) is (order == "visible")
    assert bool(new.meta[visibility.KEY]["covered_locations"]) is (order == "covered")
    if order != "covered":
        assert new.text == old.text and new.source_map == old.source_map


@pytest.fixture
def covered_pdf(tmp_path):
    path = tmp_path / "source.pdf"
    write_pdf(path, "one-word", separate_page=True, rotation=270)
    return path


def test_pdf_and_json_exclude_covered_text(covered_pdf, packet_factory, monkeypatch):
    db, store, analysis, result, decoded, _ = packet_factory(".pdf", scores=(), source_pdf=covered_pdf)
    evidence = build_packet_set(db, analysis, store, result)
    assert COVERED_TEXT not in decoded["text"]
    assert load_packet_set(db, analysis, store) == evidence
    forged = deepcopy(evidence)
    forged["coverage"][visibility.KEY]["covered_locations"][0]["boxes"][0][0] += .1
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
    assert export["coverage"][visibility.KEY] == evidence["coverage"][visibility.KEY]
    assert COVERED_TEXT not in json.dumps(export)
    rendered = engine.render_pdf(snapshot)
    assert not rendered.refused
    pdf_text = " ".join(engine.pdf_text(rendered.pdf).split())
    assert "Page 2. 1 covered copyable text span." in pdf_text
    assert "Excluded from writing checks." in pdf_text and COVERED_TEXT not in pdf_text


@pytest.mark.parametrize("multimedia_package", ["covered_pdf"], indirect=True)
def test_signed_package_replays_covered_text(multimedia_package, covered_pdf):
    f = multimedia_package
    built = package.build(f.db, f.row, store=f.store, key=f.key, exported_at=datetime.now(UTC))
    replay = verify.verify_package(built.path, expect_key_id=f.key.key_id, source_document=covered_pdf)
    assert replay.ok, replay.as_dict()
    assert "Verified the covered PDF text inspection" in replay.checks[-1].detail
    with ZipFile(built.path) as archive:
        evidence = json.loads(archive.read("evidence.json"))
        assert evidence["coverage"][visibility.KEY]["covered_locations"][0]["page"] == 2
        assert COVERED_TEXT not in archive.read("evidence.json").decode()


def test_paid_analysis_preserves_the_finding_when_no_text_can_be_scored(billing, tmp_path, monkeypatch):
    monkeypatch.setattr(v2_evidence_packets, "get_session", billing.sessions)
    monkeypatch.setattr(v2_evidence_packets, "default_store", lambda: billing.store)
    billing.client.app.include_router(v2_evidence_packets.router)
    path = tmp_path / "covered.pdf"
    write_pdf(path, "black")
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
        pytest.fail("Covered text must never be sent to a writing model")

    models = SimpleNamespace(default_device="cpu", lease=unexpected_lease)
    outcome = pipelines.run_analysis(ident, profile="quick", sessions=billing.sessions, store=billing.store, models=models)
    assert outcome.status == "done", (outcome.error, outcome.stages)
    with billing.sessions() as db:
        analysis = db.get(Analysis, ident)
        result = json.loads(billing.store.read_bytes(analysis.result_hash))
        evidence = load_packet_set(db, analysis, billing.store)
        assert project(result)["decision"]["verbal"] == visibility.COVERED_HEADLINE
        assert customer_assessment(result)["confidence_pct"] is None
        assert COVERED_TEXT not in json.dumps(result)
        assert evidence["source"]["chars"] == 0
        assert evidence["coverage"][visibility.KEY]["covered_locations"][0]["page"] == 1
    response = billing.client.get(f"/api/v2/analyses/{ident}/evidence-packets", headers=billing.headers)
    assert response.status_code == 200, response.text
    assert response.json()["coverage"][visibility.KEY] == evidence["coverage"][visibility.KEY]
    assert COVERED_TEXT not in response.text
