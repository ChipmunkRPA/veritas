"""Check actual signed document packages and the prepaid documentary result."""

import io
import json
import uuid
import zipfile
from datetime import UTC, datetime
from html import escape
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from docx import Document
from test_opendocument_spreadsheet import ods
from test_pdf_credentials import _verify_public_package
from test_provenance_gate_signed_asset import sign
from test_provenance_gate_signed_asset import signing_material as signing_material
from test_reading_packages import epub, impress, powerpoint
from test_spreadsheet_cells import workbook
from test_stripe_billing import billing as billing
from test_stripe_billing import counts, event_for, signed

from app.batches.service import detect_kind
from app.billing import purchase_notice
from app.detectors.text.adapters.provenance import OfficeCredentialsDetector
from app.detectors.text.extract import extract
from app.engine import pipelines, provenance_gate
from app.engine.detector import StageOutputs
from app.engine.stages.common import detector_entry
from app.provenance import c2pa, office_credentials
from app.reports import engine, public_package, snapshot
from app.storage.db import Analysis
from app.storage.files import DOCUMENT_ARCHIVES
from scripts.review_public_content_credentials import inspect

PARAGRAPH = "This locally authored paragraph gives the document reader enough ordinary prose to preserve its exact position."
FOLLOWUP = "The second paragraph follows its recorded reading order and leaves the original source locations available for review."
CONTENT_PARTS = {
    ".docx": "word/document.xml",
    ".odt": "content.xml",
    ".pptx": "ppt/slides/slide1.xml",
    ".xlsx": "xl/cells.xml",
    ".odp": "content.xml",
    ".ods": "content.xml",
    ".epub": "OPS/chapters/first page.xhtml",
}
EXPECTED_TEXT = {
    **dict.fromkeys((".docx", ".odt"), f"Test heading\n\n{PARAGRAPH}\n\nKnown table cell"),
    **dict.fromkeys((".pptx", ".odp", ".epub"), f"{FOLLOWUP}\n\n{PARAGRAPH}"),
    **dict.fromkeys((".xlsx", ".ods"), f"{PARAGRAPH}\n\n{FOLLOWUP}"),
}
ODT_XML = """<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
  xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
  xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" office:version="1.3">
<office:body><office:text><text:h>Test heading</text:h><text:p>{paragraph}<office:annotation>
<text:p>Annotation text stays outside the body.</text:p></office:annotation></text:p>
<table:table><table:table-row><table:table-cell><text:p>Known <text:span>table</text:span><text:s/>cell</text:p>
</table:table-cell></table:table-row></table:table>
</office:text></office:body></office:document-content>"""


def source(extension):
    output = io.BytesIO()
    if extension == ".docx":
        word = Document()
        word.add_heading("Test heading", level=1)
        word.add_paragraph(PARAGRAPH)
        word.add_table(1, 1).cell(0, 0).text = "Known table cell"
        word.sections[0].header.paragraphs[0].text = "Header text stays outside the body."
        word.save(output)
    elif extension == ".odt":
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("mimetype", DOCUMENT_ARCHIVES[extension])
            archive.writestr("content.xml", ODT_XML.format(paragraph=PARAGRAPH))
    elif extension == ".xlsx":
        workbook(output, f'''
            <row r="3"><c r="B3" t="inlineStr"><is><t>{escape(PARAGRAPH)}</t></is></c></row>
            <row r="6"><c r="E6" t="inlineStr"><is><t>{escape(FOLLOWUP)}</t></is></c></row>
            ''')
    elif extension == ".ods":
        ods(output, f'''
            <table:table table:name="Notes">
              <table:table-row><table:table-cell office:value-type="string">
                <text:p>{escape(PARAGRAPH)}</text:p>
              </table:table-cell></table:table-row>
              <table:table-row><table:table-cell office:value-type="string">
                <text:p>{escape(FOLLOWUP)}</text:p>
              </table:table-cell></table:table-row>
            </table:table>''')
    else:
        make = {".pptx": powerpoint, ".odp": impress, ".epub": epub}[extension]
        make(output, passages=(PARAGRAPH, FOLLOWUP))
    return output.getvalue()


@pytest.mark.parametrize("extension", CONTENT_PARTS)
def test_batch_accepts_native_document_package(extension):
    assert detect_kind(source(extension)[:8192], f"source{extension.upper()}") == "document"


@pytest.fixture(params=CONTENT_PARTS)
def office(request, signing_material, monkeypatch):
    extension = request.param
    mime = DOCUMENT_ARCHIVES[extension]
    unsigned = source(extension)
    signer, bundle = signing_material
    monkeypatch.setattr(c2pa, "load_bundle", lambda _: bundle)
    return extension, mime, unsigned, sign(unsigned, mime, signer), bundle


def entry(data, mime):
    adapter = OfficeCredentialsDetector()
    ctx = SimpleNamespace(media=SimpleNamespace(original_bytes=lambda: data), outputs=StageOutputs(probe={"mime": mime}),
                          exhibits=SimpleNamespace(table=Mock(return_value="table")))
    return detector_entry(adapter.spec, adapter.run(ctx), None)


def test_office_credentials_reviewer_reads_original(office, tmp_path):
    extension, mime, _, content, _ = office
    path = tmp_path / ("original" + extension)
    path.write_bytes(content)

    result = inspect(path)

    assert result["media_type"] == mime
    assert result["document_binding"] is True
    assert result["pipeline_action"] == "stop_statistical_detection"
    reserved_sample = tmp_path / "receipt.asset"
    reserved_sample.write_bytes(content)
    assert inspect(reserved_sample, result["sha256"], office_extension=extension) == result
    with pytest.raises(ValueError, match="expected original SHA-256"):
        inspect(path, "0" * 64)


def test_scope_binding_and_changed_document(office, tmp_path):
    extension, mime, unsigned, content, bundle = office
    unsigned_path = tmp_path / ("unsigned" + extension)
    unsigned_path.write_bytes(unsigned)
    original_document = extract(unsigned_path, extension)
    path = tmp_path / ("signed" + extension)
    path.write_bytes(content)
    document = extract(path, extension)
    assert document.text == original_document.text == EXPECTED_TEXT[extension]
    assert document.source_map == original_document.source_map
    assert PARAGRAPH in [document.text[start:end] for start, end in document.paragraphs]
    with zipfile.ZipFile(io.BytesIO(content)) as original:
        manifest = original.getinfo(office_credentials.STORE)
        assert manifest.compress_type == zipfile.ZIP_STORED and manifest.flag_bits == 0

    report, bound = office_credentials.read(content, mime, bundle=bundle, legacy=bundle)
    assert bound and report.state == "trusted" and report.ai_declared
    verified = entry(content, mime)
    assert provenance_gate.decide({"detectors": [verified]}, "document")["action"] == "stop_statistical_detection"
    verified["metrics"].pop("office_document_binding")
    assert provenance_gate.decide({"detectors": [verified]}, "document")["action"] == "continue_detection"

    changed = io.BytesIO()
    main = CONTENT_PARTS[extension]
    with zipfile.ZipFile(io.BytesIO(content)) as original, zipfile.ZipFile(changed, "w") as replacement:
        for part in original.infolist():
            data = original.read(part)
            if part.filename == main:
                assert b"locally authored" in data
                data = data.replace(b"locally authored", b"changed")
            replacement.writestr(part, data)
    appended = io.BytesIO(content)
    with zipfile.ZipFile(appended, "a") as archive:
        archive.writestr("extra.txt", "An additional member must invalidate the signed package.")
    for data, expected in ((unsigned, "absent"), (changed.getvalue(), "invalid"), (appended.getvalue(), "invalid")):
        result, bound = office_credentials.read(data, mime, bundle=bundle, legacy=bundle)
        assert result.state == expected and not bound
        if expected == "invalid":
            assert any(code.scope == "active manifest" and code.level == "failure"
                       and code.code == "assertion.collectionHash.mismatch" for code in result.codes)
        assert provenance_gate.decide({"detectors": [entry(data, mime)]}, "document")["action"] == "continue_detection"


def test_signed_document_formats_through_prepaid_report(billing, office, tmp_path):
    extension, mime, _, content, _ = office
    purchase = billing.client.post("/api/v2/billing/checkout", headers={**billing.headers, "Idempotency-Key": str(uuid.uuid4())},
        data={"kind": "document", "profile": "full", "risk_accepted": "true", "risk_notice_version": purchase_notice.VERSION,
              "risk_notice_sha256": purchase_notice.SHA256}, files={"file": ("fixture" + extension, content, mime)})
    assert purchase.status_code == 201, purchase.text
    order = purchase.json()["order_id"]
    assert billing.created[0][0]["line_items[0][price_data][unit_amount]"] == 999
    assert signed(billing, event_for(billing, order, paid=False)).status_code == 200
    assert counts(billing) == (0, 0, 0)
    assert signed(billing, event_for(billing, order)).status_code == 200
    assert counts(billing) == (1, 1, 1)
    ident = billing.client.get(f"/api/v2/billing/orders/{order}", headers=billing.headers).json()["analysis_id"]

    model = SimpleNamespace(lease=Mock(side_effect=AssertionError("Unexpected statistical model lease")))
    result = pipelines.run_analysis(ident, profile="full", sessions=billing.sessions, store=billing.store, models=model)
    assert result.status == "done", (result.error, result.stages)
    model.lease.assert_not_called()
    assert result.outputs["provenance_gate"]["record_detector_id"] == OfficeCredentialsDetector.spec.id
    assert result.outputs["detect"]["detectors"] and all(row["status"] == "skipped" for row in result.outputs["detect"]["detectors"])
    decision = result.outputs["fuse"]["decision"]
    assert decision["provenance_record"]["kind"] == "decisive_ai" and decision["probability_ai_uncalibrated"] is None
    with billing.sessions() as db:
        analysis = db.get(Analysis, ident)
        raw = json.loads(billing.store.read_bytes(analysis.result_hash))
        public = public_package.public_result(raw)
        assert public["decision"]["code"] == "ai_supported"
        assert "Statistical writing checks were skipped" in json.dumps(public)
        frozen = snapshot.build(db, analysis, raw, report_id=str(uuid.uuid4()), revision=1, profile="standard",
            redacted=False, supersedes_report_id=None, revision_reason=None, requested_at=datetime.now(UTC),
            requested_by="local-test", template_set=engine.TEMPLATE_SET, veritas_version="test", store=billing.store)
        pdf = engine.render_pdf(frozen).pdf
        assert "Statistical writing checks were skipped" in " ".join(engine.pdf_text(pdf).split())
        _verify_public_package(billing, analysis, db, frozen, pdf, tmp_path)
