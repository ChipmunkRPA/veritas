from copy import deepcopy
from datetime import UTC, datetime

import pytest
from test_document_packets import packet_factory as packet_factory

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.evidence import document_percentage_consistency as percentages
from app.evidence.document_packets import build_packet_set, load_packet_set
from app.reports import engine, multimedia_evidence
from app.reports.specimens import specimen

STATEMENT = "The review found a café processing rate of {} percent among all submitted applications."


def test_local_txt_percentages_bind_source_and_supply_pdf_with_no_classifier_packets(packet_factory, tmp_path, monkeypatch):
    first, second = (STATEMENT.format(value) for value in ("47", "37.0"))
    text = f"Earlier findings.\n\n{first}\n\nA short section separates the figures.\n\n{second}"
    db, store, analysis, original, decoded, _ = packet_factory(scores=(), text=text)
    sample = specimen("document", "standard")
    result = deepcopy(sample.result)
    result["input"].update(sha256=analysis.sha256, filename=analysis.filename, size_bytes=analysis.size_bytes)
    result["manifest"].update(sampling=original["manifest"]["sampling"], detectors=[])
    result.update(signals=original["signals"], findings=[], exhibits=[])
    analysis.result_hash = store.put_bytes(canonical_json(result)).sha256
    monkeypatch.setattr(settings, "document_percentage_consistency", True)
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    evidence = build_packet_set(db, analysis, store, result)
    assert not evidence["packets"] and evidence["original_result_hash"] == canonical_hash(result)
    review = evidence[percentages.KEY]
    assert review["status"] == "different_values" and review["authorship_confidence_contribution"] == 0
    assert review["checked_sentences"] == 2 and review["different_claims"] == review["comparable_claims"] == 1
    (positions,) = review["groups"]
    assert [decoded["text"][row["quantity_start"]:row["quantity_end"]] for row in positions] == ["47", "37.0"]
    monkeypatch.setattr(settings, "document_percentage_consistency", False)
    assert load_packet_set(db, analysis, store) == evidence
    altered = deepcopy(evidence)
    occurrence = altered[percentages.KEY]["groups"][0][0]
    for key in ("start", "end", "quantity_start", "quantity_end"):
        occurrence[key] -= 1
    with pytest.raises(ValueError, match="committed extracted text"):
        percentages.verify_source(altered, decoded["text"], decoded["source_map"], decoded["sentences"])

    frozen = sample.model_copy(update={
        "result": result, "evidence": evidence, "case": None, "model_cards": [], "detector_names": {},
        "analysis": sample.analysis.model_copy(update={"id": analysis.id, "revision": analysis.revision}),
        "media": sample.media.model_copy(update={"sha256": analysis.sha256, "filename": analysis.filename,
            "size_bytes": analysis.size_bytes, "content_type": analysis.content_type}),
        "report": sample.report.model_copy(update={"template_set": engine.TEMPLATE_SET, "requested_at": datetime.now(UTC)}),
    })
    for redacted in (False, True):
        rendered = engine.render_pdf(frozen.model_copy(update={"report": frozen.report.model_copy(update={"redacted": redacted})}))
        assert not rendered.refused
        report_text = " ".join(engine.pdf_text(rendered.pdf).split())
        assert "Repeated percentage statements" in report_text and "unstated dates" in report_text
        assert "zero AI-authorship weight" in report_text and "/percentage_consistency" in report_text.replace(" ", ""), report_text
        assert (first in report_text and second in report_text) is not redacted
        for row in positions:
            assert f'Value [{row["quantity_start"]}, {row["quantity_end"]})' in report_text


@pytest.mark.parametrize("extension,environment,enabled", [(".txt", "dev", False), (".eml", "production", True), (".txt", "production", True)])
def test_configured_percentage_check_remains_local(packet_factory, monkeypatch, extension, environment, enabled):
    first, second = (STATEMENT.format(value) for value in ("47", "37"))
    db, store, analysis, result, *_ = packet_factory(ext=extension, scores=(), text=f"{first}\n\n{second}")
    monkeypatch.setattr(settings, "environment", environment)
    monkeypatch.setattr(settings, "document_percentage_consistency", enabled)
    evidence = build_packet_set(db, analysis, store, result)
    assert percentages.KEY not in evidence


@pytest.mark.parametrize("extension,kind,locations,caption", [
    (".docx", "docx", [{"paragraph_index": 2, "body_paragraph": 2}, {"paragraph_index": 4, "body_paragraph": 4}], "Word body paragraph"),
    (".eml", "email", [{"mime_part": "1"}, {"mime_part": "1"}], "Email body part 1"),
    (".pdf", "pdf", [{"page": 1}, {"page": 2}], "PDF page"),
])
def test_mapped_percentages_have_native_anchors_in_evidence_and_pdf(packet_factory, tmp_path, monkeypatch,
                                                            extension, kind, locations, caption):
    first, second, excluded = (STATEMENT.replace("café", "cafe").format(value) for value in (47, 37, 27))
    path = tmp_path / ("percentages" + extension)
    if extension == ".docx":
        from docx import Document

        source = Document()
        source.add_paragraph("Review overview.")
        source.add_table(rows=1, cols=1).cell(0, 0).text = excluded
        source.add_paragraph(first)
        source.add_paragraph("A second review appears below.")
        source.add_paragraph(second)
        source.add_paragraph(excluded, style="Quote")
        source.add_paragraph().add_run(excluded).font.hidden = True
        source.save(path)
    elif extension == ".eml":
        from email.message import EmailMessage

        message = EmailMessage()
        message.set_content(f"{first}\n\n{second}\n\n> {excluded}")
        path.write_bytes(message.as_bytes())
    else:
        import pymupdf

        with pymupdf.open() as source:
            for statement in (first, second):
                page = source.new_page()
                assert page.insert_textbox((50, 50, 500, 120), statement) >= 0
            source.new_page().insert_text((50, 60), excluded, render_mode=3)
            source.save(path)
    monkeypatch.setattr(settings, "document_percentage_consistency", True)
    db, store, analysis, original, decoded, _ = packet_factory(ext=extension, scores=(), source_pdf=path)
    sample = specimen("document", "standard")
    result = deepcopy(sample.result)
    result["input"].update(sha256=analysis.sha256, filename=analysis.filename, size_bytes=analysis.size_bytes)
    result["manifest"].update(sampling=original["manifest"]["sampling"], detectors=[])
    result.update(signals=original["signals"], findings=[], exhibits=[])
    analysis.result_hash = store.put_bytes(canonical_json(result)).sha256
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    evidence = build_packet_set(db, analysis, store, result)
    review = evidence[percentages.KEY]
    assert review["schema"] == percentages.MAPPED_SCHEMA and review["authorship_confidence_contribution"] == 0
    assert review["checked_sentences"] == 2 and review["different_claims"] == 1
    (positions,) = review["groups"]
    for row, expected in zip(positions, locations, strict=True):
        assert row["anchor"]["kind"] == kind
        assert row["anchor"] | expected == row["anchor"]
    assert [decoded["text"][row["quantity_start"]:row["quantity_end"]] for row in positions] == ["47", "37"]
    percentages.verify_source(evidence, decoded["text"], decoded["source_map"], decoded["sentences"], meta=decoded["meta"])
    altered = deepcopy(evidence)
    altered[percentages.KEY]["groups"][0][0]["anchor"] = positions[1]["anchor"]
    with pytest.raises(ValueError, match="separate source blocks"):
        percentages.validate(altered)
    if extension == ".eml":
        message = EmailMessage()
        message.set_content(f"<p>{first}</p><p>{second}</p>", subtype="html")
        path.write_bytes(message.as_bytes())
        assert not percentages.supported("email", {"body_content_type": "text/html"})

    frozen = sample.model_copy(update={
        "result": result, "evidence": evidence, "case": None, "model_cards": [], "detector_names": {},
        "analysis": sample.analysis.model_copy(update={"id": analysis.id, "revision": analysis.revision}),
        "media": sample.media.model_copy(update={"sha256": analysis.sha256, "filename": analysis.filename,
            "size_bytes": analysis.size_bytes, "content_type": analysis.content_type}),
        "report": sample.report.model_copy(update={"template_set": engine.TEMPLATE_SET, "requested_at": datetime.now(UTC)}),
    })
    for redacted in (False, True):
        rendered = engine.render_pdf(frozen.model_copy(update={"report": frozen.report.model_copy(update={"redacted": redacted})}))
        assert not rendered.refused
        report_text = " ".join(engine.pdf_text(rendered.pdf).split())
        assert report_text.count(caption) >= 2
        assert (first in report_text and second in report_text) is not redacted
