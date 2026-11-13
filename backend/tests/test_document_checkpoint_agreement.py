from copy import deepcopy
from email.message import EmailMessage
from html import unescape
from unittest.mock import Mock

import pytest

from app.detectors.text.adapters import agreement
from app.detectors.text.extract import extract, text_hash
from app.engine.boundary import Cancelled, Deadline
from app.engine.detector import Profile
from app.engine.runner import StageContext
from app.engine.stages import common
from app.evidence import document_matched_context, document_scope


def context(deadline):
    return StageContext("test", "detect", Profile.FULL, deadline, None, None, {}, {}, None, Mock())


def source():
    text = "First part. Second part."
    spans = [{"start": 0, "end": 11}, {"start": 12, "end": 24}]
    window = {"focus": spans[0], "spans": spans, "input_sha256": text_hash(text), "score": .8,
              "token_coverage": {"tokens": 5, "token_windows": 1}, "unit_ref": "sentence:0"}
    evidence = {"windows": [window]}
    primary = {"detector_id": document_matched_context.DETECTOR, "status": "ok", "details": {"text_evidence": evidence}}
    return text, window, evidence, {"detectors": [primary]}


@pytest.mark.parametrize(("method", "anchor", "meta"), [
    ("python-docx", {"kind": "docx", "body_path": [0], "run_mapping_complete": True}, {}),
    ("pymupdf", {"kind": "pdf", "extraction": "text_layer", "fragments": [{"start": 0, "end": 3}]}, {}),
    ("email", {"kind": "email", "quote_role": "body", "mime_part": "1"},
     {"email": {"body_content_type": "text/plain", "body_part": "1"}}),
])
def test_second_check_requires_mapped_native_source(method, anchor, meta):
    decoded = {"meta": {"method": method, **meta}, "paragraphs": [[0, 3]],
               "source_map": [{"start": 0, "end": 3, **anchor}]}
    assert agreement._supported_source(decoded)
    decoded["source_map"][0]["end"] = 2
    assert not agreement._supported_source(decoded)


@pytest.mark.parametrize(("method", "anchor", "meta"), [
    ("python-docx", {"kind": "docx", "body_path": [0, "row", 0, "cell", 0, 0], "run_mapping_complete": True}, {}),
    ("pymupdf", {"kind": "pdf", "extraction": "ocr", "fragments": [{"start": 0, "end": 3}]}, {}),
    ("pymupdf", {"kind": "pdf", "extraction": "text_layer", "fragments": [{"start": 0, "end": 3}]}, {"unmapped_pages": [2]}),
    ("email", {"kind": "email", "quote_role": "quoted", "mime_part": "1"},
     {"email": {"body_content_type": "text/plain", "body_part": "1"}}),
    ("email", {"kind": "markup", "quote_role": "unknown", "mime_part": "1"},
     {"email": {"body_content_type": "text/html", "body_part": "1"}}),
])
def test_second_check_excludes_unsupported_source_geometry(method, anchor, meta):
    decoded = {"meta": {"method": method, **meta}, "paragraphs": [[0, 3]],
               "source_map": [{"start": 0, "end": 3, **anchor}]}
    assert not agreement._supported_source(decoded)


def test_mixed_pdf_second_check_binds_complete_native_text_to_one_intact_page(tmp_path, monkeypatch):
    from dataclasses import asdict

    import pymupdf

    from app.detectors.text.segment import segment

    source = tmp_path / "mixed.pdf"
    prose = (
        "We reviewed the draft earlier this week and kept the parts that explain how the new process works. "
        "The remaining changes describe the dates and the steps each reviewer will take before the next meeting. "
        "We can share the revised document with the whole team once the source details and notes have been checked. "
        "Please bring any comments to the meeting so we can see which questions still need a written answer."
    )
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        assert page.insert_textbox(pymupdf.Rect(32, 50, 525, 700), prose, fontsize=10) >= 0
        pdf.new_page()
        pdf.save(source)
    doc = extract(source, ".pdf")
    sentences = [asdict(unit) for unit in segment(doc)]
    decoded = {"text": doc.text, "paragraphs": doc.paragraphs, "source_map": doc.source_map, "meta": doc.meta, "sentences": sentences}
    spans = [{key: row[key] for key in ("start", "end")} for row in sentences[:3]]
    window = {"focus": spans[1], "spans": spans, "input_sha256": text_hash(" ".join(doc.text[s["start"]:s["end"]] for s in spans)),
              "score": .8, "unit_ref": "sentence:1", "token_coverage": {"tokens": 56, "token_windows": 1}}
    primary = {"detector_id": document_matched_context.DETECTOR, "status": "ok", "version": "1.0.0",
               "model_revisions": {document_matched_context.MODEL: document_matched_context.REVISION},
               "details": {"text_evidence": {"windows": [window]}}}
    monkeypatch.setattr("app.detectors.text.extract._ocr_language", lambda _: {"code": "en", "status": "supported"})
    monkeypatch.setattr(agreement, "_review", lambda *_: {})
    assert not agreement._supported_source(decoded)
    assert agreement.candidates(decoded, primary) == [window]
    agreement.record(context(Deadline(60)), {"detectors": [primary]}, decoded)
    record = primary["details"]["text_evidence"][agreement.KEY]
    assert record["schema"] == document_scope.CHECKPOINT_AGREEMENT_PDF_SCHEMA
    assert record["pdf_spans"] == [{"start": 0, "end": len(doc.text), "page": 1}]
    assert record["ranges"][0]["input_sha256"] == window["input_sha256"]
    document = {"source": {"chars": len(doc.text), "method": "pymupdf"}, "coverage": {**doc.meta, agreement.KEY: record}}
    document_scope.validate_checkpoint_agreement(document, doc.text, result={"signals": [primary]}, mapping=doc.source_map)
    altered = deepcopy(document)
    altered["coverage"][agreement.KEY]["pdf_spans"][0]["page"] = 2
    with pytest.raises(ValueError, match="PDF source position"):
        document_scope.validate_checkpoint_agreement(altered)
    for corruption in (lambda fragment: fragment.update(bbox=[-2, 0, 1, 1]), lambda fragment: fragment.update(end=0)):
        broken = deepcopy(decoded)
        corruption(broken["source_map"][0]["fragments"][0])
        assert agreement.candidates(broken, primary) is None
    broken = deepcopy(decoded)
    broken["source_map"][0]["fragments"].pop(0)
    assert agreement.candidates(broken, primary) is None
    broken["meta"]["ocr"]["pages"] = [None]
    assert agreement.candidates(broken, primary) is None


@pytest.mark.parametrize("ocr_succeeds", [True, False])
def test_pdf_second_check_excludes_entire_ocr_and_partial_pages(tmp_path, monkeypatch, ocr_succeeds):
    from dataclasses import asdict

    import pymupdf
    from test_document_hybrid_pdf_coverage import NATIVE, hybrid_pdf

    from app.detectors.text import extract as extraction
    from app.detectors.text.segment import segment

    if ocr_succeeds and not extraction._ocr_runtime().get("available"):
        pytest.skip("Verified local OCR runtime is unavailable")
    if not ocr_succeeds:
        monkeypatch.setattr(extraction, "_ocr_runtime", lambda: {"available": False, "reason": "Fixture"})
    hybrid = tmp_path / "hybrid.pdf"
    hybrid_pdf(hybrid)
    mixed = tmp_path / "mixed.pdf"
    prose = (
        NATIVE + " The recorded explanations refer to the source version that we filed before the last meeting with "
        "the whole review team, and our shared notes show exactly where the next set of decisions and written responses will appear."
    )
    with pymupdf.open(hybrid) as scan, pymupdf.open() as pdf:
        pdf.insert_pdf(scan)
        page = pdf.new_page(width=612, height=792)
        assert page.insert_textbox(pymupdf.Rect(40, 25, 570, 135), prose, fontsize=12) >= 0
        pdf.save(mixed)
    doc = extract(mixed, ".pdf")
    assert doc.meta["unmapped_pages"] == []
    assert doc.meta["ocr"]["pages"][0]["status"] == ("extracted" if ocr_succeeds else "unchecked")
    assert bool(doc.meta["partially_checked_pages"]) is not ocr_succeeds
    original = doc.source_map[-1]
    assert original["page"] == 2 and doc.text[original["start"]:original["end"]] == prose
    scoped = [{key: original[key] for key in ("start", "end", "page")}]
    assert document_scope.pdf_agreement_text_spans(doc.source_map, doc.meta, doc.text) == scoped

    sentences = [asdict(unit) for unit in segment(doc)]
    before, after = ([{key: row[key] for key in ("start", "end")} for row in sentences
                      if (row["start"] < original["start"]) == on_first] for on_first in (True, False))
    candidates = []
    for i, spans in enumerate((before[:3], [before[-1], *after[:2]], after[:3])):
        content = " ".join(doc.text[span["start"]:span["end"]] for span in spans)
        candidates.append({"focus": spans[1], "spans": spans, "input_sha256": text_hash(content), "score": .8,
                           "token_coverage": {"tokens": 76, "token_windows": 1}, "unit_ref": f"sentence:{i}"})
    primary = {"detector_id": document_matched_context.DETECTOR, "status": "ok", "version": "1.0.0",
               "model_revisions": {document_matched_context.MODEL: document_matched_context.REVISION},
               "details": {"text_evidence": {"windows": candidates}}}
    decoded = {"text": doc.text, "paragraphs": doc.paragraphs, "source_map": doc.source_map,
               "meta": doc.meta, "sentences": sentences}
    monkeypatch.setattr(extraction, "_ocr_language", lambda _: {"code": "en", "status": "supported"})
    monkeypatch.setattr(agreement, "_review", lambda *_: {})
    assert agreement.candidates(decoded, primary) == candidates[-1:]
    agreement.record(context(Deadline(60)), {"detectors": [primary]}, decoded)
    record = primary["details"]["text_evidence"][agreement.KEY]
    assert record["pdf_spans"] == scoped
    assert [row["spans"] for row in record["ranges"]] == [candidates[-1]["spans"]]


def test_mixed_pdf_second_check_rejects_contexts_crossing_native_or_ocr_pages():
    window = {"focus": {"start": 1, "end": 20}, "spans": [{"start": 1, "end": 20}, {"start": 32, "end": 49}]}
    blocks = [{"start": 0, "end": 30, "page": 1}, {"start": 32, "end": 70, "page": 3}]
    assert document_scope.pdf_agreement_window_page(window, blocks) is None
    assert document_scope.pdf_agreement_window_page(window, blocks[:1]) is None
    blocks[1]["page"] = 1
    assert document_scope.pdf_agreement_window_page(window, blocks) == 1


def test_reply_scope_replays_the_exact_body_and_rejects_quoted_context():
    text, window, _, _ = source()
    body_end = len(text)
    text += "\n\n> A previous message."
    email = {"body_content_type": "text/plain", "body_part": "1.1"}
    mapping = [
        {"start": 0, "end": body_end, "kind": "email", "mime_part": "1.1", "quote_role": "body",
         "source_ranges": [[0, body_end, 0, body_end]]},
        {"start": body_end + 2, "end": len(text), "kind": "email", "mime_part": "1.1", "quote_role": "quoted",
         "source_ranges": [[body_end + 2, len(text), body_end + 2, len(text)]]},
    ]
    scope = document_scope.email_agreement_body_spans(mapping, email, len(text))
    assert scope == [{"start": 0, "end": body_end}]
    included = {**window["focus"], "spans": window["spans"], "input_sha256": window["input_sha256"], "state": "unavailable"}
    record = {"schema": document_scope.CHECKPOINT_AGREEMENT_EMAIL_SCHEMA, "body_spans": scope,
              "authorship_confidence_contribution": 0, "ranges": [included]}
    document = {"source": {"chars": len(text), "method": "email"}, "coverage": {"email": email, agreement.KEY: record}}
    document_scope.validate_checkpoint_agreement(document, text, mapping=mapping)
    wrong_source = deepcopy(document)
    wrong_source["source"]["method"] = "pymupdf"
    with pytest.raises(ValueError, match="Invalid second-check email body scope"):
        document_scope.validate_checkpoint_agreement(wrong_source)
    altered_scope = deepcopy(document)
    altered_scope["coverage"][agreement.KEY]["body_spans"][0]["end"] -= 1
    with pytest.raises(ValueError, match="scope differs from the mapped reply body"):
        document_scope.validate_checkpoint_agreement(altered_scope, text, mapping=mapping)
    altered_range = deepcopy(document)
    altered_range["coverage"][agreement.KEY]["ranges"][0]["spans"][1] = {"start": body_end + 2, "end": len(text)}
    with pytest.raises(ValueError, match="Invalid second-check source coordinates"):
        document_scope.validate_checkpoint_agreement(altered_range)


def test_html_reply_preserves_mime_entities_and_explicit_quote_boundary(tmp_path):
    raw = "<p><strong>&#70;irst</strong> &amp; second.</p>\n<p>Next paragraph.</p><blockquote><p>Earlier reply.</p></blockquote>"
    email = EmailMessage()
    email.set_content(raw, subtype="html", cte="base64")
    path = tmp_path / "reply.eml"
    path.write_bytes(email.as_bytes())
    doc = extract(path, ".eml")
    assert doc.text == "First & second.\n\nNext paragraph.\n\nEarlier reply."
    metadata = doc.meta["email"]
    assert document_scope.mapped_email_html(metadata) and metadata["body_part"] == "1"
    assert [block["quote_role"] for block in doc.source_map] == ["body", "body", "quoted"]
    assert all(unescape(raw[start:stop]) == doc.text[a:b] for a, b, start, stop in doc.source_map[0]["source_ranges"])
    scope = document_scope.email_agreement_body_spans(doc.source_map, metadata, len(doc.text))
    assert scope == [{"start": a, "end": b} for a, b in doc.paragraphs[:2]]
    record = {"schema": document_scope.CHECKPOINT_AGREEMENT_HTML_SCHEMA, "body_spans": scope,
              "authorship_confidence_contribution": 0, "ranges": []}
    evidence = {"source": {"method": "email", "chars": len(doc.text)}, "coverage": {"email": metadata, agreement.KEY: record}}
    document_scope.validate_checkpoint_agreement(evidence, doc.text, mapping=doc.source_map)
    record["schema"] = document_scope.CHECKPOINT_AGREEMENT_EMAIL_LAYOUT_SCHEMA
    with pytest.raises(ValueError, match="source format"):
        document_scope.validate_checkpoint_agreement(evidence, doc.text, mapping=doc.source_map)


@pytest.mark.parametrize("markup", [
    '<p><strong style="display:none">Hidden.</strong></p><p>Second.</p><blockquote><p>Quote.</p></blockquote>',
    '<p>First.</p><blockquote><p>Quote.</p></blockquote><p>Second.</p>',
    '<p>First.</p><p>On Friday someone wrote:</p><blockquote><p>Quote.</p></blockquote>',
])
def test_unverified_html_scope_does_not_select_reply_text(tmp_path, markup):
    email = EmailMessage()
    email.set_content(markup, subtype="html")
    path = tmp_path / "unverified.eml"
    path.write_bytes(email.as_bytes())
    doc = extract(path, ".eml")
    assert document_scope.email_agreement_body_spans(doc.source_map, doc.meta["email"], len(doc.text)) is None


@pytest.mark.parametrize("paragraphs", [1, 4])
def test_html_short_reply_records_new_schema_without_admitting_old_schema(tmp_path, monkeypatch, paragraphs):
    from dataclasses import asdict

    from app.detectors.text.segment import segment

    body = [
        "We reviewed the draft earlier this week and kept the parts that explain how the new process works.",
        "The remaining changes describe the dates and the steps each reviewer will take before the next meeting.",
        "We can share the revised document with the whole team once the source details and notes have been checked.",
        "Please bring any comments to the meeting so we can see which questions still need a written answer.",
    ]
    blocks = [" ".join(body)] if paragraphs == 1 else body
    msg = EmailMessage()
    msg.set_content("".join(f"<p>{block}</p>" for block in blocks) + "<blockquote><p>Earlier note.</p></blockquote>",
                    subtype="html", cte="quoted-printable")
    path = tmp_path / "reply.eml"
    path.write_bytes(msg.as_bytes())
    doc = extract(path, ".eml")
    units = [{"start": unit.start, "end": unit.end} for unit in segment(doc)]
    windows = []
    for i, spans in enumerate((units[:2], units[-2:])):
        content = " ".join(doc.text[span["start"]:span["end"]] for span in spans)
        windows.append({"focus": spans[0], "spans": spans, "input_sha256": text_hash(content), "score": .8,
                        "token_coverage": {"tokens": 42, "token_windows": 1}, "unit_ref": f"sentence:{i}"})
    primary = {"detector_id": document_matched_context.DETECTOR, "status": "ok", "version": "1.0.0",
               "model_revisions": {document_matched_context.MODEL: document_matched_context.REVISION},
               "details": {"text_evidence": {"windows": windows}}}
    decoded = {"text": doc.text, "paragraphs": doc.paragraphs, "source_map": doc.source_map, "meta": doc.meta,
               "sentences": [asdict(unit) for unit in segment(doc)]}
    monkeypatch.setattr("app.detectors.text.extract._ocr_language", lambda _: {"code": "en", "status": "supported"})
    monkeypatch.setattr(agreement, "_review", lambda *_: {})
    agreement.record(context(Deadline(60)), {"detectors": [primary]}, decoded)
    evidence = primary["details"]["text_evidence"][agreement.KEY]
    assert evidence["schema"] == document_scope.CHECKPOINT_AGREEMENT_HTML_SHORT_LAYOUT_SCHEMA
    assert len(evidence["body_spans"]) == paragraphs
    assert [row["input_sha256"] for row in evidence["ranges"]] == [windows[0]["input_sha256"]]
    document = {"source": {"chars": len(doc.text), "method": "email"},
                "coverage": {"email": doc.meta["email"], agreement.KEY: evidence}}
    document_scope.validate_checkpoint_agreement(document, doc.text, result={"signals": [primary]}, mapping=doc.source_map)
    if paragraphs == 1:
        decoded["sentences"].pop(0)
        assert agreement.candidates(decoded, primary) is None
    evidence["schema"] = document_scope.CHECKPOINT_AGREEMENT_HTML_SCHEMA
    with pytest.raises(ValueError, match="Invalid second-check email body scope"):
        document_scope.validate_checkpoint_agreement(document, doc.text, mapping=doc.source_map)


@pytest.mark.parametrize("paragraphs", [2, 3, 5, 6])
def test_new_reply_layout_keeps_exact_windows_and_rejects_legacy_scope(monkeypatch, paragraphs):
    paragraph = (
        "We compared the recorded version of the document against the file saved with our earlier review, "
        "and the next review will include the dates, the names of the reviewers, and every comment received since the meeting."
    )
    blocks = [paragraph] * paragraphs
    text = "\n\n".join([*blocks, "> A previous message."])
    email = {"body_content_type": "text/plain", "body_part": "1.1"}
    mapping = []
    start = 0
    for index, block in enumerate([*blocks, "> A previous message."]):
        end = start + len(block)
        mapping.append({"start": start, "end": end, "kind": "email", "mime_part": "1.1",
                        "quote_role": "body" if index < paragraphs else "quoted",
                        "source_ranges": [[start, end, start, end]]})
        start = end + 2
    spans = [{key: row[key] for key in ("start", "end")} for row in mapping]
    windows = []
    for i, pair in enumerate((spans[:2], spans[-2:])):
        content = " ".join(text[s["start"]:s["end"]] for s in pair)
        windows.append({"focus": pair[0], "spans": pair, "input_sha256": text_hash(content), "score": .8,
                        "token_coverage": {"tokens": 30, "token_windows": 1}, "unit_ref": f"sentence:{i}"})
    primary = {"detector_id": document_matched_context.DETECTOR, "status": "ok", "version": "1.0.0",
               "model_revisions": {document_matched_context.MODEL: document_matched_context.REVISION},
               "details": {"text_evidence": {"windows": windows}}}
    decoded = {"meta": {"method": "email", "email": email}, "paragraphs": list(spans),
               "sentences": list(spans), "text": text, "source_map": mapping}
    monkeypatch.setattr("app.detectors.text.extract._ocr_language", lambda _: {"code": "en", "status": "supported"})
    monkeypatch.setattr(agreement, "_review", lambda *_: {})
    agreement.record(context(Deadline(60)), {"detectors": [primary]}, decoded)
    result = primary["details"]["text_evidence"][agreement.KEY]
    assert result["schema"] == document_scope.CHECKPOINT_AGREEMENT_EMAIL_LAYOUT_SCHEMA
    assert result["body_spans"] == spans[:paragraphs]
    assert [row["input_sha256"] for row in result["ranges"]] == [windows[0]["input_sha256"]]
    assert result["authorship_confidence_contribution"] == 0
    document = {"source": {"chars": len(text), "method": "email"}, "coverage": {"email": email, agreement.KEY: result}}
    document_scope.validate_checkpoint_agreement(document, text, result={"signals": [primary]}, mapping=mapping)
    result["schema"] = document_scope.CHECKPOINT_AGREEMENT_EMAIL_SCHEMA
    with pytest.raises(ValueError, match="Invalid second-check email body scope"):
        document_scope.validate_checkpoint_agreement(document, text, mapping=mapping)


def test_review_timeout_preserves_a_parent_budget_and_primary_response(monkeypatch):
    text, window, evidence, detected = source()
    monkeypatch.setattr(agreement, "candidates", lambda *_: [window])
    clock = [0.0]
    deadline = Deadline(45, clock=lambda: clock[0])
    ctx = context(deadline)

    def time_out(optional, *args, **kwargs):
        assert optional.deadline.remaining() == 15
        clock[0] = 16
        optional.check("forced test")

    monkeypatch.setattr(common, "run_detectors", time_out)
    agreement.record(ctx, detected, {"text": text})
    assert window["score"] == .8 and evidence[agreement.KEY]["ranges"][0]["state"] == "unavailable"
    assert ctx.deadline.remaining() == 29


def test_review_does_not_start_when_only_the_finish_reserve_remains(monkeypatch):
    text, window, evidence, detected = source()
    monkeypatch.setattr(agreement, "candidates", lambda *_: [window])
    runner = Mock()
    monkeypatch.setattr(common, "run_detectors", runner)
    agreement.record(context(Deadline(29)), detected, {"text": text})
    assert evidence[agreement.KEY]["ranges"][0]["state"] == "unavailable"
    runner.assert_not_called()


def test_parent_cancellation_still_propagates(monkeypatch):
    deadline = Deadline(60)

    def cancelled(optional, *args, **kwargs):
        deadline.lose_lease()
        optional.check("forced test")

    monkeypatch.setattr(common, "run_detectors", cancelled)
    with pytest.raises(Cancelled):
        agreement._review(context(deadline), "source", [{}])
    assert deadline.lease_lost


def test_invalid_second_response_is_unavailable_and_cannot_be_rebound(monkeypatch):
    text, window, evidence, detected = source()
    monkeypatch.setattr(agreement, "candidates", lambda *_: [window])
    monkeypatch.setattr(agreement, "_review", lambda *_: {
        "sentence:0": {"input_sha256": window["input_sha256"], "response_exact_hex": float(2).hex(), "tokens": 5},
    })
    agreement.record(context(Deadline(60)), detected, {"text": text})
    result = {"signals": detected["detectors"]}
    document = {"coverage": {agreement.KEY: deepcopy(evidence[agreement.KEY])}, "source": {"chars": len(text)}}
    row = document["coverage"][agreement.KEY]["ranges"][0]
    assert row["state"] == "unavailable" and "response_exact_hex" not in row
    document_scope.validate_checkpoint_agreement(document, text, result=result)
    row["input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different source input"):
        document_scope.validate_checkpoint_agreement(document, text)
    with pytest.raises(ValueError, match="differs from the frozen result"):
        document_scope.validate_checkpoint_agreement(document, text, result=result)
