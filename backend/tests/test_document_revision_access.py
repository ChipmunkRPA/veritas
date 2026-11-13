from __future__ import annotations

import base64
import hashlib
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from html import unescape
from io import BytesIO
from tempfile import TemporaryDirectory

import pymupdf
import pytest
from docx import Document
from docx.oxml import OxmlElement
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_image_reference_access import reference_http as reference_http

from app.api import auth, v2_document_revision
from app.core.config import settings
from app.evidence import document_revision
from app.evidence import document_revision_writing as paired
from app.reports.document_revision_pdf import _side
from app.storage.db import Analysis, Artefact, BillingOrderRow
from scripts.review_document_revision_export import review


def test_paired_plain_text_boundary_replays_both_scales_and_preserves_opt_out(monkeypatch):
    earlier = ("The team approved the earlier draft after its last meeting. The editor checked the original version "
               "for clarity and accuracy.\n\nThe published article records the procedures for the next review.")
    later = earlier.replace("The editor checked the original version for clarity and accuracy.",
                            "The editor carefully examined the complete draft to improve its organization and readability.")
    monkeypatch.setattr(paired, "_ocr_language", lambda _: {"code": "en", "status": "supported"})
    a, b = earlier.encode(), later.encode()
    result, old, new = document_revision.compare_with_text(a, b, ".txt")
    assert paired.paired_input(result, old, new) is None
    candidate = paired.paired_input(result, old, new, allow_boundary=True)
    assert candidate is not None and candidate.across_paragraphs
    for side, source in (("earlier", old), ("later", new)):
        focus = candidate.geometry[side]
        assert focus["text"] == source[focus["start"]:focus["end"]]
        assert focus["center_input_sha256"] == paired.text_hash(focus["text"])
    contextual = [{"earlier": .2, "later": .6}, {"earlier": .2, "later": .6}]
    exact = [{"earlier": .4, "later": .6}, {"earlier": .4, "later": .6}]
    proof = paired.result(candidate, contextual, center_responses=exact)
    assert proof["schema"] == paired.BOUNDARY_SCHEMA and proof["status"] == "joint_increase"
    assert proof["authorship_confidence_contribution"] == 0
    assert paired.verify(result, old, new, proof) == "joint_increase"
    reversed_exact = [dict(earlier=reading["later"], later=reading["earlier"]) for reading in exact]
    assert paired.result(candidate, contextual, center_responses=reversed_exact)["status"] == "no_joint_increase"
    proof["later"]["center_input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not replay"):
        paired.verify(result, old, new, proof)
    malformed = later.replace("\n\n", " ")
    changed_layout, old, new = document_revision.compare_with_text(a, malformed.encode(), ".txt")
    assert paired.paired_input(changed_layout, old, new, allow_boundary=True) is None
    unavailable = paired.result(None)
    assert unavailable["schema"] == paired.WIDE_SCHEMA and "center_responses" not in unavailable
    assert paired.verify(changed_layout, old, new, unavailable) == "outside_scope"


def _email(text):
    message = EmailMessage()
    message["From"] = "fixture@example.invalid"
    message.set_content(text, cte="quoted-printable")
    message.add_alternative("<p>Excluded HTML body</p>", subtype="html")
    message.add_attachment(b"Excluded attachment body", maintype="application", subtype="octet-stream", filename="fixture.bin")
    return message.as_bytes()


def _html_email(body, *, quote="Previous scheduling discussion.", cte="quoted-printable"):
    markup = f"<html><body>{body}<blockquote><p>{quote}</p></blockquote></body></html>"
    message = EmailMessage()
    message["From"] = "fixture@example.invalid"
    message.set_content(markup, subtype="html", cte=cte)
    message.add_attachment(b"Private reference sheet", maintype="application", subtype="octet-stream", filename="notes.bin")
    return message.as_bytes()


def _word(body, omitted):
    word = Document()
    word.add_paragraph("A shared opening.").runs[0].bold = True
    word.add_table(rows=1, cols=1).cell(0, 0).text = "Table " + omitted
    word.add_paragraph()
    paragraph = word.add_paragraph(body)
    tracked = OxmlElement("w:ins")
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "Tracked " + omitted
    run.append(text)
    tracked.append(run)
    paragraph._p.append(tracked)
    word.sections[0].header.paragraphs[0].text = "Header " + omitted
    word.sections[0].footer.paragraphs[0].text = "Footer " + omitted
    file = BytesIO()
    word.save(file)
    return file.getvalue()


def _pdf_draft(term, rotation):
    with pymupdf.open() as document:
        title = document.new_page()
        title.insert_text((60, 76), "Release plan - annual review", fontsize=18)
        title.insert_text((60, 112), "The first page keeps the amber shipment.", fontsize=13)
        detail = document.new_page()
        detail.insert_text((60, 76), "Delivery details", fontsize=18)
        detail.insert_text((60, 112), f"The second shipment will use the {term} packaging.", fontsize=13)
        detail.set_rotation(rotation)
        return document.tobytes()


def _install(fixture, monkeypatch):
    @contextmanager
    def sessions():
        with Session(fixture.engine) as db:
            yield db

    monkeypatch.setattr(v2_document_revision, "get_session", sessions)
    monkeypatch.setattr(v2_document_revision, "default_store", lambda: fixture.store)
    monkeypatch.setattr(settings, "document_revision_comparison_enabled", True)
    return fixture


def _saved(f, aid, extension, data, *, status="done"):
    retained = f.store.put_bytes(data)
    with Session(f.engine) as db:
        db.add(Analysis(id=aid, kind="document", filename="later" + extension, content_type="text/plain", case_id="reference-case",
                        status=status, sha256=retained.sha256, size_bytes=retained.size_bytes, result_hash="f" * 64))
        db.commit()
    return retained


def _url(aid):
    return f"/api/v2/analyses/{aid}/document-revision-comparison"


def _upload(data, extension=".eml"):
    return {"earlier": ("earlier" + extension, data, "application/octet-stream")}


def test_pdf_comparison_binds_repeated_text_to_both_original_pages(reference_http, monkeypatch, tmp_path):
    f = _install(reference_http, monkeypatch)
    monkeypatch.setattr(settings, "text_checkpoint_agreement_review", False)
    earlier, later = _pdf_draft("amber", 90), _pdf_draft("green", 0)
    retained = _saved(f, "pdf-draft", ".pdf", later)
    url = _url("pdf-draft")
    assert f.client.get(url, headers=f.viewer).json() == {"available": True, "extension": ".pdf"}
    response = f.client.post(url, headers=f.viewer, files=_upload(earlier, ".pdf"), data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    data = response.json()
    comparison = data["comparison"]
    assert comparison["format"] == "pdf" and comparison["coordinate_space"] == "pdf_native_text_unicode_codepoint"
    assert comparison["earlier"]["sha256"] == hashlib.sha256(earlier).hexdigest()
    assert comparison["later"]["sha256"] == retained.sha256
    (change,) = comparison["changes"]
    marked_pixels = []
    for side, source, word, rotation in (("earlier", earlier, "amber", 90), ("later", later, "green", 0)):
        excerpt = change[side]
        assert excerpt["text"] == word
        assert excerpt["changed_tokens"] == [[excerpt["start"], excerpt["end"]]]
        (location,) = excerpt["pdf_pages"]
        assert (location["page"], location["page_rotation"]) == (2, rotation)
        preview = location["preview"]
        with pymupdf.open(stream=source, filetype="pdf") as original:
            page = original[1]
            box = tuple(page.search_for(word)[0])
            assert location["word_regions"] == [{"start": excerpt["start"], "end": excerpt["end"], "bbox": list(box)}]
            page.set_rotation(0)
            scale = preview["render_scale"]
            pixels = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            full_page = Image.frombytes("RGB", (pixels.width, pixels.height), pixels.samples)
            crop = full_page.crop(preview["source_crop_pixels"])
        raw = base64.b64decode(preview["crop_png"], validate=True)
        marked = base64.b64decode(preview["highlighted_png"], validate=True)
        assert hashlib.sha256(raw).hexdigest() == preview["crop_sha256"]
        assert hashlib.sha256(marked).hexdigest() == preview["highlighted_sha256"]
        with Image.open(BytesIO(raw)) as image, Image.open(BytesIO(marked)) as overlay:
            assert image.tobytes() == crop.tobytes() and overlay.size == crop.size
            assert overlay.tobytes() != crop.tobytes()
            marked_pixels.append((overlay.size, overlay.tobytes()))
    with pymupdf.open(stream=base64.b64decode(data["pdf"], validate=True), filetype="pdf") as report:
        pages = " ".join(page.get_text() for page in report)
        assert "Page 2 · highlighted source crop" in pages and "C01 · PDF text" in pages
        embedded = set()
        for page in report:
            for image in page.get_images():
                with Image.open(BytesIO(report.extract_image(image[0])["image"])) as pixels:
                    embedded.add((pixels.size, pixels.convert("RGB").tobytes()))
        assert set(marked_pixels) <= embedded
    old, new, exported = (tmp_path / filename for filename in ("earlier.pdf", "later.pdf", "comparison.json"))
    old.write_bytes(earlier)
    new.write_bytes(later)
    exported.write_text(json.dumps(comparison))
    assert review(exported, old, new)["pdf_previews_verified"] == 2
    change["later"]["pdf_pages"][0]["page"] = 1
    exported.write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="does not replay"):
        review(exported, old, new)


def test_single_page_pdf_paired_review_keeps_page_evidence_and_replays(reference_http, monkeypatch, tmp_path):
    f = _install(reference_http, monkeypatch)
    monkeypatch.setattr(settings, "text_detector_mode", "candidate")
    monkeypatch.setattr(settings, "text_window_sentences", 3)
    monkeypatch.setattr(settings, "text_checkpoint_agreement_review", True)
    prefix = "The committee met on Monday morning to review the budget. "
    earlier_sentence = "The draft gives the survey date and meeting location."
    later_sentence = "The revision includes a detailed account of the survey and its outcomes."
    suffix = " The final minutes will be available after the next meeting."

    def pdf(sentence, layout="single"):
        with pymupdf.open() as document:
            page = document.new_page()
            right = 325 if layout == "columns" else 530
            assert page.insert_textbox((54, 55, right, 300), prefix + sentence + suffix, fontsize=11, lineheight=1.32) >= 0
            if layout == "columns":
                assert page.insert_textbox((350, 56, 530, 220), "Another column contains independent notes for the committee.", fontsize=11) >= 0
            elif layout == "two_pages":
                document.new_page().insert_text((54, 55), "The second page contains separate notes.")
            elif layout == "rotation":
                page.set_rotation(90)
            return document.tobytes()

    older, newer = pdf(earlier_sentence), pdf(later_sentence)
    _saved(f, "paired-pdf", ".pdf", newer)
    url = _url("paired-pdf")
    assert f.client.get(url, headers=f.viewer).json()["paired_check_available"] is True
    scored = []

    def score(candidate):
        scored.append(candidate)
        return [{"earlier": .1, "later": .4}, {"earlier": .3, "later": .6}]

    monkeypatch.setattr(paired, "score", score)
    standard = f.client.post(url, headers=f.viewer, files=_upload(older, ".pdf"))
    assert standard.status_code == 200 and not scored
    response = f.client.post(url, headers=f.viewer, files=_upload(older, ".pdf"),
                             data={"include_paired_check": "true", "include_pdf": "true"})
    assert response.status_code == 200, response.text
    payload = response.json()
    comparison = payload["comparison"]
    check = comparison["paired_check"]
    assert check["status"] == "joint_increase" and check["authorship_confidence_contribution"] == 0
    assert {k: v for k, v in comparison.items() if k != "paired_check"} == standard.json()["comparison"]
    assert [check[key]["text"] for key in ("earlier", "later")] == [earlier_sentence, later_sentence]
    assert check["change_ids"] == [change["id"] for change in comparison["changes"]]
    assert all(change[side]["pdf_pages"][0]["word_regions"] for change in comparison["changes"] for side in ("earlier", "later"))
    with pymupdf.open(stream=base64.b64decode(payload["pdf"]), filetype="pdf") as report:
        text = " ".join(" ".join(page.get_text().split()) for page in report)
        assert "Experimental paired writing check" in text and "Page 1 · highlighted source crop" in text
        assert "zero authorship confidence weight" in text and "ModernBERT" not in text and "Tropa" not in text
    old, new, exported = (tmp_path / name for name in ("earlier.pdf", "later.pdf", "comparison.json"))
    old.write_bytes(older)
    new.write_bytes(newer)
    exported.write_text(json.dumps(comparison))
    assert review(exported, old, new)["paired_scope_and_saved_response_arithmetic"] == "joint_increase"
    for layout in ("two_pages", "rotation", "columns"):
        aid = f"paired-pdf-{layout}"
        _saved(f, aid, ".pdf", pdf(later_sentence, layout))
        outcome = f.client.post(_url(aid), headers=f.viewer, files=_upload(pdf(earlier_sentence, layout), ".pdf"),
                                data={"include_paired_check": "true"})
        assert outcome.status_code == 200, outcome.text
        assert outcome.json()["comparison"]["paired_check"]["status"] == "outside_scope"
        assert outcome.json()["comparison"]["changes"]
    assert len(scored) == 1


def test_pdf_drafts_with_unmapped_or_locked_pages_are_refused():
    source = _pdf_draft("green", 0)
    with pymupdf.open() as document:
        blank = document.new_page()
        blank_data = document.tobytes()
        blank.insert_text((50, 50), "The text is locked.")
        locked = document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="review-fixture", owner_pw="review-owner")
    for unsupported in (blank_data, locked):
        with pytest.raises(ValueError, match="unlocked PDFs with mapped text on every page"):
            document_revision.compare(unsupported, source, ".pdf")


@pytest.mark.parametrize("extension", [".txt", ".docx"])
def test_opt_in_paired_check_replays_exact_focus_and_preserves_unscored_comparison(
    reference_http, monkeypatch, tmp_path, extension,
):
    f = _install(reference_http, monkeypatch)
    monkeypatch.setattr(settings, "text_detector_mode", "candidate")
    monkeypatch.setattr(settings, "text_window_sentences", 3)
    monkeypatch.setattr(settings, "text_checkpoint_agreement_review", True)
    lead = "The committee met on Monday morning to review the budget.\n"
    tail = " The final minutes will be available after the next meeting."
    old_focus = "The draft gives  the survey date and meeting location."
    new_focus = "The revision includes a detailed account of the survey and its outcomes."
    if extension == ".docx":
        prefix = "A shared opening.\n\n"
        earlier_file, later_file = (_word(lead + focus + tail, f"Excluded{index}")
                                    for index, focus in enumerate((old_focus, new_focus)))
    else:
        prefix = "Meeting minutes.\r\n\r\n"
        lead = lead.replace("\n", "\r\n")
        earlier_file, later_file = ((prefix + lead + focus + tail).encode() for focus in (old_focus, new_focus))
    earlier, later = (prefix + lead + focus + tail for focus in (old_focus, new_focus))
    _saved(f, "paired-draft", extension, later_file)
    url = _url("paired-draft")
    assert f.client.get(url, headers=f.viewer).json()["paired_check_available"] is True
    inputs = []

    def measure(candidate):
        inputs.append(candidate)
        return [{"earlier": .1, "later": .4}, {"earlier": .3, "later": .6}]

    monkeypatch.setattr(paired, "score", measure)
    ordinary = f.client.post(url, headers=f.viewer, files=_upload(earlier_file, extension))
    assert ordinary.status_code == 200 and "paired_check" not in ordinary.json()["comparison"] and not inputs
    response = f.client.post(url, headers=f.viewer, files=_upload(earlier_file, extension),
                             data={"include_pdf": "true", "include_paired_check": "true"})
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    body = response.json()
    comparison = body["comparison"]
    check = comparison["paired_check"]
    assert check["status"] == "joint_increase" and check["authorship_confidence_contribution"] == 0
    if extension == ".docx":
        assert "top-level body paragraph" in check["scope"]
        assert comparison["coordinate_space"] == "docx_body_unicode_codepoint"
        assert all(change[side]["paragraphs"][0]["paragraph"] == 3
                   for change in comparison["changes"] for side in ("earlier", "later"))
    assert {k: v for k, v in comparison.items() if k != "paired_check"} == ordinary.json()["comparison"]
    for side, source, focus in (("earlier", earlier, old_focus), ("later", later, new_focus)):
        excerpt = check[side]
        assert source[excerpt["start"]:excerpt["end"]] == excerpt["text"] == focus
        assert excerpt["context_spans"][1] == [excerpt["start"], excerpt["end"]]
        assert len(excerpt["context_spans"]) == 3
        assert excerpt["input_sha256"] == hashlib.sha256(getattr(inputs[0], side).encode()).hexdigest()
    assert check["change_ids"] == [change["id"] for change in comparison["changes"]]
    with pymupdf.open(stream=base64.b64decode(body["pdf"]), filetype="pdf") as report:
        text = " ".join(" ".join(page.get_text().split()) for page in report)
        assert "Experimental paired writing check" in text and "zero authorship confidence weight" in text
        assert "Check 1" in text and "Check 2" in text and "Linked wording changes C01" in text
        assert any(page.get_links() for page in report)
        assert "Tropa" not in text and "ModernBERT" not in text
    source_old, source_new, exported = (tmp_path / name for name in (f"earlier{extension}", f"later{extension}", "comparison.json"))
    source_old.write_bytes(earlier_file)
    source_new.write_bytes(later_file)
    exported.write_text(json.dumps(comparison))
    assert review(exported, source_old, source_new)["paired_scope_and_saved_response_arithmetic"] == "joint_increase"
    check["later"]["input_sha256"] = "0" * 64
    exported.write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="paired writing check does not replay"):
        review(exported, source_old, source_new)

    unchanged = f.client.post(url, headers=f.viewer, files=_upload(later_file, extension),
                              data={"include_paired_check": "true"})
    assert unchanged.status_code == 200 and unchanged.json()["comparison"]["paired_check"]["status"] == "outside_scope"
    if extension == ".docx":
        split = Document()
        for paragraph in (prefix.strip(), lead.strip(), old_focus + tail):
            split.add_paragraph(paragraph)
        file = BytesIO()
        split.save(file)
        different_layout = f.client.post(url, headers=f.viewer, files=_upload(file.getvalue(), extension),
                                         data={"include_paired_check": "true"})
        assert different_layout.status_code == 200
        assert different_layout.json()["comparison"]["paired_check"]["status"] == "outside_scope"
    assert len(inputs) == 1

    def unavailable(_candidate):
        raise RuntimeError("A private diagnostic")

    monkeypatch.setattr(paired, "score", unavailable)
    failure = f.client.post(url, headers=f.viewer, files=_upload(earlier_file, extension), data={"include_paired_check": "true"})
    assert failure.status_code == 200 and failure.json()["comparison"]["paired_check"]["status"] == "unavailable"
    assert "A private diagnostic" not in failure.text


def test_word_comparison_uses_exact_body_paragraph_coordinates_in_api_pdf_and_replay(reference_http, monkeypatch, tmp_path):
    f = _install(reference_http, monkeypatch)
    monkeypatch.setattr(settings, "text_checkpoint_agreement_review", False)
    texts = ("📎 Send the crème  brûlée on Tuesday.", "📎 Send the café  order on Thursday.")
    earlier, later = (_word(text, f"Excluded{index}") for index, text in enumerate(texts))
    _saved(f, "word-draft", ".DOCX", later)
    url = _url("word-draft")
    assert f.client.get(url, headers=f.viewer).json() == {"available": True, "extension": ".docx"}
    wrong = f.client.post(url, headers=f.viewer, files=_upload(earlier, ".txt"))
    assert wrong.status_code == 422
    bad = f.client.post(url, headers=f.viewer, files=_upload(b"Not a Word document", ".docx"))
    assert bad.status_code == 422
    response = f.client.post(url, headers=f.viewer, files=_upload(earlier, ".DOCX"), data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    result = response.json()
    comparison = result["comparison"]
    assert comparison["format"] == "docx" and comparison["coordinate_space"] == "docx_body_unicode_codepoint"
    assert len(comparison["changes"]) == 1 and "Excluded" not in response.text
    change = comparison["changes"][0]
    assert change["role"] == "body"
    for name, body in zip(("earlier", "later"), texts, strict=True):
        source = "A shared opening.\n\n" + body
        excerpt = change[name]
        (anchor,) = excerpt["paragraphs"]
        assert anchor["paragraph"] == 3 and anchor["body_index"] == 3
        assert excerpt["text"] == source[excerpt["start"]:excerpt["end"]] == body[anchor["start"]:anchor["end"]]
        assert all(excerpt["start"] <= a < b <= excerpt["end"] for a, b in excerpt["changed_tokens"])
        assert comparison[name]["decoded_text_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    with pymupdf.open(stream=base64.b64decode(result["pdf"]), filetype="pdf") as report:
        report_text = "\n".join(page.get_text() for page in report)
    for name in ("earlier", "later"):
        (anchor,) = change[name]["paragraphs"]
        assert f'Paragraph 3 [{anchor["start"]}, {anchor["end"]})' in report_text
    assert "Word body text" in report_text and "Excluded" not in report_text
    assert "brûlée" in report_text and "Thursday" in report_text

    old_path, new_path, export = (tmp_path / name for name in ("earlier.docx", "later.docx", "report.json"))
    old_path.write_bytes(earlier)
    new_path.write_bytes(later)
    export.write_text(json.dumps(comparison))
    assert review(export, old_path, new_path)["body_paragraph_ranges_verified"] == 2
    change["later"]["paragraphs"][0]["body_index"] -= 1
    export.write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="does not replay"):
        review(export, old_path, new_path)


def test_email_comparison_preserves_roles_and_exact_original_excerpts_in_json_and_pdf(reference_http, monkeypatch, tmp_path):
    f = _install(reference_http, monkeypatch)
    monkeypatch.setattr(settings, "text_checkpoint_agreement_review", False)
    previous = "📎 Send the crème  brûlée on Tuesday.\n\n-- \nOld signature\n\n> Review the earlier draft."
    latest = "📎 Send the café  order on Thursday.\n\n-- \nNew signature\n\n> Review the final draft."
    earlier = _email(previous)
    retained = _saved(f, "email-draft", ".eml", _email(latest))
    staged = tmp_path / "staged"
    staged.mkdir()
    monkeypatch.setattr(document_revision, "TemporaryDirectory", lambda **kw: TemporaryDirectory(dir=staged, **kw))

    def state():
        with Session(f.engine) as db:
            return (db.execute(select(Analysis.id, Analysis.result_hash).order_by(Analysis.id)).all(),
                    db.execute(select(Artefact.id)).all(), db.execute(select(BillingOrderRow.id)).all(),
                    sorted(path for path in f.store.root.rglob("*") if path.is_file()))

    before = state()
    assert f.client.get(_url("email-draft"), headers=f.viewer).json() == {"available": True, "extension": ".eml"}
    response = f.client.post(_url("email-draft"), headers=f.viewer, files=_upload(earlier), data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    result = response.json()
    comparison = result["comparison"]
    assert comparison["coordinate_space"] == "decoded_source_unicode_codepoint"
    assert comparison["earlier"]["sha256"] == hashlib.sha256(earlier).hexdigest()
    assert comparison["later"]["sha256"] == retained.sha256
    roles = {change["role"] for change in comparison["changes"]}
    assert roles == {"body", "quoted", "signature"}
    assert "Excluded" not in response.text
    for change in comparison["changes"]:
        for side, text in (("earlier", previous), ("later", latest)):
            excerpt = change[side]
            assert excerpt["text"] == text[excerpt["start"]:excerpt["end"]]
            assert all(excerpt["start"] <= a < b <= excerpt["end"] for a, b in excerpt["changed_tokens"])
    assert any("crème  brûlée" in change["earlier"]["text"] for change in comparison["changes"])
    with pymupdf.open(stream=base64.b64decode(result["pdf"], validate=True), filetype="pdf") as pdf:
        body = " ".join(" ".join(page.get_text().split()) for page in pdf)
        for part in ("crème brûlée", "café order", "Quoted text", "Signature", retained.sha256):
            assert part in body
        assert pdf.metadata["author"] == "Chipmunk Verify"
    assert not list(staged.iterdir()) and state() == before

    mislabeled = f.client.post(_url("email-draft"), headers=f.viewer, files=_upload(b"This is plain text with no email headers."))
    assert mislabeled.status_code == 422 and not list(staged.iterdir())


@pytest.mark.parametrize("html", [False, True])
def test_paired_email_checks_only_the_bound_body_context(reference_http, monkeypatch, tmp_path, html):
    f = _install(reference_http, monkeypatch)
    monkeypatch.setattr(settings, "text_detector_mode", "candidate")
    monkeypatch.setattr(settings, "text_window_sentences", 3)
    monkeypatch.setattr(settings, "text_checkpoint_agreement_review", True)
    heading = "The committee shared the schedule in a separate note."
    lead = "The committee met on Monday morning to review the budget. "
    tail = " The final minutes will be available after the next meeting."
    old_focus = "The draft gives the survey date and meeting location."
    new_focus = "The revision includes a detailed account of the survey and its outcomes."

    def message(focus, quote):
        paragraph = lead + focus + tail
        if html:
            return _html_email(f"<p>{heading}</p><p>{paragraph}</p>", quote=quote, cte="base64")
        return _email(f"{heading}\n\n{paragraph}\n\n-- \nMeeting coordinator\n\n> {quote}\n")

    older_quote, newer_quote = "Older quotation about projected revenue.", "Revised quotation about travel costs."
    earlier, later = message(old_focus, older_quote), message(new_focus, newer_quote)
    _saved(f, "paired-mail", ".eml", later)
    url = _url("paired-mail")
    assert f.client.get(url, headers=f.viewer).json()["paired_check_available"] is True
    inputs = []

    def measure(candidate):
        inputs.append((candidate.earlier, candidate.later))
        return [{"earlier": .1, "later": .4}, {"earlier": .3, "later": .6}]

    monkeypatch.setattr(paired, "score", measure)
    standard = f.client.post(url, headers=f.viewer, files=_upload(earlier)).json()["comparison"]
    assert "paired_check" not in standard and not inputs
    response = f.client.post(url, headers=f.viewer, files=_upload(earlier),
                             data={"include_pdf": "true", "include_paired_check": "true"})
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    payload = response.json()
    comparison = payload["comparison"]
    check = comparison["paired_check"]
    assert {key: value for key, value in comparison.items() if key != "paired_check"} == standard
    assert check["status"] == "joint_increase" and check["authorship_confidence_contribution"] == 0
    assert inputs == [(lead + old_focus + tail, lead + new_focus + tail)]
    assert check["change_ids"] == [change["id"] for change in comparison["changes"] if change["role"] == "body"]
    for side, original, focus in (("earlier", earlier, old_focus), ("later", later, new_focus)):
        mime = BytesParser(policy=policy.default).parsebytes(original).get_body(preferencelist=("plain", "html"))
        decoded = mime.get_content()
        if html:
            decoded = f"{heading}\n\n{lead}{focus}{tail}"
        row = check[side]
        assert decoded[row["start"]:row["end"]] == row["text"] == focus
        assert comparison[side]["mime_part"]
    with pymupdf.open(stream=base64.b64decode(payload["pdf"]), filetype="pdf") as report:
        text = " ".join(" ".join(page.get_text().split()) for page in report)
        assert "Experimental paired writing check" in text and "MIME part" in text
        assert "covers only the reply body" in text and "ModernBERT" not in text
        assert any(page.get_links() for page in report)
    old, new, report = (tmp_path / name for name in ("earlier.eml", "later.eml", "comparison.json"))
    old.write_bytes(earlier)
    new.write_bytes(later)
    report.write_text(json.dumps(comparison))
    assert review(report, old, new)["paired_scope_and_saved_response_arithmetic"] == "joint_increase"
    only_quote = f.client.post(url, headers=f.viewer, files=_upload(message(new_focus, older_quote)),
                               data={"include_paired_check": "true"})
    assert only_quote.status_code == 200
    assert only_quote.json()["comparison"]["paired_check"]["status"] == "outside_scope"
    assert len(inputs) == 1


def test_html_reply_comparison_preserves_split_words_and_mime_positions(reference_http, monkeypatch, tmp_path):
    f = _install(reference_http, monkeypatch)
    earlier = _html_email("<p>📎 Bring the ca<strong>f</strong>&#233; order.</p><p>Meet at noon.</p>")
    later = _html_email("<p>📎 Bring the bi<em>st</em>ro order.</p><p>Meet at noon.</p>",
                        quote="Confidential quote change.", cte="base64")
    _saved(f, "html-reply", ".eml", later)
    response = f.client.post(_url("html-reply"), headers=f.viewer, files=_upload(earlier), data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    result = response.json()
    comparison = result["comparison"]
    assert comparison["schema"] == "chipmunk.document-revision-comparison/2"
    assert comparison["coordinate_space"] == document_revision.HTML_COORDINATES
    (change,) = comparison["changes"]
    assert change["role"] == "body"
    assert "Confidential quote change" not in response.text and "Private reference sheet" not in response.text
    for side, data, expected, pieces in (("earlier", earlier, "café", ["ca", "f", "&#233;"]),
                                         ("later", later, "bistro", ["bi", "st", "ro"])):
        part, _ = BytesParser(policy=policy.default).parsebytes(data).iter_parts()
        raw = part.get_content()
        assert comparison[side]["mime_part"] == "1.1"
        assert comparison[side]["decoded_html_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
        assert comparison[side]["decoded_html_characters"] == len(raw)
        excerpt = change[side]
        assert excerpt["text"] == expected and excerpt["start"] == len("📎 Bring the ")
        (token,) = excerpt["html_tokens"]
        assert [[token["start"], token["end"]]] == excerpt["changed_tokens"]
        assert [raw[a:b] for a, b in token["source_ranges"]] == pieces
        assert "".join(unescape(piece) for piece in pieces) == expected
    with pymupdf.open(stream=base64.b64decode(result["pdf"], validate=True), filetype="pdf") as pdf:
        body = " ".join(" ".join(page.get_text().split()) for page in pdf)
        assert "Decoded HTML positions" in body and "MIME part 1.1" in body
        assert all(word in body for word in ("café", "bistro")) and "Confidential quote change" not in body
        for side in ("earlier", "later"):
            assert all(f"[{a}, {b})" in body for a, b in change[side]["html_tokens"][0]["source_ranges"])

    old_path, new_path, export = (tmp_path / name for name in ("earlier.eml", "later.eml", "report.json"))
    old_path.write_bytes(earlier)
    new_path.write_bytes(later)
    export.write_text(json.dumps(comparison))
    assert review(export, old_path, new_path)["decoded_html_ranges_verified"] == 6
    change["later"]["html_tokens"][0]["source_ranges"][0][0] += 1
    export.write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="does not replay"):
        review(export, old_path, new_path)


@pytest.mark.parametrize("paragraphs", [2, 3, 5, 6])
def test_html_reply_comparison_ignores_quote_and_format_changes(paragraphs):
    tail = "".join(f"<p>Agenda item {i} is unchanged.</p>" for i in range(1, paragraphs))
    earlier = _html_email("<p>Crème &amp; tea.</p>" + tail)
    same = _html_email("<p><strong>Cr&#232;me</strong> &#38; tea.</p>" + tail, quote="Unrelated new quote.", cte="base64")
    deleted = _html_email("<p>Crème.</p>" + tail)
    assert document_revision.compare(earlier, same, ".eml")["changes"] == []
    (change,) = document_revision.compare(earlier, deleted, ".eml")["changes"]
    assert change["role"] == "body" and change["earlier"]["text"] == "& tea"
    assert len(change["earlier"]["html_tokens"]) == 2 and change["later"]["html_tokens"] == []


def test_html_reply_comparison_refuses_ambiguous_or_mixed_body_formats():
    valid = _html_email("<p>First paragraph.</p><p>Next paragraph.</p>")
    for body in ('<p style="display:none">Hidden text.</p><p>Next paragraph.</p>',
                 "<p>First paragraph.</p><blockquote><p>Interleaved quote.</p></blockquote><p>Next paragraph.</p>",
                 "<p>First paragraph.</p><p>--\nSignature inside body.</p>"):
        with pytest.raises(ValueError, match="Use HTML-only replies"):
            document_revision.compare(_html_email(body), valid, ".eml")
    with pytest.raises(ValueError, match="same body type"):
        document_revision.compare(_email("Two paragraphs.\n\nThis is the second."), valid, ".eml")


def test_auth_scope_format_mismatch_and_retained_later_integrity(reference_http, monkeypatch):
    f = _install(reference_http, monkeypatch)
    retained = _saved(f, "text-draft", ".txt", "🙂 Crème brûlée.".encode())
    _saved(f, "pending-draft", ".txt", b"Queued draft", status="queued")
    _saved(f, "csv-draft", ".csv", b"Outside the draft comparison")
    url = _url("text-draft")
    earlier = _upload("🚀 Déjà vu.".encode(), ".txt")
    assert f.client.post(url, files=earlier).status_code == 401
    assert f.client.post(url, headers=f.outsider, files=earlier).status_code == 404
    assert f.client.post(_url("pending-draft"), headers=f.viewer, files=earlier).status_code == 404
    assert f.client.get(_url("csv-draft"), headers=f.viewer).json()["available"] is False
    assert f.client.post(url, headers=f.viewer, files=_upload(_email("A different file type"))).status_code == 422
    assert f.client.post(url, headers=f.viewer, files=_upload(b"x" * (document_revision.MAX_BYTES + 1), ".txt")).status_code == 413
    for setting, value in (("environment", "prod"), ("document_revision_comparison_enabled", False)):
        with monkeypatch.context() as patch:
            patch.setattr(settings, setting, value)
            assert not f.client.get(url, headers=f.viewer).json()["available"]
            assert f.client.post(url, headers=f.viewer, files=earlier).status_code == 404

    f.client.cookies.set(auth.SESSION_COOKIE, f.session)
    assert f.client.post(url, files=earlier).status_code == 403
    named = f.client.post(url, headers={auth.CSRF_HEADER: f.csrf}, files=earlier)
    assert named.status_code == 200, named.text
    assert named.json()["comparison"]["changes"][0]["earlier"]["text"] == "🚀 Déjà vu"

    retained.path.chmod(0o644)
    retained.path.write_bytes(retained.path.read_bytes()[:-1] + b"!")
    assert f.client.post(url, headers=f.viewer, files=earlier).status_code == 409
    retained.path.unlink()
    assert not f.client.get(url, headers=f.viewer).json()["available"]
    assert f.client.post(url, headers=f.viewer, files=earlier).status_code == 404


def test_refuses_an_attachment_selected_as_the_email_body(reference_http, monkeypatch):
    f = _install(reference_http, monkeypatch)
    ambiguous = EmailMessage()
    ambiguous["From"] = "fixture@example.invalid"
    ambiguous.make_mixed()
    named = EmailMessage()
    named.set_content("Private attachment contents")
    named.set_param("name", "private.txt", header="Content-Type")
    body = EmailMessage()
    body.set_content("This is the actual message body.")
    ambiguous.attach(named)
    ambiguous.attach(body)
    assert ambiguous.get_body(preferencelist=("plain",)).get_filename() == "private.txt"
    _saved(f, "ambiguous-later", ".eml", ambiguous.as_bytes())
    _saved(f, "plain-later", ".eml", _email("The later message body."))
    for aid, content in (("ambiguous-later", _email("An earlier message body.")), ("plain-later", ambiguous.as_bytes())):
        response = f.client.post(_url(aid), headers=f.viewer, files=_upload(content), data={"include_pdf": "true"})
        assert response.status_code == 422 and "is an attachment" in response.text
        assert "Private attachment contents" not in response.text


def test_inactive_revisions_are_hidden_and_refused(reference_http, monkeypatch):
    f = _install(reference_http, monkeypatch)
    _saved(f, "withdrawn-draft", ".txt", b"This was withdrawn")
    _saved(f, "old-draft", ".txt", b"This was superseded")
    _saved(f, "new-draft", ".txt", b"This is the active draft")
    with Session(f.engine) as db:
        db.get(Analysis, "withdrawn-draft").withdrawn_at = datetime.now(UTC)
        db.get(Analysis, "new-draft").supersedes_id = "old-draft"
        db.commit()
    for aid, reason in (("withdrawn-draft", "Revision withdrawn"), ("old-draft", "Revision superseded")):
        assert not f.client.get(_url(aid), headers=f.viewer).json()["available"]
        response = f.client.post(_url(aid), headers=f.viewer, files=_upload(b"The earlier wording", ".txt"))
        assert response.status_code == 409 and reason in response.text
    assert f.client.get(_url("new-draft"), headers=f.viewer).json()["available"]


def test_pdf_marks_only_changed_code_points_and_escapes_excerpt():
    result = document_revision.compare("🚀 Déjà <end>".encode(), "🙂 Déjà <début>".encode(), ".txt")
    earlier, later = result["changes"][0]["earlier"], result["changes"][0]["later"]
    assert "<mark>🚀</mark> Déjà &lt;<mark>end</mark>" in _side("Earlier draft", earlier)
    assert "<mark>🙂</mark> Déjà &lt;<mark>début</mark>" in _side("Analyzed version", later)


def test_adjacent_sentence_pair_uses_exact_text_exports_and_preserves_legacy_replay(reference_http, monkeypatch, tmp_path):
    f = _install(reference_http, monkeypatch)
    monkeypatch.setattr(settings, "text_detector_mode", "candidate")
    monkeypatch.setattr(settings, "text_window_sentences", 3)
    monkeypatch.setattr(settings, "text_checkpoint_agreement_review", True)
    first = "The committee met on Monday morning to review the budget."
    last = "The final minutes will be available after the next meeting."
    old_focus = "The draft gives the survey date and location.\r\n The registration desk opens early."
    new_focus = "The report explains when and where the survey took place.\r\n Attendees can register before the meeting."
    old, new = (f"{first} {focus} {last}".encode() for focus in (old_focus, new_focus))
    _saved(f, "two-sentence-draft", ".txt", new)
    url = _url("two-sentence-draft")
    measured = []

    def score(candidate):
        measured.append(candidate)
        return [{"earlier": .1, "later": .4}, {"earlier": .3, "later": .6}]

    monkeypatch.setattr(paired, "score", score)
    form = {"include_paired_check": "true", "include_pdf": "true"}
    response = f.client.post(url, headers=f.viewer, files=_upload(old, ".txt"), data=form)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    payload = response.json()
    comparison = payload["comparison"]
    check = comparison["paired_check"]
    assert check["schema"] == paired.WIDE_SCHEMA and check["status"] == "joint_increase"
    assert check["authorship_confidence_contribution"] == 0
    assert check["change_ids"] == [change["id"] for change in comparison["changes"]]
    assert len(measured) == 1
    for side, raw, focus in (("earlier", old, old_focus), ("later", new, new_focus)):
        excerpt = check[side]
        assert raw.decode()[excerpt["start"]:excerpt["end"]] == excerpt["text"] == focus
        spans = excerpt["context_spans"]
        assert len(spans) == 4 and [spans[1][0], spans[2][1]] == [excerpt["start"], excerpt["end"]]
        assert excerpt["input_sha256"] == hashlib.sha256(getattr(measured[0], side).encode()).hexdigest()
    with pymupdf.open(stream=base64.b64decode(payload["pdf"]), filetype="pdf") as document:
        text = " ".join(" ".join(page.get_text().split()) for page in document)
        assert "Earlier sentences [" in text and "Analyzed sentences [" in text
        assert "two adjacent changed sentences" in text and "zero authorship confidence weight" in text
        assert "ModernBERT" not in text and "Tropa" not in text
        assert any(page.get_links() for page in document)

    earlier_path, later_path, json_path = (tmp_path / name for name in ("old.txt", "new.txt", "comparison.json"))
    earlier_path.write_bytes(old)
    later_path.write_bytes(new)
    json_path.write_text(json.dumps(comparison))
    assert review(json_path, earlier_path, later_path)["paired_scope_and_saved_response_arithmetic"] == "joint_increase"
    check["later"]["context_spans"][2][0] += 1
    json_path.write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="paired writing check does not replay"):
        review(json_path, earlier_path, later_path)

    comparison["paired_check"] = paired.result(None, draft_format="plain_text", allow_wide=False)
    json_path.write_text(json.dumps(comparison))
    assert review(json_path, earlier_path, later_path)["paired_scope_and_saved_response_arithmetic"] == "outside_scope"
    crossed = old.replace(b"\r\n ", b"\r\n\r\n")
    nonadjacent = old.replace(b"The final minutes", b"Further minutes")
    for unsupported in (new, crossed, nonadjacent):
        result = f.client.post(url, headers=f.viewer, files=_upload(unsupported, ".txt"), data={
            "include_paired_check": "true",
        })
        assert result.status_code == 200
        check = result.json()["comparison"]["paired_check"]
        assert check["schema"] == paired.WIDE_SCHEMA and check["status"] == "outside_scope"
    assert len(measured) == 1

    for extension, source in (
        (".docx", _word(new.decode().replace("\r\n", " "), "later")),
        (".eml", _email(new.decode().replace("\r\n", " "))),
    ):
        earlier_text = old.decode().replace("\r\n", " ")
        prior = _word(earlier_text, "earlier") if extension == ".docx" else _email(earlier_text)
        aid = "same-two-sentences-" + extension[1:]
        _saved(f, aid, extension, source)
        result = f.client.post(_url(aid), headers=f.viewer, files=_upload(prior, extension), data={
            "include_paired_check": "true",
        })
        assert result.status_code == 200
        check = result.json()["comparison"]["paired_check"]
        assert check["schema"] == paired.SCHEMA and check["status"] == "outside_scope"
    assert len(measured) == 1
