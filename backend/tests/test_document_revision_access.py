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
from docx.oxml.ns import qn
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


def _word_boundary(center, *, interposed=None, relocated=False):
    word = Document()
    word.add_paragraph("A published introduction.")
    word.add_table(rows=1, cols=1).cell(0, 0).text = "Omitted table text."
    opening = "The committee approved the earlier draft after its last meeting."
    tail = "The published article records the procedures for the next review."
    if relocated:
        word.add_paragraph(opening)
    else:
        word.add_paragraph(f"{opening} {center}")
    if interposed == "table":
        word.add_table(rows=1, cols=1).cell(0, 0).text = "Intervening table."
    elif interposed == "blank":
        word.add_paragraph()
    word.add_paragraph(f"{center} {tail}" if relocated else tail)
    file = BytesIO()
    word.save(file)
    return file.getvalue()


def test_word_boundary_maps_both_native_paragraphs_and_preserves_opt_out(reference_http, monkeypatch, tmp_path):
    f = _install(reference_http, monkeypatch)
    monkeypatch.setattr(settings, "text_detector_mode", "candidate")
    monkeypatch.setattr(settings, "text_window_sentences", 3)
    monkeypatch.setattr(settings, "text_checkpoint_agreement_review", True)
    monkeypatch.setattr(paired, "_ocr_language", lambda _: {"code": "en", "status": "supported"})
    old_focus = "The editor checked the original version for clarity and accuracy."
    new_focus = "The editor carefully examined the complete draft to improve its organization and readability."
    earlier_file, later_file = (_word_boundary(focus) for focus in (old_focus, new_focus))
    _saved(f, "paired-word-boundary", ".docx", later_file)
    url = _url("paired-word-boundary")
    requests = []

    def measure(candidate):
        requests.append(candidate)
        context = [{"earlier": .2, "later": .6}, {"earlier": .2, "later": .6}]
        exact = [{"earlier": .4, "later": .6}, {"earlier": .4, "later": .6}]
        return context, exact

    def submit(file=earlier_file, include_paired=True):
        response = f.client.post(url, headers=f.viewer, files=_upload(file, ".docx"), data={
            "include_pdf": "true", "include_paired_check": str(include_paired).lower(),
        })
        assert response.status_code == 200
        return response.json()

    monkeypatch.setattr(paired, "score_boundary", measure)
    monkeypatch.setattr(settings, "document_paired_boundary_review", False)
    prior = submit()["comparison"]
    assert prior["paired_check"]["schema"] == paired.SCHEMA
    assert prior["paired_check"]["status"] == "outside_scope"
    assert "earlier" not in prior["paired_check"]
    monkeypatch.setattr(settings, "document_paired_boundary_review", True)
    ordinary = submit(include_paired=False)["comparison"]
    assert "paired_check" not in ordinary and not requests
    payload = submit()
    report = payload["comparison"]
    check = report["paired_check"]
    assert {k: v for k, v in report.items() if k != "paired_check"} == ordinary
    assert check["schema"] == paired.WORD_BOUNDARY_SCHEMA
    assert check["status"] == "joint_increase" and check["authorship_confidence_contribution"] == 0
    for side, source, focus in (("earlier", earlier_file, old_focus), ("later", later_file, new_focus)):
        paragraph_text = {n: (index, p.text) for n, index, p in document_revision.docx_body_paragraphs(source)}
        row = check[side]
        assert row["text"] == focus
        assert [(p["paragraph"], p["body_index"]) for p in row["context_paragraphs"]] == [(2, 2), (2, 2), (3, 3)]
        contexts = [paragraph_text[p["paragraph"]][1][p["start"]:p["end"]] for p in row["context_paragraphs"]]
        assert contexts[1] == focus and row["input_sha256"] == paired.text_hash(" ".join(contexts))
    with pymupdf.open(stream=base64.b64decode(payload["pdf"]), filetype="pdf") as pdf:
        text = " ".join(" ".join(page.get_text().split()) for page in pdf)
        assert "Previous sentence · Paragraph 2" in text and "Next sentence · Paragraph 3" in text
        assert "Changed sentence · Paragraph 2" in text and "Changed sentence alone" in text
        assert "Linked wording changes C01" in text and any(page.get_links() for page in pdf)
    earlier_path, later_path, export = (tmp_path / name for name in ("earlier.docx", "later.docx", "comparison.json"))
    earlier_path.write_bytes(earlier_file)
    later_path.write_bytes(later_file)
    export.write_text(json.dumps(prior))
    assert review(export, earlier_path, later_path)["paired_scope_and_saved_response_arithmetic"] == "outside_scope"
    export.write_text(json.dumps(report))
    assert review(export, earlier_path, later_path)["paired_word_sentence_ranges_verified"] == 6
    check["later"]["context_paragraphs"][2]["body_index"] -= 1
    export.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="does not replay"):
        review(export, earlier_path, later_path)
    for layout in ("table", "blank"):
        unsafe = _word_boundary(old_focus, interposed=layout)
        assert submit(unsafe)["comparison"]["paired_check"]["status"] == "outside_scope"
    moved = _word_boundary(old_focus, relocated=True)
    assert submit(moved)["comparison"]["paired_check"]["status"] == "outside_scope"
    assert len(requests) == 1


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
    assert f.client.get(url, headers=f.viewer).json() == {
        "available": True, "extension": ".pdf", "earlier_extensions": [".txt", ".docx", ".pdf"],
    }
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


def test_paired_txt_neighboring_context_preserves_the_changed_sentence_and_replays(reference_http, monkeypatch, tmp_path):
    f = _install(reference_http, monkeypatch)
    monkeypatch.setattr(settings, "text_detector_mode", "candidate")
    monkeypatch.setattr(settings, "text_window_sentences", 3)
    monkeypatch.setattr(settings, "text_checkpoint_agreement_review", True)
    earlier = ("The committee met on Monday morning to review the budget. The draft gives  the survey date and meeting "
               "location. The final minutes will be available after the next meeting. "
               "All attendees will receive a copy of those minutes at the end of the week.")
    later = earlier.replace("The draft gives  the survey date and meeting location.",
                            "The revision includes a detailed account of the survey and its outcomes.")
    _saved(f, "paired-contexts", ".txt", later.encode())
    url = _url("paired-contexts")
    measured = []

    def measure(inputs, *, allow_partial, deadline):
        assert allow_partial and deadline.budget_s == paired.REVIEW_SECONDS and len(inputs) == 4
        measured.append(inputs)
        return [[.15, .18, .1, .5], [.2, .23, .3, .6]]

    monkeypatch.setattr(paired, "_score", measure)
    form = {"include_pdf": "true", "include_paired_check": "true"}
    result = f.client.post(url, headers=f.viewer, files=_upload(earlier.encode(), ".txt"), data=form)
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    body = result.json()
    comparison = body["comparison"]
    check = comparison["paired_check"]
    assert check["schema"] == paired.INTERIOR_SCHEMA and check["status"] == "joint_increase"
    assert check["authorship_confidence_contribution"] == 0 and check["change_ids"] == ["C01"]
    assert check["later"]["text"] == "The revision includes a detailed account of the survey and its outcomes."
    neighbor, = check["neighboring_contexts"]
    assert neighbor["status"] == "joint_increase"
    for side, source, model_input in zip(("earlier", "later"), (earlier, later), measured[0][2:], strict=True):
        row = neighbor[side]
        assert row["text"] == source[row["start"]:row["end"]]
        assert row["context_spans"][0] == [check[side]["start"], check[side]["end"]]
        assert row["input_sha256"] == paired.text_hash(model_input)
    with pymupdf.open(stream=base64.b64decode(body["pdf"]), filetype="pdf") as report:
        text = " ".join(" ".join(page.get_text().split()) for page in report)
        assert "Additional scored contexts" in text and "Linked wording changes C01" in text
        assert f"Analyzed sentence [{check['later']['start']}, {check['later']['end']})" in text
        assert f"Analyzed scored context [{neighbor['later']['start']}, {neighbor['later']['end']})" in text
        assert "Tropa" not in text and "ModernBERT" not in text
    old_path, new_path, export = (tmp_path / path for path in ("earlier.txt", "later.txt", "comparison.json"))
    old_path.write_text(earlier)
    new_path.write_text(later)
    export.write_text(json.dumps(comparison))
    assert review(export, old_path, new_path)["paired_scope_and_saved_response_arithmetic"] == "joint_increase"
    neighbor["later"]["context_spans"][2][0] += 1
    export.write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="paired writing check does not replay"):
        review(export, old_path, new_path)

    monkeypatch.setattr(paired, "_score", lambda *a, **k: [[.15, .18, None, None], [.2, .23, .3, .6]])
    fallback = f.client.post(url, headers=f.viewer, files=_upload(earlier.encode(), ".txt"), data=form)
    ordinary = fallback.json()["comparison"]["paired_check"]
    assert ordinary["schema"] == paired.SCHEMA and ordinary["status"] == "no_joint_increase"
    assert "neighboring_contexts" not in ordinary
    assert paired.verify(fallback.json()["comparison"], earlier, later, ordinary) == "no_joint_increase"

    from app.engine.models import MemoryBudgetExceeded

    deadlines = []

    def limited(inputs, *, allow_partial=False, deadline):
        deadlines.append(deadline)
        if allow_partial:
            raise MemoryBudgetExceeded("Test activation limit")
        assert len(inputs) == 2
        return [[.15, .18], [.2, .23]]

    monkeypatch.setattr(paired, "_score", limited)
    reduced = f.client.post(url, headers=f.viewer, files=_upload(earlier.encode(), ".txt"), data=form)
    assert reduced.json()["comparison"]["paired_check"] == ordinary
    assert len(deadlines) == 2 and deadlines[0] is deadlines[1]


def test_word_comparison_uses_exact_body_paragraph_coordinates_in_api_pdf_and_replay(reference_http, monkeypatch, tmp_path):
    f = _install(reference_http, monkeypatch)
    monkeypatch.setattr(settings, "text_checkpoint_agreement_review", False)
    texts = ("📎 Send the crème  brûlée on Tuesday.", "📎 Send the café  order on Thursday.")
    earlier, later = (_word(text, f"Excluded{index}") for index, text in enumerate(texts))
    _saved(f, "word-draft", ".DOCX", later)
    url = _url("word-draft")
    assert f.client.get(url, headers=f.viewer).json() == {
        "available": True, "extension": ".docx", "earlier_extensions": [".txt", ".docx", ".pdf"],
    }
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


def _word_with_table(value, *, label="Review", layout="simple", bold=False, body=True):
    word = Document()
    if body:
        word.add_paragraph("A shared opening paragraph.")
    if layout == "moved":
        word.add_paragraph("Extra body paragraph.")
    table = word.add_table(rows=3, cols=2)
    for row, cells in enumerate((("Item", "Wording"), (label, value), ("Second row", "Unchanged"))):
        for col, text in enumerate(cells):
            table.cell(row, col).paragraphs[0].add_run(text).bold = bold
    if layout == "merged":
        table.cell(1, 0).merge(table.cell(2, 0))
    if layout == "nested":
        table.cell(2, 1).add_table(rows=1, cols=1)
    if layout == "tracked":
        table.cell(2, 1).paragraphs[0]._p.append(OxmlElement("w:ins"))
    out = BytesIO()
    word.save(out)
    return out.getvalue()


def test_word_table_comparison_keeps_exact_native_cell_ranges_across_api_pdf_and_replay(reference_http, monkeypatch, tmp_path):
    f = _install(reference_http, monkeypatch)
    monkeypatch.setattr(settings, "text_checkpoint_agreement_review", False)
    before, after = "📎 Send the crème  brûlée Tuesday.", "📎 Send the café  order Thursday."
    earlier, later = _word_with_table(before), _word_with_table(after, bold=True)
    _saved(f, "word-table-draft", ".docx", later)
    response = f.client.post(_url("word-table-draft"), headers=f.viewer, files=_upload(earlier, ".docx"), data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    result = response.json()
    comparison = result["comparison"]
    assert comparison["schema"] == "chipmunk.document-revision-comparison/5" and not comparison["changes"]
    tables = comparison["word_tables"]
    assert tables["coordinate_space"] == "docx_table_cell_unicode_codepoint" and not tables["skipped"]
    (table,) = tables["compared"]
    assert (table["earlier_body_index"], table["later_body_index"], table["earlier_table"], table["later_table"]) == (1, 1, 1, 1)
    assert all((change["row"], change["column"]) == (2, 2) for change in table["changes"])
    for change in table["changes"]:
        for key, source in (("earlier", before), ("later", after)):
            cell = change[key]
            assert cell["text"] == source[cell["start"]:cell["end"]]
            assert cell["paragraphs"] == [{"paragraph": 1, "cell_index": 1, "start": cell["start"], "end": cell["end"]}]
            assert all(cell["start"] <= a < b <= cell["end"] for a, b in cell["changed_tokens"])
    with pymupdf.open(stream=base64.b64decode(result["pdf"]), filetype="pdf") as report:
        text = " ".join(" ".join(page.get_text().split()) for page in report)
    assert "Word table wording changes" in text and "Row 2, column 2" in text
    assert "brûlée" in text and "Thursday" in text
    a, b, export = (tmp_path / name for name in ("earlier.docx", "later.docx", "report.json"))
    a.write_bytes(earlier)
    b.write_bytes(later)
    export.write_text(json.dumps(comparison))
    assert review(export, a, b)["word_cell_token_ranges_verified"] >= 6
    table["changes"][0]["row"] += 1
    export.write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="does not replay"):
        review(export, a, b)
    historical, *_ = document_revision.compare_with_sources(earlier, later, ".docx", include_word_tables=False)
    export.write_text(json.dumps(historical))
    assert "word_tables_compared" not in review(export, a, b)
    strict, *_ = document_revision.compare_with_sources(earlier, later, ".docx", strict_word_tables=True)
    assert strict["schema"] == "chipmunk.document-revision-comparison/4"
    export.write_text(json.dumps(strict))
    assert review(export, a, b)["word_cell_token_ranges_verified"] >= 6


@pytest.mark.parametrize(("layout", "label", "reason"), [
    ("merged", "Review", "unsupported_layout"), ("nested", "Review", "unsupported_layout"),
    ("tracked", "Review", "unsupported_layout"), ("moved", "Review", "no_matching_position"),
])
def test_word_table_comparison_discloses_skips_and_preserves_other_text(layout, label, reason):
    first = _word_with_table("Earlier value")
    second = _word_with_table("Later value", layout=layout, label=label)
    coverage = document_revision.compare(first, second, ".docx")["word_tables"]
    assert not coverage["compared"]
    assert {row["reason"] for row in coverage["skipped"]} == {reason}


def test_word_table_keeps_matching_rows_when_one_label_changed():
    first = _word_with_table("Earlier value")
    second = _word_with_table("Later value", label="Changed row label")
    coverage = document_revision.compare(first, second, ".docx")["word_tables"]
    [table] = coverage["compared"]
    assert table["unchecked_rows"] == {"earlier": 1, "later": 1}
    assert not table["changes"] and not coverage["skipped"]


def _word_alignment(rows, *, moved=False, duplicate_caption=False):
    word = Document()
    word.add_paragraph("A stable introduction.")
    if moved:
        word.add_paragraph("A new paragraph before the table.")
    for _ in range(1 + duplicate_caption):
        table = word.add_table(rows=len(rows), cols=len(rows[0]))
        caption = OxmlElement("w:tblCaption")
        caption.set(qn("w:val"), "Historical estimates")
        table._tbl.tblPr.append(caption)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                table.cell(i, j).text = value
    out = BytesIO()
    word.save(out)
    return out.getvalue()


def test_word_table_aligns_moved_rows_and_columns_with_exact_coverage(reference_http, monkeypatch, tmp_path):
    old = _word_alignment([("Key", "Value", "Units"), ("Stable row", "13", "count"), ("Removed row", "20", "count")])
    new = _word_alignment([("Key", "Units", "Value"), ("Added row", "count", "99"), ("Stable row", "count", "14")], moved=True)
    f = _install(reference_http, monkeypatch)
    _saved(f, "aligned-word-draft", ".docx", new)
    response = f.client.post(_url("aligned-word-draft"), headers=f.viewer, files=_upload(old, ".docx"), data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    result = response.json()
    comparison = result["comparison"]
    [table] = comparison["word_tables"]["compared"]
    [change] = table["changes"]
    assert table["match"] == "caption" and table["earlier_body_index"] == 1 and table["later_body_index"] == 2
    assert table["unchecked_rows"] == {"earlier": 1, "later": 1}
    assert (change["earlier_row"], change["earlier_column"], change["row"], change["column"]) == (2, 2, 3, 3)
    assert [change[side]["text"] for side in ("earlier", "later")] == ["13", "14"]
    with pymupdf.open(stream=base64.b64decode(result["pdf"]), filetype="pdf") as report:
        text = " ".join(" ".join(page.get_text().split()) for page in report)
    assert "Earlier table 1 · Row 2, column 2" in text and "Analyzed table 1 · Row 3, column 3" in text
    assert "Unchecked rows: 1 earlier, 1 analyzed." in text
    a, b, export = (tmp_path / name for name in ("earlier.docx", "analyzed.docx", "comparison.json"))
    a.write_bytes(old)
    b.write_bytes(new)
    export.write_text(json.dumps(comparison))
    assert review(export, a, b)["word_cell_token_ranges_verified"] == 2
    change["earlier_column"] = 3
    export.write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="does not replay"):
        review(export, a, b)


def test_word_table_skips_duplicate_captions_and_only_unchecked_duplicate_rows_and_columns():
    head = ("Key", "Value", "Units")
    first = _word_alignment([head, ("One", "13", "number"), ("Two", "14", "number"), ("Three", "15", "number")])
    changed = [head, ("One", "13", "number"), ("One", "999", "number"), ("Three", "16", "number")]
    duplicate_caption = document_revision.compare(first, _word_alignment(changed, duplicate_caption=True), ".docx")["word_tables"]
    assert not duplicate_caption["compared"]
    assert {table["reason"] for table in duplicate_caption["skipped"]} == {"caption_unmatched"}
    [table] = document_revision.compare(first, _word_alignment(changed), ".docx")["word_tables"]["compared"]
    assert table["unchecked_rows"] == {"earlier": 2, "later": 2}
    [change] = table["changes"]
    assert change["later"]["text"] == "16"
    repeated = [("Key", "Value", "Value"), *changed[1:]]
    assert not document_revision.compare(first, _word_alignment(repeated), ".docx")["word_tables"]["compared"]


def test_word_table_comparison_has_a_total_token_cap_and_ignores_formatting():
    earlier = _word_with_table("Unchanged")
    reformatted = _word_with_table("Unchanged", bold=True)
    [table] = document_revision.compare(earlier, reformatted, ".docx")["word_tables"]["compared"]
    assert not table["changes"]
    oversized = _word_with_table(" ".join(f"word{i}" for i in range(2001)))
    result = document_revision.compare(earlier, oversized, ".docx")["word_tables"]
    assert not result["compared"] and result["skipped"][0]["reason"] == "comparison_limit"


def test_word_table_can_be_the_only_readable_text_in_both_drafts():
    earlier, later = (_word_with_table(text, body=False) for text in ("Before", "After"))
    comparison = document_revision.compare(earlier, later, ".docx")
    assert not comparison["changes"] and comparison["earlier"]["decoded_characters"] == 0
    assert comparison["word_tables"]["compared"][0]["changes"][0]["later"]["text"] == "After"


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


def test_cross_format_word_to_pdf_comparison_replays_both_source_locations(reference_http, monkeypatch, tmp_path):
    f = _install(reference_http, monkeypatch)
    monkeypatch.setattr(settings, "text_checkpoint_agreement_review", True)
    from scripts.rebuild_writing_word_boundary_sources import docx

    earlier = docx("The shipment will use amber packaging.")
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((60, 80), "The shipment will use green packaging.")
        later = document.tobytes()
    _saved(f, "mixed-draft", ".pdf", later)
    response = f.client.post(_url("mixed-draft"), headers=f.viewer, files=_upload(earlier, ".docx"),
                             data={"include_pdf": "true", "include_paired_check": "true"})
    assert response.status_code == 200, response.text
    payload = response.json()
    result = payload["comparison"]
    assert result["schema"] == "chipmunk.document-revision-comparison/3" and result["format"] == "mixed"
    assert result["earlier"]["coordinate_space"] == "docx_body_unicode_codepoint"
    assert result["later"]["coordinate_space"] == "pdf_native_text_unicode_codepoint"
    assert "paired_check" not in result
    [change] = result["changes"]
    assert change["earlier"]["text"] == "amber" and change["later"]["text"] == "green"
    assert change["earlier"]["paragraphs"] == [{"paragraph": 1, "body_index": 0, "start": 22, "end": 27}]
    [location] = change["later"]["pdf_pages"]
    assert location["page"] == 1 and location["word_regions"][0]["start"] == 22
    assert hashlib.sha256(base64.b64decode(location["preview"]["crop_png"])).hexdigest() == location["preview"]["crop_sha256"]
    with pymupdf.open(stream=base64.b64decode(payload["pdf"]), filetype="pdf") as report:
        report_text = " ".join(" ".join(page.get_text().split()) for page in report)
        assert "Earlier draft · Word body text" in report_text and "Analyzed version · PDF text layer" in report_text
        assert "Page 1 · highlighted source crop" in report_text
    earlier_path, later_path, export = (tmp_path / name for name in ("earlier.docx", "later.pdf", "draft.json"))
    earlier_path.write_bytes(earlier)
    later_path.write_bytes(later)
    export.write_text(json.dumps(result))
    assert review(export, earlier_path, later_path)["body_paragraph_ranges_verified"] == 1
    assert review(export, earlier_path, later_path)["pdf_previews_verified"] == 1
    result["earlier"]["coordinate_space"] = "decoded_source_unicode_codepoint"
    export.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="does not replay"):
        review(export, earlier_path, later_path)
    refused = f.client.post(_url("mixed-draft"), headers=f.viewer, files=_upload(_email("A message")))
    assert refused.status_code == 422


def test_cross_format_word_to_pdf_locates_one_change_across_two_pages(reference_http, monkeypatch, tmp_path):
    from scripts.rebuild_writing_word_boundary_sources import docx

    f = _install(reference_http, monkeypatch)
    earlier = docx("The first page keeps the yellow shipment.\n\nInstructions for packing:\n\nThe second shipment will use the blue packaging.")
    with pymupdf.open() as pdf:
        first = pdf.new_page()
        first.insert_text((60, 80), "The first page keeps the amber shipment.")
        second = pdf.new_page()
        second.insert_text((60, 80), "Rules for packing:")
        second.insert_text((60, 140), "The second shipment will use the green packaging.")
        later = pdf.tobytes()
    _saved(f, "multipage-mixed", ".pdf", later)
    response = f.client.post(_url("multipage-mixed"), headers=f.viewer, files=_upload(earlier, ".docx"),
                             data={"include_pdf": "true", "include_paired_check": "true"})
    assert response.status_code == 200, response.text
    payload = response.json()
    result = payload["comparison"]
    assert "all 2 upright, single-column pages" in result["scope"] and "paired_check" not in result
    assert [[page["page"] for page in change["later"]["pdf_pages"]] for change in result["changes"]] == [[1, 2], [2]]
    with pymupdf.open(stream=base64.b64decode(payload["pdf"]), filetype="pdf") as report:
        report_text = " ".join(" ".join(page.get_text().split()) for page in report)
        assert "Page 1 · highlighted source crop" in report_text and "Page 2 · highlighted source crop" in report_text
    earlier_path, later_path, export = (tmp_path / name for name in ("earlier.docx", "later.pdf", "draft.json"))
    earlier_path.write_bytes(earlier)
    later_path.write_bytes(later)
    export.write_text(json.dumps(result))
    verified = review(export, earlier_path, later_path)
    assert verified["pdf_previews_verified"] == 3 and verified["body_paragraph_ranges_verified"] == 3
    result["changes"][0]["later"]["pdf_pages"][1]["page"] = 1
    export.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="does not replay"):
        review(export, earlier_path, later_path)


@pytest.mark.parametrize("layout", ["rotated", "unmapped", "two_columns", "excess_pages"])
def test_cross_format_pdf_declines_unsupported_multipage_layouts(reference_http, monkeypatch, layout):
    f = _install(reference_http, monkeypatch)
    with pymupdf.open() as pdf:
        first = pdf.new_page()
        first.insert_text((60, 80), "The first page has selectable text.")
        second = pdf.new_page()
        if layout != "unmapped":
            second.insert_text((60, 80), "The second page has selectable text.")
        if layout == "two_columns":
            second.insert_text((360, 130), "An independent text column.")
        if layout == "rotated":
            second.set_rotation(90)
        if layout == "excess_pages":
            for _ in range(11):
                pdf.new_page().insert_text((60, 80), "Additional selectable text.")
        later = pdf.tobytes()
    _saved(f, "unsupported-mixed", ".pdf", later)
    response = f.client.post(_url("unsupported-mixed"), headers=f.viewer,
                             files=_upload(b"The earlier shipment was approved.", ".txt"))
    assert response.status_code == 422 and "comparison" not in response.json()


def test_cross_format_accepts_twelve_unchanged_pdf_pages():
    paragraphs = [f"Shipment {index} passed its packaging review." for index in range(1, 13)]
    with pymupdf.open() as pdf:
        for paragraph in paragraphs:
            pdf.new_page().insert_text((60, 80), paragraph)
        earlier = pdf.tobytes()
    later = "\n\n".join(paragraphs).encode()
    result, *_ = document_revision.compare_with_sources(earlier, later, ".txt", earlier_extension=".pdf")
    assert result["changes"] == [] and "all 12 upright, single-column pages" in result["scope"]


@pytest.mark.parametrize("text_options", [{"render_mode": 3}, {"fill_opacity": 0}])
def test_cross_format_declines_hidden_text_on_later_pdf_page(reference_http, monkeypatch, text_options):
    f = _install(reference_http, monkeypatch)
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((60, 80), "The first page has selectable text.")
        second = pdf.new_page()
        second.insert_text((60, 80), "The second page has selectable text.")
        second.insert_text((60, 140), "The shipment was cancelled.", **text_options)
        later = pdf.tobytes()
    _saved(f, "hidden-layer", ".pdf", later)
    response = f.client.post(_url("hidden-layer"), headers=f.viewer,
                             files=_upload(b"The shipment was approved.", ".txt"))
    assert response.status_code == 422 and "detected on page(s) 2" in response.json()["detail"]
    assert "cancelled" not in response.text


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
