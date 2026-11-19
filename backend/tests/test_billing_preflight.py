"""Readability checks run with actual decoders, before a payment can be created."""
from __future__ import annotations

import io
import shutil
import stat
import struct
import subprocess
import time
import wave
import zipfile
import zlib
from email.message import EmailMessage
from html import escape

import pytest
from PIL import Image

from app.billing import preflight

WORDS = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
WRITING = "This fixture has enough ordinary writing for the local analysis to check its full body after payment."


def png() -> bytes:
    data = io.BytesIO()
    Image.new("RGB", (16, 16), (20, 40, 60)).save(data, "PNG")
    return data.getvalue()


def wav(*, samples=800) -> bytes:
    data = io.BytesIO()
    with wave.open(data, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\x01\x00" * samples)
    return data.getvalue()


def verify(data: bytes, filename: str, kind: str) -> None:
    source = io.BytesIO(data)
    source.seek(min(7, len(data)))
    position = source.tell()
    try:
        preflight.validate_upload(source, filename, kind)
    finally:
        assert source.tell() == position


def document_fixture(path, text):
    if path.suffix == ".docx":
        from docx import Document

        document = Document()
        document.add_paragraph(text)
        document.save(path)
    elif path.suffix == ".odt":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("mimetype", "application/vnd.oasis.opendocument.text")
            archive.writestr("content.xml", f'''
                <office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
                                         xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
                    <office:body><office:text><text:p>{escape(text)}</text:p></office:text></office:body>
                </office:document-content>''')
    elif path.suffix in {".pptx", ".epub"}:
        from test_reading_packages import epub, powerpoint

        make = {".pptx": powerpoint, ".epub": epub}[path.suffix]
        make(path, passages=(text, ""))
    else:
        templates = {".html": "<main><p>{}</p></main>", ".eml": "Content-Type: text/plain\r\n\r\n{}",
                     ".rtf": r"{{\rtf1\ansi {}}}", ".md": "# {}", ".txt": "{}", ".xhtml": (
                         '<?xml version="1.0" encoding="UTF-8"?>'
                         '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>{}</p></body></html>')}
        value = escape(text) if path.suffix in {".html", ".xhtml"} else text
        path.write_text(templates[path.suffix].format(value), encoding="utf-8")
    return path.read_bytes()


def test_valid_image_and_truncated_magic_only_image():
    image = png()
    verify(image, "private-file.png", "image")
    with pytest.raises(preflight.PreflightError, match="could not be fully decoded"):
        verify(image[:40], "private-file.png", "image")


def test_image_pixel_limit_is_checked_before_decoding():
    image = bytearray(png())
    image[16:24] = struct.pack(">II", 9000, 9000)
    image[29:33] = struct.pack(">I", zlib.crc32(image[12:29]))
    with pytest.raises(preflight.PreflightError, match="80 MP"):
        verify(bytes(image), "huge.png", "image")


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Local media binaries required")
def test_audio_needs_decodable_samples_and_positive_duration():
    verify(wav(), "voice.wav", "audio")
    with pytest.raises(preflight.PreflightError):
        verify(wav(samples=0), "empty-voice.wav", "audio")
    with pytest.raises(preflight.PreflightError):
        verify(b"RIFF\x24\x00\x00\x00WAVE", "broken.wav", "audio")


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Local media binaries required")
def test_video_needs_an_actual_decodable_frame(tmp_path):
    path = tmp_path / "valid.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=32x32:r=2",
                    "-t", "1", "-c:v", "mpeg4", "-threads", "1", str(path)], check=True, timeout=10,
                   stdin=subprocess.DEVNULL, capture_output=True)
    verify(path.read_bytes(), "clip.mp4", "video")
    with pytest.raises(preflight.PreflightError):
        verify(path.read_bytes()[:32], "broken.mp4", "video")


def test_readable_pdf_and_password_protected_or_broken_pdf():
    import pymupdf

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 50), "A sample document for file preparation.")
        good = document.tobytes()
        locked = document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="fixture-owner", user_pw="fixture-reader")
    verify(good, "article.pdf", "document")
    with pytest.raises(preflight.PreflightError, match="password-protected"):
        verify(locked, "locked.pdf", "document")
    with pytest.raises(preflight.PreflightError):
        verify(b"%PDF-1.7\ntruncated document", "broken.pdf", "document")


def test_scan_without_embedded_text_can_still_reach_pdf_checkout():
    import pymupdf

    with pymupdf.open() as document:
        document.new_page()
        page = document.new_page()
        page.insert_image(pymupdf.Rect(20, 20, 200, 200), stream=png())
        assert not page.get_text().strip()
        verify(document.tobytes(), "image-only-scan.pdf", "document")


def test_completely_blank_pdf_is_refused_before_checkout():
    import pymupdf

    with pymupdf.open() as document:
        document.new_page()
        document.new_page()
        data = document.tobytes()
    with pytest.raises(preflight.PreflightError, match="Every page in this PDF is blank.*No payment was started"):
        verify(data, "blank.pdf", "document")


@pytest.mark.parametrize("layout", ["later_text", "later_annotation", "form", "over_page_limit", "repaired"])
def test_pdf_checkout_remains_eligible_without_proof_every_page_is_blank(layout):
    import pymupdf

    with pymupdf.open() as document:
        document.new_page()
        page = document.new_page()
        if layout == "later_text":
            page.insert_text((50, 50), WRITING)
        elif layout == "later_annotation":
            page.add_text_annot((50, 50), "Saved marginal note.")
        elif layout == "form":
            document.xref_set_key(document.pdf_catalog(), "AcroForm", "<</Fields []>>")
        elif layout == "over_page_limit":
            for _ in range(preflight.MAX_BLANK_PDF_CHECK_PAGES - 1):
                document.new_page()
        data = document.tobytes()
    if layout == "repaired":
        before_index = data.rsplit(b"\nstartxref\n", 1)[0]
        data = before_index + b"\nstartxref\n0\n%%EOF\n"
        with pymupdf.open(stream=data) as repaired:
            assert repaired.is_repaired
    verify(data, "possible-content.pdf", "document")


@pytest.mark.parametrize("layout", ["alternative", "related", "named_attachment"])
def test_checkout_reads_html_body_when_plain_alternative_is_blank_or_named(tmp_path, layout):
    from app.detectors.text import email_body
    from app.detectors.text.extract import extract

    message = EmailMessage()
    if layout == "named_attachment":
        message.set_content(f"<p>{WRITING}</p>", subtype="html")
        message.make_mixed()
        attachment = EmailMessage()
        attachment.set_content("Named attachment words belong to a different document.")
        attachment.set_param("name", "notes.txt")
        message.attach(attachment)
        part = "1.1"
    else:
        message.set_content(" \t\n")
        message.add_alternative(f"<html><body><p>{WRITING}</p></body></html>", subtype="html")
        part = "1.2"
        if layout == "related":
            html = list(message.iter_parts())[1]
            html["Content-ID"] = "<body@example.org>"
            html.add_related(png(), maintype="image", subtype="png", cid="<image@example.org>")
            html.set_param("start", "<body@example.org>")
            html.set_payload(list(reversed(list(html.iter_parts()))))
            part = "1.2.2"
    data = message.as_bytes()
    path = tmp_path / "message.eml"
    path.write_bytes(data)
    verify(data, path.name, "document")
    doc = extract(path, ".eml")
    assert doc.text == WRITING
    assert doc.meta["email"]["body_part"] == part
    assert doc.meta["email"][email_body.SELECTION_KEY] == email_body.SELECTION_POLICY
    assert extract(path, ".eml", use_current_email_body=False).text != WRITING


@pytest.mark.parametrize("layout", ["unrelated", "attachment", "dangling", "duplicate", "short_plain"])
def test_checkout_does_not_promote_unrelated_html_or_override_nonblank_plain(layout):
    message = EmailMessage()
    message.set_content("Short plain body." if layout == "short_plain" else " \t\n")
    html = f"<p>{WRITING}</p>"
    if layout in {"unrelated", "attachment"}:
        message.make_mixed()
        part = EmailMessage()
        part.set_content(html, subtype="html")
        if layout == "attachment":
            part.set_param("name", "attached.html")
        message.attach(part)
    else:
        message.add_alternative(html, subtype="html")
        if layout in {"dangling", "duplicate"}:
            part = list(message.iter_parts())[1]
            part["Content-ID"] = "<body@example.org>"
            part.add_related(png(), maintype="image", subtype="png", cid="<body@example.org>")
            part.set_param("start", "<body@example.org>" if layout == "duplicate" else "<missing@example.org>")
    with pytest.raises(preflight.PreflightError):
        verify(message.as_bytes(), "message.eml", "document")


def test_docx_requires_a_readable_document_package():
    from docx import Document

    document = Document()
    document.add_paragraph(WRITING)
    good = io.BytesIO()
    document.save(good)
    verify(good.getvalue(), "article.docx", "document")
    unrelated = io.BytesIO()
    with zipfile.ZipFile(unrelated, "w") as archive:
        archive.writestr("unrelated.txt", "This zip is not a Word document.")
    with pytest.raises(preflight.PreflightError):
        verify(unrelated.getvalue(), "fake.docx", "document")
    with pytest.raises(preflight.PreflightError):
        verify(good.getvalue()[:100], "truncated.docx", "document")


def test_docx_requires_readable_body_text_but_accepts_a_table():
    from docx import Document

    document = Document()
    document.sections[0].header.paragraphs[0].text = "A readable header outside the writing check."
    document.add_paragraph("\u200b \t")
    content = io.BytesIO()
    document.save(content)
    with pytest.raises(preflight.PreflightError, match="no readable body text"):
        verify(content.getvalue(), "header-only.docx", "document")

    document.add_table(rows=1, cols=1).cell(0, 0).text = WRITING
    content = io.BytesIO()
    document.save(content)
    verify(content.getvalue(), "table-only.docx", "document")


@pytest.mark.parametrize("location", ["body", "table", "hyperlink"])
@pytest.mark.parametrize("strict", [False, True])
def test_docx_checkout_requires_some_writing_outside_proven_hidden_runs(location, strict):
    from docx import Document
    from docx.oxml import OxmlElement
    from test_strict_word import _strict_package

    document = Document()
    paragraph = (document.add_table(rows=1, cols=1).cell(0, 0).paragraphs[0]
                 if location == "table" else document.add_paragraph())
    first = paragraph.add_run("  🧪 First   hidden words.\t")
    second = paragraph.add_run("\u00a0" + WRITING)
    first.font.hidden = second.font.hidden = True
    if location == "hyperlink":
        hyperlink = OxmlElement("w:hyperlink")
        hyperlink.append(second._r)
        paragraph._p.append(hyperlink)

    def package():
        content = io.BytesIO()
        document.save(content)
        data = content.getvalue()
        return _strict_package(data) if strict else data

    with pytest.raises(preflight.PreflightError, match="only text marked hidden.*No payment was started"):
        verify(package(), "hidden.docx", "document")
    paragraph.add_run(" \t" + WRITING).font.hidden = False
    verify(package(), "visible-and-hidden.docx", "document")


def test_rtf_preflight_requires_the_signature_and_readable_body():
    verify(rb"{\rtf1\ansi A readable rich text document contains ordinary words and ends with two visible literal braces \{ and \}.}",
           "article.RTF", "document")
    for data, reason in (
        (b"Plain text saved with the wrong extension.", "not a readable Rich Text"),
        (rb"{\rtf1\ansi{\info{\title Readable title only}} }", "no readable text"),
        (rb"{\rtf1\ansi This document lacks a closing brace.", "incomplete"),
        (rb"{\rtf1\ansi Body text with an embedded {\*\object\bin3 xyz}.}", "binary content"),
    ):
        with pytest.raises(preflight.PreflightError, match=reason):
            verify(data, "article.rtf", "document")


@pytest.mark.parametrize("inline", [True, False])
def test_docx_checkout_reads_filled_controls_but_rejects_placeholder_only(inline):
    from docx import Document
    from test_document_source_mapping import word_control

    for placeholder in (False, True):
        document = Document()
        paragraph = document.add_paragraph()
        run = paragraph.add_run(WRITING)
        word_control([run._r if inline else paragraph._p], placeholder=placeholder)
        package = io.BytesIO()
        document.save(package)
        if placeholder:
            with pytest.raises(preflight.PreflightError, match="no readable body text"):
                verify(package.getvalue(), "blank.docx", "document")
        else:
            verify(package.getvalue(), "filled.docx", "document")


def test_text_checks_encoding_and_readability_without_running_models():
    verify(("A UTF-8 document — with punctuation. " + WRITING).encode(), "article.txt", "document")
    verify(("A UTF-16 document with punctuation. " + WRITING).encode("utf-16"), "article.txt", "document")
    verify(("Readable 👩\u200d🔬 text includes Acct\u200bNumber, a word\u00adbreak and \u2067עברית\u2069. " + WRITING).encode(),
           "article.txt", "document")
    for data in (b" \r\n\t", b"\x00\xff\xfe\x80", b"", "\u2067\u2069\u202e\u202c".encode()):
        with pytest.raises(preflight.PreflightError):
            verify(data, "article.txt", "document")


@pytest.mark.parametrize(("name", "export"), [
    ("letter.DOTX", "DOCX or PDF"),
    ("deck.potm", "PPTX, ODP or PDF"),
    ("workbook.xltx", "XLSX, ODS, CSV or PDF"),
    ("letter.ott", "DOCX or PDF"),
    ("deck.otp", "PPTX, ODP or PDF"),
    ("workbook.ots", "XLSX, ODS, CSV or PDF"),
])
def test_template_extensions_offer_a_readable_export_before_checkout(tmp_path, name, export):
    payload = document_fixture(tmp_path / "office-container.docx", WRITING)
    with pytest.raises(preflight.PreflightError, match=export):
        verify(payload, name, "document")


@pytest.mark.parametrize(("name", "readable", "empty"), [
    ("article.html", f"<main><p>{WRITING}</p></main>".encode(), b"<!-- comment --><style>section {color: red}</style>"),
    ("article.markdown", f"# {WRITING}".encode(), b"```python\ncode_without_prose()\n```"),
    ("article.eml", f"Content-Type: text/html\r\n\r\n<p>{WRITING}</p>".encode(),
     b"Content-Type: text/html\r\n\r\n<script>hidden content</script><p>&nbsp;</p>"),
])
def test_checkout_checks_extracted_writing_instead_of_markup(name, readable, empty):
    verify(readable, name, "document")
    with pytest.raises(preflight.PreflightError, match="no readable text"):
        verify(empty, name, "document")


@pytest.mark.parametrize("source", [
    f"~~~text\n{WRITING}\n~~~\n",
    f"   ````text\n{WRITING}\n```\n   `````\n",
    f"> ~~~text\n> {WRITING}\n> ~~~\n",
    f"- ~~~text\n  {WRITING}\n  ~~~\n",
    f"~~~text\n{WRITING}",
])
def test_checkout_counts_prose_outside_commonmark_fences(source):
    with pytest.raises(preflight.PreflightError, match="no readable text outside fenced code blocks"):
        verify(source.encode(), "code.md", "document")
    verify(f"{WRITING}\r\n\r\n{source}".encode("utf-16"), "mixed.markdown", "document")


def test_email_uses_mime_charset_and_requires_a_readable_body():
    email = (b"From: author@example.com\r\nTo: reader@example.com\r\nSubject: Example\r\n"
             b"Content-Type: text/plain; charset=iso-8859-1\r\nContent-Transfer-Encoding: 8bit\r\n\r\n"
             b"An ordinary caf\xe9 update with a clearly readable email body.\r\n" + WRITING.encode())
    verify(email, "message.eml", "document")
    for empty in (b"From: author@example.com\r\nSubject: Empty message\r\n\r\n",
                  b"Content-Type: multipart/mixed; boundary=fixture\r\n\r\n--fixture\r\n"
                  b"Content-Type: application/octet-stream\r\nContent-Disposition: attachment; filename=sample.bin\r\n"
                  b"Content-Transfer-Encoding: base64\r\n\r\nYWJj\r\n--fixture--\r\n"):
        with pytest.raises(preflight.PreflightError, match="no readable"):
            verify(empty, "empty.eml", "document")


@pytest.mark.parametrize("extension", ["docx", "odt", "pptx", "epub", "rtf", "txt", "md", "html", "xhtml", "eml"])
def test_checkout_word_floor_applies_to_the_same_extracted_text_across_formats(tmp_path, extension):
    path = tmp_path / f"article.{extension}"
    with pytest.raises(preflight.PreflightError, match="at least 15 readable words"):
        verify(document_fixture(path, WORDS.rsplit(" ", 1)[0]), path.name, "document")
    verify(document_fixture(path, WORDS), path.name, "document")


@pytest.mark.parametrize("extension", ["docx", "odt", "pptx", "epub", "txt", "html"])
def test_checkout_uses_character_limit_after_normalizing_the_body(tmp_path, monkeypatch, extension):
    limit = 120
    monkeypatch.setenv("VERITAS_MAX_DOCUMENT_CHARS", str(limit))
    path = tmp_path / f"article.{extension}"
    tail = "x" * (limit - len(WORDS) - 1)
    # The padding normalizes to one space and the HTML entity becomes one character.
    text = WORDS + " " * 20 + ("&" + tail[1:] if extension == "html" else tail)
    verify(document_fixture(path, text), path.name, "document")
    with pytest.raises(preflight.PreflightError, match=r"120(?: extracted characters|-character writing limit)"):
        verify(document_fixture(path, text + "x"), path.name, "document")


def test_checkout_word_floor_excludes_unselected_email_bodies_and_attachments():
    message = EmailMessage()
    message.set_content(WORDS.rsplit(" ", 1)[0])
    message.add_alternative(f"<p>{WRITING}</p>", subtype="html")
    message.add_attachment(WRITING, filename="attachment.txt")
    with pytest.raises(preflight.PreflightError, match="selected email body needs at least 15 readable words"):
        verify(message.as_bytes(), "selected-body.eml", "document")


def test_xhtml_requires_valid_text_encoding(tmp_path):
    source = document_fixture(tmp_path / "article.xhtml", WORDS)
    with pytest.raises(preflight.PreflightError, match="unreadable or damaged"):
        verify(source + b"\xff", "article.xhtml", "document")


def test_character_cap_precedes_bibliography_projection(monkeypatch):
    from test_document_bibliography import PROSE, REFERENCES

    monkeypatch.setenv("VERITAS_ENVIRONMENT", "dev")
    monkeypatch.setenv("VERITAS_TEXT_BIBLIOGRAPHY_SCOPE_REVIEW", "true")
    monkeypatch.setenv("VERITAS_MAX_DOCUMENT_CHARS", str(len(PROSE)))
    verify(PROSE.encode(), "body.txt", "document")
    payload = "\n\n".join([PROSE, "Works Cited", *REFERENCES]).encode()
    with pytest.raises(preflight.PreflightError, match="extracted characters"):
        verify(payload, "body-and-bibliography.txt", "document")


def test_checkout_applies_the_same_raw_plain_text_byte_limit_as_analysis(monkeypatch):
    from app.core.config import Settings
    from app.detectors.text.extract import RAW_BYTES_PER_CHAR

    default_limit = Settings.model_fields["max_document_chars"].default
    monkeypatch.setenv("VERITAS_MAX_DOCUMENT_CHARS", str(default_limit))
    byte_limit = default_limit * RAW_BYTES_PER_CHAR
    source = WORDS.encode().ljust(byte_limit)
    verify(source, "padded.txt", "document")
    with pytest.raises(preflight.PreflightError, match=rf"{byte_limit + 1:,} bytes.*No payment was started"):
        verify(source + b" ", "padded.txt", "document")


@pytest.mark.parametrize(("element", "text_element"), [("w:del", "w:delText"), ("w:moveFrom", "w:t")])
@pytest.mark.parametrize("hidden_body", [False, True])
def test_word_checkout_can_retain_revision_evidence_without_visible_body_text(element, text_element, hidden_body):
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    document = Document()
    paragraph = document.add_paragraph()
    change = OxmlElement(element)
    change.set(qn("w:id"), "1")
    run = OxmlElement("w:r")
    text = OxmlElement(text_element)
    text.text = "Earlier wording."
    run.append(text)
    change.append(run)
    paragraph._p.append(change)
    if hidden_body:
        document.add_paragraph().add_run(WRITING).font.hidden = True
    package = io.BytesIO()
    document.save(package)
    verify(package.getvalue(), "tracked.docx", "document")


def test_temporary_source_is_private_and_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight.tempfile, "tempdir", str(tmp_path))
    worker = preflight._worker

    def inspect(path, kind, extension):
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        worker(path, kind, extension)

    monkeypatch.setattr(preflight, "_worker", inspect)
    verify(png(), "private.png", "image")
    assert list(tmp_path.iterdir()) == []


def test_timeout_kills_child_and_removes_temporary_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "WALL_SECONDS", 0.00001)
    monkeypatch.setattr(preflight.tempfile, "tempdir", str(tmp_path))
    started = time.monotonic()
    with pytest.raises(preflight.PreflightError, match="time limit"):
        verify(png(), "source.png", "image")
    assert time.monotonic() - started < 3
    assert list(tmp_path.iterdir()) == []
