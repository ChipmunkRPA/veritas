from __future__ import annotations

import dataclasses
import io
from email.message import EmailMessage
from types import SimpleNamespace

import docx
import pymupdf
from PIL import Image

from app.detectors.text.adapters import learned
from app.detectors.text.extract import extract, from_string, text_hash
from app.detectors.text.pipeline import extract_within
from app.detectors.text.segment import segment


def test_unicode_offsets_and_whitespace_map_to_decoded_source(tmp_path):
    original = "😀 Hello  world.\r\n\r\n中文段落。\nThis stays together."
    path = tmp_path / "example.txt"
    path.write_bytes(original.encode("utf-8"))
    document = extract(path, ".txt")
    assert document.text == "😀 Hello world.\n\n中文段落。 This stays together."
    assert document.meta["offset_unit"] == "unicode_codepoint"
    assert document.meta["extracted_text_sha256"] == text_hash(document.text)
    for anchor in document.source_map:
        for start, end, source_start, source_end in anchor["source_ranges"]:
            actual = document.text[start:end]
            source = original[source_start:source_end]
            assert actual == source or actual == " " and source.isspace()
    # A UTF-16 JavaScript offset after the emoji would be one larger.
    assert document.text.index("Hello") == 2
    assert extract_within(path, ".txt").source_map == document.source_map


def test_pdf_repeated_text_rotation_and_dehyphenated_word_geometry(tmp_path):
    path = tmp_path / "repeat.pdf"
    pdf = pymupdf.open()
    first = pdf.new_page()
    first.insert_text((50, 60), "This is an ex-\nample of repeated text.")
    first.set_rotation(90)
    second = pdf.new_page()
    second.insert_text((120, 240), "This is an example of repeated text.")
    pdf.save(path)
    pdf.close()
    document = extract(path, ".pdf")
    assert document.text == "This is an example of repeated text.\n\nThis is an example of repeated text."
    assert [a["page"] for a in document.source_map] == [1, 2]
    assert document.source_map[0]["page_rotation"] == 90
    a = document.text.index("example")
    parts = [f for f in document.source_map[0]["fragments"] if f["start"] == a]
    assert len(parts) == 2
    assert parts[0]["bbox"][1] < parts[1]["bbox"][1]
    assert parts[0]["bbox"][0] > 90  # not the unrelated text at x=50 on line one
    assert document.source_map[1]["bbox"][1] > 200


def test_scanned_and_mixed_pdf_pages_are_explicitly_unchecked(tmp_path):
    path = tmp_path / "mixed.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((30, 40), "Readable words on one page.")
    page = pdf.new_page()
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(stream, format="PNG")
    page.insert_image(pymupdf.Rect(10, 10, 100, 100), stream=stream.getvalue())
    pdf.save(path)
    pdf.close()
    document = extract(path, ".pdf")
    assert document.meta["unmapped_pages"] == [2]
    assert document.meta["ocr"]["status"] == "partial"
    assert document.meta["ocr"]["unchecked_pages"] == [2]
    assert "not checked" in document.meta["coverage_warning"]
    assert {a["page"] for a in document.source_map} == {1}


def test_docx_body_order_merged_cells_and_structure_anchors(tmp_path):
    path = tmp_path / "ordered.docx"
    document = docx.Document()
    document.add_paragraph("Before the table 😀.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).merge(table.cell(0, 1)).text = "One merged cell."
    document.add_paragraph("After the table 中文.")
    document.save(path)
    extracted = extract(path, ".docx")
    assert extracted.text == "Before the table 😀.\n\nOne merged cell.\n\nAfter the table 中文."
    assert len(extracted.source_map) == 3
    assert "cell" in extracted.source_map[1]["body_path"]
    assert all(a["run_mapping_complete"] for a in extracted.source_map)
    assert "no stable page numbers" in extracted.meta["coverage_warning"]


def test_email_prefers_plain_body_keeps_quotes_and_does_not_authenticate_headers(tmp_path):
    path = tmp_path / "message.eml"
    mail = EmailMessage()
    mail["From"] = "sender@example.test"
    mail["Authentication-Results"] = "claimed.test; dkim=pass"
    mail.set_content("This is the latest reply.\n\n> Earlier quoted paragraph.\n\n-- \nA signature.")
    mail.add_alternative("<p>Alternate HTML should not be counted twice.</p>", subtype="html")
    mail.add_attachment(b"An attachment is not part of the body.", maintype="text", subtype="plain", filename="note.txt")
    path.write_bytes(mail.as_bytes())
    document = extract(path, ".eml")
    assert "latest reply" in document.text and "> Earlier quoted" in document.text
    assert "Alternate HTML" not in document.text and "not part of the body" not in document.text
    assert [a["quote_role"] for a in document.source_map] == ["body", "quoted", "signature"]
    assert document.meta["email"]["headers_authenticated"] is False
    assert len(document.meta["email"]["attachments"]) == 1
    assert len({a["mime_part"] for a in document.source_map}) == 1


def test_learned_windows_record_the_actual_model_input_not_only_focus(monkeypatch):
    doc = from_string("😀 First sentence has several words.\n\nSecond sentence adds some useful context. Third sentence finishes this test.")
    sentences = segment(doc)
    calls = []

    class Classifier:
        def predict(self, texts, **kwargs):
            calls.append(texts)
            self.last_coverage = [{"tokens": 8, "token_windows": 1, "covered_tokens": 8} for _ in texts]
            return [0.8 for _ in texts]

    classifier = Classifier()
    monkeypatch.setattr(learned.DesklibDetector, "backend", lambda self, ctx: classifier)
    monkeypatch.setattr(learned.settings, "text_window_sentences", 3)
    ctx = SimpleNamespace(outputs=SimpleNamespace(document={"text": doc.text, "paragraphs": doc.paragraphs, "meta": doc.meta,
                          "source_map": doc.source_map, "sentences": [dataclasses.asdict(s) for s in sentences]}),
                          lease=SimpleNamespace(get=lambda _: SimpleNamespace(revision="fixture")),
                          deadline=SimpleNamespace(check=lambda _: None))
    result = learned.DesklibDetector().run(ctx)
    windows = result.details["text_evidence"]["windows"]
    assert len(windows) == len(sentences) == 3
    assert len(windows[1]["spans"]) == 3
    for window, actual_input in zip(windows, calls[0], strict=True):
        reconstructed = " ".join(doc.text[s["start"]:s["end"]] for s in window["spans"])
        assert reconstructed == actual_input
        assert text_hash(actual_input) == window["input_sha256"]
    assert windows[1]["focus"] == {"start": sentences[1].start, "end": sentences[1].end}
    assert windows[1]["input_sha256"] != text_hash(sentences[1].text)
