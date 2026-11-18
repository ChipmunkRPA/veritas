from __future__ import annotations

import dataclasses
import io
from email.message import EmailMessage
from types import SimpleNamespace

import docx
import pymupdf
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image

from app.detectors.text.adapters import learned
from app.detectors.text.extract import extract, from_string, text_hash
from app.detectors.text.pipeline import extract_within
from app.detectors.text.segment import segment
from app.evidence import document_native_locations
from app.evidence.document_packets import _source_anchors


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


def word_hyperlink(paragraph, *labels):
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), paragraph.part.relate_to(
        "https://example.invalid/word-source", RELATIONSHIP_TYPE.HYPERLINK, is_external=True))
    for label in labels:
        hyperlink.append(paragraph.add_run(label)._r)
    paragraph._p.append(hyperlink)


def test_docx_saved_hyperlink_text_preserves_runs_and_unicode_source_ranges(tmp_path):
    path = tmp_path / "linked.docx"
    document = docx.Document()
    before, first, second, after = "A reference 😀 to  ", "the café", " 📘 source", " remains in this paragraph."
    paragraph = document.add_paragraph(before)
    word_hyperlink(paragraph, first, second)
    paragraph.add_run(after)
    document.save(path)

    extracted = extract_within(path, ".docx")
    assert extracted.text == "A reference 😀 to the café 📘 source remains in this paragraph."
    assert extracted.meta[document_native_locations.WORD_RUN_MAP_KEY] == document_native_locations.WORD_RUN_MAP
    block, = extracted.source_map
    assert block["run_mapping_complete"]
    expected_ends = [len(before), len(before + first), len(before + first + second), len(before + first + second + after)]
    assert block["runs"] == [
        {"run": i, "start": start, "end": end}
        for i, (start, end) in enumerate(zip([0, *expected_ends[:-1]], expected_ends, strict=True))
    ]

    packets = []
    for identifier, value in (("T01", "café 📘"), ("T02", "remains")):
        offset = extracted.text.index(value)
        packets.append({"id": identifier, "anchors": _source_anchors(
            extracted.source_map, offset, offset + len(value), len(extracted.text),
        )})
    locations = document_native_locations.build({"source": {"method": "python-docx"}, "packets": packets})["packets"]
    original = before + first + second + after
    for identifier, value in (("T01", "café 📘"), ("T02", "remains")):
        item, = locations[identifier]
        assert item == {"anchor_index": 0, "coordinate": "word_paragraph",
                        "ranges": [[original.index(value), original.index(value) + len(value)]]}


def test_docx_grid_omissions_keep_cells_in_their_native_rows(tmp_path):
    path = tmp_path / "omitted-cells.docx"
    document = docx.Document()
    table = document.add_table(rows=3, cols=3)
    table.cell(0, 0).text = "Top left."
    table.cell(0, 1).text = "Top center."
    table.cell(0, 2).text = "Top right."
    table.cell(1, 1).merge(table.cell(1, 2)).text = "Middle merged value."
    table.cell(2, 0).text = "Bottom left."
    table.cell(2, 1).text = "Bottom center."
    for row_number, cell_index, tag in ((1, 0, "gridBefore"), (2, -1, "gridAfter")):
        row = table.rows[row_number]._tr
        row.remove(row.tc_lst[cell_index])
        omitted = OxmlElement(f"w:{tag}")
        omitted.set(qn("w:val"), "1")
        row.get_or_add_trPr().append(omitted)
    document.save(path)

    extracted = extract(path, ".docx")
    assert [(block["body_path"], extracted.text[block["start"]:block["end"]]) for block in extracted.source_map] == [
        ([0, "row", 0, "cell", 0, 1], "Top left."), ([0, "row", 0, "cell", 1, 1], "Top center."),
        ([0, "row", 0, "cell", 2, 1], "Top right."), ([0, "row", 1, "cell", 1, 1], "Middle merged value."),
        ([0, "row", 2, "cell", 0, 1], "Bottom left."), ([0, "row", 2, "cell", 1, 1], "Bottom center."),
    ]


def word_control(nodes, *, placeholder=False):
    control, properties, content = (OxmlElement(tag) for tag in ("w:sdt", "w:sdtPr", "w:sdtContent"))
    if placeholder:
        properties.append(OxmlElement("w:showingPlcHdr"))
    control.extend((properties, content))
    nodes[0].addprevious(control)
    content.extend(nodes)
    return control


def test_docx_reads_stored_control_text_without_placeholders(tmp_path):
    path = tmp_path / "filled-form.docx"
    document = docx.Document()
    first = document.add_paragraph("Project office: ")
    inline_hidden = first.add_run("Mérida 😀")
    inline_hidden.font.hidden = True
    word_control([inline_hidden._r])
    first.add_run(" was confirmed.")
    word_control([first.add_run("Choose an office")._r], placeholder=True)
    block = document.add_paragraph("Stored field answer.")
    hidden = block.add_run(" Private field note.")
    hidden.font.hidden = True
    control = word_control([block._p])
    binding = OxmlElement("w:dataBinding")
    binding.set(qn("w:xpath"), "/ignored/source")
    control.find(qn("w:sdtPr")).append(binding)
    word_control([document.add_paragraph("Enter your answer")._p], placeholder=True)
    table = document.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    cell_p = cell.add_paragraph("Table field answer.")
    word_control([word_control([cell_p._p])])
    table_path = [3, "row", 0, "cell", 0, 2, "control", 0, "control", 0]
    document.save(path)

    result = extract_within(path, ".docx")
    assert result.text == "Project office: Mérida 😀 was confirmed.\n\nStored field answer. Private field note.\n\nTable field answer."
    assert [block["body_path"] for block in result.source_map] == [[0], [1, "control", 0], table_path]
    inline = result.source_map[0]
    assert not inline["run_mapping_complete"] and inline["runs"] == []
    assert inline["source_ranges"] == [[0, len("Project office: Mérida 😀 was confirmed."), 0, len("Project office: Mérida 😀 was confirmed.")]]
    assert result.meta["word_text_visibility"]["locations"] == [
        {"body_path": [0], "body_paragraph": 1, "run": 1, "run_basis": "extracted",
         "source_start": len("Project office: "), "source_end": len("Project office: Mérida 😀")},
        {"body_path": [1, "control", 0], "body_paragraph": None, "run": 1,
         "source_start": len("Stored field answer."), "source_end": len("Stored field answer. Private field note.")}
    ]
    assert extract(path, ".docx", include_word_content_controls=False).text == "Project office: was confirmed."


def test_docx_skips_accepted_revision_ranges_in_a_paragraph_with_controls(tmp_path):
    path = tmp_path / "filled-and-tracked.docx"
    document = docx.Document()
    paragraph = document.add_paragraph()
    word_control([paragraph.add_run("Text from a filled control. ")._r])
    inserted = paragraph.add_run("A tracked sentence.")._r
    revision = OxmlElement("w:ins")
    inserted.addprevious(revision)
    revision.append(inserted)
    document.save(path)

    result = extract_within(path, ".docx")
    assert result.text == "Text from a filled control."
    assert result.word_review["writing_input"] == "unsupported_tracked_changes"
    assert result.word_review["unsupported_marks"] == 1
    assert result.word_review["rows"] == []


def test_docx_does_not_relabel_cells_or_rows_around_wrapped_table_parts(tmp_path):
    path = tmp_path / "wrapped-table.docx"
    document = docx.Document()
    document.add_paragraph("Useful introduction.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "The first cell is wrapped."
    table.cell(0, 1).text = "The second cell must not be relabeled."
    word_control([table.cell(0, 0)._tc])
    second_table = document.add_table(rows=2, cols=1)
    second_table.cell(1, 0).text = "The second row must not be relabeled."
    word_control([second_table.rows[0]._tr])
    document.add_paragraph("Useful conclusion.")
    document.save(path)

    result = extract_within(path, ".docx")
    assert result.text == "Useful introduction.\n\nUseful conclusion."
    assert [block["body_path"] for block in result.source_map] == [[0], [3]]
    assert "tables with controls around entire rows or cells" in result.meta["coverage_warning"]


def test_rtf_codepage_unicode_paragraphs_and_extracted_locations(tmp_path):
    path = tmp_path / "article.rtf"
    path.write_bytes(rb"{\rtf1\ansi\ansicpg1252\uc1{\info{\title Metadata is outside the body}} "
                     b"Caf\xe9 " + rb"report \u-10179?\u-8704?.\par Next \b paragraph\b0.}")
    document = extract_within(path, ".rtf")
    assert document.text == "Café report 😀.\n\nNext paragraph."
    assert [document.text[start:end] for start, end in document.paragraphs] == ["Café report 😀.", "Next paragraph."]
    assert document.meta["method"] == "rtf"
    assert all(anchor["kind"] == "markup" and anchor["source_ranges"] for anchor in document.source_map)
    assert [anchor["paragraph_index"] for anchor in document.source_map] == [0, 1]


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
