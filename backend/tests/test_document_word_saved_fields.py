from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from test_document_source_mapping import word_control
from test_strict_word import _strict_package

from app.billing.preflight import PreflightError, validate_upload
from app.core.canonical import canonical_hash, canonical_json
from app.detectors.text.extract import WORD_CONTENT_CONTROLS_KEY, extract, text_hash
from app.evidence import document_native_locations as locations
from app.evidence import document_repeat, document_scope
from app.reports import public_verify, verify_document_source


def add_field(paragraph, instruction, value, *, hidden=False):
    run = paragraph.add_run(value)
    if hidden:
        run.font.hidden = True
    simple = OxmlElement("w:fldSimple")
    simple.set(qn("w:instr"), instruction)
    run._r.addprevious(simple)
    simple.append(run._r)
    return simple


def word_bytes(word, *, strict=False):
    buffer = BytesIO()
    word.save(buffer)
    return _strict_package(buffer.getvalue()) if strict else buffer.getvalue()


@pytest.mark.parametrize("strict", [False, True])
def test_saved_word_field_in_exact_extraction_and_original_replay(strict, tmp_path):
    word = Document()
    word.add_paragraph("First paragraph confirms this template has enough ordinary text for local writing analysis.")
    block = word.add_paragraph("😀  The saved date was ")
    add_field(block, ' DATE \\@ "SECRET_FIELD_INSTRUCTION" ', " 15  February 2020 ")
    block.add_run("and the next item is ")
    add_field(block, "SEQ Table", "2")
    block.add_run(".")
    raw = word_bytes(word, strict=strict)
    original = tmp_path / "simple.docx"
    original.write_bytes(raw)
    validate_upload(BytesIO(raw), original.name, "document")

    for modern in (True, False):
        doc = extract(original, ".docx", include_word_saved_fields=modern)
        coverage = {WORD_CONTENT_CONTROLS_KEY: doc.meta[WORD_CONTENT_CONTROLS_KEY]}
        if modern:
            record = doc.meta[locations.SAVED_FIELDS_KEY]
            coverage[locations.SAVED_FIELDS_KEY] = record
            assert "😀 The saved date was 15 February 2020 and the next item is 2." in doc.text
            assert record["found"] == record["included"] == 2
            first, second = record["locations"]
            assert (first["body_index"], first["field_index"], second["field_index"]) == (1, 1, 3)
            start = len("😀  The saved date was  ")
            assert (first["source_start"], first["source_end"]) == (start, start + len("15  February 2020"))
            assert doc.text[first["start"]:first["end"]] == "15 February 2020"
            assert doc.text[second["start"]:second["end"]] == "2"
            assert not doc.source_map[1]["run_mapping_complete"]
        else:
            assert "15 February" not in doc.text and locations.SAVED_FIELDS_KEY not in doc.meta
        assert "SECRET_FIELD_INSTRUCTION" not in doc.text
        source = {"method": "python-docx", "sha256": doc.meta["original_sha256"], "chars": len(doc.text),
                  "extracted_text_sha256": text_hash(doc.text), "source_map_sha256": canonical_hash(doc.source_map)}
        evidence = {"source": source, "coverage": coverage, "packets": [], "version": document_scope.VERSION,
                    "repeated_passages": document_repeat.build(source, doc.text, doc.source_map)}
        bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
                  "source_commitment": {"sha256": source["sha256"], "bytes": len(raw)}}
        archive_io = BytesIO()
        with ZipFile(archive_io, "w") as archive:
            archive.writestr("evidence.json", canonical_json(evidence))
        with ZipFile(archive_io) as archive:
            detail = verify_document_source.check(archive, bundle, original)
        assert ("Verified the saved Word field display values" in detail) == modern
        if modern:
            first["source_start"] += 1
            first["source_end"] += 1
            tampered = BytesIO()
            with ZipFile(tampered, "w") as archive:
                archive.writestr("evidence.json", canonical_json(evidence))
            with ZipFile(tampered) as archive, pytest.raises(ValueError, match="Saved Word fields differ"):
                verify_document_source.check(archive, bundle, original)


def test_hidden_nested_external_and_complex_fields_do_not_enter_writing_text(tmp_path):
    word = Document()
    block = word.add_paragraph("Ordinary body.")
    add_field(block, 'INCLUDETEXT "https://example.invalid/private.txt"', "External content")
    add_field(block, "REF private", "Hidden content", hidden=True)
    outer = add_field(block, "DATE", "Outer")
    inner = add_field(block, "DATE", "Nested field")
    outer.append(inner)
    block = word.add_paragraph("Control paragraph.")
    word_control([add_field(block, "DATE", "Controlled content")])
    block = word.add_paragraph("Complex field paragraph.")
    block.add_run()._r.append(OxmlElement("w:fldChar"))
    add_field(block, "DATE", "Complex context")
    cell = word.add_table(rows=1, cols=1).cell(0, 0)
    add_field(cell.paragraphs[0], "DATE", "Table context")
    path = tmp_path / "skipped.docx"
    path.write_bytes(word_bytes(word))
    doc = extract(path, ".docx")
    assert doc.text == "Ordinary body.\n\nControl paragraph.\n\nComplex field paragraph."
    fields = doc.meta[locations.SAVED_FIELDS_KEY]
    assert fields["found"] == 7 and fields["included"] == 0 and fields["locations"] == []


def test_saved_field_projection_preserves_tracked_document_offsets(tmp_path):
    word = Document()
    field = word.add_paragraph("The saved date is ")
    add_field(field, "DATE", "15 February 2020")
    block = word.add_paragraph("An existing sentence. ")
    inserted = block.add_run("Added wording remains in the accepted review.")._r
    revision = OxmlElement("w:ins")
    inserted.addprevious(revision)
    revision.append(inserted)
    path = tmp_path / "tracked.docx"
    path.write_bytes(word_bytes(word))
    baseline = extract(path, ".docx", include_word_saved_fields=False)
    modern = extract(path, ".docx")
    assert modern.text == baseline.text and modern.source_map == baseline.source_map
    assert modern.word_review == baseline.word_review and modern.word_review["status"] == "found"
    assert modern.meta[locations.SAVED_FIELDS_KEY]["included"] == 0


def test_open_complex_field_excludes_later_paragraphs_until_its_closing_marker(tmp_path):
    word = Document()
    for index, phase in enumerate(("begin", None, "end", None)):
        block = word.add_paragraph(f"Body paragraph {index + 1}. ")
        if phase:
            value = OxmlElement("w:fldChar")
            value.set(qn("w:fldCharType"), phase)
            block.add_run()._r.append(value)
        else:
            add_field(block, "DATE", ("Inside complex field" if index == 1 else "Outside complex field"))
    path = tmp_path / "cross-paragraph.docx"
    path.write_bytes(word_bytes(word))
    doc = extract(path, ".docx")
    record = doc.meta[locations.SAVED_FIELDS_KEY]
    assert record["found"] == 2 and record["included"] == 1
    assert record["locations"][0]["text"] == "Outside complex field"
    assert "Inside complex field" not in doc.text


def test_tiny_saved_fields_are_readable_with_no_checkout(tmp_path):
    word = Document()
    add_field(word.add_paragraph(), "DATE", "2/15/2008")
    source = word_bytes(word)
    path = tmp_path / "too-short.docx"
    path.write_bytes(source)
    assert extract(path, ".docx").text == "2/15/2008"
    with pytest.raises(PreflightError, match="fewer than 15 readable words. No payment was started"):
        validate_upload(BytesIO(source), path.name, "document")
