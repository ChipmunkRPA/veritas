from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree
from test_document_bibliography import PROSE, REFERENCES
from test_strict_word import _strict_package

from app.billing.preflight import _document
from app.core.canonical import canonical_hash, canonical_json
from app.detectors.text.extract import extract, text_hash
from app.detectors.text.pipeline import extract_within
from app.detectors.text.strict_word import CONTENT_TYPES, RELATIONS, TRANSITIONAL_RELATIONSHIP
from app.evidence import document_bibliography, document_repeat, document_scope
from app.evidence import document_native_locations as locations
from app.evidence.document_packets import _source_anchors
from app.reports import public_verify, verify_document_source

FOOTNOTE = "The museum retained  résumé 🧪 and café in its log. The archive lists the sources used during the review."
ENDNOTE = "The committee will review the signed minutes and respond at the next meeting."


def _reference(paragraph, kind, number, *, hidden=False):
    run = paragraph.add_run()
    if hidden:
        run.font.hidden = True
    node = OxmlElement(f"w:{kind}Reference")
    node.set(qn("w:id"), str(number))
    run._r.append(node)


def _word(path, *, strict=False, bibliography=False, note_only=False, external=False):
    document = Document()
    body = document.add_paragraph("" if note_only else PROSE)
    _reference(body, "footnote", 2)
    _reference(body, "footnote", 2)
    _reference(body, "footnote", 3, hidden=True)
    if not note_only:
        _reference(document.add_paragraph("The editor supplied independent records."), "endnote", 2)
    if bibliography:
        for value in ("References", *REFERENCES):
            document.add_paragraph(value)
    stream = BytesIO()
    document.save(stream)
    with ZipFile(stream) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}

    declarations = etree.fromstring(parts["[Content_Types].xml"])
    relationships = etree.fromstring(parts["word/_rels/document.xml.rels"])
    for kind, value in (("footnote", FOOTNOTE), ("endnote", ENDNOTE)):
        name = f"saved/{kind}-part.xml"
        notes = OxmlElement(f"w:{kind}s")
        for number, contents in enumerate(("Omitted separator.", value, "Hidden word reference.", "Orphan note."), 1):
            note = OxmlElement(f"w:{kind}")
            note.set(qn("w:id"), str(number))
            if number == 1:
                note.set(qn("w:type"), "separator")
            para, run, text = (OxmlElement(f"w:{tag}") for tag in ("p", "r", "t"))
            text.text = contents
            run.append(text)
            para.append(run)
            note.append(para)
            notes.append(note)
        parts[f"word/{name}"] = etree.tostring(notes)
        etree.SubElement(declarations, CONTENT_TYPES + "Override", PartName=f"/word/{name}",
                         ContentType=f"application/vnd.openxmlformats-officedocument.wordprocessingml.{kind}s+xml")
        relation = {"Id": f"rIdNote-{kind}", "Type": TRANSITIONAL_RELATIONSHIP + kind + "s", "Target": name}
        if external and kind == "footnote":
            relation.update(Target="https://example.invalid/never-read.xml", TargetMode="External")
        etree.SubElement(relationships, RELATIONS + "Relationship", **relation)
    parts["[Content_Types].xml"] = etree.tostring(declarations)
    parts["word/_rels/document.xml.rels"] = etree.tostring(relationships)
    updated = BytesIO()
    with ZipFile(updated, "w") as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    data = updated.getvalue()
    path.write_bytes(_strict_package(data) if strict else data)
    return path


@pytest.mark.parametrize("strict", [False, True])
def test_linked_notes_preserve_reference_and_unicode_paragraph_positions(tmp_path, strict):
    path = _word(tmp_path / "notes.docx", strict=strict)
    decoded = extract_within(path, ".docx")
    assert all(text not in decoded.text for text in ("Omitted separator", "Hidden word reference", "Orphan note"))
    assert decoded.meta[locations.WORD_NOTES_KEY] == {"schema": locations.WORD_NOTES, "references": 2, "included": 2, "paragraphs": 2}
    footnote, endnote = [row for row in decoded.source_map if row.get("part_kind") in locations.WORD_NOTE_LABELS]
    for index, kind, source in ((0, "footnote", footnote), (1, "endnote", endnote)):
        assert (source["part_kind"], source["note_id"], source["note_paragraph_index"]) == (kind, 2, 0)
        assert source["reference_body_index"] == index and source["reference_run_index"] == 1
        assert source["reference_child_index"] == 0
    start = decoded.text.index("résumé")
    phrase = "résumé 🧪 and café"
    anchor, = _source_anchors(decoded.source_map, start, start + len(phrase), len(decoded.text))
    coverage = {locations.WORD_NOTES_KEY: decoded.meta[locations.WORD_NOTES_KEY]}
    evidence = {"source": {"method": "python-docx"}, "coverage": coverage, "packets": [{"id": "T01", "anchors": [anchor]}]}
    coverage[locations.KEY] = locations.build(evidence)
    locations.validate(evidence)
    row, = coverage[locations.KEY]["packets"]["T01"]
    a, b = row["ranges"][0]
    assert row["coordinate"] == "word_note_paragraph" and FOOTNOTE[a:b] == phrase


def test_note_only_checkout_and_external_relationship_exclusion(tmp_path):
    note_only = _word(tmp_path / "note-only.docx", note_only=True)
    _document(note_only, ".docx")
    assert extract(note_only, ".docx").text == FOOTNOTE.replace("retained  ", "retained ")
    external = _word(tmp_path / "external.docx", external=True)
    body = extract(external, ".docx")
    assert "résumé" not in body.text and ENDNOTE in body.text
    assert body.meta[locations.WORD_NOTES_KEY]["included"] == 1


def test_notes_do_not_obscure_existing_terminal_reference_exclusion(tmp_path):
    path = _word(tmp_path / "citations.docx", bibliography=True)
    doc = document_bibliography.body_view(extract(path, ".docx"))
    assert doc.meta[document_bibliography.KEY]["entry_count"] == 4
    assert "résumé" in doc.text and ENDNOTE in doc.text and "doi:10.1234" not in doc.text


@pytest.mark.parametrize("with_notes", [False, True])
def test_original_replay_obeys_the_saved_word_note_policy(tmp_path, with_notes):
    original = _word(tmp_path / "strict.docx", strict=True)
    doc = extract(original, ".docx", include_word_linked_notes=with_notes)
    assert ("résumé" in doc.text) is with_notes
    source = {"method": "python-docx", "sha256": doc.meta["original_sha256"], "chars": len(doc.text),
              "extracted_text_sha256": text_hash(doc.text), "source_map_sha256": canonical_hash(doc.source_map)}
    coverage = {locations.WORD_NOTES_KEY: doc.meta[locations.WORD_NOTES_KEY]} if with_notes else {}
    evidence = {"source": source, "version": document_scope.VERSION, "coverage": coverage, "packets": [],
                "repeated_passages": document_repeat.build(source, doc.text, doc.source_map)}
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": source["sha256"], "bytes": original.stat().st_size}}

    def replay():
        data = BytesIO()
        with ZipFile(data, "w") as archive:
            archive.writestr("evidence.json", canonical_json(evidence))
        with ZipFile(data) as archive:
            return verify_document_source.check(archive, bundle, original)

    assert "extracted text and source mapping matched" in replay()
    if with_notes:
        coverage.clear()
        with pytest.raises(ValueError, match="source mapping differs"):
            replay()
