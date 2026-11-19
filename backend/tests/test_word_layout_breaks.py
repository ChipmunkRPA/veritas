# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import pytest
from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from test_document_saved_link_hosts import _hyperlink
from test_document_word_saved_fields import word_bytes

from app.billing.preflight import validate_upload
from app.core.canonical import canonical_hash, canonical_json
from app.detectors.text.extract import extract, text_hash
from app.evidence import document_native_locations as locations
from app.evidence import (
    document_repeat,
    document_saved_link_hosts,
    document_scope,
    document_tracked_changes,
    document_word_visibility,
)
from app.evidence.document_packets import _source_anchors
from app.reports import public_verify, verify_document_source


def _source_range(doc, value):
    first = doc.text.index(value)
    anchor, = _source_anchors(doc.source_map, first, first + len(value), len(doc.text))
    return anchor


def _selected(anchor):
    return locations.clip_source_ranges(anchor["source_ranges"], anchor["selected_start"], anchor["selected_end"])


@pytest.mark.parametrize("strict", [False, True])
def test_page_and_column_breaks_preserve_body_and_table_token_offsets(strict, tmp_path):
    docx = Document()
    paragraph = docx.add_paragraph("🧪 The earlier sentence ends at the page boundary.")
    page = paragraph.add_run()
    page.add_break(WD_BREAK.PAGE)
    page.add_text("The following paragraph text starts after the page boundary.")
    cell = docx.add_table(rows=1, cols=1).cell(0, 0).paragraphs[0]
    column = cell.add_run("The first column ends here.")
    column.add_break(WD_BREAK.COLUMN)
    column.add_text("Column text resumes on the next column.")
    soft = docx.add_paragraph("This is an ordinary line break.").add_run()
    soft.add_break(WD_BREAK.LINE)
    soft.add_text("The word mid")
    soft._r.append(OxmlElement("w:lastRenderedPageBreak"))
    soft.add_text("point stays together across a rendering marker.")
    raw = word_bytes(docx, strict=strict)
    path = tmp_path / "source.docx"
    path.write_bytes(raw)
    validate_upload(BytesIO(raw), path.name, "document")

    current = extract(path, ".docx")
    historical = extract(path, ".docx", normalize_word_layout_breaks=False)
    assert current.meta[locations.WORD_LAYOUT_BREAK_KEY] == locations.WORD_LAYOUT_BREAK
    assert locations.WORD_LAYOUT_BREAK_KEY not in historical.meta
    assert ". The following" in current.text and ".The following" in historical.text
    assert ". Column text" in current.text and ".Column text" in historical.text
    assert "break. The word midpoint" in current.text and "break. The word midpoint" in historical.text
    for label, prefix, body_path in (("The following", "🧪 The earlier sentence ends at the page boundary.", [0]),
                                     ("Column text", "The first column ends here.", [1, "row", 0, "cell", 0, 1])):
        anchor = _source_range(current, label)
        old_anchor = _source_range(historical, label)
        start = len(prefix) + 1
        assert anchor["body_path"] == body_path
        assert _selected(anchor) == [[start, start + len(label)]]
        assert _selected(old_anchor) == [[start - 1, start - 1 + len(label)]]

    for reading in (current, historical):
        source = {"method": "python-docx", "sha256": reading.meta["original_sha256"], "chars": len(reading.text),
                  "extracted_text_sha256": text_hash(reading.text), "source_map_sha256": canonical_hash(reading.source_map),
                  document_word_visibility.COMMITMENT: canonical_hash(reading.meta[document_word_visibility.KEY]),
                  document_tracked_changes.COMMITMENT: canonical_hash(reading.word_review)}
        evidence = {"source": source, "coverage": {**reading.meta, document_tracked_changes.KEY: reading.word_review},
                    "packets": [], "version": document_scope.VERSION,
                    "repeated_passages": document_repeat.build(source, reading.text, reading.source_map)}
        bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
                  "source_commitment": {"sha256": source["sha256"], "bytes": len(raw)}}
        data = BytesIO()
        with ZipFile(data, "w") as archive:
            archive.writestr("evidence.json", canonical_json(evidence))
        with ZipFile(data) as archive:
            assert "Original SHA-256, extracted text and source mapping matched" in verify_document_source.check(archive, bundle, path)


def test_saved_link_and_writing_evidence_share_layout_break_coordinates(tmp_path):
    docx = Document()
    paragraph = docx.add_paragraph("😀 The following source is linked in the next column.")
    paragraph.add_run().add_break(WD_BREAK.COLUMN)
    _hyperlink(paragraph, "https://records.example.org/manual", "https://archive.example.net/manual")
    path = tmp_path / "linked.docx"
    path.write_bytes(word_bytes(docx))
    for current in (False, True):
        doc = extract(path, ".docx", normalize_word_layout_breaks=current)
        record = document_saved_link_hosts.inspect(path, ".docx", normalize_word_layout_breaks=current)
        [finding] = record["findings"]
        assert _selected(_source_range(doc, "records.example.org")) == finding["location"]["ranges"]


def test_page_break_inside_revision_stays_outside_accepted_projection(tmp_path):
    docx = Document()
    paragraph = docx.add_paragraph("This is ordinary text before an unsupported revision.")
    run = paragraph.add_run("Proposed inserted text.")
    run.add_break(WD_BREAK.PAGE)
    run.add_text("The next part remains in the same tracked insertion.")
    revision = OxmlElement("w:ins")
    revision.set(qn("w:id"), "7")
    run._r.addprevious(revision)
    revision.append(run._r)
    path = tmp_path / "unsupported.docx"
    path.write_bytes(word_bytes(docx))
    reading = extract(path, ".docx")
    assert reading.word_review["writing_input"] == document_tracked_changes.INPUT_UNSUPPORTED
    assert "Proposed inserted text" not in reading.text
