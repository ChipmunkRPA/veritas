from __future__ import annotations

from zipfile import ZipFile

import pytest
from test_reading_packages import PASSAGE, epub
from test_reading_packages import packet_factory as packet_factory

from app.billing.preflight import PreflightError, _document
from app.core.canonical import canonical_json
from app.detectors.text.extract import extract
from app.engine.stages import document as stage
from app.evidence import document_native_locations as locations
from app.evidence.document_packets import _source_anchors, build_packet_set
from app.reports import public_verify, verify_document_source


def _chapter(tmp_path, content, *, body_attributes=""):
    book = tmp_path / "base.epub"
    epub(book)
    changed = tmp_path / "visibility.epub"
    with ZipFile(book) as source, ZipFile(changed, "w") as target:
        for entry in source.infolist():
            data = source.read(entry)
            if entry.filename.endswith(".xhtml"):
                prefix = data.split(b"<body>", 1)[0]
                data = prefix + f"<body{body_attributes}>{content}</body></html>".encode()
            target.writestr(entry, data)
    return changed


@pytest.mark.parametrize(("attributes", "accepted"), [
    ("", True),
    (' aria-hidden="true"', True),
    (' style="display: none"', False),
    (' style="display: block; display: none"', True),
    (' hidden="hidden"', False),
])
def test_checkout_and_extraction_use_same_epub_chapter_visibility(tmp_path, attributes, accepted):
    path = _chapter(tmp_path, f"<p>{PASSAGE}</p>", body_attributes=attributes)
    doc = extract(path, ".epub")
    assert bool(doc.text) is accepted
    if accepted:
        _document(path, ".epub")
        assert doc.text == f"{PASSAGE}\n\n{PASSAGE}"
    else:
        with pytest.raises(PreflightError, match="at least 15 readable words.*No payment was started"):
            _document(path, ".epub")


@pytest.mark.parametrize("current", [False, True])
def test_epub_report_replays_native_locations_under_its_recorded_policy(tmp_path, packet_factory, current):
    path = _chapter(tmp_path, '<p aria-hidden="true">Visible 🦊 writing. ' + PASSAGE + '</p>'
                    '<p style="color: black; DiSpLaY: none !important">Saved hidden writing. ' + PASSAGE + '</p>'
                    '<p><span style="display:none">Omitted inline.</span>Final writing. ' + PASSAGE + '</p>')
    db, store, analysis, result, decoded, _ = packet_factory(".epub", source_pdf=path, use_epub_visibility=current)
    evidence = build_packet_set(db, analysis, store, result)
    key = locations.EPUB_VISIBILITY_KEY
    assert evidence["coverage"].get(key) == (locations.EPUB_VISIBILITY if current else None)
    assert ("Visible 🦊 writing." in decoded["text"]) is current
    assert ("Saved hidden writing." in decoded["text"]) is not current
    assert ("Omitted inline." in decoded["text"]) is not current
    start = decoded["text"].index("Final writing.")
    anchors = _source_anchors(decoded["source_map"], start, start + len("Final writing."), len(decoded["text"]))
    assert [(anchor["part_index"], anchor["paragraph_index"]) for anchor in anchors] == [(0, 1)]
    example = {"source": evidence["source"], "coverage": evidence["coverage"],
               "packets": [{"id": "T01", "anchors": anchors}]}
    expected_offset = 0 if current else len("Omitted inline.")
    assert locations.build(example)["packets"]["T01"] == [{"anchor_index": 0, "coordinate": "epub_block",
                                                           "ranges": [[expected_offset, expected_offset + 14]]}]

    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    proof = tmp_path / "proof.zip"
    with ZipFile(proof, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with ZipFile(proof) as archive:
        assert "extracted text and source mapping matched" in verify_document_source.check(archive, bundle, path)

    if current:
        del evidence["coverage"][key]
    else:
        evidence["coverage"][key] = locations.EPUB_VISIBILITY
    with ZipFile(proof, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with ZipFile(proof) as archive, pytest.raises(ValueError, match="source mapping differs"):
        verify_document_source.check(archive, bundle, path)


def test_epub_restart_reextracts_but_existing_report_keeps_its_reader(tmp_path, monkeypatch):
    source = _chapter(tmp_path, f"<p>{PASSAGE}</p>")
    monkeypatch.setattr(stage.settings, "text_bibliography_scope_review", False)
    monkeypatch.setattr(stage.settings, "text_doi_reference_review", False)
    checkpoint = {"meta": stage.decode_document(source, ".epub")[0].meta}
    assert stage.current_decode_output(checkpoint)
    del checkpoint["meta"][locations.EPUB_VISIBILITY_KEY]
    assert not stage.current_decode_output(checkpoint)
    assert stage.current_decode_output(checkpoint, require_current_layout_reader=False)
