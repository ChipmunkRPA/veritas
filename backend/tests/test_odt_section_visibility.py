from __future__ import annotations

import json
from zipfile import ZIP_STORED, ZipFile

import pytest
from test_document_packets import packet_factory as packet_factory

from app.billing.preflight import PreflightError, _document
from app.core.canonical import canonical_json
from app.detectors.text.extract import extract
from app.evidence import document_native_locations, document_revision
from app.evidence.document_packets import build_packet_set
from app.reports import public_verify, verify_document_source
from app.storage.files import ODT_MIME
from scripts.review_document_revision_export import review

HIDDEN = "This private paragraph contains enough words to pay for a checkout if the viewer's saved visibility is ignored."
CONDITIONAL = "This paragraph requires a display calculation, so leave it out of the extracted visible writing."
VISIBLE = (
    "The library opens at nine on Wednesday morning.",
    "The reading group meets in the north room next Tuesday.",
    "Send any questions to the coordinator before the monthly meeting.",
)
KEY = document_native_locations.ODT_VISIBILITY_KEY
POLICY = document_native_locations.ODT_VISIBILITY


def odt(path, *, visible=VISIBLE, hidden=HIDDEN):
    first, middle, last = visible
    content = (
        f'<text:p>{first}</text:p><text:section text:name="Visible" text:display="true">'
        f'<text:section text:name="Nested" text:display="none"><text:p>{hidden}</text:p></text:section>'
        f'<text:p>{middle}</text:p></text:section>'
        '<text:section text:name="Conditional" text:display="condition" text:condition="unresolved">'
        f'<text:p>{CONDITIONAL}</text:p></text:section><text:p>{last}</text:p>'
    )
    with ZipFile(path, "w") as archive:
        archive.writestr("mimetype", ODT_MIME, compress_type=ZIP_STORED)
        archive.writestr("content.xml", '<office:document-content '
                         'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
                         'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
                         f'<office:body><office:text>{content}</office:text></office:body></office:document-content>')
    return path


def test_odt_preflight_and_extraction_use_document_section_scope(tmp_path):
    source = odt(tmp_path / "notes.odt")
    _document(source, ".odt")
    document = extract(source, ".odt")
    assert document.text == "\n\n".join(VISIBLE)
    assert document.meta[KEY] == POLICY
    assert [row["paragraph_index"] for row in document.source_map] == [0, 1, 2]
    assert [document.text[row["start"]:row["end"]] for row in document.source_map] == list(VISIBLE)

    source = odt(tmp_path / "hidden.odt", visible=("Visible introduction.", "", ""))
    with pytest.raises(PreflightError, match="at least 15 readable words.*No payment was started"):
        _document(source, ".odt")
    source = odt(tmp_path / "hidden-only.odt", visible=("", "", ""))
    with pytest.raises(PreflightError, match="no readable body paragraphs"):
        _document(source, ".odt")


@pytest.mark.parametrize("modern", [True, False])
def test_odt_original_evidence_replays_recorded_visibility(tmp_path, packet_factory, modern):
    source = odt(tmp_path / "notes.odt")
    db, store, analysis, result, decoded, _ = packet_factory(".odt", source_pdf=source, skip_hidden_odt_sections=modern)
    evidence = build_packet_set(db, analysis, store, result)
    assert evidence["coverage"].get(KEY) == (POLICY if modern else None)
    assert (HIDDEN in decoded["text"]) is not modern
    assert (CONDITIONAL in decoded["text"]) is not modern
    if modern:
        assert HIDDEN not in json.dumps(evidence)
        assert document_native_locations.ODT_VISIBILITY_SCOPE in " ".join(evidence["limitations"])
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    proof = tmp_path / "proof.zip"

    def write():
        with ZipFile(proof, "w") as archive:
            archive.writestr("evidence.json", canonical_json(evidence))

    write()
    with ZipFile(proof) as archive:
        assert verify_document_source.check(archive, bundle, source)
    if modern:
        del evidence["coverage"][KEY]
    else:
        evidence["coverage"][KEY] = POLICY
    write()
    with ZipFile(proof) as archive, pytest.raises(ValueError, match="source mapping differs"):
        verify_document_source.check(archive, bundle, source)


@pytest.mark.parametrize("modern", [True, False])
def test_odt_draft_export_replays_recorded_visibility(tmp_path, modern):
    original = odt(tmp_path / "first.odt")
    revised = odt(tmp_path / "second.odt", hidden=HIDDEN.replace("private", "confidential"))
    comparison, _, _, _ = document_revision.compare_with_sources(
        original.read_bytes(), revised.read_bytes(), ".odt", skip_hidden_odt_sections=modern,
    )
    assert comparison.get(KEY) == (POLICY if modern else None)
    assert bool(comparison["changes"]) is not modern
    export = tmp_path / "comparison.json"
    export.write_text(json.dumps(comparison))
    assert review(export, original, revised)["source_hashes_verified"] == 2
    if modern:
        del comparison[KEY]
    else:
        comparison[KEY] = POLICY
    export.write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="does not replay"):
        review(export, original, revised)
