from __future__ import annotations

from zipfile import ZIP_STORED, ZipFile

import pytest
from test_document_packets import packet_factory as packet_factory

from app.billing.preflight import _document
from app.core.canonical import canonical_json
from app.detectors.text.extract import extract
from app.evidence import document_native_locations as native
from app.evidence import document_revision
from app.evidence.document_packets import build_packet_set
from app.reports import public_verify, verify_document_source
from app.storage.files import ODT_MIME

TARGET = "le résumé prévu"
STALE = "le brouillon précédent"
PREFIX = "The café board approved "
SUFFIX = " for the next community consultation."
END = "Participants can submit questions before the meeting."


def make_odt(path, *, kind="reference", example="matching"):
    tag = "reference-mark" if kind == "reference" else "bookmark"
    start = f'<text:{tag}-start text:name="local"/>'
    end = f'<text:{tag}-end text:name="local"/>'
    target = f'<text:p>The review describes {start}{TARGET}{end} in plain language.</text:p>'
    value, mode = (STALE, "text") if example == "stale" else ("2", "page") if example == "page" else (TARGET, "text")
    field = f'<text:{kind}-ref text:ref-name="local" text:reference-format="{mode}">{value}</text:{kind}-ref>'
    paragraph = f'<text:p text:style-name="Standard">{PREFIX}{field}<text:s/>{SUFFIX[1:]}</text:p>'
    if example == "missing":
        target = f'<text:p>The review describes {TARGET} in plain language.</text:p>'
    elif example == "duplicate":
        target += target
    elif example == "hidden_target":
        target = f'<text:section text:name="Hidden" text:display="none">{target}</text:section>'
    elif example == "hidden_field":
        paragraph = f'<text:section text:name="Hidden" text:display="none">{paragraph}</text:section>'
    elif example == "point":
        target = target.replace(start, f'<text:{tag} text:name="local"/>').replace(end, "")
    elif example == "hidden_paragraph":
        target = target.replace("<text:p>", '<text:p><text:hidden-paragraph text:is-hidden="true"/>')
    elif example == "mixed_modes":
        page = f'<text:{kind}-ref text:ref-name="local" text:reference-format="page">2</text:{kind}-ref>'
        paragraph = paragraph.replace(field, page + " " + field)
    with ZipFile(path, "w") as archive:
        archive.writestr("mimetype", ODT_MIME, compress_type=ZIP_STORED)
        archive.writestr("content.xml", '<office:document-content '
                         'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
                         'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
                         f'<office:body><office:text>{target}{paragraph}<text:p>{END}</text:p>'
                         '</office:text></office:body></office:document-content>')
    return path


@pytest.mark.parametrize("kind", ["reference", "bookmark"])
def test_matching_saved_text_reference_restores_writing_and_exact_offsets(tmp_path, kind):
    source = make_odt(tmp_path / "notes.odt", kind=kind)
    _document(source, ".odt")
    doc = extract(source, ".odt")
    earlier = extract(source, ".odt", include_odt_text_references=False)
    assert doc.meta[native.ODT_TEXT_REFERENCES_KEY] == native.ODT_TEXT_REFERENCES
    assert native.ODT_TEXT_REFERENCES_KEY not in earlier.meta
    block = doc.source_map[1]
    assert block["paragraph_index"] == 1
    assert doc.text[block["start"]:block["end"]] == PREFIX + TARGET + SUFFIX
    start = block["start"] + len(PREFIX)
    assert native.clip_source_ranges(block["source_ranges"], start, start + len(TARGET)) == [
        [len(PREFIX), len(PREFIX + TARGET)]]
    assert earlier.text.count(TARGET) == 1 and doc.text.count(TARGET) == 2


@pytest.mark.parametrize("example", ["stale", "page", "duplicate", "hidden_target", "hidden_field", "missing", "point",
                                     "hidden_paragraph", "mixed_modes"])
def test_only_unique_matching_text_in_ordinary_paragraphs_is_retained(tmp_path, example):
    source = make_odt(tmp_path / "notes.odt", example=example)
    doc = extract(source, ".odt")
    earlier = extract(source, ".odt", include_odt_text_references=False)
    assert doc.text == earlier.text and doc.source_map == earlier.source_map
    assert STALE not in doc.text


@pytest.mark.parametrize("modern", [True, False])
def test_odt_public_evidence_replays_the_recorded_reference_policy(tmp_path, packet_factory, modern):
    source = make_odt(tmp_path / "notes.odt")
    db, store, analysis, result, _, _ = packet_factory(".odt", source_pdf=source, include_odt_text_references=modern)
    evidence = build_packet_set(db, analysis, store, result)
    key = native.ODT_TEXT_REFERENCES_KEY
    assert evidence["coverage"].get(key) == (native.ODT_TEXT_REFERENCES if modern else None)
    packet, = evidence["packets"]
    assert (TARGET in packet["focus"]["text"]) is modern
    positions = evidence["coverage"][native.KEY]["packets"][packet["id"]]
    assert all(row["coordinate"] == "odt_paragraph" for row in positions)
    if modern:
        assert native.ODT_TEXT_REFERENCES_SCOPE in " ".join(evidence["limitations"])
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
        del evidence["coverage"][key]
    else:
        evidence["coverage"][key] = native.ODT_TEXT_REFERENCES
    write()
    with ZipFile(proof) as archive, pytest.raises(ValueError, match="source mapping differs"):
        verify_document_source.check(archive, bundle, source)


def test_odt_draft_comparison_preserves_preexisting_reference_semantics(tmp_path):
    first = make_odt(tmp_path / "before.odt", example="matching")
    second = make_odt(tmp_path / "after.odt", example="stale")
    comparison, _, _, _ = document_revision.compare_with_sources(first.read_bytes(), second.read_bytes(), ".odt")
    assert comparison["changes"] == []
