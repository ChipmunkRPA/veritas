from __future__ import annotations

import json
from zipfile import ZipFile

import pytest
from test_document_packets import packet_factory as packet_factory

from app.billing.preflight import PreflightError, _document
from app.core.canonical import canonical_json
from app.detectors.text.extract import extract, rtf_text
from app.evidence import document_native_locations as locations
from app.evidence.document_packets import build_packet_set
from app.evidence.document_revision import compare_with_sources
from app.reports import public_verify, verify_document_source
from scripts.review_document_revision_export import review

KEY = locations.RTF_VISIBILITY_KEY
POLICY = locations.RTF_VISIBILITY
PRIVATE = "These hidden words could supply enough writing for analysis without any actual readable words in the body."
VISIBLE = (
    "The library opens at nine on Wednesday morning.",
    "Café readings are usually held in the north room on Tuesday.",
    "Send any questions to the coordinator before the monthly meeting.",
)


def rtf(path, *, hidden=PRIVATE):
    path.write_bytes((
        r"{\rtf1\ansi\ansicpg1252 " + VISIBLE[0] + r"\par\v " + hidden
        + r"\par\v0 Caf\'e9 readings are usually held in the north room on Tuesday."
        + r"\par {\v This private run is embedded but }" + VISIBLE[2] + "}"
    ).encode("ascii"))
    return path


@pytest.mark.parametrize(("raw", "expected"), [
    (rb"{\rtf1\ansi Start {\v Hide \u233? \emdash {\b more} {\v0 this} hide.} End.}", "Start this End."),
    (rb"{\rtf1\ansi\v1 Hidden \plain Visible {\v Hidden {\plain included} hidden} end.}", "Visible included end."),
    (rb"{\rtf1\ansi This shows \\v literally and braces \{\v hidden\v0\}.}", r"This shows \v literally and braces {}."),
    (rb"{\rtf1\ansi {\v\uc2\u233xy \'e9 \{hidden\}\emdash}\u233? visible.}", "é visible."),
    (rb'{\rtf1\ansi {\v Hidden\par}{\field{\*\fldinst{ HYPERLINK "https://example.invalid"}}'
     rb"{\fldrslt Displayed link}} after.}", "\nDisplayed link after."),
    (rb'{\rtf1\ansi First {\v{\field{\*\fldinst{HYPERLINK "https://example.invalid"}}'
     rb"{\fldrslt hidden label}}} after.}", "First  after."),
])
def test_direct_visibility_preserves_decoding_and_group_scope(raw, expected):
    assert rtf_text(raw) == expected


def test_saved_field_label_is_independent_of_hidden_text(tmp_path):
    earlier = (rb'{\rtf1\ansi Show \\v and {\field{\*\fldinst{ HYPERLINK "https://example.invalid"}}'
               rb'{\fldrslt saved label}} over these next few ordinary words.}')
    later = earlier[:-1] + rb'{\v Unprinted note.}}'
    assert rtf_text(earlier) == rtf_text(later) == r"Show \v and saved label over these next few ordinary words."
    assert 'saved label("https://example.invalid")' in rtf_text(earlier, skip_hidden_rtf_runs=False)
    old = tmp_path / "earlier.rtf"
    new = tmp_path / "later.rtf"
    old.write_bytes(earlier)
    new.write_bytes(later)
    comparison, *_ = compare_with_sources(old.read_bytes(), new.read_bytes(), ".rtf")
    assert comparison["changes"] == []


def test_preflight_counts_only_visible_body_words(tmp_path):
    source = rtf(tmp_path / "visible.rtf")
    _document(source, ".rtf")
    doc = extract(source, ".rtf")
    assert doc.text == "\n\n".join(VISIBLE)
    assert doc.meta[KEY] == POLICY
    assert [row["paragraph_index"] for row in doc.source_map] == [0, 2, 3]
    assert [doc.text[row["start"]:row["end"]] for row in doc.source_map] == list(VISIBLE)

    for prefix, error in [("", "no readable text"), ("Visible only. ", "at least 15 readable words")]:
        path = tmp_path / "too-little.rtf"
        path.write_bytes((r"{\rtf1\ansi " + prefix + r"\v " + PRIVATE + "}").encode("ascii"))
        with pytest.raises(PreflightError, match=error):
            _document(path, ".rtf")


@pytest.mark.parametrize("modern", [True, False])
def test_original_evidence_replays_recorded_rtf_visibility(tmp_path, packet_factory, modern):
    source = rtf(tmp_path / "source.rtf")
    db, store, analysis, result, decoded, _ = packet_factory(".rtf", source_pdf=source, skip_hidden_rtf_runs=modern)
    evidence = build_packet_set(db, analysis, store, result)
    assert evidence["coverage"].get(KEY) == (POLICY if modern else None)
    assert (PRIVATE in decoded["text"]) is not modern
    if modern:
        assert PRIVATE not in json.dumps(evidence)
        assert any(locations.RTF_VISIBILITY_SCOPE in line for line in evidence["limitations"])
        anchors = [anchor for packet in evidence["packets"] for anchor in packet["anchors"]]
        assert any(anchor["paragraph_index"] == 2 for anchor in anchors)
        assert all(anchor["paragraph_index"] != 1 for anchor in anchors)
        source_blocks = (VISIBLE[0], "", *VISIBLE[1:])
        mapped = evidence["coverage"][locations.KEY]["packets"]
        for packet in evidence["packets"]:
            for item in mapped[packet["id"]]:
                anchor = packet["anchors"][item["anchor_index"]]
                block = source_blocks[anchor["paragraph_index"]]
                assert item["coordinate"] == "rtf_paragraph"
                assert "".join(block[start:end] for start, end in item["ranges"]) == decoded["text"][
                    anchor["selected_start"]:anchor["selected_end"]]
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
def test_draft_export_replays_the_saved_visibility_rule(tmp_path, modern):
    original = rtf(tmp_path / "earlier.rtf")
    revised = rtf(tmp_path / "later.rtf", hidden=PRIVATE.replace("hidden", "concealed"))
    compared, *_ = compare_with_sources(original.read_bytes(), revised.read_bytes(), ".rtf", skip_hidden_rtf_runs=modern)
    assert compared.get(KEY) == (POLICY if modern else None)
    assert bool(compared["changes"]) is not modern
    output = tmp_path / "comparison.json"
    output.write_text(json.dumps(compared))
    assert review(output, original, revised)["source_hashes_verified"] == 2
    if modern:
        del compared[KEY]
    else:
        compared[KEY] = POLICY
    output.write_text(json.dumps(compared))
    with pytest.raises(ValueError, match="does not replay"):
        review(output, original, revised)
