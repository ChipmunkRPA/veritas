from __future__ import annotations

import zipfile
from copy import deepcopy
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from html import unescape

import docx
import pytest
from test_document_packets import packet_factory as packet_factory

from app.core.canonical import canonical_hash, canonical_json
from app.evidence import document_native_locations as locations
from app.evidence import document_tracked_changes as tracked
from app.evidence.document_packets import (
    DocumentPacketIntegrityError,
    build_packet_set,
    packet_hash,
    validate_packet_set,
)
from app.reports import engine, multimedia_evidence, public_package, public_verify, verify_document_source
from app.reports.specimens import specimen

FIRST = "😀 First sentence has enough wording to inspect."
MIDDLE = "The  middle sentence includes a source position."
LAST = "The final sentence explains the report's purpose."


def _prepare(ext, path):
    if ext == ".txt":
        path.write_bytes(("  " + FIRST + "\r\n\r\n" + MIDDLE + " " + LAST + "\r\n").encode("utf-16"))
    elif ext == ".docx":
        document = docx.Document()
        document.add_paragraph(FIRST)
        table = document.add_table(rows=1, cols=1)
        paragraph = table.cell(0, 0).paragraphs[0]
        paragraph.add_run("  The ")
        paragraph.add_run(" middle sentence includes a source position.")
        document.add_paragraph(LAST)
        document.save(path)
    else:
        message = EmailMessage(policy=policy.SMTP)
        message.set_content(
            f"<p>{FIRST}</p><p>The <strong>middle</strong> sentence includes a source &amp; reference.</p>"
            f"<blockquote><p>{LAST}</p></blockquote>",
            subtype="html",
            cte="quoted-printable",
        )
        message.add_attachment("Do not include this text in analysis.", filename="excluded.txt")
        path.write_bytes(message.as_bytes())


def _snapshot(evidence, analysis, result):
    snapshot = specimen("document", "standard")
    frozen = {**deepcopy(snapshot.result), "signals": result["signals"]}
    frozen["input"].update(
        sha256=analysis.sha256,
        filename=analysis.filename,
        mime_sniffed=analysis.content_type,
        size_bytes=analysis.size_bytes,
    )
    frozen["manifest"]["sampling"] = deepcopy(result["manifest"]["sampling"])
    evidence.update(
        analysis_id=snapshot.analysis.id,
        revision=snapshot.analysis.revision,
        original_result_hash=canonical_hash(frozen),
    )
    evidence["sha256"] = packet_hash(evidence)
    return snapshot.model_copy(
        update={
            "result": frozen,
            "evidence": evidence,
            "media": snapshot.media.model_copy(
                update={
                    "sha256": analysis.sha256,
                    "filename": analysis.filename,
                    "content_type": analysis.content_type,
                    "size_bytes": analysis.size_bytes,
                }
            ),
        }
    )


@pytest.mark.parametrize(
    ("ext", "coordinate", "label"),
    [
        (".txt", "decoded_text", "Original decoded text characters"),
        (".docx", "word_paragraph", "Original Word paragraph characters"),
        (".eml", "decoded_email_html", "Decoded email HTML source characters"),
    ],
)
def test_original_text_locations_follow_native_source_in_exports_and_replay(
    packet_factory, tmp_path, monkeypatch, ext, coordinate, label
):
    original = tmp_path / ("source" + ext)
    _prepare(ext, original)
    db, store, analysis, result, decoded, _ = packet_factory(ext, source_pdf=original)
    evidence = build_packet_set(db, analysis, store, result)
    (packet,) = evidence["packets"]
    native = evidence["coverage"][locations.KEY]["packets"][packet["id"]]
    assert len(native) == 3
    assert all(row["coordinate"] == coordinate for row in native)
    for row in native:
        anchor = packet["anchors"][row["anchor_index"]]
        excerpt = decoded["text"][anchor["selected_start"] : anchor["selected_end"]]
        if ext == ".txt":
            text = original.read_bytes().decode("utf-16")
        elif ext == ".docx":
            text = [FIRST, "  " + MIDDLE, LAST][anchor["paragraph_index"]]
            if anchor["paragraph_index"] == 1:
                assert "cell" in anchor["body_path"]
        else:
            message = BytesParser(policy=policy.default).parsebytes(original.read_bytes())
            text = message.get_body(preferencelist=("html",)).get_content()
        parts = "".join(text[start:end] for start, end in row["ranges"])
        assert " ".join(unescape(parts).split()) == excerpt

    if ext == ".eml":
        assert len(native[1]["ranges"]) == 3 and native[1]["ranges"][0][1] < native[1]["ranges"][1][0]
        assert {anchor["quote_role"] for anchor in packet["anchors"]} == {"body", "quoted"}

    snapshot = _snapshot(evidence, analysis, result)
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    export = public_package._evidence(snapshot)
    assert export["coverage"][locations.KEY] == evidence["coverage"][locations.KEY]
    assert "txt.desklib" not in str(export) and "PRIVATE MODEL NAME" not in str(export)
    rendered = engine.render_pdf(snapshot)
    pdf = " ".join(engine.pdf_text(rendered.pdf).split())
    assert not rendered.refused and label in pdf
    assert f"). {label}" in pdf
    assert all(f"[{a}, {b})" in pdf for row in native for a, b in row["ranges"])

    bundle = {
        "schema": public_verify.SCHEMA,
        "analysis": {"kind": "document"},
        "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes},
    }
    path = tmp_path / "evidence.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("evidence.json", canonical_json(export))
    with zipfile.ZipFile(path) as archive:
        assert "Verified decoded original text locations" in verify_document_source.check(
            archive, bundle, original
        )


def test_location_malformed_gaps_abstain_and_drift_is_rejected(packet_factory):
    assert locations._clip([[0, 5, 2, 7], [6, 9, 8, 11]], 3, 9) == []
    assert locations._clip([[0, 2, 0, 5]], 1, 2) == []
    assert locations._clip([[0, 1, 0, 5], [1, 2, 8, 9]], 0, 2) == [[0, 5], [8, 9]]
    db, store, analysis, result, _, _ = packet_factory()
    evidence = build_packet_set(db, analysis, store, result)
    forged = deepcopy(evidence)
    first = forged["coverage"][locations.KEY]["packets"][forged["packets"][0]["id"]][0]
    first["ranges"][0][0] += 1
    forged["sha256"] = packet_hash(forged)
    with pytest.raises(DocumentPacketIntegrityError, match="Original text locations"):
        validate_packet_set(
            forged,
            analysis_id=analysis.id,
            revision=analysis.revision,
            original_result_hash=analysis.result_hash,
            source_sha256=analysis.sha256,
        )


def test_pdf_positions_do_not_claim_decoded_character_locations(packet_factory):
    db, store, analysis, result, _, _ = packet_factory(".pdf")
    evidence = build_packet_set(db, analysis, store, result)
    assert locations.KEY not in evidence["coverage"]
    assert all(anchor["page"] == 1 for packet in evidence["packets"] for anchor in packet["anchors"])


def test_word_track_changes_match_original_xml_and_stay_out_of_redacted_pdf(packet_factory, tmp_path, monkeypatch):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from lxml import etree

    original = tmp_path / "tracked.docx"
    document = docx.Document()
    document.add_paragraph(FIRST)
    middle = document.add_paragraph(MIDDLE)
    document.add_paragraph(LAST)
    middle._p.insert(0, etree.Comment("This does not affect element coordinates"))
    for kind, leaf, text in (("del", "delText", "Former wording 😀"), ("ins", "t", "Replacement wording 😀")):
        change, run, content = (OxmlElement(f"w:{name}") for name in (kind, "r", leaf))
        change.set(qn("w:author"), "PRIVATE REVISION AUTHOR")
        content.text = text
        run.append(content)
        if kind == "ins":
            run.append(OxmlElement("w:br"))
            second = OxmlElement("w:t")
            second.text = "Second  line of the insertion"
            run.append(second)
        change.append(run)
        middle._p.append(change)
    properties = middle._p.get_or_add_pPr()
    mark_properties = OxmlElement("w:rPr")
    mark_properties.append(OxmlElement("w:del"))
    properties.append(mark_properties)
    document.save(original)
    moved = tmp_path / "moved.docx"
    with zipfile.ZipFile(original) as package, zipfile.ZipFile(moved, "w", zipfile.ZIP_DEFLATED) as output:
        for item in package.namelist():
            data = package.read(item)
            if item in ("[Content_Types].xml", "_rels/.rels"):
                data = data.replace(b"word/document.xml", b"word/main.xml")
            output.writestr(item.replace("word/document.xml", "word/main.xml").replace("word/_rels/document.xml.rels", "word/_rels/main.xml.rels"), data)
    original = moved
    db, store, analysis, result, decoded, _ = packet_factory(".docx", source_pdf=original)
    source = docx.Document(original)
    review = tracked.inspect(source.element.body, str(source.part.partname).lstrip("/"))
    assert review["part"] == "word/main.xml"
    assert review["status"] == "partial" and review["unsupported_marks"] == 1
    assert [row["text"] for row in review["rows"]] == ["Former wording 😀", "Replacement wording 😀\nSecond  line of the insertion"]
    assert [(row["paragraph"], row["body_index"], row["inline_index"]) for row in review["rows"]] == [(2, 1, 2), (2, 1, 3)]
    assert all(row["text"] not in decoded["text"] for row in review["rows"])

    from app.storage.artefacts import register_artefact
    from app.storage.db import AnalysisStage

    decoded[tracked.KEY] = review
    stage = register_artefact(db, store, canonical_json(decoded), media_type="application/json", role="stage_output",
                              created_by="decode", analysis_id=analysis.id)
    db.get(AnalysisStage, (analysis.id, "decode")).output_artefact_id = stage.id
    result["manifest"]["sampling"]["document_source"]["decode_sha256"] = stage.sha256
    result["manifest"]["sampling"][tracked.COMMITMENT] = canonical_hash(review)
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    evidence = build_packet_set(db, analysis, store, result)
    assert evidence["coverage"][tracked.KEY] == review
    assert "PRIVATE REVISION AUTHOR" not in str(evidence)
    snapshot = _snapshot(evidence, analysis, result)
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    export = public_package._evidence(snapshot)
    assert export["coverage"][tracked.KEY] == review
    assert public_package.public_result(snapshot.result)[tracked.COMMITMENT] == canonical_hash(review)
    pdf = engine.render_pdf(snapshot)
    assert not pdf.refused
    printed = engine.pdf_text(pdf.pdf)
    assert "Replacement wording" in printed and "Second  line of the insertion" in printed
    redacted = snapshot.model_copy(update={"report": snapshot.report.model_copy(update={"redacted": True})})
    assert multimedia_evidence.view(redacted)["word_tracked_changes"] is None
    assert "Replacement wording" not in engine.pdf_text(engine.render_pdf(redacted).pdf)
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    archive_path = tmp_path / "tracked.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("evidence.json", canonical_json(export))
    with zipfile.ZipFile(archive_path) as archive:
        assert "Verified 2 retained tracked Word edit locations" in verify_document_source.check(archive, bundle, original)
    forged = deepcopy(export)
    forged["coverage"][tracked.KEY]["rows"][0]["inline_index"] += 1
    with pytest.raises(ValueError, match="source commitment"):
        tracked.verify_source(forged, review, expected=canonical_hash(review))
    forged["coverage"][tracked.KEY]["rows"][1]["inline_index"] += 1
    with pytest.raises(ValueError, match="original XML positions"):
        tracked.verify_source(forged, review, expected=canonical_hash(forged["coverage"][tracked.KEY]))
