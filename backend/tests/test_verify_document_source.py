from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from email.message import EmailMessage

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from test_document_packets import packet_factory as packet_factory
from test_document_source_mapping import word_control, word_hyperlink, write_pdf_with_wrapped_hyphens
from test_public_evidence_package import rewrite
from test_public_package_integration import multimedia_package as multimedia_package
from test_strict_word import _strict_package

from app.cli import main
from app.core.canonical import canonical_hash, canonical_json
from app.detectors.text import email_body
from app.detectors.text.extract import WORD_CONTENT_CONTROLS_KEY, extract, text_hash
from app.evidence import document_native_locations, document_repeat, document_scope
from app.evidence.document_packets import build_packet_set, packet_hash
from app.reports import package, public_verify, verify, verify_document_source


def test_legacy_html_replays_with_the_recorded_encoding(packet_factory, tmp_path):
    original = tmp_path / "saved.htm"
    raw = ('<html><head><meta charset="windows-1252"></head><body><main>'
           '<p>First café sentence contains enough useful words.</p>'
           '<p>Second sentence adds “quoted words” and an exact € figure to the original.</p>'
           '<p>Third sentence completes the source.</p></main></body></html>')
    original.write_bytes(raw.encode("windows-1252"))
    db, store, analysis, result, _, _ = packet_factory(".htm", source_pdf=original)
    evidence = build_packet_set(db, analysis, store, result)
    key = document_scope.HTML_LEGACY_ENCODING_KEY
    assert evidence["coverage"][key] == document_scope.HTML_LEGACY_ENCODING
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    proof = tmp_path / "html-proof.zip"
    with zipfile.ZipFile(proof, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with zipfile.ZipFile(proof) as archive:
        assert "extracted text and source mapping matched" in verify_document_source.check(archive, bundle, original)
    del evidence["coverage"][key]
    with zipfile.ZipFile(proof, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with zipfile.ZipFile(proof) as archive, pytest.raises(ValueError, match="source mapping differs"):
        verify_document_source.check(archive, bundle, original)


@pytest.mark.parametrize("extension", [".html", ".xhtml", ".eml"])
@pytest.mark.parametrize("current", [False, True])
def test_html_table_cell_offsets_replay_under_the_recorded_policy(packet_factory, tmp_path, extension, current):
    rows = ("<table><tbody><tr><th>An old district's attendance record includes visitors</th>"
            "<td>New attendance permits are recorded separately for weekend visits</td></tr></tbody></table>")
    if extension == ".eml":
        message = EmailMessage()
        message.set_content(f"<html><body>{rows}</body></html>", subtype="html")
        data = message.as_bytes()
    else:
        data = f"<html><body><main>{rows}</main></body></html>".encode()
    original = tmp_path / ("table" + extension)
    original.write_bytes(data)
    db, store, analysis, result, _, _ = packet_factory(extension, source_pdf=original,
                                                      split_html_cells=current, scores=(0.9,))
    evidence = build_packet_set(db, analysis, store, result)
    key = document_scope.HTML_CELL_BOUNDARY_KEY
    assert evidence["coverage"].get(key) == (document_scope.HTML_CELL_BOUNDARY if current else None)
    assert any(document_scope.HTML_CELL_BOUNDARY_NOTICE in warning for warning in evidence["limitations"]) is current
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    proof = tmp_path / "table-proof.zip"
    with zipfile.ZipFile(proof, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with zipfile.ZipFile(proof) as archive:
        assert "extracted text and source mapping matched" in verify_document_source.check(archive, bundle, original)
    if current:
        del evidence["coverage"][key]
    else:
        evidence["coverage"][key] = document_scope.HTML_CELL_BOUNDARY
    with zipfile.ZipFile(proof, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with zipfile.ZipFile(proof) as archive, pytest.raises(ValueError, match="source mapping differs|plain-text or source-mapped"):
        verify_document_source.check(archive, bundle, original)


@pytest.mark.parametrize("extension", [".txt", ".eml", ".docx"])
def test_replays_exact_native_text_formats(extension, packet_factory, tmp_path):
    if extension == ".docx":
        original = tmp_path / "source.docx"
        document = Document()
        repeated = " ".join(f"Word {i}" for i in range(18))
        document.add_paragraph(repeated)
        document.add_table(rows=1, cols=1).cell(0, 0).text = repeated
        document.save(original)
        decoded = extract(original, extension)
        source = {"method": decoded.meta["method"], "sha256": decoded.meta["original_sha256"],
                  "extracted_text_sha256": text_hash(decoded.text), "source_map_sha256": canonical_hash(decoded.source_map),
                  "chars": len(decoded.text)}
        evidence = {"source": source, "version": document_scope.VERSION, "packets": [],
                    "repeated_passages": document_repeat.build(source, decoded.text, decoded.source_map)}
        assert evidence["repeated_passages"]["status"] == "matched"
    else:
        db, store, analysis, result, _, original = packet_factory(extension)
        evidence = build_packet_set(db, analysis, store, result)
        assert evidence["packets"]
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": evidence["source"]["sha256"], "bytes": original.stat().st_size}}
    with zipfile.ZipFile(tmp_path / "evidence.zip", "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with zipfile.ZipFile(tmp_path / "evidence.zip") as archive:
        assert "extracted text and source mapping matched" in verify_document_source.check(archive, bundle, original)


def test_html_alternative_replays_with_its_recorded_body_selection(packet_factory, tmp_path):
    original = tmp_path / "source.eml"
    text = ("First sentence contains enough useful words. Second sentence adds more context to this test. "
            "Third sentence completes the source.")
    message = EmailMessage()
    message.set_content(" \t\n")
    message.add_alternative(f"<html><body><p>{text}</p></body></html>", subtype="html")
    message.add_attachment("Named attachment must remain outside the analysis.", filename="notes.txt")
    original.write_bytes(message.as_bytes())
    db, store, analysis, result, _, _ = packet_factory(".eml", source_pdf=original)
    evidence = build_packet_set(db, analysis, store, result)
    email = evidence["coverage"]["email"]
    assert email["body_part"] == "1.1.2"
    assert email[email_body.SELECTION_KEY] == email_body.SELECTION_POLICY
    assert {anchor["mime_part"] for row in evidence["packets"] for anchor in row["anchors"]} == {"1.1.2"}
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    proof = tmp_path / "proof.zip"
    with zipfile.ZipFile(proof, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with zipfile.ZipFile(proof) as archive:
        assert "extracted text and source mapping matched" in verify_document_source.check(archive, bundle, original)
    del email[email_body.SELECTION_KEY]
    with zipfile.ZipFile(proof, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with zipfile.ZipFile(proof) as archive, pytest.raises(ValueError, match="source mapping differs"):
        verify_document_source.check(archive, bundle, original)


@pytest.mark.parametrize("extension", [".html", ".xhtml", ".eml"])
@pytest.mark.parametrize("current", [False, True])
def test_static_html_replays_the_recorded_policy(packet_factory, tmp_path, extension, current):
    original = tmp_path / ("source" + extension)
    html = ("<p>First sentence contains <span hidden>unseen </span>enough useful words.</p>"
            "<p>Second sentence adds more context to the exact original.</p>"
            "<p>Third sentence completes the document source.</p>")
    if extension == ".eml":
        message = EmailMessage()
        message.set_content(f"<html><body>{html}</body></html>", subtype="html", cte="quoted-printable")
        original.write_bytes(message.as_bytes())
    else:
        original.write_bytes(f"<html><body><main>{html}</main></body></html>".encode("utf-16"))
    db, store, analysis, result, _, _ = packet_factory(extension, source_pdf=original, use_static_html=current)
    evidence = build_packet_set(db, analysis, store, result)
    key = document_scope.HTML_STATIC_KEY
    assert (key in evidence["coverage"]) is current
    if current:
        assert evidence["coverage"][key] == (
            document_scope.XHTML_STATIC_POLICY if extension == ".xhtml" else document_scope.HTML_STATIC_POLICY)
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    proof = tmp_path / "html-proof.zip"
    renamed = tmp_path / "original.upload"
    renamed.write_bytes(original.read_bytes())
    with zipfile.ZipFile(proof, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with zipfile.ZipFile(proof) as archive:
        assert "extracted text and source mapping matched" in verify_document_source.check(archive, bundle, renamed)
    if current:
        del evidence["coverage"][key]
        with zipfile.ZipFile(proof, "w") as archive:
            archive.writestr("evidence.json", canonical_json(evidence))
        with zipfile.ZipFile(proof) as archive, pytest.raises(ValueError, match="source mapping differs|source-mapped HTML EML"):
            verify_document_source.check(archive, bundle, renamed)


@pytest.mark.parametrize("current", [False, True])
def test_pdf_evidence_replays_with_the_recorded_line_wrap_policy(packet_factory, tmp_path, current):
    path = tmp_path / "wrapped.pdf"
    write_pdf_with_wrapped_hyphens(path)
    db, store, analysis, result, decoded, _ = packet_factory(".pdf", source_pdf=path, preserve_pdf_wrap_hyphens=current)
    evidence = build_packet_set(db, analysis, store, result)
    assert evidence["packets"]
    key = document_native_locations.PDF_LINE_WRAP_KEY
    assert (key in evidence["coverage"]) is current
    assert ("operator-approved" in decoded["text"]) is current
    assert ("reconstruction" in decoded["text"]) is current
    assert any("Wrapped words follow the saved PDF text" in line for line in evidence["limitations"]) is current
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    proof = tmp_path / "proof.zip"
    with zipfile.ZipFile(proof, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
        for packet in evidence["packets"]:
            for asset in packet["artifacts"].values():
                digest = asset["sha256"]
                if f"assets/{digest}.png" not in archive.namelist():
                    archive.writestr(f"assets/{digest}.png", store.read_bytes(digest))
    with zipfile.ZipFile(proof) as archive:
        assert "extracted text and source mapping matched" in verify_document_source.check(archive, bundle, path)
        entries = {name: archive.read(name) for name in archive.namelist()}
    if current:
        del evidence["coverage"][key]
    else:
        evidence["coverage"][key] = document_native_locations.PDF_LINE_WRAP
    entries["evidence.json"] = canonical_json(evidence)
    with zipfile.ZipFile(proof, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(proof) as archive, pytest.raises(ValueError, match="source mapping differs"):
        verify_document_source.check(archive, bundle, path)


@pytest.mark.parametrize("modern", [True, False])
def test_original_replay_uses_the_recorded_word_control_projection(modern, tmp_path):
    original = tmp_path / "saved-template.docx"
    native = Document()
    text = " ".join(f"setup{i}" for i in range(32))
    native.add_paragraph(text)
    word_control([native.add_paragraph(text)._p])
    native.save(original)
    decoded = extract(original, ".docx", include_word_content_controls=modern)
    source = {"method": "python-docx", "sha256": decoded.meta["original_sha256"],
              "extracted_text_sha256": text_hash(decoded.text), "source_map_sha256": canonical_hash(decoded.source_map),
              "chars": len(decoded.text)}
    evidence = {"source": source, "version": document_scope.VERSION, "packets": [],
                "coverage": {WORD_CONTENT_CONTROLS_KEY: decoded.meta[WORD_CONTENT_CONTROLS_KEY]} if modern else {},
                "repeated_passages": document_repeat.build(source, decoded.text, decoded.source_map)}
    assert evidence["repeated_passages"]["status"] == ("matched" if modern else "not_assessed")
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": source["sha256"], "bytes": original.stat().st_size}}
    archive_path = tmp_path / "form-evidence.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with zipfile.ZipFile(archive_path) as archive:
        assert "extracted text and source mapping matched" in verify_document_source.check(archive, bundle, original)


@pytest.mark.parametrize("grid", [None, document_native_locations.WORD_TABLE_GRID_V1, document_native_locations.WORD_TABLE_GRID])
def test_original_replay_uses_the_recorded_word_table_grid(grid, tmp_path):
    original = tmp_path / "omitted-grid-cell.docx"
    native = Document()
    repeated = " ".join(f"reading{i}" for i in range(30))
    native.add_paragraph(repeated)
    table = native.add_table(rows=1, cols=3)
    table.cell(0, 1).text = "First visible value."
    table.cell(0, 2).text = repeated
    row = table.rows[0]._tr
    row.remove(row.tc_lst[0])
    omitted = OxmlElement("w:gridBefore")
    omitted.set(qn("w:val"), "1")
    row.get_or_add_trPr().append(omitted)
    if grid is not None:
        word_control([row])
    native.save(original)

    decoded = extract(original, ".docx", map_word_table_grid=grid is not None,
                      map_word_table_controls=grid == document_native_locations.WORD_TABLE_GRID)
    if grid == document_native_locations.WORD_TABLE_GRID_V1:
        assert [anchor["body_path"] for anchor in decoded.source_map] == [[0]]
        assert decoded.text == repeated
    else:
        column = int(grid is not None)
        assert [anchor["body_path"] for anchor in decoded.source_map] == [
            [0], [1, "row", 0, "cell", column, 1], [1, "row", 0, "cell", column + 1, 1],
        ]
        assert decoded.text == f"{repeated}\n\nFirst visible value.\n\n{repeated}"
    source = {"method": "python-docx", "sha256": decoded.meta["original_sha256"],
              "extracted_text_sha256": text_hash(decoded.text), "source_map_sha256": canonical_hash(decoded.source_map),
              "chars": len(decoded.text)}
    key = document_native_locations.WORD_TABLE_GRID_KEY
    coverage = {WORD_CONTENT_CONTROLS_KEY: decoded.meta[WORD_CONTENT_CONTROLS_KEY]}
    if grid is not None:
        coverage[key] = decoded.meta[key]
    evidence = {"source": source, "version": document_scope.VERSION, "packets": [], "coverage": coverage,
                "repeated_passages": document_repeat.build(source, decoded.text, decoded.source_map)}
    assert evidence["repeated_passages"]["status"] == (
        "not_assessed" if grid == document_native_locations.WORD_TABLE_GRID_V1 else "matched")
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": source["sha256"], "bytes": original.stat().st_size}}
    archive_path = tmp_path / "grid-evidence.zip"
    def replay():
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("evidence.json", canonical_json(evidence))
        with zipfile.ZipFile(archive_path) as archive:
            return verify_document_source.check(archive, bundle, original)

    assert "extracted text and source mapping matched" in replay()
    coverage[key] = (document_native_locations.WORD_TABLE_GRID_V1 if grid != document_native_locations.WORD_TABLE_GRID_V1
                     else document_native_locations.WORD_TABLE_GRID)
    with pytest.raises(ValueError, match="source mapping differs"):
        replay()


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("modern", [False, True])
def test_original_replay_uses_the_recorded_word_run_mapping(modern, strict, tmp_path):
    original = tmp_path / "linked.docx"
    native = Document()
    paragraph = native.add_paragraph("Please review the current reference 😀 to ")
    word_hyperlink(paragraph, "the café", " 📘 source")
    paragraph.add_run(" for full information about the schedule.")
    native.save(original)
    if strict:
        original.write_bytes(_strict_package(original.read_bytes()))

    decoded = extract(original, ".docx", map_word_hyperlink_runs=modern)
    assert decoded.text == "Please review the current reference 😀 to the café 📘 source for full information about the schedule."
    assert decoded.source_map[0]["run_mapping_complete"] is modern
    assert len(decoded.source_map[0]["runs"]) == (4 if modern else 2)
    source = {"method": "python-docx", "sha256": decoded.meta["original_sha256"],
              "extracted_text_sha256": text_hash(decoded.text), "source_map_sha256": canonical_hash(decoded.source_map),
              "chars": len(decoded.text)}
    key = document_native_locations.WORD_RUN_MAP_KEY
    evidence = {"source": source, "version": document_scope.VERSION, "packets": [],
                "coverage": {key: decoded.meta[key]} if modern else {},
                "repeated_passages": document_repeat.build(source, decoded.text, decoded.source_map)}
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": source["sha256"], "bytes": original.stat().st_size}}
    archive_path = tmp_path / "run-map.zip"

    def replay():
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("evidence.json", canonical_json(evidence))
        with zipfile.ZipFile(archive_path) as archive:
            return verify_document_source.check(archive, bundle, original)

    assert "extracted text and source mapping matched" in replay()
    evidence["coverage"] = {} if modern else {key: document_native_locations.WORD_RUN_MAP}
    with pytest.raises(ValueError, match="source mapping differs"):
        replay()


@pytest.mark.parametrize("multimedia_package", ["document"], indirect=True)
def test_cli_rejects_a_resigned_source_range_shift(multimedia_package, tmp_path, capsys):
    f = multimedia_package
    built = package.build(f.db, f.row, store=f.store, key=f.key, exported_at=datetime.now(UTC))
    original = f.store.path_for(f.analysis.sha256)
    args = ["--expect-key-id", f.key.key_id, "--source-document", str(original), "--json"]
    assert main(["verify", str(built.path), *args]) == 0
    original_receipt = json.loads(capsys.readouterr().out)
    assert original_receipt["checks"][-1]["ok"]
    assert "native PDF images (3)" in original_receipt["checks"][-1]["detail"]

    def change_anchor(files):
        evidence = json.loads(files["evidence.json"])
        evidence["packets"][0]["anchors"][0]["selected_start"] += 1
        evidence["sha256"] = packet_hash(evidence)
        files["evidence.json"] = canonical_json(evidence)
        bundle = json.loads(files["bundle.json"])
        bundle["evidence"].update(set_sha256=evidence["sha256"], sha256=hashlib.sha256(files["evidence.json"]).hexdigest())
        bundle["reports"][0]["evidence_set_sha256"] = evidence["sha256"]
        files["bundle.json"] = canonical_json(bundle)

    altered = rewrite(built.path, tmp_path / "shifted.zip", change_anchor, f.key)
    assert verify.verify_package(altered, expect_key_id=f.key.key_id).ok
    assert main(["verify", str(altered), *args]) == 1
    altered_receipt = json.loads(capsys.readouterr().out)
    assert "source positions differ" in altered_receipt["checks"][-1]["detail"]


@pytest.mark.parametrize("multimedia_package", ["hybrid_document", "scanned_document"], indirect=True)
def test_replays_signed_ocr_positions_and_source_pixels(multimedia_package, tmp_path, monkeypatch):
    f = multimedia_package
    built = package.build(f.db, f.row, store=f.store, key=f.key, exported_at=datetime.now(UTC))
    assert verify.verify_package(built.path, expect_key_id=f.key.key_id).ok
    original = f.store.path_for(f.analysis.sha256)
    replay = verify.verify_package(built.path, expect_key_id=f.key.key_id, source_document=original)
    assert replay.ok, replay.as_dict()
    assert "PDF images, including OCR source rasters (4)" in replay.checks[-1].detail

    def change_ocr_coverage(files):
        evidence = json.loads(files["evidence.json"])
        page = evidence["coverage"]["ocr"]["pages"][0]
        page["mean_recognition_confidence"] -= 1
        evidence["sha256"] = packet_hash(evidence)
        files["evidence.json"] = canonical_json(evidence)
        bundle = json.loads(files["bundle.json"])
        bundle["evidence"].update(set_sha256=evidence["sha256"], sha256=hashlib.sha256(files["evidence.json"]).hexdigest())
        bundle["reports"][0]["evidence_set_sha256"] = evidence["sha256"]
        files["bundle.json"] = canonical_json(bundle)

    altered = rewrite(built.path, tmp_path / "altered-ocr.zip", change_ocr_coverage, f.key)
    assert verify.verify_package(altered, expect_key_id=f.key.key_id).ok
    replay = verify.verify_package(altered, expect_key_id=f.key.key_id, source_document=original)
    assert not replay.ok
    assert "coverage differs" in replay.checks[-1].detail

    from app.detectors.text import extract as text_extract

    monkeypatch.setattr(text_extract, "_ocr_runtime", lambda: {"available": False, "reason": "Test without OCR"})
    replay = verify.verify_package(built.path, expect_key_id=f.key.key_id, source_document=original)
    assert not replay.ok
    assert "verified OCR data are installed" in replay.checks[-1].detail
