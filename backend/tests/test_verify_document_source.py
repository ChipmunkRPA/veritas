from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime

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
from app.detectors.text.extract import WORD_CONTENT_CONTROLS_KEY, extract, text_hash
from app.evidence import document_native_locations, document_repeat, document_scope
from app.evidence.document_packets import build_packet_set, packet_hash
from app.reports import package, public_verify, verify, verify_document_source


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
