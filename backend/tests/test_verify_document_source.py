from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime

import pytest
from docx import Document
from test_document_packets import packet_factory as packet_factory
from test_public_evidence_package import rewrite
from test_public_package_integration import multimedia_package as multimedia_package

from app.cli import main
from app.core.canonical import canonical_hash, canonical_json
from app.detectors.text.extract import extract, text_hash
from app.evidence import document_repeat, document_scope
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


@pytest.mark.parametrize("multimedia_package", ["hybrid_document"], indirect=True)
def test_explicitly_refuses_to_replay_ocr(multimedia_package):
    f = multimedia_package
    built = package.build(f.db, f.row, store=f.store, key=f.key, exported_at=datetime.now(UTC))
    assert verify.verify_package(built.path, expect_key_id=f.key.key_id).ok
    replay = verify.verify_package(built.path, expect_key_id=f.key.key_id, source_document=f.store.path_for(f.analysis.sha256))
    assert not replay.ok
    assert "contains OCR evidence" in replay.checks[-1].detail
