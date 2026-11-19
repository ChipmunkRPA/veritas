from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
from copy import deepcopy
from types import SimpleNamespace
from zipfile import ZipFile

import pymupdf
import pytest
from PIL import Image
from test_document_packets import packet_factory as packet_factory

from app.core.canonical import canonical_json
from app.detectors.text import extract as extraction
from app.detectors.text.segment import segment
from app.engine.stages import document as stage
from app.evidence import document_scope
from app.evidence.document_packets import build_packet_set
from app.reports import public_verify, verify_document_source
from app.storage.objects import ObjectStore

TEXT = ("This is a scanned document with enough clear English words for reliable local transcription. "
        "Every selected sentence should be compared with the original page image. "
        "Recognition quality does not establish whether someone used artificial writing tools.")


def scanned_pdf(path, *, rotation=0, mixed=False):
    source = pymupdf.open()
    page = source.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(55, 60, 550, 720), TEXT, fontsize=17)
    raster = page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes("png")
    pdf = pymupdf.open()
    if mixed:
        page = pdf.new_page(width=612, height=792)
        page.insert_text((40, 50), "Exact text-layer contents remain unchanged.")
    page = pdf.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=raster)
    page.set_rotation(rotation)
    pdf.save(path)
    pdf.close()
    source.close()


def test_real_local_ocr_scanned_rotated_mixed_page(tmp_path):
    if not extraction._ocr_runtime().get("available"):
        pytest.skip("Verified local OCR fixture data are not provisioned")
    path = tmp_path / "mixed-rotated.pdf"
    scanned_pdf(path, rotation=90, mixed=True)
    doc = extraction.extract(path, ".pdf")
    assert doc.text.startswith("Exact text-layer contents remain unchanged.\n\n")
    assert "scanned document" in doc.text and "original page image" in doc.text
    assert doc.meta["ocr"]["status"] == "complete"
    assert doc.meta["ocr"]["pages_without_text"] == [2]
    assert doc.source_map[0]["extraction"] == "text_layer"
    anchor = next(a for a in doc.source_map if a.get("extraction") == "ocr")
    assert anchor["page"] == 2 and anchor["page_rotation"] == 90
    raster = anchor["ocr"]["raster"]
    content = base64.b64decode(doc.source_rasters[raster["sha256"]])
    assert hashlib.sha256(content).hexdigest() == raster["sha256"]
    with Image.open(io.BytesIO(content)) as image:
        assert image.size == (raster["width"], raster["height"])
    assert raster["measured_dpi_x"] == pytest.approx(200)
    for word in anchor["fragments"]:
        assert doc.text[word["start"]:word["end"]] == word["text"]
        assert 0 <= word["recognition_confidence"] <= 100
        assert 0 <= word["bbox"][0] < word["bbox"][2] <= anchor["page_width"]
        assert 0 <= word["bbox"][1] < word["bbox"][3] <= anchor["page_height"]


def test_blank_separators_leave_ocr_slots_for_saved_scan(tmp_path):
    if not extraction._ocr_runtime().get("available"):
        pytest.skip("Verified local OCR fixture data are not provisioned")
    scan = tmp_path / "scan.pdf"
    scanned_pdf(scan)
    path = tmp_path / "scan-after-blank-separators.pdf"
    blank_count = extraction.OCR_MAX_PAGES
    with pymupdf.open() as result, pymupdf.open(scan) as original:
        for _ in range(blank_count):
            result.new_page(width=144, height=216)
        result.insert_pdf(original)
        result.save(path)

    legacy = extraction.extract(path, ".pdf")
    assert legacy.text == "" and document_scope.PDF_OCR_BLANK_PAGES not in legacy.meta["ocr"]
    assert "budget exhausted" in legacy.meta["ocr"]["pages"][-1]["reason"]

    current = extraction.extract(path, ".pdf", skip_blank_pdf_pages=True)
    reference = extraction.extract(scan, ".pdf")
    assert current.text == reference.text and "scanned document" in current.text
    assert current.meta["ocr"]["version"] == document_scope.PDF_OCR_BLANK_VERSION
    assert current.meta["ocr"][document_scope.PDF_OCR_BLANK_PAGES] == list(range(1, blank_count + 1))
    assert current.meta["ocr"]["status"] == "complete"
    assert current.meta["ocr"]["unchecked_pages"] == []
    assert [(row["page"], row["status"]) for row in current.meta["ocr"]["pages"]] == [(blank_count + 1, "extracted")]
    assert {row["page"] for row in current.source_map} == {blank_count + 1}
    assert current.source_rasters == reference.source_rasters


@pytest.mark.parametrize("kind", ["empty", "empty_arrays", "vector", "annotation", "widget", "link"])
def test_blank_page_requires_absence_of_saved_ops_and_annotations(tmp_path, kind):
    path = tmp_path / "page.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=400)
        box = pymupdf.Rect(10, 10, 200, 100)
        if kind == "empty_arrays":
            doc.xref_set_key(page.xref, "Contents", "[]")
            doc.xref_set_key(page.xref, "Annots", "[]")
        elif kind == "vector":
            page.draw_line((10, 10), (200, 100))
        elif kind == "annotation":
            page.add_freetext_annot(box, "Visible annotation")
        elif kind == "widget":
            widget = pymupdf.Widget()
            widget.field_name = "Test field"
            widget.field_value = "Visible saved value"
            widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
            widget.rect = box
            page.add_widget(widget)
        elif kind == "link":
            page.insert_link({"kind": pymupdf.LINK_GOTO, "from": box, "page": 0})
        doc.save(path)
    with pymupdf.open(path) as doc:
        assert extraction._structurally_blank_pdf_page(doc[0]) is (kind in {"empty", "empty_arrays"})


@pytest.mark.parametrize("current", [False, True])
def test_original_replays_historical_ocr_and_rejects_a_forged_blank_subset(tmp_path, packet_factory, current):
    if not extraction._ocr_runtime().get("available"):
        pytest.skip("Verified local OCR fixture data are not provisioned")
    scan = tmp_path / "scan.pdf"
    scanned_pdf(scan)
    path = tmp_path / "scan-with-separator.pdf"
    with pymupdf.open() as result, pymupdf.open(scan) as original:
        result.new_page(width=144, height=216)
        result.insert_pdf(original)
        result.save(path)
    db, store, analysis, result, _, _ = packet_factory(".pdf", source_pdf=path, scores=(), skip_blank_pdf_pages=current)
    evidence = build_packet_set(db, analysis, store, result)
    ocr = evidence["coverage"]["ocr"]
    key = document_scope.PDF_OCR_BLANK_PAGES
    assert ocr.get(key) == ([1] if current else None)
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    archive_path = tmp_path / "proof.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with ZipFile(archive_path) as archive:
        summary = verify_document_source.check(archive, bundle, path)
    assert ("Verified the structurally blank PDF pages" in summary) is current
    if current:
        ocr[key] = []
        with ZipFile(archive_path, "w") as archive:
            archive.writestr("evidence.json", canonical_json(evidence))
        with ZipFile(archive_path) as archive, pytest.raises(ValueError, match="coverage differs from the re-extracted original"):
            verify_document_source.check(archive, bundle, path)


def _fake_tsv(confidence=95):
    words = ["本地😀", "识别文字", "保留原始页面", "每个文字位置", "需要人工复核", "不能证明作者身份"]
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    return header + "\n".join(f"5\t1\t1\t1\t1\t{i + 1}\t{20 + i * 110}\t40\t100\t30\t{confidence}\t{word}" for i, word in enumerate(words))


@pytest.mark.parametrize(("clockwise", "expected"), [
    (0, [7.2, 14.4, 43.2, 25.2]), (90, [14.4, 748.8, 25.2, 784.8]),
    (180, [568.8, 766.8, 604.8, 777.6]), (270, [586.8, 7.2, 597.6, 43.2]),
])
def test_ocr_unicode_offsets_and_orientation_inverse(tmp_path, monkeypatch, clockwise, expected):
    path = tmp_path / "unicode.pdf"
    scanned_pdf(path)
    monkeypatch.setattr(extraction, "_ocr_runtime", lambda: {"available": True, "engine_version": "fixture", "language_data_sha256": {}, "languages": "fixture"})
    monkeypatch.setattr(extraction, "_ocr_call", lambda *a, osd=False, **kw: f"Rotate: {clockwise}\nOrientation confidence: 9\nScript: Han\nScript confidence: 8\n" if osd else _fake_tsv())
    monkeypatch.setattr(extraction, "_ocr_language", lambda text: {"status": "supported", "code": "zh", "confidence": 0.99})
    doc = extraction.extract(path, ".pdf")
    anchor = doc.source_map[0]
    assert doc.text.startswith("本地😀 ")
    assert anchor["fragments"][0]["end"] == 3  # Unicode codepoints, not UTF-16 units
    raster = anchor["ocr"]["raster"]
    assert raster["rotation_clockwise"] == clockwise
    assert (raster["width"], raster["height"]) == ((2200, 1700) if clockwise in (90, 270) else (1700, 2200))
    # Pixel box [20,40,120,70] mapped back through a clockwise-90 correction.
    assert anchor["fragments"][0]["bbox"] == pytest.approx(expected)


@pytest.mark.parametrize("failure", ["confidence", "language", "timeout"])
def test_uncertain_ocr_pages_never_become_scored_text(tmp_path, monkeypatch, failure):
    path = tmp_path / "unreadable.pdf"
    scanned_pdf(path, mixed=True)
    monkeypatch.setattr(extraction, "_ocr_runtime", lambda: {"available": True, "engine_version": "fixture", "language_data_sha256": {}, "languages": "fixture"})

    def call(*args, osd=False, **kwargs):
        if osd or failure == "timeout":
            raise subprocess.TimeoutExpired("fixture", 1)
        return _fake_tsv(20 if failure == "confidence" else 95)

    monkeypatch.setattr(extraction, "_ocr_call", call)
    monkeypatch.setattr(extraction, "_ocr_language", lambda text: {"status": "unknown", "code": None, "confidence": None, "reason": "Language uncertain."})
    doc = extraction.extract(path, ".pdf")
    assert doc.text == "Exact text-layer contents remain unchanged."
    assert doc.meta["unmapped_pages"] == [2] and doc.meta["ocr"]["status"] == "partial"
    assert doc.meta["ocr"]["pages"][0]["status"] == "unchecked"
    assert doc.source_rasters  # Even failed recognition keeps the actual input, when it existed.
    assert "must not be treated as human-authored" in doc.meta["coverage_warning"]


def test_decode_freezes_rasters_and_resume_requires_them(tmp_path, monkeypatch):
    path = tmp_path / "source.txt"
    path.write_text(TEXT)
    doc = extraction.from_string(TEXT)
    image = io.BytesIO()
    Image.new("L", (12, 8), "white").save(image, format="PNG")
    raw = image.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    doc.source_rasters[digest] = base64.b64encode(raw).decode()
    doc.meta["ocr"] = {"pages": [{"raster": {"sha256": digest}}]}
    monkeypatch.setattr(stage.settings, "text_doi_reference_review", False)
    monkeypatch.setattr(stage.common, "media_path", lambda ctx: path)
    monkeypatch.setattr(stage, "decode_document", lambda *args, **kwargs: (doc, segment(doc)))
    store = ObjectStore(tmp_path / "objects")
    output = stage.decode(SimpleNamespace(analysis_id="ocr-test", store=store, outputs={}, state={"kind": "document"}))
    assert output["artefacts"][0]["sha256"] == digest and store.read_bytes(digest) == raw
    assert "source_rasters" not in output and base64.b64encode(raw).decode() not in json.dumps(output)
    assert stage.current_decode_output(output)
    missing = deepcopy(output)
    missing["artefacts"] = []
    assert not stage.current_decode_output(missing)
    old = deepcopy(output)
    del old["meta"]["extraction_version"]
    assert not stage.current_decode_output(old)


@pytest.mark.parametrize("mode", ["fallback", "artefact"])
def test_mixed_pdf_unchecked_page_withholds_whole_document_verdict_but_preserves_measurements(tmp_path, monkeypatch, mode):
    path = tmp_path / "partly-readable.pdf"
    scanned_pdf(path, mixed=True)
    monkeypatch.setattr(extraction, "_ocr_runtime", lambda: {"available": False, "reason": "OCR unavailable in fixture"})
    doc = extraction.extract(path, ".pdf")
    assert doc.meta["unmapped_pages"] == [2]
    score_key = "probability_ai_calibrated" if mode == "artefact" else "probability_ai_uncalibrated"
    decision = {"code": "no_ai_indicators", "verbal": "No AI indicators", "band": "likely_authentic",
                "confidence": "medium", "reason": "Measured extracted text only", "fusion_mode": mode,
                score_key: 0.08, "fuser_sha256": "a" * 64 if mode == "artefact" else None}
    measured = {"id": "check1", "score": 0.08}
    original = {"decision": decision, "signals": [measured], "caveats": [],
                "assessments": [{"hypothesis": "H1_whole_content_synthesis", "applicable": True, "probability": 0.08}],
                "v1": {"verdict": {"probability_ai": 0.08, "caveats": []}, "sentences": [{"text": doc.text, "p_ai": 0.08}]}}
    result = stage.enforce_extraction_scope(original, doc)
    assert result["decision"]["code"] == "not_assessable"
    assert result["decision"]["probability_ai_calibrated"] is None
    assert result["decision"]["probability_ai_uncalibrated"] is None
    assert result["v1"]["verdict"]["probability_ai"] is None
    assert result["assessments"][0]["applicable"] is False
    assert "Pages 2" in result["decision"]["reason"]
    assert result["signals"] == [measured] and result["v1"]["sentences"] == original["v1"]["sentences"]
    assert original["decision"][score_key] == 0.08  # preserve original measured object
