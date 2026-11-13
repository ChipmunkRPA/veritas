from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pymupdf
import pytest
from test_document_packets import packet_factory as packet_factory

from app.api import v2_evidence_packets
from app.core.canonical import canonical_hash
from app.detectors.text import extract as extraction
from app.engine.customer_assessment import customer_assessment
from app.engine.fusion.registry import Selection
from app.engine.stages import document as stage
from app.evidence.document_packets import (
    _pdf_plates,
    _source_anchors,
    build_packet_set,
    load_packet_set,
    packet_hash,
)
from app.reports import engine, multimedia_evidence, public_package
from app.reports.specimens import specimen
from app.storage.objects import ObjectIntegrityError, ObjectStore

NATIVE = (
    "The visible caption contains text that can be checked. "
    "A second sentence preserves enough wording to record a source context. "
    "A third sentence keeps the evidence fixture independent of the scan."
)
SCAN_LINES = (
    "This raster-only paragraph begins with an independent sentence.",
    "Its second line lies below the native caption.",
    "Its third line can be read only by viewing the scan.",
)
SCAN_BOX = (40, 150, 570, 735)
WARNING = "Page 1 was partially checked."


def hybrid_pdf(path, *, box=SCAN_BOX, overlay=False, native=NATIVE, rotation=0, overlapping_caption=False):
    x0, y0, x1, y1 = box
    with pymupdf.open() as source, pymupdf.open() as pdf:
        scan = source.new_page(width=x1 - x0, height=y1 - y0)
        page = pdf.new_page(width=612, height=792)
        caption = (
            pymupdf.Rect(x0 + 4, y0 + 4, x1 - 4, y0 + 43)
            if overlapping_caption
            else pymupdf.Rect(40, 25, 570, 135)
        )
        for index, line in enumerate(SCAN_LINES):
            scan.insert_text((10, 25 + 22 * index), line, fontsize=14)
        page.insert_image(
            pymupdf.Rect(box), stream=scan.get_pixmap(matrix=pymupdf.Matrix(3, 3)).tobytes("png")
        )
        assert page.insert_textbox(caption, native, fontsize=12) >= 0
        if overlay:
            for index, line in enumerate(SCAN_LINES):
                page.insert_text((x0 + 10, y0 + 25 + 22 * index), line, fontsize=14, render_mode=3)
        page.set_rotation(rotation)
        pdf.save(path)


@pytest.mark.parametrize("rotation", [0, 90])
def test_scan_below_native_caption_is_partially_checked_if_ocr_is_unavailable(
    tmp_path, monkeypatch, rotation
):
    path = tmp_path / "hybrid.pdf"
    hybrid_pdf(path, rotation=rotation)
    monkeypatch.setattr(
        extraction, "_ocr_runtime", lambda: {"available": False, "reason": "OCR unavailable in fixture."}
    )
    doc = extraction.extract(path, ".pdf")
    assert doc.text == NATIVE and SCAN_LINES[0] not in doc.text
    assert doc.meta["unmapped_pages"] == []
    assert doc.meta["partially_checked_pages"] == [
        {
            "page": 1,
            "image_boxes": [list(SCAN_BOX)],
            "coordinate_space": "unrotated-page-points",
            "reason": "OCR unavailable in fixture.",
        }
    ]
    assert doc.meta["ocr"]["status"] == "partial"
    assert doc.meta["ocr"]["pages"][0]["status"] == "unchecked"
    assert doc.meta["ocr"]["pages_without_text"] == []
    assert WARNING in doc.meta["coverage_warning"]
    assert all(row["extraction"] == "text_layer" for row in doc.source_map)


@pytest.mark.parametrize("overlay", [False, True])
def test_small_figure_and_searchable_scan_keep_mapped_coverage(tmp_path, overlay):
    path = tmp_path / "covered.pdf"
    hybrid_pdf(path, box=SCAN_BOX if overlay else (40, 150, 260, 315), overlay=overlay)
    doc = extraction.extract(path, ".pdf")
    assert doc.meta["partially_checked_pages"] == []
    assert doc.meta["unmapped_pages"] == []
    assert doc.meta["ocr"]["status"] == "not_needed"
    assert (SCAN_LINES[0] in doc.text) == overlay


def test_decode_error_explains_the_unchecked_scan_when_caption_is_too_short(tmp_path, monkeypatch):
    path = tmp_path / "short.pdf"
    hybrid_pdf(path, native="A short native caption.", overlapping_caption=True)
    monkeypatch.setattr(stage.common, "media_path", lambda _: path)
    with pytest.raises(ValueError, match=r"too little text.*Page 1 was partially checked"):
        stage.decode(SimpleNamespace())


@pytest.mark.parametrize(("code", "score"), [("no_ai_indicators", 0.04), ("ai_supported", 0.96)])
def test_normal_fuse_path_withholds_both_document_wide_directions(tmp_path, monkeypatch, code, score):
    path = tmp_path / "partially-checked.pdf"
    hybrid_pdf(
        path,
        native="This native caption provides enough text for analysis. The scanned body remains in the same mapped area.",
        overlapping_caption=True,
    )
    monkeypatch.setattr(stage.common, "media_path", lambda _: path)
    monkeypatch.setattr(stage.settings, "text_detector_mode", "legacy")
    ctx = SimpleNamespace(
        analysis_id="hybrid-pdf-fixture",
        state={},
        outputs={},
        exhibits=SimpleNamespace(add=lambda *args, **kwargs: "fixture", exhibits=[], artefacts=[]),
    )
    decoded = stage.decode(ctx)
    ctx.outputs["decode"] = decoded
    assert stage.current_decode_output(decoded)
    old = deepcopy(decoded)
    old["meta"]["extraction_version"] = "document-extraction/3"
    assert not stage.current_decode_output(old)
    decision = {
        "code": code,
        "verbal": "Fixture estimate",
        "band": "uncertain",
        "confidence": "medium",
        "reason": "Authored test score",
        "fusion_mode": "artefact",
        "fuser_sha256": "a" * 64,
        "probability_ai_calibrated": score,
    }
    measured = {
        "id": "authored-check",
        "modality": "text",
        "scope": "global",
        "raw_score": score,
        "status": "ok",
        "calibrated_probability": score,
        "calibration_ref": "authored-calibration",
        "details": {"text_evidence": {"windows": [{"score": score}]}},
    }
    monkeypatch.setattr(
        stage.common, "select_fusion", lambda *args: Selection(kind="document", mode="artefact")
    )
    monkeypatch.setattr(
        stage.common,
        "artefact_fusion",
        lambda _ctx, _kind, _selection, out, _entries: {
            **out,
            "decision": decision,
            "signals": [measured],
            "condition_profile": {"slice_id": "fixture", "within_validated_envelope": True},
        },
    )
    result = stage.fuse(ctx)
    assert result["decision"]["code"] == "not_assessable"
    assert result["decision"]["probability_ai_calibrated"] is None
    assert result["decision"]["probability_ai_uncalibrated"] is None
    assert WARNING in result["decision"]["reason"]
    assert result["signals"][0]["raw_score"] == score
    assert result["signals"][0]["details"] == measured["details"]
    assert result["signals"][0]["calibrated_probability"] is None
    assert result["signals"][0]["calibration_ref"] is None
    assert measured["calibrated_probability"] == score
    assert result["condition_profile"]["within_validated_envelope"] is False
    view = customer_assessment({**result, "kind": "document"})
    assert view["direction"] == "unavailable" and view["confidence_pct"] is None


def test_partial_page_binds_to_public_evidence_and_pdf(packet_factory, tmp_path, monkeypatch):
    path = tmp_path / "source-hybrid.pdf"
    hybrid_pdf(path)
    monkeypatch.setattr(
        extraction, "_ocr_runtime", lambda: {"available": False, "reason": "OCR unavailable in fixture."}
    )
    db, store, analysis, result, decoded, _ = packet_factory(".pdf", source_pdf=path)
    evidence = build_packet_set(db, analysis, store, result)
    stored = load_packet_set(db, analysis, store)
    assert (
        stored == evidence
        and stored["coverage"]["partially_checked_pages"] == decoded["meta"]["partially_checked_pages"]
    )
    assert any(WARNING in line for line in evidence["limitations"])
    assert evidence["packets"][0]["anchors"][0]["page"] == 1
    assert "second sentence" in evidence["packets"][0]["focus"]["text"].lower()
    assert evidence["packets"][0]["artifacts"]["page1_crop"]["media_type"] == "image/png"
    monkeypatch.setattr(v2_evidence_packets, "get_session", lambda: nullcontext(db))
    monkeypatch.setattr(v2_evidence_packets, "default_store", lambda: store)
    response = v2_evidence_packets.image_evidence(analysis.id)
    assert response.status_code == 200 and json.loads(response.body) == evidence

    forged = deepcopy(evidence)
    forged["coverage"].pop("partially_checked_pages")
    forged["sha256"] = packet_hash(forged)
    with pytest.raises(ObjectIntegrityError, match="partial-page coverage"):
        multimedia_evidence.verify_assets(forged, store)

    snapshot = specimen("document", "standard")
    report_result = {**deepcopy(snapshot.result), "signals": result["signals"]}
    report_result["input"]["sha256"] = analysis.sha256
    report_result["manifest"]["sampling"] = deepcopy(result["manifest"]["sampling"])
    evidence = {
        **evidence,
        "analysis_id": snapshot.analysis.id,
        "revision": snapshot.analysis.revision,
        "original_result_hash": canonical_hash(report_result),
    }
    evidence["sha256"] = packet_hash(evidence)
    snapshot = snapshot.model_copy(
        update={
            "result": report_result,
            "evidence": evidence,
            "media": snapshot.media.model_copy(update={"sha256": analysis.sha256}),
        }
    )
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    assert public_package._evidence(snapshot)["coverage"] == evidence["coverage"]
    rendered = engine.render_pdf(snapshot)
    text = " ".join(engine.pdf_text(rendered.pdf).split())
    assert not rendered.refused and WARNING in text
    assert "mapped text and unchecked image areas" in text
    assert "Page 1. OCR unavailable in fixture." in text
    assert "Unchecked bounds in unrotated PDF points [40.0, 150.0, 570.0, 735.0]." in text


def test_native_ocr_does_not_raise_the_recognition_quality_of_the_image_region(tmp_path, monkeypatch):
    path = tmp_path / "hybrid-uncertain.pdf"
    hybrid_pdf(path)
    monkeypatch.setattr(
        extraction,
        "_ocr_runtime",
        lambda: {
            "available": True,
            "engine_version": "fixture",
            "language_data_sha256": {},
            "languages": "fixture",
        },
    )
    monkeypatch.setattr(extraction, "_ocr_language", lambda _text: {"status": "supported", "code": "en"})
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    native = [f"5\t1\t1\t1\t1\t{i + 1}\t{120 + i * 120}\t100\t100\t30\t99\tNative{i}" for i in range(10)]
    scan = [f"5\t1\t2\t1\t1\t{i + 1}\t{120 + i * 120}\t500\t100\t30\t20\tScanned{i}" for i in range(6)]
    monkeypatch.setattr(
        extraction,
        "_ocr_call",
        lambda *args, osd=False, **kwargs: "Rotate: 0" if osd else "\n".join([header, *native, *scan]),
    )
    doc = extraction.extract(path, ".pdf")
    record = doc.meta["ocr"]["pages"][0]
    assert doc.text == NATIVE and "Scanned0" not in doc.text
    assert record["status"] == "unchecked" and record["mean_recognition_confidence"] == 20
    assert "Native" not in record["recognized_text"]
    assert "quality is too low" in doc.meta["partially_checked_pages"][0]["reason"]
    assert record["raster"]["sha256"] in doc.source_rasters


def test_ocr_does_not_rejoin_a_paragraph_across_the_image_boundary():
    inside = {"bbox": [50, 155, 65, 166], "paragraph": [2, 1]}
    outside = {"bbox": [50, 120, 65, 131], "paragraph": [2, 1]}
    assert extraction._ocr_image_words([outside, inside], list(SCAN_BOX)) is None
    assert extraction._ocr_image_words([{**outside, "paragraph": [1, 1]}, inside], list(SCAN_BOX)) == [inside]


def test_real_local_hybrid_ocr_preserves_text_geometry_and_frozen_evidence(
    packet_factory, tmp_path, monkeypatch
):
    if not extraction._ocr_runtime().get("available"):
        pytest.skip("Verified local OCR runtime is unavailable")
    path = tmp_path / "hybrid-local.pdf"
    hybrid_pdf(path, rotation=90)
    store = ObjectStore(tmp_path / "stage-objects")
    monkeypatch.setattr(stage.common, "media_path", lambda _ctx: path)
    decoded = stage.decode(SimpleNamespace(analysis_id="hybrid-local-ocr", store=store))
    assert decoded["text"].count(NATIVE) == 1
    assert all(line in decoded["text"] for line in SCAN_LINES[:2])
    assert "third line can be read only by viewing the scan." in decoded["text"]
    assert decoded["meta"]["unmapped_pages"] == decoded["meta"]["partially_checked_pages"] == []
    assert stage.current_decode_output(decoded)
    (record,) = decoded["meta"]["ocr"]["pages"]
    assert record["coverage_scope"] == "image_region" and record["status"] == "extracted"
    assert record["image_box"] == list(SCAN_BOX)
    assert (
        "native caption" in record["recognized_text"]
        and "The visible caption" not in record["recognized_text"]
    )
    raster_sha = record["raster"]["sha256"]
    assert hashlib.sha256(store.read_bytes(raster_sha)).hexdigest() == raster_sha
    a, b, c, d, e, f = record["raster"]["input_to_page"]
    for word in record["words"]:
        x0, y0, x1, y1 = word["input_bbox"]
        corners = [(a * x + c * y + e, b * x + d * y + f) for x in (x0, x1) for y in (y0, y1)]
        x, y = zip(*corners, strict=True)
        assert word["bbox"] == pytest.approx([min(x), min(y), max(x), max(y)])
        assert SCAN_BOX[0] <= min(x) < max(x) <= SCAN_BOX[2]
        assert SCAN_BOX[1] <= min(y) < max(y) <= SCAN_BOX[3]
    ctx = SimpleNamespace(outputs={"decode": decoded})
    doc, _ = stage.document_input(ctx)
    measured = {"modality": "text", "scope": "global", "raw_score": 0.95, "calibrated_probability": 0.97}
    answer = stage.enforce_extraction_scope(
        {
            "decision": {
                "code": "ai_supported",
                "verbal": "Fixture",
                "band": "uncertain",
                "confidence": "medium",
                "reason": "Authored response fixture",
                "fusion_mode": "artefact",
                "fuser_sha256": "a" * 64,
                "probability_ai_calibrated": 0.97,
            },
            "signals": [measured],
        },
        doc,
    )
    assert answer["decision"]["probability_ai_calibrated"] is None
    assert "text transcribed from image areas" in answer["decision"]["reason"]
    assert (
        answer["signals"][0]["raw_score"] == 0.95 and answer["signals"][0]["calibrated_probability"] is None
    )
    db, packet_store, analysis, result, saved, _ = packet_factory(
        ".pdf", source_pdf=path, scores=(0.1, 0.1, 0.1, 0.93)
    )
    evidence = build_packet_set(db, analysis, packet_store, result)
    (packet,) = evidence["packets"]
    assert {row["extraction"] for row in packet["anchors"]} == {"text_layer", "ocr"}
    ocr_anchors = [row for row in packet["anchors"] if row["extraction"] == "ocr"]
    assert all(row["ocr"]["image_box"] == list(SCAN_BOX) for row in ocr_anchors)
    assert packet["artifacts"]["page1_ocr_input"]["source_raster_sha256"] == raster_sha
    assert any("Authorship confidence has not been validated" in line for line in evidence["limitations"])
    assert packet["focus"]["text"] == saved["text"][packet["focus"]["start"] : packet["focus"]["end"]]
    assert all(saved["text"][word["start"]:word["end"]] == word["text"]
               for anchor in ocr_anchors for word in anchor["highlighted_words"])
    multimedia_evidence.verify_assets(evidence, packet_store, result=result)
    anchors = _source_anchors(saved["source_map"], 0, len(saved["text"]), len(saved["text"]))
    assert {row["extraction"] for row in anchors} == {"text_layer", "ocr"}
    for order in (anchors, list(reversed(anchors))):
        plates = _pdf_plates(path.read_bytes(), order, packet_store)
        assert plates["page1_ocr_input"][1]["source_raster_sha256"] == raster_sha
    snapshot = specimen("document", "standard")
    report_result = {
        **deepcopy(snapshot.result),
        "signals": result["signals"],
        "decision": answer["decision"],
    }
    report_result["input"]["sha256"] = analysis.sha256
    report_result["manifest"]["sampling"] = deepcopy(result["manifest"]["sampling"])
    evidence = {
        **evidence,
        "analysis_id": snapshot.analysis.id,
        "revision": snapshot.analysis.revision,
        "original_result_hash": canonical_hash(report_result),
    }
    evidence["sha256"] = packet_hash(evidence)
    snapshot = snapshot.model_copy(
        update={
            "result": report_result,
            "evidence": evidence,
            "media": snapshot.media.model_copy(update={"sha256": analysis.sha256}),
        }
    )
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: packet_store)
    exported = public_package._evidence(snapshot)
    assert exported["packets"][0]["anchors"] == evidence["packets"][0]["anchors"]
    rendered = engine.render_pdf(snapshot)
    text = " ".join(engine.pdf_text(rendered.pdf).split())
    assert (
        not rendered.refused and "Page 1 combines mapped text with text transcribed from image areas" in text
    )
    assert "PDF page 1, recognized text" in text
