from __future__ import annotations

import json
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pymupdf
import pytest
from test_document_packets import packet_factory as packet_factory

from app.api import v2_evidence_packets
from app.core.canonical import canonical_hash
from app.detectors.text.extract import extract
from app.engine.customer_assessment import customer_assessment
from app.engine.fusion.registry import Selection
from app.engine.stages import document as stage
from app.evidence.document_packets import build_packet_set, load_packet_set, packet_hash
from app.reports import engine, multimedia_evidence, public_package
from app.reports.specimens import specimen
from app.storage.objects import ObjectIntegrityError

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


def hybrid_pdf(path, *, box=SCAN_BOX, overlay=False, native=NATIVE, rotation=0):
    x0, y0, x1, y1 = box
    with pymupdf.open() as source, pymupdf.open() as pdf:
        scan = source.new_page(width=x1 - x0, height=y1 - y0)
        page = pdf.new_page(width=612, height=792)
        assert page.insert_textbox(pymupdf.Rect(40, 25, 570, 135), native, fontsize=12) >= 0
        for index, line in enumerate(SCAN_LINES):
            scan.insert_text((10, 25 + 22 * index), line, fontsize=10)
        page.insert_image(pymupdf.Rect(box), stream=scan.get_pixmap().tobytes("png"))
        if overlay:
            for index, line in enumerate(SCAN_LINES):
                page.insert_text((x0 + 10, y0 + 25 + 22 * index), line, fontsize=10, render_mode=3)
        page.set_rotation(rotation)
        pdf.save(path)


@pytest.mark.parametrize("rotation", [0, 90])
def test_scan_below_native_caption_is_partially_checked(tmp_path, rotation):
    path = tmp_path / "hybrid.pdf"
    hybrid_pdf(path, rotation=rotation)
    doc = extract(path, ".pdf")
    assert doc.text == NATIVE and SCAN_LINES[0] not in doc.text
    assert doc.meta["unmapped_pages"] == []
    assert doc.meta["partially_checked_pages"] == [
        {"page": 1, "image_boxes": [list(SCAN_BOX)], "coordinate_space": "unrotated-page-points"}
    ]
    assert doc.meta["ocr"]["status"] == "partial" and not doc.meta["ocr"]["pages"]
    assert WARNING in doc.meta["coverage_warning"]
    assert all(row["extraction"] == "text_layer" for row in doc.source_map)


@pytest.mark.parametrize("overlay", [False, True])
def test_small_figure_and_searchable_scan_keep_mapped_coverage(tmp_path, overlay):
    path = tmp_path / "covered.pdf"
    hybrid_pdf(path, box=SCAN_BOX if overlay else (40, 150, 260, 315), overlay=overlay)
    doc = extract(path, ".pdf")
    assert doc.meta["partially_checked_pages"] == []
    assert doc.meta["unmapped_pages"] == []
    assert doc.meta["ocr"]["status"] == "not_needed"
    assert (SCAN_LINES[0] in doc.text) == overlay


def test_decode_error_explains_the_unchecked_scan_when_caption_is_too_short(tmp_path, monkeypatch):
    path = tmp_path / "short.pdf"
    hybrid_pdf(path, native="A short native caption.")
    monkeypatch.setattr(stage.common, "media_path", lambda _: path)
    with pytest.raises(ValueError, match=r"too little text.*Page 1 was partially checked"):
        stage.decode(SimpleNamespace())


@pytest.mark.parametrize(("code", "score"), [("no_ai_indicators", 0.04), ("ai_supported", 0.96)])
def test_normal_fuse_path_withholds_both_document_wide_directions(tmp_path, monkeypatch, code, score):
    path = tmp_path / "partially-checked.pdf"
    hybrid_pdf(path)
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
    old["meta"]["extraction_version"] = "document-extraction/2"
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
