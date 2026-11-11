from __future__ import annotations

import base64
import dataclasses
import io
import json
import math
from copy import deepcopy

import numpy as np
import pymupdf
import pytest
from PIL import Image
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.text.extract import extract, text_hash
from app.detectors.text.segment import segment
from app.evidence.document_packets import (
    DocumentPacketIntegrityError,
    build_packet_set,
    load_packet_set,
    packet_hash,
    persist_packet_set,
    validate_packet_set,
    verify_context_window,
)
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, AnalysisStage, Artefact, Base
from app.storage.objects import ObjectStore


@pytest.fixture
def packet_factory(tmp_path):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    store = ObjectStore(tmp_path / "objects")
    with Session(engine) as db:
        def create(ext=".txt", scores=(0.1, 0.9, 0.2), ocr=False, text=None, source_pdf=None):
            path = source_pdf if source_pdf is not None else tmp_path / ("source" + ext)
            text = text or "First sentence contains enough useful words. Second sentence adds more context to this test. Third sentence completes the source."
            if ext == ".pdf" and source_pdf is None:
                pdf = pymupdf.open()
                page = pdf.new_page()
                page.insert_textbox(pymupdf.Rect(50, 50, 400, 300), text, fontsize=14)
                if ocr:
                    raster = page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes("png")
                    pdf.close()
                    pdf = pymupdf.open()
                    page = pdf.new_page()
                    page.insert_image(page.rect, stream=raster)
                page.set_rotation(90)
                pdf.save(path)
                pdf.close()
            elif ext != ".pdf":
                path.write_text("😀 " + text)
            source = store.put_bytes(path.read_bytes())
            doc = extract(path, ext)
            for content in doc.source_rasters.values():
                register_artefact(db, store, base64.b64decode(content), media_type="image/png", role="preview", created_by="document-ocr/1")
            sents = segment(doc)
            decoded = {"text": doc.text, "paragraphs": doc.paragraphs, "source_map": doc.source_map,
                       "meta": doc.meta, "sentences": [dataclasses.asdict(s) for s in sents], "words": sum(s.n_words for s in sents)}
            analysis = Analysis(id="document-fixture", kind="document", filename=path.name, content_type="application/pdf" if ext == ".pdf" else "text/plain",
                                size_bytes=source.size_bytes, sha256=source.sha256, status="done", revision=1, result_schema_version=2)
            db.add(analysis)
            db.flush()
            stage = register_artefact(db, store, canonical_json(decoded), media_type="application/json", role="stage_output",
                                      created_by="decode", analysis_id=analysis.id)
            db.add(AnalysisStage(analysis_id=analysis.id, stage="decode", status="done", output_artefact_id=stage.id))
            db.flush()
            windows = []
            for i, score in enumerate(scores):
                context = sents[max(0, i - 1):i + 2]
                windows.append({"unit_ref": f"sentence:{i}", "focus": {"start": sents[i].start, "end": sents[i].end},
                                "spans": [{"start": s.start, "end": s.end} for s in context],
                                "input_sha256": text_hash(" ".join(s.text for s in context)), "score": score,
                                "token_coverage": {"tokens": 30, "covered_tokens": 30, "token_windows": 1}})
            signal = {"id": "txt.desklib:global", "detector_id": "txt.desklib", "display_id": "S01", "status": "ok",
                      "name": "PRIVATE MODEL NAME", "details": {"text_evidence": {"version": "scored-context/1",
                      "offset_unit": "unicode_codepoint", "extracted_text_sha256": text_hash(doc.text), "windows": windows}}}
            result = {"input": {"sha256": source.sha256}, "manifest": {"sampling": {"document_source": {
                       "version": doc.meta["source_map_version"], "source_sha256": source.sha256,
                       "decode_sha256": stage.sha256, "source_map_sha256": canonical_hash(doc.source_map),
                       "extracted_text_sha256": text_hash(doc.text)}}}, "signals": [signal],
                       "findings": [{"id": "F1", "evidence_ids": [signal["id"]]}]}
            analysis.result_hash = canonical_hash(result)
            store.put_bytes(canonical_json(result))
            return db, store, analysis, result, decoded, path
        yield create


def test_frozen_contexts_are_bound_to_source_and_input_with_no_model_names(packet_factory):
    db, store, analysis, result, decoded, _ = packet_factory()
    original = deepcopy(result)
    evidence = build_packet_set(db, analysis, store, result)
    assert result == original and analysis.result_hash == canonical_hash(original)
    assert len(evidence["packets"]) == 1
    packet = evidence["packets"][0]
    assert packet["focus"]["text"].startswith("Second sentence")
    assert packet["context"]["text"] == decoded["text"]
    assert packet["supporting_windows"][0]["text"] == " ".join(s["text"] for s in decoded["sentences"])
    assert packet["finding_ids"] == ["F1"]
    assert packet["artifacts"] == {}
    assert packet["evidence_type"] == "statistical"
    assert "PRIVATE MODEL" not in json.dumps(evidence)
    assert "txt.desklib" not in json.dumps(evidence)
    assert evidence["sha256"] == packet_hash(evidence)
    assert evidence["coverage"]["scored_windows"] == 3
    assert evidence["coverage"]["candidate_spans"] == 1


def test_other_check_disagreement_is_retained_for_selected_context(packet_factory):
    db, store, analysis, result, _, _ = packet_factory()
    second = deepcopy(result["signals"][0])
    second["id"] = "txt.hf.hello_simpleai:global"
    second["display_id"] = "S02"
    for window in second["details"]["text_evidence"]["windows"]:
        window["score"] = 0.1
    result["signals"].append(second)
    analysis.result_hash = canonical_hash(result)
    evidence = build_packet_set(db, analysis, store, result)
    assert [w["score"] for w in evidence["packets"][0]["supporting_windows"]] == [0.9, 0.1]


def _add_ratio(result, decoded, score=1.4, threshold=0.9505):
    text = decoded["text"]
    token_ids = [0, *range(1, len(text) + 1)]
    context = {"id": "context:0", "spans": [{"start": 0, "end": len(text)}], "text": text,
               "input_sha256": text_hash(text), "input_token_ids": token_ids,
               "token_ids_sha256": canonical_hash(token_ids), "source_token_range": [0, len(text)],
               "scored_token_range": [0, len(text)], "token_offsets": [[i, i + 1] for i in range(len(text))],
               "anchor_token_count": 1, "context_limit": 1024, "stride": 768}
    windows = []
    for i, sentence in enumerate(decoded["sentences"]):
        start, end = sentence["start"], sentence["end"]
        windows.append({"unit_ref": f"sentence:{i}", "focus": {"start": start, "end": end},
                        "spans": context["spans"], "input_sha256": context["input_sha256"],
                        "score": score, "score_exact_hex": float(score).hex(), "threshold_exact_hex": float(threshold).hex(),
                        "score_kind": "perplexity_ratio", "threshold": threshold, "ai_when": "lt", "is_calibrated": False,
                        "scored_spans": [{"start": start, "end": end}], "scored_token_indices": list(range(start, end)),
                        "tokens_scored": end - start, "mean_negative_log_likelihood": score * 2,
                        "mean_cross_entropy": 2.0, "neighbor_expansion": False, "context_ids": ["context:0"],
                        "input_semantics": "Source envelope; exact input committed by token IDs.",
                        "token_coverage": {"tokens": end - start, "covered_tokens": end - start, "token_windows": 1}})
    signal = {"id": "binoculars_document", "detector_id": "txt.binoculars", "display_id": "S02", "status": "ok",
              "details": {"text_evidence": {"version": "scored-context/2", "offset_unit": "unicode_codepoint",
                         "extracted_text_sha256": text_hash(text), "shared_tokenization_verified": True,
                         "windows": windows, "contexts": [context], "omitted_windows": []}}}
    result["signals"].append(signal)
    return signal


def test_raw_ratio_above_one_is_retained_as_opposing_actual_context(packet_factory):
    db, store, analysis, result, decoded, _ = packet_factory()
    _add_ratio(result, decoded)
    analysis.result_hash = canonical_hash(result)
    evidence = build_packet_set(db, analysis, store, result)
    packet = evidence["packets"][0]
    classifier, ratio = packet["supporting_windows"]
    assert classifier["direction"] == "supports_ai" and classifier["score_kind"] == "classifier_probability"
    assert ratio["score"] == 1.4 and ratio["direction"] == "opposes_ai"
    assert ratio["threshold"] == 0.9505 and not ratio["is_calibrated"]
    assert "raw ratio" in [m["unit"] for m in packet["measurements"]]
    assert "1 are on the opposing side" in packet["observation"]
    assert ratio["text"] == decoded["text"]  # Exact input has more context than the highlighted sentence.
    assert ratio["scored_spans"] == [{"start": packet["focus"]["start"], "end": packet["focus"]["end"]}]
    assert "binoculars" not in json.dumps(evidence).lower()
    assert evidence["coverage"]["scored_windows"] == 6
    verify_context_window(ratio, decoded["text"])


def test_ratio_boundary_can_select_a_passage_even_when_classifier_opposes(packet_factory):
    db, store, analysis, result, decoded, _ = packet_factory(scores=(0.1, 0.2, 0.3))
    _add_ratio(result, decoded, score=0.8)
    analysis.result_hash = canonical_hash(result)
    evidence = build_packet_set(db, analysis, store, result)
    assert len(evidence["packets"]) == 3
    assert all([w["direction"] for w in packet["supporting_windows"]] == ["opposes_ai", "supports_ai"] for packet in evidence["packets"])


def test_six_card_bound_reserves_actual_disagreements_instead_of_hiding_them(packet_factory):
    text = " ".join(f"Sentence number {i} contains enough useful words for this exact source example." for i in range(8))
    db, store, analysis, result, decoded, _ = packet_factory(text=text, scores=(0.9,) * 8)
    signal = _add_ratio(result, decoded, score=0.8)
    for window in signal["details"]["text_evidence"]["windows"][5:]:
        window.update(score=1.4, score_exact_hex=(1.4).hex(), mean_negative_log_likelihood=2.8)
    analysis.result_hash = canonical_hash(result)
    evidence = build_packet_set(db, analysis, store, result)
    assert len(evidence["packets"]) == 6 and evidence["omitted_windows"] == 2
    opposing = [p for p in evidence["packets"] if any(w["direction"] == "opposes_ai" for w in p["supporting_windows"])]
    assert len(opposing) == 3
    assert {p["focus"]["start"] for p in opposing} == {s["start"] for s in decoded["sentences"][5:]}
    assert "reserved for disagreement" in evidence["coverage"]["selection_policy"]
    assert evidence["coverage"]["opposing_windows_recorded"] == evidence["coverage"]["opposing_windows_displayed"] == 3
    assert evidence["coverage"]["opposing_windows_omitted"] == 0


def test_opposing_responses_outside_display_limit_are_counted_explicitly(packet_factory):
    text = " ".join(f"Sentence number {i} contains enough useful words for this exact source example." for i in range(8))
    db, store, analysis, result, decoded, _ = packet_factory(text=text, scores=(0.9,) * 8)
    _add_ratio(result, decoded, score=1.4)
    analysis.result_hash = canonical_hash(result)
    evidence = build_packet_set(db, analysis, store, result)
    assert evidence["coverage"]["opposing_windows_recorded"] == 8
    assert evidence["coverage"]["opposing_windows_displayed"] == 6
    assert evidence["coverage"]["opposing_windows_omitted"] == 2
    assert any("2 recorded opposing" in line for line in evidence["limitations"])


def test_ratio_boundary_direction_survives_canonical_rounding(packet_factory):
    db, store, analysis, result, decoded, _ = packet_factory(scores=(0.1, 0.2, 0.3))
    _add_ratio(result, decoded, score=0.9504999)
    # The stage/result serializer rounds to six decimal places before packet construction.
    result = json.loads(canonical_json(result))
    analysis.result_hash = canonical_hash(result)
    evidence = build_packet_set(db, analysis, store, result)
    ratio = evidence["packets"][0]["supporting_windows"][1]
    assert ratio["score"] == ratio["threshold"] == 0.9505
    assert ratio["direction"] == "supports_ai"
    assert float.fromhex(ratio["score_exact_hex"]) < float.fromhex(ratio["threshold_exact_hex"])
    verify_context_window(ratio, decoded["text"])


@pytest.mark.parametrize("tamper", ["tokens", "offsets", "source", "reduction", "mean", "direction"])
def test_ratio_token_context_source_or_reduction_tamper_is_rejected(packet_factory, tamper):
    db, store, analysis, result, decoded, _ = packet_factory()
    _add_ratio(result, decoded)
    analysis.result_hash = canonical_hash(result)
    ratio = build_packet_set(db, analysis, store, result)["packets"][0]["supporting_windows"][1]
    if tamper == "tokens":
        ratio["inference_contexts"][0]["input_token_ids"][1] += 1
    elif tamper == "offsets":
        ratio["inference_contexts"][0]["token_offsets"][0] = [-1, 0]
    elif tamper == "source":
        ratio["inference_contexts"][0]["text"] = "x" + ratio["text"][1:]
    elif tamper == "reduction":
        ratio["scored_token_indices"].pop()
    elif tamper == "mean":
        ratio["mean_cross_entropy"] *= 2
    else:
        ratio["direction"] = "supports_ai"
    with pytest.raises(DocumentPacketIntegrityError):
        verify_context_window(ratio, decoded["text"])


def test_successful_nonlocal_checks_are_coverage_gaps_not_negative_evidence(packet_factory):
    db, store, analysis, result, _, _ = packet_factory()
    result["signals"].append({"id": "stylometry", "detector_id": "txt.stylometry", "display_id": "S03", "status": "ok"})
    result["signals"].append({"id": "sentence_consensus", "detector_id": "txt.stylometry", "display_id": "S04", "status": "ok"})
    analysis.result_hash = canonical_hash(result)
    evidence = build_packet_set(db, analysis, store, result)
    gaps = [c for c in evidence["coverage"]["checks"] if c["status"] == "context_not_recorded"]
    assert len(gaps) == 1  # Derived consensus is not another independent observation.
    assert "not opposing evidence" in gaps[0]["limitation"]
    assert len(evidence["packets"][0]["supporting_windows"]) == 1


def test_low_scores_do_not_create_suspicious_passages(packet_factory):
    db, store, analysis, result, _, _ = packet_factory(scores=(0.1, 0.2, 0.3))
    evidence = build_packet_set(db, analysis, store, result)
    assert evidence["packets"] == []
    assert evidence["coverage"]["scored_windows"] == 3
    assert "does not establish" in evidence["limitations"][0]


@pytest.mark.parametrize("bad", ["input", "offset", "extraction"])
def test_inconsistent_source_or_scored_context_is_rejected(packet_factory, bad):
    db, store, analysis, result, _, _ = packet_factory()
    window = result["signals"][0]["details"]["text_evidence"]["windows"][1]
    if bad == "input":
        window["input_sha256"] = "0" * 64
    elif bad == "offset":
        window["focus"]["start"] = -1
    else:
        result["manifest"]["sampling"]["document_source"]["decode_sha256"] = "0" * 64
    analysis.result_hash = canonical_hash(result)
    with pytest.raises(DocumentPacketIntegrityError):
        build_packet_set(db, analysis, store, result)


def test_reads_do_not_create_or_change_evidence_and_historical_mappings_are_not_invented(packet_factory):
    db, store, analysis, result, _, _ = packet_factory()
    before = db.scalar(select(func.count()).select_from(Artefact))
    assert load_packet_set(db, analysis, store) is None
    assert db.scalar(select(func.count()).select_from(Artefact)) == before
    historical = deepcopy(result)
    historical["manifest"]["sampling"] = {}
    analysis.result_hash = canonical_hash(historical)
    assert persist_packet_set(db, analysis, store, result=historical) is None
    assert db.scalar(select(func.count()).select_from(Artefact)) == before
    analysis.result_hash = canonical_hash(result)
    built = persist_packet_set(db, analysis, store, result=result)
    count = db.scalar(select(func.count()).select_from(Artefact))
    assert persist_packet_set(db, analysis, store, result=result) == built
    assert load_packet_set(db, analysis, store) == built
    assert db.scalar(select(func.count()).select_from(Artefact)) == count


def test_pdf_crops_are_exactly_the_recorded_source_render_with_rotation_bound(packet_factory):
    db, store, analysis, result, _, path = packet_factory(".pdf")
    evidence = build_packet_set(db, analysis, store, result)
    packet = evidence["packets"][0]
    assets = packet["artifacts"]
    assert set(assets) == {"page1_context", "page1_crop", "page1_overlay"}
    assert packet["anchors"][0]["page_rotation"] == 90
    art = assets["page1_crop"]
    with pymupdf.open(path) as pdf:
        pdf[0].set_rotation(0)
        pixmap = pdf[0].get_pixmap(matrix=pymupdf.Matrix(art["render_scale"], art["render_scale"]), alpha=False)
    rendered = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    x0, y0, x1, y1 = art["source_bbox"]
    scale = art["render_scale"]
    expected = rendered.crop((math.floor(x0 * scale), math.floor(y0 * scale), math.ceil(x1 * scale), math.ceil(y1 * scale)))
    crop = Image.open(io.BytesIO(store.read_bytes(art["sha256"])))
    np.testing.assert_array_equal(np.asarray(crop), np.asarray(expected))
    overlay = Image.open(io.BytesIO(store.read_bytes(assets["page1_overlay"]["sha256"])))
    assert np.any(np.asarray(overlay) != np.asarray(crop))
    assert all("assets/" in asset["url"] for asset in assets.values())


def test_tampered_context_rejected_even_with_recomputed_packet_hash(packet_factory):
    db, store, analysis, result, _, _ = packet_factory()
    evidence = build_packet_set(db, analysis, store, result)
    evidence["packets"][0]["context"]["text"] = "!" + evidence["packets"][0]["context"]["text"][1:]
    # Keep length but alter the selected span, not just unrelated context.
    p = evidence["packets"][0]
    i = p["focus"]["start"] - p["context"]["start"]
    p["context"]["text"] = p["context"]["text"][:i] + "!" + p["context"]["text"][i + 1:]
    evidence["sha256"] = packet_hash(evidence)
    with pytest.raises(DocumentPacketIntegrityError):
        validate_packet_set(evidence, analysis_id=analysis.id, revision=1,
                            original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)


def test_ocr_packet_crops_exact_frozen_engine_input_and_hides_implementation_names(packet_factory):
    from app.detectors.text.extract import _ocr_runtime
    if not _ocr_runtime().get("available"):
        pytest.skip("Verified local OCR fixture data are not provisioned")
    db, store, analysis, result, decoded, _ = packet_factory(".pdf", ocr=True)
    evidence = build_packet_set(db, analysis, store, result)
    packet = evidence["packets"][0]
    input_asset = packet["artifacts"]["page1_ocr_input"]
    crop_asset = packet["artifacts"]["page1_crop"]
    frozen = decoded["meta"]["ocr"]["pages"][0]["raster"]
    assert input_asset["sha256"] == frozen["sha256"]
    with Image.open(io.BytesIO(store.read_bytes(frozen["sha256"]))) as image:
        original = image.convert("RGB").rotate(frozen["rotation_clockwise"], expand=True)
    expected = original.crop(crop_asset["source_crop_pixels"])
    with Image.open(io.BytesIO(store.read_bytes(crop_asset["sha256"]))) as actual:
        assert np.array_equal(np.asarray(actual), np.asarray(expected))
    assert all("recognition_confidence" in word for word in packet["anchors"][0]["highlighted_words"])
    serialized = json.dumps(evidence)
    assert "tesseract" not in serialized.lower() and "traineddata" not in serialized
    assert "not AI" in serialized


def test_canonical_rounding_does_not_lose_exact_crop_pixel_bounds(tmp_path):
    from app.evidence.document_packets import _pdf_plates
    pdf = pymupdf.open()
    pdf.new_page(width=612, height=792)
    source = pdf.tobytes()
    pdf.close()
    raster = Image.fromarray(np.tile(np.arange(1700, dtype=np.uint8), (2200, 1)))
    stream = io.BytesIO()
    raster.save(stream, format="PNG")
    store = ObjectStore(tmp_path / "objects")
    obj = store.put_bytes(stream.getvalue())
    anchor = {"kind": "pdf", "page": 1, "page_rotation": 0, "page_width": 612, "page_height": 792,
              "highlight_boxes": [[47.88, 86.4, 533.88, 114.48]], "ocr": {"raster": {
                  "sha256": obj.sha256, "width": 1700, "height": 2200, "base_width": 1700, "base_height": 2200,
                  "rotation_clockwise": 0, "input_to_page": [0.36, 0, 0, 0.36, 0, 0], "colorspace": "grayscale"}}}
    crop, metadata = _pdf_plates(source, [anchor], store)["page1_crop"]
    exported = json.loads(canonical_json(metadata))
    # The 6-decimal serialized scale gives an extra column at this boundary.
    x0, y0, x1, y1 = exported["source_bbox"]
    sx, sy = exported["render_scale"], exported["render_scale_y"]
    rounded = (math.floor(x0 * sx), math.floor(y0 * sy), math.ceil(x1 * sx), math.ceil(y1 * sy))
    assert rounded != tuple(exported["source_crop_pixels"])
    assert np.array_equal(np.asarray(crop), np.asarray(raster.convert("RGB").crop(exported["source_crop_pixels"])))
