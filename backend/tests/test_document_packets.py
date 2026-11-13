from __future__ import annotations

import base64
import dataclasses
import io
import json
import math
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pymupdf
import pytest
from PIL import Image
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.text.extract import extract, text_hash
from app.detectors.text.segment import segment
from app.evidence import document_passage_coverage, document_unselected_responses
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
            elif ext == ".eml" and source_pdf is None:
                from email.message import EmailMessage

                message = EmailMessage()
                message.set_content("😀 " + text)
                message.add_alternative("<p>Alternate fixture body.</p>", subtype="html")
                message.add_attachment("Excluded fixture attachment.", filename="private-attachment-name.txt")
                path.write_bytes(message.as_bytes())
            elif ext != ".pdf" and source_pdf is None:
                path.write_text("😀 " + text)
            source = store.put_bytes(path.read_bytes())
            doc = extract(path, ext)
            for content in doc.source_rasters.values():
                register_artefact(db, store, base64.b64decode(content), media_type="image/png", role="preview", created_by="document-ocr/1")
            sents = segment(doc)
            decoded = {"text": doc.text, "paragraphs": doc.paragraphs, "source_map": doc.source_map,
                       "meta": doc.meta, "sentences": [dataclasses.asdict(s) for s in sents], "words": sum(s.n_words for s in sents)}
            content_type = {".pdf": "application/pdf", ".eml": "message/rfc822",
                            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}.get(ext, "text/plain")
            analysis = Analysis(id="document-fixture", kind="document", filename=path.name, content_type=content_type,
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
    ranges = [{"start": sentence["start"], "end": sentence["end"]} for sentence in decoded["sentences"]]
    assert packet["inspection_scope"] == {"basis": "scored_context", "spans": ranges}
    assert [(anchor["selected_start"], anchor["selected_end"]) for anchor in packet["anchors"]] == [
        (span["start"], span["end"]) for span in ranges
    ]
    assert "navigation sentence" in packet["interpretation"]
    assert packet["finding_ids"] == ["F1"]
    assert packet["artifacts"] == {}
    assert packet["evidence_type"] == "statistical"
    assert "PRIVATE MODEL" not in json.dumps(evidence)
    assert "txt.desklib" not in json.dumps(evidence)
    assert evidence["sha256"] == packet_hash(evidence)
    assert evidence["coverage"]["scored_windows"] == 3
    assert evidence["coverage"]["candidate_spans"] == 1


def test_short_email_maps_opposing_context_only_and_unassessed_quote_without_cards(packet_factory):
    body = "Please send the agenda. The date changed.\n\n> This was in the old message."
    db, store, analysis, result, decoded, _ = packet_factory(".eml", scores=(.2,), text=body)
    evidence = build_packet_set(db, analysis, store, result)
    record = evidence["coverage"][document_passage_coverage.KEY]
    assert evidence["packets"] == []
    additional = evidence["coverage"][document_unselected_responses.KEY]
    assert additional["affects_overall_confidence"] is False and additional["omitted_windows"] == 0
    row, = additional["rows"]
    assert row["id"] == "U01" and row["direction"] == "opposes_ai" and row["response"] == .2
    assert row["text"].endswith("The date changed.") and "old message" not in row["text"]
    assert all(source["quote_role"] == "body" for source in row["sources"])
    assert document_unselected_responses.verify_source(evidence, decoded["text"], decoded["source_map"], result=result) == 1
    assert record["counts"] == {
        state: sum(not char.isspace() for char in sentence["text"])
        for state, sentence in zip(("response_range", "context_only", "no_local_record"), decoded["sentences"], strict=True)
    }
    for row, sentence in zip(record["ranges"], decoded["sentences"], strict=True):
        assert (row["start"], row["end"]) == (sentence["start"], sentence["end"])
    assert [row["quote_role"] for row in record["ranges"]] == ["body", "body", "quoted"]
    document_passage_coverage.verify_source(evidence, decoded["text"], decoded["source_map"], result=result)

    forged = deepcopy(evidence)
    changed = forged["coverage"][document_passage_coverage.KEY]
    first, second = changed["ranges"][:2]
    first["state"], second["state"] = second["state"], first["state"]
    for row in (first, second):
        changed["counts"][row["state"]] = row["characters"]
    forged["sha256"] = packet_hash(forged)
    with pytest.raises(ValueError, match="differs from the frozen responses"):
        document_passage_coverage.verify_source(forged, decoded["text"], decoded["source_map"], result=result)


def test_additional_responses_preserve_selected_cards_and_reject_forged_source_or_direction(packet_factory):
    db, store, analysis, result, decoded, _ = packet_factory(".pdf")
    evidence = build_packet_set(db, analysis, store, result)
    rows = evidence["coverage"][document_unselected_responses.KEY]["rows"]
    assert len(evidence["packets"]) == 1 and len(rows) == 2
    assert all(row["focus"] != {key: evidence["packets"][0]["focus"][key] for key in ("start", "end")} for row in rows)
    assert all(row["direction"] == "opposes_ai" and all(source["page"] == 1 for source in row["sources"]) for row in rows)
    forged = deepcopy(evidence)
    forged["coverage"][document_unselected_responses.KEY]["rows"][0]["sources"][0]["page"] = 2
    with pytest.raises(ValueError, match="differs from its original text or location"):
        document_unselected_responses.verify_source(forged, decoded["text"], decoded["source_map"])
    forged = deepcopy(evidence)
    forged["coverage"][document_unselected_responses.KEY]["rows"][0]["direction"] = "supports_ai"
    with pytest.raises(ValueError, match="differ from the frozen classifier records"):
        document_unselected_responses.verify_source(forged, decoded["text"], decoded["source_map"], result=result)


def test_additional_response_limits_preserve_exact_input_and_drop_incomplete_windows(packet_factory):
    text = " ".join(f"Paragraph sentence number {index} includes exact recorded writing." for index in range(18))
    db, store, analysis, result, decoded, _ = packet_factory(scores=(.1,) * 18, text=text)
    result["signals"][0]["details"]["text_evidence"]["windows"][0]["token_coverage"]["omitted_tokens"] = 1
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    evidence = build_packet_set(db, analysis, store, result)
    record = evidence["coverage"][document_unselected_responses.KEY]
    assert record["eligible_windows"] == 17 and record["omitted_windows"] == 5
    assert len(record["rows"]) == 12 and record["rows"][0]["focus"]["start"] == decoded["sentences"][1]["start"]
    assert load_packet_set(db, analysis, store) == evidence


def test_additional_responses_replay_canonical_values_and_keep_exact_direction_when_available(packet_factory):
    db, store, analysis, result, decoded, _ = packet_factory(scores=(.4999998, .4999998, .2))
    windows = result["signals"][0]["details"]["text_evidence"]["windows"]
    windows[1]["score_exact_hex"] = float(windows[1]["score"]).hex()
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    evidence = build_packet_set(db, analysis, store, result)
    first, second, _ = evidence["coverage"][document_unselected_responses.KEY]["rows"]
    assert first["direction"] == "supports_ai" and second["direction"] == "opposes_ai"
    assert first["response"] == second["response"] == .5
    assert load_packet_set(db, analysis, store) == evidence


def test_partial_classifier_has_no_exact_coverage_and_ratio_keeps_only_committed_token_positions():
    text = "A B C D E"
    mapping = [{"start": 0, "end": len(text), "kind": "text"}]
    a, b = {"start": 0, "end": 1}, {"start": 2, "end": 3}
    classify = {"version": "scored-context/1", "offset_unit": "unicode_codepoint",
                "extracted_text_sha256": text_hash(text),
                "windows": [{"focus": a, "spans": [a, b], "token_coverage": {"tokens": 2, "covered_tokens": 1}}]}
    ratio = {"version": "scored-context/2", "offset_unit": "unicode_codepoint", "extracted_text_sha256": text_hash(text),
             "windows": [{"score_kind": "perplexity_ratio", "context_ids": ["context"], "scored_token_indices": [2]}],
             "contexts": [{"id": "context", "source_token_range": [1, 3], "token_offsets": [[4, 5], [8, 9]]}]}
    result = {"signals": [{"status": "ok", "details": {"text_evidence": item}} for item in (classify, ratio)]}
    record = document_passage_coverage.build(result, text, mapping)
    assert [(row["state"], row["start"], row["end"]) for row in record["ranges"]] == [
        ("no_local_record", 0, 3), ("context_only", 4, 5), ("no_local_record", 6, 7), ("response_range", 8, 9),
    ]
    assert record["counts"] == {"response_range": 1, "context_only": 1, "no_local_record": 3}


def test_coverage_counts_every_source_block_even_when_the_export_range_list_is_bounded():
    blocks = document_passage_coverage.MAX_RANGES + 12
    text = "\n\n".join("A" for _ in range(blocks))
    mapping = [{"start": 3 * i, "end": 3 * i + 1, "kind": "text"} for i in range(blocks)]
    record = document_passage_coverage.build({}, text, mapping)
    assert record["counts"] == {"response_range": 0, "context_only": 0, "no_local_record": blocks}
    assert len(record["ranges"]) == document_passage_coverage.MAX_RANGES and record["omitted_ranges"] == 12
    document_passage_coverage.verify_source(
        {"source": {"chars": len(text)}, "coverage": {document_passage_coverage.KEY: record}}, text, mapping, result={},
    )


def test_pdf_page_coverage_distinguishes_recorded_text_and_unread_source_pages():
    text = "A é\n\nC D\n\nE F\n\nG H\n\nI J"
    spans = [(0, 3), (5, 8), (10, 13), (15, 18), (20, 23)]
    mapping = [{"start": a, "end": b, "kind": "pdf", "page": page, "extraction": mode}
               for (a, b), page, mode in zip(spans, (1, 3, 4, 5, 5),
                   ("text_layer", "text_layer", "ocr", "text_layer", "ocr"), strict=True)]
    ocr = {"pages": [{"page": 2, "status": "unchecked"}, {"page": 4, "status": "extracted"},
                     {"page": 5, "status": "extracted", "coverage_scope": "image_region"}]}
    partial = [{"page": 3}]
    meta = {"method": "pymupdf", "pages": 6, "partially_checked_pages": partial}
    first, third = ({"start": a, "end": b} for a, b in spans[:2])
    window = {"focus": first, "spans": [first, third], "token_coverage": {"tokens": 4}}
    signal = {"status": "ok", "details": {"text_evidence": {"version": "scored-context/1",
        "offset_unit": "unicode_codepoint", "extracted_text_sha256": text_hash(text), "windows": [window]}}}
    result = {"signals": [signal]}
    record = document_passage_coverage.build(result, text, mapping, pdf_meta=meta)
    pdf = record["pdf_pages"]
    assert pdf["unmapped_page_ranges"] == [[2, 2], [6, 6]]
    assert [(row["page"], row["extraction"], row["unchecked_image_areas"]) for row in pdf["mapped_pages"]] == [
        (1, "text_layer", False), (3, "text_layer", True), (4, "ocr", False), (5, "mixed", False)]
    assert [row["counts"] for row in pdf["mapped_pages"]] == [
        {"response_range": 2, "context_only": 0, "no_local_record": 0},
        {"response_range": 0, "context_only": 2, "no_local_record": 0},
        {"response_range": 0, "context_only": 0, "no_local_record": 2},
        {"response_range": 0, "context_only": 0, "no_local_record": 4},
    ]
    document = {"source": {"method": "pymupdf", "chars": len(text)}, "coverage": {
        document_passage_coverage.KEY: record, "ocr": ocr, "partially_checked_pages": partial, "unmapped_pages": [2, 6]}}
    document_passage_coverage.verify_source(document, text, mapping, result=result, meta=meta)
    document_passage_coverage.verify_source(document, text, mapping, meta=meta)


def test_pdf_page_replay_checks_counts_beyond_the_retained_range_limit():
    first_page = document_passage_coverage.MAX_RANGES + 2
    total = first_page + 3
    text = "A\n\n" * total
    mapping = [{"start": i * 3, "end": i * 3 + 1, "page": 1 if i < first_page else 2, "kind": "pdf"}
               for i in range(total)]
    meta = {"method": "pymupdf", "pages": 2}
    record = document_passage_coverage.build({}, text, mapping, pdf_meta=meta)
    document = {"source": {"method": "pymupdf", "chars": len(text)}, "coverage": {document_passage_coverage.KEY: record}}
    assert record["omitted_ranges"] == 5
    assert [row["counts"]["no_local_record"] for row in record["pdf_pages"]["mapped_pages"]] == [first_page, 3]
    document_passage_coverage.verify_source(document, text, mapping, meta=meta)

    first, second = record["pdf_pages"]["mapped_pages"]
    first["counts"]["no_local_record"] -= 1
    second["counts"]["no_local_record"] += 1
    document_passage_coverage.validate(document)
    with pytest.raises(ValueError, match="differs from the original extraction"):
        document_passage_coverage.verify_source(document, text, mapping, meta=meta)


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


def test_opt_in_paragraph_review_preserves_detector_scores_and_binds_both_inputs(packet_factory, monkeypatch):
    from app.core.config import Settings, settings
    from app.detectors.text.adapters.learned import FourClassTextDetector
    from app.engine.boundary import Deadline
    from app.evidence.document_context import validate

    body = "\n\n".join((
        "This opening paragraph has enough source words to describe the project and introduce the current report to local readers.",
        "The review compares the supplied paragraph with its original published context so local readers can inspect the exact supporting input.",
        "A second passage describes the relevant material and illustrates how paragraph boundaries preserve original source coordinates during document review.",
        "The closing text gives the reviewer enough detail to understand the result and to check its limitations in context.",
    ))
    db, store, analysis, result, decoded, _ = packet_factory(".txt", scores=(0.1, 0.9, 0.2, 0.1), text=body)

    class Classifier:
        def __init__(self):
            self.calls = []
            self.incomplete = False

        def tok(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split()) + 2))}

        def predict(self, inputs, **kwargs):
            self.calls.append(inputs)
            self.last_coverage = [{"tokens": 18, "covered_tokens": 18, "token_windows": 1} for _ in inputs]
            if self.incomplete and len(inputs) == 1 and kwargs.get("batch") == 2:
                self.last_coverage[0].update(covered_tokens=17, omitted_tokens=1)
            return [0.1, 0.9, 0.2, 0.1] if len(inputs) == 4 else [0.81]

    classifier = Classifier()
    monkeypatch.setattr(FourClassTextDetector, "backend", lambda self, ctx: classifier)
    context = SimpleNamespace(outputs=SimpleNamespace(document=decoded), deadline=Deadline(),
                              lease={FourClassTextDetector.MODEL_ID: SimpleNamespace(revision="test")})
    assert Settings.model_fields["text_isolated_paragraph_review"].default is False
    monkeypatch.setattr(settings, "text_isolated_paragraph_review", False)
    ordinary = FourClassTextDetector().run(context)
    assert len(classifier.calls) == 2 and "isolated_paragraph_review" not in ordinary.details["text_evidence"]
    monkeypatch.setattr(settings, "text_isolated_paragraph_review", True)
    enabled = FourClassTextDetector().run(context)
    assert len(classifier.calls) == 5 and classifier.calls[-1] == [body.split("\n\n")[1]]
    assert ordinary.unit_scores == enabled.unit_scores and ordinary.aggregate == enabled.aggregate
    result["signals"][0]["details"] = enabled.details
    analysis.result_hash = canonical_hash(result)
    evidence = build_packet_set(db, analysis, store, result)
    packet, = evidence["packets"]
    comparison = packet["context_comparison"]
    isolated = comparison["isolated_paragraph"]
    assert isolated["supports_inspection"] is True and comparison["affects_overall_confidence"] is False
    assert isolated["paragraph"]["text"] == classifier.calls[-1][0]
    assert isolated["context"]["text"] == classifier.calls[2][1]
    assert isolated["paragraph"]["spans"] == [{"start": decoded["paragraphs"][1][0], "end": decoded["paragraphs"][1][1]}]
    validate(comparison, focus=packet["focus"], chars=len(decoded["text"]), text=decoded["text"])
    for member in ("context", "paragraph"):
        for mutation in ("range", "hash"):
            forged = deepcopy(comparison)
            item = forged["isolated_paragraph"][member]
            if mutation == "range":
                item["spans"][0]["start"] += 1
            else:
                item["input_sha256"] = "0" * 64
            with pytest.raises(ValueError):
                validate(forged, focus=packet["focus"], chars=len(decoded["text"]), text=decoded["text"])
    forged = deepcopy(comparison)
    forged["isolated_paragraph"]["paragraph"]["token_coverage"]["omitted_tokens"] = 1
    with pytest.raises(ValueError, match="complete token coverage"):
        validate(forged, focus=packet["focus"], chars=len(decoded["text"]), text=decoded["text"])
    classifier.incomplete = True
    uncovered = FourClassTextDetector().run(context)
    assert "isolated_paragraph_review" not in uncovered.details["text_evidence"]
    assert uncovered.unit_scores == ordinary.unit_scores and uncovered.aggregate == ordinary.aggregate


def test_email_quote_can_be_in_forward_input_outside_highlighted_ratio(packet_factory):
    body = "First sentence contains enough useful words. Second sentence adds more context.\n\n> This quotation completes the source."
    db, store, analysis, result, decoded, _ = packet_factory(".eml", scores=(0.1, 0.2, 0.3), text=body)
    _add_ratio(result, decoded, score=0.8)
    analysis.result_hash = canonical_hash(result)
    packet = build_packet_set(db, analysis, store, result)["packets"][1]
    assert packet["inspection_scope"]["basis"] == "ratio_reduction"
    assert {anchor["quote_role"] for anchor in packet["anchors"]} == {"body"}
    ratio = next(window for window in packet["supporting_windows"] if window["score_kind"] == "perplexity_ratio")
    assert "> This quotation" in ratio["text"]
    assert "A recorded check read material marked as an email quote or signature outside the highlighted range. That surrounding text can influence the response." in packet["limitations"]
    assert packet["focus"]["text"] == "Second sentence adds more context."


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
    assert "Opposing responses: 1" in packet["observation"]
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
    assert all(packet["inspection_scope"] == {
        "basis": "ratio_reduction", "spans": [{"start": packet["focus"]["start"], "end": packet["focus"]["end"]}],
    } for packet in evidence["packets"])


def test_a_rehashed_narrower_classifier_highlight_is_rejected(packet_factory):
    db, store, analysis, result, _, _ = packet_factory()
    document = build_packet_set(db, analysis, store, result)
    packet, = document["packets"]
    packet["inspection_scope"]["spans"] = [
        {"start": packet["focus"]["start"], "end": packet["focus"]["end"]},
    ]
    document["sha256"] = packet_hash(document)
    with pytest.raises(DocumentPacketIntegrityError, match="invalid inspection scope"):
        validate_packet_set(document, analysis_id=analysis.id, revision=analysis.revision,
                            original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)


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
