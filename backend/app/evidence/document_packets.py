"""Immutable text evidence bound to extraction, actual scored contexts and source pages.

Reading never performs extraction, scoring or rendering. Historical results without
source commitments need a new analysis; their old offsets are not silently reinterpreted.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
import uuid
from collections.abc import Callable
from typing import Any

from PIL import Image, ImageDraw
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.public_projection import alias_signal
from app.core.canonical import canonical_hash, canonical_json
from app.detectors.text.extract import SOURCE_MAP_VERSION, text_hash
from app.evidence import (
    document_matched_context,
    document_paragraph_priority,
    document_passage_coverage,
    document_passage_position,
    document_repeat,
    document_scope,
    document_unselected_responses,
    email_alternatives,
)
from app.storage.artefacts import get_artefact, register_artefact
from app.storage.db import Analysis, AnalysisStage, Artefact
from app.storage.objects import ObjectIntegrityError, ObjectStore

SCHEMA = "chipmunk.document-evidence/1"
VERSION = document_scope.VERSION
CREATOR = "document-evidence-packets/1"
MAX_PACKETS = 6
MAX_PAGES_PER_PACKET = 3
_NS = uuid.UUID("e17ab568-436e-43a8-bc69-939c3a8007a6")
_SHA = re.compile(r"[a-f0-9]{64}")


class DocumentPacketIntegrityError(ObjectIntegrityError):
    pass


def _verified(store: ObjectStore, sha256: str) -> bytes:
    content = store.read_bytes(sha256)
    if hashlib.sha256(content).hexdigest() != sha256:
        raise DocumentPacketIntegrityError("Document evidence bytes no longer match their hash")
    return content


def packet_hash(document: dict[str, Any]) -> str:
    return canonical_hash({key: value for key, value in document.items() if key != "sha256"})


def current_set_id(analysis: Analysis) -> str:
    return str(uuid.uuid5(_NS, f"{analysis.id}:{analysis.result_hash}:{VERSION}"))


def _range(span: dict, size: int) -> tuple[int, int]:
    a, b = span.get("start"), span.get("end")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (a, b)) or not 0 <= a < b <= size:
        raise DocumentPacketIntegrityError("Text evidence has an invalid codepoint range")
    return a, b


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _raw_value(window: dict, key: str, default: float | None = None) -> float:
    value = window.get(key, default)
    exact = window.get(key + "_exact_hex")
    if exact is None:
        return value
    try:
        raw = float.fromhex(exact)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DocumentPacketIntegrityError("Invalid exact text-response value") from exc
    if not math.isfinite(raw) or not _number(value) or abs(raw - value) > 0.00000051:
        raise DocumentPacketIntegrityError("Exact text-response value differs from its display value")
    return raw


def verify_context_window(window: dict, text: str) -> None:
    """Verify frozen source mapping and token-input commitments without inference.

    Historical classifier windows keep their exact sentence-space join. Ratio
    windows additionally commit the actual paired forward sequences, including
    anchor IDs, rolling context boundaries and which tokens entered the ratio.
    """
    spans = [_range(span, len(text)) for span in window["spans"]]
    if not spans or any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise DocumentPacketIntegrityError("A scored text context has overlapping ranges")
    actual = " ".join(text[a:b] for a, b in spans)
    if text_hash(actual) != window.get("input_sha256") or ("text" in window and window["text"] != actual):
        raise DocumentPacketIntegrityError("Recorded input context does not match the extracted text")
    score, kind = _raw_value(window, "score"), window.get("score_kind", "classifier_probability")
    if not _number(score) or score < 0 or (kind == "classifier_probability" and score > 1):
        raise DocumentPacketIntegrityError("The text check has an invalid local score")
    if kind not in {"classifier_probability", "perplexity_ratio"}:
        raise DocumentPacketIntegrityError("Unknown text evidence score kind")
    threshold = _raw_value(window, "threshold", 0.5)
    ai_when = window.get("ai_when", "gte")
    if not _number(threshold) or threshold <= 0 or ai_when != ("lt" if kind == "perplexity_ratio" else "gte"):
        raise DocumentPacketIntegrityError("The text check has an invalid operating boundary")
    direction = "supports_ai" if (score < threshold if ai_when == "lt" else score >= threshold) else "opposes_ai"
    if window.get("direction", direction) != direction or window.get("is_calibrated", False) is not False:
        raise DocumentPacketIntegrityError("A text response misstates its threshold direction or calibration")
    if kind != "perplexity_ratio":
        return
    contexts = window.get("inference_contexts") or []
    if not contexts or len(contexts) > 16 or not window.get("shared_tokenization_verified"):
        raise DocumentPacketIntegrityError("The raw-ratio check lacks bounded verified paired input contexts")
    ids, selected = set(), {}
    for context in contexts:
        if context.get("id") in ids:
            raise DocumentPacketIntegrityError("Duplicate paired input context")
        ids.add(context.get("id"))
        cs = [_range(s, len(text)) for s in context.get("spans", [])]
        if len(cs) != 1:
            raise DocumentPacketIntegrityError("A paired token input has no exact source envelope")
        start, end = context.get("source_token_range", [None, None])
        low, high = context.get("scored_token_range", [None, None])
        anchor = context.get("anchor_token_count")
        token_ids, offsets = context.get("input_token_ids", []), context.get("token_offsets", [])
        limit, stride = context.get("context_limit"), context.get("stride")
        if (any(type(x) is not int for x in (start, end, low, high, anchor, limit, stride))
                or not 0 <= start <= low < high <= end or anchor not in {0, 1}
                or not 1 <= stride <= limit <= 1024 or len(token_ids) > limit
                or len(token_ids) != end - start + anchor or len(offsets) != end - start
                or any(type(x) is not int or x < 0 for x in token_ids)
                or canonical_hash(token_ids) != context.get("token_ids_sha256")):
            raise DocumentPacketIntegrityError("The paired token sequence or its coverage commitment is inconsistent")
        if "shared_input_anchor_id" in window:
            chosen = window["shared_input_anchor_id"]
            defaults = window.get("paired_tokenizer_default_eos_ids")
            if (not isinstance(defaults, list) or len(defaults) != 2 or defaults[0] != chosen
                    or any(value is not None and (type(value) is not int or value < 0) for value in defaults)
                    or (anchor == 1 and token_ids[0] != chosen) or (anchor == 0 and chosen is not None)):
                raise DocumentPacketIntegrityError("The shared input anchor differs from its recorded provenance")
        if any(len(o) != 2 or any(type(v) is not int for v in o) or not 0 <= o[0] <= o[1] <= len(text) for o in offsets):
            raise DocumentPacketIntegrityError("The paired token offsets are not source codepoint ranges")
        mapped = [(a, b) for a, b in offsets if a < b]
        if not mapped or cs[0] != (min(a for a, _ in mapped), max(b for _, b in mapped)):
            raise DocumentPacketIntegrityError("The token input envelope does not match its token offsets")
        actual_context = text[cs[0][0]:cs[0][1]]
        if context.get("text") != actual_context or context.get("input_sha256") != text_hash(actual_context):
            raise DocumentPacketIntegrityError("The paired token source excerpt differs from the frozen source")
        for index in range(low, high):
            if index in selected:
                raise DocumentPacketIntegrityError("Rolling contexts score the same token twice")
            selected[index] = offsets[index - start]
    if spans != [(min(c["spans"][0]["start"] for c in contexts), max(c["spans"][0]["end"] for c in contexts))]:
        raise DocumentPacketIntegrityError("The ratio context is not the full recorded input envelope")
    scored_spans = [_range(s, len(text)) for s in window.get("scored_spans", [])]
    expected = sorted(index for index, (a, b) in selected.items() if any(b > lo and a < hi for lo, hi in scored_spans))
    if (len(scored_spans) != 1 or expected != window.get("scored_token_indices") or len(expected) < 12
            or window.get("tokens_scored") != len(expected)):
        raise DocumentPacketIntegrityError("The raw ratio's scored tokens do not match its reduction span")
    numerator, denominator = _raw_value(window, "mean_negative_log_likelihood"), _raw_value(window, "mean_cross_entropy")
    if (not _number(numerator) or numerator < 0 or not _number(denominator) or denominator <= 0
            or not math.isclose(score, numerator / denominator, rel_tol=2e-5, abs_tol=2e-5)):
        raise DocumentPacketIntegrityError("The recorded raw ratio differs from its loss means")


def _decode(db: Session, analysis: Analysis, store: ObjectStore, result: dict) -> dict:
    binding = result.get("manifest", {}).get("sampling", {}).get("document_source") or {}
    if binding.get("version") != SOURCE_MAP_VERSION or binding.get("source_sha256") != analysis.sha256:
        raise DocumentPacketIntegrityError("This analysis does not contain a verified document source mapping")
    row = db.get(AnalysisStage, (analysis.id, "decode"))
    if row is None or row.status != "done" or not row.output_artefact_id:
        raise DocumentPacketIntegrityError("The frozen document extraction is unavailable")
    artifact = get_artefact(db, row.output_artefact_id, analysis_id=analysis.id)
    if artifact.sha256 != binding.get("decode_sha256"):
        raise DocumentPacketIntegrityError("The extraction no longer matches the published document result")
    decoded = json.loads(_verified(store, artifact.sha256))
    if (text_hash(decoded["text"]) != binding.get("extracted_text_sha256")
            or canonical_hash(decoded.get("source_map", [])) != binding.get("source_map_sha256")
            or decoded.get("meta", {}).get("original_sha256") != analysis.sha256):
        raise DocumentPacketIntegrityError("The text or its source mapping no longer matches the published result")
    return decoded


def _source_anchors(mapping: list[dict], a: int, b: int, size: int) -> list[dict]:
    anchors = []
    for source in mapping:
        start, end = _range(source, size)
        if start >= b or end <= a:
            continue
        anchor = copy.deepcopy(source)
        anchor["selected_start"], anchor["selected_end"] = max(start, a), min(end, b)
        if source.get("kind") == "pdf":
            anchor["highlight_boxes"] = [f["bbox"] for f in source.get("fragments", [])
                                         if _range(f, size)[0] < b and f["end"] > a]
        # The packet's exact excerpt is already bound to the full frozen mapping.
        if source.get("extraction") == "ocr":
            anchor["highlighted_words"] = [copy.deepcopy(f) for f in source.get("fragments", [])
                                           if _range(f, size)[0] < b and f["end"] > a]
        # Do not repeat all unrelated words or decoded source ranges in the export.
        anchor.pop("fragments", None)
        anchors.append(anchor)
    return anchors


def _inspection_anchors(mapping: list[dict], scope: dict, size: int) -> list[dict]:
    return [anchor for span in scope["spans"]
            for anchor in _source_anchors(mapping, span["start"], span["end"], size)]


def _rect(box: list, width: float, height: float) -> tuple[float, float, float, float]:
    if len(box) != 4 or any(not isinstance(v, int | float) or not math.isfinite(v) for v in box):
        raise DocumentPacketIntegrityError("A PDF source box is invalid")
    x0, y0, x1, y1 = box
    if not 0 <= x0 < x1 <= width + 0.01 or not 0 <= y0 < y1 <= height + 0.01:
        raise DocumentPacketIntegrityError("A PDF source box is outside its page")
    return x0, y0, x1, y1


def _png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    clean = Image.new("RGB", image.size)
    clean.paste(image.convert("RGB"))
    clean.save(output, format="PNG")
    return output.getvalue()


def _pdf_plates(
    source: bytes, anchors: list[dict], store: ObjectStore | None = None, *,
    read_ocr_raster: Callable[[str], bytes] | None = None,
) -> dict[str, tuple[Image.Image, dict]]:
    import pymupdf

    grouped: dict[int, list[dict]] = {}
    for anchor in anchors:
        if anchor.get("kind") == "pdf" and anchor.get("highlight_boxes"):
            grouped.setdefault(anchor["page"], []).append(anchor)
    plates = {}
    with pymupdf.open(stream=source, filetype="pdf") as pdf:
        for number in sorted(grouped)[:MAX_PAGES_PER_PACKET]:
            if not 1 <= number <= len(pdf):
                raise DocumentPacketIntegrityError("A PDF source page is missing")
            page = pdf[number - 1]
            saved_rotation = page.rotation
            # Text geometry uses the unrotated page space. Render that same space
            # and record the transform; never search text and pick a repeated match.
            page.set_rotation(0)
            width, height = page.rect.width, page.rect.height
            boxes = []
            for anchor in grouped[number]:
                if abs(anchor["page_width"] - width) > 0.01 or abs(anchor["page_height"] - height) > 0.01 or anchor["page_rotation"] != saved_rotation:
                    raise DocumentPacketIntegrityError("The source PDF page geometry changed")
                boxes.extend(_rect(box, width, height) for box in anchor["highlight_boxes"])
            bounding = (max(0, min(b[0] for b in boxes) - 18), max(0, min(b[1] for b in boxes) - 18),
                        min(width, max(b[2] for b in boxes) + 18), min(height, max(b[3] for b in boxes) + 18))
            rasters = [anchor.get("ocr", {}).get("raster") for anchor in grouped[number]
                       if anchor.get("extraction") == "ocr" or anchor.get("ocr")]
            frozen = rasters[0] if rasters else None
            if rasters:
                if frozen is None or any(raster != frozen for raster in rasters):
                    raise DocumentPacketIntegrityError("An OCR page has inconsistent frozen input rasters")
                if read_ocr_raster is not None:
                    raster_bytes = read_ocr_raster(frozen["sha256"])
                elif store is not None:
                    raster_bytes = _verified(store, frozen["sha256"])
                else:
                    raise DocumentPacketIntegrityError("An OCR page requires its exact input raster")
                with Image.open(io.BytesIO(raster_bytes)) as opened:
                    engine_input = opened.convert("RGB")
                if engine_input.size != (frozen["width"], frozen["height"]):
                    raise DocumentPacketIntegrityError("The frozen OCR raster dimensions changed")
                original = engine_input.rotate(frozen["rotation_clockwise"], expand=True)
                if original.size != (frozen["base_width"], frozen["base_height"]):
                    raise DocumentPacketIntegrityError("The OCR input rotation changed")
                zoom, zoom_y = original.width / width, original.height / height
                plates[f"page{number}_ocr_input"] = (engine_input, {"page": number, "role": "ocr_input",
                     "source_raster_sha256": frozen["sha256"], "input_to_page": frozen["input_to_page"],
                     "confidence_semantics": "OCR recognition quality, not AI-generation confidence", "_exact_png": raster_bytes})
            else:
                zoom = zoom_y = min(2.0, 1200 / max(width, height))
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                original = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            x0, y0, x1, y1 = bounding
            pixel_box = (max(0, math.floor(x0 * zoom)), max(0, math.floor(y0 * zoom_y)),
                         min(original.width, math.ceil(x1 * zoom)), min(original.height, math.ceil(y1 * zoom_y)))
            crop = original.crop(pixel_box)
            overlay = crop.copy()
            draw = ImageDraw.Draw(overlay, "RGBA")
            for bx0, by0, bx1, by1 in boxes:
                draw.rectangle((bx0 * zoom - pixel_box[0], by0 * zoom_y - pixel_box[1],
                                bx1 * zoom - pixel_box[0], by1 * zoom_y - pixel_box[1]),
                               fill=(245, 167, 49, 55), outline=(160, 92, 12, 220), width=1)
            context = original.copy()
            ImageDraw.Draw(context).rectangle(pixel_box, outline=(210, 130, 20), width=2)
            metadata = {"page": number, "source_bbox": list(bounding), "render_scale": zoom,
                        "page_rotation_removed": saved_rotation, "renderer": f"pymupdf-{pymupdf.VersionBind}",
                        "source_crop_pixels": list(pixel_box), "source_raster_dimensions": list(original.size),
                        "source_crop_coordinate_space": "unrotated-page-raster-pixels"}
            if frozen:
                metadata.update(source_raster_sha256=frozen["sha256"], render_scale_y=zoom_y,
                                extraction="ocr", colorspace=frozen["colorspace"])
            for role, image in (("context", context), ("crop", crop), ("overlay", overlay)):
                plates[f"page{number}_{role}"] = (image, {**metadata, "role": role})
    return plates


def validate_packet_set(document: dict[str, Any], *, analysis_id: str, revision: int,
                        original_result_hash: str, source_sha256: str) -> None:
    if (document.get("schema") != SCHEMA or document.get("version") not in document_scope.VERSIONS
            or document.get("status") != "ready" or document.get("analysis_id") != analysis_id
            or document.get("revision") != revision or document.get("original_result_hash") != original_result_hash
            or document.get("source", {}).get("sha256") != source_sha256 or packet_hash(document) != document.get("sha256")):
        raise DocumentPacketIntegrityError("Document evidence does not match its source, result or hash")
    size = document["source"].get("chars")
    if not isinstance(size, int) or size < 0 or document["source"].get("offset_unit") != "unicode_codepoint":
        raise DocumentPacketIntegrityError("Document evidence has no valid text coordinate space")
    try:
        document_repeat.validate(document)
        document_matched_context.validate(document)
        document_paragraph_priority.validate(document)
        document_passage_coverage.validate(document)
        document_passage_position.validate(document)
        document_unselected_responses.validate(document)
        document_scope.validate_checkpoint_agreement(document)
        if email_alternatives.KEY in document.get("coverage", {}):
            email_alternatives.validate_set(document, canonical_hash(document["coverage"][email_alternatives.KEY]))
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise DocumentPacketIntegrityError("Document review evidence has invalid source commitments or positions") from error
    ids = set()
    for packet in document.get("packets", []):
        if packet.get("id") in ids or packet.get("evidence_type") != "statistical":
            raise DocumentPacketIntegrityError("Ambiguous document evidence")
        ids.add(packet["id"])
        a, b = _range(packet["focus"], size)
        ca, cb = _range(packet["context"], size)
        if not ca <= a < b <= cb or len(packet["focus"]["text"]) != b - a or len(packet["context"]["text"]) != cb - ca:
            raise DocumentPacketIntegrityError("A document excerpt is inconsistent with its recorded scope")
        if packet["context"]["text"][a - ca:b - ca] != packet["focus"]["text"]:
            raise DocumentPacketIntegrityError("The highlighted text is not within its context")
        for window in packet["supporting_windows"]:
            if text_hash(window["text"]) != window["input_sha256"]:
                raise DocumentPacketIntegrityError("A scored context does not match its actual input")
        if document["version"] in document_scope.SCOPE_VERSIONS:
            try:
                document_scope.validate(packet, size)
            except (ValueError, KeyError, TypeError) as error:
                raise DocumentPacketIntegrityError("A document packet has an invalid inspection scope") from error
        if document["version"] in document_scope.CONTEXT_VERSIONS or "context_comparison" in packet:
            from app.evidence.document_context import validate

            try:
                validate(packet["context_comparison"], focus=packet["focus"], chars=size)
            except (ValueError, KeyError, TypeError) as error:
                raise DocumentPacketIntegrityError("A document context comparison has invalid scope or measurements") from error
        for artifact in packet.get("artifacts", {}).values():
            if not _SHA.fullmatch(str(artifact.get("sha256", ""))) or artifact.get("media_type") != "image/png":
                raise DocumentPacketIntegrityError("Invalid document evidence image")


def public_ocr_coverage(meta: dict) -> dict | None:
    coverage = copy.deepcopy(meta.get("ocr"))
    for page in (coverage or {}).get("pages", []):
        for private_key in ("engine_version", "language_data_sha256", "languages_requested"):
            page.pop(private_key, None)
    return coverage


def build_packet_set(db: Session, analysis: Analysis, store: ObjectStore, result: dict[str, Any]) -> dict[str, Any]:
    if analysis.kind != "document" or canonical_hash(result) != analysis.result_hash or result.get("input", {}).get("sha256") != analysis.sha256:
        raise DocumentPacketIntegrityError("Document evidence requires an intact published document result")
    source = _verified(store, analysis.sha256)
    decoded = _decode(db, analysis, store, result)
    text, mapping = decoded["text"], decoded["source_map"]
    binding = result["manifest"]["sampling"]["document_source"]
    public_ocr = public_ocr_coverage(decoded["meta"])
    groups: dict[tuple[int, int], list[dict]] = {}
    scored = 0
    check_coverage = []
    seen_checks = set()
    matched_review = None
    checkpoint_review = None
    for signal in result.get("signals", []):
        evidence = signal.get("details", {}).get("text_evidence") or {}
        if signal.get("status") != "ok":
            continue
        # The fused sentence consensus can reuse the style check's identity. It
        # is not another independent source of measured local input evidence.
        check_id = (signal.get("detector_id", signal["id"]), canonical_hash(evidence))
        if check_id in seen_checks:
            continue
        seen_checks.add(check_id)
        ref = {"id": alias_signal(signal["id"]), "display_id": signal.get("display_id")}
        if evidence.get("version") not in {"scored-context/1", "scored-context/2"}:
            check_coverage.append({"signal_ref": ref, "status": "context_not_recorded", "scored_windows": 0,
                                   "limitation": "This check has document-level or heuristic measurements without a frozen local inference context. Its aggregate response is not word-level authorship evidence; absence of a local flag is not opposing evidence."})
            continue
        if evidence.get("extracted_text_sha256") != text_hash(text) or evidence.get("offset_unit") != "unicode_codepoint":
            raise DocumentPacketIntegrityError("A text check refers to a different extracted document")
        if document_scope.CHECKPOINT_AGREEMENT_KEY in evidence:
            if checkpoint_review is not None or not document_matched_context.matches(
                signal["detector_id"], signal["detector_version"], signal["model_revisions"]
            ):
                raise DocumentPacketIntegrityError("Second writing check has an invalid primary binding")
            checkpoint_review = evidence[document_scope.CHECKPOINT_AGREEMENT_KEY]
        try:
            review = document_matched_context.from_signal(signal, ref)
        except (ValueError, KeyError, TypeError) as error:
            raise DocumentPacketIntegrityError("Matched writing review has an invalid check binding") from error
        if review is not None:
            if matched_review is not None:
                raise DocumentPacketIntegrityError("Matched writing review refers to more than one check")
            matched_review = review
        context_lookup = {c["id"]: c for c in evidence.get("contexts", [])}
        if len(context_lookup) != len(evidence.get("contexts", [])):
            raise DocumentPacketIntegrityError("The paired text check repeated a context identity")
        check_coverage.append({"signal_ref": ref, "status": "partial" if evidence.get("omitted_windows") else "recorded",
                               "scored_windows": len(evidence.get("windows", [])),
                               "omitted_windows": evidence.get("omitted_windows", []),
                               "unscored_first_token": evidence.get("unscored_first_token", False)})
        for window in evidence.get("windows", []):
            a, b = _range(window["focus"], len(text))
            kind = window.get("score_kind", "classifier_probability")
            output_window = {"signal_ref": ref, "spans": window["spans"], "input_sha256": window["input_sha256"],
                             "score": window["score"], "score_kind": kind, "is_calibrated": False,
                             "threshold": window.get("threshold", 0.5), "ai_when": window.get("ai_when", "gte"),
                             "token_coverage": window.get("token_coverage", {})}
            if kind == "perplexity_ratio":
                try:
                    output_window["inference_contexts"] = [copy.deepcopy(context_lookup[key]) for key in window["context_ids"]]
                except KeyError as exc:
                    raise DocumentPacketIntegrityError("A raw-ratio response refers to a missing actual input") from exc
                for field in ("scored_spans", "scored_token_indices", "tokens_scored", "mean_negative_log_likelihood",
                              "mean_cross_entropy", "neighbor_expansion", "input_semantics"):
                    output_window[field] = copy.deepcopy(window[field])
                for field in ("score_exact_hex", "threshold_exact_hex", "mean_negative_log_likelihood_exact_hex", "mean_cross_entropy_exact_hex"):
                    if field in window:
                        output_window[field] = window[field]
                for field in ("shared_input_anchor_id", "paired_tokenizer_default_eos_ids", "anchor_policy"):
                    if field in window:
                        output_window[field] = copy.deepcopy(window[field])
                output_window["shared_tokenization_verified"] = evidence.get("shared_tokenization_verified", False)
            if review is not None:
                try:
                    document_matched_context.apply_to_window(output_window, window["focus"], review)
                except (ValueError, KeyError, TypeError) as error:
                    raise DocumentPacketIntegrityError("Matched writing review differs from the frozen input") from error
            verify_context_window(output_window, text)
            spans = [_range(span, len(text)) for span in window["spans"]]
            if ((kind == "classifier_probability" and (a, b) not in spans)
                    or not min(x for x, _ in spans) <= a < b <= max(y for _, y in spans)):
                raise DocumentPacketIntegrityError("The focus is not part of the actual scored context")
            if kind == "perplexity_ratio" and not any(s["start"] <= a < b <= s["end"] for s in window["scored_spans"]):
                raise DocumentPacketIntegrityError("The focus is not within the ratio reduction span")
            output_window["text"] = " ".join(text[x:y] for x, y in spans)
            raw, boundary = _raw_value(output_window, "score"), _raw_value(output_window, "threshold")
            supports = raw < boundary if output_window["ai_when"] == "lt" else raw >= boundary
            output_window["direction"] = "supports_ai" if supports else "opposes_ai"
            output_window["direction_semantics"] = "Side of this check's raw operating threshold only; not validated proof of authorship."
            scored += 1
            groups.setdefault((a, b), []).append(output_window)
    candidates = [key for key, values in groups.items() if any(document_scope.supports_inspection(w) for w in values)]
    matched = sorted(key for key in candidates if document_matched_context.packet_record(groups[key]) is not None)
    reply_review = bool(checkpoint_review and checkpoint_review["schema"] in document_scope.CHECKPOINT_AGREEMENT_EMAIL_PARAGRAPHS)
    mixed_pdf_review = bool(checkpoint_review and checkpoint_review["schema"] == document_scope.CHECKPOINT_AGREEMENT_PDF_SCHEMA)
    priority = (sorted({(row["start"], row["end"]) for row in checkpoint_review["ranges"]}.intersection(candidates))
                if reply_review or mixed_pdf_review else matched)
    # Ratios and classifier outputs are not commensurate. Reserve half the
    # bounded display for disagreement so selecting corroborated passages does
    # not systematically hide real opposing observations elsewhere in the file.
    remaining = set(candidates).difference(priority)
    conflicts = sorted(key for key in remaining if any(document_scope.inspection_direction(w) is False for w in groups[key]))
    agreed = sorted(remaining.difference(conflicts))
    reserved = min(MAX_PACKETS // 2, MAX_PACKETS - len(priority))
    selected = priority + conflicts[:reserved]
    selected += agreed[:MAX_PACKETS - len(selected)]
    selected += conflicts[reserved:reserved + MAX_PACKETS - len(selected)]
    selected.sort()
    opposing_total = sum(document_scope.inspection_direction(w) is False for values in groups.values() for w in values)
    opposing_shown = sum(document_scope.inspection_direction(w) is False for key in selected for w in groups[key])
    document = {"schema": SCHEMA, "status": "ready", "id": current_set_id(analysis),
                "version": document_scope.MATCHED_VERSION if matched_review is not None else VERSION,
                "analysis_id": analysis.id, "revision": analysis.revision, "original_result_hash": analysis.result_hash,
                "source": {"sha256": analysis.sha256, "extracted_text_sha256": text_hash(text), "chars": len(text),
                           "method": decoded["meta"].get("method"), "offset_unit": "unicode_codepoint",
                           "source_map_sha256": binding["source_map_sha256"], "decode_sha256": binding["decode_sha256"]},
                "packets": [], "coverage": {"scored_windows": scored, "candidate_spans": len(candidates),
                                              "checks": check_coverage,
                                              "opposing_windows_recorded": opposing_total,
                                              "opposing_windows_displayed": opposing_shown,
                                              "opposing_windows_omitted": opposing_total - opposing_shown,
                                              "selection_policy": "Up to half the displayed passages are reserved for disagreement; remaining slots use other candidate passages, then any remaining disagreements. Each group follows source order; incomparable score kinds are not pooled.",
                                              "unmapped_pages": decoded["meta"].get("unmapped_pages", []),
                                              "ocr": public_ocr, "email": decoded["meta"].get("email")},
                "omitted_windows": max(0, len(candidates) - len(selected)), "limitations": []}
    if partially_checked := decoded["meta"].get("partially_checked_pages"):
        document["coverage"]["partially_checked_pages"] = partially_checked
    if alternative_review := email_alternatives.result_review(result):
        document["coverage"][email_alternatives.KEY] = alternative_review
        email_alternatives.validate_set(document, canonical_hash(alternative_review))
        email_alternatives.verify_source(document, source)
    if checkpoint_review is not None:
        document["coverage"][document_scope.CHECKPOINT_AGREEMENT_KEY] = checkpoint_review
        document_scope.validate_checkpoint_agreement(document, text, result=result, mapping=mapping)
        if reply_review or mixed_pdf_review:
            document["coverage"]["selection_policy"] = (
                "Passages revisited by the second writing check are prioritized for inclusion. Remaining passages use the standard "
                "disagreement and source-order policy."
            ) if mixed_pdf_review else (
                "Passages revisited by the second reply-body check are prioritized for inclusion. Remaining passages use the standard "
                "disagreement and source-order policy."
            )
    if matched_review is not None:
        document["coverage"][document_matched_context.KEY] = matched_review
        document["coverage"]["selection_policy"] = (
            "Sentences selected by the short-passage review are prioritized for inclusion. Other passages follow the standard "
            "disagreement and source-order policy. Each check keeps its own recorded response."
        )
    try:
        meta = decoded["meta"]
        pages = meta.get("pages")
        pdf_meta = meta if meta.get("method") == "pymupdf" and type(pages) is int and pages > 0 else None
        document["coverage"][document_passage_coverage.KEY] = document_passage_coverage.build(
            result, text, mapping, pdf_meta=pdf_meta)
    except (ValueError, KeyError, TypeError) as error:
        raise DocumentPacketIntegrityError("Local passage coverage differs from the frozen inputs") from error
    document["repeated_passages"] = document_repeat.build(document["source"], text, mapping)
    quoted_roles = {"quoted", "signature"}
    quoted_ranges = [_range(source, len(text)) for source in mapping if source.get("quote_role") in quoted_roles]
    for index, (a, b) in enumerate(selected, 1):
        windows = groups[a, b]
        ca, cb = min(s["start"] for w in windows for s in w["spans"]), max(s["end"] for w in windows for s in w["spans"])
        focus = {"start": a, "end": b, "text": text[a:b]}
        scope = document_scope.build(focus, windows, len(text))
        ratio_scope = scope["basis"] == "ratio_reduction"
        anchors = _inspection_anchors(mapping, scope, len(text))
        refs = {w["signal_ref"]["id"] for w in windows}
        original_refs = {s["id"] for s in result["signals"] if alias_signal(s["id"]) in refs}
        findings = [f["id"] for f in result.get("findings", []) if original_refs.intersection(f.get("evidence_ids", []))]
        artifacts = {}
        for key, (image, metadata) in _pdf_plates(source, anchors, store).items() if decoded["meta"].get("method") == "pymupdf" else []:
            payload = metadata.pop("_exact_png", None) or _png(image)
            art = register_artefact(db, store, payload, media_type="image/png", role="preview", created_by=CREATOR,
                                   analysis_id=analysis.id)
            artifacts[key] = {"sha256": art.sha256, "media_type": "image/png", "width": image.width, "height": image.height,
                              "url": f"/api/v2/analyses/{analysis.id}/evidence-packets/{document['id']}/assets/{art.sha256}.png",
                              **metadata}
        supports = sum(document_scope.supports_inspection(w) for w in windows)
        opposing = sum(document_scope.inspection_direction(w) is False for w in windows)
        limits = ["The recorded responses are inspection clues. They do not establish whether a person used AI.",
                  "Responses are uncalibrated. Templates, translation, editing and unfamiliar languages can trigger them.",
                  "Each check uses its own recorded operating threshold. Threshold direction is not a validated probability or proof of authorship."]
        if any(w["score_kind"] == "perplexity_ratio" for w in windows):
            limits.append("The raw language-pattern ratio is lower on its AI-like side and can exceed 1. It is not a probability. Its sentence reduction uses tokens predicted within the recorded rolling contexts; short sentences may include neighbouring sentences.")
        roles = {anchor.get("quote_role") for anchor in anchors}
        if roles & quoted_roles:
            if "body" in roles:
                limits.append("The highlight includes message text and material marked as an email quote or signature. The response applies to the measured range and cannot identify who wrote either part.")
            else:
                limits.append("The highlight falls in material marked as an email quote or signature. The response does not establish whether the sender wrote it.")
        elif quoted_ranges:
            input_spans = (span for window in windows for context in (window.get("inference_contexts") or [window])
                           for span in context["spans"])
            if any(start < span["end"] and span["start"] < end for span in input_spans for start, end in quoted_ranges):
                limits.append("A recorded check read material marked as an email quote or signature outside the highlighted range. That surrounding text can influence the response.")
        if any(anchor.get("kind") == "pdf" for anchor in anchors):
            limits.append("Page highlights locate intersecting words in the frozen extraction; they are not edits to the source PDF.")
        if any(anchor.get("extraction") == "ocr" for anchor in anchors):
            limits.append("This excerpt was transcribed from the frozen page image. Recognition confidence measures transcription quality, not AI authorship; inspect the source crop for recognition mistakes.")
        interpretation = ("The highlighted range contains the tokens used for a writing-pattern ratio. Surrounding text can influence the response."
                          if ratio_scope else
                          "The highlight shows the measured context. The navigation sentence is a place to start reading, and the response does not locate generated words.")
        review_marker = document_matched_context.packet_record(windows)
        title = "Measured writing-pattern range" if ratio_scope else "Scored context for inspection"
        if review_marker is not None:
            title = "Sentence to review"
            interpretation = ("The selected sentence is a review location. The recorded writing check read it with "
                              "both neighboring sentences. The response does not identify the author or generated words.")
        packet = {"id": f"T{index:02d}", "title": title,
                                    "evidence_type": "statistical",
                                    "finding_ids": findings, "signal_refs": [w["signal_ref"] for w in windows],
                                    "focus": focus, "inspection_scope": scope,
                                    "context": {"start": ca, "end": cb, "text": text[ca:cb]},
                                    "supporting_windows": windows, "anchors": anchors, "artifacts": artifacts,
                                    "observation": f"Recorded responses on the AI-like side of their checks: {supports}. Opposing responses: {opposing}.",
                                    "interpretation": interpretation,
                                    "measurements": [{"label": "Language-pattern ratio" if w["score_kind"] == "perplexity_ratio" else "Context response",
                                                      "value": w["score"], "unit": "raw ratio" if w["score_kind"] == "perplexity_ratio" else "uncalibrated score"} for w in windows],
                                    "limitations": limits}
        if review_marker is not None:
            packet[document_matched_context.KEY] = review_marker
            packet["observation"] = ("The recorded response on this three-sentence context selected its middle "
                                     "sentence for review.")
        document["packets"].append(packet)
    position = document_passage_position.from_result(result)
    if position is not None:
        document["coverage"][document_passage_position.KEY] = position
        document_passage_position.verify_source(document, text, mapping)
    warning = decoded["meta"].get("coverage_warning")
    if warning:
        document["limitations"].append(warning)
    if not scored:
        document["limitations"].append("No successful learned check recorded exact scored contexts for this analysis.")
    elif not document["packets"] and (position is None or position["status"] != "flagged"):
        document["limitations"].append("No recorded context reached the selection boundary. This does not establish human authorship.")
    if any(c["status"] == "context_not_recorded" for c in check_coverage):
        document["limitations"].append("Some successful document-level or heuristic checks do not provide exact local input proof. See check coverage; their aggregate scores are not treated as local authorship evidence.")
    if any(c.get("omitted_windows") for c in check_coverage):
        document["limitations"].append("Some local responses lack enough scored tokens or exceed the bounded input-proof record. These omissions are listed in check coverage and are not negative findings.")
    unselected = document_unselected_responses.build(result, text, mapping, document["packets"])
    if unselected is not None:
        document["coverage"][document_unselected_responses.KEY] = unselected
    if opposing_total > opposing_shown:
        document["limitations"].append(f"{opposing_total - opposing_shown} recorded opposing local response(s) are outside the bounded displayed passages. Their omission is not agreement with an AI-authorship interpretation.")
    document = json.loads(canonical_json(document))
    from app.evidence.document_context import build as compare_contexts

    for packet in document["packets"]:
        packet["context_comparison"] = compare_contexts(result, text, packet["focus"])
    try:
        document_paragraph_priority.apply(document, result, text, mapping)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise DocumentPacketIntegrityError("Paragraph review order has invalid source commitments or positions") from error
    document = json.loads(canonical_json(document))
    for packet in document["packets"]:
        for window in packet["supporting_windows"]:
            verify_context_window(window, text)
    document["sha256"] = packet_hash(document)
    validate_packet_set(document, analysis_id=analysis.id, revision=analysis.revision,
                        original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    if matched_review is not None:
        try:
            document_matched_context.verify_source(document, text, mapping)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise DocumentPacketIntegrityError("Matched writing review differs from the frozen source") from error
    register_artefact(db, store, canonical_json(document), media_type="application/json", role="exhibit_payload",
                     created_by=CREATOR, analysis_id=analysis.id, artefact_id=document["id"])
    return document


def load_packet_set(db: Session, analysis: Analysis, store: ObjectStore, *, set_id: str | None = None) -> dict | None:
    query = select(Artefact).where(Artefact.analysis_id == analysis.id, Artefact.created_by == CREATOR,
                                   Artefact.role == "exhibit_payload", Artefact.media_type == "application/json")
    if set_id is not None:
        query = query.where(Artefact.id == set_id)
    artifact = db.scalars(query.order_by(Artefact.created_at.desc(), Artefact.id)).first()
    if artifact is None:
        return None
    document = json.loads(_verified(store, artifact.sha256))
    if document.get("id") != artifact.id:
        raise DocumentPacketIntegrityError("Document evidence identity differs from its frozen object")
    validate_packet_set(document, analysis_id=analysis.id, revision=analysis.revision,
                        original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    if document["version"] in document_scope.REPEAT_VERSIONS:
        result = json.loads(_verified(store, analysis.result_hash))
        decoded = _decode(db, analysis, store, result)
        try:
            document_repeat.verify_source(document, decoded["text"], decoded["source_map"])
            document_matched_context.verify_source(document, decoded["text"], decoded["source_map"], result=result)
            document_paragraph_priority.verify_source(document, decoded["text"], decoded["source_map"], result=result)
            document_passage_coverage.verify_source(document, decoded["text"], decoded["source_map"],
                                                    result=result, meta=decoded["meta"])
            document_unselected_responses.verify_source(document, decoded["text"], decoded["source_map"], result=result)
            document_passage_position.verify_source(document, decoded["text"], decoded["source_map"], result=result)
            document_scope.validate_checkpoint_agreement(document, decoded["text"], result=result, mapping=decoded["source_map"])
            alternatives = email_alternatives.result_review(result)
            email_alternatives.validate_set(document, canonical_hash(alternatives) if alternatives is not None else None)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise DocumentPacketIntegrityError("Document review evidence differs from the frozen text or source mapping") from error
    store.verify(analysis.sha256)
    return document


def persist_packet_set(db: Session, analysis: Analysis, store: ObjectStore, *, result: dict | None = None) -> dict | None:
    if analysis.kind != "document" or not analysis.result_hash:
        return None
    previous = load_packet_set(db, analysis, store, set_id=current_set_id(analysis))
    if previous is not None:
        return previous
    result = result if result is not None else json.loads(_verified(store, analysis.result_hash))
    if result.get("manifest", {}).get("sampling", {}).get("document_source", {}).get("version") != SOURCE_MAP_VERSION:
        return None
    return build_packet_set(db, analysis, store, result)


def unavailable_packet_set(analysis: Analysis) -> dict:
    return {"schema": SCHEMA, "status": "not_generated" if analysis.kind == "document" else "not_applicable",
            "id": None, "version": VERSION, "sha256": None, "analysis_id": analysis.id, "revision": analysis.revision,
            "original_result_hash": analysis.result_hash, "source": {"sha256": analysis.sha256, "offset_unit": "unicode_codepoint"},
            "packets": [], "coverage": {}, "omitted_windows": 0,
            "limitations": ["Reanalyse the document to record its extraction mapping and the exact contexts that were scored."]}
