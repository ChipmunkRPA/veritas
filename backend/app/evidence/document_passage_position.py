"""Source-bound writing position evidence for exactly three mapped sentences."""

from __future__ import annotations

import math
import re

from app.core.canonical import canonical_hash
from app.evidence import document_long_passage_position, document_matched_context
from app.evidence.document_native_locations import clip_source_ranges
from app.evidence.document_scope import check_range

KEY = "passage_position_review"
COMMITMENT = "writing_position_review_sha256"
SCHEMA = "document-passage-position-review/1"
MAPPED_SCHEMA = "document-passage-position-review/3"
SHORT_SCHEMAS = {SCHEMA, MAPPED_SCHEMA}
POINT = 2.7579683180027486
MAX_CHARS = 1800
LABELS = ("opening", "middle", "closing")
SCOPES = ((0, 1), (0, 1, 2), (1, 2))
SHA256 = re.compile(r"[0-9a-f]{64}")


def geometry(text, mapping, method, sentences, *, locator=None):
    from app.detectors.text.extract import _ocr_language

    if ((method != "plain" and locator is None)
            or not 0 < len(text) <= MAX_CHARS or len(mapping) != 1 or len(sentences) != 3):
        return None
    block = mapping[0]
    if (block.get("kind") != (locator or {}).get("kind", "text")
            or (block.get("start"), block.get("end")) != (0, len(text))
            or "\ufffd" in text or any(unit.paragraph != 0 for unit in sentences)):
        return None
    ranges = block.get("source_ranges") or []
    if locator is None:
        if len(ranges) != 1:
            return None
        start, end, source_start, source_end = ranges[0]
        if (start, end) != (0, len(text)) or not 0 <= source_start < source_end or end - start != source_end - source_start:
            return None
    elif len(clip_source_ranges(ranges, 0, len(text))) != 1:
        return None
    if (not 12 <= sentences[1].n_words <= 65 or any(not 7 <= sentences[i].n_words <= 80 for i in (0, 2))
            or sentences[0].start != 0 or sentences[-1].end != len(text)):
        return None
    language = _ocr_language(text)
    if language.get("code") != "en" or language.get("status") != "supported":
        return None
    rows = []
    for unit in sentences:
        if text[unit.start:unit.end] != unit.text:
            return None
        original = clip_source_ranges(ranges, unit.start, unit.end)
        if len(original) != 1:
            return None
        lo, hi = original[0]
        rows.append({"start": unit.start, "end": unit.end, "source_start": lo, "source_end": hi})
    return rows


def mapped_locator(mapping, meta):
    if len(mapping) != 1:
        return None
    block = mapping[0]
    kind = block.get("kind")
    if meta.get("method") == "python-docx" and kind == "docx":
        path = block["body_path"]
        if (len(path) == 1 and meta.get("paragraph_count") == 1 and meta.get("percentage_body_paragraphs") == path
                and block.get("run_mapping_complete") is True):
            return {"kind": kind, "paragraph_index": block["paragraph_index"], "body_path": path}
    if meta.get("method") == "email" and kind == "email":
        email = meta.get("email") or {}
        if (email.get("body_content_type") == "text/plain" and block.get("mime_part") == email.get("body_part")
                and block.get("quote_role") == "body"):
            return {"kind": kind, "mime_part": block["mime_part"]}
    return None


def _validate_locator(document, locator):
    if not isinstance(locator, dict):
        raise ValueError("Invalid original location for the writing position check")
    coverage = document["coverage"]
    method = document["source"].get("method")
    kind = locator.get("kind")
    if method == "python-docx" and kind == "docx":
        path = locator.get("body_path")
        if (set(locator) == {"kind", "paragraph_index", "body_path"}
                and type(locator["paragraph_index"]) is int and locator["paragraph_index"] == 0
                and isinstance(path, list) and len(path) == 1 and type(path[0]) is int and path[0] >= 0):
            return
    if method == "email" and kind == "email":
        email = coverage.get("email") or {}
        part = locator.get("mime_part")
        if (set(locator) == {"kind", "mime_part"} and isinstance(part, str) and re.fullmatch(r"1(?:\.[1-9][0-9]*)*", part)
                and email.get("body_content_type") == "text/plain" and email.get("body_part") == part):
            return
    raise ValueError("Invalid original location for the writing position check")


def inputs(text, spans):
    words = [text[row["start"]:row["end"]] for row in spans]
    return [(" ".join(words[j] for j in selected),
             sum(len(words[j]) + 1 for j in selected if j < i),
             len(words[i])) for i, selected in enumerate(SCOPES)]


def _amount(value):
    if not isinstance(value, str):
        raise ValueError("An exact writing response is missing")
    response = float.fromhex(value)
    if not math.isfinite(response) or response.hex() != value:
        raise ValueError("Invalid exact writing response")
    return response


def record(spans, readings, focus_text, *, locator=None):
    left, center, right = (_amount(item["response_hex"]) for item in readings)
    contrast = center - max(left, right)
    return {**unassessed(spans, focus_text, locator=locator),
            "status": "flagged" if contrast > POINT else "no_flag", "readings": readings, "contrast_hex": contrast.hex()}


def from_signal(signal, ref):
    payload = (signal.get("details", {}).get("text_evidence") or {}).get(KEY)
    if payload is None:
        return None
    if not document_matched_context.matches(signal["detector_id"], signal["detector_version"], signal["model_revisions"]):
        raise ValueError("The writing position check has a different classifier binding")
    return {**payload, "signal_ref": ref}


def from_result(result):
    from app.api.public_projection import alias_signal

    rows = [review for signal in result.get("signals", [])
            if (review := from_signal(signal, {"id": alias_signal(signal["id"]), "display_id": signal.get("display_id")}))]
    if len(rows) > 1:
        raise ValueError("Repeated writing position evidence")
    return rows[0] if rows else None


def validate_commitment(document, expected):
    value = (document.get("coverage") or {}).get(KEY)
    if expected != (canonical_hash(value) if value is not None else None):
        raise ValueError("Writing position evidence differs from its published result commitment")


def unassessed(spans, focus_text, *, locator=None):
    row = {"schema": SCHEMA, "status": "unassessed", "authorship_confidence_contribution": 0,
           "sentences": spans, "focus_text": focus_text}
    if locator is not None:
        row.update(schema=MAPPED_SCHEMA, source_locator=locator)
    return row


def validate_readings(readings):
    if len(readings) != 3:
        raise ValueError("Three complete writing responses required")
    for row in readings:
        if (set(row) != {"input_sha256", "response_hex", "focus_tokens", "neighbor_tokens"}
                or not SHA256.fullmatch(row["input_sha256"])
                or any(type(row[key]) is not int or row[key] <= 0 for key in ("focus_tokens", "neighbor_tokens"))
                or row["focus_tokens"] + row["neighbor_tokens"] > 512):
            raise ValueError("Incomplete writing input coverage")
        _amount(row["response_hex"])
    return readings


def validate(document, *, published_checks=None):
    value = (document.get("coverage") or {}).get(KEY)
    if value is None:
        return None
    schema = value.get("schema")
    mapped = schema == MAPPED_SCHEMA
    if (schema not in SHORT_SCHEMAS | document_long_passage_position.SCHEMAS
            or value.get("authorship_confidence_contribution") != 0
            or not mapped and document["source"].get("method") != "plain"):
        raise ValueError("Invalid writing position scope")
    checks = {row["signal_ref"]["id"] for row in document.get("coverage", {}).get("checks", [])}
    ref = value.get("signal_ref") or {}
    if (ref.get("id") not in checks or published_checks is not None and ref["id"] not in published_checks
            or not isinstance(ref.get("display_id"), str) or not re.fullmatch(r"S[0-9]{2,6}", ref["display_id"])):
        raise ValueError("Writing position evidence has no recorded check")
    if schema in document_long_passage_position.SCHEMAS:
        return document_long_passage_position.validate(value, document["source"])
    required = {"schema", "status", "authorship_confidence_contribution", "sentences", "signal_ref", "focus_text"}
    if mapped:
        required.add("source_locator")
    if value.get("status") != "unassessed":
        required.update(("readings", "contrast_hex"))
    if set(value) != required:
        raise ValueError("Invalid writing position scope")
    locator = value.get("source_locator")
    if mapped:
        _validate_locator(document, locator or {})
    spans = value["sentences"]
    if len(spans) != 3:
        raise ValueError("Three complete writing positions required")
    size = document["source"]["chars"]
    first_source = spans[0]["source_start"]
    if type(first_source) is not int or first_source < 0:
        raise ValueError("Writing positions do not span the one mapped paragraph")
    previous = original_end = -1
    for row in spans:
        a, b = check_range(row, size)
        lo, hi = row["source_start"], row["source_end"]
        if (set(row) != {"start", "end", "source_start", "source_end"} or a <= previous
                or type(lo) is not int or type(hi) is not int or not original_end < lo < hi
                or (hi - lo < b - a if mapped else (lo, hi) != (a + first_source, b + first_source))):
            raise ValueError("Writing positions are not contiguous with the original source map")
        previous, original_end = b, hi
    if spans[0]["start"] != 0 or spans[-1]["end"] != size:
        raise ValueError("Writing positions do not span the one mapped paragraph")
    if not isinstance(value["focus_text"], str) or len(value["focus_text"]) != spans[1]["end"] - spans[1]["start"]:
        raise ValueError("Writing position evidence has an invalid focus excerpt")
    if value["status"] == "unassessed":
        return value
    readings = validate_readings(value["readings"])
    expected = record(spans, readings, value["focus_text"], locator=locator)
    if expected != {key: item for key, item in value.items() if key != "signal_ref"}:
        raise ValueError("The writing position direction does not replay")
    return value


def verify_source(document, text, mapping, *, result=None, meta=None):
    from app.detectors.text.extract import from_string, text_hash
    from app.detectors.text.segment import segment

    value = validate(document)
    if result is not None and canonical_hash(value) != canonical_hash(from_result(result)):
        raise ValueError("Writing position evidence differs from the recorded analysis")
    if value is None:
        return False
    sentences = segment(from_string(text))
    method = document["source"]["method"]
    if value["schema"] in document_long_passage_position.SCHEMAS:
        return document_long_passage_position.verify_source(value, text, mapping, method, sentences)
    locator = value.get("source_locator")
    if locator is not None and mapped_locator(mapping, meta or {}) != locator:
        raise ValueError("Writing sentence locator differs from the original source")
    mapped = geometry(text, mapping, method, sentences, locator=locator)
    if mapped != value["sentences"] or value["focus_text"] != text[mapped[1]["start"]:mapped[1]["end"]]:
        raise ValueError("Writing sentence positions differ from the original source")
    if value["status"] == "unassessed":
        return False
    for (content, _, _), row in zip(inputs(text, mapped), value["readings"], strict=True):
        if text_hash(content) != row["input_sha256"]:
            raise ValueError("Writing input differs from the original source")
    return True
