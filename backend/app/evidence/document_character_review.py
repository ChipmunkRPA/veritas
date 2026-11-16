"""Bind a character-adjusted writing response to the unchanged document source."""

from __future__ import annotations

import math
import re

from app.core.canonical import canonical_hash
from app.detectors.text.character_view import SCHEMA as INPUT_VIEW
from app.detectors.text.character_view import derive
from app.evidence.document_native_locations import clip_source_ranges
from app.evidence.document_scope import CHARACTER_REVIEW_COMMITMENT as COMMITMENT
from app.evidence.document_scope import CHARACTER_REVIEW_KEY as KEY
from app.evidence.document_scope import check_range, mapped_email_html

SCHEMA = "document-writing-character-review/1"
MAX_RANGES = 6
MAX_CHARS = 1800
MAX_CHANGES = 24
METHODS = {"plain": "text", "python-docx": "docx", "email": "email", "pymupdf": "pdf"}
STATE = {
    "changed": "The writing response changed direction after character adjustment.",
    "different": "The adjusted readings gave different directions.",
    "consistent": "The response direction held after character adjustment.",
}
HASH = re.compile(r"[0-9a-f]{64}")


def from_result(result):
    return (result.get("manifest", {}).get("sampling") or {}).get(KEY)


def result_commitment(result):
    record = from_result(result)
    return canonical_hash(record) if record is not None else None


def _locator(block, meta, start, end):
    kind = block["kind"]
    if kind == "pdf":
        if block.get("extraction") == "text_layer" and any(
            item["start"] < end and start < item["end"] for item in block.get("fragments", ())
        ):
            return {"kind": kind, "page": block["page"]}
        return None
    ranges = clip_source_ranges(block.get("source_ranges") or (), start, end)
    if not ranges:
        return None
    if kind == "docx":
        path = block.get("body_path", ())
        return {"kind": kind, "ranges": ranges, "body_path": path} if len(path) == 1 and block.get("run_mapping_complete") else None
    if kind == "email":
        email = meta.get("email") or {}
        html = mapped_email_html(email)
        if (not (html or email.get("body_content_type") == "text/plain")
                or block.get("mime_part") != email.get("body_part") or block.get("quote_role") != "body"):
            return None
        return {"kind": kind, "ranges": ranges, "mime_part": block["mime_part"], "html": html}
    return {"kind": kind, "ranges": ranges}


def geometry(text, spans, mapping, meta):
    """Keep the center and its exact input fragments in one mapped source block."""
    from app.detectors.text.extract import text_hash

    if len(spans) != 3 or not (kind := METHODS.get(meta.get("method"))):
        return None
    bounds = [check_range(span, len(text)) for span in spans]
    a, b = bounds[1]
    lo, hi = bounds[0][0], bounds[2][1]
    if hi - lo > MAX_CHARS or any(left[1] > right[0] for left, right in zip(bounds, bounds[1:])):
        return None
    focus, center = derive(text[a:b])
    if not center:
        return None
    block = next((block for block in mapping if block.get("kind") == kind and block["start"] <= lo < hi <= block["end"]), None)
    if block is None or (place := _locator(block, meta, a, b)) is None:
        return None
    values = [text[start:end] for start, end in bounds]
    variants = [derive(values[0]), (focus, center), derive(values[2])]
    count = sum(len(changes) for _, changes in variants)
    if count > MAX_CHANGES:
        return None
    raw = " ".join(values)
    adjusted = " ".join(value for value, _ in variants)
    changes = [{"offset": start + i, "codepoint": f"U+{ord(text[start + i]):04X}", "replacement": replacement}
               for (start, _), (_, items) in zip(bounds, variants, strict=True) for i, replacement in items]
    return ({"spans": spans, "excerpt": text[a:b], "location": place, "source_input_sha256": text_hash(raw),
             "adjusted_input_sha256": text_hash(adjusted), "changes": changes}, adjusted)


def response_state(as_submitted, adjusted_primary, adjusted_second):
    raw, primary, second = (float.fromhex(value["response_exact_hex"]) >= .5
                            for value in (as_submitted, adjusted_primary, adjusted_second))
    if primary != second:
        return "different"
    if raw == primary:
        return "consistent"
    return "changed"


def _validate_response(reading):
    if set(reading) != {"response_exact_hex", "tokens"} or type(reading["tokens"]) is not int or not 0 < reading["tokens"] <= 512:
        raise ValueError("Invalid character-adjusted writing response")
    value = float.fromhex(reading["response_exact_hex"])
    if not math.isfinite(value) or not 0 <= value <= 1 or value.hex() != reading["response_exact_hex"]:
        raise ValueError("Invalid character-adjusted writing response")


def validate(document, expected):
    record = (document.get("coverage") or {}).get(KEY)
    if record is None and expected is None:
        return
    if (not isinstance(record, dict) or set(record) != {"schema", "input_view", "ranges", "authorship_confidence_contribution"}
            or record["schema"] != SCHEMA or record["input_view"] != INPUT_VIEW
            or type(record["authorship_confidence_contribution"]) is not int
            or record["authorship_confidence_contribution"] != 0 or not isinstance(record["ranges"], list)
            or canonical_hash(record) != expected or not 0 < len(record["ranges"]) <= MAX_RANGES):
        raise ValueError("Invalid character-adjusted writing record")
    chars, method = document["source"]["chars"], document["source"]["method"]
    previous = 0
    for row in record["ranges"]:
        if set(row) != {"spans", "excerpt", "location", "source_input_sha256", "adjusted_input_sha256", "changes",
                        "as_submitted", "adjusted_primary", "adjusted_second", "state"}:
            raise ValueError("Incomplete character-adjusted writing range")
        if not isinstance(row["spans"], list) or len(row["spans"]) != 3:
            raise ValueError("A character-adjusted writing review needs three source spans")
        bounds = [check_range(span, chars) for span in row["spans"]]
        start, stop = bounds[1]
        if (start < previous or bounds[-1][1] - bounds[0][0] > MAX_CHARS
                or not isinstance(row["excerpt"], str) or len(row["excerpt"]) != stop - start
                or row["location"].get("kind") != METHODS.get(method)
                or any(left[1] > right[0] for left, right in zip(bounds, bounds[1:]))
                or not 1 <= len(row["changes"]) <= MAX_CHANGES
                or any(not HASH.fullmatch(row[key]) for key in ("source_input_sha256", "adjusted_input_sha256"))
                or row["source_input_sha256"] == row["adjusted_input_sha256"]):
            raise ValueError("Character-adjusted writing coordinates differ")
        center = [{"offset": start + offset, "codepoint": f"U+{ord(row['excerpt'][offset]):04X}", "replacement": value}
                  for offset, value in derive(row["excerpt"])[1]]
        if (not center or [item for item in row["changes"] if start <= item["offset"] < stop] != center
                or any(not any(a <= item["offset"] < b for a, b in bounds) for item in row["changes"])):
            raise ValueError("Character adjustments differ from the indicated source excerpt")
        replies = [row[key] for key in ("as_submitted", "adjusted_primary", "adjusted_second")]
        for reply in replies:
            _validate_response(reply)
        if row["state"] != response_state(*replies):
            raise ValueError("Invalid comparison of the writing responses")
        previous = stop


def verify_source(document, text, mapping, meta, *, sentences=None):
    record = (document.get("coverage") or {}).get(KEY)
    validate(document, document["source"].get(COMMITMENT))
    if record is None:
        return
    if sentences is not None:
        source_windows = {tuple((row["start"], row["end"]) for row in group)
                          for group in zip(sentences, sentences[1:], sentences[2:])}
    for row in record["ranges"]:
        positions = tuple((span["start"], span["end"]) for span in row["spans"])
        if sentences is not None and positions not in source_windows:
            raise ValueError("Adjusted writing review differs from the extracted sentence positions")
        material = geometry(text, row["spans"], mapping, meta)
        if material is None or any(row[key] != value for key, value in material[0].items()):
            raise ValueError("Adjusted writing inputs differ from the source text")
