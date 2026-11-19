"""Locate one exact selectable PDF heading without interpreting its statement."""

from __future__ import annotations

import math
import re
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.detectors.text.extract import Document

KEY = "printed_ai_use_heading"
SCHEMA = "chipmunk.printed-ai-use-heading/1"
COMMITMENT = "printed_ai_use_heading_sha256"
POLICY = "printed_ai_use_heading_review"
TITLE = "Printed AI-use heading"
LIMITATION = ("A printed heading can be part of a quotation or example. "
              "This observation carries zero weight in the AI assessment.")
MAX_PAGES = 12

_PRECISION = 6
_HEADINGS = {"ai use disclosure", "ai usage disclosure", "generative ai use disclosure", "generative ai usage disclosure"}
_WORD = re.compile(r"\S+")
_SHA = re.compile(r"[a-f0-9]{64}")
_MIN_CONTRAST, _RENDER_SCALE, _MAX_WORD_PIXELS = 24, 1.5, 1_000_000
_FIELDS = {"schema", "text", "start", "end", "page", "block", "box", "word_boxes", "affects_overall_confidence"}


def _heading(text) -> bool:
    return isinstance(text, str) and text.removesuffix(":").casefold() in _HEADINGS


def _valid_box(box) -> bool:
    return (isinstance(box, list) and len(box) == 4
            and all(type(value) is int or (type(value) is float and math.isfinite(value) and value == round(value, _PRECISION))
                    for value in box)
            and 0 <= box[0] < box[2] and 0 <= box[1] < box[3])


def validate(record, *, source_chars: int, pages: int) -> None:
    if (not isinstance(record, dict) or set(record) != _FIELDS or record["schema"] != SCHEMA
            or record["affects_overall_confidence"] is not False or not _heading(record["text"])
            or type(source_chars) is not int or type(pages) is not int or not 1 <= pages <= MAX_PAGES
            or any(type(record[key]) is not int for key in ("start", "end", "page", "block"))
            or not 0 <= record["start"] < record["end"] <= source_chars
            or record["end"] - record["start"] != len(record["text"])
            or not 1 <= record["page"] <= pages or record["block"] < 0 or not _valid_box(record["box"])):
        raise ValueError("Invalid printed AI-use heading location")
    expected = record["text"].split(" ")
    words = record["word_boxes"]
    if not isinstance(words, list) or len(words) != len(expected):
        raise ValueError("Incomplete printed AI-use heading word boxes")
    x0, y0, x1, y1 = record["box"]
    previous_left = x0
    for word, text in zip(words, expected, strict=True):
        if (not isinstance(word, dict) or set(word) != {"text", "box"} or word["text"] != text
                or not _valid_box(word["box"])):
            raise ValueError("Invalid printed AI-use heading word box")
        a, b, c, d = word["box"]
        if not (x0 <= previous_left <= a < c <= x1 and y0 <= b < d <= y1):
            raise ValueError("Printed AI-use heading word box falls outside its native line")
        previous_left = a


def validate_evidence(document) -> None:
    """Require the optional public record and its source commitment to agree."""
    source = document["source"]
    if KEY not in document and COMMITMENT not in source:
        return
    if KEY not in document or COMMITMENT not in source or source.get("method") != "pymupdf":
        raise ValueError("The printed AI-use heading must be bound to its PDF source")
    from app.core.canonical import canonical_hash

    record = document[KEY]
    validate(record, source_chars=source.get("chars"), pages=MAX_PAGES)
    if canonical_hash(record) != source[COMMITMENT]:
        raise ValueError("The printed AI-use heading differs from its source commitment")


def _native_line(page, block_number, text):
    from pymupdf import Rect

    matches = []
    for block in page.get_text("rawdict")["blocks"]:
        if block["number"] != block_number or block["type"] != 0:
            continue
        for line in block["lines"]:
            if tuple(line["dir"]) != (1.0, 0.0):
                continue
            glyphs = [char for run in line["spans"] for char in run["chars"]]
            value = "".join(char["c"] for char in glyphs)
            if " ".join(value.split()) != text or any(len(char["c"]) != 1 for char in glyphs):
                continue
            boxes = []
            for match in _WORD.finditer(value):
                rect = Rect(glyphs[match.start()]["bbox"])
                for char in glyphs[match.start() + 1:match.end()]:
                    rect.include_rect(Rect(char["bbox"]))
                boxes.append({"text": match[0], "box": [round(point, _PRECISION) for point in rect]})
            matches.append(([round(point, _PRECISION) for point in line["bbox"]], boxes))
    return matches[0] if len(matches) == 1 else None


def _has_contrast(page, words) -> bool:
    from pymupdf import Matrix, Rect, csGRAY

    for word in words:
        rect = Rect(word["box"])
        pixels = (math.ceil(rect.width * _RENDER_SCALE) + 2) * (math.ceil(rect.height * _RENDER_SCALE) + 2)
        if pixels > _MAX_WORD_PIXELS:
            return False
        image = page.get_pixmap(matrix=Matrix(_RENDER_SCALE, _RENDER_SCALE), clip=rect, colorspace=csGRAY, alpha=False)
        samples = image.samples
        if not samples or max(samples) - min(samples) < _MIN_CONTRAST:
            return False
    return True


def capture(path: Path, doc: Document) -> dict | None:
    """Return the exact location from the local original and its decoded mapping."""
    import pymupdf

    from app.evidence import document_pdf_visibility as visibility
    from app.evidence.document_interval_locations import locate
    from app.evidence.document_scope import _complete_pdf_fragments

    pages = doc.meta.get("pages")
    raw_digest = doc.meta.get("original_sha256")
    if (doc.meta.get("method") != "pymupdf" or type(pages) is not int or not 1 <= pages <= MAX_PAGES
            or not isinstance(raw_digest, str) or not _SHA.fullmatch(raw_digest)):
        return None
    eligible = [block for block in doc.source_map if block["kind"] == "pdf" and _heading(doc.text[block["start"]:block["end"]])]
    if len(eligible) != 1:
        return None
    block, = eligible
    page_number, start, end = block["page"], block["start"], block["end"]
    record = doc.meta.get(visibility.KEY) or {}
    if (block.get("extraction") != "text_layer" or block.get("page_rotation") != 0
            or any(page_number not in record.get(key, ()) for key in (
                "examined_pages", "replacement_examined_pages", "covered_examined_pages"))
            or any(row["page"] == page_number for key in ("locations", "replacement_differences", "covered_locations")
                   for row in record.get(key, ()))
            or page_number in doc.meta.get("unmapped_pages", ())
            or any(row["page"] == page_number for row in doc.meta.get("partially_checked_pages", ()))
            or not _complete_pdf_fragments(block, doc.text, start, end)):
        return None
    geometry = locate("pymupdf", [block], start, end)
    if len(geometry) != 1:
        return None
    text = doc.text[start:end]
    try:
        original = path.read_bytes()
        if sha256(original).hexdigest() != raw_digest:
            return None
        with pymupdf.open(stream=original, filetype="pdf") as pdf:
            if not pdf.is_pdf or pdf.needs_pass or len(pdf) != pages or not 1 <= page_number <= pages:
                return None
            page = pdf[page_number - 1]
            match = _native_line(page, block["block"], text)
            if match is None:
                return None
            box, words = match
            height = block["page_height"]
            if (not height * .055 <= box[1] < box[3] <= height * .935
                    or [word["box"] for word in words] != geometry[0]["boxes"]):
                return None
            result = {"schema": SCHEMA, "text": text, "start": start, "end": end, "page": page_number,
                      "block": block["block"], "box": box, "word_boxes": words, "affects_overall_confidence": False}
            validate(result, source_chars=len(doc.text), pages=pages)
            return result if _has_contrast(page, words) else None
    except (OSError, RuntimeError, ValueError):
        return None


def verify_original(record, path: Path, doc: Document) -> None:
    validate(record, source_chars=len(doc.text), pages=doc.meta.get("pages"))
    if capture(path, doc) != record:
        raise ValueError("The printed AI-use heading differs from the original PDF")
