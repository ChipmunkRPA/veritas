"""Exact repetition in reliably mapped extracted text. No inference or calibration."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict

from app.core.canonical import canonical_hash
from app.evidence import document_scope

SCHEMA = "document-repeated-passages/1"
MIN_WORDS = 30
MAX_CHARACTERS = 1600
MAX_GROUPS = 4
MAX_OCCURRENCES = 3
COVERAGE = (
    "mapped_blocks",
    "compared_blocks",
    "excluded_short_or_long_blocks",
    "excluded_ocr_blocks",
    "excluded_quoted_or_signature_blocks",
    "excluded_other_blocks",
    "matching_groups",
    "omitted_groups",
)
SOURCE_KEYS = ("sha256", "extracted_text_sha256", "source_map_sha256")


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def _count(value) -> bool:
    return type(value) is int and value >= 0


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _commitment(source: dict) -> dict:
    return {
        "schema": SCHEMA,
        "offset_unit": "unicode_codepoint",
        "normalization": "frozen_extraction_only",
        "authorship_confidence_contribution": 0,
        "source_sha256": source["sha256"],
        "extracted_text_sha256": source["extracted_text_sha256"],
        "source_map_sha256": source["source_map_sha256"],
    }


def _status(compared: int, matched: int) -> str:
    return "matched" if matched else "no_match" if compared >= 2 else "not_assessed"


def _sized(text: str) -> bool:
    return len(text) <= MAX_CHARACTERS and len(text.split()) >= MIN_WORDS


def _anchor(entry: dict) -> dict | None:
    kind = entry.get("kind")
    if kind == "pdf" and entry.get("extraction") == "text_layer":
        keys = ("page", "block", "bbox", "coordinate_space")
    elif kind == "docx":
        keys = ("paragraph_index", "body_path")
    elif kind in {"text", "email"} and (kind == "text" or entry.get("quote_role") == "body"):
        ranges = entry.get("source_ranges") or []
        if not ranges or ranges[0][0] != entry["start"] or ranges[-1][1] != entry["end"]:
            return None
        anchor = {"kind": kind, "source_start": ranges[0][2], "source_end": ranges[-1][3]}
        if kind == "email":
            anchor["mime_part"] = entry["mime_part"]
        return anchor
    else:
        return None
    return {"kind": kind, **{key: entry[key] for key in keys}} if all(key in entry for key in keys) else None


def build(source: dict, text: str, mapping: list[dict]) -> dict:
    """Compare whole mapped blocks in extraction order and retain a bounded display."""
    _require(
        len(text) == source["chars"]
        and _digest(text) == source["extracted_text_sha256"]
        and canonical_hash(mapping) == source["source_map_sha256"],
        "Repeated passages require the frozen extracted text and source mapping",
    )
    coverage = dict.fromkeys(COVERAGE, 0)
    by_text = defaultdict(list)
    previous_end = 0
    for entry in mapping:
        start, end = entry["start"], entry["end"]
        _require(
            _count(start) and _count(end) and previous_end <= start < end <= len(text),
            "Repeated passages require ordered nonoverlapping source blocks",
        )
        previous_end = end
        coverage["mapped_blocks"] += 1
        if entry.get("extraction") == "ocr":
            coverage["excluded_ocr_blocks"] += 1
        elif entry.get("kind") == "email" and entry.get("quote_role") in {"quoted", "signature"}:
            coverage["excluded_quoted_or_signature_blocks"] += 1
        elif (anchor := _anchor(entry)) is None:
            coverage["excluded_other_blocks"] += 1
        elif not _sized(excerpt := text[start:end]):
            coverage["excluded_short_or_long_blocks"] += 1
        else:
            coverage["compared_blocks"] += 1
            by_text[excerpt].append({"start": start, "end": end, "anchor": anchor})
    matches = [(excerpt, occurrences) for excerpt, occurrences in by_text.items() if len(occurrences) >= 2]
    coverage["matching_groups"] = len(matches)
    coverage["omitted_groups"] = max(0, len(matches) - MAX_GROUPS)
    groups = [
        {
            "id": f"RP{index:02d}",
            "text": excerpt,
            "text_sha256": _digest(excerpt),
            "word_count": len(excerpt.split()),
            "total_occurrences": len(occurrences),
            "occurrences": occurrences[:MAX_OCCURRENCES],
        }
        for index, (excerpt, occurrences) in enumerate(matches[:MAX_GROUPS], 1)
    ]
    return {
        **_commitment(source),
        "status": _status(coverage["compared_blocks"], len(matches)),
        "coverage": coverage,
        "groups": groups,
    }


def _valid_docx_path(path) -> bool:
    if not isinstance(path, list) or not path or not _count(path[0]) or (len(path) - 1) % 5:
        return False
    for offset in range(1, len(path), 5):
        row, r, cell, c, paragraph = path[offset : offset + 5]
        if row != "row" or cell != "cell" or not all(_count(i) for i in (r, c, paragraph)):
            return False
    return True


def _validate_anchor(anchor: dict, size: int) -> None:
    _require(isinstance(anchor, dict), "Repeated text needs a source anchor")
    kind = anchor.get("kind")
    if kind in {"text", "email"}:
        keys = {"kind", "source_start", "source_end"}
        if kind == "email":
            keys.add("mime_part")
            _require(
                isinstance(anchor.get("mime_part"), str)
                and bool(re.fullmatch(r"[1-9]\d*(?:\.[1-9]\d*)*", anchor["mime_part"])),
                "Repeated email text needs its exact body part",
            )
        a, b = anchor.get("source_start"), anchor.get("source_end")
        _require(
            anchor.keys() == keys and _count(a) and _count(b) and b - a >= size,
            "Repeated text needs exact decoded source offsets",
        )
    elif kind == "docx":
        _require(
            anchor.keys() == {"kind", "paragraph_index", "body_path"}
            and _count(anchor.get("paragraph_index"))
            and _valid_docx_path(anchor.get("body_path")),
            "Repeated DOCX text needs its exact paragraph path",
        )
    elif kind == "pdf":
        bbox = anchor.get("bbox")
        _require(
            anchor.keys() == {"kind", "page", "block", "bbox", "coordinate_space"}
            and _count(anchor.get("page"))
            and anchor["page"] > 0
            and _count(anchor.get("block"))
            and anchor["coordinate_space"] == "unrotated-page-points"
            and isinstance(bbox, list)
            and len(bbox) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in bbox)
            and bbox[0] < bbox[2]
            and bbox[1] < bbox[3],
            "Repeated PDF text needs its exact native block geometry",
        )
    else:
        raise ValueError("Repeated text has an unsupported source anchor")


def validate(document: dict) -> None:
    """Offline check of shape, bounds and commitments without a parser or source text."""
    if document.get("version") != document_scope.VERSION:
        _require(
            "repeated_passages" not in document,
            "Historical documents cannot contain repeated-passage evidence",
        )
        return
    source, repeat = document["source"], document["repeated_passages"]
    commitment = _commitment(source)
    _require(
        isinstance(repeat, dict)
        and repeat.keys() == {*commitment, "status", "coverage", "groups"}
        and all(repeat[key] == value for key, value in commitment.items())
        and type(repeat["authorship_confidence_contribution"]) is int
        and all(
            isinstance(source.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", source[key])
            for key in SOURCE_KEYS
        )
        and _count(source["chars"]),
        "Repeated passages differ from their source commitments",
    )
    coverage, groups = repeat["coverage"], repeat["groups"]
    _require(
        isinstance(coverage, dict)
        and coverage.keys() == set(COVERAGE)
        and all(_count(n) for n in coverage.values()),
        "Invalid repeated-passage coverage counts",
    )
    compared, matched = coverage["compared_blocks"], coverage["matching_groups"]
    _require(
        coverage["mapped_blocks"] == sum(coverage[key] for key in COVERAGE[1:6]) <= source["chars"]
        and 2 * matched <= compared
        and repeat["status"] == _status(compared, matched)
        and isinstance(groups, list)
        and len(groups) == min(MAX_GROUPS, matched)
        and coverage["omitted_groups"] == matched - len(groups),
        "Repeated-passage coverage is inconsistent",
    )
    ranges = []
    distinct_text = set()
    displayed_occurrences = 0
    first_positions = []
    for index, group in enumerate(groups, 1):
        _require(isinstance(group, dict), "Invalid exact repeated-passage group")
        excerpt, occurrences = group["text"], group["occurrences"]
        total = group["total_occurrences"]
        _require(
            group.keys() == {"id", "text", "text_sha256", "word_count", "total_occurrences", "occurrences"}
            and group["id"] == f"RP{index:02d}"
            and isinstance(excerpt, str)
            and _sized(excerpt)
            and excerpt == " ".join(excerpt.split())
            and excerpt not in distinct_text
            and group["text_sha256"] == _digest(excerpt)
            and _count(group["word_count"])
            and group["word_count"] == len(excerpt.split())
            and _count(total)
            and total >= 2
            and isinstance(occurrences, list)
            and len(occurrences) == min(total, MAX_OCCURRENCES),
            "Invalid exact repeated-passage group",
        )
        distinct_text.add(excerpt)
        displayed_occurrences += total
        end = 0
        for occurrence in occurrences:
            _require(isinstance(occurrence, dict), "A repeated passage needs a source range")
            a, b = occurrence["start"], occurrence["end"]
            _require(
                occurrence.keys() == {"start", "end", "anchor"}
                and _count(a)
                and _count(b)
                and end <= a < b <= source["chars"]
                and b - a == len(excerpt),
                "A repeated passage differs from its extracted text range",
            )
            end = b
            ranges.append((a, b))
            _validate_anchor(occurrence["anchor"], len(excerpt))
        first_positions.append(occurrences[0]["start"])
    ordered = sorted(ranges)
    _require(
        first_positions == sorted(first_positions)
        and all(end <= start for (_, end), (start, _) in zip(ordered, ordered[1:]))
        and displayed_occurrences + 2 * coverage["omitted_groups"] <= compared,
        "Repeated-passage positions or totals overlap",
    )


def verify_source(document: dict, text: str, mapping: list[dict]) -> None:
    """Verify actual excerpts and anchors against the immutable extraction."""
    validate(document)
    if document.get("version") == document_scope.VERSION:
        _require(
            document["repeated_passages"] == build(document["source"], text, mapping),
            "Repeated passages differ from the frozen text or source mapping",
        )
