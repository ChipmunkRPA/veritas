# SPDX-License-Identifier: Apache-2.0
"""Validate saved PDF wording commitments without importing the PDF reader."""

from __future__ import annotations

import hashlib
import re
from math import isfinite

from app.core.canonical import canonical_hash

KEY = "pdf_saved_wording"
COMMITMENT = f"{KEY}_sha256"
SCHEMA = "chipmunk.pdf-saved-wording/1"
MAX_BYTES = 24 * 1024 * 1024
MAX_SAVES = 5
MAX_PAGES = 24
MAX_WORDS = 3500
MAX_CHANGES = 8
MAX_EXCERPT_WORDS = 18
MAX_EXCERPT_CHARS = 320
_HASH = re.compile(r"[a-f0-9]{64}")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _has_changes(row):
    return bool(row["changed_ranges"] or row["earlier_only_pages"] or row["current_only_pages"])


def from_result(result):
    return (result.get("manifest", {}).get("sampling") or {}).get(KEY)


def result_commitment(result):
    record = from_result(result)
    return canonical_hash(record) if record is not None else None


def current_locations(record):
    if record["status"] == "changed":
        for index, change in enumerate(record["comparisons"][-1]["changes"], 1):
            if current := change["current"]:
                yield index, current


def _valid_location(row, page_limit, *, current):
    if row is None:
        return True
    fields = {"page", "page_xref", "rotation", "width", "height", "word_range", "excerpt_word_range", "boxes"}
    if set(row) != (fields | {"excerpt", "excerpt_sha256"} if current else fields):
        return False
    low, high = row["word_range"]
    first, last = row["excerpt_word_range"]
    width, height = row["width"], row["height"]
    boxes = row["boxes"]
    bounds = (row["page"], row["page_xref"], low, high, first, last, row["rotation"])
    if (any(type(value) is not int for value in bounds) or not 1 <= row["page"] <= page_limit
            or row["page_xref"] <= 0 or row["rotation"] not in {0, 90, 180, 270}
            or not 0 <= low == first < last <= high <= MAX_WORDS or not 1 <= last - first == len(boxes) <= MAX_EXCERPT_WORDS
            or not all(isfinite(value) and value > 0 for value in (width, height))):
        return False
    for x0, y0, x1, y1 in boxes:
        if not 0 <= x0 < x1 <= width + .01 or not 0 <= y0 < y1 <= height + .01:
            return False
    if current:
        text = row["excerpt"]
        return (0 < len(text) <= MAX_EXCERPT_CHARS and len(text.split()) == len(boxes)
                and _sha(text.encode()) == row["excerpt_sha256"])
    return True


def _validate_comparison(row, earlier, current):
    if (row["earlier_version"] != earlier["version"] or type(row["changed_ranges"]) is not int
            or row["changed_ranges"] < 0 or len(row["changes"]) != min(row["changed_ranges"], MAX_CHANGES)):
        raise ValueError("A wording comparison is outside its recorded revision")
    pairs = row["checked"] + row["unassessed"]
    for index, name, scope in ((0, "earlier_only_pages", earlier), (1, "current_only_pages", current)):
        covered = [pair[index] for pair in pairs] + row[name]
        if sorted(covered) != list(range(1, scope["pages"] + 1)):
            raise ValueError("PDF page coverage differs from its recorded saves")
    if any(pair not in row["checked"] for pair in row["order_only"]):
        raise ValueError("Inconsistent coverage for pages with reordered text")
    for change in row["changes"]:
        old, new = change["earlier"], change["current"]
        if old is None and new is None:
            raise ValueError("A wording change has no location")
        kind = "replace" if old and new else "delete" if old else "insert"
        located = any((old is None or pair[0] == old["page"]) and (new is None or pair[1] == new["page"])
                      for pair in row["checked"])
        if (change["kind"] != kind or not located or old and new and old["page_xref"] != new["page_xref"]
                or not _valid_location(old, earlier["pages"], current=False)
                or not _valid_location(new, current["pages"], current=True)):
            raise ValueError("A wording change differs from its matched native PDF pages")


def validate(evidence, expected):
    review = evidence.get("coverage", {}).get(KEY)
    if (expected is None) != (review is None) or evidence["source"].get(COMMITMENT) != expected:
        raise ValueError("Saved wording differs from the published document commitment")
    if review is None:
        return
    record = review["inspection"]
    if (evidence["source"].get("method") != "pymupdf" or canonical_hash(record) != expected
            or record["schema"] != SCHEMA or record["affects_overall_confidence"] is not False
            or record["status"] not in {"unavailable", "changed", "no_change", "no_history"}
            or type(record["source_bytes"]) is not int or record["source_bytes"] < 1):
        raise ValueError("Saved wording is outside its PDF source or scope")
    saves, comparisons = validate_saves(record)
    if ((record["status"] == "changed") != bool(comparisons and _has_changes(comparisons[-1]))
            or any(_has_changes(row) for row in comparisons[:-1])):
        raise ValueError("PDF wording changes require the corresponding comparison")
    for row, earlier in zip(comparisons, saves[1:]):
        if not 1 <= max(saves[0]["pages"], earlier["pages"]) <= MAX_PAGES:
            raise ValueError("A wording comparison exceeds its page scope")
        _validate_comparison(row, earlier, saves[0])
    targets = {f"change{index}_{role}": side["page"] for index, side in current_locations(record) for role in ("crop", "overlay")}
    artifacts = review["artifacts"]
    if "pictures" in record:
        from app.evidence import document_pdf_picture_binding

        picture_assets = {key: value for key, value in artifacts.items() if key.startswith("picture")}
        document_pdf_picture_binding.validate(record["pictures"], picture_assets, size=record["source_bytes"])
        artifacts = {key: value for key, value in artifacts.items() if key not in picture_assets}
    if set(artifacts) != set(targets):
        raise ValueError("PDF wording changes require their corresponding page crops")
    for key, asset in artifacts.items():
        if (asset["media_type"] != "image/png" or not _HASH.fullmatch(asset["sha256"]) or asset["page"] != targets[key]
                or asset["role"] != key.rsplit("_", 1)[1] or min(asset["width"], asset["height"]) <= 0
                or max(asset["width"], asset["height"]) > 1200):
            raise ValueError("A wording preview differs from the current PDF page")


def validate_saves(record):
    saves, comparisons = record["saves"], record["comparisons"]
    levels = len(saves), len(comparisons)
    if not 0 <= levels[0] <= MAX_SAVES or not max(0, levels[0] - 2) <= levels[1] <= max(0, levels[0] - 1):
        raise ValueError("Saved PDF revision coverage is inconsistent")
    before = None
    for save in saves:
        if (not 0 < save["xref_offset"] < save["prefix_bytes"] <= record["source_bytes"] or not _HASH.fullmatch(save["sha256"])
                or type(save["version"]) is not int or save["version"] < 1 or save["xref_kind"] not in {"table", "stream"}
                or type(save["pages"]) is not int or save["pages"] < 0
                or before is not None and (save["prefix_bytes"] > before["xref_offset"] or save["version"] != before["version"] - 1)):
            raise ValueError("A saved PDF boundary is invalid")
        before = save
    if record["status"] == "no_history" and (levels != (1, 0) or saves[0]["version"] != 1):
        raise ValueError("PDF history status disagrees with its recorded saves")
    return saves, comparisons
