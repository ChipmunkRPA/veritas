"""Report the text edits explicitly marked in the main Word document part."""

from __future__ import annotations

import hashlib
import re

from app.core.canonical import canonical_hash

KEY = "word_tracked_changes"
COMMITMENT = "word_tracked_changes_sha256"
SCHEMA = "document-word-tracked-changes/1"
MAX_ROWS = 24
MAX_EXCERPT = 240
SCOPE = "Checks inline Track Changes insertions and deletions in top-level Word body paragraphs. Tables, headers, footers, paragraph marks, moves and formatting are outside the measured scope."
LIMITATION = "These are markings stored in the submitted file. They do not establish who wrote the text, when it changed, or whether AI was involved."
SHA256 = re.compile(r"[0-9a-f]{64}")
PART_NAME = re.compile(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*/)*[A-Za-z0-9_][A-Za-z0-9_.-]*\.xml")


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _revision_text(element, kind, tag):
    content = []
    for run in element:
        if run.tag != tag["r"]:
            return None
        for item in run:
            if item.tag == tag["rPr"]:
                continue
            if len(item):
                return None
            if item.tag == tag["t" if kind == "inserted" else "delText"]:
                content.append(item.text or "")
            elif item.tag == tag["tab"]:
                content.append("\t")
            elif item.tag == tag["br"] and item.get(tag["type"], "textWrapping") == "textWrapping":
                content.append("\n")
            else:
                return None
    return "".join(content)


def inspect(body, part):
    """Inspect the already parsed Word XML. Positions count all sibling elements."""
    from docx.oxml.ns import qn

    inline_names = ("p", "r", "rPr", "ins", "del", "t", "delText", "tab", "br", "type")
    tag = {name: qn(f"w:{name}") for name in inline_names}
    kinds = {tag["ins"]: "inserted", tag["del"]: "deleted"}
    marks = set(kinds) | {qn(f"w:{name}") for name in (
        "moveFrom", "moveTo", "moveFromRangeStart", "moveToRangeStart", "cellIns", "cellDel", "cellMerge")}
    visited = set()
    rows, totals = [], dict.fromkeys(kinds.values(), 0)
    unsupported, paragraph = 0, 0
    for body_index, parent in enumerate(item for item in body if isinstance(item.tag, str)):
        if parent.tag != tag["p"]:
            continue
        paragraph += 1
        for inline_index, item in enumerate(child for child in parent if isinstance(child.tag, str)):
            if (kind := kinds.get(item.tag)) is None:
                continue
            visited.add(item)
            text = _revision_text(item, kind, tag)
            if not text or not text.strip():
                unsupported += 1
                continue
            totals[kind] += 1
            if len(rows) < MAX_ROWS:
                rows.append({"kind": kind, "paragraph": paragraph, "body_index": body_index, "inline_index": inline_index,
                             "text": text[:MAX_EXCERPT], "characters": len(text), "text_sha256": _sha(text)})
    unsupported += sum(item not in visited for item in body.iter() if item.tag in marks)
    measured = sum(totals.values())
    return {"schema": SCHEMA, "part": part, "status": "partial" if unsupported else "found" if measured else "no_match",
            "scope": SCOPE, "limitation": LIMITATION,
            "paragraphs": paragraph, **totals, "unsupported_marks": unsupported, "omitted_excerpts": measured - len(rows),
            "rows": rows, "authorship_confidence_contribution": 0}


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(COMMITMENT)


def validate(document, expected):
    review = (document.get("coverage") or {}).get(KEY)
    if review is None and expected is None:
        return
    if (not isinstance(review, dict) or canonical_hash(review) != expected or document["source"]["method"] != "python-docx"
            or review["schema"] != SCHEMA or not PART_NAME.fullmatch(review["part"])
            or review["scope"] != SCOPE or review["limitation"] != LIMITATION
            or type(review["authorship_confidence_contribution"]) is not int
            or review["authorship_confidence_contribution"] != 0):
        raise ValueError("Tracked Word edits differ from their source commitment")
    fields = ("paragraphs", "inserted", "deleted", "unsupported_marks", "omitted_excerpts")
    if any(type(review[k]) is not int or review[k] < 0 for k in fields):
        raise ValueError("Invalid tracked edit coverage")
    measured = review["inserted"] + review["deleted"]
    rows = review["rows"]
    status = "partial" if review["unsupported_marks"] else "found" if measured else "no_match"
    if (review["status"] != status or len(rows) != min(measured, MAX_ROWS)
            or review["omitted_excerpts"] != measured - len(rows)):
        raise ValueError("Invalid tracked edit disposition")
    previous = (-1, -1)
    counts = {"inserted": 0, "deleted": 0}
    for row in rows:
        kind, text, length = row["kind"], row["text"], row["characters"]
        path = row["body_index"], row["inline_index"]
        if (kind not in counts or any(type(i) is not int or i < 0 for i in path) or path <= previous
                or type(row["paragraph"]) is not int or not 1 <= row["paragraph"] <= min(review["paragraphs"], path[0] + 1)
                or type(length) is not int or length < 1 or not isinstance(text, str)
                or len(text) != min(length, MAX_EXCERPT) or not SHA256.fullmatch(row["text_sha256"])
                or (length <= MAX_EXCERPT and _sha(text) != row["text_sha256"])):
            raise ValueError("A tracked Word edit has invalid text or native coordinates")
        counts[kind] += 1
        previous = path
    if any(counts[k] > review[k] for k in counts):
        raise ValueError("Tracked edit counts differ from displayed excerpts")


def verify_source(document, recorded, *, expected):
    validate(document, expected)
    review = (document.get("coverage") or {}).get(KEY)
    if review is not None and review != recorded:
        raise ValueError("Tracked Word edits differ from the original XML positions")
    return len(review["rows"]) if review is not None else 0
