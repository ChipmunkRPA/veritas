"""Identify a terminal list of complete citations outside the writing input."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace

KEY = "writing_bibliography"
SCHEMA = "chipmunk.writing-bibliography/1"
HEADLINE = "Writing assessment limited to body text"
HEADINGS = {"references", "bibliography", "works cited"}
MIN_ENTRIES, MAX_ENTRIES = 4, 96
METHODS = {"plain": "text", "python-docx": "docx", "pymupdf": "pdf"}
_NUMBER = re.compile(r"^(?:\[\d{1,3}\]|\d{1,3}[.)])\s+")
_NAME_WORD = r"[^\W\d_][\w’'.-]{0,45}"
_AUTHOR = re.compile(rf"^(?:{_NAME_WORD}(?: {_NAME_WORD}){{0,3}},\s{{0,2}}|{_NAME_WORD} )[A-Z]{{1,4}}\b")
_YEAR = re.compile(r"\b(?:18|19|20)\d{2}[a-z]?\b")
_DOI = re.compile(r"\b10\.\d{4,9}/[\w.():/;-]{3,150}", re.I)
_PAGES = re.compile(r"\b\d{1,4}(?:\s*\(\d{1,4}\))?\s*:\s*[a-z]?\d+(?:\s*[-–]\s*[a-z]?\d+)?\b", re.I)
_APA_PAGES = re.compile(r"\b\d{1,4}(?:\s*\(\d{1,4}\))?,\s*(?:pp\.\s*)?\d+\s*[-–]\s*\d+\b", re.I)
_HASH = re.compile(r"[a-f0-9]{64}")


def _digest(value):
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def _entry(value):
    if not 32 <= len(value) <= 560:
        return False
    citation = _NUMBER.sub("", value, count=1)
    if not _AUTHOR.match(citation):
        return False
    year = _YEAR.search(citation, 0, 175)
    return bool(year and (_DOI.search(citation) or _PAGES.search(citation, year.end()) or _APA_PAGES.search(citation, year.end())))


def _location(mapped):
    if mapped["kind"] == "text":
        first, last = mapped["source_ranges"][0], mapped["source_ranges"][-1]
        return {"kind": "text", "start": first[2], "end": last[3]}
    if mapped["kind"] == "docx":
        path = mapped["body_path"]
        if len(path) == 1:
            return {"kind": "docx", "body_index": path[0]}
    if mapped["kind"] == "pdf" and mapped.get("extraction") != "ocr":
        return {"kind": "pdf", "page": mapped["page"]}
    return None


def measure(doc):
    text, paragraphs, mapping, meta = doc.text, doc.paragraphs, doc.source_map, doc.meta
    expected_kind = METHODS.get(meta.get("method"))
    if (expected_kind is None or len(paragraphs) != len(mapping) or len(paragraphs) < MIN_ENTRIES + 2
            or meta.get("unmapped_pages") or meta.get("partially_checked_pages")
            or (meta.get("pdf_text_visibility") or {}).get("covered_locations")
            or any(page.get("status") == "extracted" for page in (meta.get("ocr") or {}).get("pages", []))):
        return None

    for entries, index in enumerate(range(len(paragraphs) - 1, 0, -1)):
        start, end = paragraphs[index]
        value = text[start:end]
        if value.casefold() in HEADINGS:
            if entries < MIN_ENTRIES:
                return None
            body_end = paragraphs[index - 1][1]
            if len(text[:body_end].split()) < 40 or text[body_end:start] != "\n\n":
                return None
            tail = mapping[index:]
            if any(row.get("kind") != expected_kind for row in tail):
                return None
            locations = [_location(row) for row in tail]
            if None in locations:
                return None
            return {"schema": SCHEMA, "heading": value.casefold(), "body_chars": body_end, "section_start": start,
                    "section_end": len(text), "section_sha256": _digest(text[start:]), "entry_count": entries,
                    "first_location": locations[0], "last_location": locations[-1]}
        if entries == MAX_ENTRIES or not _entry(value):
            return None
    return None


def body_view(doc):
    """Keep the original prefix and source map so all retained offsets stay identical."""
    record = measure(doc)
    if record is None:
        return doc
    end = record["body_chars"]
    text = doc.text[:end]
    paragraphs = [(a, b) for a, b in doc.paragraphs if b <= end]
    source_map = [row for row in doc.source_map if row["end"] <= end]
    meta = {**doc.meta, KEY: record, "chars": end, "words": len(text.split()), "paragraphs": len(paragraphs),
            "extracted_text_sha256": _digest(text)}
    return replace(doc, text=text, paragraphs=paragraphs, source_map=source_map, meta=meta)


def notice(record):
    return f"The writing input excluded a terminal reference list with {record['entry_count']} citation entries."


def location_label(record):
    first, last = record["first_location"], record["last_location"]
    if first["kind"] == "pdf":
        return f"PDF page {first['page']}" if first["page"] == last["page"] else f"PDF pages {first['page']}–{last['page']}"
    if first["kind"] == "docx":
        return f"Word body blocks {first['body_index'] + 1}–{last['body_index'] + 1}"
    return f"Source characters {first['start'] + 1}–{last['end']}"


def validate(evidence):
    record = evidence.get("coverage", {}).get(KEY)
    if record is None:
        return
    source = evidence["source"]
    kind = METHODS.get(source["method"])
    if (type(record) is not dict or set(record) != {"schema", "heading", "body_chars", "section_start", "section_end",
                 "section_sha256", "entry_count", "first_location", "last_location"}
            or record["schema"] != SCHEMA or record["heading"] not in HEADINGS or kind is None
            or type(record["entry_count"]) is not int or not MIN_ENTRIES <= record["entry_count"] <= MAX_ENTRIES
            or not all(type(record[key]) is int for key in ("body_chars", "section_start", "section_end"))
            or record["body_chars"] != source["chars"] or record["section_start"] != record["body_chars"] + 2
            or record["section_end"] < record["section_start"] + 32 * record["entry_count"]
            or not isinstance(record["section_sha256"], str) or not _HASH.fullmatch(record["section_sha256"])):
        raise ValueError("Invalid bibliography exclusion record")
    first, last = record["first_location"], record["last_location"]
    indices = {"text": ("start", "end"), "docx": ("body_index",), "pdf": ("page",)}[kind]
    for location in (first, last):
        if (type(location) is not dict or set(location) != {"kind", *indices} or location["kind"] != kind
                or any(type(location[key]) is not int or location[key] < (1 if kind == "pdf" else 0) for key in indices)
                or kind == "text" and location["start"] >= location["end"]):
            raise ValueError("Invalid bibliography source location")
    if first[indices[-1]] > last[indices[0]]:
        raise ValueError("Bibliography source locations are out of order")


def verify_source(evidence, meta):
    validate(evidence)
    if evidence.get("coverage", {}).get(KEY) != meta.get(KEY):
        raise ValueError("The recorded bibliography differs from the source")
