"""Locate complete repeated body sentences with different literal percentages."""

from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal

from app.core.canonical import FLOAT_DP
from app.evidence.document_repeat import block_anchor, validate_block_anchor

KEY = "percentage_consistency"
SCHEMA = "document-percentage-consistency/1"
MAPPED_SCHEMA = "document-percentage-consistency/2"
MAPPED_METHODS = {"python-docx": "docx", "email": "email", "pymupdf": "pdf"}
OCCURRENCE_FIELDS = {"start", "end", "quantity_start", "quantity_end", "text"}
SCOPE = "Compares complete body sentences with the same wording, punctuation and unit apart from one percentage value. The comparison ignores letter case and spacing."
LIMITATION = "Compare the surrounding context before interpreting a difference. The sentences could refer to different unstated dates, groups or denominators. This check carries zero AI-authorship weight."
MAPPED_LIMITATION = LIMITATION + " Locations refer to the extracted text and its page, paragraph or email body. Check the submitted document to confirm how it displays the text."
MAX_BLOCK_LENGTH = 6000
MAX_TEXT_LENGTH = 40_000
MAX_SENTENCE_LENGTH = 500
MAX_GROUPS = 4
MAX_OCCURRENCES = 4
NUMBER = re.compile(r"(?<![\w.,%$£€+−–\-/])(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?(?![\w.,/−–\-])")
PERCENTAGE = re.compile(r"(?<![\w.,%$£€+−–\-/])(?P<value>(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?) *"
                        r"(?P<unit>%|percent(?:age points?)?)(?![\w/−–\-])", re.I)
WORD = re.compile(r"[^\W\d_]+")
QUALIFIER = re.compile(r"\b(?:about|approximately|around|nearly|roughly|almost|at least|at most|more than|less than|under|over|"
                       r"up to|above|below|from|between)\s*$", re.I)
QUOTED = re.compile(r'["“”«»‘]|(?<!\w)\x27|\x27(?!\w)')


def percentage(text):
    """Return an exact numeral and key only for one strict, self-contained value occurrence."""
    if (not 35 <= len(text) <= MAX_SENTENCE_LENGTH or not text.endswith(".") or QUOTED.search(text)
            or len(WORD.findall(text)) < 7 or text.partition(" ")[0].startswith(("#", "*", "-", "§", "|"))):
        return None
    matches = list(PERCENTAGE.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    start, end = match.span("value")
    visible = text[start:end]
    # Other figures (including years and citations) remain part of the exact key.
    # A different percentage, or a range before this value, is outside scope.
    if len(visible) > 15 or QUALIFIER.search(text[:start]) or re.search(r"\b(?:to|and|or)\s*$", text[:start], re.I) and NUMBER.search(text[:start]):
        return None
    key = (" ".join(text[:start].casefold().split()), " ".join(text[end:].casefold().split()))
    value = Decimal(visible.replace(",", ""))
    return (key, (start, end), value) if 0 <= value <= 100 else None


def supported(method, email):
    return method in {"plain", "pymupdf", "python-docx"} or (
        method == "email" and (email or {}).get("body_content_type") == "text/plain"
    )


def source_block(row, method, eligibility):
    if method == "plain":
        return {} if row.get("kind") == "text" and row.get("source_ranges") else None
    if row.get("kind") != MAPPED_METHODS[method]:
        return None
    anchor = block_anchor(row)
    if anchor is None:
        return None
    if method == "python-docx":
        path = anchor["body_path"]
        if not row.get("run_mapping_complete") or len(path) != 1 or path[0] not in eligibility:
            return None
        return anchor | {"body_paragraph": eligibility[path[0]]}
    if method == "pymupdf":
        if anchor["page"] not in eligibility:
            return None
        return anchor | {"bbox": [round(value, FLOAT_DP) for value in anchor["bbox"]]}
    return anchor


def build(source, text, mapping, sentences, *, meta=None):
    records = defaultdict(list)
    skipped = 0
    candidates = {}
    method = source["method"]
    mapped = method != "plain"
    meta = meta or {}
    if method == "python-docx":
        eligibility = {index: ordinal for ordinal, index in enumerate(meta.get("percentage_body_paragraphs", ()), 1)
                       if index is not None}
    else:
        eligibility = set(meta.get("percentage_text_pages", ()))
    if len(text) > MAX_TEXT_LENGTH:
        skipped = len(mapping)
    else:
        for index, row in enumerate(mapping):
            start, end = row["start"], row["end"]
            anchor = source_block(row, method, eligibility)
            if (anchor is None or not 0 <= start < end <= len(text) or end - start > MAX_BLOCK_LENGTH
                    or ">" in text[start:end]):
                skipped += 1
                continue
            candidates[index] = anchor
    checked = 0
    for unit in sentences:
        index = unit["paragraph"]
        if index not in candidates:
            continue
        start, end = unit["start"], unit["end"]
        block = mapping[index]
        if not block["start"] <= start < end <= block["end"] or unit["text"] != text[start:end]:
            raise ValueError("Percentage evidence needs the frozen mapped sentence positions")
        if (hit := percentage(unit["text"])) is None:
            continue
        checked += 1
        key, (lo, hi), value = hit
        occurrence = {"start": start, "end": end, "text": unit["text"], "quantity_start": start + lo, "quantity_end": start + hi}
        if mapped:
            occurrence["anchor"] = candidates[index]
        records[key].append((value, index, occurrence))
    results = []
    for group in records.values():
        first = group[0]
        distinct_blocks = {entry[1] for entry in group}
        if len(distinct_blocks) < 2 or len({entry[0] for entry in group}) < 2:
            continue
        # If an earlier value is repeated with a disagreement in its own
        # paragraph, a later repeat can still supply the other source block.
        opposite = next((entry for entry in group if entry[0] != first[0] and entry[1] != first[1]), None)
        if opposite is None:
            first = next(entry for entry in group if entry[0] != first[0])
            opposite = next(entry for entry in group if entry[0] != first[0] and entry[1] != first[1])
        observations = sorted([first, opposite, *[r for r in group if r not in (first, opposite)][:MAX_OCCURRENCES - 2]],
                              key=lambda entry: entry[2]["start"])
        results.append([row for _, _, row in observations])
    comparable = sum(len({entry[1] for entry in group}) >= 2 for group in records.values())
    return {"schema": MAPPED_SCHEMA if mapped else SCHEMA,
            "status": "different_values" if results else "checked" if comparable else "not_assessed",
            "scope": SCOPE, "limitation": MAPPED_LIMITATION if mapped else LIMITATION, "authorship_confidence_contribution": 0,
            "extracted_text_sha256": source["extracted_text_sha256"], "source_map_sha256": source["source_map_sha256"],
            "checked_sentences": checked, "comparable_claims": comparable, "excluded_blocks": skipped, "different_claims": len(results),
            "groups": results[:MAX_GROUPS]}


def validate(document):
    if KEY not in document:
        return
    record = document[KEY]
    source = document["source"]
    method = source.get("method")
    mapped = method in MAPPED_METHODS
    email = (document.get("coverage") or {}).get("email") or {}
    groups = record["groups"]
    checked, comparable, skipped, found = (record[key] for key in ("checked_sentences", "comparable_claims", "excluded_blocks", "different_claims"))
    if (not supported(method, email) or record["schema"] != (MAPPED_SCHEMA if mapped else SCHEMA)
            or record["scope"] != SCOPE or record["limitation"] != (MAPPED_LIMITATION if mapped else LIMITATION)
            or type(record["authorship_confidence_contribution"]) is not int or record["authorship_confidence_contribution"] != 0
            or not all(record[k] == source[k] for k in ("extracted_text_sha256", "source_map_sha256"))
            or not all(type(x) is int and 0 <= x <= source["chars"] for x in (checked, comparable, skipped, found))
            or record["status"] != ("different_values" if found else "checked" if comparable else "not_assessed")
            or found > comparable or comparable * 2 > checked or len(groups) != min(MAX_GROUPS, found)):
        raise ValueError("Invalid percentage consistency record")
    used = set()
    fields = OCCURRENCE_FIELDS | {"anchor"} if mapped else OCCURRENCE_FIELDS
    for group in groups:
        readings = set()
        locations = defaultdict(set)
        previous_end = 0
        if not 2 <= len(group) <= MAX_OCCURRENCES:
            raise ValueError("Invalid percentage evidence group")
        for occurrence in group:
            start, end, lo, hi = (occurrence[k] for k in ("start", "end", "quantity_start", "quantity_end"))
            hit = percentage(occurrence["text"])
            if (occurrence.keys() != fields or hit is None or not all(type(x) is int for x in (start, end, lo, hi))
                    or not previous_end <= start <= lo < hi <= end <= source["chars"]
                    or end - start != len(occurrence["text"]) or (lo - start, hi - start) != hit[1]
                    or (start, end) in used):
                raise ValueError("Percentage reading differs from its recorded evidence")
            if mapped:
                anchor = occurrence["anchor"]
                base_anchor = {key: value for key, value in anchor.items() if key != "body_paragraph"} if method == "python-docx" else anchor
                validate_block_anchor(base_anchor, end - start)
                if (anchor["kind"] != MAPPED_METHODS[method]
                        or method == "email" and anchor["mime_part"] != email.get("body_part")
                        or method == "python-docx" and (len(anchor["body_path"]) != 1
                            or type(anchor.get("body_paragraph")) is not int
                            or not 1 <= anchor["body_paragraph"] <= min(anchor["paragraph_index"], anchor["body_path"][0]) + 1)):
                    raise ValueError("Percentage reading has the wrong source location")
                identity = tuple(anchor.get(key) for key in ("page", "block", "paragraph_index", "source_start", "source_end"))
                locations[hit[2]].add(identity)
            used.add((start, end))
            previous_end = end
            readings.add((hit[0], hit[2]))
        if len({key for key, _ in readings}) != 1 or len(readings) < 2:
            raise ValueError("Percentages do not differ in one repeated sentence")
        if mapped and len(set().union(*locations.values())) < 2:
            raise ValueError("Different readings require separate source blocks")


def verify_source(document, text, mapping, sentences, *, meta=None):
    validate(document)
    if KEY in document and document[KEY] != build(document["source"], text, mapping, sentences, meta=meta):
        raise ValueError("Percentage evidence differs from the committed extracted text")
