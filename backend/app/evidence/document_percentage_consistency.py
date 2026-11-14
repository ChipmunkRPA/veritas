"""Locate complete repeated body sentences with different literal percentages."""

from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal

KEY = "percentage_consistency"
SCHEMA = "document-percentage-consistency/1"
SCOPE = "Compares complete body sentences with the same wording, punctuation and unit apart from one percentage value. The comparison ignores letter case and spacing."
LIMITATION = "Compare the surrounding context before interpreting a difference. The sentences could refer to different unstated dates, groups or denominators. This check carries zero AI-authorship weight."
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


def available_block(row):
    return row.get("kind") == "text" and bool(row.get("source_ranges"))


def build(source, text, mapping, sentences):
    records = defaultdict(list)
    skipped = 0
    candidates = set()
    if len(text) > MAX_TEXT_LENGTH:
        skipped = len(mapping)
    else:
        for index, row in enumerate(mapping):
            start, end = row["start"], row["end"]
            if (not available_block(row) or not 0 <= start < end <= len(text) or end - start > MAX_BLOCK_LENGTH
                    or ">" in text[start:end]):
                skipped += 1
                continue
            candidates.add(index)
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
        records[key].append((value, index, {"start": start, "end": end, "text": unit["text"],
                                           "quantity_start": start + lo, "quantity_end": start + hi}))
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
    return {"schema": SCHEMA, "status": "different_values" if results else "checked" if comparable else "not_assessed",
            "scope": SCOPE, "limitation": LIMITATION, "authorship_confidence_contribution": 0,
            "extracted_text_sha256": source["extracted_text_sha256"], "source_map_sha256": source["source_map_sha256"],
            "checked_sentences": checked, "comparable_claims": comparable, "excluded_blocks": skipped, "different_claims": len(results),
            "groups": results[:MAX_GROUPS]}


def validate(document):
    if KEY not in document:
        return
    record = document[KEY]
    source = document["source"]
    groups = record["groups"]
    checked, comparable, skipped, found = (record[key] for key in ("checked_sentences", "comparable_claims", "excluded_blocks", "different_claims"))
    if (source.get("method") != "plain" or record["schema"] != SCHEMA or record["scope"] != SCOPE or record["limitation"] != LIMITATION
            or type(record["authorship_confidence_contribution"]) is not int or record["authorship_confidence_contribution"] != 0
            or not all(record[k] == source[k] for k in ("extracted_text_sha256", "source_map_sha256"))
            or not all(type(x) is int and 0 <= x <= source["chars"] for x in (checked, comparable, skipped, found))
            or record["status"] != ("different_values" if found else "checked" if comparable else "not_assessed")
            or found > comparable or comparable * 2 > checked or len(groups) != min(MAX_GROUPS, found)):
        raise ValueError("Invalid percentage consistency record")
    used = set()
    for group in groups:
        readings = set()
        previous_end = 0
        if not 2 <= len(group) <= MAX_OCCURRENCES:
            raise ValueError("Invalid percentage evidence group")
        for occurrence in group:
            start, end, lo, hi = (occurrence[k] for k in ("start", "end", "quantity_start", "quantity_end"))
            hit = percentage(occurrence["text"])
            if (hit is None or not previous_end <= start <= lo < hi <= end <= source["chars"]
                    or end - start != len(occurrence["text"]) or (lo - start, hi - start) != hit[1]
                    or (start, end) in used):
                raise ValueError("Percentage reading differs from its recorded evidence")
            used.add((start, end))
            previous_end = end
            readings.add((hit[0], hit[2]))
        if len({key for key, _ in readings}) != 1 or len(readings) < 2:
            raise ValueError("Percentages do not differ in one repeated sentence")


def verify_source(document, text, mapping, sentences):
    validate(document)
    if KEY in document and document[KEY] != build(document["source"], text, mapping, sentences):
        raise ValueError("Percentage evidence differs from the committed extracted text")
