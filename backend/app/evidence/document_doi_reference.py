"""Compare mapped references with a pinned, offline bibliographic snapshot."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from importlib.resources import files

from app.core.canonical import FLOAT_DP, canonical_hash
from app.evidence import document_bibliography, document_pdf_visibility, document_word_visibility
from app.evidence.document_percentage_consistency import validate_value_location, value_location
from app.evidence.document_publication_ids import METHODS as IDENTIFIER_METHODS
from app.evidence.document_publication_ids import _anchor, _validate_anchor

KEY = "doi_reference_consistency"
POLICY = "doi_reference_review"
COMMITMENT = "doi_reference_consistency_sha256"
SCHEMA = "document-doi-reference/1"
SNAPSHOT_VERSION = "2026-11-18"
SNAPSHOT_SHA256 = "dec264757b798d6dbf0d02732293f1eae3c101c27b1ed904db08dcb6f4facd96"
SNAPSHOT = {"provider": "Crossref", "as_of": SNAPSHOT_VERSION, "sha256": SNAPSHOT_SHA256, "records": 364}
METHODS = {name: IDENTIFIER_METHODS[name] for name in ("plain", "python-docx", "odt", "pymupdf", "rtf")}
MAX_ENTRIES = 256
MAX_LOCATIONS = 8
SCOPE = ("Checks up to 256 mapped entries under a References, Bibliography, or Works Cited heading against a local Crossref snapshot. "
         "A matching title and first author identify the cited work.")
LIMITATION = ("The snapshot covers selected publications. A difference may reflect a citation error, incomplete registry metadata, "
              "or a different version. This check carries zero weight in the AI assessment.")
DOI = re.compile(r"(?<![\w.])10\.\d{4,9}/[A-Za-z0-9][A-Za-z0-9._;()/+:-]{1,149}", re.I)
WORD = re.compile(r"[^\W_]+", re.UNICODE)
SHA = re.compile(r"[a-f0-9]{64}")


def _normalized_words(text):
    for match in WORD.finditer(text):
        token = "".join(ch for ch in unicodedata.normalize("NFKD", match.group().casefold()) if not unicodedata.combining(ch))
        if token:
            yield token, match.start(), match.end()


def _words(text):
    return tuple(word for word, _, _ in _normalized_words(text))


def _registered_title(title):
    return 6 <= len(_words(title)) <= 65


@lru_cache(maxsize=1)
def _registry():
    data = files("app.evidence").joinpath("data", f"crossref-citations-{SNAPSHOT_VERSION}.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != SNAPSHOT_SHA256:
        raise ValueError("The pinned DOI registry bytes differ")
    snapshot = json.loads(data)
    records = snapshot["records"]
    if (snapshot["schema"] != "chipmunk.crossref-citations/1" or snapshot["as_of"] != SNAPSHOT_VERSION
            or len(records) != SNAPSHOT["records"]):
        raise ValueError("The pinned DOI registry shape differs")
    starts = defaultdict(list)
    for doi, row in records.items():
        if not row["family"] or not (family := _words(row["family"])):
            continue
        for title in row["titles"]:
            if 6 <= len(words := _words(title)) <= 65:
                starts[words[0]].append((doi, words, family, title))
    return records, starts


def _located_dois(text):
    for match in DOI.finditer(text):
        literal = match.group().rstrip(".,;:")
        yield literal, match.start(), match.start() + len(literal)


def _matches(text, index):
    located = list(_normalized_words(text))
    words = tuple(row[0] for row in located)
    matches = set()
    for start in range(1, min(48, len(words))):
        for doi, title, family, registered_title in index.get(words[start], ()):
            end, width = start + len(title), len(family)
            if words[start:end] == title and any(words[k:k + width] == family for k in range(start - width + 1)):
                matches.add((doi, located[start][1], located[end - 1][2], registered_title))
    return matches


def _distinct_titles(left, right):
    left_words, right_words = set(_words(left)), set(_words(right))
    return 20 * len(left_words & right_words) <= 9 * len(left_words | right_words)


def _compare(text, registry, index):
    if not 32 <= len(text) <= 640:
        return "unassessed", None
    hits = list(_located_dois(text))
    if len(hits) != 1:
        return "unassessed", None
    literal, start, end = hits[0]
    doi = literal.casefold()
    if doi not in registry:
        return "unassessed", None
    candidates = _matches(text[:start], index)
    matching = {item[0] for item in candidates}
    if matching == {doi}:
        return "consistent", {"text": literal, "start": start, "end": end}
    if len(matching) != 1:
        return "unassessed", None
    cited, title_start, title_end, cited_title = min(candidates)
    if cited in registry[doi]["related"] or doi in registry[cited]["related"]:
        return "unassessed", None
    cited_excerpt = text[title_start:title_end]
    titles = [value for value in registry[doi]["titles"] if _registered_title(value)]
    if not titles or not all(_distinct_titles(cited_excerpt, title) for title in titles):
        return "unassessed", None
    return "different_work", {
        "start": start, "end": end, "text": literal,
        "title_start": title_start, "title_end": title_end, "cited_title": cited_excerpt,
        "printed": {"doi": doi, "title": titles[0]}, "cited": {"doi": cited, "title": cited_title},
    }


def _outside_visibility(doc, kind):
    if kind == "pdf":
        visibility = doc.meta.get(document_pdf_visibility.KEY) or {}
        return {row["page"] for name in ("locations", "covered_locations", "replacement_differences")
                for row in visibility.get(name, ())}
    if kind == "docx":
        visibility = doc.meta.get(document_word_visibility.KEY) or {}
        return {tuple(row["body_path"]) for row in visibility.get("locations", ()) if len(row["body_path"]) == 1}
    return set()


def _visible_anchor(block, kind, meta, excluded):
    anchor = _anchor(block, kind, meta)
    if anchor is None:
        return None
    if kind == "pdf" and (block.get("extraction") == "ocr" or anchor["page"] in excluded):
        return None
    if kind == "docx" and (len(anchor["body_path"]) != 1 or tuple(anchor["body_path"]) in excluded):
        return None
    return anchor


def capture(doc):
    """Inspect the whole decoded document before any writing-only projection."""
    method = doc.meta.get("method")
    kind = METHODS.get(method)
    if not kind or (doc.word_review and doc.word_review.get("status") != "no_match"):
        return None
    if kind == "docx":
        visibility = doc.meta.get(document_word_visibility.KEY) or {}
        if visibility.get("hidden_runs", 0) > len(visibility.get("locations", ())):
            return None
    mapping, text = doc.source_map, doc.text
    missing = _outside_visibility(doc, kind)
    headings = [i for i, block in enumerate(mapping) if text[block["start"]:block["end"]].casefold().removesuffix(":")
                in document_bibliography.HEADINGS and _visible_anchor(block, kind, doc.meta, missing) is not None]
    if len(headings) != 1:
        return None
    entries = mapping[headings[0] + 1:headings[0] + 1 + MAX_ENTRIES]
    if not entries:
        return None
    registry, index = _registry()
    counts = {"consistent": 0, "different_work": 0, "unassessed": 0}
    rows = []
    for block in entries:
        counts["unassessed"] += 1
        anchor = _visible_anchor(block, kind, doc.meta, missing)
        if anchor is None:
            continue
        offset = block["start"]
        status, result = _compare(text[offset:block["end"]], registry, index)
        if result is None:
            continue
        if kind == "pdf":
            anchor["bbox"] = [round(value, FLOAT_DP) for value in anchor["bbox"]]
        for key in ("start", "end", "title_start", "title_end"):
            if key in result:
                result[key] += offset
        pieces = [("location", "start", "end")]
        if status == "different_work":
            pieces.append(("title_location", "title_start", "title_end"))
        try:
            for field, first, last in pieces:
                start, end = result[first], result[last]
                result[field] = value_location(block, start, end)
                validate_value_location(result[field], anchor, end - start)
        except ValueError:
            continue
        counts["unassessed"] -= 1
        counts[status] += 1
        if status == "different_work" and len(rows) < MAX_LOCATIONS:
            rows.append({"anchor": anchor, **result})
    return {"schema": SCHEMA, "snapshot": SNAPSHOT, "scope": SCOPE, "limitation": LIMITATION, "affects_overall_confidence": False,
            "source": {"sha256": doc.meta["original_sha256"], "method": method, "chars": len(text),
                       "extracted_text_sha256": doc.meta["extracted_text_sha256"]},
            "references": len(entries), **counts, "locations": rows}


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(COMMITMENT)


def validate(document):
    source = document["source"]
    record, expected = document.get(KEY), source.get(COMMITMENT)
    if KEY not in document and COMMITMENT not in source:
        return
    fields = {"schema", "snapshot", "scope", "limitation", "affects_overall_confidence", "source",
              "references", "consistent", "different_work", "unassessed", "locations"}
    if (not isinstance(record, dict) or record.keys() != fields or canonical_hash(record) != expected or record["schema"] != SCHEMA
            or record["snapshot"] != SNAPSHOT or record["affects_overall_confidence"] is not False
            or record["scope"] != SCOPE or record["limitation"] != LIMITATION):
        raise ValueError("DOI reference evidence differs from the published inspection")
    full = record["source"]
    kind = METHODS.get(full["method"])
    counts = (record["references"], record["consistent"], record["different_work"], record["unassessed"])
    bibliography = document.get("coverage", {}).get(document_bibliography.KEY)
    if (full.keys() != {"method", "sha256", "chars", "extracted_text_sha256"} or not kind
            or full["method"] != source["method"] or full["sha256"] != source["sha256"]
            or type(full["chars"]) is not int or full["chars"] != (bibliography["section_end"] if bibliography else source["chars"])
            or (not bibliography and full["extracted_text_sha256"] != source["extracted_text_sha256"])
            or not SHA.fullmatch(full["extracted_text_sha256"])
            or any(type(value) is not int or value < 0 for value in counts)
            or not 0 < counts[0] <= MAX_ENTRIES or sum(counts[1:]) != counts[0]
            or not isinstance(record["locations"], list) or len(record["locations"]) != min(MAX_LOCATIONS, counts[2])):
        raise ValueError("DOI reference evidence has invalid source or coverage")
    cursor = 0
    for row in record["locations"]:
        if row.keys() != {"start", "end", "title_start", "title_end", "text", "cited_title",
                          "anchor", "location", "title_location", "printed", "cited"}:
            raise ValueError("Invalid DOI reference location")
        start, end, title_start, title_end = (row[key] for key in ("start", "end", "title_start", "title_end"))
        printed, cited = row["printed"], row["cited"]
        if (any(type(value) is not int for value in (start, end, title_start, title_end))
                or not cursor <= title_start < title_end <= start < end <= full["chars"]
                or row["anchor"]["kind"] != kind or len(row["text"]) != end - start
                or len(row["cited_title"]) != title_end - title_start or not DOI.fullmatch(row["text"])
                or printed.keys() != {"doi", "title"} or cited.keys() != {"doi", "title"}
                or row["text"].casefold() != printed["doi"] or not DOI.fullmatch(cited["doi"]) or printed["doi"] == cited["doi"]
                or _words(row["cited_title"]) != _words(cited["title"]) or not _registered_title(cited["title"])
                or not _registered_title(printed["title"]) or not _distinct_titles(cited["title"], printed["title"])):
            raise ValueError("DOI discrepancy differs from the retained title or coordinates")
        _validate_anchor(row["anchor"], end - start)
        validate_value_location(row["location"], row["anchor"], end - start)
        validate_value_location(row["title_location"], row["anchor"], title_end - title_start)
        cursor = end


def verify_record(document, recorded, *, commitment):
    validate(document)
    if document["source"].get(COMMITMENT) != commitment or document.get(KEY) != recorded:
        raise ValueError("DOI reference evidence differs from the frozen extraction")


def verify_original(document, doc):
    if KEY in document:
        verify_record(document, capture(doc), commitment=document["source"].get(COMMITMENT))
