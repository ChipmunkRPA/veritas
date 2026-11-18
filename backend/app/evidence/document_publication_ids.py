# SPDX-License-Identifier: Apache-2.0
"""Match clearly labeled publication identifiers and compare their check digits."""

from __future__ import annotations

import re

from app.core.canonical import FLOAT_DP
from app.evidence import document_native_locations as native_text
from app.evidence import document_pdf_visibility, document_scope
from app.evidence.document_percentage_consistency import validate_value_location, value_location
from app.evidence.document_repeat import block_anchor, validate_block_anchor

KEY = "publication_identifier_consistency"
SCHEMA = "document-publication-identifiers/1"
METHODS = {"plain": "text", "email": "email", "python-docx": "docx", "pymupdf": "pdf", "html": "decoded_html",
           "odt": "odt_paragraph", "rtf": "rtf_paragraph"}
SCOPE = ("Checks the final check digit of clearly labeled ISBN-10, ISBN-13 and ISSN values in mapped text. "
         "It uses the preceding digits from the submitted file.")
LIMITATION = ("A disagreement can come from a typing or transcription error. Matching digits alone cannot confirm assignment, "
              "publication or authorship. This check carries zero weight in the AI assessment.")
MAX_LOCATIONS = 8

_SEP = r"[-\u2010\u2011\u2013 ]?"
_LABEL = re.compile(r"(?<![\w-])(?P<scheme>ISBN(?:[- ](?:10|13)(?![0-9]))?|[ep]?ISSN(?:-L)?)(?: *#? *: *| +|[-\u2010\u2011\u2013])", re.I)
_BOUND = r"(?![\w/-]|[-\u2010\u2011\u2013][0-9Xx])"
_FORMATS = {
    "ISBN-13": re.compile(rf"(?:9{_SEP}7{_SEP}[89]{_SEP})(?:[0-9]{_SEP}){{9}}[0-9]{_BOUND}"),
    "ISBN-10": re.compile(rf"(?:[0-9]{_SEP}){{9}}[0-9Xx]{_BOUND}"),
    "ISSN": re.compile(rf"[0-9]{{4}}(?:[-\u2010\u2011\u2013]| )[0-9]{{3}}[0-9Xx]{_BOUND}|[0-9]{{7}}[0-9Xx]{_BOUND}"),
}
_EXPLICITLY_WRONG = re.compile(r"\b(?:invalid|incorrect|misprinted|cancell?ed|erroneous) *\(? *$", re.I)
_SUFFIX_WRONG = re.compile(r" *\((?:invalid|incorrect|misprinted|cancell?ed|erroneous)\)", re.I)
_FOLLOWING_NUMBER = re.compile(r" [0-9Xx]")
_FOLLOWING_ISBN_LABEL = re.compile(r" (?:10|13)[- ]digit ISBN *:", re.I)
_NORMAL = re.compile(r"[-\u2010\u2011\u2013 ]")


def _digits(text, kind):
    number = _NORMAL.sub("", text).upper()
    # The 979-0 prefix belongs to printed music identifiers.
    if kind == "ISBN-13" and number.startswith("9790"):
        return None
    return number


def _expected(number):
    digits = map(int, number[:-1])
    size = len(number)
    if size == 13:
        return str(-sum((1 + 2 * (i % 2)) * digit for i, digit in enumerate(digits)) % 10)
    result = -sum((size - i) * digit for i, digit in enumerate(digits)) % 11
    return "X" if result == 10 else str(result)


def readings(text, start, end):
    for label in _LABEL.finditer(text, start, end):
        if _EXPLICITLY_WRONG.search(text[max(start, label.start() - 24):label.start()]):
            continue
        scheme = label["scheme"].upper()
        kinds = ("ISBN-13", "ISBN-10") if scheme == "ISBN" else (scheme.replace(" ", "-"),) if scheme.startswith("ISBN") else ("ISSN",)
        for kind in kinds:
            match = _FORMATS[kind].match(text, label.end(), end)
            if match is None:
                continue
            if (_FOLLOWING_NUMBER.match(text, match.end(), end)
                    and not _FOLLOWING_ISBN_LABEL.match(text, match.end(), end)):
                break
            number = _digits(match.group(), kind)
            if number is None or _SUFFIX_WRONG.match(text, match.end(), end):
                break
            yield kind, match, number[-1], _expected(number)
            break


def _anchor(block, kind, meta):
    if block.get("kind") == kind:
        return block_anchor(block)
    if block.get("kind") != "markup" or not block.get("source_ranges"):
        return None
    if kind == "decoded_html" and document_scope.valid_html_article_scope(meta.get(document_scope.HTML_ARTICLE_KEY)):
        start, end = meta[document_scope.HTML_ARTICLE_KEY]["scope"]
        return {"kind": kind, "source_start": start, "source_end": end}
    if (kind in native_text.PARAGRAPH_COORDINATES.values()
            and meta.get(native_text.PARAGRAPH_MAP_KEY) == native_text.PARAGRAPH_MAP):
        return {"kind": kind, "paragraph_index": block["paragraph_index"]}
    return None


def _validate_anchor(anchor, width):
    kind = anchor["kind"]
    if kind == "decoded_html":
        validate_block_anchor({**anchor, "kind": "text"}, width)
    elif kind in native_text.PARAGRAPH_COORDINATES.values():
        if (set(anchor) != {"kind", "paragraph_index"} or type(anchor["paragraph_index"]) is not int
                or anchor["paragraph_index"] < 0):
            raise ValueError("Invalid publication identifier paragraph")
    else:
        validate_block_anchor(anchor, width)


def build(source, text, mapping, meta):
    kind = METHODS.get(source["method"])
    if kind is None:
        return None
    ordinary_word = set(meta.get("percentage_body_paragraphs", ()))
    visibility = meta.get(document_pdf_visibility.KEY) or {}
    unsuitable_pages = {row["page"] for key in ("locations", "covered_locations", "replacement_differences")
                        for row in visibility.get(key, ())}
    checked = disagreements = 0
    locations = []
    for block in mapping:
        if ((anchor := _anchor(block, kind, meta)) is None
                or (kind == "docx" and (len(anchor["body_path"]) != 1 or anchor["body_path"][0] not in ordinary_word))
                or (kind == "pdf" and anchor["page"] in unsuitable_pages)):
            continue
        if kind == "pdf":
            anchor["bbox"] = [round(value, FLOAT_DP) for value in anchor["bbox"]]
        for scheme, match, written, expected in readings(text, block["start"], block["end"]):
            start, end = match.span()
            native = value_location(block, start, end)
            try:
                validate_value_location(native, anchor, end - start)
            except ValueError:
                continue
            checked += 1
            if written != expected:
                disagreements += 1
                if len(locations) < MAX_LOCATIONS:
                    locations.append({"kind": scheme, "start": start, "end": end, "text": match.group(),
                                      "written": written, "expected": expected, "anchor": anchor, "location": native})
    if not checked:
        return None
    return {"schema": SCHEMA, "scope": SCOPE, "limitation": LIMITATION, "affects_overall_confidence": False,
            "checked": checked, "disagreements": disagreements, "locations": locations}


def validate(document):
    if KEY not in document:
        return
    source, record = document["source"], document[KEY]
    if (set(record) != {"schema", "scope", "limitation", "affects_overall_confidence", "checked", "disagreements", "locations"}
            or record["schema"] != SCHEMA or record["scope"] != SCOPE or record["limitation"] != LIMITATION
            or record["affects_overall_confidence"] is not False or source["method"] not in METHODS
            or type(record["checked"]) is not int or type(record["disagreements"]) is not int
            or not 0 < record["checked"] <= source["chars"] or not 0 <= record["disagreements"] <= record["checked"]
            or not isinstance(record["locations"], list) or len(record["locations"]) != min(MAX_LOCATIONS, record["disagreements"])):
        raise ValueError("Invalid publication identifier coverage")
    cursor = 0
    for row in record["locations"]:
        if set(row) != {"kind", "start", "end", "text", "written", "expected", "anchor", "location"}:
            raise ValueError("Invalid publication identifier location")
        kind, start, end, text, anchor = (row[key] for key in ("kind", "start", "end", "text", "anchor"))
        if (type(start) is not int or type(end) is not int or not cursor <= start < end <= source["chars"]
                or not isinstance(text, str) or end - start != len(text) or kind not in _FORMATS
                or not _FORMATS[kind].fullmatch(text) or anchor["kind"] != METHODS[source["method"]]):
            raise ValueError("Publication identifier differs from its source coordinates")
        number = _digits(text, kind)
        if number is None or number[-1] != row["written"] or _expected(number) != row["expected"] or row["written"] == row["expected"]:
            raise ValueError("Publication identifier disagrees with its computed check digit")
        _validate_anchor(anchor, end - start)
        validate_value_location(row["location"], anchor, end - start)
        cursor = end


def verify_source(document, text, mapping, meta):
    if KEY in document and build(document["source"], text, mapping, meta) != document[KEY]:
        raise ValueError("Publication identifiers differ from the verified source text")
