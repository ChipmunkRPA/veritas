# SPDX-License-Identifier: Apache-2.0
"""Locate explicit weekday and calendar-date disagreements in mapped text."""

from __future__ import annotations

import re
from datetime import date

from app.core.canonical import FLOAT_DP
from app.evidence.document_percentage_consistency import value_location
from app.evidence.document_repeat import block_anchor, validate_block_anchor

KEY = "calendar_consistency"
SCHEMA = "document-calendar-consistency/1"
METHODS = {"plain": "text", "email": "email", "python-docx": "docx", "pymupdf": "pdf"}
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
MONTH_INDEX = {name.casefold(): index for index, name in enumerate(MONTHS, 1)}
MAXIMUM = 6
SCOPE = "Checks explicit English weekdays paired with a day, a spelled-out month and a four-digit year from 1900 to 2099."
LIMITATION = "An editing mistake or data-entry error can cause a disagreement. This observation carries zero weight in the AI assessment."
_MONTH = "|".join(MONTHS)
_ORDINAL = r"(?:st|nd|rd|th)?"
_DATE = re.compile(
    rf"(?<!\w)(?P<weekday>{'|'.join(WEEKDAYS)})(?:\s*,\s*|\s+)"
    rf"(?:(?P<month_first>{_MONTH})\s+(?P<day_last>[0-9]{{1,2}}){_ORDINAL}(?:\s*,\s*|\s+)"
    rf"|(?P<day_first>[0-9]{{1,2}}){_ORDINAL}\s+(?P<month_last>{_MONTH})\s+)"
    r"(?P<year>19[0-9]{2}|20[0-9]{2})(?![\w/]|[.,][0-9])", re.IGNORECASE,
)


def _reading(match):
    weekday = match["weekday"].capitalize()
    month = MONTH_INDEX[(match["month_first"] or match["month_last"]).casefold()]
    day = int(match["day_last"] or match["day_first"])
    year = int(match["year"])
    try:
        candidate = date(year, month, day)
    except ValueError:
        return {"reason": "invalid_date", "stated_weekday": weekday, "expected_weekday": None, "iso_date": None}
    expected = WEEKDAYS[candidate.weekday()]
    if weekday == expected:
        return None
    return {"reason": "weekday_mismatch", "stated_weekday": weekday, "expected_weekday": expected, "iso_date": candidate.isoformat()}


def build(source, text, mapping):
    kind = METHODS[source["method"]]
    blocks = checked = mismatches = 0
    locations = []
    for block in mapping:
        if block.get("kind") != kind or block.get("quote_role") in {"quoted", "signature"}:
            continue
        anchor = block_anchor(block)
        if anchor is None:
            continue
        if kind == "pdf":
            anchor["bbox"] = [round(value, FLOAT_DP) for value in anchor["bbox"]]
        start, end = block["start"], block["end"]
        if not 0 <= start < end <= len(text):
            raise ValueError("Calendar evidence needs mapped text positions")
        blocks += 1
        for match in _DATE.finditer(text, start, end):
            lo, hi = match.span()
            evidence = value_location(block, lo, hi)
            if evidence is None:
                continue
            checked += 1
            if issue := _reading(match):
                mismatches += 1
                if len(locations) < MAXIMUM:
                    locations.append({"start": lo, "end": hi, "text": match.group(), "anchor": anchor,
                                      "location": evidence, **issue})
    return {"schema": SCHEMA, "affects_overall_confidence": False, "scope": SCOPE, "limitation": LIMITATION,
            "mapped_blocks": blocks, "checked_dates": checked, "disagreements": mismatches, "locations": locations}


def _validate_location(row):
    anchor, location = row["anchor"], row["location"]
    validate_block_anchor(anchor, row["end"] - row["start"])
    if anchor["kind"] == "pdf":
        x0, y0, x1, y1 = anchor["bbox"]
        if set(location) != {"boxes"} or not isinstance(location["boxes"], list) or not location["boxes"]:
            raise ValueError("Calendar evidence needs the aligned PDF date boxes")
        for left, top, right, bottom in location["boxes"]:
            if not (x0 - .01 <= left < right <= x1 + .01 and y0 - .01 <= top < bottom <= y1 + .01):
                raise ValueError("Calendar date boxes exceed their mapped PDF block")
    else:
        if set(location) != {"ranges"} or not isinstance(location["ranges"], list) or not location["ranges"]:
            raise ValueError("Calendar evidence needs the original date positions")
        previous_end = anchor.get("source_start", 0)
        for start, end in location["ranges"]:
            if not (type(start) is type(end) is int and previous_end <= start < end <= anchor.get("source_end", end)):
                raise ValueError("Calendar dates exceed their original text positions")
            previous_end = end


def validate(document):
    if KEY not in document:
        return
    record = document[KEY]
    source = document["source"]
    blocks, checked, disagreements = (record[key] for key in ("mapped_blocks", "checked_dates", "disagreements"))
    locations = record["locations"]
    kind = METHODS.get(source["method"])
    if (set(record) != {"schema", "affects_overall_confidence", "scope", "limitation", "mapped_blocks",
                       "checked_dates", "disagreements", "locations"} or record["schema"] != SCHEMA
            or record["scope"] != SCOPE or record["limitation"] != LIMITATION or record["affects_overall_confidence"] is not False
            or kind is None or not all(type(value) is int for value in (blocks, checked, disagreements))
            or not 0 <= disagreements <= checked <= source["chars"] or not 0 <= blocks <= source["chars"]
            or checked and not blocks or not isinstance(locations, list) or len(locations) != min(MAXIMUM, disagreements)):
        raise ValueError("Invalid calendar evidence coverage")
    previous = 0
    for row in locations:
        a, b, text = (row[key] for key in ("start", "end", "text"))
        match = _DATE.fullmatch(text)
        issue = _reading(match) if match else None
        if (issue is None or set(row) != {"start", "end", "text", "anchor", "location", *issue}
                or any(row[key] != value for key, value in issue.items()) or row["anchor"]["kind"] != kind
                or type(a) is not int or type(b) is not int or not previous <= a < b <= source["chars"] or b - a != len(text)):
            raise ValueError("A calendar disagreement differs from its exact source phrase")
        _validate_location(row)
        previous = b


def verify_source(document, text, mapping):
    if KEY in document and build(document["source"], text, mapping) != document[KEY]:
        raise ValueError("Calendar date evidence differs from the verified source text")
