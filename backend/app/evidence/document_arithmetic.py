# SPDX-License-Identifier: Apache-2.0
"""Check explicit numeric equations in reliably mapped document text."""

from __future__ import annotations

import re
from decimal import Decimal, localcontext
from heapq import merge

from app.core.canonical import FLOAT_DP
from app.evidence import document_length
from app.evidence.document_percentage_consistency import validate_value_location, value_location
from app.evidence.document_repeat import block_anchor, validate_block_anchor

KEY = "arithmetic_consistency"
SCHEMA = "document-arithmetic-consistency/1"
LENGTH_SCHEMA = "document-arithmetic-consistency/2"
METHODS = {"plain": "text", "email": "email", "python-docx": "docx", "pymupdf": "pdf"}
MAXIMUM = 6
SCOPE = "Checks complete two-number equations in mapped text blocks or parentheses. Supports addition, subtraction and multiplication, with matching currency symbols for addition and subtraction."
LENGTH_SCOPE = (SCOPE + " Checks explicit length conversions at the stated result precision using international units. "
                "Approximate, survey-foot and nautical-mile contexts are outside the conversion check.")
LIMITATION = "A typing, transcription or calculation error can cause a disagreement. This check carries zero weight in the AI assessment."
LENGTH_LIMITATION = ("Rounding, nominal dimensions, typing, transcription or calculation can cause disagreements. "
                     "This check carries zero weight in the AI assessment.")
_NUMBER = r"(?:[0-9]{1,3}(?:,[0-9]{3}){1,3}|[0-9]{1,12})(?:\.[0-9]{1,6})?"
_VALUE = rf"[+\-−]?[$£€]?{_NUMBER}"
_EXPRESSION = rf"(?P<left>{_VALUE}) *(?P<operator>[+\-−*×]) *(?P<right>{_VALUE}) *= *(?P<stated>{_VALUE})"
_EQUATION = re.compile(_EXPRESSION)
_PARENTHESES = re.compile(rf"\( *(?P<expression>{_EXPRESSION}) *\)")


def _operand(token):
    number = token.lstrip("+-−")
    currency = number[0] if number[0] in "$£€" else ""
    sign = "-" if token[0] in "-−" else ""
    return Decimal(sign + number.removeprefix(currency).replace(",", "")), currency


def _reading(match):
    left, right, stated = (_operand(match[name]) for name in ("left", "right", "stated"))
    operator = match["operator"]
    currency = left[1]
    if right[1] != currency or stated[1] != currency or (currency and operator in "*×"):
        return None
    with localcontext() as context:
        context.prec = 40  # The longest admitted product has no more than 36 significant digits.
        if operator == "+":
            calculated = left[0] + right[0]
        elif operator in "-−":
            calculated = left[0] - right[0]
        else:
            calculated = left[0] * right[0]
        digits = format(abs(calculated), ",f" if currency else "f")
    if calculated == stated[0]:
        return {}
    if "." in digits:
        digits = digits.rstrip("0").rstrip(".")
    sign = "-" if calculated < 0 else ""
    return {"stated": match["stated"], "calculated": sign + currency + digits}


def _arithmetic_readings(text, start, end):
    whole = _EQUATION.fullmatch(text, start, end)
    group = 0 if whole else "expression"
    for match in (whole,) if whole else _PARENTHESES.finditer(text, start, end):
        if (issue := _reading(match)) is not None:
            yield match.span(group), match.group(group), issue, False


def build(source, text, mapping, *, include_lengths=False):
    kind = METHODS[source["method"]]
    checked = disagreements = 0
    lengths = length_disagreements = 0
    mixed_definitions = include_lengths and bool(document_length.SPECIAL_UNITS.search(text))
    locations = []
    for block in mapping:
        if block.get("kind") != kind:
            continue
        anchor = block_anchor(block)
        if anchor is None:
            continue
        start, end = block["start"], block["end"]
        if not 0 <= start < end <= len(text):
            raise ValueError("Arithmetic evidence needs mapped text positions")
        if kind == "pdf":
            anchor["bbox"] = [round(value, FLOAT_DP) for value in anchor["bbox"]]
        measurements = _arithmetic_readings(text, start, end)
        if include_lengths:
            measurements = merge(measurements, document_length.readings(text, start, end, mixed_definitions=mixed_definitions),
                                 key=lambda reading: reading[0])
        for (low, high), expression, issue, is_length in measurements:
            native = value_location(block, low, high)
            try:
                validate_value_location(native, anchor, high - low)
            except ValueError:
                continue  # Keep only expressions whose source coordinates can be represented.
            checked += 1
            lengths += is_length
            if issue:
                disagreements += 1
                length_disagreements += is_length
                if len(locations) < MAXIMUM:
                    locations.append({"start": low, "end": high, "text": expression, "anchor": anchor,
                                      "location": native, **issue})
    result = {"schema": SCHEMA, "affects_overall_confidence": False, "scope": SCOPE, "limitation": LIMITATION,
              "checked_equations": checked, "disagreements": disagreements, "locations": locations}
    if lengths:
        result.update(schema=LENGTH_SCHEMA, scope=LENGTH_SCOPE, limitation=LENGTH_LIMITATION,
                      length_conversions_checked=lengths, length_conversion_disagreements=length_disagreements)
    return result


def validate(document):
    if KEY not in document:
        return
    record, source = document[KEY], document["source"]
    checked, disagreements, locations = (record[key] for key in ("checked_equations", "disagreements", "locations"))
    kind = METHODS.get(source["method"])
    include_lengths = record["schema"] == LENGTH_SCHEMA
    scope, limitation = (LENGTH_SCOPE, LENGTH_LIMITATION) if include_lengths else (SCOPE, LIMITATION)
    fields = {"schema", "affects_overall_confidence", "scope", "limitation", "checked_equations", "disagreements", "locations"}
    if include_lengths:
        fields.update(("length_conversions_checked", "length_conversion_disagreements"))
    if (set(record) != fields or record["schema"] not in {SCHEMA, LENGTH_SCHEMA}
            or record["scope"] != scope or record["limitation"] != limitation or record["affects_overall_confidence"] is not False
            or kind is None or type(checked) is not int or type(disagreements) is not int
            or not 0 <= disagreements <= checked <= source["chars"]
            or not isinstance(locations, list) or len(locations) != min(MAXIMUM, disagreements)):
        raise ValueError("Invalid arithmetic evidence coverage")
    lengths, length_disagreements = (record.get(key, 0) for key in ("length_conversions_checked", "length_conversion_disagreements"))
    if (include_lengths and (type(lengths) is not int or type(length_disagreements) is not int
            or not 0 < lengths <= checked or not 0 <= length_disagreements <= min(lengths, disagreements)
            or disagreements - length_disagreements > checked - lengths)):
        raise ValueError("Invalid length conversion coverage")
    previous = 0
    retained_lengths = 0
    for row in locations:
        low, high, text, anchor = (row[key] for key in ("start", "end", "text", "anchor"))
        match = _EQUATION.fullmatch(text)
        expected = _reading(match) if match else document_length.conversion(text) if include_lengths else None
        if (not expected or set(row) != {"start", "end", "text", "anchor", "location", *expected}
                or any(row[key] != value for key, value in expected.items()) or anchor["kind"] != kind
                or type(low) is not int or type(high) is not int or not previous <= low < high <= source["chars"]
                or high - low != len(text)):
            raise ValueError("An arithmetic disagreement differs from its source expression")
        validate_block_anchor(anchor, high - low)
        validate_value_location(row["location"], anchor, high - low)
        retained_lengths += match is None
        previous = high
    if retained_lengths > length_disagreements or len(locations) - retained_lengths > disagreements - length_disagreements:
        raise ValueError("Arithmetic evidence has inconsistent disagreement counts")


def verify_source(document, text, mapping):
    if KEY in document and build(document["source"], text, mapping,
                                 include_lengths=document[KEY]["schema"] == LENGTH_SCHEMA) != document[KEY]:
        raise ValueError("Arithmetic evidence differs from the verified source text")
