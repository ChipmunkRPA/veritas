# SPDX-License-Identifier: Apache-2.0
"""Check explicit length equalities at the precision displayed in a document."""

from __future__ import annotations

import re
from fractions import Fraction

# Exact international units in metres. NIST Handbook 133, Appendix E.
_FACTORS = {
    "mm": "0.001", "cm": "0.01", "m": "1", "km": "1000",
    "in": "0.0254", "ft": "0.3048", "yd": "0.9144", "mi": "1609.344",
}
_ALIASES = {
    "mm": ("millimeter", "millimetre"), "cm": ("centimeter", "centimetre"), "m": ("meter", "metre"),
    "km": ("kilometer", "kilometre"), "in": ("inch", "inches"), "ft": ("foot", "feet"),
    "yd": ("yard",), "mi": ("mile",),
}
_WORDS = {word: unit for unit, words in _ALIASES.items() for word in words}
_WORDS |= {word + "s": unit for word, unit in _WORDS.items() if word not in {"inch", "inches", "foot", "feet"}}
_METRES = {name: Fraction(_FACTORS[name]) for name in _FACTORS} | {name: Fraction(_FACTORS[unit]) for name, unit in _WORDS.items()}
_UNIT = rf"(?:{'|'.join(_FACTORS)}|(?i:{'|'.join(_WORDS)}))"
_NUMBER = r"(?:[0-9]{1,3}(?:,[0-9]{3}){1,3}|[0-9]{1,12})(?:\.[0-9]{1,12})?"
_LENGTH = (rf"(?P<amount>{_NUMBER}) +(?P<from_unit>{_UNIT})(?: *= *| +(?i:equals) +)"
           rf"(?P<stated>(?P<digits>{_NUMBER}) +(?P<to_unit>{_UNIT}))")
_CONVERSION = re.compile(_LENGTH)
_PARENTHESIZED = re.compile(rf"\( *(?P<conversion>{_LENGTH}) *\)")
_QUALIFIER = re.compile(r"[≈~±]|\b(?:approx(?:imat\w+)?|about|roughly|around|nominal\w*|tolerance\w*)\b", re.I)
SPECIAL_UNITS = re.compile(r"\b(?:survey +(?:foot|feet|miles?)|(?:international +)?nautical +miles?|usft)\b", re.I)
_SIMPLE = (_FACTORS.keys() - {"ft", "mi"}) | {word for word, unit in _WORDS.items() if unit not in {"ft", "mi"}}


def conversion(text):
    """Return None for unsupported units and an empty result for compatible rounding."""
    row = _CONVERSION.fullmatch(text)
    if row is None:
        return None
    left, right = (_METRES[row[key].lower()] for key in ("from_unit", "to_unit"))
    if left == right:
        return None
    amount, stated = (Fraction(row[key].replace(",", "")) for key in ("amount", "digits"))
    decimals = len(row["digits"].partition(".")[2])
    resolution = 10 ** decimals
    expected = amount * left / right
    if abs(expected - stated) * resolution <= Fraction(1, 2):
        return {}
    digits = str(round(expected * resolution))
    if decimals:
        digits = digits.rjust(decimals + 1, "0")
        digits = f"{digits[:-decimals]}.{digits[-decimals:]}"
    return {"stated": row["stated"], "calculated": f"{digits} {row['to_unit']}"}


def readings(text, start, end, *, mixed_definitions):
    if _QUALIFIER.search(text, start, end):
        return
    whole = _CONVERSION.fullmatch(text, start, end)
    group = 0 if whole else "conversion"
    for row in (whole,) if whole else _PARENTHESIZED.finditer(text, start, end):
        if mixed_definitions and any(row[key].lower() not in _SIMPLE for key in ("from_unit", "to_unit")):
            continue
        if (issue := conversion(row[group])) is not None:
            yield row.span(group), row[group], issue, True
