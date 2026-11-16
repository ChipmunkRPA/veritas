# SPDX-License-Identifier: MIT
"""Pair printed figures between drafts using unique surrounding wording."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

SCHEMA = "chipmunk.draft-value-comparison/1"
SCOPE = (
    "Compares figures and dates in uniquely matched body text. Equivalent number formatting is ignored. "
    "Check the source context to tell whether a change is a correction, a new fact or a mistake."
)
MAX_RESULTS = 8
MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
NUMERAL = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
FIGURE = re.compile(
    r"(?<![\w.,%$£€:/+−–-])(?:"
    r"(?P<iso>[0-9]{4}-[0-9]{2}-[0-9]{2})|"
    rf"(?P<month>{'|'.join(MONTHS)})[ \t]+(?P<day>[0-9]{{1,2}}),?[ \t]+(?P<year>[0-9]{{4}})|"
    rf"(?P<currency>[$£€][ \t]*)?(?P<number>[+−-]?{NUMERAL})"
    r"(?:[ \t]+(?P<scale>thousand|million|billion)(?!\w))?"
    r"(?:[ \t]*(?P<percentage>%|percent(?:age points?)?|per cent)(?!\w))?"
    r")(?![\w%$£€:/+−–-]|[.,][0-9])",
    re.I,
)
WORD = r"[^\W\d_]+(?:['’-][^\W\d_]+)*"
LEFT = re.compile(rf"(?P<words>{WORD}(?:\s+{WORD}){{0,4}})\s*$")
RIGHT = re.compile(rf"\s*(?P<words>{WORD}(?:\s+{WORD}){{0,4}})")
LEFT_BOUNDARY = re.compile(r"(?:^|[.!?])\s*$")
RIGHT_BOUNDARY = re.compile(r"\s*(?:[.!?](?![0-9])|$)")
QUOTED_LINE = re.compile(r"(?m)^\s*>")
SCALE = {"thousand": 3, "million": 6, "billion": 9}


@dataclass(frozen=True)
class Reading:
    kind: str
    value: Decimal | date
    span: tuple[int, int]
    context: tuple[int, int]


def _literal(match: re.Match) -> tuple[str, str, Decimal | date] | None:
    if iso := match["iso"]:
        try:
            return "date", "", date.fromisoformat(iso)
        except ValueError:
            return None
    if month := match["month"]:
        try:
            return "date", "", date(int(match["year"]), MONTHS.index(month.casefold()) + 1, int(match["day"]))
        except ValueError:
            return None
    number = match["number"].replace(",", "").replace("−", "-")
    currency, percent = match["currency"], match["percentage"]
    if len(number) > 19 or (currency and percent):
        return None
    value = Decimal(number).scaleb(SCALE.get((match["scale"] or "").casefold(), 0))
    if currency:
        return "currency", currency.strip(), value
    if percent:
        kind = "percentage_points" if "point" in percent.casefold() else "percentage"
        return kind, "", value
    return "number", "", value


def readings(text, blocks, *, words_per_side: int = 3):
    grouped = defaultdict(list)
    for start, end in blocks:
        block = text[start:end]
        if QUOTED_LINE.search(block):
            continue
        for match in FIGURE.finditer(block):
            literal = _literal(match)
            if literal is None:
                continue
            a, b = match.span()
            left, right = LEFT.search(block, max(0, a - 160), a), RIGHT.match(block, b, min(len(block), b + 160))
            left_words = left["words"].casefold().split() if left else []
            right_words = right["words"].casefold().split() if right else []
            first, last = left.start("words") if left else a, right.end("words") if right else b
            if len(left_words) >= words_per_side and len(right_words) >= words_per_side:
                key = (tuple(left_words[-words_per_side:]), tuple(right_words[:words_per_side:]), "within")
            elif len(right_words) >= 4 and LEFT_BOUNDARY.search(block, max(0, first - 160), first):
                key = (tuple(left_words), tuple(right_words[:4]), "starts")
            elif len(left_words) >= 4 and RIGHT_BOUNDARY.match(block, last):
                key = (tuple(left_words[-4:]), tuple(right_words), "ends")
            else:
                continue
            kind, unit, value = literal
            grouped[(kind, unit, *key)].append(Reading(kind, value, (start + a, start + b), (start + first, start + last)))
    return grouped


def compare(earlier: dict, later: dict):
    """Return distinct matches and require a unique occurrence in each supplied draft."""
    matched = [(first[0], later[key][0]) for key, first in earlier.items()
               if len(first) == 1 and len(later.get(key, ())) == 1]
    matched.sort(key=lambda pair: pair[1].span)
    return matched
