"""Reconcile a printed euro summary with an explicit VAT basis."""

from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image, ImageOps

from app.evidence import image_printed_totals_binding as binding
from app.evidence.image_printed_totals import ocr, transcript, union

_AMOUNT = binding.EURO_MONEY
_MONEY = re.compile(
    rf"(?:(?P<prefix_sign>-)\s*)?(?:EUR|€)\s*(?P<prefix>{_AMOUNT})(?=$|\s|[.!])|"
    rf"(?P<suffix_sign>-)?\s*(?P<suffix>{_AMOUNT})\s*(?:EUR|€)(?=$|\s|[.!])",
    re.I,
)
_CHARACTERS = str.maketrans({"’": "'", "‘": "'", "−": "-", "–": "-", "—": "-"})
_PERCENT = r"[0-9]{1,2}(?:[.,][0-9]{1,2})?\s*%"
_INCLUDE = re.compile(rf"all prices include {_PERCENT} vat\.?\Z", re.I)
_EXCLUDE = re.compile(rf"plus {_PERCENT} vat on the subtotal\.?\Z", re.I)
_OTHER_MONEY = re.compile(r"[$£]\s*[0-9]|[0-9]\s*(?:USD|GBP)\b", re.I)
_LABELS = (
    (
        "subtotal_inclusive",
        re.compile(r"subtotal\s*\(incl\.? vat\)\s*:?|the subtotal including vat is", re.I),
    ),
    ("subtotal_exclusive", re.compile(r"subtotal:|the subtotal is", re.I)),
    ("tax_inline", re.compile(rf"plus {_PERCENT} vat on the subtotal,? amounting to", re.I)),
    ("tax", re.compile(rf"vat\s*\({_PERCENT}\):", re.I)),
    ("discount", re.compile(r"less\s*:?\s*rebate per agreement(?: of)?\s*:?", re.I)),
    ("total", re.compile(r"total:|the total amount due is", re.I)),
)


@dataclass(frozen=True)
class Amount:
    role: str
    literal: str
    value: int
    row: tuple[int, int, int, int]
    box: tuple[int, int, int, int]
    avg_quality: float


def _money(text):
    matches = list(_MONEY.finditer(text.translate(_CHARACTERS)))
    if len(matches) != 1:
        return None
    match = matches[0]
    literal = match["prefix"] or match["suffix"]
    if match["prefix_sign"] or match["suffix_sign"]:
        literal = "-" + literal
    return match, literal


def _field(line):
    text = transcript(line).translate(_CHARACTERS)
    for role, pattern in _LABELS:
        label = pattern.match(text)
        if not label:
            continue
        residue = text[label.end() :]
        reading = _money(residue)
        if reading is None:
            return None
        matched, literal = reading
        if residue[: matched.start()].strip() or residue[matched.end() :].strip() not in {"", "."}:
            return None
        start, end = label.end() + matched.start(), label.end() + matched.end()
        cursor = 0
        amount_words = []
        for word in line:
            following = cursor + len(word.text)
            if following > start and cursor < end:
                amount_words.append(word)
            cursor = following + 1
        if not amount_words:
            return None
        return Amount(
            role,
            literal,
            binding.euro_value(literal),
            union(line),
            union(amount_words),
            sum(w.confidence for w in amount_words) / len(amount_words),
        )
    return None


def _is_rule(line, image):
    text = transcript(line).translate(_CHARACTERS).replace(" ", "")
    if re.fullmatch(r"[-_]+", text):
        return True
    x, y, right, bottom = union(line)
    width, height = right - x, bottom - y
    if any(char.isdigit() for char in text) or height > 6 or width < max(60, 20 * height):
        return False
    with image.crop((x, y, right, bottom)).convert("L") as patch:
        for row in range(height):
            with patch.crop((0, row, width, row + 1)) as pixels:
                if sum(pixel < 144 for pixel in pixels.getdata()) >= 0.35 * width:
                    return True
    return False


def _summary(lines, image):
    parsed = [_field(line) for line in lines]
    subtotals = [
        i for i, field in enumerate(parsed) if field is not None and field.role.startswith("subtotal")
    ]
    if len(subtotals) != 1:
        return None
    index = subtotals[0]
    subtotal = parsed[index]
    tails = [
        i
        for i in range(index + 1, min(len(lines), index + 8))
        if parsed[i] is not None and parsed[i].role == "total"
    ]
    if len(tails) != 1:
        return None
    last = tails[0]
    for line in lines[last + 1 :]:
        after = transcript(line).translate(_CHARACTERS)
        if _MONEY.search(after) or _OTHER_MONEY.search(after):
            return None
    preceding = transcript(lines[index - 1]).translate(_CHARACTERS) if index else ""
    included = subtotal.role == "subtotal_inclusive"
    if (included and _EXCLUDE.fullmatch(preceding)) or (not included and _INCLUDE.fullmatch(preceding)):
        return None
    terms = [parsed[i] for i in range(index + 1, last) if not _is_rule(lines[i], image)]
    if any(term is None for term in terms):
        return None
    middle_roles = [term.role for term in terms]
    if included:
        if middle_roles not in ([], ["discount"]):
            return None
    elif not (
        (middle_roles in (["tax"], ["tax", "discount"]) and _EXCLUDE.fullmatch(preceding))
        or middle_roles in (["tax_inline"], ["tax_inline", "discount"])
    ):
        return None
    return ("tax_included" if included else "tax_added"), (subtotal, *terms, parsed[last])


def _confirm(binary, image, field, deadline):
    x, y, right, bottom = field.box
    with image.crop((x, max(0, y - 1), right, min(image.height, bottom + 1))) as patch:
        for zoom in (2, 3):
            with patch.resize((patch.width * zoom, patch.height * zoom), Image.Resampling.BICUBIC).convert(
                "L"
            ) as large:
                a, b = large.width - 1, large.height - 1
                border = sorted(large.getpixel(point) for point in ((0, 0), (a, 0), (0, b), (a, b)))[2]
                with ImageOps.expand(large, border=16, fill=border) as isolated:
                    second = ocr(binary, isolated, 7, deadline)
            reading = _money(transcript(second))
            if reading is None:
                continue
            quality = sum(w.confidence for w in second) / len(second)
            if binding.euro_value(reading[1]) != field.value:
                if quality >= 75:
                    return False
            elif quality >= 80:
                return True
    return False


def find(binary, image, lines, deadline):
    """Return the source-bound calculation when all displayed amounts agree on reread."""
    summary = _summary(lines, image)
    if summary is None:
        return None
    basis, fields = summary
    previous_bottom = 0
    for field in fields:
        x, y, right, bottom = field.row
        if (
            field.avg_quality < 78
            or not 9 <= bottom - y <= 72
            or right - x > 1000
            or y < previous_bottom
            or (field.role == "discount") != (field.value < 0)
        ):
            return None
        previous_bottom = bottom
    calculated = sum(field.value for field in fields[:-1])
    if calculated < 0 or not all(_confirm(binary, image, field, deadline) for field in fields):
        return None
    delta = fields[-1].value - calculated
    roles = {"subtotal_exclusive": "subtotal", "subtotal_inclusive": "subtotal", "tax_inline": "tax"}
    return {
        "schema": binding.EURO_SCHEMA,
        "basis": basis,
        "currency": "€",
        "calculated_cents": calculated,
        "delta_cents": delta,
        "fields": {
            roles.get(field.role, field.role): {
                "printed": field.literal,
                "source_region": list(field.row),
                "amount_region": list(field.box),
            }
            for field in fields
        },
    }
