"""Verify printed summary arithmetic and original image positions."""

from __future__ import annotations

import re

from app.core.canonical import canonical_hash

KEY = "printed_total_comparison"
COMMITMENT = KEY + "_sha256"
SCHEMA = "image-printed-total-comparison/1"
EURO_SCHEMA = "image-printed-total-comparison/2"
TITLE = "Printed total mismatch"
PACKET_ID = "PT01"
LIMITATION = (
    "Another adjustment or an ordinary accounting or transcription error can explain the difference. "
    "This observation carries zero weight in the AI assessment."
)
ROLES = {"subtotal": "Printed subtotal", "tax": "Printed tax", "total": "Printed total"}
CURRENCIES = "$£€"
CASH = r"[0-9]{1,3}(?:,[0-9]{3}){1,2}|[0-9]{1,8}"
MONEY = rf"(?:{CASH})\.[0-9]{{2}}"
EURO_MONEY = (
    r"(?:[0-9]{1,8}[.,][0-9]{2}|[0-9]{1,3}(?:,[0-9]{3}){1,2}\.[0-9]{2}|"
    r"[0-9]{1,3}(?:\.[0-9]{3}){1,2},[0-9]{2}|[0-9]{1,3}(?:'[0-9]{3}){1,2}\.[0-9]{2})"
)
MIN_DELTA_CENTS = 2


def size_supported(size):
    width, height = size
    return 240 <= min(size) <= max(size) <= 2048 and width * height <= 3_000_000


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(KEY)


def assets(document):
    return (document.get(KEY) or {}).get("artifacts", {})


def cash_value(token):
    return int(token.replace(",", "").replace(".", ""))


def euro_value(token):
    return cash_value(token.replace("'", ""))


def roles(record):
    if record["schema"] == SCHEMA:
        return tuple(ROLES)
    ordered = ["subtotal"]
    if record["basis"] == "tax_added":
        ordered.append("tax")
    if "discount" in record["fields"]:
        ordered.append("discount")
    return (*ordered, "total")


def amount_display(record, token):
    return f"-{record['currency']}{token[1:]}" if token.startswith("-") else f"{record['currency']}{token}"


def calculated_display(record):
    cents = record["calculated_cents"]
    integer = f"{cents // 100:,}"
    decimal = "."
    if record["schema"] == EURO_SCHEMA:
        printed = record["fields"]["total"]["printed"]
        grouping = next((char for char in printed[:-3] if char in ",.'"), "")
        integer = integer.replace(",", grouping)
        decimal = printed[-3]
    return f"{record['currency']}{integer}{decimal}{cents % 100:02d}"


def _box(value, dimensions):
    if not isinstance(value, list) or len(value) != 4 or any(type(point) is not int for point in value):
        raise ValueError("Printed amount position needs four pixel coordinates")
    x, y, right, bottom = value
    width, height = dimensions
    if not (0 <= x < right <= width and 0 <= y < bottom <= height):
        raise ValueError("Printed amount position falls outside the submitted image")
    return x, y, right, bottom


def validate(document, commitment, *, stopped=False):
    source = document["source"]
    reading = document.get(KEY)
    if reading is None and commitment is None and COMMITMENT not in source:
        return
    if (
        stopped
        or not isinstance(reading, dict)
        or set(reading) != {"record", "artifacts"}
        or document.get("schema") != "chipmunk.image-evidence/1"
    ):
        raise ValueError("Invalid printed total evidence")
    record, images = reading["record"], reading["artifacts"]
    if not isinstance(record, dict) or not isinstance(images, dict):
        raise ValueError("Invalid printed total evidence")
    dimensions = (source["width"], source["height"])
    is_euro = record.get("schema") == EURO_SCHEMA
    currencies = ("€",) if is_euro else tuple(CURRENCIES)
    record_keys = {"schema", "source_sha256", "affects_overall_confidence", "currency", "calculated_cents",
                   "delta_cents", "fields"}
    if (
        set(record) != (record_keys | {"basis"} if is_euro else record_keys)
        or record["schema"] not in (SCHEMA, EURO_SCHEMA)
        or record["affects_overall_confidence"] is not False
        or record["source_sha256"] != source["sha256"]
        or record["currency"] not in currencies
        or (is_euro and record["basis"] not in ("tax_added", "tax_included"))
        or not isinstance(record["fields"], dict)
        or source.get("exif_orientation") != 1
        or not size_supported(dimensions)
        or source.get(COMMITMENT) != commitment
        or canonical_hash(record) != commitment
    ):
        raise ValueError("Printed total evidence differs from the committed image or values")
    ordered = roles(record)
    if set(record["fields"]) != set(ordered) or set(images) != set(ordered):
        raise ValueError("Printed total evidence omits or adds a calculation term")
    previous_bottom = 0
    values = []
    for role in ordered:
        field, image = record["fields"][role], images[role]
        amount_pattern = ("-" if role == "discount" else "") + EURO_MONEY if is_euro else MONEY
        if set(field) != {"printed", "source_region", "amount_region"} or not re.fullmatch(amount_pattern, field["printed"], re.ASCII):
            raise ValueError("Invalid printed amount")
        values.append(euro_value(field["printed"]) if is_euro else cash_value(field["printed"]))
        if role == "discount" and values[-1] >= 0:
            raise ValueError("The printed discount must be a deduction")
        x, y, right, bottom = _box(field["source_region"], dimensions)
        left_amount, top_amount, right_amount, bottom_amount = _box(field["amount_region"], dimensions)
        if (
            not (
                x <= left_amount < right_amount <= right
                and y <= top_amount < bottom_amount <= bottom
                and (9 if is_euro else 12) <= bottom - y <= 72
                and right - x <= 1000
                and previous_bottom <= y
            )
            or (image["width"], image["height"], image["media_type"]) != (right - x, bottom - y, "image/png")
            or not re.fullmatch(r"[a-f0-9]{64}", image["sha256"])
        ):
            raise ValueError("Invalid crop or position for printed total evidence")
        previous_bottom = bottom
    calculated, delta = record["calculated_cents"], record["delta_cents"]
    if (
        type(calculated) is not int
        or type(delta) is not int
        or calculated < 0
        or calculated != sum(values[:-1])
        or delta != values[-1] - calculated
        or abs(delta) < MIN_DELTA_CENTS
    ):
        raise ValueError("Printed amounts do not reproduce the checked disagreement")
