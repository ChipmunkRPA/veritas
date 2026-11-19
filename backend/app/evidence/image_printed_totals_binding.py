"""Verify the arithmetic and original positions of three printed amounts."""

from __future__ import annotations

import re

from app.core.canonical import canonical_hash

KEY = "printed_total_comparison"
COMMITMENT = KEY + "_sha256"
SCHEMA = "image-printed-total-comparison/1"
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


def calculated_display(record):
    cents = record["calculated_cents"]
    return f"{record['currency']}{cents // 100:,}.{cents % 100:02d}"


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
    dimensions = (source["width"], source["height"])
    if (
        set(record)
        != {
            "schema",
            "source_sha256",
            "affects_overall_confidence",
            "currency",
            "calculated_cents",
            "delta_cents",
            "fields",
        }
        or record["schema"] != SCHEMA
        or record["affects_overall_confidence"] is not False
        or record["source_sha256"] != source["sha256"]
        or record["currency"] not in tuple(CURRENCIES)
        or source.get("exif_orientation") != 1
        or not size_supported(dimensions)
        or source.get(COMMITMENT) != commitment
        or canonical_hash(record) != commitment
        or set(record["fields"]) != set(ROLES)
        or set(images) != set(ROLES)
    ):
        raise ValueError("Printed total evidence differs from the committed image or values")
    previous_bottom = 0
    for role in ROLES:
        field, image = record["fields"][role], images[role]
        if set(field) != {"printed", "source_region", "amount_region"} or not re.fullmatch(
            MONEY, field["printed"], re.ASCII
        ):
            raise ValueError("Invalid printed subtotal, tax or total")
        x, y, right, bottom = _box(field["source_region"], dimensions)
        left_amount, top_amount, right_amount, bottom_amount = _box(field["amount_region"], dimensions)
        if (
            not (
                x <= left_amount < right_amount <= right
                and y <= top_amount < bottom_amount <= bottom
                and 12 <= bottom - y <= 72
                and right - x <= 1000
                and previous_bottom <= y
            )
            or (image["width"], image["height"], image["media_type"]) != (right - x, bottom - y, "image/png")
            or not re.fullmatch(r"[a-f0-9]{64}", image["sha256"])
        ):
            raise ValueError("Invalid crop or position for printed total evidence")
        previous_bottom = bottom
    subtotal, tax, total = (cash_value(record["fields"][role]["printed"]) for role in ROLES)
    calculated, delta = record["calculated_cents"], record["delta_cents"]
    if (
        type(calculated) is not int
        or type(delta) is not int
        or calculated != subtotal + tax
        or delta != total - calculated
        or abs(delta) < MIN_DELTA_CENTS
    ):
        raise ValueError("Printed amounts do not reproduce the checked disagreement")
