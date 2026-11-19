"""Validate signed source positions for printed identifier comparisons."""
from __future__ import annotations

import re

from app.core.canonical import canonical_hash

KEY = "printed_identifier_comparison"
COMMITMENT = KEY + "_sha256"
SCHEMA = "image-printed-identifier-comparison/1"
TITLE = "Printed identifier mismatch"
LIMITATION = ("These two fields are labeled as an entry and its confirmation. A typing error or an ordinary edit can explain the mismatch. "
              "This observation carries zero weight in the AI assessment.")
PLATES = {"entry": "Printed identifier", "confirmation": "Printed confirmation"}
FIELDS = {"reference id": "Reference ID", "record id": "Record ID", "invoice id": "Invoice ID"}
IDENTIFIER = r"[A-Z]{2,5}-[0-9]{6,12}"
MAX_SIDE = 2048
MAX_PIXELS = 3_000_000


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(KEY)


def assets(document):
    return (document.get(KEY) or {}).get("artifacts", {})


def size_supported(size):
    width, height = size
    return min(size) >= 240 and max(size) <= MAX_SIDE and width * height <= MAX_PIXELS


def validate(document, commitment, *, stopped=False):
    source = document["source"]
    evidence = document.get(KEY)
    if evidence is None and commitment is None and COMMITMENT not in source:
        return
    if (stopped or not isinstance(evidence, dict) or set(evidence) != {"record", "artifacts"}
            or document.get("schema") != "chipmunk.image-evidence/1"):
        raise ValueError("Invalid printed identifier evidence")
    record, images = evidence["record"], evidence["artifacts"]
    width, height = source["width"], source["height"]
    if (set(record) != {"schema", "source_sha256", "affects_overall_confidence", "field", "values"}
            or record["schema"] != SCHEMA or record["affects_overall_confidence"] is not False
            or record["source_sha256"] != source["sha256"] or record["field"] not in FIELDS.values()
            or source.get("exif_orientation") != 1 or not size_supported((width, height))
            or source.get(COMMITMENT) != commitment or canonical_hash(record) != commitment
            or set(record["values"]) != set(PLATES) or set(images) != set(PLATES)):
        raise ValueError("Printed identifier evidence differs from the bound image")
    first, second = (record["values"][role] for role in PLATES)
    for role, item in zip(PLATES, (first, second), strict=True):
        box, asset = item["region"], images[role]
        if (set(item) != {"value", "region"} or not re.fullmatch(IDENTIFIER, item["value"], re.ASCII)
                or not isinstance(box, list) or len(box) != 4 or not all(type(n) is int for n in box)):
            raise ValueError("Invalid printed identifier reading")
        x, y, right, bottom = box
        if (not (0 <= x < right <= width and 0 <= y < bottom <= height and 12 <= bottom-y <= 64 and right-x <= 920)
                or (asset["width"], asset["height"], asset["media_type"]) != (right-x, bottom-y, "image/png")
                or not re.fullmatch(r"[a-f0-9]{64}", asset["sha256"])):
            raise ValueError("Invalid printed identifier crop")
    if first["value"] == second["value"] or first["region"][3] + 5 > second["region"][1]:
        raise ValueError("Printed identifiers must disagree on two separate lines")
