# SPDX-License-Identifier: Apache-2.0
"""Bind a static PDF text field to both saved readings and its rendered area."""

from __future__ import annotations

import re
from math import ceil, floor, isfinite

from app.core.canonical import canonical_hash

KEY = "pdf_form_value_differences"
COMMITMENT = f"{KEY}_sha256"
POLICY = "pdf_static_form_review"
SCHEMA = "chipmunk.pdf-form-value-differences/1"
MAX_PAGES = 6
MAX_WIDGETS = 96
MAX_REVIEWED = 12
MAX_CROP_BYTES = 512 * 1024
SCALE = 3
TITLE = "Saved PDF form values"
LIMITATION = ("The stored form value differs from the saved page appearance shown here. An outdated appearance or "
              "form formatting can explain the difference. PDF viewers may display the field differently. "
              "This observation carries zero weight in the AI assessment.")
WORDING = re.compile(r"(?=.{5,54}\Z)(?=.*[A-Za-z])[A-Za-z0-9]+(?: +[A-Za-z0-9]+){1,7}\Z")
HASH = re.compile(r"[a-f0-9]{64}\Z")


def from_result(result):
    return (result.get("manifest", {}).get("sampling") or {}).get(COMMITMENT)


def normalized(text):
    return " ".join(text.casefold().split())


def rectangles(box, page):
    return (isinstance(box, list) and len(box) == 4 and isinstance(page, list) and len(page) == 2
            and all(type(n) in (int, float) and isfinite(n) for n in (*box, *page))
            and 0 <= box[0] < box[2] <= page[0] <= 7200 and 0 <= box[1] < box[3] <= page[1] <= 7200
            and 69.99 <= box[2] - box[0] < 450.01 and 9.99 <= box[3] - box[1] <= 65.01)


def crop_role(index):
    return f"field_{index + 1}"


def validate_record(record):
    if (not isinstance(record, dict) or set(record) != {"schema", "affects_overall_confidence", "rows"}
            or record["schema"] != SCHEMA or record["affects_overall_confidence"] is not False
            or not isinstance(record["rows"], list) or not 1 <= len(record["rows"]) <= MAX_REVIEWED):
        raise ValueError("Invalid saved PDF form observation")
    previous = (0, 0)
    for row in record["rows"]:
        if (not isinstance(row, dict) or set(row) != {"page", "xref", "page_size", "box", "saved_value", "display_value", "crop"}
                or type(row["page"]) is not int or not 1 <= row["page"] <= MAX_PAGES
                or type(row["xref"]) is not int or not 1 <= row["xref"] < 1000
                or (row["page"], row["xref"]) <= previous or not rectangles(row["box"], row["page_size"])):
            raise ValueError("Invalid saved PDF form field position")
        previous = row["page"], row["xref"]
        saved, visible = (row[key] for key in ("saved_value", "display_value"))
        if (any(not isinstance(value, str) or len(value) > 54 or not WORDING.fullmatch(value.strip(" "))
                for value in (saved, visible)) or len(saved.split()) != len(visible.split())
                or normalized(saved) == normalized(visible)):
            raise ValueError("Saved PDF form values are outside the literal comparison")
        crop = row["crop"]
        if (not isinstance(crop, dict) or set(crop) != {"sha256", "pixels_sha256", "width", "height"}
                or any(not isinstance(crop[k], str) or not HASH.fullmatch(crop[k]) for k in ("sha256", "pixels_sha256"))):
            raise ValueError("Invalid saved PDF form crop")
        x0, y0, x1, y1 = row["box"]
        for size, start, end in (("width", x0, x1), ("height", y0, y1)):
            expected = ceil(end * SCALE) - floor(start * SCALE)
            # Position fields round native points to six decimal places.
            if type(crop[size]) is not int or abs(expected - crop[size]) > 1:
                raise ValueError("Saved PDF form crop lies outside its field rectangle")


def validate(document, expected):
    source = document["source"]
    review = document.get(KEY)
    if review is None and KEY not in document and COMMITMENT not in source and expected is None:
        return
    if (source.get("method") != "pymupdf" or not isinstance(review, dict)
            or set(review) != {"inspection", "artifacts"} or expected is None
            or source.get(COMMITMENT) != expected or canonical_hash(review["inspection"]) != expected):
        raise ValueError("Saved PDF form evidence differs from its published source commitment")
    record, artifacts = review["inspection"], review["artifacts"]
    validate_record(record)
    if not isinstance(artifacts, dict) or set(artifacts) != {crop_role(i) for i, _ in enumerate(record["rows"])}:
        raise ValueError("Saved PDF form fields require their raster evidence")
    for index, row in enumerate(record["rows"]):
        crop, asset = row["crop"], artifacts[crop_role(index)]
        if (not isinstance(asset, dict) or asset.get("media_type") != "image/png"
                or any(asset.get(key) != crop[key] for key in ("sha256", "width", "height"))
                or type(asset.get("size_bytes")) is not int or not 0 < asset["size_bytes"] <= MAX_CROP_BYTES):
            raise ValueError("Saved PDF form raster differs from its field geometry")
