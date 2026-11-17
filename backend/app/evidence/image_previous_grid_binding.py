# SPDX-License-Identifier: Apache-2.0
"""Source bindings for an earlier JPEG compression observation."""

from __future__ import annotations

import math
import re

from app.core.canonical import canonical_hash

KEY = "jpeg_prior_grid_review"
COMMITMENT = "jpeg_prior_grid_review_sha256"
LEGACY_SCHEMA = "image-jpeg-prior-grid/1"
SCHEMA = "image-jpeg-prior-grid/2"
TITLE = "Local JPEG compression history"
SCOPES = {
    LEGACY_SCHEMA: "This check compares an earlier JPEG grid across supported upright JPEG images.",
    SCHEMA: "The grid is measured in the JPEG's stored pixel order. The outline follows the displayed image orientation.",
}
LIMITATION = "Blur, retouching, and other ordinary edits can change the grid. This observation carries zero AI confidence weight."
STATUS_TEXT = {
    "matched": "The highlighted area retains less of the earlier JPEG block pattern than the surrounding image.",
    "no_region": "An earlier JPEG grid was measured. No region showed the required difference from its surroundings.",
    "no_prior_grid": "No repeatable earlier JPEG grid was measured across the picture.",
    "outside_profile": "This JPEG falls outside the supported image and compression profile.",
    "earlier_region": "The earlier JPEG or regional check already supplied a region.",
    "provenance": "Verified Content Credentials stopped the analysis before this check.",
    "unavailable": "The local compression check could not finish during this analysis.",
}
BASIS = ("The earlier pattern was weaker inside the rectangle in both measured directions. "
         "At least three sides retained more of the pattern.")
PLATES = {"context": "Highlighted area in the submitted image", "crop": "Unmarked crop from the submitted image"}
EARLIER_CHECKS = frozenset({"img.forensic.jpeg_dq", "loc.native_regional"})
_ASSESSABLE = {"matched", "no_region", "no_prior_grid", "earlier_region"}
_SHA = re.compile(r"[a-f0-9]{64}")


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(KEY)


def assets(document):
    return (document.get(KEY) or {}).get("artifacts", {})


def earlier_region(exhibits):
    return any(item["kind"] == "regions" and item["created_by"] in EARLIER_CHECKS and item.get("regions")
               for item in exhibits)


def orient_region(region, orientation, width, height):
    """Map a half-open rectangle between the stored and displayed pixel grids."""
    left, top, right, bottom = region
    if orientation in (5, 6, 7, 8):
        left, top, right, bottom = top, left, bottom, right
        width, height = height, width
    if orientation in (2, 3, 6, 7):
        left, right = width - right, width - left
    if orientation in (3, 4, 7, 8):
        top, bottom = height - bottom, height - top
    return [left, top, right, bottom]


def _numbers(value, size, lo, hi):
    return (isinstance(value, list) and len(value) == size
            and all(type(number) in {int, float} and math.isfinite(number) and lo <= number <= hi for number in value))


def _validate_reading(record, source, pictures):
    width, height = source["width"], source["height"]
    orientation = source.get("exif_orientation")
    orientations = (1,) if record["schema"] == LEGACY_SCHEMA else range(1, 9)
    status, measurement, region = (record[key] for key in ("status", "measurement", "region"))
    if status in _ASSESSABLE and (type(orientation) is not int or orientation not in orientations
                                 or min(width, height) < 320 or width * height > 1_048_576):
        raise ValueError("The JPEG compression reading uses an unsupported source geometry")
    if status not in {"matched", "no_region"}:
        if measurement is not None or region is not None or pictures:
            raise ValueError("An unmeasured JPEG observation has unexpected local evidence")
        return
    offset = measurement["grid_offset"]
    quality = measurement["estimated_earlier_quality"]
    if (not isinstance(offset, list) or len(offset) != 2 or any(type(n) is not int or n not in range(8) for n in offset)
            or offset == [0, 0] or type(quality) is not int or quality not in range(45, 81, 5)
            or not _numbers(measurement["background_support"], 2, 0., 2.)):
        raise ValueError("Invalid JPEG compression history measurements")
    expected = {"grid_offset", "estimated_earlier_quality", "background_support"}
    if status == "no_region":
        if set(measurement) != expected or region is not None or pictures:
            raise ValueError("A JPEG review without a region has unexpected local evidence")
        return
    if (not isinstance(region, list) or len(region) != 4 or any(type(n) is not int for n in region)
            or set(measurement) != expected | {"inside_support", "surrounding_support", "supported_cells"}):
        raise ValueError("Invalid JPEG region measurements")
    x, y, right, bottom = region
    size = right - x, bottom - y
    inverse = {6: 8, 8: 6}.get(orientation, orientation)
    stored_x, stored_y, _, _ = orient_region(region, inverse, width, height)
    if (not 0 <= x < right <= width or not 0 <= y < bottom <= height
            or stored_x % 16 != offset[0] or stored_y % 16 != offset[1]
            or any(n not in {32, 48, 64, 80, 96, 128, 160} for n in size)):
        raise ValueError("The JPEG region does not follow its measured pixel grid")
    inside, outside = measurement["inside_support"], measurement["surrounding_support"]
    if not (_numbers(inside, 2, -1., 1.) and _numbers(outside, 2, -1., 1.)
            and all(a < b for a, b in zip(inside, outside, strict=True))):
        raise ValueError("The earlier JPEG pattern did not weaken in both measured directions")
    cells = measurement["supported_cells"]
    if (not isinstance(cells, list) or len(cells) != 2 or any(type(n) is not int for n in cells)
            or not 1 <= cells[0] <= size[0] * size[1] // 256 or not 1 <= cells[1] <= (sum(size) // 4) + 16):
        raise ValueError("Invalid JPEG compression coverage")
    if set(pictures) != set(PLATES):
        raise ValueError("The JPEG region needs both the highlighted image and source crop")
    for name, (w, h) in {"context": (width, height), "crop": size}.items():
        asset = pictures[name]
        if (not _SHA.fullmatch(str(asset.get("sha256", "")))
                or (asset.get("media_type"), asset.get("width"), asset.get("height")) != ("image/png", w, h)):
            raise ValueError("A JPEG evidence picture differs from its recorded dimensions")


def validate(document, commitment, *, stopped=None):
    review = document.get(KEY)
    source = document["source"]
    if review is None and commitment is None and COMMITMENT not in source:
        return
    if not isinstance(review, dict) or set(review) != {"record", "artifacts"} or document.get("schema") != "chipmunk.image-evidence/1":
        raise ValueError("JPEG compression history requires an image evidence record")
    record, pictures = review["record"], review["artifacts"]
    status = record["status"]
    if (not isinstance(commitment, str) or not _SHA.fullmatch(commitment)
            or set(record) != {"schema", "source_sha256", "affects_overall_confidence", "status", "observation",
                              "scope", "limitation", "basis", "measurement", "region"}
            or canonical_hash(record) != commitment or source.get(COMMITMENT) != commitment
            or record["schema"] not in SCOPES or record["source_sha256"] != source["sha256"]
            or record["affects_overall_confidence"] is not False
            or record["observation"] != STATUS_TEXT.get(status) or status not in STATUS_TEXT
            or record["scope"] != SCOPES[record["schema"]] or record["limitation"] != LIMITATION
            or record["basis"] != (BASIS if status == "matched" else None)
            or (stopped is not None and (status == "provenance") != stopped)):
        raise ValueError("JPEG compression history differs from its source commitment")
    _validate_reading(record, source, pictures)
