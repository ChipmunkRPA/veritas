"""Portable validation for the comparison of two APNG display views."""
from __future__ import annotations

import re

KEY = "png_animation"
SCHEMA = "png-animation-opening/1"
TITLE = "Animated PNG opening comparison"
ROLES = {"fallback": "Static fallback image", "opening": "First animation frame"}
REPORT_PROFILES = frozenset({"summary", "standard", "forensic"})
MAX_EDGE = 4096
MAX_PIXELS = 4_194_304
MAX_PREVIEW_EDGE = 320
MAX_ASSET_BYTES = 384 * 1024
MAX_INPUT_BYTES = 24 * 1024 * 1024
FIELDS = ("frame_count", "changed_pixels", "bounds")
SCOPE = ("Compares the separate static fallback and the first full-canvas replacement frame of an 8-bit RGB or RGBA PNG. "
         "Counts use display-oriented pixels on white. Later frames are outside this comparison.")
LIMITATION = ("Separate fallback art is allowed in animated PNGs. The difference carries zero weight "
              "in the AI assessment.")
_SHA = re.compile(r"[a-f0-9]{64}")
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def preview_size(width, height):
    scale = min(1, MAX_PREVIEW_EDGE / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def projection(review):
    return {**{key: review[key] for key in ("schema", "source_sha256", *FIELDS, "authorship_confidence_contribution")},
            "previews": {role: {key: asset[key] for key in ("exhibit_id", "sha256")}
                         for role, asset in review["previews"].items()}}


def assets(record, source):
    width, height = source.get("width"), source.get("height")
    if (not isinstance(record, dict)
            or set(record) != {"schema", "source_sha256", *FIELDS, "previews", "authorship_confidence_contribution"}
            or record["schema"] != SCHEMA or record["source_sha256"] != source.get("sha256")
            or not _SHA.fullmatch(str(record["source_sha256"]))
            or source.get("coordinate_space") != "exif-oriented-original"
            or type(record["authorship_confidence_contribution"]) is not int
            or record["authorship_confidence_contribution"] != 0
            or type(width) is not int or type(height) is not int
            or not 0 < width <= MAX_EDGE or not 0 < height <= MAX_EDGE or width * height > MAX_PIXELS
            or type(record["frame_count"]) is not int or not 2 <= record["frame_count"] <= 2**32
            or type(record["changed_pixels"]) is not int or not 0 <= record["changed_pixels"] <= width * height
            or not isinstance(record["previews"], dict)):
        raise ValueError("Invalid animated PNG comparison")
    changed, bounds, previews = record["changed_pixels"], record["bounds"], record["previews"]
    if not changed:
        if bounds is not None or previews:
            raise ValueError("Matching animated PNG views cannot contain difference evidence")
        return {}
    if (not isinstance(bounds, list) or len(bounds) != 4 or any(type(value) is not int for value in bounds)):
        raise ValueError("Invalid animated PNG difference bounds")
    x, y, w, h = bounds
    if not (x >= 0 and y >= 0 and 0 < w <= width - x and 0 < h <= height - y and changed <= w * h):
        raise ValueError("Animated PNG difference falls outside the source")
    if previews and previews.keys() != ROLES.keys():
        raise ValueError("Animated PNG evidence requires a pair of previews")
    w, h = preview_size(width, height)
    indexed = {}
    for asset in previews.values():
        if (not isinstance(asset, dict) or set(asset) != {"exhibit_id", "sha256"}
                or not isinstance(asset["exhibit_id"], str) or not _ID.fullmatch(asset["exhibit_id"])
                or not isinstance(asset["sha256"], str) or not _SHA.fullmatch(asset["sha256"])
                or asset["exhibit_id"] in indexed):
            raise ValueError("Invalid animated PNG preview identity")
        indexed[asset["exhibit_id"]] = {"sha256": asset["sha256"], "media_type": "image/png", "width": w, "height": h}
    return indexed
