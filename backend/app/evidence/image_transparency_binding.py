"""Validate signed PNG transparency measurements without an image runtime."""
from __future__ import annotations

import re

KEY = "png_transparency"
SCHEMA = "png-transparency/1"
ROLES = ("visible_crop", "revealed_crop")
REPORT_PROFILES = frozenset({"summary", "standard", "forensic"})
TILE = 128
MIN_PAIRS = 32
MAX_EDGE = 4096
MAX_PIXELS = 4_194_304
MAX_ASSET_BYTES = 64 * 1024
COUNTS = ("width", "height", "fully_transparent_pixels", "qualifying_tiles")
_SHA = re.compile(r"[a-f0-9]{64}")
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def projection(review):
    result = {key: review[key] for key in ("schema", "source_sha256", *COUNTS, "authorship_confidence_contribution")}
    location = review["location"]
    result["location"] = None if location is None else {
        "bounds": location["bounds"], "qualifying_pairs": location["qualifying_pairs"],
        **{role: {key: location[role][key] for key in ("exhibit_id", "sha256")} for role in ROLES},
    }
    return result


def assets(record, source):
    _require(isinstance(record, dict)
             and set(record) == {"schema", "source_sha256", *COUNTS, "authorship_confidence_contribution", "location"}
             and record["schema"] == SCHEMA and record["source_sha256"] == source.get("sha256")
             and bool(_SHA.fullmatch(str(record["source_sha256"])))
             and source.get("coordinate_space") == "exif-oriented-original"
             and type(record["authorship_confidence_contribution"]) is int
             and record["authorship_confidence_contribution"] == 0
             and all(type(record[key]) is int for key in COUNTS), "Invalid PNG transparency measurement")
    width, height, transparent, tiles = (record[key] for key in COUNTS)
    _require((width, height) == (source.get("width"), source.get("height"))
             and 0 < width <= MAX_EDGE and 0 < height <= MAX_EDGE and width * height <= MAX_PIXELS
             and 0 <= transparent <= width * height
             and 0 <= tiles <= ((width + TILE - 1) // TILE) * ((height + TILE - 1) // TILE)
             and tiles * (MIN_PAIRS // 2 + 1) <= transparent, "Invalid PNG transparency coverage")
    loc = record["location"]
    if loc is None:
        return {}
    _require(tiles and isinstance(loc, dict) and set(loc) == {"bounds", "qualifying_pairs", *ROLES}
             and isinstance(loc["bounds"], dict) and set(loc["bounds"]) == {"x", "y", "w", "h"},
             "Invalid PNG transparency location")
    x, y, w, h = (loc["bounds"][k] for k in ("x", "y", "w", "h"))
    _require(all(type(n) is int for n in (x, y, w, h, loc["qualifying_pairs"]))
             and 0 <= x < width and 0 <= y < height and x % TILE == y % TILE == 0
             and w == min(TILE, width - x) and h == min(TILE, height - y)
             and MIN_PAIRS <= loc["qualifying_pairs"] <= (w - 1) * h + (h - 1) * w,
             "PNG transparency crop falls outside the recorded grid")
    result = {}
    for role in ROLES:
        asset = loc[role]
        _require(isinstance(asset, dict) and set(asset) == {"exhibit_id", "sha256"} and isinstance(asset["exhibit_id"], str)
                 and bool(_ID.fullmatch(asset["exhibit_id"])) and asset["exhibit_id"] not in result
                 and isinstance(asset["sha256"], str) and bool(_SHA.fullmatch(asset["sha256"])),
                 "Invalid PNG transparency crop identity")
        result[asset["exhibit_id"]] = {"sha256": asset["sha256"], "media_type": "image/png", "width": w, "height": h}
    return result
