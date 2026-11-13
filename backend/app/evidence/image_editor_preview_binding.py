"""Bounded signed JPEG editor-preview records, independent of image runtimes."""

from __future__ import annotations

import re

KEY = "thumbnail_evidence"
SCHEMA = "chipmunk.jpeg-editor-preview/1"
REPORT_PROFILES = frozenset({"standard", "summary", "forensic"})
CARRIER = "jpeg_editor_resource"
BOX = ("x", "y", "w", "h")
SOURCE_ROLES = ("main_image", "embedded_preview")
FOOTPRINT = "measured_footprint"
MAX_REGIONS = 3
DISPLAY_EDGE = 384
PREVIEW_EDGE = 512
MAX_ASSET_BYTES = 1_048_576
_SHA256 = re.compile(r"[0-9a-f]{64}")
_EXHIBIT_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(n):
    return type(n) is int and n > 0


def _box(box, size, coordinate_space=None):
    _require(isinstance(box, dict) and set(box) == set(BOX) | ({"coordinate_space"} if coordinate_space else set())
             and (coordinate_space is None or box["coordinate_space"] == coordinate_space),
             "Thumbnail evidence has an invalid coordinate space or box")
    x, y, width, height = (box[key] for key in BOX)
    _require(type(x) is int and type(y) is int and _positive(width) and _positive(height)
             and 0 <= x < x + width <= size[0] and 0 <= y < y + height <= size[1],
             "Thumbnail evidence box is outside its source")
    return width, height


def asset_groups(document, source):
    """Validate the signed projection and return the declared PNG references."""
    _require(isinstance(document, dict) and set(document) == {"schema", "source_sha256", "preview_carrier",
             "main_exif_orientation", "preview_alignment_orientation", "authorship_confidence_contribution", "regions"}
             and document["schema"] == SCHEMA and document["preview_carrier"] == CARRIER
             and document["source_sha256"] == source.get("sha256") and source.get("exif_orientation") == 1
             and type(document["main_exif_orientation"]) is int and document["main_exif_orientation"] == 1
             and type(document["preview_alignment_orientation"]) is int and document["preview_alignment_orientation"] == 0
             and type(document["authorship_confidence_contribution"]) is int and document["authorship_confidence_contribution"] == 0,
             "Thumbnail evidence has an invalid carrier, orientation, source or confidence")
    main_size = (source.get("width"), source.get("height"))
    _require(all(_positive(n) and n >= 160 for n in main_size) and main_size[0] * main_size[1] <= 24_000_000,
             "Thumbnail evidence has unsupported source geometry")
    rows = document["regions"]
    _require(isinstance(rows, list) and 1 <= len(rows) <= MAX_REGIONS, "Thumbnail evidence has an invalid region roster")
    exhibits = set()
    for index, row in enumerate(rows, 1):
        _require(isinstance(row, dict) and set(row) in ({"region", *SOURCE_ROLES, "marked_preview_pixels", "crops"},
                  {"region", *SOURCE_ROLES, "marked_preview_pixels", "crops", "footprint_grid"})
                 and type(row["region"]) is int and row["region"] == index, "Thumbnail evidence has an invalid region")
        sizes = {key: _box(row[key], size) for key, size in zip(SOURCE_ROLES, (main_size, (PREVIEW_EDGE, PREVIEW_EDGE)), strict=True)}
        if "footprint_grid" in row:
            sizes[FOOTPRINT] = _box(row["footprint_grid"], (PREVIEW_EDGE, PREVIEW_EDGE), "aligned-preview-grid")
        grid_width, grid_height = sizes.get(FOOTPRINT, (PREVIEW_EDGE, PREVIEW_EDGE))
        marked = row["marked_preview_pixels"]
        _require(_positive(marked) and marked <= grid_width * grid_height, "Thumbnail evidence has invalid marked pixels")
        crops = row["crops"]
        _require(isinstance(crops, dict) and set(crops) == set(sizes), "Thumbnail evidence has an invalid crop roster")
        assets = {}
        for key, crop in crops.items():
            _require(isinstance(crop, dict) and set(crop) == {"exhibit_id", "sha256", "display_width", "display_height", "display_reduced"}
                     and isinstance(crop["exhibit_id"], str) and _EXHIBIT_ID.fullmatch(crop["exhibit_id"])
                     and crop["exhibit_id"] not in exhibits and isinstance(crop["sha256"], str) and _SHA256.fullmatch(crop["sha256"]),
                     "Thumbnail evidence has an invalid crop identity")
            exhibit = crop["exhibit_id"]
            exhibits.add(exhibit)
            width, height = crop["display_width"], crop["display_height"]
            box_width, box_height = sizes[key]
            reduce = key != FOOTPRINT and max(box_width, box_height) > DISPLAY_EDGE
            _require(type(crop["display_reduced"]) is bool and crop["display_reduced"] == reduce and _positive(width) and _positive(height)
                     and (max(width, height) == DISPLAY_EDGE and width <= box_width and height <= box_height if reduce
                          else (width, height) == (box_width, box_height)), "Thumbnail evidence has invalid display geometry")
            assets[exhibit] = {"media_type": "image/png", "sha256": crop["sha256"], "width": width, "height": height}
        yield assets
