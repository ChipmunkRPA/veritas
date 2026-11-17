# SPDX-License-Identifier: Apache-2.0
"""Bind visible picture changes to retained PDF pages and rendered pixels."""

from __future__ import annotations

from math import ceil, floor, isfinite

from app.evidence.document_pdf_history_binding import _HASH, MAX_PAGES, validate_saves

SCHEMA = "chipmunk.pdf-saved-pictures/1"
MAX_EDGE = 1200
MAX_PICTURES = 8
MAX_CHANGES = 4
ROLES = ("crop", "overlay")


def raster_box(points, scale):
    x0, y0, x1, y1 = points
    return [ceil(x0 * scale), ceil(y0 * scale), floor(x1 * scale), floor(y1 * scale)]


def changed(comparison):
    return comparison["changed_regions"] > 0


def incomplete(comparisons):
    if any(not any(page["compared"] for page in row["pages"]) for row in comparisons):
        return "no_comparable_pictures"
    if any(any(row["unpaired_pages"]) or any(page["skipped"] for page in row["pages"]) for row in comparisons):
        return "partial_picture_coverage"
    return None


def locations(record):
    if record["status"] == "changed":
        yield from enumerate(record["comparisons"][-1]["changes"], 1)


def _validate_location(row, pairs):
    scale = row["render_scale"]
    a, b, c, d = row["picture_box"]
    if (row["page_xref"] < 1 or (row["earlier_page"], row["current_page"]) not in pairs
            or row["page_rotation"] not in (0, 90, 180, 270) or scale not in (1, 2)
            or not all(isfinite(value) for value in (a, b, c, d)) or not (0 <= a < c and 0 <= b < d)):
        raise ValueError("A picture change has no comparable PDF placement")
    x0, y0, x1, y1 = raster_box(row["picture_box"], scale)
    left, top, right, bottom = row["region_pixels"]
    exact, tested = row["matching_tiles"], row["tested_tiles"]
    if (not all(type(value) is int for value in row["region_pixels"])
            or not (x0 <= left < right <= x1 and y0 <= top < bottom <= y1)
            or not (64 <= x1 - x0 <= MAX_EDGE and 64 <= y1 - y0 <= MAX_EDGE)
            or not 0 < exact <= tested == ((x1 - x0) // 16) * ((y1 - y0) // 16) or exact * 20 < tested * 13
            or not 0 < row["marked_pixels"] <= (right - left) * (bottom - top)
            or not _HASH.fullmatch(row["mask_sha256"])):
        raise ValueError("A picture change is outside the compared page pixels")


def _validate_comparison(row, earlier, current):
    if (row["earlier_version"] != earlier["version"] or type(row["changed_regions"]) is not int
            or row["changed_regions"] < 0 or len(row["changes"]) != min(row["changed_regions"], MAX_CHANGES)
            or not 1 <= min(earlier["pages"], current["pages"]) <= max(earlier["pages"], current["pages"]) <= MAX_PAGES):
        raise ValueError("A picture comparison differs from its PDF saves")
    pairs = set()
    for page in row["pages"]:
        if (type(page["compared"]) is not int or not 0 <= page["compared"] <= MAX_PICTURES
                or type(page["skipped"]) is not int or not 0 <= page["skipped"] <= MAX_PICTURES * 2):
            raise ValueError("A picture comparison has invalid page coverage")
        if page["compared"]:
            pairs.add((page["earlier"], page["current"]))
    for side, key, save in ((0, "earlier", earlier), (1, "current", current)):
        accounted = [page[key] for page in row["pages"]] + row["unpaired_pages"][side]
        if sorted(accounted) != list(range(1, save["pages"] + 1)):
            raise ValueError("Picture page coverage differs from its PDF saves")
    for change in row["changes"]:
        _validate_location(change, pairs)


def validate(record, artifacts, *, size):
    if (record["schema"] != SCHEMA or record["source_bytes"] != size or record["affects_overall_confidence"] is not False
            or record["status"] not in {"unavailable", "changed", "no_change", "no_history"}):
        raise ValueError("Saved picture evidence is outside its PDF source or scope")
    saves, comparisons = validate_saves(record)
    if ((record["status"] == "changed") != bool(comparisons and changed(comparisons[-1]))
            or any(changed(row) for row in comparisons[:-1])
            or record["status"] == "no_change" and
            (not comparisons or saves[-1]["version"] != 1 or incomplete(comparisons))):
        raise ValueError("Saved picture status differs from the recorded comparisons")
    for row, earlier in zip(comparisons, saves[1:]):
        _validate_comparison(row, earlier, saves[0])
    expected = {f"picture{index}_{role}": (role, row) for index, row in locations(record) for role in ROLES}
    if set(artifacts) != set(expected):
        raise ValueError("Saved picture changes require their corresponding page crops")
    for key, asset in artifacts.items():
        role, row = expected[key]
        box = row["region_pixels"] if role == "crop" else raster_box(row["picture_box"], row["render_scale"])
        if (asset["media_type"] != "image/png" or not _HASH.fullmatch(asset["sha256"])
                or asset["role"] != role or asset["page"] != row["current_page"] or asset["source_pixels"] != box
                or (asset["width"], asset["height"]) != (box[2] - box[0], box[3] - box[1])):
            raise ValueError("A saved picture crop differs from its page coordinates")
