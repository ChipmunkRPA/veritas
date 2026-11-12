"""Standard-library binding for exact two-area image difference evidence."""
from __future__ import annotations

import re
from collections.abc import Callable

from app.evidence.image_copy_similarity_binding import SKIPPED_CREDENTIALS, box_bounds, url

VERSION = "image-evidence-15"
KEY = "embedded_pair_difference"
BASE = "prior_embedded_pair_evidence"
SCHEMA = "same-image-exact-difference/1"
BASE_VERSIONS = frozenset({"image-evidence-12", "image-evidence-13", "image-evidence-14"})
# Native held scope, including the two separately replayed archived canvas sizes.
SIZES = frozenset({(768, 640), (768, 768), (1088, 688), (944, 944)})
NOTE = "Matching areas are aligned on the source pixel grid. The highlighted pixels differ between those areas. Either area may have come first. The comparison carries zero weight in AI authorship confidence."
STATUS_TEXT = {
    "local_difference": ("A localized pixel difference was measured between the two displayed areas. "
                         "The image may contain other matching areas, including resized copies."),
    "matching_without_difference": "The pixels matched throughout the two compared areas.",
    "no_supported_pair": "The check found no qualifying two-area comparison.",
    "unassessed_ambiguous_match": "Additional matching detail made the two-area comparison ambiguous.",
    "outside_scope": "The exact two-area comparison covers selected native sizes of upright RGB PNGs.",
    "unassessed_runtime": "The exact two-area comparison could not be completed for this file.",
    SKIPPED_CREDENTIALS: "Trusted Content Credentials already record AI involvement in this image.",
}
ROLES = ("area_a", "area_b")
ASSETS = {"area_a": "Area A original crop", "area_b": "Area B original crop",
          "overlay_a": "Area A differing pixels", "overlay_b": "Area B differing pixels",
          "difference_mask": "Exact binary difference mask"}
_SHA = re.compile(r"[0-9a-f]{64}")


def base_document(document: dict) -> dict:
    if document.get("version") != VERSION:
        return document
    lineage = document[BASE]
    return {key: value for key, value in document.items() if key not in {BASE, KEY}} | lineage


def artifacts(document: dict) -> dict:
    if document.get("version") != VERSION:
        return {}
    pair = (document.get(KEY) or {}).get("pair")
    return pair["artifacts"] if pair is not None else {}


def validate_scope(document: dict, skip: bool) -> None:
    if document.get("version") == VERSION and (document[KEY]["status"] == SKIPPED_CREDENTIALS) != skip:
        raise ValueError("Exact two-area coverage differs from the frozen signed provenance result")


def _need(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_witnesses(points, area, tile: int = 16) -> bool:
    x, y, right, bottom = area
    width, height = right - x, bottom - y
    if (not isinstance(points, list) or not 12 <= len(points) <= 6000
            or any(not isinstance(point, list) or len(point) != 2 or any(type(value) is not int for value in point)
                   for point in points)):
        return False
    if points != [list(point) for point in sorted({tuple(point) for point in points})]:
        return False
    if (not all(x <= a <= right - tile and y <= b <= bottom - tile for a, b in points)
            or (min(a for a, _ in points), min(b for _, b in points),
                max(a for a, _ in points) + tile, max(b for _, b in points) + tile) != area):
        return False
    cells = {(min(2, 3 * (a - x) // width), min(2, 3 * (b - y) // height)) for a, b in points}
    return len(cells) >= 6


def _validate_pair(document, pair):
    source = document["source"]
    width, height = source["width"], source["height"]
    _need(isinstance(pair, dict) and set(pair) == {"id", "regions", "difference_regions", "matching_pixels", "different_pixels",
                                                   "witnesses", "artifacts"} and pair["id"] == "D01",
          "The exact two-area proof is malformed")
    _need(set(pair["regions"]) == set(ROLES) == set(pair["difference_regions"]), "Two source areas are required")
    a, b = (box_bounds(pair["regions"][role], width, height) for role in ROLES)
    aw, ah = a[2] - a[0], a[3] - a[1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    _need((width, height) in SIZES and source["exif_orientation"] == 1 and min(aw, ah) >= 160
          and b == (a[0] + dx, a[1] + dy, a[2] + dx, a[3] + dy) and (dx > 0 or (dx == 0 and dy > 0))
          and max(abs(dx), abs(dy)) >= 192 and (abs(dx) >= aw + 24 or abs(dy) >= ah + 24),
          "The exact comparison exceeds the tested native geometry")
    _need(_valid_witnesses(pair["witnesses"], a), "Exact texture witnesses do not span the comparison")
    changed, matching = pair["different_pixels"], pair["matching_pixels"]
    aa, bb = (box_bounds(pair["difference_regions"][role], width, height) for role in ROLES)
    _need(bb == (aa[0] + dx, aa[1] + dy, aa[2] + dx, aa[3] + dy)
          and aa[0] >= a[0] + 8 and aa[1] >= a[1] + 8 and aa[2] <= a[2] - 8 and aa[3] <= a[3] - 8,
          "The exact difference must be inside both compared areas")
    area = aw * ah
    difference_area = (aa[2] - aa[0]) * (aa[3] - aa[1])
    _need(type(changed) is int and type(matching) is int and 48 <= changed <= difference_area
          and matching + changed == area and 100 * changed <= 15 * area and difference_area <= .45 * area,
          "The exact difference pixel count is outside its support")
    _need(set(pair["artifacts"]) == set(ASSETS), "The exact comparison is missing a required image or mask")
    for asset in pair["artifacts"].values():
        digest = asset.get("sha256", "")
        _need(set(asset) == {"sha256", "media_type", "width", "height", "url"} and _SHA.fullmatch(str(digest))
              and asset["media_type"] == "image/png" and (asset["width"], asset["height"]) == (aw, ah)
              and asset["url"] == url(document, digest), "An exact comparison image differs from its source region")


def validate(document: dict, hash_document: Callable[[dict], str]) -> None:
    if document.get("version") != VERSION:
        _need(KEY not in document and BASE not in document, "Historical image evidence contains a newer comparison")
        return
    lineage = document.get(BASE)
    _need(isinstance(lineage, dict) and set(lineage) == {"id", "version", "sha256"}
          and lineage["version"] in BASE_VERSIONS and lineage["id"] != document["id"]
          and _SHA.fullmatch(str(lineage["sha256"])), "The exact comparison lacks its frozen base evidence")
    _need(hash_document(base_document(document)) == lineage["sha256"], "The exact comparison changes prior image evidence")
    record = document.get(KEY)
    _need(isinstance(record, dict) and set(record) == {"schema", "source_sha256", "status", "interpretation",
                                                       "authorship_confidence_contribution", "pair"},
          "The exact two-area record is malformed")
    _need(record["schema"] == SCHEMA and record["source_sha256"] == document["source"]["sha256"]
          and record["interpretation"] == NOTE and type(record["authorship_confidence_contribution"]) is int
          and record["authorship_confidence_contribution"] == 0 and record["status"] in STATUS_TEXT
          and (record["pair"] is not None) == (record["status"] == "local_difference"),
          "The exact comparison differs from its source or evidence scope")
    if record["pair"] is not None:
        _validate_pair(document, record["pair"])
