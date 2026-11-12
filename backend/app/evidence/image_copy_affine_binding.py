"""Public structure for source-bound, confidence-neutral geometric matches."""

from __future__ import annotations

import math
import re

KEY = "copy_affine"
VERSION = "image-evidence-13"
LARGE_VERSION = "image-evidence-14"
JPEG_BLOCK_AFFINE_VERSION = "image-evidence-17"
LARGE_VERSIONS = frozenset({LARGE_VERSION, JPEG_BLOCK_AFFINE_VERSION})
AFFINE_VERSIONS = LARGE_VERSIONS | {VERSION}
SCHEMA = "same-image-affine-support/1"
LARGE_SCHEMA = "same-image-affine-support/2"
ROLES = ("area_a", "area_b")
STATUS_TEXT = {
    "supported_pair": "Matching detail was found after geometric alignment.",
    "no_pair": "None of the distinct candidates examined had enough shared detail after alignment.",
    "unsupported_geometry": "This check requires a long side of 513–1,024 px and a short side of at least 384 px.",
    "unassessed_match_budget": "Too many candidate matches were present to show a distinct comparison.",
}
NOTE = (
    "The check compares textured neighborhoods after alignment. Area A shows the selected region and Area B maps it to the second crop. "
    "Repeated objects and image editing can both cause a match. "
    "The evidence does not establish which area came first or whether AI was involved."
)
LARGE_COVERAGE_TEXT = (
    "This check covers images with a long side of 513–1,024 px and a short side of at least 384 px, "
    "or a long side of 1,025–2,048 px and a short side of at least 768 px."
)
_SHA = re.compile(r"[a-f0-9]{64}")


def supported(width: int, height: int, *, version: str = VERSION) -> bool:
    short, long = sorted((width, height))
    if long <= 1024:
        return short >= 384 and long >= 513
    return version in LARGE_VERSIONS and short >= 768 and long <= 2048


def search_geometry(width: int, height: int) -> dict:
    long = max(width, height)
    if long <= 1024:
        return {"basis": "source", "width": width, "height": height}
    scale = 1024 / long
    return {"basis": "reduced", "width": round(width * scale), "height": round(height * scale)}


def status_text(status: str, version: str) -> str:
    return LARGE_COVERAGE_TEXT if status == "unsupported_geometry" and version in LARGE_VERSIONS else STATUS_TEXT[status]


def validate(document, *, box_bounds, url, credential_status):
    version = document.get("version")
    if version not in AFFINE_VERSIONS:
        if KEY in document:
            raise ValueError("Geometric copy support requires its own evidence version")
        return
    source, record = document["source"], document.get(KEY)
    large = version in LARGE_VERSIONS
    fields = {"schema", "source_sha256", "status", "pairs", "authorship_confidence_contribution", "interpretation"}
    if large:
        fields.add("proposal_search")
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("Malformed geometric copy evidence")
    width, height = source["width"], source["height"]
    status, pairs = record["status"], record["pairs"]
    if (record["schema"] != (LARGE_SCHEMA if large else SCHEMA)
            or record["source_sha256"] != source["sha256"] or record["interpretation"] != NOTE
            or type(record["authorship_confidence_contribution"]) is not int or record["authorship_confidence_contribution"] != 0
            or status not in {*STATUS_TEXT, credential_status} or not isinstance(pairs, list) or len(pairs) > 3
            or bool(pairs) != (status == "supported_pair")
            or (status != credential_status and (status == "unsupported_geometry") == supported(width, height, version=version))):
        raise ValueError("Geometric copy evidence has invalid scope or confidence")
    if large:
        search = None if status in {credential_status, "unsupported_geometry"} else search_geometry(width, height)
        recorded = record["proposal_search"]
        if recorded != search or (search and any(type(recorded[key]) is not int for key in ("width", "height"))):
            raise ValueError("Geometric copy search uses a different working grid")
    for i, pair in enumerate(pairs, 1):
        if (set(pair) != {"id", "seed", "matrix", "regions", "selected_pixels", "artifacts"} or pair["id"] != f"G{i:02d}"
                or set(pair["regions"]) != set(ROLES) or set(pair["selected_pixels"]) != set(ROLES)
                or set(pair["artifacts"]) != {*ROLES, "mask_a", "mask_b"}):
            raise ValueError("Malformed geometric copy pair")
        box_bounds(pair["seed"], width, height)
        matrix = pair["matrix"]
        if (not isinstance(matrix, list) or len(matrix) != 2 or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
                or any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 8192
                       for row in matrix for value in row)):
            raise ValueError("Invalid geometric copy transform")
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        if not .06 <= abs(determinant) <= 16.01:
            raise ValueError("Geometric copy transform is outside its supported scale")
        for role in ROLES:
            region = pair["regions"][role]
            box_bounds(region, width, height)
            selected = pair["selected_pixels"][role]
            if type(selected) is not int or not 300 <= selected <= region["w"] * region["h"]:
                raise ValueError("Invalid geometric copy mask size")
            for name in (role, "mask_" + role[-1]):
                asset = pair["artifacts"][name]
                if (set(asset) != {"sha256", "media_type", "width", "height", "url"} or asset["media_type"] != "image/png"
                        or not _SHA.fullmatch(str(asset["sha256"])) or asset["url"] != url(document, asset["sha256"])
                        or (asset["width"], asset["height"]) != (region["w"], region["h"])):
                    raise ValueError("Invalid geometric copy crop or mask")
