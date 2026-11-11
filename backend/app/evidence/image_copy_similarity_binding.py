"""Stdlib structural validation for committed automatic copy-similarity evidence."""

from __future__ import annotations

import re
from collections.abc import Callable

VERSION = "image-evidence-7"
SCHEMA = "same-image-reflection-similarity/1"
NOTE = (
    "Two areas may have similar detail after a horizontal reflection. Copying, intentional symmetry, reflections "
    "and repeated patterns can all produce this result. The comparison cannot identify which area came first "
    "or establish that the image was edited or made with AI."
)
STATUS_TEXT = {
    "supported_pair": "Similar areas were found after horizontal reflection.",
    "no_pair": "This check found no qualifying reflected pair.",
    "unsupported_geometry": "This check requires native dimensions of 384 px or more on both sides and a long side of 513–1,024 px.",
    "unassessed_match_budget": "The check found too many candidate matches to show a distinct comparison.",
}
_SHA = re.compile(r"[a-f0-9]{64}")
ROLES = ("area_a", "area_b")

MIN_SIDE = 384
MIN_LONG_SIDE = 513
MAX_SIDE = 1024
PAIR_LIMIT = 3


def supported_size(width: int, height: int) -> bool:
    return min(width, height) >= MIN_SIDE and MIN_LONG_SIDE <= max(width, height) <= MAX_SIDE


def box_bounds(box: dict, width: int, height: int) -> tuple[int, int, int, int]:
    if (
        not isinstance(box, dict)
        or set(box) != {"x", "y", "w", "h"}
        or any(type(v) is not int for v in box.values())
    ):
        raise ValueError("Copy similarity needs exact source-pixel coordinates")
    x, y, w, h = (box[k] for k in ("x", "y", "w", "h"))
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError("Copy similarity region is outside the submitted image")
    return x, y, x + w, y + h


def url(document: dict, digest: str) -> str:
    return f"/api/v2/analyses/{document['analysis_id']}/evidence-packets/{document['id']}/images/{digest}.png"


def prior_evidence(document: dict) -> dict:
    base = dict(document)
    base.pop("copy_similarity", None)
    lineage = base.pop("prior_image_evidence")
    base.update(lineage)
    return base


def validate_similarity(document: dict, canonical_hash: Callable[[dict], str]) -> None:
    """Validate standalone public structure and both exact native-crop references."""
    if document.get("version") != VERSION:
        if {"copy_similarity", "prior_image_evidence"} & document.keys():
            raise ValueError("Earlier image evidence cannot contain automatic copy similarity")
        return
    lineage = document.get("prior_image_evidence")
    if (
        not isinstance(lineage, dict)
        or set(lineage) != {"id", "version", "sha256"}
        or lineage["version"] != "image-evidence-6"
        or lineage["id"] == document.get("id")
        or not _SHA.fullmatch(str(lineage["sha256"]))
    ):
        raise ValueError("Copy similarity lacks frozen prior evidence")
    base = prior_evidence(document)
    if canonical_hash({k: v for k, v in base.items() if k != "sha256"}) != lineage["sha256"]:
        raise ValueError("Copy similarity changes the frozen prior evidence")
    source = document["source"]
    size = source["width"], source["height"]
    supported = supported_size(*size)
    record = document.get("copy_similarity")
    if not isinstance(record, dict) or set(record) != {
        "schema",
        "source_sha256",
        "coordinate_space",
        "status",
        "authorship_confidence_contribution",
        "pairs",
        "interpretation",
    }:
        raise ValueError("Copy similarity has an invalid record")
    status, pairs = record["status"], record["pairs"]
    if (
        record["schema"] != SCHEMA
        or record["source_sha256"] != source["sha256"]
        or record["coordinate_space"] != "exif-oriented-original"
        or type(record["authorship_confidence_contribution"]) is not int
        or record["authorship_confidence_contribution"] != 0
        or record["interpretation"] != NOTE
        or status not in STATUS_TEXT
        or (status == "unsupported_geometry") == supported
        or not isinstance(pairs, list)
        or len(pairs) > PAIR_LIMIT
        or bool(pairs) != (status == "supported_pair")
    ):
        raise ValueError("Copy similarity has an inconsistent source, result or coverage")
    for index, pair in enumerate(pairs, 1):
        if (
            not isinstance(pair, dict)
            or set(pair) != {"id", "transform", "regions", "artifacts"}
            or pair["id"] != f"R{index:02d}"
            or pair["transform"] != "horizontal_reflection"
            or not isinstance(pair["regions"], dict)
            or not isinstance(pair["artifacts"], dict)
            or set(pair["regions"]) != set(ROLES)
            or set(pair["artifacts"]) != set(ROLES)
        ):
            raise ValueError("Copy similarity has malformed paired areas")
        boxes = []
        for role in ROLES:
            region, asset = pair["regions"][role], pair["artifacts"][role]
            bounds = box_bounds(region, *size)
            boxes.append((bounds[0], bounds[1], region["w"], region["h"]))
            if (
                not isinstance(asset, dict)
                or asset.get("media_type") != "image/png"
                or not _SHA.fullmatch(str(asset.get("sha256")))
                or (asset.get("width"), asset.get("height")) != (region["w"], region["h"])
                or asset.get("url") != url(document, asset["sha256"])
            ):
                raise ValueError("Copy similarity has an invalid native crop")
        ax, ay, aw, ah = boxes[0]
        bx, by, bw, bh = boxes[1]
        overlaps = ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah
        if (aw, ah) != (bw, bh) or overlaps:
            raise ValueError("The paired areas must be separate and share a native size")
