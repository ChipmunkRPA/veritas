"""Stdlib structural validation for committed automatic copy-similarity evidence."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

V7 = "image-evidence-7"
V8 = "image-evidence-8"
V9 = "image-evidence-9"
V10 = "image-evidence-10"
V11 = "image-evidence-11"
V12 = "image-evidence-12"
V13 = "image-evidence-13"
VERSION = V12
TRANSLATION_VERSIONS = frozenset({V10, V11, V12, V13})
BASIS_VERSIONS = frozenset({V11, V12, V13})
REFLECTION_VERSIONS = frozenset({V7, V8, V9}) | TRANSLATION_VERSIONS
LARGE_IMAGE_VERSIONS = TRANSLATION_VERSIONS | {V9}
SCHEMA = "same-image-reflection-similarity/1"
TRANSLATION_SCHEMA = "same-image-copy-similarity/1"
SKIPPED_CREDENTIALS = "skipped_trusted_ai_credentials"
SKIPPED_CREDENTIALS_TEXT = "Trusted Content Credentials cover this image and declare AI involvement. The two copy checks were skipped."
TRANSLATION_NOTE = (
    "The two crops show similar image detail without mirroring. Copies and naturally repeated detail can both produce "
    "a match. The crop boundaries enclose the matched detail. The origin of either area and AI involvement remain undetermined."
)
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
    SKIPPED_CREDENTIALS: SKIPPED_CREDENTIALS_TEXT,
}
LARGE_COVERAGE_TEXT = (
    "This check covers images with a long side of 513–1,024 px and a short side of at least 384 px, "
    "or a long side of 1,025–2,048 px and a short side of at least 768 px."
)
_SHA = re.compile(r"[a-f0-9]{64}")
ROLES = ("area_a", "area_b")

MIN_SIDE = 384
MIN_LONG_SIDE = 513
MAX_SIDE = 1024
LARGE_MIN_SIDE = 768
LARGE_MAX_SIDE = 2048
LARGE_MAX_PIXELS = 4_194_304
PAIR_LIMIT = 3


def supported_size(width: int, height: int, *, version: str = VERSION) -> bool:
    short, long = sorted((width, height))
    if long <= MAX_SIDE:
        return short >= MIN_SIDE and long >= MIN_LONG_SIDE
    return version in LARGE_IMAGE_VERSIONS and short >= LARGE_MIN_SIDE and long <= LARGE_MAX_SIDE and width * height <= LARGE_MAX_PIXELS


def status_text(status: str, *, version: str) -> str:
    if status == "unsupported_geometry" and version in LARGE_IMAGE_VERSIONS:
        return LARGE_COVERAGE_TEXT
    return STATUS_TEXT[status]


def credential_skip(source_sha256: str) -> dict:
    return {"status": SKIPPED_CREDENTIALS, "source_sha256": source_sha256}


def result_skip(result: dict) -> dict | None:
    """Read only the explicit, source-bound scope committed by the verified gate."""
    scope = ((result.get("manifest") or {}).get("sampling") or {}).get("image_copy_coverage")
    if scope is None:
        return None
    source = (result.get("input") or {}).get("sha256")
    decision = (result.get("decision") or {}).get("provenance_record") or {}
    if not isinstance(source, str) or not _SHA.fullmatch(source) or scope != credential_skip(source) or decision.get("kind") != "decisive_ai":
        raise ValueError("Copy check coverage has no source-bound signed AI declaration")
    return scope


def validate_copy_scope(document: dict, scope: dict | None) -> None:
    """Bind the presented skip to the result and require both checks to agree."""
    if document.get("version") not in {V12, V13}:
        if scope is not None:
            raise ValueError("Earlier image evidence has no recorded copy check coverage")
        return
    if scope is not None and scope != credential_skip(document["source"]["sha256"]):
        raise ValueError("Copy check coverage differs from the source-bound result")
    if any((record["status"] == SKIPPED_CREDENTIALS) != (scope is not None) for record in copy_records(document)):
        raise ValueError("Copy check coverage differs from the source-bound result")


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
    base.pop("copy_translation", None)
    base.pop("copy_affine", None)
    lineage = base.pop("prior_image_evidence")
    base.update(lineage)
    return base


def validate_similarity(document: dict, canonical_hash: Callable[[dict], str]) -> None:
    """Validate standalone public structure and both exact native-crop references."""
    version = document.get("version")
    if ("copy_translation" in document) != (version in TRANSLATION_VERSIONS):
        raise ValueError("The same-orientation copy check differs from the evidence version")
    if version not in REFLECTION_VERSIONS:
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
    _validate_check(document, document.get("copy_similarity"), size, supported_size(*size, version=version), reflected=True)
    if version in TRANSLATION_VERSIONS:
        _validate_check(document, document["copy_translation"], size, supported_size(*size, version=V8), reflected=False)
    if version in {V12, V13}:
        records = tuple(copy_records(document))
        if len({record["status"] == SKIPPED_CREDENTIALS for record in records}) != 1:
            raise ValueError("The copy checks have different signed credential coverage")


def copy_records(document: dict) -> Iterator[dict]:
    if document.get("version") in REFLECTION_VERSIONS:
        yield document["copy_similarity"]
    if document.get("version") in TRANSLATION_VERSIONS:
        yield document["copy_translation"]
    if document.get("version") == V13:
        yield document["copy_affine"]


def _validate_check(document: dict, record: dict, size: tuple[int, int], supported: bool, *, reflected: bool) -> None:
    has_basis = not reflected and document["version"] in BASIS_VERSIONS
    fields = {
        "schema",
        "source_sha256",
        "coordinate_space",
        "status",
        "authorship_confidence_contribution",
        "pairs",
        "interpretation",
    }
    if has_basis:
        fields.add("basis")
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("Copy similarity has an invalid record")
    status, pairs = record["status"], record["pairs"]
    skipped = status == SKIPPED_CREDENTIALS
    if (
        record["schema"] != (SCHEMA if reflected else TRANSLATION_SCHEMA)
        or record["source_sha256"] != document["source"]["sha256"]
        or record["coordinate_space"] != "exif-oriented-original"
        or type(record["authorship_confidence_contribution"]) is not int
        or record["authorship_confidence_contribution"] != 0
        or record["interpretation"] != (NOTE if reflected else TRANSLATION_NOTE)
        or status not in STATUS_TEXT
        or (skipped and document["version"] not in {V12, V13})
        or (not skipped and (status == "unsupported_geometry") == supported)
        or not isinstance(pairs, list)
        or len(pairs) > PAIR_LIMIT
        or bool(pairs) != (status == "supported_pair")
    ):
        raise ValueError("Copy similarity has an inconsistent source, result or coverage")
    if has_basis:
        basis = record["basis"]
        expected = ("native_patch_alignment", "geometric_features") if pairs else (None,)
        if basis not in expected:
            raise ValueError("Copy similarity has an invalid matching basis")
    for index, pair in enumerate(pairs, 1):
        if (
            not isinstance(pair, dict)
            or set(pair) != {"id", "transform", "regions", "artifacts"}
            or pair["id"] != f"{'R' if reflected else 'T'}{index:02d}"
            or pair["transform"] != ("horizontal_reflection" if reflected else "same_orientation")
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
        equal_size = reflected or record.get("basis") == "native_patch_alignment"
        if overlaps or (equal_size and (aw, ah) != (bw, bh)):
            raise ValueError("The paired areas must be separate and aligned areas must share a native size")
