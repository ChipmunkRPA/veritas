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
V14 = "image-evidence-14"
JPEG_BLOCK_VERSION = "image-evidence-16"
JPEG_BLOCK_AFFINE_VERSION = "image-evidence-17"
JPEG_OFFGRID_VERSION = "image-evidence-18"
JPEG_OFFGRID_AFFINE_VERSION = "image-evidence-19"
JPEG_OFFGRID_VERSIONS = frozenset({JPEG_OFFGRID_VERSION, JPEG_OFFGRID_AFFINE_VERSION})
JPEG_BLOCK_VERSIONS = JPEG_OFFGRID_VERSIONS | {JPEG_BLOCK_VERSION, JPEG_BLOCK_AFFINE_VERSION}
VERSION = V12
AFFINE_VERSIONS = frozenset({V13, V14, JPEG_BLOCK_AFFINE_VERSION, JPEG_OFFGRID_AFFINE_VERSION})
CREDENTIAL_VERSIONS = AFFINE_VERSIONS | {V12, JPEG_BLOCK_VERSION, JPEG_OFFGRID_VERSION}
TRANSLATION_VERSIONS = CREDENTIAL_VERSIONS | {V10, V11}
BASIS_VERSIONS = CREDENTIAL_VERSIONS | {V11}
REFLECTION_VERSIONS = frozenset({V7, V8, V9}) | TRANSLATION_VERSIONS
LARGE_IMAGE_VERSIONS = TRANSLATION_VERSIONS | {V9}
SCHEMA = "same-image-reflection-similarity/1"
TRANSLATION_SCHEMA = "same-image-copy-similarity/1"
SKIPPED_CREDENTIALS = "skipped_trusted_ai_credentials"
JPEG_BLOCK_BASIS = "jpeg_brightness_blocks"
JPEG_BLOCK_STATUS_TEXT = {
    "supported_pair": "The complete encoded brightness blocks matched exactly in both areas.",
    "no_pair": "The JPEG brightness-block check found no qualifying repeated area.",
    "unsupported_profile": "The brightness-block check requires an upright JPEG within its tested size and encoding scope.",
    "unavailable": "The JPEG brightness-block check could not read the encoded blocks.",
    "unassessed_match_budget": "The brightness-block check found too many competing offsets.",
}
OFFGRID_MIN_SIDE = 32
OFFGRID_MAX_SIDE = 118
OFFGRID_BASIS = "jpeg_offgrid_texture"
OFFGRID_STATUS_TEXT = {
    "supported_pair": "The aligned crops have closely matching color values inside and more different surroundings on at least three sides.",
    "no_pair": "The between-block JPEG check found no qualifying repeated area at least 32 pixels wide and tall.",
    "unsupported_profile": "The between-block check covers upright color JPEGs with the tested standard compression tables.",
    "unavailable": "The between-block JPEG check could not complete.",
    "unassessed_match_budget": "Too many competing offsets prevented a distinct between-block comparison.",
}
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
    if document.get("version") not in CREDENTIAL_VERSIONS:
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
    if version in CREDENTIAL_VERSIONS:
        records = tuple(copy_records(document))
        if len({record["status"] == SKIPPED_CREDENTIALS for record in records}) != 1:
            raise ValueError("The copy checks have different signed credential coverage")


def copy_records(document: dict) -> Iterator[dict]:
    if document.get("version") in REFLECTION_VERSIONS:
        yield document["copy_similarity"]
    if document.get("version") in TRANSLATION_VERSIONS:
        yield document["copy_translation"]
    if document.get("version") in AFFINE_VERSIONS:
        yield document["copy_affine"]


def _validate_offgrid(record: dict | None, jpeg_status: str | None) -> bool:
    if record is None:
        if jpeg_status == "no_pair":
            raise ValueError("Completed JPEG block comparison lacks its between-block coverage")
        return False
    if not isinstance(record, dict) or set(record) != {"status", "measurement"} or jpeg_status != "no_pair":
        raise ValueError("Between-block JPEG coverage does not follow the completed block comparison")
    status, measurement = record["status"], record["measurement"]
    if status not in OFFGRID_STATUS_TEXT or (status == "supported_pair") != (measurement is not None):
        raise ValueError("Between-block JPEG comparison has an invalid status")
    if measurement is None:
        return False
    if not isinstance(measurement, dict) or set(measurement) != {"interior_mean", "interior_p90", "perimeter_means"}:
        raise ValueError("Between-block JPEG comparison has an invalid measurement")
    mean, p90, perimeter = (measurement[k] for k in ("interior_mean", "interior_p90", "perimeter_means"))
    if not (_rgb_level(mean, 3.2) and _rgb_level(p90, 6) and isinstance(perimeter, list) and len(perimeter) == 4
            and all(value is None or _rgb_level(value, 255) for value in perimeter)):
        raise ValueError("Between-block JPEG differences are outside the measured limits")
    # Recorded levels are rounded to five decimal places.
    if sum(value is not None and value + 0.00005 >= max(7, 2.5 * mean) for value in perimeter) < 3:
        raise ValueError("Between-block JPEG comparison lacks surrounding contrast")
    return True


def _rgb_level(value: object, ceiling: float) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and 0 <= value <= ceiling


def _validate_check(document: dict, record: dict, size: tuple[int, int], supported: bool, *, reflected: bool) -> None:
    has_basis = not reflected and document["version"] in BASIS_VERSIONS
    has_jpeg_review = not reflected and document["version"] in JPEG_BLOCK_VERSIONS
    has_offgrid_review = not reflected and document["version"] in JPEG_OFFGRID_VERSIONS
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
    if has_jpeg_review:
        fields.add("jpeg_block_review")
    if has_offgrid_review:
        fields.add("jpeg_offgrid_review")
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
        or (skipped and document["version"] not in CREDENTIAL_VERSIONS)
        or (not skipped and (status == "unsupported_geometry") == supported)
        or not isinstance(pairs, list)
        or len(pairs) > PAIR_LIMIT
        or bool(pairs) != (status == "supported_pair")
    ):
        raise ValueError("Copy similarity has an inconsistent source, result or coverage")
    offgrid_pair = False
    if has_offgrid_review:
        offgrid_pair = _validate_offgrid(record["jpeg_offgrid_review"], record["jpeg_block_review"])
        if record["jpeg_offgrid_review"] is not None and (not supported or document["source"]["exif_orientation"] != 1):
            raise ValueError("Between-block JPEG coverage is outside its native source geometry")
    if offgrid_pair and len(pairs) != 1:
        raise ValueError("Between-block JPEG evidence must contain one measured pair")
    if has_basis:
        basis = record["basis"]
        expected = ("native_patch_alignment", "geometric_features") if pairs else (None,)
        if has_jpeg_review and record["jpeg_block_review"] == "supported_pair":
            expected = (JPEG_BLOCK_BASIS,)
        if offgrid_pair:
            expected = (OFFGRID_BASIS,)
        if basis not in expected:
            raise ValueError("Copy similarity has an invalid matching basis")
    jpeg_pair = has_jpeg_review and record["jpeg_block_review"] == "supported_pair"
    if has_jpeg_review:
        jpeg_status = record["jpeg_block_review"]
        if (jpeg_status is not None and (jpeg_status not in JPEG_BLOCK_STATUS_TEXT
                or (status != "no_pair" and not (status == "supported_pair" and (jpeg_pair or offgrid_pair)))
                or (jpeg_status != "unsupported_profile" and document["source"]["exif_orientation"] != 1))
                or jpeg_pair and not pairs):
            raise ValueError("JPEG block coverage differs from the completed copy check")
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
        equal_size = reflected or offgrid_pair or record.get("basis") in {"native_patch_alignment", JPEG_BLOCK_BASIS}
        if overlaps or (equal_size and (aw, ah) != (bw, bh)):
            raise ValueError("The paired areas must be separate and aligned areas must share a native size")
        if jpeg_pair and (any(value % 8 for box in boxes for value in box) or not 24 <= min(aw, ah) <= max(aw, ah) <= 128):
            raise ValueError("JPEG copy areas do not follow the supported encoded block grid")
        if offgrid_pair and (not ((bx - ax) % 8 or (by - ay) % 8)
                             or not OFFGRID_MIN_SIDE <= min(aw, ah) <= max(aw, ah) <= OFFGRID_MAX_SIDE):
            raise ValueError("Between-block JPEG pair is outside its measured geometry")
