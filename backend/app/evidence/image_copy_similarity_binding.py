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
SMALL_PNG_COPY_VERSION = "image-evidence-20"
SMALL_PNG_COPY_AFFINE_VERSION = "image-evidence-21"
TURNED_PNG_COPY_VERSION = "image-evidence-22"
TURNED_PNG_COPY_AFFINE_VERSION = "image-evidence-23"
TURNED_JPEG_COPY_VERSION = "image-evidence-24"
TURNED_JPEG_COPY_AFFINE_VERSION = "image-evidence-25"
EXPANDED_JPEG_ROTATION_VERSION = "image-evidence-26"
EXPANDED_JPEG_ROTATION_AFFINE_VERSION = "image-evidence-27"
NATIVE_JPEG_ROTATION_VERSION = "image-evidence-28"
NATIVE_JPEG_ROTATION_AFFINE_VERSION = "image-evidence-29"
DISJOINT_JPEG_ROTATION_VERSION = "image-evidence-30"
DISJOINT_JPEG_ROTATION_AFFINE_VERSION = "image-evidence-31"
WEBP_EXACT_COPY_VERSION = "image-evidence-32"
WEBP_EXACT_COPY_AFFINE_VERSION = "image-evidence-33"
WEBP_EXACT_REFLECTION_VERSION = "image-evidence-34"
WEBP_EXACT_REFLECTION_AFFINE_VERSION = "image-evidence-35"
WEBP_EXACT_REFLECTION_VERSIONS = frozenset({WEBP_EXACT_REFLECTION_VERSION, WEBP_EXACT_REFLECTION_AFFINE_VERSION})
WEBP_EXACT_COPY_VERSIONS = WEBP_EXACT_REFLECTION_VERSIONS | {WEBP_EXACT_COPY_VERSION, WEBP_EXACT_COPY_AFFINE_VERSION}
PNG_EXACT_REFLECTION_VERSION = "image-evidence-36"
PNG_EXACT_REFLECTION_AFFINE_VERSION = "image-evidence-37"
PNG_EXACT_REFLECTION_VERSIONS = frozenset({PNG_EXACT_REFLECTION_VERSION, PNG_EXACT_REFLECTION_AFFINE_VERSION})
EXACT_REFLECTION_VERSIONS = WEBP_EXACT_REFLECTION_VERSIONS | PNG_EXACT_REFLECTION_VERSIONS
DISJOINT_JPEG_ROTATION_VERSIONS = frozenset({DISJOINT_JPEG_ROTATION_VERSION, DISJOINT_JPEG_ROTATION_AFFINE_VERSION})
NATIVE_JPEG_ROTATION_VERSIONS = DISJOINT_JPEG_ROTATION_VERSIONS | {NATIVE_JPEG_ROTATION_VERSION, NATIVE_JPEG_ROTATION_AFFINE_VERSION}
DISJOINT_ROTATION_MARGIN = 16
TURNED_PNG_VERSIONS = PNG_EXACT_REFLECTION_VERSIONS | {TURNED_PNG_COPY_VERSION, TURNED_PNG_COPY_AFFINE_VERSION}
SMALL_PNG_VERSIONS = TURNED_PNG_VERSIONS | {SMALL_PNG_COPY_VERSION, SMALL_PNG_COPY_AFFINE_VERSION}
EXPANDED_JPEG_ROTATION_VERSIONS = NATIVE_JPEG_ROTATION_VERSIONS | {EXPANDED_JPEG_ROTATION_VERSION, EXPANDED_JPEG_ROTATION_AFFINE_VERSION}
TURNED_JPEG_VERSIONS = EXPANDED_JPEG_ROTATION_VERSIONS | {TURNED_JPEG_COPY_VERSION, TURNED_JPEG_COPY_AFFINE_VERSION}
SMALL_PNG_MIN_SIDE, SMALL_PNG_MAX_SIDE = 384, 768
SMALL_PNG_CROP_MIN, SMALL_PNG_CROP_MAX = 16, 56
WEBP_EXACT_COPY_MIN, WEBP_EXACT_COPY_MAX = 512, 768
SMALL_PNG_BASIS = "png_exact_small_copy"
TURNED_PNG_BASIS = "png_exact_turned_copy"
WEBP_EXACT_COPY_BASIS = "webp_exact_copy"
WEBP_EXACT_COPY_KEY = "webp_exact_copy_review"
WEBP_EXACT_REFLECTION_BASIS = "webp_exact_reflected_copy"
WEBP_EXACT_REFLECTION_KEY = "webp_exact_reflection_review"
PNG_EXACT_REFLECTION_BASIS = "png_exact_reflected_copy"
PNG_EXACT_REFLECTION_KEY = "png_exact_reflection_review"
REFLECTION_TRANSFORMS = ("horizontal_reflection", "vertical_reflection")
EXACT_REFLECTION_NOTE = (
    "The displayed crops match after reflection. Reflected copies or repeated graphic elements can produce a match. "
    "The comparison cannot establish which area came first or whether AI was involved."
)
PNG_COPY_TRANSFORMS = {0: "same_orientation", 1: "rotate_90_ccw", 2: "rotate_180"}
PNG_COPY_TURNS = {transform: turns for turns, transform in PNG_COPY_TRANSFORMS.items()}
SMALL_PNG_STATUS_TEXT = {
    "supported_pair": "Every RGB pixel in the two crops matches exactly. Pixels outside the crops differ on at least three sides.",
    "no_pair": "The small-copy check found no qualifying exact pair.",
    "unsupported_profile": "The small-copy check covers upright, static 8-bit RGB PNGs with sides from 384 to 768 pixels.",
    "unavailable": "The small-copy check could not complete.",
    "ambiguous_pairs": "Multiple exact pairs made this comparison ambiguous.",
    "unassessed_match_budget": "Too many competing offsets prevented a distinct small-copy comparison.",
}
WEBP_EXACT_COPY_STATUS_TEXT = {
    **SMALL_PNG_STATUS_TEXT,
    "no_pair": "The lossless WebP check found no qualifying exact pair with or without right-angle rotation.",
    "unsupported_profile": "Exact comparison requires an opaque lossless WebP still with sides from 512 to 768 pixels.",
}
WEBP_EXACT_REFLECTION_STATUS_TEXT = {
    **WEBP_EXACT_COPY_STATUS_TEXT,
    "no_pair": "The reflection check found no qualifying exact pair after a horizontal or vertical flip.",
    "unassessed_match_budget": "Too many matching areas prevented the reflection check from isolating a pair.",
    "ambiguous_pairs": "Several reflected pairs met the evidence requirements, so no single pair is shown.",
    "unavailable": "The exact reflection check could not complete.",
}
PNG_EXACT_REFLECTION_STATUS_TEXT = {
    **WEBP_EXACT_REFLECTION_STATUS_TEXT,
    "unsupported_profile": SMALL_PNG_STATUS_TEXT["unsupported_profile"],
}
JPEG_OFFGRID_VERSIONS = TURNED_JPEG_VERSIONS | {JPEG_OFFGRID_VERSION, JPEG_OFFGRID_AFFINE_VERSION}
JPEG_BLOCK_VERSIONS = JPEG_OFFGRID_VERSIONS | {JPEG_BLOCK_VERSION, JPEG_BLOCK_AFFINE_VERSION}
VERSION = V12
AFFINE_VERSIONS = frozenset({V13, V14, JPEG_BLOCK_AFFINE_VERSION, JPEG_OFFGRID_AFFINE_VERSION,
                             SMALL_PNG_COPY_AFFINE_VERSION, TURNED_PNG_COPY_AFFINE_VERSION,
                             TURNED_JPEG_COPY_AFFINE_VERSION, EXPANDED_JPEG_ROTATION_AFFINE_VERSION,
                             NATIVE_JPEG_ROTATION_AFFINE_VERSION, DISJOINT_JPEG_ROTATION_AFFINE_VERSION,
                             WEBP_EXACT_COPY_AFFINE_VERSION, WEBP_EXACT_REFLECTION_AFFINE_VERSION, PNG_EXACT_REFLECTION_AFFINE_VERSION})
CREDENTIAL_VERSIONS = AFFINE_VERSIONS | {V12} | JPEG_BLOCK_VERSIONS | SMALL_PNG_VERSIONS | WEBP_EXACT_COPY_VERSIONS
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
TURNED_JPEG_BASIS = "jpeg_turned_copy"
OFFGRID_STATUS_TEXT = {
    "supported_pair": "The aligned crops have closely matching color values inside and more different surroundings on at least three sides.",
    "no_pair": ("The between-block JPEG check found no qualifying repeated area. "
                "The minimum crop side is 32 or 56 pixels, depending on the JPEG compression profile."),
    "unsupported_profile": "The between-block check covers upright color JPEGs with the tested standard compression tables.",
    "unavailable": "The between-block JPEG check could not complete.",
    "unassessed_match_budget": "Too many competing offsets prevented a distinct between-block comparison.",
}
TURNED_JPEG_STATUS_TEXT = {
    "supported_pair": "After alignment, the crops have similar color inside and more different surroundings on at least three sides.",
    "no_pair": "The JPEG rotation check found no qualifying pair after a quarter turn or half turn.",
    "unsupported_profile": "The JPEG rotation check covers upright color JPEGs within the tested compression settings.",
    "unavailable": "The JPEG rotation check could not complete.",
    "ambiguous": "More than one pair fit the JPEG rotation check, so it could not select distinct areas.",
    "unassessed_match_budget": "Too many matching points prevented a distinct JPEG rotation comparison.",
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


def native_rotation_size(width: int, height: int) -> bool:
    short, long = sorted((width, height))
    return short >= 1024 and 1024 < long <= 3072


def exact_copy_size(width: int, height: int, *, webp: bool) -> bool:
    lower, upper = (WEBP_EXACT_COPY_MIN, WEBP_EXACT_COPY_MAX) if webp else (SMALL_PNG_MIN_SIDE, SMALL_PNG_MAX_SIDE)
    return lower <= min(width, height) <= max(width, height) <= upper


def disjoint_copy_regions(first, second) -> bool:
    gap = DISJOINT_ROTATION_MARGIN
    return all(ax + aw + gap <= bx or bx + bw + gap <= ax or ay + ah + gap <= by or by + bh + gap <= ay
               for ax, ay, aw, ah in first for bx, by, bw, bh in second)


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
    return _validate_jpeg_measurement(measurement)


def rotation_review_follows(offgrid_review: dict | None, *, expanded: bool) -> bool:
    if offgrid_review is None:
        return False
    status = offgrid_review["status"]
    return status == "no_pair" or expanded and status == "unsupported_profile"


def _validate_turned_jpeg(record: dict | None, offgrid_review: dict | None, *, expanded: bool, native: bool,
                          allow_disjoint: bool) -> int:
    should_run = native or rotation_review_follows(offgrid_review, expanded=expanded)
    if record is None and not should_run:
        return 0
    if not should_run or not isinstance(record, dict):
        raise ValueError("Rotated JPEG coverage does not follow its recorded between-block comparison")
    if set(record) == {"status", "measurement", "measurements"}:
        values = record["measurements"]
        if (not native or not allow_disjoint or record["status"] != "supported_pair" or record["measurement"] is not None
                or not isinstance(values, list) or len(values) != 2 or not all(_validate_jpeg_measurement(value) for value in values)):
            raise ValueError("Two distinct rotations require two measured pairs on the native JPEG grid")
        return 2
    if set(record) != {"status", "measurement"}:
        raise ValueError("Rotated JPEG coverage has unexpected fields")
    status, measurement = record["status"], record["measurement"]
    if status not in TURNED_JPEG_STATUS_TEXT or (status == "supported_pair") != (measurement is not None):
        raise ValueError("Rotated JPEG comparison has an invalid status")
    return int(_validate_jpeg_measurement(measurement))


def _validate_jpeg_measurement(measurement: dict | None) -> bool:
    if measurement is None:
        return False
    if not isinstance(measurement, dict) or set(measurement) != {"interior_mean", "interior_p90", "perimeter_means"}:
        raise ValueError("JPEG copy comparison has an invalid measurement")
    mean, p90, perimeter = (measurement[k] for k in ("interior_mean", "interior_p90", "perimeter_means"))
    if not (_rgb_level(mean, 3.2) and _rgb_level(p90, 6) and isinstance(perimeter, list) and len(perimeter) == 4
            and all(value is None or _rgb_level(value, 255) for value in perimeter)):
        raise ValueError("JPEG copy differences are outside the measured limits")
    # Recorded levels are rounded to five decimal places.
    if sum(value is not None and value + 0.00005 >= max(7, 2.5 * mean) for value in perimeter) < 3:
        raise ValueError("JPEG copy comparison lacks surrounding contrast")
    return True


def _rgb_level(value: object, ceiling: float) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and 0 <= value <= ceiling


def _validate_exact_copy(review) -> bool:
    if review is None:
        return False
    if not isinstance(review, dict) or set(review) != {"status", "measurement"}:
        raise ValueError("Exact copy comparison has an invalid record")
    status, measurement = review["status"], review["measurement"]
    if status not in SMALL_PNG_STATUS_TEXT or (status == "supported_pair") != (measurement is not None):
        raise ValueError("Exact copy comparison has an invalid status")
    if measurement is None:
        return False
    if not isinstance(measurement, dict) or set(measurement) != {"matching_pixels", "different_color_values",
                                                               "perimeter_mean_absolute_difference"}:
        raise ValueError("Exact copy comparison has an invalid measurement")
    count, colors, rim = (measurement[key] for key in ("matching_pixels", "different_color_values",
                                                        "perimeter_mean_absolute_difference"))
    if (type(count) is not int or type(colors) is not int or not 64 <= colors <= count
            or not isinstance(rim, list) or len(rim) != 4
            or any(type(value) not in (int, float) or not 0 <= value <= 255 for value in rim)
            or sum(value >= 3.0 for value in rim) < 3):
        raise ValueError("Exact copy comparison has invalid pixel or perimeter values")
    return True


def _validate_check(document: dict, record: dict, size: tuple[int, int], supported: bool, *, reflected: bool) -> None:
    has_basis = not reflected and document["version"] in BASIS_VERSIONS
    has_jpeg_review = not reflected and document["version"] in JPEG_BLOCK_VERSIONS
    has_offgrid_review = not reflected and document["version"] in JPEG_OFFGRID_VERSIONS
    has_turned_jpeg_review = not reflected and document["version"] in TURNED_JPEG_VERSIONS
    webp = not reflected and document["version"] in WEBP_EXACT_COPY_VERSIONS
    reflection_review = not reflected and document["version"] in EXACT_REFLECTION_VERSIONS
    reflection_key = WEBP_EXACT_REFLECTION_KEY if webp else PNG_EXACT_REFLECTION_KEY
    reflection_basis = WEBP_EXACT_REFLECTION_BASIS if webp else PNG_EXACT_REFLECTION_BASIS
    has_exact_review = webp or (not reflected and document["version"] in SMALL_PNG_VERSIONS)
    exact_key = WEBP_EXACT_COPY_KEY if webp else "small_png_copy_review"
    exact_size = has_exact_review and exact_copy_size(*size, webp=webp)
    native_rotation = not reflected and document["version"] in NATIVE_JPEG_ROTATION_VERSIONS and native_rotation_size(*size)
    supported |= native_rotation
    supported |= exact_size
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
    if has_turned_jpeg_review:
        fields.add("jpeg_turned_review")
    if has_exact_review:
        fields.add(exact_key)
    if reflection_review:
        fields.add(reflection_key)
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
        or record["interpretation"] != (NOTE if reflected else EXACT_REFLECTION_NOTE
                                       if reflection_review and record.get("basis") == reflection_basis else TRANSLATION_NOTE)
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
    turned_jpeg_pair = has_turned_jpeg_review and _validate_turned_jpeg(
        record["jpeg_turned_review"], record["jpeg_offgrid_review"], expanded=document["version"] in EXPANDED_JPEG_ROTATION_VERSIONS,
        native=native_rotation and not skipped, allow_disjoint=document["version"] in DISJOINT_JPEG_ROTATION_VERSIONS)
    if native_rotation:
        if record["jpeg_block_review"] is not None or record["jpeg_offgrid_review"] is not None:
            raise ValueError("Native rotation evidence cannot attest to the smaller JPEG comparisons")
        nested = record["jpeg_turned_review"]
        if (not skipped and status != ("supported_pair" if turned_jpeg_pair else "no_pair")
                or nested is not None and nested["status"] != "unsupported_profile"
                and document["source"]["exif_orientation"] != 1):
            raise ValueError("Native rotation evidence differs from its recorded source or result")
    if turned_jpeg_pair and len(pairs) != turned_jpeg_pair:
        raise ValueError("Rotated JPEG evidence must contain the recorded measured pairs")
    reflected_exact_pair = False
    if reflection_review and (reflection := record[reflection_key]) is not None:
        reflected_exact_pair = _validate_exact_copy(reflection)
        legacy_review = record[exact_key]
        previous = "unsupported_profile" if reflection["status"] == "unsupported_profile" else "no_pair"
        if (not exact_size or not legacy_review or legacy_review["status"] != previous
                or document["copy_similarity"]["status"] not in {"no_pair", "unsupported_geometry"}
                or (document.get("copy_affine") or {}).get("status") not in {None, "no_pair", "unsupported_geometry"}
                or status != ("supported_pair" if reflected_exact_pair else "no_pair")
                or (reflected_exact_pair and len(pairs) != 1)):
            raise ValueError("Reflected exact copy coverage differs from the completed source checks")
    exact_pair = False
    if has_exact_review and (review := record[exact_key]) is not None:
        exact_pair = _validate_exact_copy(review)
        if (not exact_size
                or (review["status"] != "unsupported_profile" and document["source"]["exif_orientation"] != 1)
                or status != ("supported_pair" if exact_pair or reflected_exact_pair else "no_pair")
                or (exact_pair and len(pairs) != 1)):
            raise ValueError("Exact copy coverage differs from the native source or recorded result")
    if has_basis:
        basis = record["basis"]
        expected = ("native_patch_alignment", "geometric_features") if pairs else (None,)
        if has_jpeg_review and record["jpeg_block_review"] == "supported_pair":
            expected = (JPEG_BLOCK_BASIS,)
        if offgrid_pair:
            expected = (OFFGRID_BASIS,)
        if turned_jpeg_pair:
            expected = (TURNED_JPEG_BASIS,)
        if exact_pair:
            if webp:
                expected = (WEBP_EXACT_COPY_BASIS,)
            else:
                expected = ((SMALL_PNG_BASIS, TURNED_PNG_BASIS) if document["version"] in TURNED_PNG_VERSIONS
                            else (SMALL_PNG_BASIS,))
        if reflected_exact_pair:
            expected = (reflection_basis,)
        if basis not in expected:
            raise ValueError("Copy similarity has an invalid matching basis")
    jpeg_pair = has_jpeg_review and record["jpeg_block_review"] == "supported_pair"
    if has_jpeg_review:
        jpeg_status = record["jpeg_block_review"]
        if (jpeg_status is not None and (jpeg_status not in JPEG_BLOCK_STATUS_TEXT
                or (status != "no_pair" and not (status == "supported_pair" and (jpeg_pair or offgrid_pair or turned_jpeg_pair)))
                or (jpeg_status != "unsupported_profile" and document["source"]["exif_orientation"] != 1))
                or jpeg_pair and not pairs):
            raise ValueError("JPEG block coverage differs from the completed copy check")
    measured_boxes = []
    for index, pair in enumerate(pairs, 1):
        turned = record.get("basis") in {TURNED_PNG_BASIS, TURNED_JPEG_BASIS}
        transforms = (("horizontal_reflection",) if reflected else
                      REFLECTION_TRANSFORMS if reflected_exact_pair else
                      tuple(PNG_COPY_TRANSFORMS.values()) if record.get("basis") == WEBP_EXACT_COPY_BASIS else
                      (PNG_COPY_TRANSFORMS[1], PNG_COPY_TRANSFORMS[2]) if turned else ("same_orientation",))
        if (
            not isinstance(pair, dict)
            or set(pair) != {"id", "transform", "regions", "artifacts"}
            or pair["id"] != f"{'R' if reflected else 'T'}{index:02d}"
            or pair["transform"] not in transforms
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
        equal_size = (reflected or offgrid_pair or exact_pair or reflected_exact_pair or turned_jpeg_pair
                      or record.get("basis") in {"native_patch_alignment", JPEG_BLOCK_BASIS})
        aligned_size = (bh, bw) if pair["transform"] == PNG_COPY_TRANSFORMS[1] else (bw, bh)
        if overlaps or (equal_size and (aw, ah) != aligned_size):
            raise ValueError("The paired areas must be separate and aligned areas must share a native size")
        measurement_key = reflection_key if reflected_exact_pair else exact_key
        if (exact_pair or reflected_exact_pair) and (not SMALL_PNG_CROP_MIN <= min(aw, ah) <= max(aw, ah) <= SMALL_PNG_CROP_MAX
                           or record[measurement_key]["measurement"]["matching_pixels"] != aw * ah
                           or (pair["transform"] == PNG_COPY_TRANSFORMS[0]
                               and pair["artifacts"]["area_a"]["sha256"] != pair["artifacts"]["area_b"]["sha256"])):
            carrier = "Lossless WebP" if webp else "Small PNG"
            raise ValueError(f"{carrier} copy pair differs from its exact crop geometry or pixels")
        if jpeg_pair and (any(value % 8 for box in boxes for value in box) or not 24 <= min(aw, ah) <= max(aw, ah) <= 128):
            raise ValueError("JPEG copy areas do not follow the supported encoded block grid")
        if offgrid_pair and (not ((bx - ax) % 8 or (by - ay) % 8)
                             or not OFFGRID_MIN_SIDE <= min(aw, ah) <= max(aw, ah) <= OFFGRID_MAX_SIDE):
            raise ValueError("Between-block JPEG pair is outside its measured geometry")
        if turned_jpeg_pair and not OFFGRID_MIN_SIDE <= min(aw, ah) <= max(aw, ah) <= OFFGRID_MAX_SIDE:
            raise ValueError("Rotated JPEG pair is outside its measured geometry")
        measured_boxes.append(boxes)
    if turned_jpeg_pair == 2 and not disjoint_copy_regions(*measured_boxes):
        raise ValueError("The two rotated copy pairs must have four distinct source crops")
