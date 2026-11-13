"""Model-free bindings for a native image interpolation observation."""
from __future__ import annotations

import math
import re

from app.core.canonical import canonical_hash

KEY = "native_interpolation_review"
COMMITMENT = "native_interpolation_review_sha256"
SCHEMA = "image-native-interpolation/1"
EXTENDED_SCHEMA = "image-native-interpolation/2"
ROTATION_SCHEMA = "image-native-interpolation/3"
QUANTIZED_SCHEMA = "image-native-interpolation/4"
SCOPE = "This check requires a 2048 × 1536, 16-bit RGB PNG without ancillary chunks and a repeatable interpolation pattern."
ROTATION_SCOPE = SCOPE + " It also looks for local interpolation consistent with a small rotation."
QUANTIZED_SCOPE = ROTATION_SCOPE + " This check also compares local stored-value steps with the remaining pixel pattern."
EXTENDED_SCOPE = (
    "This check requires a 16-bit RGB PNG without ancillary chunks, a repeatable interpolation pattern, "
    "and one of these sizes: 2048 × 1536, 2304 × 1728, 2368 × 1344 or 1792 × 1792."
)
LEGACY_SIZE = (2048, 1536)
SUPPORTED_SIZES = frozenset({LEGACY_SIZE, (2304, 1728), (2368, 1344), (1792, 1792)})
SCOPES = {SCHEMA: SCOPE, EXTENDED_SCHEMA: EXTENDED_SCOPE, ROTATION_SCHEMA: ROTATION_SCOPE, QUANTIZED_SCHEMA: QUANTIZED_SCOPE}
REGION_STATUSES = frozenset({"matched", "rotation", "quantized_resampling"})
LIMITATION = "An ordinary edit can change the interpolation pattern. The observation carries zero AI confidence weight."
STATUS = {
    "unsupported_png": "The PNG did not match the supported input profile.",
    "no_repeatable_stencil": "The image did not have a repeatable interpolation pattern for this check.",
    "no_match": "No qualifying interpolation shift was found in the checked areas.",
    "matched": "A compact region showed a different interpolation pattern from the surrounding image.",
    "rotation": "A compact region showed an interpolation pattern consistent with a small local rotation.",
    "quantized_resampling": "A compact region used larger pixel-value steps and retained a repeating camera-style pattern.",
    "provenance": "The analysis stopped at verified provenance before this check ran.",
    "unavailable": "The check could not finish during the analysis.",
}
ROTATION_NO_MATCH = "No qualifying interpolation shift or local rotation was found in the checked areas."
QUANTIZED_NO_MATCH = "No qualifying interpolation shift, local rotation or lower-depth resampling pattern was found in the checked areas."
ROTATION_OBSERVATION = ("The red and blue channels showed a matching angled pattern in two parts of the crop. "
                        "This supports a local resampling explanation.")
QUANTIZED_OBSERVATION = ("The values in each color channel mostly followed one code step. Red and blue showed opposite "
                         "two-pixel patterns across the crop. Together these support a local resampling explanation.")
ROLES = ("context", "crop")
SHA = re.compile(r"[a-f0-9]{64}")


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(KEY)


def assets(document):
    return (document.get(KEY) or {}).get("artifacts", {})


def observation(schema, status):
    if schema == QUANTIZED_SCHEMA and status == "no_match":
        return QUANTIZED_NO_MATCH
    if schema == ROTATION_SCHEMA and status == "no_match":
        return ROTATION_NO_MATCH
    return STATUS[status]


def basis(status):
    return {"rotation": ROTATION_OBSERVATION, "quantized_resampling": QUANTIZED_OBSERVATION}.get(status)


def validate(document, commitment):
    review = document.get(KEY)
    if review is None and commitment is None and document["source"].get(COMMITMENT) is None:
        return
    if document.get("schema") != "chipmunk.image-evidence/1" or not isinstance(review, dict):
        raise ValueError("Interpolation review requires image evidence")
    source = document["source"]
    record = review["record"]
    pictures = review["artifacts"]
    status = record["status"]
    schema = record["schema"]
    scope = SCOPES.get(schema)
    if (not isinstance(commitment, str) or not SHA.fullmatch(commitment)
            or canonical_hash(record) != commitment or source.get(COMMITMENT) != commitment
            or scope is None or record["source_sha256"] != source["sha256"]
            or status not in STATUS or record["observation"] != observation(schema, status)
            or (status == "rotation" and schema not in {ROTATION_SCHEMA, QUANTIZED_SCHEMA})
            or (status == "quantized_resampling" and schema != QUANTIZED_SCHEMA)
            or record["scope"] != scope or record["limitation"] != LIMITATION
            or record["affects_overall_confidence"] is not False):
        raise ValueError("Interpolation evidence differs from its source commitment")
    if (schema in {ROTATION_SCHEMA, QUANTIZED_SCHEMA} and record.get("basis") != basis(status)
            or schema not in {ROTATION_SCHEMA, QUANTIZED_SCHEMA} and "basis" in record):
        raise ValueError("Interpolation evidence does not describe the recorded basis")
    quantized = record.get("quantized_pattern")
    if status == "quantized_resampling":
        if (not isinstance(quantized, dict) or quantized.keys() != {"code_step", "minimum_channel_fraction", "phase_contrast"}
                or type(quantized["code_step"]) is not int or quantized["code_step"] not in {256, 257}
                or type(quantized["minimum_channel_fraction"]) not in {int, float} or not .95 <= quantized["minimum_channel_fraction"] <= 1
                or type(quantized["phase_contrast"]) not in {int, float} or not 0 < quantized["phase_contrast"] <= 3):
            raise ValueError("Invalid lower-depth interpolation reading")
    elif quantized is not None or (schema != QUANTIZED_SCHEMA and "quantized_pattern" in record):
        raise ValueError("Unexpected lower-depth interpolation reading")
    checked = status == "no_match" or status in REGION_STATUSES
    count, phase, support = (record[key] for key in ("supported_cells", "phase", "reference_closure"))
    if checked:
        size = source["width"], source["height"]
        sizes = SUPPORTED_SIZES if schema == EXTENDED_SCHEMA else {LEGACY_SIZE}
        if (size not in sizes or source.get("exif_orientation") != 1
                or type(count) is not int or not 0 < count <= (size[0] // 32) * (size[1] // 32) or type(phase) is not int or not 0 <= phase < 4
                or type(support) not in (int, float) or not math.isfinite(support) or not 0 <= support <= 1):
            raise ValueError("Interpolation evidence has invalid checked coverage")
    elif (count, phase, support) != (0, None, None):
        raise ValueError("An unassessed interpolation check has measured coverage")
    box = record["region"]
    digest = record["decoded_rgb16_sha256"]
    if status not in REGION_STATUSES:
        if box is not None or digest is not None or pictures:
            raise ValueError("Interpolation source images require a measured region")
        return
    x0, y0, x1, y1 = box
    alignment = 32 if status == "matched" else 16
    if (any(type(v) is not int or v % alignment for v in box)
            or not 0 <= x0 < x1 <= size[0] or not 0 <= y0 < y1 <= size[1]
            or not 64 <= x1 - x0 <= 192 or not 64 <= y1 - y0 <= 192
            or (x1 - x0) % 32 or (y1 - y0) % 32
            or not SHA.fullmatch(str(digest)) or set(pictures) != set(ROLES)):
        raise ValueError("Interpolation region or source image is invalid")
    for role, (width, height) in zip(ROLES, ((size[0] // 2, size[1] // 2), (x1 - x0, y1 - y0)), strict=True):
        asset = pictures[role]
        if (asset["media_type"] != "image/png" or not SHA.fullmatch(asset["sha256"])
                or (asset["width"], asset["height"]) != (width, height)):
            raise ValueError("Interpolation source images do not match their geometry")
