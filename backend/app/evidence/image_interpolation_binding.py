"""Model-free bindings for a native image interpolation observation."""
from __future__ import annotations

import math
import re

from app.core.canonical import canonical_hash

KEY = "native_interpolation_review"
COMMITMENT = "native_interpolation_review_sha256"
SCHEMA = "image-native-interpolation/1"
SCOPE = "This check requires a 2048 × 1536, 16-bit RGB PNG without ancillary chunks and a repeatable interpolation pattern."
LIMITATION = "An ordinary edit can change the interpolation pattern. The observation carries zero AI confidence weight."
STATUS = {
    "unsupported_png": "The PNG did not match the supported input profile.",
    "no_repeatable_stencil": "The image did not have a repeatable interpolation pattern for this check.",
    "no_match": "No qualifying interpolation shift was found in the checked areas.",
    "matched": "A compact region showed a different interpolation pattern from the surrounding image.",
    "provenance": "The analysis stopped at verified provenance before this check ran.",
    "unavailable": "The check could not finish during the analysis.",
}
ROLES = ("context", "crop")
SHA = re.compile(r"[a-f0-9]{64}")


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(KEY)


def assets(document):
    return (document.get(KEY) or {}).get("artifacts", {})


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
    if (not isinstance(commitment, str) or not SHA.fullmatch(commitment)
            or canonical_hash(record) != commitment or source.get(COMMITMENT) != commitment
            or record["schema"] != SCHEMA or record["source_sha256"] != source["sha256"]
            or status not in STATUS or record["observation"] != STATUS[status]
            or record["scope"] != SCOPE or record["limitation"] != LIMITATION
            or record["affects_overall_confidence"] is not False):
        raise ValueError("Interpolation evidence differs from its source commitment")
    checked = status in {"no_match", "matched"}
    count, phase, support = (record[key] for key in ("supported_cells", "phase", "reference_closure"))
    if checked:
        if ((source["width"], source["height"], source.get("exif_orientation")) != (2048, 1536, 1)
                or type(count) is not int or not 0 < count <= 3072 or type(phase) is not int or not 0 <= phase < 4
                or type(support) not in (int, float) or not math.isfinite(support) or not 0 <= support <= 1):
            raise ValueError("Interpolation evidence has invalid checked coverage")
    elif (count, phase, support) != (0, None, None):
        raise ValueError("An unassessed interpolation check has measured coverage")
    box = record["region"]
    digest = record["decoded_rgb16_sha256"]
    if status != "matched":
        if box is not None or digest is not None or pictures:
            raise ValueError("Interpolation source images require a measured region")
        return
    x0, y0, x1, y1 = box
    if (any(type(v) is not int or v % 32 for v in box)
            or not 0 <= x0 < x1 <= 2048 or not 0 <= y0 < y1 <= 1536
            or not 64 <= x1 - x0 <= 192 or not 64 <= y1 - y0 <= 192
            or not SHA.fullmatch(str(digest)) or set(pictures) != set(ROLES)):
        raise ValueError("Interpolation region or source image is invalid")
    for role, (width, height) in zip(ROLES, ((1024, 768), (x1 - x0, y1 - y0)), strict=True):
        asset = pictures[role]
        if (asset["media_type"] != "image/png" or not SHA.fullmatch(asset["sha256"])
                or (asset["width"], asset["height"]) != (width, height)):
            raise ValueError("Interpolation source images do not match their geometry")
