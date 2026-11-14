# SPDX-License-Identifier: Apache-2.0
"""Compare detailed blocks in two same-framing JPEGs."""

from __future__ import annotations

import numpy as np

from app.detectors.image import reference_compare, reference_exact_jpeg, reference_structure_jpeg

SCHEMA = "chipmunk.image-reference-comparison/14"
METHOD = "exact_decoded_jpeg_pair_identity"
MIN_SIDE, MAX_SIDE = 512, 768
QUALITY = 90
MIN_EXACT_NUMERATOR, MIN_EXACT_DENOMINATOR = 17, 20
INTERPRETATION = (
    "Identical decoded blocks support the same framing in both JPEGs. Highlights show local differences between "
    "the delivered images. Ordinary editing can produce these differences. The earlier copy's origin and the cause "
    "of the changes require separate evidence."
)


def compare(earlier, submitted, earlier_bytes, submitted_bytes):
    """Locate differences only after independently matching the original JPEG grids."""
    if (earlier.size != submitted.size or not MIN_SIDE <= min(earlier.size) <= max(earlier.size) <= MAX_SIDE
            or not all(reference_structure_jpeg._jpeg_profile(image, payload, (QUALITY,), allow_progressive=True) for image, payload in
                       ((earlier, earlier_bytes), (submitted, submitted_bytes)))
            or earlier.quantization != submitted.quantization):
        return None, None

    a, b = np.asarray(earlier), np.asarray(submitted)
    proof = reference_exact_jpeg.alignment_proof(a, b)
    if (proof is None or MIN_EXACT_DENOMINATOR * proof["exact_tiles"] < MIN_EXACT_NUMERATOR * proof["full_tiles"]):
        return None, None

    # The two files already have matching quantization and most blocks have identical decoded samples.
    # Another encode can create differences outside the native changed area.
    result, mask = reference_compare.compare(a, b)
    if result["status"] != "comparable":
        return None, None
    for region in result["regions"]:
        region["reference_region"] = {key: region[key] for key in ("x", "y", "w", "h")}
    proof["reference_decoded_rgb_sha256"] = proof.pop("expected_decoded_rgb_sha256")
    return {**result, "alignment": {**proof, "method": METHOD}}, mask
