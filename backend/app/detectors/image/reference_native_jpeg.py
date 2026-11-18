# SPDX-License-Identifier: Apache-2.0
"""Locate local sample differences between same-framing JPEG copies."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import reference_compare, reference_exact_jpeg, reference_structure_jpeg

SCHEMA = "chipmunk.image-reference-comparison/28"
METHOD = "native_jpeg_sample_difference"
QUALITIES = (75,)
MIN_SIDE, MAX_SIDE = 256, 768
MIN_EXACT_SHARE = .85
RESIDUAL_FLOOR = 4.
COLOR_WEIGHT = .75
MIN_AREA = 32
CLOSURE_WIDTH = 5
INTERPRETATION = ("Highlights show local brightness or color differences between two aligned JPEG copies. Retouching "
                  "and JPEG re-encoding can cause these differences. This evidence carries zero weight in the AI assessment.")


def _samples(data):
    """Decode full-size JPEG brightness and color samples before RGB conversion."""
    with Image.open(io.BytesIO(data), formats=["JPEG"]) as image:
        image.draft("YCbCr", image.size)
        if image.mode != "YCbCr":
            raise ValueError("JPEG component samples unavailable")
        return np.asarray(image, dtype=np.int16)


def measure_pair(earlier, submitted, original, questioned, *, qualities=QUALITIES):
    """Require matching standard JPEG tables and exact detailed anchors across the framing."""
    if (earlier.size != submitted.size or not MIN_SIDE <= min(earlier.size) <= max(earlier.size) <= MAX_SIDE
            or not reference_structure_jpeg._jpeg_profile(earlier, original, qualities)
            or not reference_structure_jpeg._jpeg_profile(submitted, questioned, qualities)
            or earlier.quantization != submitted.quantization):
        return "different_delivery_profile", None, None
    rgb = np.asarray(earlier), np.asarray(submitted)
    if np.array_equal(*rgb):
        return "identical_pixels", None, np.zeros(rgb[0].shape[:2], dtype=bool)
    proof = reference_exact_jpeg.alignment_proof(*rgb)
    if proof is None or proof["exact_tiles"] < MIN_EXACT_SHARE * proof["full_tiles"]:
        return "native_alignment_unresolved", None, None
    # A routine decode/re-encode can change local samples even with the same quantization tables.
    if np.array_equal(reference_exact_jpeg.encoded_reference(rgb[0], submitted), rgb[1]):
        return "codec_reencode_equivalent", None, np.zeros(rgb[0].shape[:2], dtype=bool)

    try:
        channels = np.abs(_samples(original) - _samples(questioned))
    except (OSError, ValueError, SyntaxError):
        return "component_read_unavailable", None, None
    difference = np.maximum(channels[:, :, 0], COLOR_WEIGHT * channels[:, :, 1:].max(axis=2))
    binary = (difference >= RESIDUAL_FLOOR).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSURE_WIDTH, CLOSURE_WIDTH))
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # Morphological closing joins small gaps. Unchanged samples may not enter the published mask.
    joined &= difference > 0
    regions, mask, omitted = reference_compare._regions(joined, MIN_AREA)
    for region in regions:
        region["reference_region"] = {key: region[key] for key in ("x", "y", "w", "h")}
    proof["reference_decoded_rgb_sha256"] = proof.pop("expected_decoded_rgb_sha256")
    return "comparable", {"status": "comparable", "alignment": {**proof, "method": METHOD},
                          "regions": regions, "additional_regions": omitted}, mask


def compare(earlier, submitted, original, questioned):
    """Promote only a measured local change supported by both matching files."""
    _, result, mask = measure_pair(earlier, submitted, original, questioned)
    if result is None or not result["regions"]:
        return None, None
    return result, mask
