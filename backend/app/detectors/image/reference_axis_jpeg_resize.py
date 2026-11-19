# SPDX-License-Identifier: MIT
"""Compare two copies through a reproduced width/height resize and JPEG delivery."""

from __future__ import annotations

import io
from hashlib import sha256

import numpy as np
from PIL import Image

from app.detectors.image import (
    jpeg_bitstream,
    reference_exact_jpeg,
    reference_lossless_webp,
    reference_native_jpeg,
    reference_png_exact_resize,
    reference_resize,
    reference_structure_jpeg,
)
from app.detectors.image.reference_axis_resize import MIN_AXIS_DRIFT

SCHEMA = "chipmunk.image-reference-comparison/30"
QUALITIES = (75, 90)
FULL_GRID = "exact_full_jpeg_pixel_reproduction"
LOCAL_DIFFERENCE = "reproduced_jpeg_background_with_local_sample_difference"
INTERPRETATION = ("The comparison reproduced a separate width and height resize and a matching JPEG delivery. "
                  "Highlights show local sample differences between the two supplied copies. Ordinary editing and "
                  "JPEG compression can affect these differences. This comparison carries zero weight in the AI assessment.")


def _encode(pixels, tables):
    with io.BytesIO() as output:
        Image.fromarray(pixels).save(output, "JPEG", qtables=tables, subsampling=2)
        return output.getvalue()


def _projections(reference, submitted, size, tables):
    bounds = (0, 0, reference.shape[1], reference.shape[0])
    for name, kernel in reference_png_exact_resize.KERNELS.items():
        projected = reference_png_exact_resize._project(reference, bounds, size, kernel)
        delivery = _encode(projected, tables)
        for writes in (1, 2):
            with Image.open(io.BytesIO(delivery), formats=["JPEG"]) as image:
                pixels = np.array(image.convert("RGB"))
            yield pixels, delivery, {
                "resize_kernel": name, "jpeg_writes": writes, "body_sha256": sha256(delivery).hexdigest(),
                "rgb_sha256": sha256(pixels.tobytes()).hexdigest(),
            }
            if writes == 1 and not np.array_equal(pixels, submitted):
                delivery = _encode(pixels, tables)
            else:
                break


def compare(earlier, questioned, reference_bytes, submitted_bytes):
    """Verify exact delivery equivalence or retain differences with detailed shared anchors."""
    rw, rh = earlier.size
    sw, sh = questioned.size
    sx, sy = sw / rw, sh / rh
    exact = reference_png_exact_resize
    if (min(rw, rh, sw, sh) < reference_native_jpeg.MIN_SIDE
            or max(rw, rh) > exact.MAX_REFERENCE_SIDE or max(sw, sh) > exact.MAX_SUBMITTED_SIDE
            or not exact.MIN_SCALE <= min(sx, sy) <= max(sx, sy) <= exact.MAX_SCALE
            or abs(sx / sy - 1) < MIN_AXIS_DRIFT
            or earlier.format != "PNG" or not reference_lossless_webp.supported(earlier, reference_bytes, png=True)
            or not reference_structure_jpeg._jpeg_profile(questioned, submitted_bytes, QUALITIES)):
        return None, None

    spec = jpeg_bitstream.read(submitted_bytes)
    proof = {"jpeg_quality": spec.ijg_quality, "submitted_over_reference_xy": [[sw, rw], [sh, rh]]}
    reference, submitted = np.array(earlier), np.array(questioned)
    submitted_tiles = reference_exact_jpeg.blocks(submitted)
    unchanged, plausible = [], []
    for pixels, delivery, recipe in _projections(reference, submitted, (sw, sh), questioned.quantization):
        if np.array_equal(pixels, submitted):
            unchanged.append(recipe)
            continue
        projected_tiles = reference_exact_jpeg.blocks(pixels)
        equal_tiles = (projected_tiles == submitted_tiles).all(axis=(-1, -2, -3))
        if equal_tiles.mean() < reference_native_jpeg.MIN_EXACT_SHARE:
            continue
        difference = np.any(pixels != submitted, axis=2)
        rows, columns = np.where(difference)
        bound = (columns.max() + 1 - columns.min()) * (rows.max() + 1 - rows.min())
        if bound <= exact.MAX_MISMATCH_BOX * difference.size:
            plausible.append((pixels, delivery, recipe))
    if unchanged:
        return {"status": "comparable", "regions": [], "affects_overall_confidence": False,
                "axis_jpeg_resize": {**proof, "basis": FULL_GRID, "projection": unchanged,
                                     "matched_pixels": sw * sh}}, np.zeros((sh, sw), dtype=bool)

    accepted = []
    for pixels, delivery, recipe in plausible:
        with Image.open(io.BytesIO(delivery), formats=["JPEG"]) as expected:
            status, result, mask = reference_native_jpeg.measure_pair(expected, questioned, delivery, submitted_bytes,
                                                                    qualities=QUALITIES)
        if status == "comparable" and result is not None:
            accepted.append((pixels, result, mask, recipe))
    if not accepted:
        return None, None
    first, result, mask, _ = accepted[0]
    if any(not np.array_equal(first, pixels) or not np.array_equal(mask, other_mask)
           for pixels, _other_result, other_mask, _recipe in accepted[1:]):
        return None, None
    for region in result["regions"]:
        region["reference_region"] = reference_resize.native_crop(region, (rw, rh), (sw, sh))
    return result | {"affects_overall_confidence": False,
                     "axis_jpeg_resize": {**proof, "basis": LOCAL_DIFFERENCE,
                                          "projection": [recipe for _, _, _, recipe in accepted]}}, mask
