# SPDX-License-Identifier: MIT
"""Compare a full earlier image after independently scaling its two axes."""

from __future__ import annotations

from hashlib import sha256

import numpy as np
from PIL import Image

from app.detectors.image import reference_compare, reference_lossless_webp, reference_resize
from app.detectors.image import reference_png_exact_resize as exact

MIN_SIDE = 160
MIN_AXIS_DRIFT = .10
FULL_GRID = "exact_full_grid"
DISTRIBUTED_TEXTURE = "exact_distributed_texture"


def _match(reference, submitted):
    height, width = reference.shape[:2]
    full = (0, 0, width, height)
    size = submitted.shape[1::-1]
    projections = [(label, exact._project(reference, full, size, kernel)) for label, kernel in exact.KERNELS.items()]
    exact_kernels = [label for label, projection in projections if np.array_equal(projection, submitted)]
    if exact_kernels:
        return submitted, np.zeros(submitted.shape[:2], dtype=bool), [], exact_kernels, FULL_GRID

    textured = exact._textured_tiles(submitted)
    if not exact._spread([(row, column) for row, column, _ in textured]):
        return None
    match = None
    kernels = []
    for label, projection in projections:
        support = exact._supported(projection, submitted, textured)
        if support is None:
            continue
        if match is None:
            equal, cells = support
            match = projection, ~equal, cells
        elif not np.array_equal(match[0], projection):
            return None
        kernels.append(label)
    if match is None:
        return None
    return *match, kernels, DISTRIBUTED_TEXTURE


def compare(earlier: Image.Image, submitted: Image.Image, earlier_bytes: bytes, submitted_bytes: bytes
            ) -> tuple[dict | None, np.ndarray | None]:
    """Return native regions and the supporting projection for a checked lossless pair."""
    rw, rh = earlier.size
    sw, sh = submitted.size
    if (min(rw, rh, sw, sh) < MIN_SIDE or max(rw, rh) > exact.MAX_REFERENCE_SIDE
            or max(sw, sh) > exact.MAX_SUBMITTED_SIDE):
        return None, None
    sx, sy = sw / rw, sh / rh
    if (not exact.MIN_SCALE <= min(sx, sy) <= max(sx, sy) <= exact.MAX_SCALE
            or abs(sx / sy - 1) < MIN_AXIS_DRIFT
            or not reference_lossless_webp.supported(earlier, earlier_bytes, png=True)
            or not reference_lossless_webp.supported(submitted, submitted_bytes, png=True)):
        return None, None

    match = _match(np.asarray(earlier), np.asarray(submitted))
    if match is None:
        return None, None
    projection, difference, cells, kernels, basis = match
    regions, mask, omitted = reference_compare._regions(difference, max(48, round(difference.size * .0004)))
    for region in regions:
        region["reference_region"] = reference_resize.native_crop(region, earlier.size, submitted.size)
    return {
        "status": "comparable", "regions": regions, "additional_regions": omitted, "affects_overall_confidence": False,
        "axis_resize": {
            "basis": basis,
            "submitted_over_reference_xy": [[sw, rw], [sh, rh]],
            "matched_pixels": int(difference.size - difference.sum()),
            "matched_tiles": [list(cell) for cell in cells],
            "supported_resize_kernels": kernels,
            "projected_rgb_sha256": sha256(projection.tobytes()).hexdigest(),
            "difference_mask_sha256": sha256(difference.tobytes()).hexdigest(),
        },
    }, mask
