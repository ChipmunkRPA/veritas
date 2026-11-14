# SPDX-License-Identifier: Apache-2.0
"""Locate lost fine detail across two matching JPEG grids."""

from __future__ import annotations

import cv2
import numpy as np

from app.detectors.image import (
    jpeg_dq,
    reference_compare,
    reference_exact_jpeg,
    reference_same_jpeg,
    reference_structure_jpeg,
)

SCHEMA = "chipmunk.image-reference-comparison/15"
METHOD = "exact_jpeg_pair_local_detail_loss"
INTERPRETATION = (
    "Matching blocks support the same framing. Fine luminance detail is weaker in the highlighted areas while broader "
    "structure remains similar. Local blur can cause this pattern. This comparison carries no weight in the AI estimate."
)
TILE = 16
MIN_COMPONENT = 4
_FREQUENCIES = np.indices((8, 8)).sum(axis=0)
_BANDS = ((_FREQUENCIES >= 1) & (_FREQUENCIES <= 2), (_FREQUENCIES >= 3) & (_FREQUENCIES <= 5))


def _vectors(coefficients, rows, columns):
    values = (coefficients.y[:2 * rows, :2 * columns] * coefficients.table)
    blocks = values.reshape(rows, 2, columns, 2, 8, 8).transpose(0, 2, 1, 3, 4, 5)
    return [blocks[..., band].reshape(rows, columns, -1) for band in _BANDS]


def compare(earlier, submitted, earlier_bytes, submitted_bytes):
    """Return a supplementary loss-of-detail region only when the pair qualifies."""
    if (earlier_bytes == submitted_bytes or earlier.size != submitted.size
            or not reference_same_jpeg.MIN_SIDE <= min(earlier.size) <= max(earlier.size) <= reference_same_jpeg.MAX_SIDE
            or not all(reference_structure_jpeg._jpeg_profile(image, body, (reference_same_jpeg.QUALITY,))
                       for image, body in ((earlier, earlier_bytes), (submitted, submitted_bytes)))):
        return None, None
    a, b = np.asarray(earlier), np.asarray(submitted)
    proof = reference_exact_jpeg.alignment_proof(a, b)
    if (proof is None or reference_same_jpeg.MIN_EXACT_DENOMINATOR * proof["exact_tiles"]
            < reference_same_jpeg.MIN_EXACT_NUMERATOR * proof["full_tiles"]):
        return None, None
    try:
        left, right = (jpeg_dq.read_coefficients(body) for body in (earlier_bytes, submitted_bytes))
    except ImportError:
        return None, None
    if (left is None or right is None or left.y.shape != right.y.shape or not np.array_equal(left.table, right.table)):
        return None, None
    height, width = a.shape[:2]
    rows, columns = height // TILE, width // TILE
    old, new = (_vectors(item, rows, columns) for item in (left, right))
    strength = [(np.square(x).sum(axis=2), np.square(y).sum(axis=2)) for x, y in zip(old, new, strict=True)]
    broad, fine = [np.sqrt(later / np.maximum(before, 1)) for before, later in strength]
    agreement = (old[0] * new[0]).sum(axis=2) / np.sqrt(np.maximum(strength[0][0] * strength[0][1], 1))
    selected = ((strength[1][0] >= (1.25 * TILE) ** 2) & (broad >= .65) & (broad <= 1.10)
                & (fine <= .75) & (fine <= broad - .12) & (agreement >= .90))
    closed = cv2.morphologyEx(selected.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8),
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # Only reconnect a bounded gap in the measured tiles. Keep both the exhibited
    # region and the directly supporting count so an export records the difference.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    components = sorted((i for i, row in enumerate(stats) if i and row[cv2.CC_STAT_AREA] >= MIN_COMPONENT),
                        key=lambda i: (-stats[i, cv2.CC_STAT_AREA], i))
    if not components:
        return None, None
    kept = components[:reference_compare.MAX_REGIONS]
    supported = np.isin(labels, kept)
    direct_counts = np.bincount(labels[selected], minlength=len(stats))
    mask = np.zeros((height, width), bool)
    mask[:rows * TILE, :columns * TILE] = supported.repeat(TILE, axis=0).repeat(TILE, axis=1)
    regions = []
    for i in kept:
        x, y, w, h, area = (int(value) for value in stats[i])
        bounds = {"x": x * TILE, "y": y * TILE, "w": w * TILE, "h": h * TILE}
        regions.append({**bounds, "reference_region": bounds, "selected_pixels": area * TILE ** 2,
                        "directly_measured_tiles": int(direct_counts[i])})
    proof["reference_decoded_rgb_sha256"] = proof.pop("expected_decoded_rgb_sha256")
    return {"status": "comparable", "alignment": {**proof, "method": METHOD}, "regions": regions,
            "additional_regions": len(components) - len(kept)}, mask
