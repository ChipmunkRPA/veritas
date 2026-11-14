"""Exact same-framing comparison for an opaque lossless WebP and a lossless copy."""
from __future__ import annotations

import hashlib

import numpy as np

from app.detectors.image import reference_compare, reference_png_exact_resize
from app.detectors.image.lossless_webp import lossless_webp_size

SCHEMA = "chipmunk.image-reference-comparison/18"
UNASSESSED = "webp_framing_unassessed"
INTERPRETATION = (
    "The comparison measures exact decoded RGB differences between two lossless files of the same framing. "
    "Ordinary processing can change these samples. The source of the change requires separate evidence."
)
LIMITATION = (
    "Exact WebP comparison needs compatible opaque images with matching dimensions and enough exact shared detail."
)


def supported(source, body, *, png=False):
    if source.mode != "RGB" or getattr(source, "n_frames", 1) != 1 or source.getexif().get(274, 1) != 1:
        return False
    if source.format == "WEBP":
        return lossless_webp_size(body) == source.size
    return (png and source.format == "PNG" and body[24:26] == b"\x08\x02"
            and not any(key in source.info for key in ("icc_profile", "gamma", "srgb", "chromaticity", "transparency")))


def compare(reference, submitted):
    if reference.shape != submitted.shape:
        return {"status": UNASSESSED, "regions": []}, None
    changed = np.any(reference != submitted, axis=2)
    matched = []
    if changed.any():
        support = reference_png_exact_resize._supported(
            reference, submitted, reference_png_exact_resize._textured_tiles(submitted))
        if support is None:
            return {"status": UNASSESSED, "regions": []}, None
        _, matched = support
    regions, mask, omitted = reference_compare._regions(changed, max(48, round(changed.size * .0004)))
    return {
        "status": "comparable", "regions": regions, "additional_regions": omitted,
        "exact_webp": {
            "changed_pixels": int(changed.sum()),
            "matched_tiles": [list(tile) for tile in matched],
            "earlier_rgb_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
            "submitted_rgb_sha256": hashlib.sha256(submitted.tobytes()).hexdigest(),
        },
    }, mask
