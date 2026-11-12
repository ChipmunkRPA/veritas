"""Compare a corroborated same-framing reference at different native dimensions."""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.detectors.image import reference_alignment, reference_compare

KERNELS = (Image.Resampling.BICUBIC, Image.Resampling.BILINEAR, Image.Resampling.LANCZOS)
MAX_ASPECT_DRIFT = .008
MIN_SCALE, MAX_SCALE = .50, 1.50
MIN_VOTES = 2


def native_crop(region: dict, original_size: tuple[int, int], submitted_size: tuple[int, int]) -> dict:
    """Map half-open submitted edges outward to integral original crop coordinates."""
    x, y, width, height = (region[key] for key in ("x", "y", "w", "h"))
    rw, rh = original_size
    sw, sh = submitted_size
    left, top = x * rw // sw, y * rh // sh
    right = ((x + width) * rw + sw - 1) // sw
    bottom = ((y + height) * rh + sh - 1) // sh
    return {"x": left, "y": top, "w": right - left, "h": bottom - top}


def compare(reference: np.ndarray, submitted: np.ndarray, carrier: Image.Image) -> tuple[dict, np.ndarray | None]:
    """Project the reference for scoring while preserving native evidence coordinates."""
    rh, rw = reference.shape[:2]
    sh, sw = submitted.shape[:2]
    sx, sy = sw / rw, sh / rh
    if ((rw, rh) == (sw, sh) or min(rw, rh, sw, sh) < reference_alignment.MIN_SIDE
            or min(sx, sy) < MIN_SCALE or max(sx, sy) > MAX_SCALE or abs(sx / sy - 1) > MAX_ASPECT_DRIFT):
        return {"status": "unsupported_resize_geometry", "regions": []}, None
    original = Image.fromarray(reference)
    candidates = [np.asarray(original.resize((sw, sh), kernel)) for kernel in KERNELS]
    support = reference_alignment.measure(candidates[0], submitted)
    if support is None or support["reference_minus_submitted"] != {"x": 0, "y": 0}:
        return {"status": "same_framing_unresolved", "regions": []}, None
    metadata = {"framing_check": support, "submitted_over_reference_xy": [[sw, rw], [sh, rh]]}
    votes = np.zeros((sh, sw), np.uint8)
    supported = []
    for kernel, projected in zip(KERNELS, candidates, strict=True):
        record, mask = reference_compare.compare_across_deliveries(projected, submitted, carrier)
        if record["status"] == "comparable" and mask is not None:
            votes += mask
            supported.append(kernel.name.lower())
    if len(supported) < MIN_VOTES:
        return {"status": "resize_delivery_unresolved", "regions": [], **metadata}, None
    regions, mask, omitted = reference_compare._regions(votes >= MIN_VOTES, max(48, round(votes.size * .0015)))
    for region in regions:
        region["reference_region"] = native_crop(region, (rw, rh), (sw, sh))
    return {"status": "comparable", "regions": regions, "additional_regions": omitted,
            "supported_resize_kernels": supported, **metadata}, mask
