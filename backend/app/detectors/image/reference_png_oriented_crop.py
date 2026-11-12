"""Verify an integer crop after a right-angle orientation of an earlier image."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from app.detectors.image import reference_compare, reference_png_orientation
from app.detectors.image import reference_png_exact_resize as exact

SCHEMA = "chipmunk.image-reference-comparison/7"
MAX_REFERENCE_SIDE = 1024
MIN_CROP_FRACTION, MAX_CROP_FRACTION = .55, .90
MAX_PROPOSALS = 32
ANCHOR_SAMPLES = ((0, 0), (3, 11), (8, 4), (12, 17), (17, 7))


def _packed(pixels: np.ndarray) -> np.ndarray:
    rgb = pixels.astype(np.uint32)
    return rgb[:, :, 0] << 16 | rgb[:, :, 1] << 8 | rgb[:, :, 2]


def _offsets(reference: np.ndarray, submitted: np.ndarray, tiles: list) -> np.ndarray:
    height, width = submitted.shape
    rows, cols = reference.shape[0] - height + 1, reference.shape[1] - width + 1
    if min(rows, cols) < 1:
        return np.empty((0, 2), dtype=int)
    votes = np.zeros((rows, cols), dtype=np.uint8)
    for _, _, (y_bounds, x_bounds) in tiles:
        agrees = np.ones((rows, cols), dtype=bool)
        for dy, dx in ANCHOR_SAMPLES:
            y, x = y_bounds.start + dy, x_bounds.start + dx
            agrees &= reference[y:y + rows, x:x + cols] == submitted[y, x]
        votes += agrees
    return np.argwhere(votes >= exact.MIN_TILES)


def compare(reference: np.ndarray, submitted: np.ndarray, carrier: Image.Image) -> tuple[dict | None, np.ndarray | None]:
    height, width = reference.shape[:2]
    submitted_height, submitted_width = submitted.shape[:2]
    retained = submitted_height * submitted_width / (height * width)
    if (carrier.format != "PNG" or carrier.mode != "RGB" or carrier.n_frames != 1
            or max(height, width) > MAX_REFERENCE_SIDE or min(submitted_height, submitted_width) < 160
            or not MIN_CROP_FRACTION <= retained <= MAX_CROP_FRACTION):
        return None, None
    tiles = exact._textured_tiles(submitted)
    if not exact._spread([(row, column) for row, column, _ in tiles]):
        return None, None

    match = None
    submitted_packed = _packed(submitted)
    # The identity competes with all seven orientations for the same crop.
    for transform, oriented in reference_png_orientation._projections(reference):
        offsets = _offsets(_packed(oriented), submitted_packed, tiles)
        if len(offsets) > MAX_PROPOSALS:
            return None, None
        for y, x in offsets:
            x, y = int(x), int(y)
            projected = oriented[y:y + submitted_height, x:x + submitted_width]
            support = exact._supported(projected, submitted, tiles)
            if support is None:
                continue
            if match is not None or transform is None:
                return None, None
            equal, witnesses = support
            match = transform, x, y, projected, equal, witnesses
    if match is None:
        return None, None

    transform, x, y, projected, equal, witnesses = match
    difference = ~equal
    regions, mask, omitted = reference_compare._regions(difference, max(48, round(difference.size * .0004)))
    projected_crop = {"x": x, "y": y, "w": submitted_width, "h": submitted_height}
    native_crop = reference_png_orientation._native_bounds(projected_crop, width, height, transform)
    for region in regions:
        shifted = {**region, "x": region["x"] + x, "y": region["y"] + y}
        region["reference_region"] = reference_png_orientation._native_bounds(shifted, width, height, transform)
    return {"status": "comparable", "regions": regions, "additional_regions": omitted, "exact_oriented_crop": {
        "transform": transform, "projected_crop": projected_crop, "native_crop": native_crop,
        "matched_pixels": int(equal.sum()), "matched_tiles": [list(cell) for cell in witnesses],
        "projected_rgb_sha256": hashlib.sha256(projected.tobytes()).hexdigest(),
        "difference_mask_sha256": hashlib.sha256(difference.tobytes()).hexdigest(),
    }}, mask
