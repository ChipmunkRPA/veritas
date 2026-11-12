"""Verify whole-image right-angle orientations by exact RGB correspondence."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from app.detectors.image import reference_compare
from app.detectors.image import reference_png_exact_resize as exact

SCHEMA = "chipmunk.image-reference-comparison/6"
MAX_PIXELS = 4_000_000
LABELS = {
    "rotate_90_clockwise": "90-degree clockwise rotation",
    "rotate_180": "180-degree rotation",
    "rotate_90_counterclockwise": "90-degree counterclockwise rotation",
    "mirror_horizontal": "left-to-right reflection",
    "mirror_vertical": "top-to-bottom reflection",
    "reflect_main_diagonal": "main-diagonal reflection",
    "reflect_other_diagonal": "opposite-diagonal reflection",
}


def _projections(reference):
    yield None, reference
    for angle, key in ((-1, "rotate_90_clockwise"), (2, "rotate_180"), (1, "rotate_90_counterclockwise")):
        yield key, np.rot90(reference, angle)
    yield "mirror_horizontal", reference[:, ::-1]
    yield "mirror_vertical", reference[::-1]
    yield "reflect_main_diagonal", reference.transpose(1, 0, 2)
    yield "reflect_other_diagonal", reference[::-1, ::-1].transpose(1, 0, 2)


def _native_bounds(region, width, height, transform):
    x, y, w, h = (region[key] for key in ("x", "y", "w", "h"))
    right, bottom = x + w, y + h
    left, top, right, bottom = {
        "rotate_90_clockwise": (y, height - right, bottom, height - x),
        "rotate_180": (width - right, height - bottom, width - x, height - y),
        "rotate_90_counterclockwise": (width - bottom, x, width - y, right),
        "mirror_horizontal": (width - right, y, width - x, bottom),
        "mirror_vertical": (x, height - bottom, right, height - y),
        "reflect_main_diagonal": (y, x, bottom, right),
        "reflect_other_diagonal": (width - bottom, height - right, width - y, height - x),
    }[transform]
    return {"x": left, "y": top, "w": right - left, "h": bottom - top}


def compare(reference: np.ndarray, submitted: np.ndarray, carrier: Image.Image) -> tuple[dict | None, np.ndarray | None]:
    if (carrier.format != "PNG" or carrier.mode != "RGB" or carrier.n_frames != 1
            or reference.shape[:2] not in (submitted.shape[:2], submitted.shape[1::-1])
            or min(submitted.shape[:2]) < 160 or submitted.shape[0] * submitted.shape[1] > MAX_PIXELS):
        return None, None
    textured = exact._textured_tiles(submitted)
    if not exact._spread([(row, column) for row, column, _ in textured]):
        return None, None

    match = None
    for transform, projected in _projections(reference):
        if projected.shape != submitted.shape:
            continue
        tiles = [(r, c, bounds) for r, c, bounds in textured if np.array_equal(projected[bounds], submitted[bounds])]
        if not exact._spread([(r, c) for r, c, _ in tiles]):
            continue
        support = exact._supported(projected, submitted, tiles)
        if support is None:
            continue
        if transform is None or match is not None:
            return None, None
        equal, cells = support
        match = transform, projected, equal, cells
    if match is None:
        return None, None

    transform, projected, equal, cells = match
    difference = ~equal
    regions, mask, omitted = reference_compare._regions(difference, max(48, round(difference.size * .0004)))
    for region in regions:
        region["reference_region"] = _native_bounds(region, reference.shape[1], reference.shape[0], transform)
    return {
        "status": "comparable", "regions": regions, "additional_regions": omitted,
        "exact_orientation": {
            "transform": transform, "matched_pixels": int(equal.sum()),
            "matched_tiles": [list(cell) for cell in cells], "rgb_sha256": hashlib.sha256(projected.tobytes()).hexdigest(),
        },
    }, mask
