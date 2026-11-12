"""Exact RGB reproduction of an earlier crop on a submitted PNG grid."""

from __future__ import annotations

import hashlib
from itertools import product

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import reference_compare, reference_crop_resize, reference_resize

KERNELS = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR,
           "lanczos": Image.Resampling.LANCZOS, "opencv_area": None}
GRID = 5
TILE_SIDE = 20
TILE_MIN_TEXTURE = .12
TILE_GRADIENT_MIN = 6
MIN_TILES = 12
MIN_AXIS = 4
MIN_GLOBAL_EQUAL = .65
MAX_MISMATCH_BOX = .40
TOP_PROPOSALS = 8
MIN_PROPOSAL_CORRELATION = .75
MAX_REFERENCE_SIDE = 768
MAX_SUBMITTED_SIDE = 845
MIN_SCALE, MAX_SCALE = .72, 1.17


def _project(reference, rectangle, size, kernel):
    x0, y0, x1, y1 = rectangle
    crop = reference[y0:y1, x0:x1]
    if kernel is None:
        return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(crop).resize(size, kernel))


def _textured_tiles(submitted):
    height, width = submitted.shape[:2]
    half = TILE_SIDE // 2
    tiles = []
    for row, column in product(range(GRID), repeat=2):
        x = round((column + .5) / GRID * width)
        y = round((row + .5) / GRID * height)
        bounds = np.s_[y-half:y+half, x-half:x+half]
        patch = submitted[bounds].astype(np.int16)
        dx = np.any(np.abs(np.diff(patch, axis=1)) >= TILE_GRADIENT_MIN, axis=2)
        dy = np.any(np.abs(np.diff(patch, axis=0)) >= TILE_GRADIENT_MIN, axis=2)
        if (dx.mean() + dy.mean()) / 2 >= TILE_MIN_TEXTURE:
            tiles.append((row, column, bounds))
    return tiles


def _spread(cells):
    return (len(cells) >= MIN_TILES and len({r for r, _ in cells}) >= MIN_AXIS
            and len({c for _, c in cells}) >= MIN_AXIS
            and all(any((r < GRID // 2 if top else r > GRID // 2)
                        and (c < GRID // 2 if left else c > GRID // 2) for r, c in cells)
                    for top, left in product((False, True), repeat=2)))


def _supported(projected, submitted, tiles):
    equal = np.all(projected == submitted, axis=2)
    if equal.mean() < MIN_GLOBAL_EQUAL:
        return None
    supports = [(r, c) for r, c, bounds in tiles if equal[bounds].all()]
    if not _spread(supports):
        return None
    ys, xs = np.where(~equal)
    if len(xs) and (xs.max() + 1 - xs.min()) * (ys.max() + 1 - ys.min()) > MAX_MISMATCH_BOX * equal.size:
        return None
    return equal, supports


def _scope(rectangle, submitted_shape):
    x0, y0, x1, y1 = rectangle
    widths = (x1 - x0, y1 - y0)
    sizes = submitted_shape[1], submitted_shape[0]
    return widths != sizes and all(round(MIN_SCALE * base) <= measured <= round(MAX_SCALE * base)
                                   for base, measured in zip(widths, sizes, strict=True))


def _candidate(reference: np.ndarray, submitted: np.ndarray):
    """Return a unique exact projection and a native difference mask if corroborated."""
    if (min(*reference.shape[:2], *submitted.shape[:2]) < 160 or max(reference.shape[:2]) > MAX_REFERENCE_SIDE
            or max(submitted.shape[:2]) > MAX_SUBMITTED_SIDE):
        return None, None
    textured = _textured_tiles(submitted)
    if not _spread([(row, column) for row, column, _ in textured]):
        return None, None
    gray_reference, gray_submitted = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in (reference, submitted))
    rough = reference_crop_resize._rough_bounds(gray_reference, gray_submitted)
    if rough is None:
        return None, None
    proposals = sorted(reference_crop_resize._scores(gray_reference, gray_submitted, rough[0]), reverse=True)
    good = None
    for correlation, rectangle, *_features in proposals[:TOP_PROPOSALS]:
        if correlation < MIN_PROPOSAL_CORRELATION or not _scope(rectangle, submitted.shape[:2]):
            continue
        for label, kernel in KERNELS.items():
            projected = _project(reference, rectangle, submitted.shape[1::-1], kernel)
            support = _supported(projected, submitted, textured)
            if support is None:
                continue
            equal, cells = support
            if good is not None:
                if rectangle != good["rectangle"] or not np.array_equal(projected, good["projected"]):
                    return None, None
                good["kernels"].append(label)
            else:
                good = {"rectangle": rectangle, "projected": projected, "equal": equal,
                        "support": cells, "kernels": [label]}
    if good is None:
        return None, None
    x0, y0, x1, y1 = good["rectangle"]
    difference = ~good["equal"]
    regions, mask, omitted = reference_compare._regions(difference, max(48, round(difference.size * .0004)))
    for region in regions:
        crop = reference_resize.native_crop(region, (x1 - x0, y1 - y0), submitted.shape[1::-1])
        region["reference_region"] = {**crop, "x": crop["x"] + x0, "y": crop["y"] + y0}
    return {"crop": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
            "kernels": good["kernels"], "exact_tiles": good["support"], "exact_pixels": int(good["equal"].sum()),
            "projected_sha256": hashlib.sha256(good["projected"].tobytes()).hexdigest(),
            "regions": regions, "omitted": omitted}, mask


def compare(reference: np.ndarray, submitted: np.ndarray, carrier: Image.Image) -> tuple[dict | None, np.ndarray | None]:
    """Compare a static RGB PNG within the bounded exact-reproduction scope."""
    if carrier.format != "PNG" or carrier.mode != "RGB" or carrier.n_frames != 1:
        return None, None
    proof, mask = _candidate(reference, submitted)
    if proof is None:
        return None, None
    crop = proof["crop"]
    return {
        "status": "comparable",
        "regions": proof["regions"],
        "additional_regions": proof["omitted"],
        "reference_crop_bounds": crop,
        "submitted_over_reference_xy": [[submitted.shape[1], crop["w"]], [submitted.shape[0], crop["h"]]],
        "supported_resize_kernels": proof["kernels"],
        "exact_projection": {
            "matched_pixels": proof["exact_pixels"],
            "matched_tiles": [[row, column] for row, column in proof["exact_tiles"]],
            "rgb_sha256": proof["projected_sha256"],
        },
    }, mask
