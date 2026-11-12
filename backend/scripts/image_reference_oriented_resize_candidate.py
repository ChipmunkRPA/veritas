# SPDX-License-Identifier: Apache-2.0
"""Try exact, unique crop-and-resize reproduction over all eight orientations."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from app.detectors.image import reference_compare, reference_crop_resize, reference_resize
from app.detectors.image import reference_png_exact_resize as exact
from app.detectors.image import reference_png_orientation as orientation


def supports(oriented, submitted, gray_submitted, tiles):
    gray_reference = cv2.cvtColor(oriented, cv2.COLOR_RGB2GRAY)
    rough = reference_crop_resize._rough_bounds(gray_reference, gray_submitted)
    if rough is None:
        return
    scores = sorted(reference_crop_resize._scores(gray_reference, gray_submitted, rough[0]), reverse=True)
    for correlation, rectangle, *_ in scores[:exact.TOP_PROPOSALS]:
        if correlation < exact.MIN_PROPOSAL_CORRELATION or not exact._scope(rectangle, submitted.shape[:2]):
            continue
        for label, kernel in exact.KERNELS.items():
            projection = exact._project(oriented, rectangle, submitted.shape[1::-1], kernel)
            support = exact._supported(projection, submitted, tiles)
            if support is not None:
                yield rectangle, label, projection, support


def compare(reference, submitted, carrier):
    if (carrier.format != "PNG" or carrier.mode != "RGB" or carrier.n_frames != 1
            or min(*reference.shape[:2], *submitted.shape[:2]) < 160
            or max(reference.shape[:2]) > exact.MAX_REFERENCE_SIDE or max(submitted.shape[:2]) > exact.MAX_SUBMITTED_SIDE):
        return None, None
    tiles = exact._textured_tiles(submitted)
    if not exact._spread([(row, col) for row, col, _ in tiles]):
        return None, None
    height, width = reference.shape[:2]
    found = None
    gray_submitted = cv2.cvtColor(submitted, cv2.COLOR_RGB2GRAY)
    for transform, oriented in orientation._projections(reference):
        for rectangle, kernel, projection, support in supports(oriented, submitted, gray_submitted, tiles):
            if transform is None:
                return None, None
            if found is None:
                found = transform, rectangle, projection, support, [kernel]
            elif found[:2] != (transform, rectangle) or not np.array_equal(found[2], projection):
                return None, None
            else:
                found[4].append(kernel)
    if found is None:
        return None, None

    transform, (x, y, right, bottom), projection, (equal, tiles), kernels = found
    difference = ~equal
    crop_size = right-x, bottom-y
    projected_crop = {"x": x, "y": y, "w": crop_size[0], "h": crop_size[1]}
    native_crop = orientation._native_bounds(projected_crop, width, height, transform)
    regions, mask, omitted = reference_compare._regions(difference, max(48, round(difference.size * .0004)))
    for region in regions:
        local = reference_resize.native_crop(region, crop_size, submitted.shape[1::-1])
        local.update(x=x + local["x"], y=y + local["y"])
        region["reference_region"] = orientation._native_bounds(local, width, height, transform)
    return {"transform": transform, "projected_crop": projected_crop, "native_crop": native_crop,
            "kernels": kernels, "matched_pixels": int(equal.sum()), "matched_tiles": [list(cell) for cell in tiles],
            "projected_rgb_sha256": hashlib.sha256(projection.tobytes()).hexdigest(),
            "difference_mask_sha256": hashlib.sha256(difference.tobytes()).hexdigest(),
            "regions": regions, "omitted_regions": omitted}, mask
