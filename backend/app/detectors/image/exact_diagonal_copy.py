# SPDX-License-Identifier: Apache-2.0
"""Find a single small, diagonally reflected RGB copy."""

from __future__ import annotations

import numpy as np

from app.detectors.image import exact_small_copy
from app.detectors.image.exact_turned_copy import _offsets, _unique_seeds
from app.evidence.image_copy_similarity_binding import DIAGONAL_AXES, exact_diagonal_size


def measure(rgb, first, second, transform):
    axis = DIAGONAL_AXES.get(transform)
    if axis is None or not exact_diagonal_size(*rgb.shape[:2]):
        return None
    return exact_small_copy.witness(rgb, first, second, turns=1, reflection_axis=axis)


def analyse(rgb):
    height, width = rgb.shape[:2]
    if not exact_diagonal_size(width, height):
        return {"status": "unsupported_profile", "measurement": None}, [], None
    seeds = _unique_seeds(rgb)
    matches = {}
    for transform, axis in DIAGONAL_AXES.items():
        reflected = np.flip(np.rot90(rgb), axis=axis)
        offsets = _offsets(seeds, _unique_seeds(reflected))
        if offsets is None:
            return {"status": "unassessed_match_budget", "measurement": None}, [], None
        for dx, dy in offsets:
            for first, (x, y, w, h) in exact_small_copy.matching_regions(rgb, reflected, dx, dy):
                second = (y, x, h, w) if axis == 0 else (width - y - h, height - x - w, h, w)
                pair = tuple(sorted((first, second)))
                key = transform, pair
                if key in matches:
                    continue
                if (measurement := measure(rgb, *pair, transform)) is not None:
                    matches[key] = measurement
                    if len(matches) > 1:
                        return {"status": "ambiguous_pairs", "measurement": None}, [], None
    if not matches:
        return {"status": "no_pair", "measurement": None}, [], None
    ((transform, pair), measurement), = matches.items()
    return {"status": "supported_pair", "measurement": measurement}, [pair], transform
