# SPDX-License-Identifier: Apache-2.0
"""Propose bounded small exact RGB matches after a single reflection."""

from __future__ import annotations

import numpy as np

from app.detectors.image import exact_small_copy as exact
from app.detectors.image.exact_turned_copy import _offsets, _unique_seeds

AXES = {"horizontal_reflection": 1, "vertical_reflection": 0}


def analyse(rgb):
    height, width = rgb.shape[:2]
    seeds = _unique_seeds(rgb)
    matches = {}
    for transform, axis in AXES.items():
        reflected = np.flip(rgb, axis=axis)
        offsets = _offsets(seeds, _unique_seeds(reflected))
        if offsets is None:
            return {"status": "unassessed_match_budget", "pairs": []}
        for dx, dy in offsets:
            for first, aligned in exact.matching_regions(rgb, reflected, dx, dy):
                x, y, w, h = aligned
                second = (width - x - w, y, w, h) if axis == 1 else (x, height - y - h, w, h)
                pair = tuple(sorted((first, second)))
                key = (transform, pair)
                if key in matches:
                    continue
                if (measurement := exact.witness(rgb, *pair, reflection_axis=axis)) is not None:
                    matches[key] = {"pair": pair, "transform": transform, "measurement": measurement}
                    if len(matches) > 1:
                        return {"status": "ambiguous_pairs", "pairs": []}
    return {"status": "supported_pair" if matches else "no_pair", "pairs": list(matches.values())}
