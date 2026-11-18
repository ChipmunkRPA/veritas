# SPDX-License-Identifier: Apache-2.0
"""Find small RGB copies after an exact right-angle rotation."""

from __future__ import annotations

import numpy as np

from app.detectors.image import exact_small_copy as exact


def _unique_seeds(pixels):
    seeds = exact.signatures(pixels)
    keys, positions, counts = np.unique(seeds, return_index=True, return_counts=True)
    unique = counts == 1
    return seeds.shape, keys[unique], positions[unique]


def _offsets(source, rotated):
    shape, keys, positions = source
    other_shape, other_keys, other_positions = rotated
    _, source_indices, other_indices = np.intersect1d(keys, other_keys, assume_unique=True, return_indices=True)
    if len(source_indices) > exact.MAX_CANDIDATES:
        return None
    y, x = np.unravel_index(positions[source_indices], shape)
    b, a = np.unravel_index(other_positions[other_indices], other_shape)
    offsets, counts = np.unique(np.column_stack((a - x, b - y)), axis=0, return_counts=True)
    ranked = sorted((-int(n), int(dx), int(dy)) for (dx, dy), n in zip(offsets, counts, strict=True) if n >= 16)
    return [(dx, dy) for _, dx, dy in ranked[:exact.MAX_OFFSETS]]


def _source_box(box, height, width, turns):
    x, y, w, h = box
    if turns == 1:
        return width - y - h, x, h, w
    return width - x - w, height - y - h, w, h


def analyse(rgb, *, allow_multiple=False, seeds=None):
    height, width = rgb.shape[:2]
    if not exact.MIN_SIDE <= min(height, width) <= max(height, width) <= exact.MAX_SIDE:
        return {"status": "unsupported_geometry", "pairs": []}
    seeds = _unique_seeds(rgb) if seeds is None else seeds
    matches = {}
    # Every clockwise 90 degree match is also counterclockwise when
    # swapping the two areas. A single quarter-turn search covers both.
    for turns in (1, 2):
        rotated = np.rot90(rgb, turns)
        offsets = _offsets(seeds, _unique_seeds(rotated))
        if offsets is None:
            return {"status": "unassessed_match_budget", "pairs": []}
        for dx, dy in offsets:
            for first, aligned in exact.matching_regions(rgb, rotated, dx, dy):
                second = _source_box(aligned, height, width, turns)
                pair = first, second
                if turns == 2 and second < first:
                    # Use the same crop order for both half-turn offsets.
                    pair = second, first
                key = tuple(sorted(pair))
                if allow_multiple:
                    key = turns, key
                if key in matches:
                    continue
                if (measurement := exact.witness(rgb, *pair, turns=turns)) is not None:
                    matches[key] = {"pair": pair, "measurement": measurement, "turns": turns}
                    if len(matches) > (2 if allow_multiple else 1):
                        return {"status": "ambiguous_pairs", "pairs": []}
    return {"status": "supported_pair" if matches else "no_pair", "pairs": list(matches.values())}
