# SPDX-License-Identifier: Apache-2.0
"""Find a small, exact translated RGB region with different surrounding pixels."""

from __future__ import annotations

import cv2
import numpy as np

MIN_SIDE, MAX_SIDE = 384, 768
MIN_CROP, MAX_CROP = 16, 56
MIN_DISPLACEMENT = 30
MAX_OFFSETS = 8
MAX_CANDIDATES = 60_000
_PERIMETER = 3


def _blocks(rgb, first, second):
    height, width = rgb.shape[:2]
    x, y, w, h = first
    a, b, _, _ = second
    if not (MIN_CROP <= min(w, h) <= max(w, h) <= MAX_CROP and (w, h) == second[2:]
            and max(abs(a - x), abs(b - y)) >= MIN_DISPLACEMENT
            and (abs(a - x) >= w or abs(b - y) >= h)
            and all(i >= _PERIMETER and i + w + _PERIMETER <= width for i in (x, a))
            and all(i >= _PERIMETER and i + h + _PERIMETER <= height for i in (y, b))):
        return None
    return (rgb[y - _PERIMETER:y + h + _PERIMETER, x - _PERIMETER:x + w + _PERIMETER],
            rgb[b - _PERIMETER:b + h + _PERIMETER, a - _PERIMETER:a + w + _PERIMETER])


def witness(rgb, first, second):
    blocks = _blocks(rgb, first, second)
    if blocks is None:
        return None
    before, after = blocks
    core = before[_PERIMETER:-_PERIMETER, _PERIMETER:-_PERIMETER]
    if not np.array_equal(core, after[_PERIMETER:-_PERIMETER, _PERIMETER:-_PERIMETER]):
        return None
    # Require actual local detail in both axes and multiple colors. A matching
    # flat swatch, straight edge, or one row of repeated pixels is insufficient.
    x_steps = np.mean(np.any(core[:, 1:] != core[:, :-1], axis=2))
    y_steps = np.mean(np.any(core[1:] != core[:-1], axis=2))
    color_count = len(np.unique(np.ascontiguousarray(core).view(np.dtype((np.void, 3)))))
    if color_count < 64 or min(x_steps, y_steps) < .45 or core.std(axis=(0, 1)).max() < 6:
        return None
    delta = np.abs(before.astype(np.int16) - after.astype(np.int16))
    border = (delta[:_PERIMETER, _PERIMETER:-_PERIMETER], delta[-_PERIMETER:, _PERIMETER:-_PERIMETER],
              delta[_PERIMETER:-_PERIMETER, :_PERIMETER], delta[_PERIMETER:-_PERIMETER, -_PERIMETER:])
    border_mean = [round(float(plane.mean()), 5) for plane in border]
    if sum(value >= 3.0 for value in border_mean) < 3:
        return None
    return {"matching_pixels": core.shape[0] * core.shape[1], "different_color_values": color_count,
            "perimeter_mean_absolute_difference": border_mean}


def _offsets(rgb):
    packed = rgb[..., 0].astype(np.uint64) << 16 | rgb[..., 1].astype(np.uint64) << 8 | rgb[..., 2].astype(np.uint64)
    height, width = packed.shape
    # Use several widely separated samples only to propose matching coordinates.
    # Full native equality is subsequently checked for every selected crop.
    view = packed[:height - 8, :width - 8]
    signatures = view.copy()
    for y, x in ((1, 7), (3, 3), (7, 1), (8, 8)):
        signatures *= np.uint64(0x9E3779B185EBCA87)
        signatures ^= packed[y:height - 8 + y, x:width - 8 + x]
    order = np.argsort(signatures, axis=None, kind="stable")
    sorted_values = signatures.ravel()[order]
    equal = sorted_values[:-1] == sorted_values[1:]
    # Ignore ambiguous matches with more than two occurrences of the same seed.
    unique_pairs = equal & ~np.r_[False, equal[:-1]] & ~np.r_[equal[1:], False]
    chosen = np.flatnonzero(unique_pairs)
    if len(chosen) > MAX_CANDIDATES:
        return None
    if not len(chosen):
        return []
    y, x = np.unravel_index(order[chosen], signatures.shape)
    b, a = np.unravel_index(order[chosen + 1], signatures.shape)
    dx, dy = a - x, b - y
    separated = np.maximum(np.abs(dx), np.abs(dy)) >= MIN_DISPLACEMENT
    dx, dy = dx[separated], dy[separated]
    offsets, counts = np.unique(np.column_stack((dx, dy)), return_counts=True, axis=0)
    ranked = sorted(((int(n), (int(tx), int(ty))) for (tx, ty), n in zip(offsets, counts, strict=True) if n >= 16),
                    key=lambda entry: (-entry[0], entry[1]))
    return [shift for _, shift in ranked[:MAX_OFFSETS]]


def _regions(rgb, dx, dy):
    height, width = rgb.shape[:2]
    x, y = max(0, -dx), max(0, -dy)
    right, bottom = min(width, width - dx), min(height, height - dy)
    equality = np.all(rgb[y:bottom, x:right] == rgb[y + dy:bottom + dy, x + dx:right + dx], axis=2)
    # Exact equality component is already the crop mask. Requiring full fill
    # removes holey graphic and palette matches.
    _, _, boxes, _ = cv2.connectedComponentsWithStats(equality.astype(np.uint8), connectivity=4)
    for left, top, w, h, area in boxes[1:].tolist():
        if MIN_CROP <= min(w, h) <= max(w, h) <= MAX_CROP and area == w * h:
            yield (x + left, y + top, w, h), (x + left + dx, y + top + dy, w, h)


def analyse(rgb):
    height, width = rgb.shape[:2]
    if not MIN_SIDE <= min(height, width) <= max(height, width) <= MAX_SIDE:
        return {"status": "unsupported_geometry", "pairs": []}
    offsets = _offsets(rgb)
    if offsets is None:
        return {"status": "unassessed_match_budget", "pairs": []}
    pairs = []
    for dx, dy in offsets:
        for first, second in _regions(rgb, dx, dy):
            if (proof := witness(rgb, first, second)) is not None:
                pairs.append({"pair": (first, second), "measurement": proof})
    if len(pairs) > 1:
        return {"status": "ambiguous_pairs", "pairs": []}
    return {"status": "supported_pair" if pairs else "no_pair", "pairs": pairs}
