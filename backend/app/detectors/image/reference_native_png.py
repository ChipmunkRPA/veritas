"""Exact localized pixel differences between two PNGs on the same framed grid."""
from __future__ import annotations

import hashlib
from collections import Counter

import cv2
import numpy as np

from app.detectors.image.reference_compare import MAX_REGIONS

SCHEMA = "chipmunk.image-reference-comparison/31"
INTERPRETATION = ("The two PNGs share the same native pixel grid. Highlighted pixels differ from the earlier copy. "
                  "Ordinary edits can also produce these differences. This comparison carries zero weight in the AI assessment.")
MAX_PIXELS = 4_000_000
MIN_SIDE = 160
TILE = 24
PIXELS_PER_REGION = 16
COLOR_KEYS = ("icc_profile", "gamma", "srgb", "chromaticity", "cicp")


def _supported(image, data):
    return (image.format == "PNG" and image.mode == "RGB" and image.n_frames == 1
            and data[24:26] == b"\x08\x02" and "transparency" not in image.info
            and image.getexif().get(274, 1) == 1 and min(image.size) >= MIN_SIDE
            and image.width * image.height <= MAX_PIXELS)


def _textured_tiles(pixels):
    height, width = pixels.shape[:2]
    cells = {}
    for y in range(0, height - TILE + 1, TILE):
        for x in range(0, width - TILE + 1, TILE):
            tile = pixels[y:y + TILE, x:x + TILE]
            scaled = tile.astype(np.int16)
            if all(np.count_nonzero(np.any(np.abs(np.diff(scaled, axis=axis)) >= 8, axis=2)) >= 24
                   for axis in (0, 1)):
                cells[x, y] = hashlib.blake2s(tile.tobytes()).digest()
    return cells


def _quadrants(earlier, submitted):
    first, second = _textured_tiles(earlier), _textured_tiles(submitted)
    first_counts, second_counts = Counter(first.values()), Counter(second.values())
    height, width = submitted.shape[:2]
    counts = [0, 0, 0, 0]
    for (x, y), signature in first.items():
        if first_counts[signature] == second_counts[signature] == 1 and second.get((x, y)) == signature:
            counts[2 * (2 * y + TILE >= height) + (2 * x + TILE >= width)] += 1
    return counts


def _measure_regions(changed):
    # Dilation associates close pieces of a glyph or thin stroke. The measured
    # mask below still contains only original differing pixels.
    nearby = cv2.dilate(changed.astype(np.uint8), np.ones((3, 3), dtype=np.uint8))
    _, labels, bounds, _ = cv2.connectedComponentsWithStats(nearby, connectivity=8)
    sizes = np.bincount(labels[changed])
    candidates = sorted((index for index in range(1, len(sizes)) if sizes[index] >= PIXELS_PER_REGION),
                        key=lambda index: (-sizes[index], index))
    mask = np.zeros(changed.shape, dtype=bool)
    regions = []
    for index in candidates:
        x, y, width, height = map(int, bounds[index, :4])
        window = np.s_[y:y + height, x:x + width]
        component = (labels[window] == index) & changed[window]
        rows, columns = np.nonzero(component)
        left, right = x + int(columns.min()), x + int(columns.max()) + 1
        top, bottom = y + int(rows.min()), y + int(rows.max()) + 1
        # The crop's selected-pixel count must be its own overlay count even
        # when an unrelated component sits inside a hollow stroke.
        if any(left < region["x"] + region["w"] and right > region["x"]
               and top < region["y"] + region["h"] and bottom > region["y"] for region in regions):
            continue
        regions.append({"x": left, "y": top, "w": right - left, "h": bottom - top,
                        "selected_pixels": int(sizes[index])})
        mask[window] |= component
        if len(regions) == MAX_REGIONS:
            break
    return regions, mask, len(candidates) - len(regions)


def compare(earlier, submitted, before, after, earlier_image, submitted_image):
    if (earlier.shape != submitted.shape or not _supported(earlier_image, before) or not _supported(submitted_image, after)
            or any(earlier_image.info.get(key) != submitted_image.info.get(key) for key in COLOR_KEYS)):
        return None, None
    changed = np.any(earlier != submitted, axis=2)
    difference = int(changed.sum())
    # Keep identical pairs on their existing route. Two unrelated images that
    # share a blank background still need exact distinctive detail on both axes.
    if not difference or difference * 50 > changed.size:
        return None, None
    support = _quadrants(earlier, submitted)
    if min(support) < 2:
        return None, None
    regions, mask, omitted = _measure_regions(changed)
    return {
        "status": "comparable", "regions": regions, "additional_regions": omitted, "affects_overall_confidence": False,
        "native_png": {"exact_common_pixels": changed.size - difference, "changed_pixels": difference,
                       "unshown_changed_pixels": difference - int(mask.sum()), "matching_tiles_by_quadrant": support,
                       "minimum_region_pixels": PIXELS_PER_REGION},
    }, mask
