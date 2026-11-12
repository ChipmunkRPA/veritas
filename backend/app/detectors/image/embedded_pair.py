"""Bounded exact RGB comparisons between two separated areas of one image."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

TILE = 16
STRIDE = 19
SAMPLES = ((0, 0), (1, 13), (4, 6), (6, 15), (8, 1), (10, 10), (14, 4), (15, 15))
MIX = (0x9e3779b185ebca87, 0xc2b2ae3d27d4eb4f, 0x165667b19e3779f9, 0x85ebca77c2b2ae63,
       0x27d4eb2f165667c5, 0xd6e8feb86659fd93, 0xa0761d6478bd642f, 0x8ebc6af09c88c6e3)
Box = tuple[int, int, int, int]
Point = tuple[int, int]


@dataclass(frozen=True)
class Pair:
    area_a: Box
    area_b: Box
    witnesses: tuple[Point, ...]
    difference: np.ndarray

    @property
    def difference_bounds(self) -> Box:
        y, x = np.nonzero(self.difference)
        return int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)


@dataclass(frozen=True)
class _Atlas:
    rgb: np.ndarray
    hashes: np.ndarray
    keys: np.ndarray
    order: np.ndarray

    def occurrences(self, x: int, y: int, minimum: int, maximum: int):
        key = self.hashes[y, x]
        left = int(np.searchsorted(self.keys, key, side="left"))
        right = int(np.searchsorted(self.keys, key, side="right"))
        if minimum <= right - left <= maximum:
            for index in self.order[left:right]:
                row, col = divmod(int(index), self.hashes.shape[1])
                yield col, row


def crop(rgb: np.ndarray, box: Box) -> np.ndarray:
    x, y, right, bottom = box
    return rgb[y:bottom, x:right]


def _atlas(rgb: np.ndarray) -> _Atlas:
    height, width = rgb.shape[:2]
    values = rgb.astype(np.uint32)
    packed = (values[:, :, 0] << 16) | (values[:, :, 1] << 8) | values[:, :, 2]
    rows, cols = height - TILE + 1, width - TILE + 1
    hashes = np.zeros((rows, cols), dtype=np.uint64)
    for (dy, dx), factor in zip(SAMPLES, MIX, strict=True):
        hashes ^= packed[dy:dy + rows, dx:dx + cols].astype(np.uint64) * np.uint64(factor)
    order = np.argsort(hashes, axis=None, kind="stable")
    return _Atlas(packed, hashes, hashes.ravel()[order], order)


def _tile(packed: np.ndarray, x: int, y: int) -> np.ndarray:
    return packed[y:y + TILE, x:x + TILE]


def _separated(a: Box, b: Box, margin: int) -> bool:
    return a[2] + margin <= b[0] or b[2] + margin <= a[0] or a[3] + margin <= b[1] or b[3] + margin <= a[1]


def _spread(points, minimum: int) -> Box | None:
    x = min(i for i, _ in points)
    y = min(j for _, j in points)
    right = max(i for i, _ in points) + TILE
    bottom = max(j for _, j in points) + TILE
    width, height = right - x, bottom - y
    if min(width, height) < minimum:
        return None
    occupied = {(min(2, 3 * (i - x) // width), min(2, 3 * (j - y) // height)) for i, j in points}
    return (x, y, right, bottom) if len(occupied) >= 6 else None


def _pair(rgb: np.ndarray, atlas: _Atlas) -> tuple[str, Pair | None]:
    shifts: dict[Point, set[Point]] = defaultdict(set)
    rows, cols = atlas.hashes.shape
    yy, xx = np.meshgrid(np.arange(0, rows, STRIDE), np.arange(0, cols, STRIDE), indexing="ij")
    samples = atlas.hashes[yy, xx].ravel()
    starts = np.searchsorted(atlas.keys, samples, side="left")
    ends = np.searchsorted(atlas.keys, samples, side="right")
    for y, x, start, end in zip(yy.ravel(), xx.ravel(), starts, ends, strict=True):
        if not 2 <= end - start <= 3:
            continue
        patch = _tile(atlas.rgb, x, y)
        if np.unique(patch).size < 48:
            continue
        for index in atlas.order[start:end]:
            b, a = divmod(int(index), cols)
            dx, dy = int(a - x), int(b - y)
            if max(abs(dx), abs(dy)) < 192 or not np.array_equal(patch, _tile(atlas.rgb, a, b)):
                continue
            if dx > 0 or (dx == 0 and dy > 0):
                shifts[dx, dy].add((int(x), int(y)))
            else:
                shifts[-dx, -dy].add((a, b))
    eligible = []
    for (dx, dy), points in shifts.items():
        if len(points) < 12 or (a := _spread(points, 160)) is None:
            continue
        b = (a[0] + dx, a[1] + dy, a[2] + dx, a[3] + dy)
        if not _separated(a, b, 24):
            continue
        difference = (crop(rgb, a) != crop(rgb, b)).any(axis=2)
        if int(difference.sum()) * 100 <= 15 * difference.size:
            eligible.append(Pair(a, b, tuple(sorted(points)), difference))
    if not eligible:
        return "no_supported_pair", None
    if len(eligible) != 1:
        return "unassessed_ambiguous_match", None
    pair = eligible[0]
    if not pair.difference.any():
        return "matching_without_difference", None
    x, y, right, bottom = pair.difference_bounds
    height, width = pair.difference.shape
    _, _, regions, _ = cv2.connectedComponentsWithStats(pair.difference.astype(np.uint8), connectivity=8)
    if min(x, y, width - right, height - bottom) < 8 or ((right - x) * (bottom - y) > .45 * pair.difference.size
                                                       or int(regions[1:, 4].max()) < 48):
        return "no_supported_pair", None
    return "local_difference", pair


def _exact_third(atlas: _Atlas, pair: Pair) -> bool:
    groups: dict[Point, set[Point]] = defaultdict(set)
    for x, y in pair.witnesses:
        for tx, ty in atlas.occurrences(x, y, 3, 8):
            window = (tx, ty, tx + TILE, ty + TILE)
            if (_separated(window, pair.area_a, 0) and _separated(window, pair.area_b, 0)
                    and np.array_equal(_tile(atlas.rgb, x, y), _tile(atlas.rgb, tx, ty))):
                groups[tx - x, ty - y].add((tx, ty))
    for points in groups.values():
        if (len(points) >= 12 and (box := _spread(points, 128)) is not None
                and _separated(box, pair.area_a, 8) and _separated(box, pair.area_b, 8)):
            return True
    return False


def _pixel_witnesses(gray: np.ndarray, matrix: np.ndarray, source: np.ndarray) -> int:
    height, width = gray.shape
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.)
    mapped = cv2.warpAffine(smooth, matrix, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    used = set()
    for x, y in source:
        x, y = round(float(x)), round(float(y))
        cell = x // TILE, y // TILE
        if not (8 <= x < width - 8 and 8 <= y < height - 8) or cell in used:
            continue
        p, q = (image[y - 8:y + 8, x - 8:x + 8].astype(np.float64).ravel() for image in (smooth, mapped))
        if min(p.std(), q.std()) >= 8 and float(np.corrcoef(p, q)[0, 1]) >= .97:
            used.add(cell)
    return len(used)


def _geometric_third(rgb: np.ndarray, pair: Pair) -> bool:
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inside = np.zeros((height, width), np.uint8)
    outside = np.full((height, width), 255, np.uint8)
    x, y, right, bottom = pair.area_a
    inside[y + 8:bottom - 8, x + 8:right - 8] = 255
    for x, y, right, bottom in (pair.area_a, pair.area_b):
        outside[max(0, y - 24):min(height, bottom + 24), max(0, x - 24):min(width, right + 24)] = 0
    feature = cv2.SIFT_create(nfeatures=1200)
    first, first_descriptors = feature.detectAndCompute(gray, inside)
    third, third_descriptors = feature.detectAndCompute(gray, outside)
    if first_descriptors is None or third_descriptors is None or len(third_descriptors) < 2:
        return False
    nearest = cv2.BFMatcher(cv2.NORM_L2).knnMatch(first_descriptors, third_descriptors, k=2)
    best = {}
    for m, n in nearest:
        if m.distance < .72 * n.distance and (m.trainIdx not in best or m.distance < best[m.trainIdx].distance):
            best[m.trainIdx] = m
    matches = sorted(best.values(), key=lambda match: (match.queryIdx, match.trainIdx))
    if len(matches) < 12:
        return False
    source = np.array([first[m.queryIdx].pt for m in matches], dtype=np.float32)
    target = np.array([third[m.trainIdx].pt for m in matches], dtype=np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=2.,
                                                 maxIters=1000, confidence=.995, refineIters=10)
    if matrix is None or inliers is None:
        return False
    kept = inliers.reshape(-1).astype(bool)
    count = int(kept.sum())
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    if count < 12 or count < .55 * len(matches) or not .60 <= scale <= 1.30 or abs(angle) > 5:
        return False
    source, target = source[kept], target[kept]
    source_box = _spread([(int(x) - 8, int(y) - 8) for x, y in source], 160)
    target_box = _spread([(int(x) - 8, int(y) - 8) for x, y in target], 128)
    return bool(source_box and target_box and _separated(target_box, pair.area_a, 8)
                and _separated(target_box, pair.area_b, 8) and _pixel_witnesses(gray, matrix, source) >= 10)


def inspect(rgb: np.ndarray) -> tuple[str, Pair | None]:
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
            or min(rgb.shape[:2]) < 384 or math.prod(rgb.shape[:2]) > 1_000_000):
        return "outside_scope", None
    atlas = _atlas(rgb)
    status, pair = _pair(rgb, atlas)
    if pair is not None and (_exact_third(atlas, pair) or _geometric_third(rgb, pair)):
        return "unassessed_ambiguous_match", None
    return status, pair
