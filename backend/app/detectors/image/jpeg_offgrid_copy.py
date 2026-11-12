# SPDX-License-Identifier: Apache-2.0
"""Find small translated JPEG regions with matching texture and distinct perimeters."""

from collections import defaultdict

import cv2
import numpy as np

MAX_REFINEMENT_SEEDS = 12_000


def _descriptors(image, x, y):
    steps = np.arange(-8, 9, 5)
    values = image[y[:, None, None] + steps[None, :, None], x[:, None, None] + steps[None, None, :]]
    vectors = values.reshape(len(x), 16).astype(np.float32)
    vectors -= vectors.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.square(vectors).mean(axis=1))
    return vectors / np.maximum(scale[:, None], 1e-6) / 4.0, scale


def _correlation(region, comparison):
    a, b = region.ravel(), comparison.ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b) / max(1e-3, np.linalg.norm(a) * np.linalg.norm(b))


def _refine_offsets(blurred, votes):
    height, width = blurred.shape
    offsets = defaultdict(set)
    ranked = sorted((item for item in votes.items() if len(item[1]) >= 2), key=lambda item: (-len(item[1]), item[0]))[:40]
    if sum(len(seeds) for _, seeds in ranked) > MAX_REFINEMENT_SEEDS:
        return None
    for (dx, dy), seeds in ranked:
        refined = defaultdict(set)
        for x, y in sorted(seeds):
            left = blurred[y - 8:y + 9, x - 8:x + 9]
            possible = (
                (_correlation(left, blurred[b - 8:b + 9, a - 8:a + 9]), a - x, b - y)
                for a in range(max(8, x + dx - 2), min(width - 9, x + dx + 2) + 1)
                for b in range(max(8, y + dy - 2), min(height - 9, y + dy + 2) + 1)
            )
            score, shift_x, shift_y = max(possible)
            if score >= .95:
                refined[shift_x, shift_y].add((x, y))
        for (shift_x, shift_y), matched in refined.items():
            if len(matched) >= 2 and (shift_x % 8 or shift_y % 8):
                offsets[shift_x, shift_y].update(matched)
    return sorted(offsets.items(), key=lambda item: (-len(item[1]), item[0]))[:12]


def _proposals(blurred):
    height, width = blurred.shape
    y, x = np.mgrid[10:height - 10:2, 10:width - 10:2]
    x, y = x.ravel(), y.ravel()
    vectors, scale = _descriptors(blurred, x, y)
    textured = scale >= 5
    x, y, vectors = x[textured], y[textured], vectors[textured]
    probes = ((x - 10) % 10 == 0) & ((y - 10) % 10 == 0)
    if len(vectors) < 12 or probes.sum() < 2:
        return []
    cv2.setRNGSeed(902)
    index = cv2.flann_Index(vectors, {"algorithm": 1, "trees": 3})
    indices, distances = index.knnSearch(vectors[probes], 12, params={"checks": 32})
    probe_x, probe_y = x[probes], y[probes]
    votes = defaultdict(set)
    for probe, neighbor in zip(*np.where(distances < .14)):
        match = indices[probe, neighbor]
        first, second = sorted(((int(probe_x[probe]), int(probe_y[probe])), (int(x[match]), int(y[match]))))
        dx, dy = second[0] - first[0], second[1] - first[1]
        if max(abs(dx), abs(dy)) >= 32:
            votes[dx, dy].add(first)
    return _refine_offsets(blurred, votes)


def _correlation_field(a, b):
    def mean(pixels):
        return cv2.boxFilter(pixels, -1, (13, 13))

    mean_a, mean_b = mean(a), mean(b)
    var_a = np.maximum(0, mean(a * a) - mean_a * mean_a)
    var_b = np.maximum(0, mean(b * b) - mean_b * mean_b)
    corr = (mean(a * b) - mean_a * mean_b) / np.sqrt(np.maximum(.001, var_a * var_b))
    return corr, np.sqrt(np.minimum(var_a, var_b))


def _rectangles(gray, blurred, offsets):
    band = cv2.GaussianBlur(gray, (0, 0), .8) - cv2.GaussianBlur(gray, (0, 0), 2.7)
    height, width = gray.shape
    kernel = np.ones((9, 9), np.uint8)
    for (dx, dy), seeds in offsets:
        left, top = max(0, -dx), max(0, -dy)
        right, bottom = min(width, width - dx), min(height, height - dy)
        a = np.s_[top:bottom, left:right]
        b = np.s_[top + dy:bottom + dy, left + dx:right + dx]
        coarse, texture = _correlation_field(blurred[a], blurred[b])
        fine, detail = _correlation_field(band[a], band[b])
        agree = (coarse > .93) & (fine > .82) & (texture >= 3) & (detail >= 1.3)
        agree[:6] = agree[-6:] = agree[:, :6] = agree[:, -6:] = False
        closed = cv2.morphologyEx(agree.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        _, _, boxes, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for x, y, w, h, area in boxes[1:].tolist():
            if not (17 <= w <= 110 and 17 <= h <= 110 and area >= .5 * w * h):
                continue
            count = sum(x - 5 <= sx - left < x + w + 5 and y - 5 <= sy - top < y + h + 5 for sx, sy in seeds)
            if not count or agree[y:y + h, x:x + w].mean() < .58:
                continue
            l, t = max(0, x - 4), max(0, y - 4)
            r, d = min(right - left, x + w + 4), min(bottom - top, y + h + 4)
            if abs(dx) < r - l and abs(dy) < d - t:
                continue
            yield count, area, ((left + l, top + t, r - l, d - t), (left + l + dx, top + t + dy, r - l, d - t))


def _perimeter(blurred, first, second):
    x, y, w, h = first
    a, b, _, _ = second
    dx, dy = a - x, b - y
    height, width = blurred.shape[:2]
    interior = np.abs(blurred[y + 6:y + h - 6, x + 6:x + w - 6]
                      - blurred[b + 6:b + h - 6, a + 6:a + w - 6]).mean(axis=2)
    mean = float(interior.mean())
    upper = float(np.quantile(interior, .90))
    strips = ((x - 7, y + 4, x - 2, y + h - 4), (x + w + 2, y + 4, x + w + 7, y + h - 4),
              (x + 4, y - 7, x + w - 4, y - 2), (x + 4, y + h + 2, x + w - 4, y + h + 7))
    perimeter = []
    for l, t, r, d in strips:
        if min(l, t, l + dx, t + dy) < 0 or max(r, r + dx) > width or max(d, d + dy) > height:
            perimeter.append(None)
        else:
            perimeter.append(float(np.abs(blurred[t:d, l:r] - blurred[t + dy:d + dy, l + dx:r + dx]).mean()))
    if mean > 3.2 or upper > 6 or sum(v is not None and v >= max(7, 2.5 * mean) for v in perimeter) < 3:
        return None
    return {"interior_mean": round(mean, 5), "interior_p90": round(upper, 5),
            "perimeter_means": [None if v is None else round(v, 5) for v in perimeter]}


def witness(rgb, pair):
    return _perimeter(cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 1.25), *pair)


def analyse(rgb):
    height, width = rgb.shape[:2]
    if not 384 <= min(height, width) <= max(height, width) <= 1024 or max(height, width) < 513:
        return "unsupported_geometry", []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.15)
    offsets = _proposals(blurred)
    if offsets is None:
        return "unassessed_match_budget", []
    if not offsets:
        return "no_pair", []
    color = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 1.25)
    for _, _, pair in sorted(_rectangles(gray, blurred, offsets), reverse=True):
        if (metrics := _perimeter(color, *pair)) is not None:
            return "supported_pair", [{"pair": pair, "measurement": metrics}]
    return "no_pair", []
