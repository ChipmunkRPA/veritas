# SPDX-License-Identifier: Apache-2.0
"""Find disjoint JPEG regions with closely matching color after a right-angle turn."""

from collections import defaultdict

import cv2
import numpy as np

from app.detectors.image.copy_move import iou
from app.detectors.image.exact_turned_copy import _source_box
from app.detectors.image.jpeg_offgrid_copy import MAX_REFINEMENT_SEEDS, _correlation_field, _descriptors

MIN_CROP_SIDE = 32


def _sample(image, *, step):
    h, w = image.shape
    y, x = np.mgrid[10:h - 10:step, 10:w - 10:step]
    x, y = x.ravel(), y.ravel()
    descriptors, contrast = _descriptors(image, x, y)
    texture = contrast >= 5
    return np.stack((x[texture], y[texture]), axis=1), descriptors[texture]


def _alignments(canvas, rotated, turns):
    h, w = canvas.shape
    rh, rw = rotated.shape
    points, desc = _sample(canvas, step=8)
    targets, rotated_desc = _sample(rotated, step=2)
    if len(targets) < 12 or len(points) < 2:
        return []
    cv2.setRNGSeed(902)
    index = cv2.flann_Index(rotated_desc, {"algorithm": 1, "trees": 3})
    nearest, distances = index.knnSearch(desc, 12, params={"checks": 36})
    votes = defaultdict(set)
    for row, col in zip(*np.where(distances < .14)):
        x, y = map(int, points[row])
        a, b = map(int, targets[nearest[row, col]])
        u, v, _, _ = _source_box((a, b, 1, 1), h, w, turns)
        if max(abs(x - u), abs(y - v)) >= 24:
            votes[a - x, b - y].add((x, y))

    refined = defaultdict(set)
    ranked = sorted((entry for entry in votes.items() if len(entry[1]) >= 2),
                    key=lambda item: (-len(item[1]), item[0]))[:45]
    if sum(len(seeds) for _, seeds in ranked) > MAX_REFINEMENT_SEEDS:
        return None
    for (dx, dy), seeds in ranked:
        for x, y in sorted(seeds):
            left, top = max(8, x + dx - 2), max(8, y + dy - 2)
            right, bottom = min(rw - 9, x + dx + 2), min(rh - 9, y + dy + 2)
            tile = canvas[y - 8:y + 9, x - 8:x + 9]
            search = rotated[top - 8:bottom + 9, left - 8:right + 9]
            scores = cv2.matchTemplate(search, tile, cv2.TM_CCOEFF_NORMED)
            _, score, _, (a, b) = cv2.minMaxLoc(scores)
            if score >= .95:
                refined[left + a - x, top + b - y].add((x, y))
    return sorted(
        ((delta, seeds) for delta, seeds in refined.items() if len(seeds) >= 2),
        key=lambda item: (-len(item[1]), item[0]),
    )[:14]


def _measure(left, right, boxes):
    (x, y, w, h), (a, b, _, _) = boxes
    inside = abs(left[y + 6:y + h - 6, x + 6:x + w - 6] - right[b + 6:b + h - 6, a + 6:a + w - 6]).mean(axis=2)
    mean, p90 = float(inside.mean()), float(np.quantile(inside, .90))
    if mean > 3.2 or p90 > 6:
        return None
    perimeters = []
    for l, t, r, d in [(-7, 4, -2, h - 4), (w + 2, 4, w + 7, h - 4), (4, -7, w - 4, -2), (4, h + 2, w - 4, h + 7)]:
        if (min(x + l, y + t, a + l, b + t) >= 0 and x + r <= left.shape[1] and a + r <= right.shape[1]
                and y + d <= left.shape[0] and b + d <= right.shape[0]):
            p = left[y + t:y + d, x + l:x + r]
            q = right[b + t:b + d, a + l:a + r]
            perimeters.append(float(np.abs(p - q).mean()))
        else:
            perimeters.append(None)
    if sum(value is not None and value >= max(7, 2.5 * mean) for value in perimeters) >= 3:
        return {"interior_mean": round(mean, 5), "interior_p90": round(p90, 5),
                "perimeter_means": [None if value is None else round(value, 5) for value in perimeters]}
    return None


def _regions(gray, rgray, band, color, turns, alignments):
    h, w = gray.shape
    rband, rcolor = [np.ascontiguousarray(np.rot90(image, turns)) for image in (band, color)]
    rh, rw = rgray.shape
    kernel = np.ones((9, 9), np.uint8)
    for (dx, dy), seeds in alignments:
        left, top, right, bottom = max(0, -dx), max(0, -dy), min(w, rw - dx), min(h, rh - dy)
        a = np.s_[top:bottom, left:right]
        b = np.s_[top + dy:bottom + dy, left + dx:right + dx]
        lo, spread = _correlation_field(gray[a], rgray[b])
        hi, detail = _correlation_field(band[a], rband[b])
        agree = (lo > .93) & (hi > .82) & (spread >= 3) & (detail >= 1.3)
        agree[:6] = agree[-6:] = agree[:, :6] = agree[:, -6:] = False
        closed = cv2.morphologyEx(agree.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        _, _, boxes, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for x, y, bw, bh, area in boxes[1:].tolist():
            if not (17 <= bw <= 110 and 17 <= bh <= 110 and area >= .5 * bw * bh):
                continue
            count = sum(x - 5 <= sx - left < x + bw + 5 and y - 5 <= sy - top < y + bh + 5 for sx, sy in seeds)
            if not count or agree[y:y + bh, x:x + bw].mean() < .58:
                continue
            l, t = max(0, x - 4), max(0, y - 4)
            r, d = min(right - left, x + bw + 4), min(bottom - top, y + bh + 4)
            first = left + l, top + t, r - l, d - t
            second = left + l + dx, top + t + dy, r - l, d - t
            source = _source_box(second, h, w, turns)
            if min(first[2:]) < MIN_CROP_SIDE or iou(first, source):
                continue
            if metrics := _measure(color, rcolor, (first, second)):
                yield count, area, (first, source), turns, metrics


def _color(rgb):
    return cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 1.25)


def witness(rgb, pair, turns):
    """Recompute the saved crop and perimeter readings on the supplied JPEG pixels."""
    _, second = pair
    x, y, w, h = second
    height, width = rgb.shape[:2]
    aligned = (y, width - x - w, h, w) if turns == 1 else (width - x - w, height - y - h, w, h)
    color = _color(rgb)
    return _measure(color, np.rot90(color, turns), (pair[0], aligned))


def _same_pair(a, b):
    return max(min(iou(a[0], b[0]), iou(a[1], b[1])), min(iou(a[0], b[1]), iou(a[1], b[0]))) >= .5


def analyse(rgb):
    height, width = rgb.shape[:2]
    if not 384 <= min(height, width) <= max(height, width) <= 1024 or max(height, width) < 513:
        return "unsupported_profile", []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.15)
    band = cv2.GaussianBlur(gray, (0, 0), .8) - cv2.GaussianBlur(gray, (0, 0), 2.7)
    color = _color(rgb)
    matches = []
    for turns in (1, 2):
        rotated = np.ascontiguousarray(np.rot90(blurred, turns))
        alignments = _alignments(blurred, rotated, turns)
        if alignments is None:
            return "unassessed_match_budget", []
        matches.extend(_regions(blurred, rotated, band, color, turns, alignments))
    if not matches:
        return "no_pair", []
    matches.sort(key=lambda item: item[:4], reverse=True)
    _, _, pair, turns, measurement = matches[0]
    if any(other_turns != turns or not _same_pair(other_pair, pair)
           for _, _, other_pair, other_turns, _ in matches[1:]):
        return "ambiguous", []
    return "supported_pair", [{"pair": pair, "turns": turns, "measurement": measurement}]
