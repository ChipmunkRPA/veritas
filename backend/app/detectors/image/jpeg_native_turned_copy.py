# SPDX-License-Identifier: Apache-2.0
"""Propose native right-angle matches on larger JPEGs with bounded local verification."""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from app.detectors.image import jpeg_turned_copy
from app.detectors.image.copy_move import iou
from app.detectors.image.exact_turned_copy import _source_box
from app.detectors.image.jpeg_offgrid_copy import _correlation_field

TILE = 160
CORNERS_PER_TILE = 28
MIN_CORNER_DISTANCE = 5
NEIGHBORS = 12
OFFSET_BIN = 4
REFINE_RADIUS = 4
MAX_SEEDS = 2400
MAX_OFFSETS = 24
WINDOW_HALF = 120
MAX_REGIONS = 72


def _features(gray):
    height, width = gray.shape
    points = []
    for y in range(0, height, TILE):
        for x in range(0, width, TILE):
            left, top = max(0, x - 6), max(0, y - 6)
            patch = gray[top : min(height, y + TILE + 6), left : min(width, x + TILE + 6)]
            corners = cv2.goodFeaturesToTrack(
                patch, CORNERS_PER_TILE, 0.025, MIN_CORNER_DISTANCE, blockSize=5
            )
            if corners is not None:
                points.extend(
                    (a + left, b + top)
                    for a, b in corners[:, 0].astype(int)
                    if max(x, 8) <= a + left < min(x + TILE, width - 8)
                    and max(y, 8) <= b + top < min(y + TILE, height - 8)
                )
    if not points:
        return np.empty((0, 2), dtype=int), np.empty((0, 4, 4), dtype=np.float32)
    centers = np.array(sorted(points), dtype=int)
    x, y = centers.T
    taps = np.array([-7, -2, 2, 7])
    samples = gray[y[:, None, None] + taps[None, :, None], x[:, None, None] + taps[None, None, :]]
    samples -= samples.mean(axis=(1, 2), keepdims=True)
    scale = np.sqrt(np.square(samples).mean(axis=(1, 2)))
    textured = scale >= 5
    return centers[textured], samples[textured] / (4 * scale[textured, None, None])


def _votes(points, descriptors, turns, shape):
    height, width = shape
    x, y = points.T
    transformed = (
        np.column_stack((y, width - 1 - x))
        if turns == 1
        else np.column_stack((width - 1 - x, height - 1 - y))
    )
    target = np.ascontiguousarray(np.rot90(descriptors, turns, axes=(1, 2)).reshape(len(points), 16))
    cv2.setRNGSeed(903)
    index = cv2.flann_Index(target, {"algorithm": 1, "trees": 3})
    neighbors, distances = index.knnSearch(
        np.ascontiguousarray(descriptors.reshape(len(points), 16)), NEIGHBORS, params={"checks": 48}
    )
    rows, columns = np.where(distances < 0.14)
    candidates = neighbors[rows, columns]
    far = np.max(np.abs(points[rows] - points[candidates]), axis=1) >= 32
    deltas = transformed[candidates[far]] - points[rows[far]]
    votes = defaultdict(set)
    for (sx, sy), (dx, dy) in zip(points[rows[far]].tolist(), deltas.tolist(), strict=True):
        votes[(round(dx / OFFSET_BIN) * OFFSET_BIN, round(dy / OFFSET_BIN) * OFFSET_BIN)].add((sx, sy))
    return sorted(
        ((delta, seeds) for delta, seeds in votes.items() if len(seeds) >= 2),
        key=lambda item: (-len(item[1]), item[0]),
    )[:MAX_OFFSETS]


def _refine(gray, rotated, votes):
    rh, rw = rotated.shape
    refined = defaultdict(set)
    for (dx, dy), seeds in votes:
        for x, y in sorted(seeds):
            left, right = max(8, x + dx - REFINE_RADIUS), min(rw - 9, x + dx + REFINE_RADIUS)
            top, bottom = max(8, y + dy - REFINE_RADIUS), min(rh - 9, y + dy + REFINE_RADIUS)
            if left > right or top > bottom:
                continue
            tile = gray[y - 8 : y + 9, x - 8 : x + 9]
            field = cv2.matchTemplate(
                rotated[top - 8 : bottom + 9, left - 8 : right + 9], tile, cv2.TM_CCOEFF_NORMED
            )
            _, score, _, (off_x, off_y) = cv2.minMaxLoc(field)
            if score >= 0.95:
                refined[left + off_x - x, top + off_y - y].add((x, y))
    return sorted(
        ((delta, seeds) for delta, seeds in refined.items() if len(seeds) >= 2),
        key=lambda item: (-len(item[1]), item[0]),
    )[:MAX_OFFSETS]


def _windows(gray_shape, rotated_shape, alignments):
    h, w = gray_shape
    rh, rw = rotated_shape
    for (dx, dy), remaining in alignments:
        pending = sorted(remaining)
        while pending:
            sx, sy = pending[0]
            left, top = max(0, -dx, sx - WINDOW_HALF), max(0, -dy, sy - WINDOW_HALF)
            right, bottom = min(w, rw - dx, sx + WINDOW_HALF + 1), min(h, rh - dy, sy + WINDOW_HALF + 1)
            inner = [(x, y) for x, y in pending if abs(x - sx) <= 67 and abs(y - sy) <= 67]
            pending = sorted(set(pending) - set(inner))
            if right - left >= 32 and bottom - top >= 32:
                yield (dx, dy), (left, top, right, bottom), [(x - left, y - top) for x, y in inner]


def _regions(gray, rotated, band, rotated_band, shifts, turns):
    h, w = gray.shape
    kernel = np.ones((9, 9), np.uint8)
    for (dx, dy), (left, top, right, bottom), seeds in shifts:
        a = np.s_[top:bottom, left:right]
        b = np.s_[top + dy : bottom + dy, left + dx : right + dx]
        lo, spread = _correlation_field(gray[a], rotated[b])
        hi, detail = _correlation_field(band[a], rotated_band[b])
        agree = (lo > 0.93) & (hi > 0.82) & (spread >= 3) & (detail >= 1.3)
        agree[:6] = agree[-6:] = agree[:, :6] = agree[:, -6:] = False
        closed = cv2.morphologyEx(agree.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        _, _, boxes, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for x, y, bw, bh, area in boxes[1:].tolist():
            if not (17 <= bw <= 110 and 17 <= bh <= 110 and area >= 0.5 * bw * bh):
                continue
            count = sum(x - 5 <= sx < x + bw + 5 and y - 5 <= sy < y + bh + 5 for sx, sy in seeds)
            if not count or agree[y : y + bh, x : x + bw].mean() < 0.58:
                continue
            l, t, r, d = (
                max(0, x - 4),
                max(0, y - 4),
                min(right - left, x + bw + 4),
                min(bottom - top, y + bh + 4),
            )
            first = (left + l, top + t, r - l, d - t)
            aligned = (left + l + dx, top + t + dy, r - l, d - t)
            second = _source_box(aligned, h, w, turns)
            if min(first[2:]) >= jpeg_turned_copy.MIN_CROP_SIDE and not iou(first, second):
                yield count, area, (first, second), aligned


def analyse(rgb):
    height, width = rgb.shape[:2]
    if not 1024 <= min(height, width) <= max(height, width) <= 3072 or max(height, width) <= 1024:
        return "unsupported_profile", []
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(luma, (0, 0), 1.15)
    points, descriptors = _features(gray)
    if len(points) < NEIGHBORS:
        return "no_pair", []
    band = cv2.GaussianBlur(luma, (0, 0), 0.8) - cv2.GaussianBlur(luma, (0, 0), 2.7)
    matches, color = [], None
    for turns in (1, 2):
        votes = _votes(points, descriptors, turns, gray.shape)
        if sum(len(seeds) for _, seeds in votes) > MAX_SEEDS:
            return "unassessed_match_budget", []
        rotated = np.rot90(gray, turns)
        offsets = _refine(gray, rotated, votes)
        windows = list(_windows(gray.shape, rotated.shape, offsets))
        if len(windows) > MAX_REGIONS:
            return "unassessed_match_budget", []
        for count, area, pair, aligned in _regions(
            gray, rotated, band, np.rot90(band, turns), windows, turns
        ):
            if color is None:
                color = jpeg_turned_copy._color(rgb)
            if witness := jpeg_turned_copy._measure(color, np.rot90(color, turns), (pair[0], aligned)):
                matches.append((count, area, pair, turns, witness))
    if not matches:
        return "no_pair", []
    _, _, pair, turns, witness = max(matches, key=lambda item: item[:4])
    if any(
        other_turn != turns or not jpeg_turned_copy._same_pair(other_pair, pair)
        for _, _, other_pair, other_turn, _ in matches
    ):
        return "ambiguous", []
    return "supported_pair", [{"pair": pair, "turns": turns, "measurement": witness}]
