"""Propose crop geometry from visual landmarks, then confirm consecutive motion."""
from __future__ import annotations

import math

import cv2
import numpy as np

from app.detectors.video import picture_crop as coarse
from app.evidence.picture_sequence import MIN_SPAN

UNIT = 32  # Fixed-point working pixels, stored in the evidence record.
SAMPLE_STEP = 6
MAX_FEATURES = 100
MAX_PROPOSALS = 16
MIN_FRACTION, MAX_FRACTION = .69, .925
MIN_INLIERS = 8
MAX_PICTURE_RMS = 3.25


def _points(image, descriptor):
    points, values = descriptor.detectAndCompute((image + 128).astype(np.uint8), None)
    return np.array([item.pt for item in points], dtype=np.float32), values


def _crop_box(scale, x, y):
    w, h = coarse.WORK_WIDTH, coarse.WORK_HEIGHT
    if not MIN_FRACTION <= scale <= MAX_FRACTION or not -1 <= x <= w * (1 - scale) + 1 or not -1 <= y <= h * (1 - scale) + 1:
        return None
    left, top = min(max(0., x), w * (1 - scale)), min(max(0., y), h * (1 - scale))
    return tuple(round(value * UNIT) for value in (left, top, left + w * scale, top + h * scale))


def _geometry(matrix):
    scale, sine = float(matrix[0, 0]), float(matrix[1, 0])
    if abs(sine) > .012 or abs(scale - float(matrix[1, 1])) > .018 or abs(float(matrix[0, 1])) > .018:
        return None
    return _crop_box(scale, *(float(offset) + (1 - scale) * .5 for offset in matrix[:, 2]))


def _geometry_proposal(first, second, matcher):
    a, descriptors_a = first
    b, descriptors_b = second
    if descriptors_a is None or descriptors_b is None or min(len(a), len(b)) < MIN_INLIERS:
        return None
    matches = [near for near, far in matcher.knnMatch(descriptors_a, descriptors_b, k=2) if near.distance < .78 * far.distance]
    if len(matches) < MIN_INLIERS:
        return None
    source = a[[item.queryIdx for item in matches]]
    target = b[[item.trainIdx for item in matches]]
    matrix, inliers = cv2.estimateAffinePartial2D(source, target, ransacReprojThreshold=1.15, maxIters=250)
    if matrix is None or not np.isfinite(matrix).all() or float(np.linalg.det(matrix[:, :2])) <= 0:
        return None
    admit = inliers.ravel() != 0
    number = int(admit.sum())
    if number < MIN_INLIERS or number * 2 < len(matches):
        return None
    inverse = matrix[0, 0] > 1
    original_support = source[admit] if inverse else target[admit]
    if any(np.ptp(original_support, axis=0) < [coarse.WORK_WIDTH * .22, coarse.WORK_HEIGHT * .23]):
        return None
    box = _geometry(cv2.invertAffineTransform(matrix) if inverse else matrix)
    return (number, box, inverse) if box else None


def _refine(narrow, wide, box):
    left, top, right, bottom = (value / UNIT for value in box)
    scale = (right - left) / coarse.WORK_WIDTH
    matrix = np.array([[scale, 0., left - (1 - scale) * .5],
                       [0., (bottom - top) / coarse.WORK_HEIGHT, top - (1 - scale) * .5]], dtype=np.float32)
    try:
        _, adjusted = cv2.findTransformECC(narrow, wide, matrix, cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 18, 1e-5), None, 5)
    except cv2.error:
        return box
    improved = _geometry(adjusted)
    if improved and max(abs(a - b) for a, b in zip(improved, box, strict=True)) <= 2 * UNIT:
        return improved
    return box


def _landmark_boxes(frames, minimum_lag):
    detector = cv2.SIFT_create(nfeatures=MAX_FEATURES, contrastThreshold=.015)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    indices = range(0, len(frames), SAMPLE_STEP)
    points = {index: _points(frames[index], detector) for index in indices}
    proposed = []
    for i in indices:
        for j in indices:
            if j - i >= minimum_lag and (match := _geometry_proposal(points[i], points[j], matcher)):
                votes, region, inverse = match
                proposed.append((votes, i, j, region, inverse))
    boxes = []
    for _, i, j, region, inverse in sorted(proposed, key=lambda row: (-row[0], row[1], row[2])):
        if any(max(abs(a - b) for a, b in zip(region, old, strict=True)) <= 1.6 * UNIT for old in boxes):
            continue
        narrow, wide = (j, i) if inverse else (i, j)
        boxes.append(_refine(frames[narrow], frames[wide], region))
        if len(boxes) == MAX_PROPOSALS:
            break
    return boxes


def _integral(frames):
    pixels = np.stack(frames, axis=0).astype(np.float64)
    return np.pad(pixels.cumsum(axis=1).cumsum(axis=2), ((0, 0), (1, 0), (1, 0)))


def _crop_vectors(sums, box):
    left, top, right, bottom = (value / UNIT for value in box)
    x = np.linspace(left, right, coarse.GRID_WIDTH + 1)
    y = np.linspace(top, bottom, coarse.GRID_HEIGHT + 1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0 + 1, coarse.WORK_WIDTH), np.minimum(y0 + 1, coarse.WORK_HEIGHT)
    dx, dy = x - x0, (y - y0)[:, None]
    upper = (1 - dx) * sums[:, y0[:, None], x0] + dx * sums[:, y0[:, None], x1]
    lower = (1 - dx) * sums[:, y1[:, None], x0] + dx * sums[:, y1[:, None], x1]
    sampled = (1 - dy) * upper + dy * lower
    area = (right - left) * (bottom - top) / (coarse.GRID_WIDTH * coarse.GRID_HEIGHT)
    return (np.diff(np.diff(sampled, axis=1), axis=2).reshape(len(sums), -1) / area).astype(np.float32)


def _verify(sums, boxes, full, power, distinct, motion, rms, minimum):
    best_rank, best_match = None, None
    for box in dict.fromkeys(boxes):
        crop = _crop_vectors(sums, box)
        distances = coarse._distance(crop, full, np.square(crop).mean(axis=1), power)
        close = (distances <= MAX_PICTURE_RMS ** 2) & distinct
        linked = close[:-1, :-1] & close[1:, 1:]
        if np.count_nonzero(linked) < coarse.MIN_EDGES:
            continue
        crop_motion, crop_rms = coarse._motion(crop)
        weaker = np.minimum(crop_rms[:, None], rms[None, :])
        stronger = np.maximum(crop_rms[:, None], rms[None, :])
        correlation = crop_motion @ motion.T / full.shape[1]
        moving = ((weaker >= coarse.MIN_MOTION_RMS_25FPS) & (weaker >= coarse.MIN_MOTION_RATIO * stronger)
                  & (correlation >= coarse.MIN_MOTION_COSINE * weaker * stronger))
        for wider, connections, changes, errors in (
            ("first", linked, moving, distances), ("second", linked.T, moving.T, distances.T),
        ):
            for start, end, lag, support in coarse._chains(connections, changes, minimum):
                index = np.arange(start, end + 1)
                error = float(np.mean(errors[index, index + lag]))
                rank = end - start, len(support), -error, -start, -lag
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_match = coarse.CropMatch(((start, end), (start + lag, end + lag)), wider, box, support)
    return best_match


def find(frames):
    """Compare centered subpixel crops and estimated off-center geometry at 25 fps."""
    minimum = math.ceil(25 * MIN_SPAN)
    if len(frames) < 2 * (minimum + 1):
        return {"centered": None, "landmarks": None}
    w, h = coarse.WORK_WIDTH, coarse.WORK_HEIGHT
    sums = _integral(frames)
    full = _crop_vectors(sums, (0, 0, w * UNIT, h * UNIT))
    power = np.square(full).mean(axis=1)
    distinct = coarse._distance(full, full, power, power) >= coarse.MIN_DIRECT_RMS ** 2
    motion, rms = coarse._motion(full)
    centered = [_crop_box(p / 100, w * (100 - p) / 200, h * (100 - p) / 200) for p in coarse.CROP_PERCENTAGES]
    anchors = _landmark_boxes(frames, minimum + 1)
    return {
        "centered": _verify(sums, centered, full, power, distinct, motion, rms, minimum),
        "landmarks": _verify(sums, anchors, full, power, distinct, motion, rms, minimum),
    }
