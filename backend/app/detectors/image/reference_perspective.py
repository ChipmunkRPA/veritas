"""Locate local differences between two projectively aligned JPEG photographs."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.detectors.image.reference_crop_resize import matching_points

SCHEMA = "chipmunk.image-reference-comparison/25"
MIN_SIDE = 320
MAX_SIDE = 1024
TILE_COUNT = 4
MIN_INLIERS = 28
MAX_REGIONS = 5
MIN_OVERLAP = .70
MIN_CORRELATION = .94
INTERPRETATION = (
    "The images share detail across the matched area. Highlighted pixels differ after accounting for perspective and overall color. "
    "Ordinary editing and resampling can cause differences. The comparison carries zero weight in the AI assessment."
)


@dataclass(frozen=True)
class Measurement:
    transform: np.ndarray
    coverage: np.ndarray
    difference: np.ndarray
    evidence: dict


def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _project(matrix, points):
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float64)[None], matrix)[0]


def _bounds(w, h):
    return [(0., 0.), (w - 1., 0.), (w - 1., h - 1.), (0., h - 1.)]


def _alignment(reference, submitted):
    match = matching_points(reference, submitted)
    if match is None:
        return None
    points, target = match
    matrix, consensus = cv2.findHomography(points, target, cv2.USAC_MAGSAC, 1.4, maxIters=1500, confidence=.997)
    if matrix is None or consensus is None or not np.all(np.isfinite(matrix)) or abs(matrix[2, 2]) < 1e-6:
        return None
    inliers = consensus.ravel() > 0
    h, w = submitted.shape
    if (inliers.sum() < max(MIN_INLIERS, .70 * len(points))
            or np.any(np.ptp(points[inliers], axis=0) < .68 * np.array((w, h)))):
        return None
    matrix = np.round(matrix / matrix[2, 2], 11)
    corners = _bounds(w, h)
    projected = _project(matrix, corners).astype(np.float32)
    polygon = cv2.contourArea(projected, oriented=True)
    denominators = np.column_stack((corners, np.ones(4))) @ matrix[2]
    if (not cv2.isContourConvex(projected) or not .52 <= polygon / (w * h) <= 2.1
            or min(denominators) < .75 or max(denominators) > 1.25):
        return None
    errors = np.linalg.norm(_project(matrix, points[inliers]) - target[inliers], axis=1)
    if np.median(errors) > .62 or np.quantile(errors, .90) > 1.25:
        return None
    return matrix, int(inliers.sum()), float(np.median(errors))


def _tiles(earlier, submitted, coverage):
    h, w = submitted.shape
    rows = []
    for i in range(TILE_COUNT):
        dy = slice(i * h // TILE_COUNT, (i + 1) * h // TILE_COUNT, 2)
        for j in range(TILE_COUNT):
            dx = slice(j * w // TILE_COUNT, (j + 1) * w // TILE_COUNT, 2)
            valid = coverage[dy, dx]
            if valid.mean() < .80:
                continue
            anchor, observed = earlier[dy, dx][valid], submitted[dy, dx][valid]
            anchor, observed = anchor - anchor.mean(), observed - observed.mean()
            if min(anchor.std(), observed.std()) < 5:
                continue
            correlation = float(np.dot(anchor, observed) / (np.linalg.norm(anchor) * np.linalg.norm(observed)))
            if correlation >= MIN_CORRELATION:
                rows.append({"row": i, "column": j, "correlation": round(min(correlation, 1.), 6)})
    quadrants = {(row["row"] // 2, row["column"] // 2) for row in rows}
    return rows if len(rows) >= 10 and len(quadrants) == 4 else None


def _tone_fit(earlier, submitted, valid):
    take = valid[::4, ::4]
    original = earlier[::4, ::4][take]
    observed = submitted[::4, ::4][take]
    fits = []
    for a, b in zip(original.T, observed.T, strict=True):
        inliers = (a > 12) & (a < 243) & (b > 12) & (b < 243)
        gain, offset = 1., float(np.median((b - a)[inliers])) if inliers.any() else 0.
        for _ in range(3):
            x, y = a[inliers], b[inliers]
            if len(x) < 200 or x.std() < 10.:
                break
            gain = float(np.mean((x - x.mean()) * (y - y.mean())) / x.var())
            offset = float(y.mean() - gain * x.mean())
            residual = np.abs(gain * a + offset - b)
            inliers &= residual <= max(3., float(np.quantile(residual[inliers], .82)))
        fits.append((gain, offset))
    fit = np.asarray(fits, dtype=np.float32).round(6)
    if np.any((fit[:, 0] < .78) | (fit[:, 0] > 1.22) | (np.abs(fit[:, 1]) > 28.)):
        return None
    return fit


def measure(reference, submitted):
    side_lengths = (*reference.shape[:2], *submitted.shape[:2])
    if min(side_lengths) < MIN_SIDE or max(side_lengths) > MAX_SIDE:
        return None
    earlier_gray, observed_gray = _gray(reference), _gray(submitted)
    if (fitted := _alignment(earlier_gray, observed_gray)) is None:
        return None
    matrix, count, median_error = fitted
    h, w = observed_gray.shape
    size = (w, h)
    source_inner = np.zeros(earlier_gray.shape, dtype=np.uint8)
    source_inner[3:-3, 3:-3] = 1
    coverage = cv2.warpPerspective(source_inner, matrix, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    coverage = cv2.erode(coverage, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT).astype(bool)
    if coverage.mean() < MIN_OVERLAP:
        return None
    aligned = cv2.warpPerspective(reference.astype(np.float32), matrix, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    before = cv2.GaussianBlur(aligned, (0, 0), 1.4)
    after = cv2.GaussianBlur(submitted.astype(np.float32), (0, 0), 1.4)
    gray_before, gray_after = _gray(before), _gray(after)
    tiles = _tiles(gray_before, gray_after, coverage)
    tone = _tone_fit(before, after, coverage) if tiles else None
    if tone is None:
        return None
    delta = np.max(np.abs(after - (before * tone[:, 0] + tone[:, 1]).clip(0., 255.)), axis=2)
    gx = cv2.Sobel(gray_before, cv2.CV_32F, 1, 0, scale=.125)
    gy = cv2.Sobel(gray_before, cv2.CV_32F, 0, 1, scale=.125)
    difference = delta - .50 * np.hypot(gx, gy)
    return Measurement(matrix, coverage, difference, {
        "submitted_to_earlier_pixel_centers": matrix.tolist(),
        "inlier_matches": count, "median_match_error_px": round(median_error, 6),
        "compared_pixels": int(coverage.sum()), "supporting_tiles": tiles,
        "tone_adjustment": np.round(tone.astype(float), 6).tolist(),
    })


def selection(measurement, point, earlier_size):
    values, coverage = measurement.difference, measurement.coverage
    possible = (values > .65 * point) & coverage
    expanded = cv2.morphologyEx(possible.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    expanded &= coverage
    _, labels, count, _ = cv2.connectedComponentsWithStats(expanded, connectivity=8)
    areas = count[:, cv2.CC_STAT_AREA]
    strong_counts = np.bincount(labels[values > point], minlength=len(areas))
    minimum = max(80, round(coverage.sum() * .00035))
    allowed = [i for i in range(1, len(areas)) if areas[i] >= minimum and strong_counts[i] >= .20 * areas[i]]
    result = np.zeros(coverage.shape, dtype=bool)
    if sum(areas[i] for i in allowed) > .14 * coverage.sum():
        return None, None, 0
    ranked = sorted(allowed, key=lambda i: (-int(strong_counts[i]), -int(areas[i]), int(i)))
    regions = []
    for index in ranked[:MAX_REGIONS]:
        x, y, w, h, pixels = map(int, count[index])
        local = labels[y:y+h, x:x+w] == index
        result[y:y+h, x:x+w] |= local
        bounds = [(x0 + x, y0 + y) for x0, y0 in _bounds(w, h)]
        corners = _project(measurement.transform, bounds)
        left, top = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
        right, bottom = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 1, earlier_size)
        regions.append({
            "x": x, "y": y, "w": w, "h": h, "selected_pixels": pixels,
            "reference_region": {"x": int(left), "y": int(top), "w": int(right - left), "h": int(bottom - top)},
            "earlier_pixel_centers": corners.round(5).tolist(),
        })
    return regions, result, len(ranked[MAX_REGIONS:])


def compare(measured, earlier_size, point):
    if measured is None:
        return None, None
    regions, mask, omitted = selection(measured, point, earlier_size)
    if regions is None:
        return None, None
    return {
        "status": "comparable", "regions": regions, "additional_regions": omitted,
        "selected_pixels": int(mask.sum()),
        "affects_overall_confidence": False,
        "perspective_check": {**measured.evidence, "minimum_difference": point},
    }, mask
