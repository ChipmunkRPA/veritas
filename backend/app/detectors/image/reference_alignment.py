"""Establish an integer correspondence between independently supplied images."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import reference_compare

MIN_MATCHES = 18
MIN_MATCH_SHARE = .60
MIN_OVERLAP = .80
MIN_TILE_SIDE = 32
TILE_SEARCH = 5
MIN_TILE_SCORE = .83
MIN_TILE_SEPARATION = .012
MAX_FRACTIONAL_OFFSET = .35
MIN_TILES = 6
MIN_SIDE = 160


def _matches(gray_reference, gray_submitted):
    extractor = cv2.ORB_create(nfeatures=2400, fastThreshold=12, edgeThreshold=18)
    earlier, a = extractor.detectAndCompute(gray_reference, None)
    later, b = extractor.detectAndCompute(gray_submitted, None)
    if a is None or b is None or min(len(a), len(b)) < MIN_MATCHES:
        return np.empty((0, 2))
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def distinctive(left, right):
        return {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(left, right, k=2)
                if len(pair) == 2 and pair[0].distance < .74 * pair[1].distance}

    forward = distinctive(b, a)
    reverse = distinctive(a, b)
    return np.array([np.subtract(earlier[i].pt, later[j].pt) for j, i in forward.items() if reverse.get(i) == j])


def _overlap(reference_shape, submitted_shape, dx, dy):
    rh, rw = reference_shape
    sh, sw = submitted_shape
    x, y = max(0, -dx), max(0, -dy)
    w, h = min(sw, rw - dx) - x, min(sh, rh - dy) - y
    if min(w, h) < MIN_SIDE or w * h < MIN_OVERLAP * max(rw * rh, sw * sh):
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _fraction(left, center, right):
    curvature = left - 2 * center + right
    return (left - right) / (2 * curvature) if curvature < -1e-6 else float("inf")


def _tile(reference, submitted, x, y, side, dx, dy):
    margin = TILE_SEARCH
    rx, ry = x + dx, y + dy
    target = submitted[y:y + side, x:x + side]
    window = reference[ry - margin:ry + side + margin, rx - margin:rx + side + margin]
    if window.shape != (side + 2 * margin, side + 2 * margin) or target.shape != (side, side) or target.std() < 5:
        return None
    scores = cv2.matchTemplate(window, target, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (px, py) = cv2.minMaxLoc(scores)
    if peak < MIN_TILE_SCORE or not (0 < px < 2 * margin and 0 < py < 2 * margin):
        return None
    # A repeated or flat peak cannot establish which native pixel is paired.
    competitors = scores.copy()
    competitors[py - 1:py + 2, px - 1:px + 2] = -1
    separation = float(peak - competitors.max())
    fractional = (float(_fraction(*map(float, scores[py, px - 1:px + 2]))),
                  float(_fraction(*map(float, scores[py - 1:py + 2, px]))))
    if separation < MIN_TILE_SEPARATION or max(map(abs, fractional)) > MAX_FRACTIONAL_OFFSET:
        return None
    return {"x": x, "y": y, "size": side, "dx": dx + px - margin, "dy": dy + py - margin,
            "correlation": round(float(peak), 6), "next_peak_difference": round(separation, 6),
            "fractional_offset": [round(value, 6) for value in fractional]}


def _tiles(reference, submitted, bounds, dx, dy):
    x, y, w, h = (bounds[key] for key in ("x", "y", "w", "h"))
    side = max(MIN_TILE_SIDE, min(96, w // 5, h // 5))
    ys = [y + round(h * f - side / 2) for f in (.20, .50, .80)]
    xs = [x + round(w * f - side / 2) for f in (.20, .50, .80)]
    return [(row, col, tile) for row, top in enumerate(ys) for col, left in enumerate(xs)
            if (tile := _tile(reference, submitted, left, top, side, dx, dy)) is not None]


def measure(original, submitted):
    """Return an independently corroborated reference-minus-submitted integer offset."""
    ref = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    sub = cv2.cvtColor(submitted, cv2.COLOR_RGB2GRAY)
    offsets = _matches(ref, sub)
    if len(offsets) < MIN_MATCHES:
        return None
    estimate = np.median(offsets, axis=0)
    dx, dy = (int(value) for value in np.rint(estimate))
    nearby = np.count_nonzero(np.max(np.abs(offsets - estimate), axis=1) <= 2)
    if nearby < max(MIN_MATCHES, MIN_MATCH_SHARE * len(offsets)):
        return None
    bounds = _overlap(ref.shape, sub.shape, dx, dy)
    if bounds is None:
        return None
    measured = _tiles(ref, sub, bounds, dx, dy)
    if len(measured) < MIN_TILES:
        return None
    # Any other distinctive displacement indicates scale, rotation or uncertain alignment.
    if len({(t["dx"], t["dy"]) for _, _, t in measured}) != 1:
        return None
    if len({row for row, _, _ in measured}) != 3 or len({col for _, col, _ in measured}) != 3:
        return None
    dx, dy = measured[0][2]["dx"], measured[0][2]["dy"]
    bounds = _overlap(ref.shape, sub.shape, dx, dy)
    if bounds is None:
        return None
    return {"method": "corroborated_integer_translation", "reference_minus_submitted": {"x": dx, "y": dy},
            "submitted_overlap": bounds, "reference_overlap": {**bounds, "x": bounds["x"] + dx, "y": bounds["y"] + dy},
            "matching_features": int(nearby), "tested_tiles": 9, "supporting_tiles": [t for _, _, t in measured]}


def _slice(region):
    x, y, w, h = (region[key] for key in ("x", "y", "w", "h"))
    return np.s_[y:y + h, x:x + w]


def compare(original: np.ndarray, submitted: np.ndarray, carrier: Image.Image) -> tuple[dict, np.ndarray | None]:
    """Compare only the common pixels, retaining both independent source coordinates."""
    if min(*original.shape[:2], *submitted.shape[:2]) < MIN_SIDE:
        return {"status": "integer_alignment_unresolved", "regions": []}, None
    alignment = measure(original, submitted)
    if alignment is None:
        return {"status": "integer_alignment_unresolved", "regions": []}, None
    submitted_bounds = alignment["submitted_overlap"]
    submitted_view = submitted[_slice(submitted_bounds)]
    if carrier.format == "JPEG" and submitted_view.shape != submitted.shape:
        return {"status": "partial_jpeg_delivery_unassessed", "alignment": alignment, "regions": []}, None
    earlier = original[_slice(alignment["reference_overlap"])]
    # Plausible earlier JPEG passes must start on the earlier image's original grid.
    result, mask = reference_compare.compare_delivered(earlier, submitted_view, carrier)
    if "estimated_native_shift" in result:
        result["overlap_residual_shift"] = result.pop("estimated_native_shift")
    if mask is not None and mask.any():
        support = np.zeros(mask.shape, np.uint8)
        qualities = []
        for quality in reference_compare.PRIOR_JPEG_QUALITIES:
            prior = reference_compare._prior_jpeg(original, quality)[_slice(alignment["reference_overlap"])]
            _, alternate = reference_compare.compare_delivered(prior, submitted_view, carrier)
            if alternate is not None:
                qualities.append(quality)
                support += alternate
        if len(qualities) < 2:
            return {**result, "status": "uncertain_delivery", "alignment": alignment, "regions": []}, None
        regions, mask, omitted = reference_compare._regions(mask & (support >= 2), max(48, round(mask.size * .0015)))
        result = {**result, "regions": regions, "additional_regions": omitted, "prior_jpeg_qualities_compared": qualities}
    if mask is not None:
        full_mask = np.zeros(submitted.shape[:2], bool)
        full_mask[_slice(submitted_bounds)] = mask
        mask = full_mask
    for region in result["regions"]:
        region["x"] += submitted_bounds["x"]
        region["y"] += submitted_bounds["y"]
        offset = alignment["reference_minus_submitted"]
        region["reference_region"] = {"x": region["x"] + offset["x"], "y": region["y"] + offset["y"],
                                      "w": region["w"], "h": region["h"]}
    return {**result, "alignment": alignment}, mask
