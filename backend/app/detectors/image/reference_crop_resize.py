"""Corroborate a whole-pixel earlier-image crop before proportional resize comparison."""
from __future__ import annotations

from itertools import product

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import reference_alignment, reference_resize

MIN_FEATURES = 20
FEATURE_RATIO = .74
MIN_INLIER_SHARE = .65
MIN_FEATURE_SPAN = .55
MAX_SHEAR = .004
MIN_AREA = .55
SEARCH_RADIUS = 2
MIN_CORRELATION = .88
MIN_MARGIN = .00006
MAX_VALIDATION_FRACTION = .14


def _rough_bounds(reference, submitted):
    extractor = cv2.SIFT_create(nfeatures=1000, contrastThreshold=.018)
    ref, ref_features = extractor.detectAndCompute(reference, None)
    sub, sub_features = extractor.detectAndCompute(submitted, None)
    if ref_features is None or sub_features is None or min(len(ref), len(sub)) < MIN_FEATURES:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def distinctive(first, second):
        return {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(first, second, k=2)
                if len(pair) == 2 and pair[0].distance < FEATURE_RATIO * pair[1].distance}

    forward, reverse = distinctive(sub_features, ref_features), distinctive(ref_features, sub_features)
    matches = [(index, partner) for index, partner in forward.items() if reverse.get(partner) == index]
    if len(matches) < MIN_FEATURES:
        return None
    points = np.array([sub[index].pt for index, _ in matches], dtype=np.float32)
    target = np.array([ref[partner].pt for _, partner in matches], dtype=np.float32)
    affine, inside = cv2.estimateAffine2D(points, target, ransacReprojThreshold=1.6, maxIters=1200, confidence=.995)
    if affine is None or inside is None:
        return None
    valid = inside.ravel() > 0
    height, width = submitted.shape
    if (valid.sum() < max(MIN_FEATURES, MIN_INLIER_SHARE * len(matches))
            or np.any(np.ptp(points[valid], axis=0) < MIN_FEATURE_SPAN * np.array((width, height)))):
        return None
    sx, sy = affine[0, 0], affine[1, 1]
    if min(sx, sy) <= 0 or max(abs(affine[0, 1]), abs(affine[1, 0])) > MAX_SHEAR * min(sx, sy):
        return None
    left, top = affine[:, 2] + .5 * (1 - np.array((sx, sy)))
    return np.rint((left, top, left + sx * width, top + sy * height)).astype(int), int(valid.sum())


def _feasible(rect, reference_shape, submitted_shape):
    left, top, right, bottom = rect
    rh, rw = reference_shape
    sh, sw = submitted_shape
    w, h = right - left, bottom - top
    return (0 <= left < right <= rw and 0 <= top < bottom <= rh and (left, top, right, bottom) != (0, 0, rw, rh)
            and w * h >= MIN_AREA * rw * rh and min(w, h) >= reference_alignment.MIN_SIDE
            and reference_resize.MIN_SCALE <= min(sw / w, sh / h) <= max(sw / w, sh / h) <= reference_resize.MAX_SCALE
            and abs((sw / w) / (sh / h) - 1) <= reference_resize.MAX_ASPECT_DRIFT)


def _scores(reference, submitted, guess):
    sh, sw = submitted.shape
    extent = guess[2:] - guess[:2]
    sigma = max(.7, .9 * float(np.mean(extent / (sw, sh))))
    ref = cv2.GaussianBlur(reference.astype(np.float32), (0, 0), sigma)
    sub = cv2.GaussianBlur(submitted.astype(np.float32), (0, 0), .9)
    gy, gx = np.indices((23, 23), dtype=np.int32) * 2 - 22
    centers = [(round(fx * sw), round(fy * sh)) for fy, fx in product((.2, .5, .8), repeat=2)]
    ys = np.array([(y + gy).ravel() for _, y in centers])
    xs = np.array([(x + gx).ravel() for x, _ in centers])
    sampled = sub[ys, xs]
    centered = sampled - sampled.mean(axis=1, keepdims=True)
    lengths = np.linalg.norm(centered, axis=1)
    usable = lengths >= 5 * np.sqrt(centered.shape[1])
    if np.count_nonzero(usable) < 6 or np.count_nonzero(usable[::2]) < 3 or np.count_nonzero(usable[1::2]) < 3:
        return []
    unit = centered / np.maximum(lengths[:, None], 1.)
    xs = ((xs + .5) / sw).astype(np.float32)
    ys = ((ys + .5) / sh).astype(np.float32)
    offsets = range(-SEARCH_RADIUS, SEARCH_RADIUS + 1)
    scores = []
    for delta in product(offsets, repeat=4):
        candidate = tuple(map(int, guess + delta))
        if not _feasible(candidate, reference.shape, submitted.shape):
            continue
        left, top, right, bottom = candidate
        measured = cv2.remap(ref, xs * (right - left) + left - .5, ys * (bottom - top) + top - .5,
                             cv2.INTER_LINEAR)
        centered_ref = measured - measured.mean(axis=1, keepdims=True)
        correlation = np.sum(centered_ref * unit, axis=1) / np.maximum(np.linalg.norm(centered_ref, axis=1), 1.)
        overall = float(np.median(correlation[usable]))
        folds = [float(np.median(correlation[start::2][usable[start::2]])) for start in (0, 1)]
        scores.append((overall, candidate, folds, correlation, usable))
    return scores


def measure(reference, submitted):
    """Find an exact earlier crop using a bounded search and independent tile support."""
    if min(*reference.shape[:2], *submitted.shape[:2]) < reference_alignment.MIN_SIDE:
        return None
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    gray_sub = cv2.cvtColor(submitted, cv2.COLOR_RGB2GRAY)
    rough = _rough_bounds(gray_ref, gray_sub)
    if rough is None:
        return None
    guess, features = rough
    scores = _scores(gray_ref, gray_sub, guess)
    if len(scores) < 2:
        return None
    ranked = sorted(scores, key=lambda value: value[0], reverse=True)
    score, rect, folds, correlations, usable = ranked[0]
    margin = score - ranked[1][0]
    if (min(score, *folds) < MIN_CORRELATION or margin < MIN_MARGIN
            or any(max(scores, key=lambda value: value[2][i])[1] != rect for i in range(2))):
        return None
    supported = np.argwhere((usable & (correlations >= MIN_CORRELATION)).reshape(3, 3))
    if len(supported) < 6 or len(set(supported[:, 0])) != 3 or len(set(supported[:, 1])) != 3:
        return None
    left, top, right, bottom = rect
    return {"x": left, "y": top, "w": right - left, "h": bottom - top, "matching_features": features,
            "median_correlation": round(score, 8), "next_crop_difference": round(margin, 8),
            "supporting_tiles": [{"row": int(y), "column": int(x), "correlation": round(float(correlations[3 * y + x]), 8)}
                                 for y, x in supported]}


def compare(reference: np.ndarray, submitted: np.ndarray, carrier: Image.Image) -> tuple[dict, np.ndarray | None]:
    bounds = measure(reference, submitted)
    if bounds is None:
        return {"status": "crop_resize_unresolved", "regions": []}, None
    x, y, w, h = (bounds[key] for key in ("x", "y", "w", "h"))
    result, mask = reference_resize.compare(reference[y:y+h, x:x+w], submitted, carrier)
    tiles = (result.get("framing_check") or {}).get("supporting_tiles", [])
    if result["status"] != "comparable" or not tiles:
        return {"status": "crop_resize_unresolved", "regions": []}, None
    fractions = np.asarray([item["fractional_offset"] for item in tiles])
    if np.max(np.median(np.abs(fractions), axis=0)) > MAX_VALIDATION_FRACTION:
        return {"status": "crop_resize_unresolved", "regions": []}, None
    for region in result["regions"]:
        region["reference_region"]["x"] += x
        region["reference_region"]["y"] += y
    return {**result, "reference_crop_bounds": bounds}, mask
