"""Bounded same-image reflection similarity in oriented native pixels.

The method finds spatially consistent mirrored patches and verifies dense local
correlation. A pair is undirected and can occur in naturally symmetric content.
"""

from __future__ import annotations

from types import MappingProxyType

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from app.detectors.image import copy_move
from app.evidence.image_copy_similarity_binding import PAIR_LIMIT, supported_size

PLAN = MappingProxyType({
    "tile_px": 128,
    "corners_per_tile": 40,
    "nms_side": 7,
    "corner_floor": 0.00001,
    "descriptor_radius": 8,
    "descriptor_step": 2,
    "min_patch_std": 7.0,
    "blur_sigma": 0.75,
    "nearest_neighbors": 6,
    "ratio": 0.80,
    "max_normalized_distance": 0.43,
    "min_separation_px": 32,
    "match_cap": 256,
    "offset_tolerance_px": 2.0,
    "dedup_radius_px": 3.0,
    "consensus_link_radius_px": 32.0,
    "min_consensus_pairs": 8,
    "refine_halo_px": 48,
    "refine_corr_side": 9,
    "refine_corr": 0.98,
    "refine_std": 5.0,
    "refine_close_side": 7,
    "refine_seed_overlap": 4,
    "refine_min_area": 256,
    "refine_exact_row_column_support": 0.95,
})
LOW_TEXTURE_PLAN = MappingProxyType({
    **PLAN,
    "corner_floor": 0.000002,
    "corners_per_tile": 60,
    "min_patch_std": 3.5,
    "min_consensus_pairs": 6,
    "refine_std": 2.75,
    "refine_corr": 0.985,
})
MISSING = "unassessed_match_budget"


def _components(pairs, plan):
    minimum = plan["min_consensus_pairs"]
    if len(pairs) < minimum:
        return []
    first, second = pairs[:, 0], pairs[:, 1]
    parameters = np.column_stack((first[:, 0] + second[:, 0], second[:, 1] - first[:, 1]))
    offsets = np.abs(parameters[:, None] - parameters)
    link = plan["consensus_link_radius_px"]
    tolerance = plan["offset_tolerance_px"]
    near = (
        (np.linalg.norm(first[:, None] - first, axis=2) <= link)
        & (np.linalg.norm(second[:, None] - second, axis=2) <= link)
        & (offsets.max(axis=2) <= tolerance)
    )
    count, membership = connected_components(csr_matrix(near), directed=False)
    found = []
    for group in range(count):
        members = np.flatnonzero(membership == group)
        if len(members) < minimum:
            continue
        center = np.median(parameters[members], axis=0)
        stable = members[np.max(np.abs(parameters[members] - center), axis=1) <= tolerance]
        if len(stable) >= minimum:
            found.append(stable)
    return sorted(found, key=lambda group: (-len(group), int(group[0])))


def keypoint_descriptors(gray, p):
    radius, tile = p["descriptor_radius"], p["tile_px"]
    response = cv2.cornerMinEigenVal(gray, blockSize=5, ksize=3)
    maxima = (response == cv2.dilate(response, np.ones((p["nms_side"],) * 2, dtype=np.uint8))) & (
        response >= p["corner_floor"]
    )
    maxima[:radius] = maxima[-radius:] = False
    maxima[:, :radius] = maxima[:, -radius:] = False
    height, width = gray.shape
    selected = []
    for top in range(0, height, tile):
        for left in range(0, width, tile):
            y, x = np.nonzero(maxima[top : top + tile, left : left + tile])
            y, x = y + top, x + left
            chosen = np.lexsort((x, y, -response[y, x]))[: p["corners_per_tile"]]
            selected.extend(zip(x[chosen], y[chosen], strict=True))
    positions = np.asarray(selected, dtype=np.int32).reshape(-1, 2)
    gx, gy = positions.T
    offset = np.arange(-radius, radius + 1, p["descriptor_step"])
    blurred = cv2.GaussianBlur(gray, (5, 5), p["blur_sigma"])
    values = blurred[
        gy[:, None, None] + offset[None, :, None], gx[:, None, None] + offset[None, None, :]
    ].astype(np.float32)
    values -= values.mean(axis=(1, 2), keepdims=True)
    lengths = np.linalg.norm(values.reshape(len(values), len(offset) ** 2), axis=1)
    keep = lengths >= p["min_patch_std"] * len(offset)
    return positions[keep], np.ascontiguousarray(values[keep] / lengths[keep, None, None])


def proposal_matches(positions, values, p):
    n = len(values)
    if n < p["min_consensus_pairs"]:
        return np.empty((0, 2, 2))
    flat = np.ascontiguousarray(values.reshape(n, -1))
    mirrored = np.ascontiguousarray(values[:, :, ::-1].reshape(n, -1))
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(flat, mirrored, k=min(n, p["nearest_neighbors"]))
    accepted = {}
    for row in matches:
        good = [
            m
            for m in row
            if np.linalg.norm(positions[m.queryIdx] - positions[m.trainIdx]) >= p["min_separation_px"]
        ]
        if len(good) < 2:
            continue
        first, second = good[:2]
        if first.distance <= p["max_normalized_distance"] and first.distance < p["ratio"] * max(
            1e-6, second.distance
        ):
            accepted[first.queryIdx] = first
    pairs = []
    for hit in sorted(accepted.values(), key=lambda m: (m.distance, m.queryIdx, m.trainIdx)):
        a, b = sorted((hit.queryIdx, hit.trainIdx), key=lambda index: tuple(positions[index]))
        if hit.queryIdx != a or b not in accepted or accepted[b].trainIdx != a:
            continue
        left, right = positions[a], positions[b]
        if any(
            np.linalg.norm(left - x) < p["dedup_radius_px"]
            and np.linalg.norm(right - y) < p["dedup_radius_px"]
            for x, y in pairs
        ):
            continue
        pairs.append((left, right))
        if len(pairs) > p["match_cap"]:
            return None
    return np.asarray(pairs, dtype=np.float64).reshape(-1, 2, 2)


def local_corr(left, right, plan):
    n = plan["refine_corr_side"]

    def average(pixels):
        return cv2.boxFilter(pixels, -1, (n, n), normalize=True)

    mu_l, mu_r = average(left), average(right)
    var_l = np.maximum(0, average(left * left) - mu_l * mu_l)
    var_r = np.maximum(0, average(right * right) - mu_r * mu_r)
    cov = average(left * right) - mu_l * mu_r
    corr = cov / np.sqrt(np.maximum(1, var_l * var_r))
    min_var = plan["refine_std"] ** 2
    valid = (corr >= plan["refine_corr"]) & (var_l >= min_var) & (var_r >= min_var)
    pad = n // 2
    valid[:pad] = valid[-pad:] = False
    valid[:, :pad] = valid[:, -pad:] = False
    return valid


def tighten_equal(bounds, exact, plan):
    """Use matching RGB in a locally supported rectangle when it delineates the edge."""
    left, top, right, bottom = bounds
    threshold = plan["refine_exact_row_column_support"]
    patch = exact[top:bottom, left:right]
    rows = np.flatnonzero(patch.mean(axis=1) >= threshold)
    cols = np.flatnonzero(patch.mean(axis=0) >= threshold)
    if (
        len(rows) < 12
        or len(cols) < 12
        or len(rows) < patch.shape[0] * 0.7
        or len(cols) < patch.shape[1] * 0.7
    ):
        return bounds
    if (rows[-1] - rows[0] + 1) > len(rows) + 2 or (cols[-1] - cols[0] + 1) > len(cols) + 2:
        return bounds
    return left + cols[0], top + rows[0], left + cols[-1] + 1, top + rows[-1] + 1


def refine(rgb, gray, inliers, p):
    height, width = gray.shape
    a, b = inliers[:, 0], inliers[:, 1]
    intercept = int(np.rint(np.median(a[:, 0] + b[:, 0])))
    dy = int(np.rint(np.median(b[:, 1] - a[:, 1])))
    pad = p["refine_halo_px"]
    left = max(0, intercept - (width - 1), int(a[:, 0].min()) - pad)
    right = min(width, intercept + 1, int(a[:, 0].max()) + pad + 1)
    top = max(0, -dy, int(a[:, 1].min()) - pad)
    bottom = min(height, height - dy, int(a[:, 1].max()) + pad + 1)
    if right - left <= p["refine_corr_side"] or bottom - top <= p["refine_corr_side"]:
        return None
    other_left, other_right = intercept - (right - 1), intercept - left + 1
    r0 = np.s_[top:bottom, left:right]
    r1 = np.s_[top + dy : bottom + dy, other_left:other_right]
    aligned = local_corr(gray[r0].astype(np.float32), gray[r1][:, ::-1].astype(np.float32), p)
    kernel = np.ones((p["refine_close_side"],) * 2, dtype=np.uint8)
    closed = cv2.morphologyEx(aligned.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    seeds = a.astype(int) - [left, top]
    inside = (
        (seeds[:, 0] >= 0) & (seeds[:, 0] < right - left) & (seeds[:, 1] >= 0) & (seeds[:, 1] < bottom - top)
    )
    seeds = seeds[inside]
    supports = np.bincount(labels[seeds[:, 1], seeds[:, 0]], minlength=n)
    candidates = [
        i
        for i in range(1, n)
        if supports[i] >= p["refine_seed_overlap"] and stats[i, -1] >= p["refine_min_area"]
    ]
    if not candidates:
        return None
    i = max(candidates, key=lambda j: (supports[j], stats[j, -1]))
    x, y, w, h, _ = stats[i]
    half = p["refine_corr_side"] // 2
    bounds = (
        max(0, x - half),
        max(0, y - half),
        min(right - left, x + w + half),
        min(bottom - top, y + h + half),
    )
    equal = np.all(rgb[r0] == rgb[r1][:, ::-1], axis=2)
    x0, y0, x1, y1 = tighten_equal(bounds, equal, p)
    a_box = tuple(int(v) for v in (left + x0, top + y0, x1 - x0, y1 - y0))
    b_box = tuple(int(v) for v in (intercept - (left + x1 - 1), top + y0 + dy, x1 - x0, y1 - y0))
    return (a_box, b_box) if copy_move.iou(a_box, b_box) == 0 else None


def _offset(pair):
    first, second = pair
    return first[0] + second[0] + first[2] - 1, second[1] - first[1]


def _merge_strips(left, right):
    if _offset(left) != _offset(right) or copy_move.iou(left[0], right[0]) <= 0:
        return None
    ax, ay, aw, ah = left[0]
    bx, by, bw, bh = right[0]
    if (ax, aw) != (bx, bw) and (ay, ah) != (by, bh):
        return None
    x, y = min(ax, bx), min(ay, by)
    w, h = max(ax + aw, bx + bw) - x, max(ay + ah, by + bh) - y
    intercept, dy = _offset(left)
    first, second = (x, y, w, h), (intercept - (x + w - 1), y + dy, w, h)
    return (first, second) if copy_move.iou(first, second) == 0 else None


def _consolidate(pairs):
    retained = []
    for candidate in pairs:
        for index, earlier in enumerate(retained):
            merged = _merge_strips(earlier, candidate)
            if merged is not None:
                retained[index] = merged
                break
        else:
            retained.append(candidate)
    return retained


def _search(rgb, gray, plan):
    positions, descriptors = keypoint_descriptors(gray, plan)
    matches = proposal_matches(positions, descriptors, plan)
    if matches is None:
        return MISSING, []
    pairs = _consolidate(
        [pair for group in _components(matches, plan) if (pair := refine(rgb, gray, matches[group], plan)) is not None]
    )
    if len(pairs) > PAIR_LIMIT:
        return MISSING, []
    return ("supported_pair" if pairs else "no_pair"), pairs


def _corroborates(rgb, pair):
    """Require agreement across the entire proposed lower-texture pair."""
    first, second = pair
    if min(first[2:]) < 24:
        return False

    def pixels(box):
        x, y, width, height = box
        return rgb[y:y + height, x:x + width]

    first, second = pixels(first), pixels(second)[:, ::-1]
    if np.mean(np.all(first == second, axis=2)) >= 0.85:
        return True

    def interior_luma(crop):
        luma = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return cv2.GaussianBlur(luma, (5, 5), 1.0)[3:-3, 3:-3].ravel()

    a, b = interior_luma(first), interior_luma(second)
    return min(np.std(a), np.std(b)) >= 3 and np.corrcoef(a, b)[0, 1] >= 0.988


def analyse(rgb: np.ndarray) -> tuple[str, list[tuple[copy_move.Box, copy_move.Box]]]:
    if not supported_size(rgb.shape[1], rgb.shape[0]):
        return "unsupported_geometry", []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    status, pairs = _search(rgb, gray, PLAN)
    if status != "no_pair":
        return status, pairs

    status, pairs = _search(rgb, gray, LOW_TEXTURE_PLAN)
    if status != "supported_pair":
        return status, pairs
    verified = [pair for pair in pairs if _corroborates(rgb, pair)]
    return ("supported_pair" if verified else "no_pair"), verified
