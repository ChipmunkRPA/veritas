"""Propose small rotated or resized copies for the existing dense mask check."""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from app.detectors.image import copy_affine, copy_move

MAX_FEATURES = 3200
MAX_MATCHES = 500
MAX_FITS = 12
MIN_MATCHES = 4
MAX_COLOR_ERROR = 8.5
MIN_DETAIL_CORRELATION = .93


def _features(gray):
    points, descriptors = cv2.SIFT_create(nfeatures=MAX_FEATURES, contrastThreshold=.009).detectAndCompute(gray, None)
    return np.array([p.pt for p in points], np.float64).reshape(-1, 2), descriptors


def _matches(points, descriptors):
    if descriptors is None or len(points) < 3:
        return np.empty((0, 2), np.intp)
    neighbours = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors, descriptors, k=min(12, len(points)))
    nearest = {}
    for index, row in enumerate(neighbours):
        remote = [hit for hit in row if np.linalg.norm(points[index] - points[hit.trainIdx]) >= 48]
        if len(remote) < 2:
            continue
        best = remote[0]
        # Several descriptor angles can represent one location. Compare to a different location.
        rival = next((hit for hit in remote[1:] if np.linalg.norm(points[best.trainIdx] - points[hit.trainIdx]) >= 5), None)
        if rival is not None and best.distance < min(185, .78 * rival.distance):
            nearest[index] = best.trainIdx
    mutual = sorted((first, second) for first, second in nearest.items()
                    if first < second and nearest.get(second) == first)
    if len(mutual) > MAX_MATCHES:
        raise copy_move.CandidateBudgetExceeded("Too many close matches in the small-copy search")
    return np.asarray(mutual, dtype=np.intp).reshape(-1, 2)


def _groups(points, matches):
    # Link neighbours in both regions. Reversed edges make the search independent of keypoint order.
    directed = np.concatenate((matches, matches[:, ::-1]))
    locations = points[directed].reshape(len(directed), 4)
    edges = cKDTree(locations).query_pairs(32, output_type="ndarray")
    graph = coo_matrix((np.ones(len(edges)), edges.T), shape=(len(locations), len(locations)))
    count, labels = connected_components(graph, directed=False)
    groups = [np.flatnonzero(labels == label) for label in range(count)]
    seen = set()
    for group in sorted(groups, key=lambda nodes: (-len(nodes), int(nodes[0]))):
        if len(group) < MIN_MATCHES:
            break
        identity = frozenset(group % len(matches))
        if identity not in seen:
            seen.add(identity)
            yield locations[group].reshape(-1, 2, 2)


def _fit(group, shape):
    first, second = (np.ascontiguousarray(side, dtype=np.float32) for side in group.transpose(1, 0, 2))
    matrix, inliers = cv2.estimateAffinePartial2D(first, second, ransacReprojThreshold=1.5, maxIters=1000)
    if matrix is None:
        return None
    first = first[inliers.ravel() != 0]
    second = second[inliers.ravel() != 0]
    # Count positions on each side, not alternate descriptor orientations.
    if min(len(np.unique(np.rint(side / 3), axis=0)) for side in (first, second)) < MIN_MATCHES:
        return None
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    angle = abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    if not .68 <= scale <= 1.48 or not 0 <= angle <= 45 or (angle < 5 and abs(math.log(scale)) < .07):
        return None
    if tuple(first.mean(axis=0)[::-1]) > tuple(second.mean(axis=0)[::-1]):
        matrix = cv2.invertAffineTransform(matrix)
        first = second
    if max(np.ptp(first, axis=0)) < 12:
        return None
    height, width = shape
    low = np.maximum(0, np.floor(first.min(axis=0) - 7)).astype(int)
    high = np.minimum([width, height], np.ceil(first.max(axis=0) + 8)).astype(int)
    if np.any(high - low > 132):
        return None
    return np.round(matrix, 6), (int(low[0]), int(low[1]), int(high[0]), int(high[1]))


def matches_in_image(rgb):
    """Yield matrices, source seeds and dense masks in group-support order."""
    native = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    points, descriptors = _features(native)
    matches = _matches(points, descriptors)
    if len(matches) < MIN_MATCHES:
        return
    gray = copy_affine.luma(rgb)
    for index, group in enumerate(_groups(points, matches)):
        if index >= MAX_FITS:
            raise copy_move.CandidateBudgetExceeded("Too many groups in the small-copy search")
        proposal = _fit(group, native.shape)
        if proposal is not None and (masks := copy_affine.dense_pair(gray, *proposal)) is not None:
            yield (*proposal, masks)


def alignment_detail(rgb, matrix, mask):
    """Measure agreement in the interior of the matched source area."""
    x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
    # Keep a halo for the filters before taking any interior samples.
    left, top = max(0, x - 6), max(0, y - 6)
    right, bottom = min(rgb.shape[1], x + width + 6), min(rgb.shape[0], y + height + 6)
    local = matrix.copy()
    local[:, 2] += local[:, :2] @ [left, top]
    size = (right - left, bottom - top)
    first = rgb[top:bottom, left:right].astype(np.float32)
    second = cv2.warpAffine(rgb.astype(np.float32), local, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    interior = cv2.erode(mask[top:bottom, left:right].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if int(interior.sum()) < 160:
        return None

    def band(pixels):
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        return (cv2.GaussianBlur(gray, (0, 0), .65) - cv2.GaussianBlur(gray, (0, 0), 2.0))[interior].astype(np.float64)

    color_error = float(np.mean(np.abs(first - second)[interior]))
    a, b = band(first), band(second)
    a -= a.mean()
    b -= b.mean()
    magnitude = np.linalg.norm(a) * np.linalg.norm(b)
    return color_error, float(a @ b / magnitude) if magnitude > 1e-7 else 0.


def confirmed(rgb):
    """Return the first candidate with independently measured color and fine-detail agreement."""
    for matrix, seed, masks in matches_in_image(rgb):
        detail = alignment_detail(rgb, matrix, masks[0])
        if detail and detail[0] <= MAX_COLOR_ERROR and detail[1] >= MIN_DETAIL_CORRELATION:
            return matrix, seed, masks
    return None
