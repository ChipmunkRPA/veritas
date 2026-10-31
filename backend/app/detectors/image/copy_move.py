"""SIFT keypoint copy-move detection (I-12; IF §4.2, §6.3): g2NN matching, grouping, RANSAC affine fits, repetitive structure.

- **Input.** Native pixels (EXIF-oriented) to luma, reduced with area interpolation to a long side of at most 2048 px (IF §4.2
  default pass). Every reported coordinate is in native px.
- **Keypoints.** OpenCV SIFT, at most 20,000 by response, sorted by position so a run is reproducible.
- **g2NN** (Amerini et al. 2011, the multiple-copy reading): the 10 nearest other descriptors d_1 ≤ … ≤ d_10 (L2); the
  neighbours up to the first ratio d_j / d_(j+1) < 0.5 are matches. A keypoint with no such break whose 10th neighbour is still
  similar has many look-alikes ("crowded"): repeated structure rather than one copy. Pairs closer than 16 px are dropped.
- **Grouping.** Each pair, in both orientations, is a point (x_a, y_a, x_b, y_b); points within 5 % of the long side of each
  other are linked, and connected components are candidate clones. The paper clusters keypoint positions hierarchically; linking
  pairs keeps the two sides of one clone together.
- **Fit.** `cv2.estimateAffine2D` with RANSAC (3 px) per component. Accepted: at least 12 inliers, scales 0.25–4, |shear| ≤ 0.5,
  the two areas at least 32 px apart and overlapping by IoU < 0.5, and the inlier keypoints at least 25 % of all keypoints inside
  the two areas (a copied patch brings its keypoints along; similar structure among other detail does not). Duplicates of one
  clone keep the fit with more inliers.
- **Repetitive structure.** A pair is marked when ≥ 25 % of its inlier keypoints have two or more partners, or when one of its
  areas takes part in three or more accepted pairs; the image is marked when ≥ 5 % of keypoints are crowded. Tiles, windows,
  text, crowds and symmetric objects form similar pairs without any editing.
- **Direction.** Either area can be the copy: area a is the upper (then left) one, and the affine maps a onto b.

The thresholds were set on synthetic fixtures (`tests/test_copy_move.py`) and on the first 60 vb R1/R2 camera originals by
`item_id`, which are also in the false-positive population (`scripts/v2/copymove_fpr.py`). They are uncalibrated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MAX_SIDE = 2048
MIN_INPUT_SIDE = 64
MAX_KEYPOINTS = 20_000
NEIGHBOURS = 10
RATIO = 0.5
CROWDED_DISTANCE = 160.0  # OpenCV SIFT descriptors have L2 norm ≈ 512; unrelated patches sit around 300–450
MIN_PAIR_DISTANCE = 16.0
LINK_FRACTION = 0.05
MIN_INLIERS = 12
MIN_COVERAGE = 0.25
RANSAC_PX = 3.0
SCALE_RANGE = (0.25, 4.0)
MAX_SHEAR = 0.5
MIN_SHIFT = 32.0
MAX_SIDE_IOU = 0.5
DUPLICATE_IOU = 0.5
LATTICE_IOU = 0.3
MULTI_SHARE = 0.25
LATTICE_PAIRS = 3
CROWDED_SHARE = 0.05
MAX_PAIRS = 10

Box = tuple[float, float, float, float]  # x, y, w, h


@dataclass(frozen=True)
class Affine:
    """Area a → area b in native px: x_b = A x_a + t."""

    matrix: tuple[tuple[float, float, float], tuple[float, float, float]]
    rotation_deg: float
    scale_x: float
    scale_y: float
    shear: float
    reflected: bool

    @property
    def translation(self) -> tuple[float, float]:
        return self.matrix[0][2], self.matrix[1][2]


@dataclass(frozen=True)
class ClonePair:
    box_a: Box
    box_b: Box
    inliers: int
    candidates: int  # pairs in the component the fit ran on
    coverage: float  # inlier keypoints / keypoints inside the two areas
    affine: Affine
    multi_share: float
    lattice: int  # accepted pairs sharing an area with this one, itself included
    repetitive: bool


@dataclass(frozen=True)
class CopyMoveResult:
    width: int
    height: int
    analysed: tuple[int, int]  # width, height of the analysed luma image
    keypoints: int
    matches: int
    crowded_share: float
    components: int
    pairs: list[ClonePair]

    @property
    def repetitive_image(self) -> bool:
        return self.crowded_share >= CROWDED_SHARE

    @property
    def distinct(self) -> list[ClonePair]:
        """Pairs not marked repetitive, in an image not marked repetitive."""
        return [] if self.repetitive_image else [p for p in self.pairs if not p.repetitive]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def decompose(linear: np.ndarray) -> tuple[float, float, float, float, bool]:
    """A = R(θ) · [[s_x, k], [0, s_y]]: rotation in degrees, scales, shear k / s_y, and whether A mirrors."""
    a, b, c, d = float(linear[0, 0]), float(linear[0, 1]), float(linear[1, 0]), float(linear[1, 1])
    scale_x = math.hypot(a, c)
    theta = math.atan2(c, a)
    k = b * math.cos(theta) + d * math.sin(theta)
    scale_y = -b * math.sin(theta) + d * math.cos(theta)
    shear = k / scale_y if abs(scale_y) > 1e-9 else math.inf
    return math.degrees(theta), scale_x, abs(scale_y), shear, a * d - b * c < 0


def _luma(pixels: np.ndarray) -> tuple[np.ndarray, float, float]:
    gray = pixels if pixels.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(pixels[..., :3]), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    long_side = max(width, height)
    if long_side <= MAX_SIDE:
        return np.ascontiguousarray(gray), 1.0, 1.0
    new_w, new_h = max(1, round(width * MAX_SIDE / long_side)), max(1, round(height * MAX_SIDE / long_side))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA), new_w / width, new_h / height


def _keypoints(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions (n, 2), sizes (n,) and descriptors (n, 128), sorted by position, size and angle."""
    keypoints, descriptors = cv2.SIFT_create(nfeatures=MAX_KEYPOINTS).detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 2:
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 128), np.float32)
    table = np.array([(k.pt[1], k.pt[0], k.size, k.angle) for k in keypoints])
    order = np.lexsort(table.T[::-1])
    return table[order][:, [1, 0]], table[order][:, 2], np.ascontiguousarray(descriptors[order])


def g2nn(descriptors: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Undirected matches (i < j) and a per-keypoint crowded flag."""
    n = len(descriptors)
    k = min(NEIGHBOURS + 1, n)
    # The ratio needs two other descriptors, in addition to the self-match.
    if k < 3:
        return [], np.zeros(n, bool)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors, descriptors, k=k)
    width = k - 1
    index = np.full((n, width), -1, np.int64)
    dist = np.full((n, width), np.inf)
    for i, row in enumerate(knn):
        others = [m for m in row if m.trainIdx != i][:width]
        index[i, :len(others)] = [m.trainIdx for m in others]
        dist[i, :len(others)] = [m.distance for m in others]
    ratios = dist[:, :-1] / np.maximum(dist[:, 1:], 1e-6)
    breaks = ratios < RATIO
    has_break = breaks.any(axis=1)
    first = np.where(has_break, breaks.argmax(axis=1), -1)
    crowded = ~has_break & (width == NEIGHBOURS) & (dist[:, -1] < CROWDED_DISTANCE)
    matches = {(min(i, int(index[i, j])), max(i, int(index[i, j]))) for i in np.flatnonzero(has_break)
               for j in range(int(first[i]) + 1)}
    return sorted(matches), crowded


def _box(points: np.ndarray, pad: float, width: int, height: int) -> Box:
    x0, y0 = max(0.0, float(points[:, 0].min()) - pad), max(0.0, float(points[:, 1].min()) - pad)
    x1, y1 = min(float(width), float(points[:, 0].max()) + pad), min(float(height), float(points[:, 1].max()) + pad)
    return round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)


def _inside(points: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    return (points[:, 0] >= x) & (points[:, 0] <= x + w) & (points[:, 1] >= y) & (points[:, 1] <= y + h)


def _fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    matrix, mask = cv2.estimateAffine2D(src.astype(np.float32), dst.astype(np.float32), method=cv2.RANSAC,
                                        ransacReprojThreshold=RANSAC_PX, maxIters=2000, confidence=0.995, refineIters=10)
    if matrix is None or mask is None:
        return None
    return matrix, mask.ravel().astype(bool)


def analyse(pixels: np.ndarray) -> CopyMoveResult:
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    gray, sx, sy = _luma(pixels)
    positions, sizes, descriptors = _keypoints(gray)
    matches, crowded = g2nn(descriptors)
    pairs = [(i, j) for i, j in matches if np.hypot(*(positions[i] - positions[j])) >= MIN_PAIR_DISTANCE]
    partners = np.zeros(len(positions), np.int64)
    for i, j in pairs:
        partners[i] += 1
        partners[j] += 1
    crowded_share = float(crowded.mean()) if len(crowded) else 0.0
    analysed = (int(gray.shape[1]), int(gray.shape[0]))
    if len(pairs) < MIN_INLIERS:
        return CopyMoveResult(width, height, analysed, len(positions), len(pairs), crowded_share, 0, [])
    scale = np.array([sx, sy])
    native = positions / scale
    ends = np.array(pairs, np.int64)
    m = len(ends)
    oriented = np.vstack([ends, ends[:, ::-1]])
    points = np.hstack([positions[oriented[:, 0]], positions[oriented[:, 1]]])
    links = cKDTree(points).query_pairs(LINK_FRACTION * max(analysed), output_type="ndarray")
    graph = coo_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(2 * m, 2 * m))
    n_components, labels = connected_components(graph, directed=False)
    seen: set[frozenset[int]] = set()
    fits: list[ClonePair] = []
    for label in range(n_components):
        nodes = np.flatnonzero(labels == label)
        key = frozenset(int(v) for v in nodes % m)
        if len(key) < MIN_INLIERS or key in seen:
            continue
        seen.add(key)
        fitted = _fit(points[nodes, :2], points[nodes, 2:])
        if fitted is None or fitted[1].sum() < MIN_INLIERS:
            continue
        matrix, inlier = fitted
        _, scale_x, scale_y, shear, _ = decompose(matrix[:, :2])
        if not (SCALE_RANGE[0] <= scale_x <= SCALE_RANGE[1] and SCALE_RANGE[0] <= scale_y <= SCALE_RANGE[1]) or abs(shear) > MAX_SHEAR:
            continue
        chosen = oriented[nodes[inlier]]
        src, dst = positions[chosen[:, 0]], positions[chosen[:, 1]]
        if np.hypot(*(src.mean(axis=0) - dst.mean(axis=0))) < MIN_SHIFT:
            continue
        pad = float(np.median(sizes[chosen.ravel()])) / 2
        box_a, box_b = _box(native[chosen[:, 0]], pad / sx, width, height), _box(native[chosen[:, 1]], pad / sy, width, height)
        if iou(box_a, box_b) >= MAX_SIDE_IOU:
            continue
        keypoints = np.unique(chosen.ravel())
        coverage = len(keypoints) / max(1, int((_inside(native, box_a) | _inside(native, box_b)).sum()))
        if coverage < MIN_COVERAGE:
            continue
        linear = np.diag(1 / scale) @ matrix[:, :2] @ np.diag(scale)
        translation = matrix[:, 2] / scale
        if (box_b[1], box_b[0]) < (box_a[1], box_a[0]):
            box_a, box_b = box_b, box_a
            inverse = cv2.invertAffineTransform(np.hstack([linear, translation[:, None]]))
            linear, translation = inverse[:, :2], inverse[:, 2]
        rotation, scale_x, scale_y, shear, reflected = decompose(linear)
        affine = Affine(matrix=(tuple(round(float(v), 6) for v in (*linear[0], translation[0])),  # type: ignore[arg-type]
                                tuple(round(float(v), 6) for v in (*linear[1], translation[1]))),
                        rotation_deg=round(rotation, 2), scale_x=round(scale_x, 4), scale_y=round(scale_y, 4),
                        shear=round(shear, 4), reflected=reflected)
        fits.append(ClonePair(box_a=box_a, box_b=box_b, inliers=int(inlier.sum()), candidates=len(nodes),
                              coverage=round(coverage, 4), affine=affine,
                              multi_share=round(float((partners[keypoints] >= 2).mean()), 4), lattice=1, repetitive=False))
    fits.sort(key=lambda p: (-p.inliers, p.box_a[1], p.box_a[0], p.box_b[1], p.box_b[0]))
    unique: list[ClonePair] = []
    for p in fits:
        if not any(min(iou(p.box_a, q.box_a), iou(p.box_b, q.box_b)) >= DUPLICATE_IOU
                   or min(iou(p.box_a, q.box_b), iou(p.box_b, q.box_a)) >= DUPLICATE_IOU for q in unique):
            unique.append(p)
    out = []
    for p in unique:
        lattice = max(sum(any(iou(side, other) >= LATTICE_IOU for other in (q.box_a, q.box_b)) for q in unique)
                      for side in (p.box_a, p.box_b))
        out.append(replace(p, lattice=lattice, repetitive=p.multi_share >= MULTI_SHARE or lattice >= LATTICE_PAIRS))
    return CopyMoveResult(width, height, analysed, len(positions), len(pairs), crowded_share, len(seen), out[:MAX_PAIRS])
