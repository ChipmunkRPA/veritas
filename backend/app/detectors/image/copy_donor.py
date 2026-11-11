"""Bounded same-image matching on an existing source-coordinate highlight.

The fixed v295/v298 method compares native luminance blocks under one
translation or horizontal reflection. It supplies pixel evidence only.
"""

from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np

from app.detectors.image import copy_move

MIN_SIDE = copy_move.MIN_INPUT_SIDE
SIDE_LIMIT = 512
PATCH = 16
BORDER = 4
MIN_TEXTURE_STD = 8.0
MIN_ANCHORS = 4
MIN_CORRELATION = 0.96
MIN_SUPPORT_FRACTION = 2 / 3
MIN_DISTINCT_DONOR_MARGIN = 0.03
TRANSFORMS = ("translation", "horizontal_reflection")


def supports_size(width: int, height: int) -> bool:
    return min(width, height) >= MIN_SIDE and max(width, height) <= SIDE_LIMIT


def _bounds(mask):
    x = np.flatnonzero(mask.any(axis=0))
    y = np.flatnonzero(mask.any(axis=1))
    return int(x[0]), int(y[0]), int(x[-1] - x[0] + 1), int(y[-1] - y[0] + 1)


def _positions(start, extent):
    available = extent - PATCH - 2 * BORDER
    if available < 0:
        return ()
    count = min(3, 1 + available // PATCH)
    return np.linspace(start + BORDER, start + BORDER + available, count, dtype=int).tolist()


def _anchors(luma, selection, box):
    left, top, width, height = box
    result = []
    for y in _positions(top, height):
        for x in _positions(left, width):
            window = np.s_[y : y + PATCH, x : x + PATCH]
            patch = luma[window]
            if selection[window].all() and patch.std() >= MIN_TEXTURE_STD:
                result.append((x - left, y - top, patch))
    return result


def _candidate_donors(luma, patches, box, reflected):
    left, top, width, height = box
    rows, columns = luma.shape[0] - height + 1, luma.shape[1] - width + 1
    response = []
    for x, y, patch in patches:
        if reflected:
            x = width - PATCH - x
            patch = np.ascontiguousarray(patch[:, ::-1])
        matches = cv2.matchTemplate(luma, patch, cv2.TM_CCOEFF_NORMED)
        response.append(matches[y : y + rows, x : x + columns])
    scores = np.stack(response)
    support = np.count_nonzero(scores >= MIN_CORRELATION, axis=0)
    median = np.median(scores, axis=0)
    yy, xx = np.ogrid[:rows, :columns]
    target_overlap = (xx < left + width) & (xx + width > left) & (yy < top + height) & (yy + height > top)
    eligible = (support >= math.ceil(len(patches) * MIN_SUPPORT_FRACTION)) & ~target_overlap
    return np.where(eligible, median, -np.inf), support


def _strongest(maps):
    best = None
    for transform, (scores, support) in zip(TRANSFORMS, maps, strict=True):
        y, x = np.unravel_index(np.argmax(scores), scores.shape)
        score = float(scores[y, x])
        if np.isfinite(score) and (best is None or score > best["correlation"]):
            best = {
                "origin": (int(x), int(y)),
                "transform": transform,
                "correlation": score,
                "supporting_anchors": int(support[y, x]),
            }
    return best


def _competing_donor(maps, origin, width, height):
    best = None
    for scores, _ in maps:
        yy, xx = np.ogrid[: scores.shape[0], : scores.shape[1]]
        intersection = np.maximum(0, width - abs(xx - origin[0])) * np.maximum(
            0, height - abs(yy - origin[1])
        )
        # Same spatial donor, including a small alignment shift, is one alternative.
        distinct = intersection * 3 < 2 * width * height
        second = float(np.max(np.where(distinct, scores, -np.inf)))
        if np.isfinite(second) and (best is None or second > best):
            best = second
    return best


def corroborate(rgb: np.ndarray, selection: np.ndarray) -> dict:
    """Evaluate one existing source-coordinate Boolean mask without modifying it."""
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or selection.dtype != bool
        or selection.shape != rgb.shape[:2]
        or not selection.any()
    ):
        raise ValueError("Upright RGB and one nonempty source-coordinate Boolean selection required")
    box = _bounds(selection)
    record = {
        "proposal_box_xywh": list(box),
        "proposal_pixels": int(selection.sum()),
        "proposal_mask_sha256": hashlib.sha256(selection.tobytes()).hexdigest(),
        "anchor_count": 0,
        "status": "unsupported_geometry",
        "donor": None,
    }
    if not supports_size(*rgb.shape[:2]):
        return record
    luma, xscale, yscale = copy_move._luma(rgb)
    if xscale != 1 or yscale != 1:
        raise ValueError("Native coordinates must be retained")
    patches = _anchors(luma, selection, box)
    record.update(anchor_count=len(patches), status="insufficient_texture")
    if len(patches) < MIN_ANCHORS:
        return record
    maps = [_candidate_donors(luma, patches, box, reflected) for reflected in (False, True)]
    best = _strongest(maps)
    record["status"] = "no_donor"
    if best is None:
        return record
    width, height = box[2:]
    second = _competing_donor(maps, best["origin"], width, height)
    margin = best["correlation"] - second if second is not None else None
    record["competing_correlation"] = second
    record["donor_margin"] = margin
    if margin is not None and margin < MIN_DISTINCT_DONOR_MARGIN:
        record["status"] = "ambiguous_repetition"
        return record
    record.update(
        status="copy_consistent",
        donor={
            "box_xywh": [*best["origin"], width, height],
            "transform": best["transform"],
            "correlation": best["correlation"],
            "supporting_anchors": best["supporting_anchors"],
        },
    )
    return record
