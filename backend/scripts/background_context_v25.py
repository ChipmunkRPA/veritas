"""Research-only face-transition context; never an AI-authorship probability."""
from __future__ import annotations

import math

import numpy as np

ROWS, COLS, MARGIN, MIN_BACKGROUND = 12, 16, .25, .25
PENALTY, STRENGTH = .35, 2.


def require(value, message):
    if not value:
        raise ValueError(message)


def describe(rgb):
    require(rgb.dtype == np.uint8 and rgb.shape == (480, 640, 3), 'Exact native 640x480 uint8 RGB required')
    return rgb.reshape(ROWS, 40, COLS, 40, 3).mean(axis=(1, 3), dtype=np.float64) / 255.


def background_mask(boxes):
    require(bool(boxes), 'Every source face box required')
    keep = np.ones((ROWS, COLS), bool)
    for box in boxes:
        require(len(box) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in box),
                'Finite native face box required')
        x, y, w, h = box
        require(0 < w <= 1280 and 0 < h <= 960 and x < 640 and y < 480 and x + w > 0 and y + h > 0,
                'Face box must intersect the native source')
        x0, y0 = max(0, x - MARGIN * w), max(0, y - MARGIN * h)
        x1, y1 = min(640, x + (1 + MARGIN) * w), min(480, y + (1 + MARGIN) * h)
        # Exclude every grid cell intersecting any expanded face box. One
        # common mask is used for all frames in this exact temporal support.
        keep[math.floor(y0 / 40):math.ceil(y1 / 40), math.floor(x0 / 40):math.ceil(x1 / 40)] = False
    return keep


def measure(unit, descriptors, boxes):
    needed = sorted({i for key in ('sampled_frames', 'context_frames') for i in unit['location'][key]})
    if not set(needed) <= descriptors.keys() or not set(needed) <= boxes.keys():
        return {'status': 'not_applicable', 'raw': None, 'reason': 'Missing native source or face context.'}
    mask = background_mask([boxes[i] for i in needed])
    if float(mask.mean()) < MIN_BACKGROUND:
        return {'status': 'not_applicable', 'raw': None, 'reason': 'Less than 25% common background support.',
                'background_cells': int(mask.sum())}
    for i in needed:
        value = descriptors[i]
        require(value.dtype == np.float64 and value.shape == (ROWS, COLS, 3)
                and np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(),
                'Complete finite normalized native background descriptor required')

    def block(centers):
        require(bool(centers) and all(i + k in needed for i in centers for k in (-1, 0, 1)),
                'Exact before/middle/after three-frame support required')
        return np.mean([np.mean([descriptors[i + k] for k in (-1, 0, 1)], axis=0)
                        for i in centers], axis=0)

    left, middle, right = [block(unit[k]) for k in ('before_centers', 'middle_centers', 'after_centers')]
    lm, rm, flank = [float(np.abs(a - b)[mask].mean()) for a, b in
                     ((middle, left), (middle, right), (left, right))]
    value = max(0., min(lm, rm) - PENALTY * flank)
    require(0 <= value <= 1 and math.isfinite(value), 'Invalid source-background transition')
    return {'status': 'ok', 'raw': value, 'left_change': lm, 'right_change': rm, 'flank_change': flank,
            'background_cells': int(mask.sum())}


def adjusted(appearance, context):
    if appearance is None or context['status'] != 'ok':
        return None
    require(type(appearance) in (int, float) and math.isfinite(appearance) and -4 <= appearance <= 4,
            'Finite bound original appearance response required')
    return float(appearance - STRENGTH * context['raw'])


def fit_point(rows, reference):
    """Use genuine calibration groups only; match the original cue's group budget."""
    require(type(reference) in (int, float) and math.isfinite(reference), 'Finite comparison reference required')
    grouped, seen = {}, set()
    for row in rows:
        require(row['role'] == 'fusion_cal' and type(row['label']) is int and row['label'] == 0 and row['id'] not in seen,
                'Unique genuine calibration cases only')
        seen.add(row['id'])
        grouped.setdefault(row['group'], []).append(row)
    complete = [part for part in grouped.values() if all(r['baseline'] is not None and r['candidate'] is not None for r in part)]
    require(len(complete) >= 25, 'At least 25 common assessed genuine groups required')
    original = [max(r['baseline'] for r in part) for part in complete]
    candidate = [max(r['candidate'] for r in part) for part in complete]
    require(all(math.isfinite(v) for v in original + candidate), 'Finite genuine group responses required')
    budget = sum(v > reference for v in original)
    require(budget < len(complete), 'A nontrivial genuine reference budget is required')
    point = sorted(candidate, reverse=True)[budget]
    return {'value': point, 'comparison': 'strict_greater_than', 'assessed_groups': len(complete),
            'unassessed_groups': len(grouped) - len(complete), 'baseline_flagged_groups': budget,
            'candidate_flagged_groups': sum(v > point for v in candidate), 'uses_ai_or_diagnostic': False,
            'scope': 'Internal matched genuine-group comparison; not AI confidence.'}
