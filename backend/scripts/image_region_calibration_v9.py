"""Region-aligned calibration, a new consumed-data hypothesis after v8 rejection.

The statistic is the highest superlevel at which a sufficiently large connected
region exists. It matches the display's area rule, unlike a single-cell maximum.
No reference mask is read here; this is not an AI probability.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def validate(floating, reliable, minimum_fraction):
    if (floating.ndim != 2 or floating.shape != reliable.shape or floating.dtype != np.float32
            or reliable.dtype != bool or not np.isfinite(floating).all()
            or not math.isfinite(minimum_fraction) or not 0 < minimum_fraction <= 1):
        raise ValueError('Complete finite float32 grid, Boolean reliability and positive area fraction required')
    return max(1, math.ceil(minimum_fraction * floating.size))


def largest_component(binary):
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0


def critical_level(floating, reliable, minimum_fraction):
    """Maximum minimum response of any eligible connected set, with plateau ties.

    With display eligibility `score > point`, a region exists iff critical>point.
    The minimum is relative to the entire grid, exactly as in display filtering.
    """
    minimum = validate(floating, reliable, minimum_fraction)
    if int(reliable.sum()) < minimum or largest_component(reliable) < minimum:
        return None
    candidates = np.unique(floating[reliable])
    lo, hi, evaluations = 0, len(candidates), 0
    while lo + 1 < hi:
        middle = (lo + hi) // 2
        evaluations += 1
        if largest_component((floating >= candidates[middle]) & reliable) >= minimum:
            lo = middle
        else:
            hi = middle
    return {'critical_level': float(candidates[lo]), 'minimum_component_cells': minimum,
        'unique_eligible_levels': int(candidates.size), 'binary_search_evaluations': evaluations}


def flood_largest(binary):
    """Independent8-neighbour graph traversal; no OpenCV component labels."""
    unseen = set(zip(*np.nonzero(binary), strict=True))
    maximum = 0
    while unseen:
        queue, size = [unseen.pop()], 0
        while queue:
            y, x = queue.pop()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = y + dy, x + dx
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
        maximum = max(maximum, size)
    return maximum


def verify_critical(floating, reliable, minimum_fraction, response):
    """A feasible plateau and infeasible strict successor certify the optimum."""
    minimum = validate(floating, reliable, minimum_fraction)
    if response is None:
        maximum = flood_largest(reliable)
        if maximum >= minimum:
            raise ValueError('A feasible connected region was incorrectly unavailable')
        return {'passed': True, 'available': False, 'largest_reliable_component': maximum}
    critical = response['critical_level']
    if not math.isfinite(critical) or not np.any(floating[reliable] == critical):
        raise ValueError('Critical level must be an actual reliable response value')
    nonstrict = flood_largest((floating >= critical) & reliable)
    strict = flood_largest((floating > critical) & reliable)
    if nonstrict < minimum or strict >= minimum or response['minimum_component_cells'] != minimum:
        raise ValueError('Critical level is not the maximum feasible region plateau')
    return {'passed': True, 'available': True, 'nonstrict_largest_component': nonstrict,
        'strict_largest_component': strict, 'minimum_component_cells': minimum}


def point_from_genuine(critical_values):
    if len(critical_values) != 60:
        raise ValueError('Exactly60 planned genuine source values required')
    values = [float(v) for v in critical_values if v is not None]
    if len(values) < 48 or not all(math.isfinite(v) for v in values):
        raise ValueError('At least48 eligible finite genuine source values required')
    point = sorted(values, reverse=True)[1]
    if point <= 0:
        raise ValueError('Positive genuine-only operating point required')
    return point
