# SPDX-License-Identifier: Apache-2.0
"""Compare local changes in next-token entropy on matched writing passages."""

from __future__ import annotations

from math import ceil, fsum, sqrt
from statistics import median

WIDTHS = (1, 2, 3)
TOKEN_BOUNDS = (24, 48, 96, 193)
UNITS = 1_000_000
ARMS = ("prominence", "nearby_prominence", "prominence_and_curvature", "curvature")
MAX_RANGES = 6


def prominence(values):
    """Sum each strict peak's height over the higher of its two valleys."""
    heights = []
    for index in range(1, len(values) - 1):
        top = values[index]
        if not values[index - 1] < top > values[index + 1]:
            continue
        valleys = []
        for positions in (range(index - 1, -1, -1), range(index + 1, len(values))):
            low = top
            for position in positions:
                height = values[position]
                if height > top:
                    break
                low = min(low, height)
            valleys.append(low)
        heights.append(top - max(valleys))
    return sum(heights)


def reduce_window(entropy, deviations, variances):
    size = len(entropy)
    total = sum(entropy)
    variation = sum(variances)
    if not (TOKEN_BOUNDS[0] <= size < TOKEN_BOUNDS[-1] and total > 0 and variation > 0):
        return None
    return [size, round(UNITS * prominence(entropy) / total),
            round(fsum(deviations) / sqrt(variation / UNITS))]


def bucket(count):
    return next(index for index, stop in enumerate(TOKEN_BOUNDS[1:]) if count < stop)


def quantile(values, proportion):
    return sorted(values)[ceil(proportion * len(values)) - 1]


def fit_scales(cases):
    values = {}
    for case in cases:
        for first, stop, count, relative, curvature in case["measurements"]:
            group = values.setdefault((stop - first, bucket(count)), {})
            group.setdefault(case["group"], []).append((relative, curvature))
    scales = {}
    for (width, length), group in sorted(values.items()):
        if len(group) < 5:
            continue
        center = [sorted(median(readings[i] for readings in local) for local in group.values()) for i in (0, 1)]
        upper = [sorted(quantile([readings[i] for readings in local], .80) for local in group.values()) for i in (0, 1)]
        scales[f"{width}/{length}"] = [[median(center[i]), max(1, median(upper[i]) - median(center[i]))]
                                        for i in (0, 1)]
    return scales


def candidates(case, scales):
    standardized = {}
    for first, stop, count, relative, curvature in case["measurements"]:
        scale = scales.get(f"{stop - first}/{bucket(count)}")
        if scale is not None:
            standardized[first, stop] = tuple(round(UNITS * (x - center) / spread)
                                               for x, (center, spread) in zip((relative, curvature), scale, strict=True))
    results = {arm: [] for arm in ARMS}
    for (first, stop), (relative, curvature) in standardized.items():
        results["prominence"].append((relative, first, stop))
        results["curvature"].append((curvature, first, stop))
        results["prominence_and_curvature"].append((min(relative, curvature), first, stop))
        left = [standardized[index, index + 1][0] for index in range(max(0, first - 3), first)
                if (index, index + 1) in standardized]
        right = [standardized[index, index + 1][0] for index in range(stop, stop + 3)
                 if (index, index + 1) in standardized]
        if len(left) >= 2 and len(right) >= 2:
            score = relative - max(median(left), median(right))
            results["nearby_prominence"].append((round(score), first, stop))
    return results


def max_distinct(candidates, incumbent):
    if len(incumbent) >= MAX_RANGES:
        return None
    disjoint = ((score, first, stop) for score, first, stop in candidates
                if all(last < first or start > stop for start, last in incumbent))
    ordered = sorted(disjoint, key=lambda item: (-item[0], item[2] - item[1], item[1]))
    return ordered[0] if ordered else None


def fit_points(controls, scales, *, fraction):
    by_group = {arm: {} for arm in ARMS}
    for case in controls:
        for arm, proposals in candidates(case, scales).items():
            selected = max_distinct(proposals, case["existing"])
            if selected is not None:
                scores = by_group[arm]
                scores.setdefault(case["group"], []).append(selected[0])
    return {arm: quantile([max(scores) for scores in groups.values()], fraction) if groups else None
            for arm, groups in by_group.items()}


def selections(case, scales, points):
    result = {}
    for arm, proposals in candidates(case, scales).items():
        chosen = max_distinct(proposals, case["existing"])
        point = points[arm]
        result[arm] = (list(chosen[1:]) if point is not None and chosen is not None and chosen[0] > point else None,
                       bool(proposals) and point is not None)
    return result


def compact_range(sentences, interval):
    first, stop = interval
    return [sentences[first][2], sentences[stop - 1][3]]


def overlap(a, b):
    shared = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return 2 * shared >= b[1] - b[0] and 10 * shared >= 3 * (max(a[1], b[1]) - min(a[0], b[0]))


def matched(intervals, truth):
    """Find a maximum one-to-one assignment with the published overlap rule."""
    occupied = {}

    def augment(index, visited):
        for target, actual in enumerate(truth):
            if target in visited or not overlap(intervals[index], actual):
                continue
            visited.add(target)
            if target not in occupied or augment(occupied[target], visited):
                occupied[target] = index
                return True
        return False

    for index in range(len(intervals)):
        augment(index, set())
    return sorted(occupied)
