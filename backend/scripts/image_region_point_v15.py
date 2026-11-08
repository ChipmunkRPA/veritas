"""Stable context means and exact activation levels for displayed regions.

Research only. Neither a ranking value nor an activation level is AI confidence.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from scripts import image_patch_head_v14 as patch

ARMS = ('old_maximum', 'old_regions', 'stable_maximum', 'stable_regions')
PRIMARY = 'stable_regions'


def stable_mean(features):
    patch.require(features.ndim == 2 and features.dtype == np.float32 and np.isfinite(features).all(),
                  'Complete native float32 features required')
    return features.mean(axis=0, dtype=np.float64).astype(np.float32)


def extract(backbone, rgb, hook=False):
    features, _old_mean, tensor_sha = patch.extract(backbone, rgb, hook=hook)
    return features, stable_mean(features), tensor_sha


def activation(scores, minimum=4):
    """Highest superlevel at which a size-qualified 8-connected region appears.

    All equal-valued cells enter together. A strict-threshold displayed region
    exists precisely when this returned level is strictly greater than its point.
    """
    values = np.asarray(scores)
    patch.require(values.ndim == 2 and values.size >= minimum >= 1
                  and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
                  'Finite complete ranking grid and feasible component size required')
    height, width = values.shape
    flat = values.ravel()
    order = np.argsort(-flat.astype(np.float64), kind='stable')
    parent = np.full(values.size, -1, np.int64)
    sizes = np.zeros(values.size, np.int64)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    cursor = 0
    while cursor < len(order):
        end, level = cursor + 1, flat[order[cursor]]
        while end < len(order) and flat[order[end]] == level:
            end += 1
        added = order[cursor:end]
        for index in added:
            parent[index], sizes[index] = index, 1
        for index in added:
            y, x = divmod(int(index), width)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width:
                        neighbor = yy * width + xx
                        if parent[neighbor] >= 0:
                            a, b = root(int(index)), root(neighbor)
                            if a != b:
                                if sizes[a] < sizes[b]:
                                    a, b = b, a
                                parent[b], sizes[a] = a, sizes[a] + sizes[b]
        if any(sizes[root(int(index))] >= minimum for index in added):
            return float(level)
        cursor = end
    raise ValueError('Every complete feasible grid must have an activation level')


def select_point(records, kind, expected_groups=60):
    patch.require(kind in ('maximum', 'regions'), 'Declared point kind required')
    groups, identities = defaultdict(dict), set()
    for record in records:
        patch.require(record['role'] == 'point' and record['label'] == 0
                      and record['variant'] in ('original', 'ordinary_patch'),
                      'Only genuine point-role originals and ordinary controls may choose a point')
        identity = (record['group'], record['variant'])
        patch.require(identity not in identities, 'Duplicate calibration variant')
        identities.add(identity)
        value = record['maximum'] if kind == 'maximum' else record['activation']
        patch.require(np.isfinite(value) and 0 <= value <= 1, 'Complete finite calibration value required')
        groups[record['group']][record['variant']] = float(value)
    patch.require(len(groups) == expected_groups >= 25
                  and all(set(v) == {'original', 'ordinary_patch'} for v in groups.values()),
                  'Both genuine variants of every declared point group required')
    ordered = sorted((max(v.values()) for v in groups.values()), reverse=True)
    budget = expected_groups * 5 // 100
    return {'value': ordered[budget], 'kind': kind, 'comparison': 'strict_greater_than',
            'calibration_groups': expected_groups, 'maximum_flagged_calibration_groups': budget,
            'scope': 'Research comparison rule; not calibrated AI confidence or a public screening threshold'}
