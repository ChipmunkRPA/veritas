"""Numerical-only reconstruction of the declared mouth learner and JSON export.

No app, decoder, neural runtime, original training module or model assets.
The scikit-learn learner is shared; weights, input matrix and export are rebuilt.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def reconstruct(source, spec, scalar_margin):
    rows = source['fitting_windows']
    if len(rows) != 3560 or any(r['role'] != 'fit' or type(r['label']) is not int or r['label'] not in (0, 1) for r in rows):
        raise ValueError('All and only3560 original fitting windows required')
    groups = sorted({r['group'] for r in rows})
    if len(groups) != 16:
        raise ValueError('All sixteen fitting actors required')
    positive_conditions = {'ai-redraw', 'ai-retime', 'muse-ai-aligned', 'muse-ai-earlier'}
    if any({r['condition'] for r in rows if r['group'] == g and r['label']} != positive_conditions for g in groups):
        raise ValueError('Every original positive fitting construction required')
    counts = Counter((r['group'], r['label'], r['condition']) for r in rows)
    conditions = {(g, y): {c for actor, label, c in counts if actor == g and label == y} for g in groups for y in (0, 1)}
    if any(not v for v in conditions.values()):
        raise ValueError('Both classes for every fitting actor required')
    weights = np.asarray([len(rows) / (2 * len(groups) * len(conditions[r['group'], r['label']]) * counts[r['group'], r['label'], r['condition']]) for r in rows], np.float64)
    matrix = np.asarray([r['features'] for r in rows], np.float64)
    if matrix.shape != (3560, 49) or not np.isfinite(matrix).all():
        raise ValueError('Complete finite original fitting matrix required')
    learner = HistGradientBoostingClassifier(**{k: v for k, v in spec.items() if k != 'method'})
    learner.fit(matrix, [r['label'] for r in rows], sample_weight=weights)
    if learner.n_iter_ != 100 or list(learner.classes_) != [0, 1]:
        raise ValueError('Complete declared binary fit required')
    trees = []
    for iteration in learner._predictors:
        if len(iteration) != 1:
            raise ValueError('One binary tree per iteration required')
        tree = []
        for node in iteration[0].nodes:
            if node['is_categorical']:
                raise ValueError('Undeclared categorical routing')
            tree.append({'value': float(node['value']), 'feature_idx': int(node['feature_idx']),
                         'num_threshold': float(node['num_threshold']), 'left': int(node['left']),
                         'right': int(node['right']), 'is_leaf': int(node['is_leaf'])})
        trees.append(tree)
    head = {'spec': spec, 'feature_names': source['head']['feature_names'], 'trees': trees,
            'base_margin': float(learner._baseline_prediction[0, 0]), 'fit_windows': len(rows),
            'fit_groups': groups, 'response_kind': 'raw signed margin; not probability', 'deployment_eligible': False}
    scalar = np.asarray([scalar_margin(r['features'], head) for r in rows])
    difference = float(np.max(np.abs(learner.decision_function(matrix) - scalar)))
    if difference > 1e-10:
        raise ValueError('Native/independent scalar export differs')
    head['native_fit_max_margin_difference'] = difference
    return head
