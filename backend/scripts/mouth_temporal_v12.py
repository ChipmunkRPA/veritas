"""Fixed small nonlinear mouth-window learner; raw margins, research only."""
from __future__ import annotations

import math

import numpy as np

from scripts.mouth_probe_v6 import source_weights
from scripts.mouth_temporal_v11 import FEATURE_NAMES

SPEC = {
    'method': 'mouth-temporal-histogram/12.0',
    'loss': 'log_loss', 'learning_rate': .05, 'max_iter': 100,
    'max_leaf_nodes': 7, 'max_depth': 3, 'min_samples_leaf': 40,
    'l2_regularization': 1., 'max_bins': 64, 'early_stopping': False,
    'categorical_features': None, 'random_state': 2718,
}
AI_CONDITIONS = {'ai-redraw', 'ai-retime', 'muse-ai-aligned', 'muse-ai-earlier'}


def matrix(values):
    x = np.asarray(values, np.float64)
    if x.ndim != 2 or x.shape[1] != 49 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Complete finite49-variable windows required')
    return x


def fit(rows):
    from sklearn.ensemble import HistGradientBoostingClassifier

    if not rows or any(r['role'] != 'fit' or type(r['label']) is not int or r['label'] not in (0, 1) for r in rows):
        raise ValueError('Only original binary fitting windows allowed')
    groups = sorted({r['group'] for r in rows})
    if len(groups) != 16 or any({r['condition'] for r in rows if r['group'] == g and r['label']} != AI_CONDITIONS for g in groups):
        raise ValueError('Every sixteen fitting actors/four positive constructions required')
    x = matrix([r['features'] for r in rows])
    model = HistGradientBoostingClassifier(**{k: v for k, v in SPEC.items() if k != 'method'})
    model.fit(x, [r['label'] for r in rows], sample_weight=source_weights(rows))
    if model.n_iter_ != 100 or model.n_features_in_ != 49 or list(model.classes_) != [0, 1]:
        raise ValueError('Fixed complete binary learner required')
    trees = []
    for iteration in model._predictors:
        if len(iteration) != 1:
            raise ValueError('Only one binary raw-margin tree per iteration permitted')
        nodes = iteration[0].nodes
        if any(n['is_categorical'] for n in nodes):
            raise ValueError('Categorical routing is not part of this method')
        trees.append([{k: float(n[k]) if k in ('value', 'num_threshold') else int(n[k])
                       for k in ('value', 'feature_idx', 'num_threshold', 'left', 'right', 'is_leaf')} for n in nodes])
    head = {'spec': SPEC, 'feature_names': list(FEATURE_NAMES), 'trees': trees,
            'base_margin': float(model._baseline_prediction[0, 0]), 'fit_windows': len(rows),
            'fit_groups': groups, 'response_kind': 'raw signed margin; not probability', 'deployment_eligible': False}
    delta = float(np.max(np.abs(model.decision_function(x) - margins(x, head))))
    if delta > 1e-10:
        raise ValueError('Exported complete raw tree response differs')
    head['native_fit_max_margin_difference'] = delta
    return model, head


def margins(values, head):
    x = matrix(values)
    if head['spec'] != SPEC or head['feature_names'] != list(FEATURE_NAMES) or len(head['trees']) != 100 or not math.isfinite(head['base_margin']):
        raise ValueError('Original complete finite model/order required')
    output = np.full(len(x), head['base_margin'], np.float64)
    for tree in head['trees']:
        if not tree or len(tree) > 13:
            raise ValueError('Fixed seven-leaf tree bound required')
        for i, row in enumerate(x):
            node = 0
            visited = set()
            while True:
                if type(node) is not int or node in visited or not 0 <= node < len(tree):
                    raise ValueError('Malformed cyclic tree routing')
                visited.add(node)
                n = tree[node]
                if n['is_leaf'] not in (0, 1) or not math.isfinite(n['value']):
                    raise ValueError('Finite explicit tree node required')
                if n['is_leaf']:
                    output[i] += n['value']
                    break
                feature = n['feature_idx']
                if type(feature) is not int or not 0 <= feature < 49 or not math.isfinite(n['num_threshold']):
                    raise ValueError('Finite numeric split required')
                node = n['left'] if row[feature] <= n['num_threshold'] else n['right']
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite raw response')
    return output
