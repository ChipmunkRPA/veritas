"""Research-only sound/mouth correspondence, not an AI probability or sync proof."""
from __future__ import annotations

import math

import numpy as np

from scripts import mouth_temporal_v12 as previous
from scripts.mouth_probe_v6 import source_weights

LAGS = (-8, -4, 0, 4, 8)
NEW_NAMES = tuple(f'{part}.{name}' for part in ('middle', 'support') for name in
    [*[f'corr_lag_{k}' for k in LAGS], 'best_lag_fraction', 'peak_minus_zero',
     'audio_std', 'aperture_std', 'usable_correlation_fraction'])
NAMES = tuple(previous.FEATURE_NAMES) + NEW_NAMES
SPEC = {**previous.SPEC, 'method': 'mouth-audio-context/14.0'}


def require(value, message):
    if not value:
        raise ValueError(message)


def envelope(pcm):
    """One non-overlapping log RMS value for every exact 640-sample/40-ms block."""
    require(pcm.dtype == np.float32 and pcm.ndim == 1 and len(pcm) >= 20 * 640
            and len(pcm) % 640 == 0 and np.isfinite(pcm).all(), 'Complete native mono16k sample blocks required')
    values = pcm.astype(np.float64).reshape(-1, 640)
    return np.log(np.maximum(np.sqrt(np.mean(values ** 2, axis=1)), 1e-8))


def apertures(trace, width, height, frames):
    require([r['frame_index'] for r in trace] == list(range(frames)) and width > 0 and height > 0,
            'Every committed native geometry frame required')
    points = np.asarray([r['points'] for r in trace], np.float64)
    require(points.shape == (frames, 7, 3) and np.isfinite(points).all(), 'Seven finite native mesh points per frame required')
    xy = points[:, :, :2] * [width, height]
    opening = np.linalg.norm(xy[:, 0] - xy[:, 1], axis=1)
    mouth = np.linalg.norm(xy[:, 2] - xy[:, 3], axis=1)
    require((mouth >= 1).all(), 'Resolved native mouth width required')
    return opening / mouth


def correlation(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    require(a.shape == b.shape and a.ndim == 1 and len(a) >= 4
            and np.isfinite(a).all() and np.isfinite(b).all(), 'Four finite corresponding observations required')
    a, b = a - a.mean(), b - b.mean()
    energy = float(np.linalg.norm(a) * np.linalg.norm(b))
    if energy <= 1e-12:
        return 0., False
    return float(np.clip(np.dot(a, b) / energy, -1., 1.)), True


def features(window, audio, aperture):
    require(audio.shape == aperture.shape and audio.ndim == 1 and np.isfinite(audio).all()
            and np.isfinite(aperture).all(), 'Complete shared-clock sound and mouth traces required')
    centers = window['centers']
    require(len(centers) >= 6 and all(type(i) is int for i in centers)
            and centers == sorted(set(centers)) and all(0 <= i < len(audio) for i in centers), 'Ordered complete before/middle/after centers required')
    n = len(audio)

    def span(a, b):
        # Fixed common anchor support for every lag; never wrap, pad or repeat.
        return set(range(max(a - 1, 8), min(b + 2, n - 8)))

    mid = span(centers[2], centers[-3])
    support = span(centers[0], centers[-1])
    output, supports = [], []
    for part in (mid, support):
        indices = np.asarray(sorted(part), np.int64)
        require(len(indices) >= 4, 'Insufficient common temporal support; do not invent a correspondence')
        pairs = [correlation(aperture[indices], audio[indices + lag]) for lag in LAGS]
        values = [p[0] for p in pairs]
        # Ties prefer zero, then smaller magnitude, then the earlier offset.
        best = min(range(5), key=lambda j: (-values[j], abs(LAGS[j]), LAGS[j]))
        output.extend([*values, LAGS[best] / 8, values[best] - values[2],
                       float(audio[indices].std()), float(aperture[indices].std()), sum(p[1] for p in pairs) / 5])
        supports.append(indices.tolist())
    require(len(output) == 20 and np.isfinite(output).all(), 'Complete finite20-variable correspondence required')
    return output, supports


def fit(rows):
    from sklearn.ensemble import HistGradientBoostingClassifier

    require(rows and all(r['role'] == 'fit' and type(r['label']) is int and r['label'] in (0, 1) for r in rows),
            'Only original binary fitting roles permitted')
    groups = sorted({r['group'] for r in rows})
    require(len(groups) == 16 and all({r['condition'] for r in rows if r['group'] == g and r['label']}
            == previous.AI_CONDITIONS for g in groups), 'Every fitting actor and AI construction required')
    x = np.asarray([r['features'] for r in rows], np.float64)
    require(x.shape == (len(rows), 69) and np.isfinite(x).all(), 'Complete fit-only69-variable matrix required')
    model = HistGradientBoostingClassifier(**{k: v for k, v in SPEC.items() if k != 'method'})
    model.fit(x, [r['label'] for r in rows], sample_weight=source_weights(rows))
    require(model.n_iter_ == 100 and model.n_features_in_ == 69 and list(model.classes_) == [0, 1],
            'Complete fixed binary learner required')
    trees = []
    for iteration in model._predictors:
        require(len(iteration) == 1 and not any(n['is_categorical'] for n in iteration[0].nodes),
                'One numeric binary tree per iteration required')
        trees.append([{k: float(n[k]) if k in ('value', 'num_threshold') else int(n[k])
                       for k in ('value', 'feature_idx', 'num_threshold', 'left', 'right', 'is_leaf')}
                      for n in iteration[0].nodes])
    head = {'spec': SPEC, 'feature_names': list(NAMES), 'trees': trees,
            'base_margin': float(model._baseline_prediction[0, 0]), 'fit_windows': len(rows),
            'fit_groups': groups, 'deployment_eligible': False, 'response_kind': 'raw margin; not AI confidence'}
    require(np.max(np.abs(model.decision_function(x) - margins(x, head))) <= 1e-10,
            'Native estimator and complete exported head differ')
    return model, head


def margins(values, head):
    x = np.asarray(values, np.float64)
    require(x.ndim == 2 and x.shape[1] == 69 and len(x) and np.isfinite(x).all(), 'Complete finite inference matrix required')
    require(head['spec'] == SPEC and head['feature_names'] == list(NAMES) and len(head['trees']) == 100
            and math.isfinite(head['base_margin']), 'Exact finite tree specification required')
    output = np.full(len(x), head['base_margin'], np.float64)
    for tree in head['trees']:
        require(0 < len(tree) <= 13, 'Fixed seven-leaf tree bound required')
        for i, row in enumerate(x):
            node, seen = 0, set()
            while True:
                require(type(node) is int and node not in seen and 0 <= node < len(tree), 'Acyclic complete tree routing required')
                seen.add(node)
                n = tree[node]
                require(n['is_leaf'] in (0, 1) and math.isfinite(n['value']), 'Finite explicit tree node required')
                if n['is_leaf']:
                    output[i] += n['value']
                    break
                require(type(n['feature_idx']) is int and 0 <= n['feature_idx'] < 69 and math.isfinite(n['num_threshold']),
                        'Finite valid numeric split required')
                node = n['left'] if row[n['feature_idx']] <= n['num_threshold'] else n['right']
    require(np.isfinite(output).all(), 'Finite raw margins required')
    return output
