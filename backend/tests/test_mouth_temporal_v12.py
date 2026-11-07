import copy
import json

import numpy as np
import pytest

from scripts import mouth_temporal_v12 as method


@pytest.fixture(scope='module')
def fitted():
    rows = []
    for actor in range(16):
        for condition in sorted(method.AI_CONDITIONS):
            for label in (0, 1):
                x = np.random.default_rng(actor * 17 + label).normal(size=49)
                x[0] = 2. * label - 1.
                rows.append({'group': str(actor), 'role': 'fit', 'condition': condition,
                             'label': label, 'features': x.tolist()})
    model, head = method.fit(rows)
    return model, head, rows


def test_complete_export_reproduces_native_estimator_on_unseen_numeric_inputs(fitted):
    model, head, _ = fitted
    x = np.random.default_rng(301).normal(size=(30, 49))
    assert len(head['trees']) == 100 and head['native_fit_max_margin_difference'] <= 1e-10
    assert np.allclose(model.decision_function(x), method.margins(x, json.loads(json.dumps(head))), rtol=0, atol=1e-10)
    assert not head['deployment_eligible']


def test_exact_split_boundary_and_nextafter_follow_native_numeric_routing(fitted):
    model, head, _ = fitted
    node = next(n for t in head['trees'] for n in t if not n['is_leaf'])
    x = np.zeros((3, 49))
    t = node['num_threshold']
    x[:, node['feature_idx']] = [np.nextafter(t, -np.inf), t, np.nextafter(t, np.inf)]
    assert np.allclose(model.decision_function(x), method.margins(x, head), rtol=0, atol=1e-10)


@pytest.mark.parametrize('fault', ['omitted_tree', 'cycle', 'unknown_feature', 'nonfinite', 'feature_order'])
def test_invalid_export_cannot_supply_a_margin(fitted, fault):
    _, original, _ = fitted
    head = copy.deepcopy(original)
    if fault == 'omitted_tree':
        head['trees'].pop()
    elif fault == 'cycle':
        head['trees'][0][0].update(is_leaf=0, left=0, right=0)
    elif fault == 'unknown_feature':
        head['trees'][0][0].update(is_leaf=0, feature_idx=49)
    elif fault == 'nonfinite':
        head['base_margin'] = float('nan')
    else:
        head['feature_names'].reverse()
    with pytest.raises(ValueError):
        method.margins(np.zeros((1, 49)), head)


def test_point_role_never_enters_fitting(fitted):
    _, _, original = fitted
    rows = copy.deepcopy(original)
    rows[0]['role'] = 'point'
    with pytest.raises(ValueError):
        method.fit(rows)


def test_missing_measurements_cannot_acquire_a_score():
    with pytest.raises(ValueError):
        method.matrix([[float('nan')] * 49])
