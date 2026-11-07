import copy

import pytest

from scripts.review_mouth_temporal_v12 import margin


def head():
    tree = [{'is_leaf': 0, 'value': 0., 'feature_idx': 0, 'num_threshold': .5, 'left': 1, 'right': 2},
            {'is_leaf': 1, 'value': -.25}, {'is_leaf': 1, 'value': .75}]
    return {'base_margin': 1., 'trees': [copy.deepcopy(tree) for _ in range(100)]}


def test_hand_calculated_complete_tree_sum_and_inclusive_boundary():
    values = [0.] * 49
    assert margin(values, head()) == -24.
    values[0] = .5
    assert margin(values, head()) == -24.
    values[0] = .5000000000000001
    assert margin(values, head()) == 76.


@pytest.mark.parametrize('fault', ['cycle', 'unknown_feature', 'missing_features', 'nonfinite_feature', 'nonfinite_leaf'])
def test_malformed_tree_or_missing_measurement_cannot_supply_a_vote(fault):
    model, values = head(), [0.] * 49
    if fault == 'cycle':
        model['trees'][0][0]['left'] = 0
    elif fault == 'unknown_feature':
        model['trees'][0][0]['feature_idx'] = 49
    elif fault == 'missing_features':
        values.pop()
    elif fault == 'nonfinite_feature':
        values[0] = float('nan')
    else:
        model['trees'][0][1]['value'] = float('inf')
    with pytest.raises(ValueError):
        margin(values, model)
