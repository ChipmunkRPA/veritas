"""Method safeguards for bounded changes and genuine-only matched review budgets."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import sface_transition_v16 as m


def vector(index):
    out = np.zeros(128)
    out[index] = 1
    return out


def toy_sequence(return_to_before=True):
    table = SimpleNamespace(seconds=lambda i: i / 24)
    centers = set(range(6, 139, 6))
    scheduled = {i + k for i in centers for k in (-1, 0, 1)}
    features = {i: vector(1 if 54 <= i <= (90 if return_to_before else 1000) else 0) for i in scheduled}
    tracks = {'same-track': [SimpleNamespace(frame_index=i) for i in sorted(scheduled)]}
    return centers, tracks, features, table


def test_bounded_change_localizes_returning_middle_appearance():
    args = toy_sequence()
    values = m.windows(*args)
    assert values[0]['response'] > .8
    assert m.overlap(m.selected(values, .5, strict=True), [54 / 24, 90 / 24]) >= .3


def test_permanent_identity_change_does_not_supply_a_bounded_positive():
    values = m.windows(*toy_sequence(False))
    assert m.selected(values, .01, strict=True) is None


def test_tracks_are_not_joined_to_create_a_manipulation_window():
    centers, _, features, table = toy_sequence(False)
    tracks = {name: [SimpleNamespace(frame_index=i) for i in features if (i < 54) == first]
              for name, first in [('before', True), ('after', False)]}
    assert m.selected(m.windows(centers, tracks, features, table), .01, strict=True) is None


@pytest.mark.parametrize('value', [np.zeros(128), np.full(128, np.nan), np.ones(127)])
def test_bad_native_vectors_do_not_become_negative_evidence(value):
    with pytest.raises(ValueError):
        m.unit(value)


def calibration():
    items, responses = [], []
    for i in range(60):
        items.append({'id': str(i), 'role': 'calibration', 'label': 0, 'group': str(i // 2), 'legacy_range_s': None})
        responses.append({'id': str(i), 'status': 'ok', 'current_windows': [],
                          'candidate_windows': [{'start_s': 1., 'end_s': 2., 'response': i / 100}]})
    return items, responses


def test_point_allows_at_most_one_extra_genuine_calibration_flag():
    items, responses = calibration()
    point = m.operating_point(items, responses, .5)
    assert point['value'] == .58
    assert sum(m.selected(r['candidate_windows'], point['value'], strict=True) is not None for r in responses) == 1
    assert not point['confidence_calibration'] and not point['identity_match_point']


def test_already_flagged_genuine_scores_cannot_raise_incremental_budget_point():
    items, responses = calibration()
    for i in range(4):
        items[i]['legacy_range_s'] = [1, 2]
        responses[i]['candidate_windows'][0]['response'] = 1000 + i
    assert m.operating_point(items, responses, .5)['value'] == .58


def test_ai_calibration_labels_are_refused():
    items, responses = calibration()
    items[0]['label'] = 1
    with pytest.raises(ValueError):
        m.operating_point(items, responses, .5)


@pytest.mark.parametrize('change', ['missing', 'failed', 'nonfinite', 'too_little_context'])
def test_incomplete_or_unrepresentative_calibration_stops_development(change):
    items, responses = calibration()
    if change == 'missing':
        responses.pop()
    elif change == 'failed':
        responses[0]['status'] = 'failed'
    elif change == 'nonfinite':
        responses[0]['candidate_windows'][0]['response'] = float('inf')
    else:
        for row in responses[:30]:
            row['candidate_windows'] = []
    with pytest.raises(ValueError):
        m.operating_point(items, responses, .5)


def test_unassessed_candidate_retains_current_range_without_original_verdict():
    item = {'id': 'a', 'group': 'g', 'family': 'uniface', 'label': 1, 'role': 'development',
            'legacy_range_s': None, 'edit_s': [1, 2]}
    response = {'id': 'a', 'status': 'not_applicable', 'current_windows': [
        {'start_s': 1., 'end_s': 2., 'response': .8}], 'candidate_windows': []}
    out = m.outcomes([item], [response], {'value': .5}, .5)[0]
    assert out['candidate_range_s'] == out['current_range_s'] == [1, 2]
    assert 'direction' not in out and 'probability_ai' not in out
