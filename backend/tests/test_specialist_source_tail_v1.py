"""Operating comparisons use calibration sources; authored controls only."""
from copy import deepcopy

import pytest
from test_specialist_calibration import fixture

from app.bench import specialists as engine
from scripts.specialist_source_tail_v1 import gates, point, predict, summarize


def genuine(n=30):
    return [{'role': 'fusion_cal', 'label': 0, 'group': f'g{i}', 'probability': (i + 1) / 100} for i in range(n)]


def test_one_source_maximum_not_thousands_of_duplicate_windows():
    rows = genuine()
    original = point(rows)
    assert original['upper_rank'] == 28 and original['threshold'] == .28
    assert original['calibration_genuine_groups_flagged'] == 2
    assert point(rows + [deepcopy(rows[0])] * 1000) == original


def test_ai_and_reserved_responses_cannot_select_operating_point():
    rows = genuine()
    original = point(rows)
    rows += [{'role': 'fusion_cal', 'label': 1, 'group': 'ai', 'probability': .99},
        {'role': 'confirmation', 'label': 0, 'group': 'reserved', 'probability': 1.}]
    assert point(rows) == original


def test_unassessed_groups_are_not_zero_imputed_or_counted_as_supported():
    rows = genuine()
    for row in rows[:6]:
        row['probability'] = None
    result = point(rows)
    assert result['status'] == 'insufficient_support' and result['threshold'] is None
    assert result['genuine_groups_with_assessed_files'] == 24
    assert result['genuine_groups_with_no_assessed_files'] == 6


def test_partially_assessed_group_keeps_explicit_missing_file_count():
    rows = genuine(25)
    rows.append({**rows[0], 'probability': None})
    result = point(rows)
    assert result['status'] == 'fitted' and result['upper_rank'] == 24
    assert result['genuine_groups_total'] == result['genuine_groups_with_assessed_files'] == 25
    assert result['unassessed_genuine_files'] == 1


def test_strict_ties_are_not_ai_flags_and_missing_ai_stays_in_denominator():
    operating = point([{**row, 'probability': .5} for row in genuine()])
    assert operating['calibration_genuine_groups_flagged'] == 0
    rows = [{'label': 1, 'group': 'a', 'probability': .5}, {'label': 1, 'group': 'b', 'probability': None}]
    result = summarize(rows, operating)
    assert result['ai_files_found'] == 0 and result['ai_not_flagged_including_inconclusive'] == 2
    assert result['exact_operating_point_ties'] == 1 and result['inconclusive'] == 1


def test_repeated_genuine_files_do_not_dominate_probability_loss():
    rows = [{'label': 0, 'group': 'a', 'probability': .9}] * 100
    rows += [{'label': 0, 'group': 'b', 'probability': .1}, {'label': 1, 'group': 'c', 'probability': .8}]
    result = summarize(rows, {'status': 'fitted', 'threshold': .5})
    assert result['source_class_balanced_brier_on_assessed_files'] == pytest.approx(((.81 + .01) / 2 + .04) / 2)
    assert result['false_ai_files'] == 100 and result['false_ai_groups'] == 1


def test_aggregate_gain_cannot_hide_condition_detection_loss():
    a = {'ai_files_found': 4, 'false_ai_files': 0, 'false_ai_groups': 0, 'ai_groups_found': 4,
        'assessed': 10, 'source_class_balanced_brier_on_assessed_files': .1}
    b = {**a, 'ai_files_found': 3, 'source_class_balanced_brier_on_assessed_files': .2}
    actual = {'all': a, 'conditions': {'easy': {**a, 'ai_files_found': 4}, 'hard': {**a, 'ai_files_found': 0}}}
    reference = {'all': b, 'conditions': {'easy': {**b, 'ai_files_found': 2}, 'hard': {**b, 'ai_files_found': 1}}}
    result = gates(actual, reference)
    assert result['more_ai_files_found'] is True and result['no_condition_ai_detection_loss'] is False


def test_prediction_matches_shared_engine_and_does_not_impute_missing():
    manifest = fixture()
    # Authored numerical fixture only; the actual study always uses25groups/class.
    artifact = engine.train(manifest, min_groups_per_class=3)
    expected = {row['id']: row['probability'] for row in engine.evaluate(manifest, artifact)['outcomes']}
    for file in manifest['files']:
        if file['role'] == 'confirmation':
            assert predict(file, artifact) == expected[file['id']]
    file = deepcopy(next(f for f in manifest['files'] if f['role'] == 'confirmation'))
    file['units'][0].update(status='skipped', raw=None, reason='No required evidence')
    assert predict(file, artifact) is None


def test_invalid_calibration_response_cannot_enter_source_tail():
    rows = genuine()
    rows[0]['probability'] = float('nan')
    with pytest.raises(ValueError):
        point(rows)
