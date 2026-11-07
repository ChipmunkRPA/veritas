"""Study safeguards: genuine-only selection and unassessed/original handling."""
import math

import pytest

from scripts import image_smaug_comparison_v7 as m


def calibration():
    items = [{'id': str(i), 'role': 'calibration', 'label': 0} for i in range(48)]
    return {'items': items}, [{'id': str(i), 'status': 'ok', 'mean_logit': i / 10} for i in range(48)]


def test_unseen_ai_responses_cannot_set_the_genuine_point():
    panel, rows = calibration()
    panel['items'].append({'id': 'ai', 'role': 'development', 'label': 1})
    rows.append({'id': 'ai', 'status': 'ok', 'mean_logit': 999})
    assert m.select_gate(panel, rows)['raw_mean_logit'] == 4.7


@pytest.mark.parametrize('failure', ['failed', 'skipped', 'not_applicable'])
def test_incomplete_calibration_does_not_admit_development(failure):
    panel, rows = calibration()
    rows[0]['status'] = failure
    with pytest.raises(ValueError):
        m.select_gate(panel, rows)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_any_nonfinite_calibration_refused(value):
    panel, rows = calibration()
    rows[3]['mean_logit'] = value
    with pytest.raises(ValueError):
        m.select_gate(panel, rows)


def sample():
    return {'id': 'a', 'group': 'g', 'label': 0, 'family': 'camera', 'scope': 'genuine',
            'format': 'native', 'role': 'development'}


def test_skipped_candidate_cannot_turn_an_inconclusive_result_into_original():
    result = m.account([sample()], [{'id': 'a', 'status': 'done',
        'interpretation': {'customer_direction': 'inconclusive', 'statistical_direction': 'inconclusive'}}],
        [{'id': 'a', 'status': 'skipped'}], {'raw_mean_logit': 4})[0]
    assert not result['candidate_positive_cue']
    assert result['baseline_direction'] == 'inconclusive'
    assert not result['combined_ai_indication']


def test_low_response_cannot_suppress_existing_ai_and_boundary_is_strict():
    for value in (-100, 4):
        row = m.account([sample()], [{'id': 'a', 'status': 'done',
            'interpretation': {'customer_direction': 'ai', 'statistical_direction': 'ai'}}],
            [{'id': 'a', 'status': 'ok', 'mean_logit': value}], {'raw_mean_logit': 4})[0]
        assert row['combined_ai_indication']
        assert not row['candidate_positive_cue']


def test_positive_original_disagreement_is_not_a_replacement_confidence():
    row = m.account([sample()], [{'id': 'a', 'status': 'done',
        'interpretation': {'customer_direction': 'original', 'statistical_direction': 'original'}}],
        [{'id': 'a', 'status': 'ok', 'mean_logit': 5}], {'raw_mean_logit': 4})[0]
    assert row['new_conflict_with_original']
    assert row['baseline_direction'] == 'original'
    assert not any('confidence' in k or 'probability' in k for k in row)


def test_provenance_result_is_not_statistical_detection():
    row = m.direction({'decision': {'code': 'ai_supported', 'probability_ai_calibrated': .99}}, {'direction': 'record'})
    assert row['provenance_led']
    assert row['statistical_direction'] == 'inconclusive'


def test_duplicate_outcomes_refused():
    with pytest.raises(ValueError):
        m.account([sample()], [{'id': 'a'}, {'id': 'a'}], [{'id': 'a'}], {'raw_mean_logit': 4})
