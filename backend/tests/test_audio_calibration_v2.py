"""Guard against leakage and false certainty in the offline calibration pilot."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import calibrate_audio_v2 as pilot


def guard(n=12):
    return {str(label): [{'speaker': f'{label}:{i}', 'maximum': .1} for i in range(n)] for label in (0, 1)}


def test_rejection_requires_unique_class_and_probability_threshold():
    assert pilot.decision(.95, guard()) == 'ai'
    assert pilot.decision(.05, guard()) == 'original'
    assert pilot.decision(.79, guard()) == 'inconclusive'
    overlapping = guard()
    for row in overlapping['0']: row['maximum'] = .99
    assert pilot.decision(.95, overlapping) == 'inconclusive'


def test_small_calibration_population_never_forces_a_label():
    assert pilot.decision(.999, guard(8)) == 'inconclusive'
    assert pilot.decision(None, guard()) == 'inconclusive'
    assert pilot.decision(float('nan'), guard()) == 'inconclusive'


def test_missing_condition_retains_calibration_identity_as_infinite():
    rows = [dict(split='cal', label=1, speaker='voice', new=.8),
            dict(split='cal', label=1, speaker='voice', new=None)]
    fitted = pilot.calibrate_guard(rows, 'new')
    assert fitted['1'] == [{'speaker': 'voice', 'maximum': None, 'conditions': 2}]
    uninformative = guard()
    for row in uninformative['1']: row['maximum'] = None
    assert pilot.decision(.01, uninformative) == 'inconclusive'


def test_calibration_cannot_accept_test_rows():
    with pytest.raises(ValueError, match='cal only'):
        pilot.calibrate_guard([dict(split='test', label=0, speaker='heldout', new=.1)], 'new')


def test_calibration_uses_worst_condition_per_identity_not_rows_as_samples():
    rows = [dict(split='cal', label=0, speaker='real', new=p) for p in (.01, .4, .2)]
    result = pilot.calibrate_guard(rows, 'new')
    assert len(result['0']) == 1
    assert result['0'][0]['maximum'] == .4


def test_probability_mapping_is_monotone():
    parameters = {'slopes': [.4], 'intercept': -.2}
    values = [pilot.probability(x, parameters) for x in (0., .1, .5, .9, 1.)]
    assert values == sorted(values)
    assert pilot.probability(None, parameters) is None


def test_empty_shared_assessed_population_is_reported_unavailable():
    result = pilot.metrics([], 'new', True)
    assert result['total'] == 0
    assert result['coverage'] is None
    assert result['brier'] is None
    assert result['false_ai_rate'] is None
    assert result['false_original_rate'] is None


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -.1, 1.1])
def test_invalid_detector_output_is_not_a_probability(value):
    assert not pilot.valid_probability(value)
