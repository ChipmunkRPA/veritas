"""Forward source binding, explicit missing observations and cached parity."""
from copy import deepcopy

import pytest

from scripts.run_genuine_edits_v24 import bind, parity, raw_alert


def source():
    location = {'kind': 'time', 'unit': 'nanoseconds', 'start': 0, 'end': 1, 'extent': 2}
    unit = {'id': 'u', 'view': 'face-appearance', 'location': location}
    plan = {'id': 'ordinary', 'sha256': 'a' * 64, 'source_geometry': {'pts_table_sha256': 'b' * 64}, 'units': [unit]}
    measured = {'id': 'ordinary', 'input_sha256': 'a' * 64, 'native_table_sha256': 'b' * 64,
        'units': [{**unit, 'status': 'ok', 'raw': .2}]}
    artifact = {'policy': {'specialists': [{'id': 'face-appearance', 'modality': 'video',
        'family': 'video-face', 'contexts': ['single-face'], 'raw_range': [-4., 4.]}]}}
    return plan, measured, artifact


@pytest.mark.parametrize('status', ['unavailable', 'failed', 'not_applicable', 'skipped'])
def test_unassessed_local_input_is_preserved_and_cannot_become_a_negative_alert(status):
    plan, measured, artifact = source()
    measured['units'][0].update(status=status, raw=None, reason='No usable input.')
    bound = bind(plan, measured, artifact)
    assert bound['units'][0]['raw'] is None and bound['units'][0]['status'] == status
    assert raw_alert(measured, 'face-appearance') is None
    measured['units'][0]['raw'] = 0.
    with pytest.raises(ValueError, match='cannot become zero'):
        bind(plan, measured, artifact)


@pytest.mark.parametrize('mutation', ['clock', 'sha', 'location', 'duplicate', 'view'])
def test_forward_diagnostic_refuses_changed_source_or_support(mutation):
    plan, measured, artifact = source()
    if mutation == 'clock':
        measured['native_table_sha256'] = 'c' * 64
    elif mutation == 'sha':
        measured['input_sha256'] = 'd' * 64
    elif mutation == 'location':
        measured['units'][0]['location'] = {'changed': True}
    elif mutation == 'duplicate':
        measured['units'].append(deepcopy(measured['units'][0]))
    else:
        measured['units'][0]['view'] = 'face-crop'
    with pytest.raises(ValueError):
        bind(plan, measured, artifact)


def test_original_native_parity_has_strict_availability_and_explicit_numeric_limit():
    _, current, _ = source()
    cached = deepcopy(current)
    cached['units'][0]['raw'] += 1e-5
    assert parity(current, cached) == pytest.approx(1e-5)
    cached['units'][0]['raw'] += .1
    with pytest.raises(ValueError, match='parity exceeded'):
        parity(current, cached)
    cached['units'][0].update(status='not_applicable', raw=None)
    with pytest.raises(ValueError, match='availability'):
        parity(current, cached)
