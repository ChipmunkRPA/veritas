"""Information preservation and source invariants on authored four-media cases."""
import json
from copy import deepcopy

import pytest
from test_specialist_calibration import fixture

from app.bench import specialists as legacy
from scripts import specialists_preserved_v4 as new


def shared():
    document = fixture()
    for spec in document['policy']['specialists']:
        spec['family'] = 'shared'
    for file in document['files']:
        file['required_families'] = ['shared']
    return document


def heads(document, mode):
    return {'policy': document['policy'], 'mode': mode, 'specialists': {
        s['id'] + '|' + c: {'status': 'fitted', 'center': [0.], 'scale': [1.], 'coefficients': [1.], 'intercept': 0.}
        for s in document['policy']['specialists'] for c in s['contexts']}}


@pytest.mark.parametrize('modality', legacy.MEDIA)
def test_per_view_features_distinguish_which_check_generated_the_maximum(modality):
    document = shared()
    file = next(f for f in document['files'] if f['modality'] == modality)
    for u in file['units']:
        u['raw'] = 4. if '.first' in u['specialist'] else -5.
    assert new.features(file, heads(document, 'family-max')) == [4.]
    assert new.features(file, heads(document, 'view-max')) == [4., 0.]
    for u in file['units']:
        u['raw'] = -5. if '.first' in u['specialist'] else 4.
    assert new.features(file, heads(document, 'family-max')) == [4.]
    assert new.features(file, heads(document, 'view-max')) == [0., 4.]


def test_negative_other_area_cannot_reduce_the_positive_view_maximum():
    document = shared()
    file = document['files'][0]
    for u in file['units']:
        u['raw'] = 4. if u['context'] == 'normal' else -5.
    before = new.features(file, heads(document, 'view-max'))
    for u in file['units']:
        if u['context'] == 'difficult':
            u['raw'] = -8.
    assert new.features(file, heads(document, 'view-max')) == before == [4., 4.]


@pytest.mark.parametrize('status', ['skipped', 'unavailable', 'not_applicable', 'failed'])
def test_missing_required_measurement_has_no_zero_feature(status):
    document = shared()
    file = document['files'][0]
    file['units'][0].update(status=status, raw=None, reason='Unavailable authored input.')
    assert new.features(file, heads(document, 'view-max')) is None


def test_equal_dimension_control_and_view_mode_share_penalty():
    document = shared()
    file = document['files'][0]
    assert new.layout(file, heads(document, 'view-max'))[1] == [2, 2]
    assert new.layout(file, heads(document, 'family-max-repeat'))[1] == [2, 2]
    assert new.layout(file, heads(document, 'family-max'))[1] == [1]


def test_all_four_media_fit_with_frozen_confirmation_and_source_checks():
    document = shared()
    artifact = new.train(document, 'view-max', minimum=3)
    result = new.evaluate(document, artifact)
    assert new.evaluate(document, json.loads(json.dumps(artifact))) == result
    assert len(result['outcomes']) == 32 and all(r['status'] == 'ok' for r in result['outcomes'])
    altered = deepcopy(document)
    for file in altered['files']:
        if file['role'] == 'confirmation':
            for unit in file['units']:
                unit['raw'] = -unit['raw']
    assert new.train(altered, 'view-max', minimum=3) == artifact
    file = next(f for f in altered['files'] if f['role'] == 'confirmation')
    file['units'][0].update(status='skipped', raw=None, reason='Required observation withheld.')
    row = next(r for r in new.evaluate(altered, artifact)['outcomes'] if r['id'] == file['id'])
    assert row['probability'] is None
    altered['files'][-1]['lineage'] = altered['files'][0]['lineage']
    with pytest.raises(ValueError, match='leakage'):
        new.audit(altered)


def test_signed_raw_maximum_cannot_be_used_to_cancel_a_different_view():
    with pytest.raises(ValueError, match='single-view reference'):
        new.train(shared(), 'raw-view-max', minimum=3)
