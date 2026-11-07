"""Source-scope correctness, not model performance, on authored four-media data."""
from copy import deepcopy

import pytest

from app.bench import specialists as legacy
from app.bench import specialists_scoped_v3 as scoped
from tests.test_specialist_calibration import fixture


def identity_head():
    return {'status': 'fitted', 'center': [0.], 'scale': [1.], 'coefficients': [1.], 'intercept': 0.}


def same_family(document):
    for spec in document['policy']['specialists']:
        spec['family'] = 'shared'
    for file in document['files']:
        file['required_families'] = ['shared']
    return document


def raw_artifact(document):
    return {'policy': document['policy'], 'specialists': {
        spec['id'] + '|' + context: identity_head()
        for spec in document['policy']['specialists'] for context in spec['contexts']}}


@pytest.mark.parametrize('modality', legacy.MEDIA)
def test_same_scope_agreement_is_distinguished_from_disagreement(modality):
    document = same_family(fixture())
    file = next(f for f in document['files'] if f['modality'] == modality)
    file['units'] = [u for u in file['units'] if u['context'] == 'normal']
    for unit in file['units']:
        unit['raw'] = 4.
    artifact = raw_artifact(document)
    assert scoped._features(file, artifact) == [4., 4.]
    assert legacy._features(file, artifact) == [4.]
    file['units'][1]['raw'] = -4.
    assert scoped._features(file, artifact) == [4., 0.]
    assert legacy._features(file, artifact) == [4.]


@pytest.mark.parametrize('modality', legacy.MEDIA)
def test_negative_other_area_cannot_cancel_positive_local_evidence(modality):
    document = same_family(fixture())
    file = next(f for f in document['files'] if f['modality'] == modality)
    for unit in file['units']:
        unit['raw'] = 4. if unit['context'] == 'normal' else -7.
    assert scoped._features(file, raw_artifact(document)) == [4., 4.]
    # One view positive in each of two different areas is not same-area agreement.
    file['units'][2]['raw'] = -4.
    file['units'][3]['raw'] = 3.
    assert scoped._features(file, raw_artifact(document)) == [4., 0.]


def test_different_semantic_targets_cannot_manufacture_agreement():
    document = same_family(fixture())
    file = document['files'][0]
    file['units'] = [u for u in file['units'] if u['context'] == 'normal']
    for unit in file['units']:
        unit['raw'] = 3.
    assert scoped._features(file, raw_artifact(document)) == [3., 3.]
    document['policy']['specialists'][1]['target'] = 'image.another_target'
    assert scoped._features(file, raw_artifact(document)) == [3., 0.]


def test_duplicate_view_cannot_manufacture_consistency_even_with_a_different_context():
    document = fixture()
    original = document['files'][0]['units'][0]
    duplicate = deepcopy(original)
    duplicate.update(id='repeat', context='difficult')
    document['files'][0]['units'].append(duplicate)
    with pytest.raises(ValueError, match='Duplicate specialist'):
        scoped.audit(document)


@pytest.mark.parametrize('status', ['failed', 'skipped', 'not_applicable', 'unavailable'])
def test_missing_planned_view_is_not_a_measured_zero_consistency(status):
    document = same_family(fixture())
    file = document['files'][0]
    file['units'][0].update(status=status, raw=None, reason='Authored unavailable check')
    assert scoped._features(file, raw_artifact(document)) is None


def test_four_media_fit_and_frozen_confirmation_and_old_artifact_refusal():
    document = same_family(fixture())
    artifact = scoped.train(document, min_groups_per_class=3)
    assert len(artifact['combiners']) == 4
    assert all(len(c['response']['coefficients']) == 2 for c in artifact['combiners'].values())
    result = scoped.evaluate(document, artifact)
    assert len(result['outcomes']) == 32 and all(r['status'] == 'ok' for r in result['outcomes'])
    revised = deepcopy(document)
    for file in revised['files']:
        if file['role'] == 'confirmation':
            for unit in file['units']:
                unit['raw'] = -unit['raw']
    assert scoped.train(revised, min_groups_per_class=3) == artifact
    assert scoped.evaluate(revised, artifact)['outcomes'] != result['outcomes']
    old = legacy.train(document, min_groups_per_class=3)
    with pytest.raises(ValueError, match='Exact fitted artifact'):
        scoped.evaluate(document, old)
    with pytest.raises(ValueError, match='Exact fitted artifact'):
        legacy.evaluate(document, artifact)


def test_unsupported_context_withholds_file_instead_of_an_original_verdict():
    document = same_family(fixture())
    for file in document['files']:
        if file['role'] == 'specialist_cal' and file['label'] == 1:
            for unit in file['units']:
                if unit['context'] == 'normal':
                    unit.update(status='unavailable', raw=None, reason='Authored missing class support')
    result = scoped.evaluate(document, scoped.train(document, min_groups_per_class=3))
    assert all(r['probability'] is None and r['status'] == 'inconclusive' for r in result['outcomes'])
