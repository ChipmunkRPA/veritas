"""Method isolation and withholding using already-public retained scalars."""
from copy import deepcopy

import pytest

from app.bench import specialists as legacy
from app.bench import specialists_scoped_v3 as scoped
from scripts import scoped_writing_v6 as study


def test_four_variants_keep_identical_local_measurements_truth_and_roles():
    documents = study.documents()
    assert len(documents) == 8
    for context in ('global', 'context'):
        original = documents[context + '-original-combined']
        shared = documents[context + '-shared-max']
        revised = documents[context + '-shared-scoped']
        assert shared == revised
        for first, second in zip(original['files'], shared['files'], strict=True):
            assert {k: v for k, v in first.items() if k != 'required_families'} == {
                k: v for k, v in second.items() if k != 'required_families'}
        assert len(shared['files']) == 512
        assert all(file['required_families'] == ['writing-views'] for file in shared['files'])
        assert all(len(file['units']) == 4 for file in shared['files'])
    assert study.engine_for('global-shared-scoped') is scoped
    assert study.engine_for('global-shared-max') is legacy


def test_common_source_model_responses_do_not_have_duplicate_same_scope_views():
    document = study.documents()['context-shared-scoped']
    scoped.audit(document)
    altered = deepcopy(document)
    unit = deepcopy(altered['files'][0]['units'][0])
    unit['id'] += '-repeat'
    altered['files'][0]['units'].append(unit)
    with pytest.raises(ValueError, match='Duplicate specialist'):
        scoped.audit(altered)


def test_identity_roundtrips_and_all_four_roles_are_preserved():
    import json

    assert json.loads(json.dumps(study.METHOD)) == study.METHOD
    for doc in study.documents().values():
        assert {role: sum(f['role'] == role for f in doc['files']) for role in legacy.ROLES} == {
            role: 128 for role in legacy.ROLES}
