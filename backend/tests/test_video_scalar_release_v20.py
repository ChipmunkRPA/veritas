"""Public packets preserve scalar scope/availability while removing source names."""
import copy

import pytest
from test_specialist_calibration import fixture

from scripts.release_video_scalars_v20 import assert_probabilities, public_manifest


def test_public_remap_preserves_source_units_roles_labels_and_missing_responses():
    original = fixture()
    before = copy.deepcopy(original)
    original['files'][0]['units'][0].update(status='not_applicable', raw=None, reason='Not assessed.')
    public = public_manifest(original)
    assert original['files'][0]['id'] == before['files'][0]['id']
    assert public['policy'] == original['policy']
    for a, b in zip(public['files'], original['files'], strict=True):
        assert a['id'] != b['id'] and a['group'] != b['group'] and a['lineage'] != b['lineage']
        assert {k: v for k, v in a.items() if k not in ('id', 'group', 'lineage')} == {
            k: v for k, v in b.items() if k not in ('id', 'group', 'lineage')}


def test_public_numeric_check_refuses_turning_missing_into_zero():
    prior = {'outcomes': [{'probability': None}], 'unit_outcomes': []}
    with pytest.raises(ValueError, match='available evidence'):
        assert_probabilities({'outcomes': [{'probability': 0.}], 'unit_outcomes': []}, prior)
