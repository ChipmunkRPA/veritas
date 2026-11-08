"""Blinding, operating controls and projection scope use authored examples."""
import copy

import pytest
from test_specialist_calibration import fixture

from scripts.run_preserved_video_v21 import blind, mapped_point


def test_reserved_blinding_preserves_truth_and_existing_missing_observations():
    original = fixture()
    file = next(f for f in original['files'] if f['role'] == 'confirmation')
    file['units'][0].update(status='not_applicable', raw=None, reason='Authored missing input.')
    before = copy.deepcopy(original)
    current = blind(original)
    assert original == before
    assert current['files'][0] == original['files'][0]
    for a, b in zip(current['files'], original['files'], strict=True):
        assert a['label'] == b['label'] and a['role'] == b['role']
        for u, v in zip(a['units'], b['units'], strict=True):
            assert u['location'] == v['location'] and u['label'] == v['label']
            if a['role'] == 'confirmation':
                assert u['raw'] is None
                if v['status'] != 'ok':
                    assert u == v


def model(slope=1., center=0.):
    return {'status': 'fitted', 'center': [center], 'scale': [1.], 'coefficients': [slope], 'intercept': 0.}


@pytest.mark.parametrize('first,second', [(0., 1.), (1., 0.)])
def test_flat_file_calibration_cannot_claim_a_point_preserving_mapping(first, second):
    artifact = {'combiners': {'video|video-face': {'response': model(first), 'calibration': model(second)}}}
    assert mapped_point(artifact)['status'] == 'unsupported'


def test_threshold_in_a_clipped_tail_cannot_claim_preserved_directions():
    artifact = {'combiners': {'video|video-face': {'response': model(center=100.), 'calibration': model()}}}
    assert mapped_point(artifact)['threshold'] is None


def test_supported_point_mapping_is_explicit_and_finite():
    artifact = {'combiners': {'video|video-face': {'response': model(), 'calibration': model()}}}
    assert 0 < mapped_point(artifact)['threshold'] < 1
