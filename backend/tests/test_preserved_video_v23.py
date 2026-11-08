"""Blinding, operating controls and projection scope use authored examples."""
import copy

import numpy as np
import pytest
from test_specialist_calibration import fixture

from app.bench import specialists as legacy
from scripts import monotone_newton_v4 as solver
from scripts.run_preserved_video_v23 import blind, fit_frozen_problem, mapped_point


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


def test_diagnostic_rows_restore_strict_python_scalars_without_changing_values():
    values = np.asarray([[.1], [-.3]], np.float64)
    for native, restored in zip(values, values.tolist(), strict=True):
        with pytest.raises(ValueError, match='Unassessed'):
            legacy._logit(model(), native)
        assert legacy._logit(model(), restored) == restored[0]
        assert np.array_equal(native, restored)


def test_optimizer_audit_holds_original_normalizer_even_if_host_differs_one_ulp():
    x = np.linspace(1., 3., 60)
    y, groups = [int(v > 2.) for v in x], [str(i) for i in range(60)]
    original = solver.fit(x, y, groups)
    original['center'][0] = float(np.nextafter(original['center'][0], np.inf))
    fitted, diagnostic = fit_frozen_problem(x, y, groups, original)
    assert fitted['center'] == original['center'] and fitted['scale'] == original['scale']
    assert diagnostic['host_center_maximum_difference'] > 0
    assert fitted['numerical_check']['independent_kkt_infinity'] <= 1e-9


def test_invalid_frozen_normalizer_cannot_be_substituted_by_host_values():
    x = np.linspace(1., 3., 60)
    y, groups = [int(v > 2.) for v in x], [str(i) for i in range(60)]
    original = solver.fit(x, y, groups)
    original['scale'][0] = 0.
    with pytest.raises(ValueError, match='normalizer'):
        fit_frozen_problem(x, y, groups, original)
