"""Independent finite-graph tests for a changed calibration statistic."""
import copy
import math

import numpy as np
import pytest

from scripts.image_region_calibration_v9 import (
    critical_level,
    flood_largest,
    point_from_genuine,
    verify_critical,
)


def test_single_peak_does_not_set_complete_region_response():
    scores = np.full((20, 20), .1, np.float32)
    scores[1:5, 1:5] = .4
    scores[15, 15] = 1
    reliable = np.ones_like(scores, bool)
    result = critical_level(scores, reliable, .04)
    assert result['critical_level'] == float(np.float32(.4)) and result['minimum_component_cells'] == 16
    assert verify_critical(scores, reliable, .04, result)['passed']


def test_strict_plateau_and_diagonal_connectivity():
    scores = np.eye(6, dtype=np.float32)
    reliable = np.ones_like(scores, bool)
    result = critical_level(scores, reliable, 6 / 36)
    assert result['critical_level'] == 1
    assert flood_largest(scores > 1) == 0 and flood_largest(scores >= 1) == 6
    assert verify_critical(scores, reliable, 6 / 36, result)['passed']


def test_reliability_does_not_shrink_minimum_area_denominator():
    scores = np.full((10, 10), .8, np.float32)
    reliable = np.zeros_like(scores, bool)
    reliable[2:5, 2:5] = True
    assert critical_level(scores, reliable, .1) is None
    assert verify_critical(scores, reliable, .1, None)['passed']
    reliable[2, 5] = True
    result = critical_level(scores, reliable, .1)
    assert result['minimum_component_cells'] == 10
    assert verify_critical(scores, reliable, .1, result)['passed']


@pytest.mark.parametrize('seed', range(12))
def test_binary_search_matches_exhaustive_graph_and_all_strict_thresholds(seed):
    rng = np.random.default_rng(seed)
    scores = rng.integers(-2, 7, size=(8, 11)).astype(np.float32) / 8
    reliable = rng.random(scores.shape) > .15
    fraction = .09
    minimum = math.ceil(fraction * scores.size)
    feasible = [float(value) for value in np.unique(scores[reliable])
        if flood_largest((scores >= value) & reliable) >= minimum]
    expected = max(feasible) if feasible else None
    result = critical_level(scores, reliable, fraction)
    assert (result['critical_level'] if result else None) == expected
    assert verify_critical(scores, reliable, fraction, result)['passed']
    for point in np.unique(scores):
        assert (expected is not None and expected > point) == (flood_largest((scores > point) & reliable) >= minimum)
    if result:
        changed = copy.deepcopy(result)
        changed['critical_level'] = float(np.nextafter(np.float32(expected), np.float32(np.inf)))
        with pytest.raises(ValueError):
            verify_critical(scores, reliable, fraction, changed)


def test_ineligible_disconnected_reliability_and_invalid_values():
    scores = np.ones((10, 10), np.float32)
    reliable = np.zeros_like(scores, bool)
    reliable[::3, ::3] = True
    assert critical_level(scores, reliable, .05) is None
    assert verify_critical(scores, reliable, .05, None)['passed']
    for invalid in (scores.astype(np.float64), np.full_like(scores, np.nan)):
        with pytest.raises(ValueError):
            critical_level(invalid, reliable, .05)
    with pytest.raises(ValueError):
        critical_level(scores, reliable, 0)


def test_genuine_only_point_keeps_strict_exceedance_budget():
    assert point_from_genuine(list(range(1, 61))) == 59
    assert point_from_genuine([.3] * 60) == .3
    assert point_from_genuine(list(range(1, 49)) + [None] * 12) == 47
    for invalid in ([1.] * 59, [1.] * 47 + [None] * 13, [0.] * 60, [float('inf')] * 60):
        with pytest.raises(ValueError):
            point_from_genuine(invalid)


def test_critical_boundary_matches_actual_two_region_display_and_pixel_counter():
    from scripts.localization_benchmark import policy
    from scripts.run_image_region_calibration_v9 import measure

    rng = np.random.default_rng(91)
    values = rng.integers(0, 12, size=(40, 50)).astype(np.float32) / 16
    reliable = rng.random(values.shape) > .1
    rule = policy()
    result = critical_level(values, reliable, rule['minimum_region_fraction'])
    critical = result['critical_level']
    truth = np.zeros((83, 127), bool)
    for point in [np.nextafter(np.float32(critical), np.float32(-np.inf)).item(), critical,
            np.nextafter(np.float32(critical), np.float32(np.inf)).item()]:
        metrics, proof = measure(values, reliable, point, [83, 127, 3], truth, rule)
        assert (metrics['predicted_positive_pixels'] > 0) == (critical > point)
        assert set(proof) == {'display_grid', 'reliable', 'selected_grid'}
