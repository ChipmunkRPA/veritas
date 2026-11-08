"""Exact component calibration and stable context, before source-data scoring."""
import itertools

import numpy as np
import pytest

from scripts import image_patch_head_v14 as patch
from scripts import image_region_point_v15 as method
from scripts import review_image_region_point_v15 as review


def test_stable_context_fixes_the_measured_accumulation_failure_without_changing_native_values():
    features = np.full((1024, 4), 16.1, np.float32)
    before = features.copy()
    expected = np.full(4, 16.1, np.float32)
    assert np.abs(features.mean(0, dtype=np.float32) - expected).max() > 1e-5
    assert np.array_equal(method.stable_mean(features), expected)
    assert np.array_equal(method.stable_mean(np.asfortranarray(features)), expected)
    assert np.array_equal(features, before)


@pytest.mark.parametrize('bad', [np.full((4, 2), np.nan, np.float32), np.ones((4, 2), np.float64)])
def test_non_native_or_nonfinite_features_are_refused(bad):
    with pytest.raises(ValueError, match='native float32'):
        method.stable_mean(bad)


def test_an_isolated_peak_does_not_set_the_region_activation():
    scores = np.zeros((32, 32), np.float32)
    scores[16, 16] = 1.
    scores[2:4, 2:4] = .75
    assert scores.max() == 1 and method.activation(scores) == .75
    assert not patch.selected(scores, .75, 71, 49).any()
    assert patch.selected(scores, .5, 71, 49).any()


def test_diagonal_regions_and_equal_level_plateaus_are_processed_together():
    scores = np.zeros((32, 32), np.float32)
    for k in range(4):
        scores[k, k] = .625
    assert method.activation(scores) == review.activation(scores) == .625
    assert not patch.selected(scores, .625, 65, 33).any()
    assert method.activation(np.ones((2, 2), np.float32)) == 1.
    assert method.activation(np.zeros((2, 2), np.float32)) == 0.


def test_all_729_small_plateau_fields_match_an_independent_search():
    for values in itertools.product((0., .5, 1.), repeat=6):
        scores = np.array(values, np.float32).reshape(2, 3)
        level = method.activation(scores)
        assert level == review.activation(scores)
        for point in (0., .5, 1.):
            assert review.has_region(scores > point) == (level > point)


def test_native_region_decision_matches_activation_at_actual_float32_points():
    rng = np.random.default_rng(15)
    for _ in range(12):
        scores = rng.random((32, 32), dtype=np.float32)
        level = method.activation(scores)
        for point in (float(np.nextafter(np.float32(level), np.float32(0))), level,
                      float(np.nextafter(np.float32(level), np.float32(1)))):
            assert bool(patch.selected(scores, point, 101, 73).any()) == (level > point)


def records():
    return [{'group': str(k), 'role': 'point', 'label': 0, 'variant': variant,
             'maximum': float(np.float32(k / 60)), 'activation': float(np.float32(k / 120))}
            for k in range(60) for variant in ('original', 'ordinary_patch')]


def test_both_calibration_rules_use_the_same_source_group_budget():
    values = records()
    for kind in ('maximum', 'regions'):
        point = method.select_point(values, kind)
        assert point['maximum_flagged_calibration_groups'] == 3
        field = 'maximum' if kind == 'maximum' else 'activation'
        assert point['value'] == values[56 * 2][field]
        assert len({r['group'] for r in values if r[field] > point['value']}) == 3


def test_ai_diagnostic_missing_and_duplicate_variants_cannot_choose_the_point():
    values = records()
    values[0]['label'] = 1
    with pytest.raises(ValueError, match='Only genuine point-role'):
        method.select_point(values, 'regions')
    values = records()
    values[0]['role'] = 'diagnostic'
    with pytest.raises(ValueError, match='Only genuine point-role'):
        method.select_point(values, 'regions')
    with pytest.raises(ValueError, match='Both genuine variants'):
        method.select_point(records()[1:], 'regions')
    values = records()
    values.append(values[0].copy())
    with pytest.raises(ValueError, match='Duplicate calibration'):
        method.select_point(values, 'regions')


@pytest.mark.parametrize('bad', [np.array([[0., 1.], [np.nan, .5]]), np.full((2, 2), -1.), np.full((2, 2), 2.)])
def test_invalid_ranking_fields_never_create_an_activation(bad):
    with pytest.raises(ValueError, match='Finite complete ranking'):
        method.activation(bad)
