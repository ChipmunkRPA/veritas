"""Prospective error-budget and independent geometry checks, without models."""
import copy

import numpy as np
import pytest

from scripts.image_patch_drift_v8 import median3, predict_regions, scores
from scripts.localization_benchmark import policy
from scripts.review_image_patch_drift_v8 import (
    display_reference,
    gates,
    median_reference,
    pixel_reference,
    readout_reference,
)
from scripts.run_image_patch_drift_v8 import choose, operating_point, variant


def test_genuine_only_operating_point_is_strict_order_statistic():
    assert operating_point(list(range(1, 61))) == 59
    assert operating_point([10.] * 60) == 10
    assert operating_point(list(range(1, 49)) + [None] * 12) == 47
    for invalid in ([1.] * 47 + [None] * 13, [1.] * 59, [0.] * 60, [float('nan')] * 60):
        with pytest.raises(ValueError):
            operating_point(invalid)


def test_disjoint_source_roles_and_fixed_family_denominators():
    originals = [{'id': f'{f}-{i}', 'source_id': f'{f}-{i}', 'family': f, 'label': 0}
        for f in ('firefly', 'sd2', 'flux-fill') for i in range(24)]
    development = [{'id': f'd-{f}-{i}-{label}', 'source_id': f'd-{f}-{i}', 'family': f, 'label': label}
        for f in ('firefly', 'sd2', 'flux-fill') for i in range(16) for label in (0, 1)]
    chosen = choose(originals, development)
    assert len(chosen) == 60 and choose(list(reversed(originals)), development) == chosen
    changed = copy.deepcopy(development)
    changed[0]['source_id'] = chosen[0]['source_id']
    with pytest.raises(ValueError, match='overlap'):
        choose(originals, changed)
    with pytest.raises(ValueError, match='48-pair'):
        choose(originals, development[:-1])


def test_same_method_reference_and_nonrectangular_source_projection():
    rng = np.random.default_rng(80)
    value = rng.random((32, 32), dtype=np.float32) * .003
    np.testing.assert_array_equal(median3(value), median_reference(value))
    for arm in ('raw', 'within_image'):
        np.testing.assert_array_equal(scores(value, arm), readout_reference(value, arm))
    floating = np.zeros((17, 23), np.float32)
    floating[2:8, 4:9] = .9
    floating[6:13, 7:11] = .9
    floating[10:16, 15:22] = .95
    reliable = np.ones_like(floating, bool)
    reliable[3:5, 5:7] = False
    truth = rng.random((129, 191)) > .8
    actual, arrays, _ = predict_regions(floating, reliable, .5, 191, 129, truth)
    encoded, selected = display_reference(floating, reliable, .5, policy(), 191, 129)
    np.testing.assert_array_equal(arrays['display_grid'], encoded)
    np.testing.assert_array_equal(arrays['selected_grid'], selected)
    assert actual == pixel_reference(selected, truth)


def test_qualification_cannot_trade_away_error_budget_or_coverage():
    b = {'overall': {'overlapping_edits': 12, 'mean_edit_iou': .1, 'genuine_highlights': 2},
        'families': {f: {'overlapping_edits': 4} for f in ('firefly', 'sd2', 'flux-fill')}}
    c = {'overall': {'overlapping_edits': 16, 'mean_edit_iou': .11, 'genuine_highlights': 3},
        'families': {f: {'overlapping_edits': 5} for f in ('firefly', 'sd2', 'flux-fill')}}
    assert all(gates(b, c, True, True).values())
    for key, bad in [('overlapping_edits', 15), ('mean_edit_iou', .099), ('genuine_highlights', 4)]:
        changed = copy.deepcopy(c)
        changed['overall'][key] = bad
        assert not all(gates(b, changed, True, True).values())
    changed = copy.deepcopy(c)
    changed['families']['sd2']['overlapping_edits'] = 3
    assert not all(gates(b, changed, True, True).values())
    assert not all(gates(b, c, False, True).values())
    assert not all(gates(b, c, True, False).values())


def test_whole_jpeg_variant_has_deterministic_bytes_and_source_geometry():
    rng = np.random.default_rng(9)
    rgb = rng.integers(0, 256, (67, 91, 3), dtype=np.uint8)
    assert variant(rgb, b'original', 'native')[0] == b'original'
    first, pixels = variant(rgb, b'original', 'jpeg75')
    assert first == variant(rgb, b'original', 'jpeg75')[0]
    assert pixels.shape == rgb.shape and not np.array_equal(rgb, pixels)
    with pytest.raises(ValueError):
        variant(rgb, b'original', 'cropped')
