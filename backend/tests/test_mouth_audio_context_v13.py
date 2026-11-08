"""Authored correspondence and source-contract tests, not detector accuracy."""
import copy
import json

import numpy as np
import pytest

from scripts import mouth_audio_context_v13 as method
from scripts import review_mouth_audio_context_v13 as reference
from scripts import run_mouth_audio_context_v13 as driver


def window():
    return {'centers': [0, 5, 10, 20, 30, 40, 50, 59]}


def test_exact_sample_blocks_have_independent_scalar_reference():
    pcm = np.random.default_rng(101).uniform(-.7, .7, 60 * 640).astype(np.float32)
    actual = method.envelope(pcm)
    expected = np.array([np.log(max(np.sqrt(sum(float(v) ** 2 for v in pcm[i:i + 640]) / 640), 1e-8))
                         for i in range(0, len(pcm), 640)])
    assert np.max(np.abs(actual - expected)) < 1e-12


def test_known_audio_shift_has_correct_sign_without_wrapping():
    audio = np.random.default_rng(13).normal(size=60)
    mouth = np.zeros(60)
    mouth[:-8] = audio[8:]
    values, supports = method.features(window(), audio, mouth)
    assert values[4] == pytest.approx(1.) and values[5] == 1.
    assert values[14] == pytest.approx(1.) and values[15] == 1.
    assert all(8 <= i < 52 for p in supports for i in p)
    expected, positions = reference.features(window(), audio.tolist(), mouth.tolist())
    assert positions == supports
    assert np.allclose(values, expected, rtol=0, atol=1e-12)


def test_constant_trace_is_explicitly_degenerate_not_sync_evidence():
    values, _ = method.features(window(), np.ones(60), np.ones(60))
    assert values == [0.] * 20


def test_edge_changes_never_wrap_into_the_common_lag_support():
    audio = np.random.default_rng(17).normal(size=60)
    mouth = np.random.default_rng(18).normal(size=60)
    values, supports = method.features(window(), audio, mouth)
    changed = mouth.copy()
    changed[:8], changed[-8:] = 1e6, -1e6
    again, same = method.features(window(), audio, changed)
    assert values == again and supports == same


@pytest.mark.parametrize('fault', ['short_pcm', 'nonfinite_pcm', 'double_pcm', 'missing_aperture', 'duplicate_centers', 'short_support'])
def test_missing_or_invalid_measurements_cannot_supply_a_vector(fault):
    pcm = np.zeros(60 * 640, np.float32)
    audio, mouth, w = np.ones(60), np.ones(60), window()
    if fault == 'short_pcm':
        pcm = pcm[:-1]
    elif fault == 'nonfinite_pcm':
        pcm[0] = np.nan
    elif fault == 'double_pcm':
        pcm = pcm.astype(np.float64)
    elif fault == 'missing_aperture':
        mouth = mouth[:-1]
    elif fault == 'duplicate_centers':
        w['centers'][1] = w['centers'][0]
    else:
        w['centers'] = [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        method.envelope(pcm)
        method.features(w, audio, mouth)


def test_native_aperture_uses_pixel_aspect_and_complete_order():
    points = [[0., 0., 0.], [0., .1, 0.], [0., 0., 0.], [.2, 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    trace = [{'frame_index': i, 'points': points} for i in range(60)]
    assert np.all(method.apertures(trace, 200, 100, 60) == .25)
    trace[0]['frame_index'] = 1
    with pytest.raises(ValueError):
        method.apertures(trace, 200, 100, 60)


def test_both_complete_scoring_heads_are_bound_before_diagnostics(tmp_path):
    names = ('candidate-head.json', 'control-head.json', 'fit.jsonl', 'calibration.jsonl', 'freeze.json')
    for name in names:
        (tmp_path / name).write_text('{}\n')
    (tmp_path / 'fit-complete.json').write_text(json.dumps({'sha256': {n: driver.sha(tmp_path / n) for n in names}}))
    driver.check_fit(tmp_path)
    (tmp_path / 'candidate-head.json').write_text('{"changed_point": 0}\n')
    with pytest.raises(ValueError):
        driver.check_fit(tmp_path)


@pytest.fixture(scope='module')
def fitted():
    rows = []
    for actor in range(16):
        for condition in sorted(method.previous.AI_CONDITIONS):
            for label in (0, 1):
                x = np.random.default_rng(actor * 17 + label).normal(size=69)
                x[49] = 2 * label - 1.
                rows.append({'group': str(actor), 'role': 'fit', 'condition': condition, 'label': label, 'features': x.tolist()})
    model, head = method.fit(rows)
    return model, head, rows


def test_new_audio_feature_splits_export_against_native_and_scalar_tree(fitted):
    model, head, _ = fitted
    x = np.random.default_rng(40).normal(size=(21, 69))
    expected = model.decision_function(x)
    assert np.allclose(method.margins(x, json.loads(json.dumps(head))), expected, rtol=0, atol=1e-10)
    assert np.allclose([reference.margin(r, head) for r in x], expected, rtol=0, atol=1e-10)
    assert any(n['feature_idx'] >= 49 for tree in head['trees'] for n in tree if not n['is_leaf'])
    assert not head['deployment_eligible']


def test_point_actor_cannot_fit_the_candidate(fitted):
    rows = copy.deepcopy(fitted[2])
    rows[0]['role'] = 'point'
    with pytest.raises(ValueError):
        method.fit(rows)


@pytest.mark.parametrize('fault', ['omitted_tree', 'cycle', 'unknown_feature', 'nonfinite', 'feature_order'])
def test_altered_tree_never_becomes_an_ai_margin(fitted, fault):
    head = copy.deepcopy(fitted[1])
    if fault == 'omitted_tree':
        head['trees'].pop()
    elif fault == 'cycle':
        head['trees'][0][0].update(is_leaf=0, left=0, right=0)
    elif fault == 'unknown_feature':
        head['trees'][0][0].update(is_leaf=0, feature_idx=69)
    elif fault == 'nonfinite':
        head['base_margin'] = float('nan')
    else:
        head['feature_names'].reverse()
    with pytest.raises(ValueError):
        method.margins(np.zeros((1, 69)), head)
