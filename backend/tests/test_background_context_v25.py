"""Known spatial/temporal interventions and source-role refusal contracts."""
import copy

import numpy as np
import pytest

from scripts import background_context_v25 as method
from scripts import review_background_context_v25 as independent
from scripts import run_background_context_v25 as run


def window():
    before, middle, after = [1, 5], [9, 13, 17], [21, 25]
    frames = sorted({i + k for i in middle for k in (-1, 0, 1)})
    context = sorted({i + k for i in before + after for k in (-1, 0, 1)})
    return {'before_centers': before, 'middle_centers': middle, 'after_centers': after,
        'location': {'sampled_frames': frames, 'context_frames': context}}


def observations(mode):
    unit = window()
    frames = sorted(set(unit['location']['sampled_frames'] + unit['location']['context_frames']))
    values, boxes = {}, {}
    for i in frames:
        rgb = np.full((480, 640, 3), 40, np.uint8)
        if i in unit['location']['sampled_frames']:
            if mode == 'scene':
                rgb[:] = 200
            elif mode == 'face':
                rgb[160:320, 240:400] = 200
        values[i] = method.describe(rgb)
        boxes[i] = [240, 160, 160, 160]
    return unit, values, boxes


def test_block_statistics_match_exact_integer_integrals_on_nonuniform_rgb():
    y, x = np.indices((480, 640))
    rgb = np.stack([x % 256, y % 256, (x * 7 + y * 13) % 256], -1).astype(np.uint8)
    expected = rgb[:40, :40].sum(axis=(0, 1), dtype=np.uint64) / (1600 * 255)
    assert np.allclose(method.describe(rgb)[0, 0], expected, atol=1e-15, rtol=0)
    assert np.allclose(method.describe(rgb), independent.describe(rgb), atol=1e-15, rtol=0)


@pytest.mark.parametrize('mode,expected', [('stable', 0.), ('face', 0.), ('scene', 160 / 255)])
def test_background_distinguishes_face_only_intervention_from_scene_change(mode, expected):
    unit, values, boxes = observations(mode)
    got = method.measure(unit, values, boxes)
    other = independent.measure(unit, values, boxes)
    assert got['raw'] == pytest.approx(expected, abs=1e-12)
    assert got['raw'] == pytest.approx(other['raw'], abs=1e-12)
    assert got['background_cells'] == other['background_cells'] == 156
    assert method.adjusted(.3, got) == pytest.approx(.3 - 2 * expected)


def test_source_motion_uses_common_exclusion_mask_and_keeps_missing_as_missing():
    unit, values, boxes = observations('face')
    boxes[next(iter(boxes))] = [0, 0, 200, 100]
    assert method.measure(unit, values, boxes)['background_cells'] == independent.measure(unit, values, boxes)['background_cells']
    del boxes[next(iter(boxes))]
    got = method.measure(unit, values, boxes)
    assert got['raw'] is None and method.adjusted(.3, got) is None


def test_insufficient_background_cannot_become_a_zero_change_observation():
    unit, values, boxes = observations('scene')
    boxes = {i: [0, 0, 640, 480] for i in boxes}
    assert method.measure(unit, values, boxes)['raw'] is None
    assert independent.measure(unit, values, boxes) is None


@pytest.mark.parametrize('box', [[0, 0, 0, 20], [0, float('nan'), 20, 20], [700, 0, 20, 20]])
def test_invalid_native_geometry_is_refused(box):
    with pytest.raises(ValueError):
        method.background_mask([box])


def genuine():
    return [{'id': str(i), 'group': str(i // 2), 'label': 0, 'role': 'fusion_cal',
             'baseline': i / 100, 'candidate': -i / 100} for i in range(60)]


def test_point_uses_source_group_maxima_and_preserves_matched_flag_budget():
    rows = genuine()
    point = method.fit_point(rows, .55)
    assert point['assessed_groups'] == 30 and point['baseline_flagged_groups'] == 2
    assert point['value'] == -.04 and point['candidate_flagged_groups'] == 2
    rows[0]['candidate'] = None
    point = method.fit_point(rows, .55)
    assert point['assessed_groups'] == 29 and point['unassessed_groups'] == 1


@pytest.mark.parametrize('change', ['ai', 'diagnostic', 'duplicate', 'nonfinite', 'too_few'])
def test_unqualified_cases_cannot_select_a_point(change):
    rows = genuine()
    if change == 'ai':
        rows[0]['label'] = 1
    elif change == 'diagnostic':
        rows[0]['role'] = 'confirmation'
    elif change == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif change == 'nonfinite':
        rows[0]['candidate'] = float('nan')
    else:
        rows = rows[:48]
    with pytest.raises(ValueError):
        method.fit_point(rows, .55)


def test_source_intervals_preserve_unknown_checks_without_original_verdict():
    row = {'id': 'case', 'group': 'source', 'family': 'inswapper', 'label': 1,
           'truth': {'time_ns': [2_000_000_000, 4_000_000_000]},
           'units': [{'baseline': .3, 'candidate': None,
                      'location': {'start': 2_000_000_000, 'end': 4_000_000_000}}]}
    base = run.score_file(row, 'current-fixed', .1)
    assert base['best_iou'] == 1 and base['flag'] is True
    missing = run.score_file(row, 'context', -.1)
    assert missing['flag'] is None and missing['best_iou'] == 0 and missing['originality_verdict'] is None


def test_changed_prospective_policy_cannot_use_an_existing_freeze(tmp_path, monkeypatch):
    lock = {'policy': {'coefficient': 2.}, 'code_sha256': {'source': 'frozen'}}
    run.write(tmp_path / 'freeze.json', lock)
    monkeypatch.setattr(run, 'identities', lambda: {**lock, 'policy': {'coefficient': 1.}})
    with pytest.raises(ValueError, match='Prospectively frozen'):
        run.check(tmp_path)


def test_fitted_points_cannot_be_replaced_between_calibration_and_diagnostics(tmp_path):
    for name, value in (('points.json', {'context': {'value': .1}}), ('freeze.json', {'method': 'fixed'})):
        run.write(tmp_path / name, value)
    (tmp_path / 'point.jsonl').write_text('original-point-cases\n')
    run.write(tmp_path / 'point-fit-complete.json', {'points_sha256': run.sha(tmp_path / 'points.json'),
        'journal_sha256': run.sha(tmp_path / 'point.jsonl'), 'freeze_sha256': run.sha(tmp_path / 'freeze.json')})
    run.check_points(tmp_path)
    (tmp_path / 'points.json').write_text('{"context":{"value":0.05}}')
    with pytest.raises(ValueError, match='fitted points'):
        run.check_points(tmp_path)
