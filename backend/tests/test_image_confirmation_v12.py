"""Fresh inference must not alter fitting, source bindings or missingness."""
import math
from copy import deepcopy

import numpy as np
import pytest
import yaml

from scripts import image_supported_calibration_v11 as calibration
from scripts import prepare_image_confirmation_v12 as selection
from scripts import prepare_image_supported_v11 as source
from scripts.review_image_confirmation_v12 import nearest_quadrants, probability
from scripts.run_image_confirmation_v12 import (
    forward,
    license_binding,
    mask_summary,
    metrics,
    paired_bootstrap,
)


def fixture():
    producer = {'kind': 'authored_synthetic_fixture', 'implementation_sha256': 'a' * 64}
    producers = {'png': producer, 'jpeg': producer}
    geometry = {'width': 100, 'height': 80}
    row = {'id': 'new', 'group': 'new-source', 'role': 'confirmation', 'label': 0, 'format': 'jpeg',
        'source_sha256': 'b' * 64, 'input_sha256': 'c' * 64, 'rgb_sha256': 'd' * 64,
        'source_geometry': geometry, 'label_provenance_sha256': 'e' * 64, 'lineage': ['new-source'],
        'units': [{'id': 'primary', 'target': 'primary', 'box': [0, 0, 100, 80], 'label': 0},
            {'id': 'quadrant.0', 'target': 'region', 'box': [0, 0, 50, 40], 'label': 0}]}
    plan = {'branches': source.BRANCHES, 'files': [row]}
    measured = {'new': {'format': 'jpeg', 'input_sha256': row['input_sha256'], 'rgb_sha256': row['rgb_sha256'],
        'native_map_sha256': 'f' * 64, 'raw_values': {'primary': .4, 'quadrant.0': .7}}}
    document = calibration.manifest(plan, measured, producers, 'primary-regions')
    artifact = {'policy': document['policy']}
    return plan, measured, artifact


def test_forward_keeps_original_policy_and_source_coordinates_without_mutation():
    plan, measured, artifact = fixture()
    before = deepcopy((plan, measured, artifact))
    result = forward(plan, measured, artifact, 'primary-regions')
    assert result['policy'] == artifact['policy']
    assert result['files'][0]['role'] == 'confirmation'
    assert result['files'][0]['units'][1]['location']['width'] == 50
    assert (plan, measured, artifact) == before


@pytest.mark.parametrize('damage', ['input', 'rgb', 'role', 'format', 'box', 'policy'])
def test_fresh_forward_refuses_source_scope_or_calibration_changes(damage):
    plan, measured, artifact = fixture()
    if damage in ('input', 'rgb'):
        measured['new'][damage + '_sha256'] = '0' * 64
    elif damage == 'role':
        plan['files'][0]['role'] = 'fusion_fit'
    elif damage == 'format':
        plan['files'][0]['format'] = 'png'
    elif damage == 'box':
        plan['files'][0]['units'][1]['box'] = [90, 0, 50, 40]
    else:
        artifact['policy']['specialists'][0]['target'] = 'another.target'
    with pytest.raises(ValueError):
        forward(plan, measured, artifact, 'primary-regions')


def test_missing_measurement_stays_unavailable_and_cannot_produce_confidence():
    plan, _, artifact = fixture()
    result = forward(plan, {}, artifact, 'primary-regions')
    assert all(u['raw'] is None and u['status'] == 'unavailable' for u in result['files'][0]['units'])
    artifact.update(specialists={'image.jpeg.primary|jpeg': {'status': 'fitted'}}, combiners={})
    assert calibration.predict(result['files'][0], artifact) is None


def test_missing_flags_remain_misses_and_inconclusive_not_genuine_decisions():
    rows = [{'label': label, 'flag': None, 'assessed': False, 'probability': None} for label in (0, 1)]
    result = metrics(rows)
    assert result['planned'] == result['inconclusive'] == 2
    assert result['missed_ai_including_inconclusive'] == 1 and result['false_ai'] == 0
    assert result['brier_on_complete_balanced_pairs'] is None


def test_identical_methods_have_exactly_zero_paired_bootstrap_differences():
    rows = [{'group': g, 'label': label, 'flag': None if g == 'b' else bool(label),
        'assessed': g != 'b', 'probability': None} for g in ('a', 'b') for label in (0, 1)]
    result = paired_bootstrap({arm: deepcopy(rows) for arm in ('current', 'primary', 'primary-regions')})
    assert result['source_groups'] == 2 and result['draws'] == 5000
    for methods in result['comparisons'].values():
        assert all(v == {'candidate_minus_reference': 0., 'percentile95_interval': [0., 0.]} for v in methods.values())


@pytest.mark.parametrize('damage', ['duplicate', 'missing-original', 'missing-method-case'])
def test_bootstrap_cannot_change_the_paired_denominator(damage):
    rows = [{'group': g, 'label': label, 'flag': False} for g in ('a', 'b') for label in (0, 1)]
    outputs = {a: deepcopy(rows) for a in ('current', 'primary', 'primary-regions')}
    if damage == 'duplicate':
        outputs['current'].append(deepcopy(rows[0]))
    elif damage == 'missing-original':
        outputs['current'] = [r for r in outputs['current'] if r['label'] == 1]
    else:
        outputs['current'] = outputs['current'][:2]
    with pytest.raises(ValueError):
        paired_bootstrap(outputs)


def test_source_inventory_excludes_unscored_prior_selections(tmp_path):
    root = tmp_path / 'backend/data'
    ids = {str(i) for i in range(288)}
    for index in range(7):
        folder = root / str(index)
        folder.mkdir(parents=True)
        source.write(folder / 'tgif-plan.json', {'metadata_revision': selection.tgif.REVISION,
            'excluded_coco_ids': sorted(ids), 'groups': {'firefly': [{'source_id': 'unscored-draft'}]}})
    found, inventory = selection.exclusion_inventory(tmp_path, root / 'next')
    assert 'unscored-draft' in found and found >= ids and len(inventory) == 7


def test_bootstrap_retains_coverage_and_complete_pair_brier_differences():
    rows = [{'group': g, 'label': y, 'flag': bool(y), 'assessed': True, 'probability': .5}
        for g in ('a', 'b') for y in (0, 1)]
    outputs = {a: deepcopy(rows) for a in ('current', 'primary', 'primary-regions')}
    for row in outputs['primary-regions']:
        row['probability'] = float(row['label'])
    for row in outputs['current']:
        row.update(assessed=False, probability=None, flag=None)
    got = paired_bootstrap(outputs)['comparisons']
    assert got['current']['coverage_rate_difference']['percentile95_interval'] == [1., 1.]
    assert 'brier_difference' not in got['current']
    assert got['primary']['brier_difference'] == {'candidate_minus_reference': -.25, 'percentile95_interval': [-.25, -.25]}


@pytest.mark.parametrize('damage', ['none', 'unrelated', 'licence', 'asset', 'dependency', 'duplicate'])
def test_commercial_records_allow_unrelated_changes_but_bind_selected_assets(tmp_path, damage):
    rows = [{'id': k, 'licence': {'class': 'commercial', 'spdx': 'MIT'},
        'source': {'sha256': 'a' * 64}, 'depends_on': []} for k in ('fused.opensdi', 'fused.convnext_xxlarge')]
    prior = {'commercial_models': {r['id']: deepcopy(r) for r in rows}, 'commercial_license_record_sha256': 'b' * 64}
    if damage == 'unrelated':
        rows.append({'id': 'new.model-free-credential-validator'})
    elif damage == 'licence':
        rows[0]['licence']['class'] = 'noncommercial'
    elif damage == 'asset':
        rows[0]['source']['sha256'] = 'c' * 64
    elif damage == 'dependency':
        rows[0]['depends_on'] = ['other.backbone']
    elif damage == 'duplicate':
        rows.append(deepcopy(rows[0]))
    path = tmp_path / 'manifest.yaml'
    path.write_text(yaml.safe_dump({'models': rows}))
    if damage in ('none', 'unrelated'):
        result = license_binding(path, prior)
        assert result['historical_producer_manifest_sha256'] == 'b' * 64
        assert result['current_manifest_sha256'] != 'b' * 64
        assert result['selected_records_equal_original']
    else:
        with pytest.raises(ValueError):
            license_binding(path, prior)


@pytest.mark.parametrize('width,height', [(641, 479), (640, 480), (73, 91), (1281, 721)])
def test_independent_source_quadrant_indexing_matches_declared_nearest_resize(width, height):
    import cv2

    raw = np.random.default_rng(17).random((512, 512), dtype=np.float32)
    units = source.units(np.zeros((height, width), dtype=bool), [[0, 0, width // 2, height // 2]])
    expected = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)[:height // 2, :width // 2].mean(dtype=np.float64)
    assert nearest_quadrants(raw, width, height, units)['quadrant.0'] == expected


def test_current_mask_misses_cannot_be_hidden_by_file_presence_score():
    rows = [{'id': 'original', 'label': 0}, {'id': 'edited', 'label': 1}]
    responses = {'original': {'current_mask_metrics': {'predicted_positive_pixels': 4, 'flagged_area_fraction': .25}},
        'edited': {'current_mask_metrics': {'iou': 0., 'pixel_f1': 0.}}}
    got = mask_summary(rows, responses)
    assert got['ai_masks_iou_at_least_0_3'] == got['ai_mean_iou_including_misses'] == 0
    assert got['genuine_inputs_with_selected_pixels'] == 1
    assert got['candidate_changes_pixel_selection'] is False


def test_independent_probability_propagates_missing_measurements():
    assert probability({}, {'status': 'failed'}, {}, 'primary') is None


def test_independent_probability_matches_engine_without_negative_region_cancellation():
    plan, measured, artifact = fixture()
    head = {'status': 'fitted', 'center': [0.], 'scale': [1.], 'coefficients': [1.], 'intercept': -.5}
    combined = {**head, 'intercept': 0.}
    artifact.update(specialists={'image.jpeg.' + target + '|jpeg': deepcopy(head) for target in ('primary', 'region')},
        combiners={'image|jpeg-local-edit': {'response': combined, 'calibration': combined}})
    record = {**measured['new'], 'status': 'ok'}
    got = probability(plan['files'][0], record, artifact, 'primary-regions')
    document = forward(plan, measured, artifact, 'primary-regions')
    assert got == calibration.predict(document['files'][0], artifact)
    assert math.isclose(got, 1 / (1 + math.exp(-.2)), abs_tol=1e-15, rel_tol=0.)
    record['raw_values']['quadrant.0'] = None
    assert probability(plan['files'][0], record, artifact, 'primary-regions') is None


def test_exact_point_ties_stay_visible_without_becoming_flags():
    got = metrics([{'label': 1, 'flag': False, 'assessed': True, 'probability': .5, 'exact_point_tie': True}])
    assert got['exact_point_ties'] == got['missed_ai_including_inconclusive'] == 1
    assert got['ai_found'] == 0
