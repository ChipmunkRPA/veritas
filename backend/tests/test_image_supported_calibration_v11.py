"""Authored branch isolation and genuine-source operating-budget regressions."""
from copy import deepcopy

import pytest

from app.bench import specialists as engine
from scripts.image_supported_calibration_v11 import ARMS, manifest, matched_points, predict


def h(value):
    return engine.digest(value)


def fixture():
    files, responses = [], {}
    for role in engine.ROLES:
        for group_index in range(4):
            group = f'{role}-{group_index}'
            for label in (0, 1):
                for branch in ('png', 'jpeg'):
                    identity = f'{group}-{label}-{branch}'
                    local = [{'id': name, 'target': name, 'box': [0, 0, 64, 64], 'label': label}
                        for name in ('primary', 'integrity')]
                    for index, box in enumerate(([0, 0, 32, 32], [32, 0, 32, 32], [0, 32, 32, 32], [32, 32, 32, 32])):
                        local.append({'id': f'quadrant.{index}', 'target': 'region', 'box': box, 'label': label if index == 0 else 0})
                    file = {'id': identity, 'group': group, 'role': role, 'format': branch, 'label': label,
                        'input_sha256': h(identity), 'rgb_sha256': h('pixels-' + identity), 'source_sha256': h('source-' + identity),
                        'lineage': ['parent-' + group], 'source_geometry': {'width': 64, 'height': 64},
                        'label_provenance_sha256': h('mask-' + identity), 'units': local}
                    files.append(file)
                    responses[identity] = {'format': branch, 'input_sha256': file['input_sha256'], 'rgb_sha256': file['rgb_sha256'],
                        'native_map_sha256': h('native-' + identity), 'raw_values': {
                            unit['id']: .15 + .7 * unit['label'] + group_index * .005 for unit in local}}
    producer = {'kind': 'authored_synthetic_fixture', 'implementation_sha256': h('authored-only')}
    return {'branches': {'png': 'authored PNG model', 'jpeg': 'authored JPEG model'}, 'files': files}, responses, {'png': producer, 'jpeg': producer}


def calibration_rows(n=30, flags=2):
    rows = [{'id': f'g{i}', 'role': 'fusion_cal', 'format': 'png', 'label': 0,
        'group': f'g{i}', 'probability': (i + 1) / 100, 'current_assessed': True, 'current_flag': i < flags}
        for i in range(n)]
    return {arm: deepcopy(rows) for arm in ARMS}


def test_png_and_jpeg_have_independent_combination_calibrations():
    plan, responses, producers = fixture()
    document = manifest(plan, responses, producers, 'primary-integrity-regions')
    # Low support belongs only to this explicitly authored numerical fixture.
    artifact = engine.train(document, min_groups_per_class=3)
    assert set(artifact['combiners']) == {'image|png-local-edit', 'image|jpeg-local-edit'}
    changed = deepcopy(responses)
    for value in changed.values():
        if value['format'] == 'jpeg':
            value['raw_values'] = {key: 1 - raw for key, raw in value['raw_values'].items()}
    revised = engine.train(manifest(plan, changed, producers, 'primary-integrity-regions'), min_groups_per_class=3)
    assert revised['combiners']['image|png-local-edit'] == artifact['combiners']['image|png-local-edit']
    assert {k: v for k, v in revised['specialists'].items() if '.png.' in k} == {
        k: v for k, v in artifact['specialists'].items() if '.png.' in k}


def test_missing_jpeg_response_is_inconclusive_without_affecting_paired_png():
    plan, responses, producers = fixture()
    document = manifest(plan, responses, producers, 'primary-regions')
    artifact = engine.train(document, min_groups_per_class=3)
    file = next(f for f in plan['files'] if f['role'] == 'confirmation' and f['format'] == 'jpeg')
    missing = deepcopy(responses)
    del missing[file['id']]
    result = engine.evaluate(manifest(plan, missing, producers, 'primary-regions'), artifact)
    by_id = {row['id']: row for row in result['outcomes']}
    assert by_id[file['id']]['probability'] is None and by_id[file['id']]['status'] == 'inconclusive'
    png = file['id'].removesuffix('jpeg') + 'png'
    assert by_id[png]['status'] == 'ok' and by_id[png]['probability'] is not None


def test_same_family_views_do_not_become_independent_votes():
    plan, responses, producers = fixture()
    document = manifest(plan, responses, producers, 'primary-integrity-regions')
    assert all(len(file['required_families']) == 1 and len(file['units']) == 6 for file in document['files'])
    artifact = engine.train(document, min_groups_per_class=3)
    expected = {row['id']: row['probability'] for row in engine.evaluate(document, artifact)['outcomes']}
    assert all(predict(file, artifact) == expected[file['id']] for file in document['files'] if file['role'] == 'confirmation')


def test_point_matches_current_false_flag_budget_not_arbitrary_shared_cut():
    points = matched_points(calibration_rows(flags=2), 'png')
    assert all(point['upper_rank'] == 28 and point['threshold'] == .28 for point in points.values())
    assert all(point['calibration_flagged_groups'] == 2 for point in points.values())


def test_zero_reference_flags_uses_strict_maximum_and_keeps_ties():
    values = calibration_rows(flags=0)
    for rows in values.values():
        for row in rows:
            row['probability'] = .5
    points = matched_points(values, 'png')
    assert all(point['upper_rank'] == 30 and point['threshold'] == .5
        and point['calibration_flagged_groups'] == 0 for point in points.values())


def test_high_reference_false_rate_cannot_allow_an_unbounded_false_budget():
    points = matched_points(calibration_rows(flags=20), 'png')
    assert all(point['matched_false_flag_budget'] == 3 and point['upper_rank'] == 27 for point in points.values())


def test_missing_or_partial_group_is_excluded_from_every_arm_without_zero_imputation():
    values = calibration_rows()
    for row in values['primary-regions'][:6]:
        row['probability'] = None
    points = matched_points(values, 'png')
    assert all(point['common_complete_genuine_groups'] == 24 and point['status'] == 'insufficient_support'
        and point['threshold'] is None for point in points.values())


def test_reserved_ai_and_other_format_observations_cannot_select_the_point():
    values = calibration_rows()
    expected = matched_points(values, 'png')
    for rows in values.values():
        for index, fields in enumerate(({'role': 'confirmation'}, {'label': 1}, {'format': 'jpeg'})):
            rows.append({**rows[0], 'id': f'excluded{index}', 'group': f'excluded{index}', 'probability': 1., **fields})
    assert matched_points(values, 'png') == expected


def test_omitted_genuine_file_cannot_disappear_from_one_arm():
    values = calibration_rows()
    values['primary-regions'].pop()
    with pytest.raises(ValueError, match='including missing evidence'):
        matched_points(values, 'png')


def test_unknown_reference_is_not_substituted_with_false():
    values = calibration_rows()
    for rows in values.values():
        rows[0].update(current_assessed=False, current_flag=False)
    with pytest.raises(ValueError, match='never an imputed'):
        matched_points(values, 'png')
