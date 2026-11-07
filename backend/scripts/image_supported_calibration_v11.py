"""Per-format specialist/combination manifests and matched genuine-source points.

These research helpers load no models and change no serving score. PNG and JPEG
use separate dependence families, so their combination fitting and calibration
remain separate. Do not call a partial cached preview fresh qualification.
"""
from __future__ import annotations

import math
from collections import defaultdict

from app.bench import specialists as engine

ARMS = {'primary': (False, False), 'primary-integrity': (True, False),
    'primary-regions': (False, True), 'primary-integrity-regions': (True, True)}
FORMATS = ('png', 'jpeg')
MINIMUM = 25
MAXIMUM_CALIBRATION_FALSE_FRACTION = .1


def family(branch):
    engine.check(branch in FORMATS, 'Explicit supported format required')
    return branch + '-local-edit'


def manifest(plan, responses, producers, arm):
    engine.check(arm in ARMS and set(producers) == set(FORMATS), 'Every supported branch needs a bound producer')
    include_integrity, include_regions = ARMS[arm]
    targets = ['primary'] + (['integrity'] if include_integrity else []) + (['region'] if include_regions else [])
    route = engine.digest({'routing': 'PNG/JPEG original byte signature before model responses',
        'branches': plan['branches'], 'source_plan_method': 'supported-image-v11'})
    specs = [{'id': f'image.{branch}.{target}', 'modality': 'image', 'family': family(branch),
        'target': 'image.local_editing_presence' if target != 'region' else 'image.region_local_editing_presence',
        'contexts': [branch], 'routing_sha256': route, 'higher_means_ai': True, 'raw_range': [0., 1.],
        'producer': producers[branch]} for branch in FORMATS for target in targets]
    files = []
    for original in plan['files']:
        branch = original['format']
        measured = responses.get(original['id'])
        if measured is not None:
            engine.check(measured['format'] == branch and measured['input_sha256'] == original['input_sha256']
                and measured['rgb_sha256'] == original['rgb_sha256'], 'Model response is not its source format/input')
        geometry = original['source_geometry']
        units = []
        for unit in original['units']:
            if unit['target'] not in targets:
                continue
            x, y, width, height = unit['box']
            raw = measured['raw_values'][unit['id']] if measured else None
            if measured:
                engine.check(engine.finite(raw) and 0 <= raw <= 1, 'Original finite measured scalar required')
            units.append({'id': unit['id'], 'specialist': f'image.{branch}.{unit["target"]}',
                'context': branch, 'routing_sha256': route,
                'context_evidence_sha256': engine.digest({'input_sha256': original['input_sha256'], 'format': branch}),
                'measurement_sha256': engine.digest({'native_map_sha256': measured['native_map_sha256'],
                    'source_box': unit['box'], 'raw': raw, 'readout': unit['id']}) if measured else engine.digest(unit),
                'location': {'kind': 'pixels', 'x': x, 'y': y, 'width': width, 'height': height,
                    'source_width': geometry['width'], 'source_height': geometry['height']},
                'raw': raw, 'status': 'ok' if measured else 'unavailable', 'label': unit['label'],
                'label_scope': 'local_unit', 'label_provenance_sha256': engine.digest({
                    'source_mask': original['label_provenance_sha256'], 'box': unit['box'], 'label': unit['label']}),
                **({} if measured else {'reason': 'Supported-branch response unavailable; no originality verdict.'})})
        files.append({**{key: original[key] for key in ('id', 'group', 'role', 'source_sha256', 'input_sha256',
            'lineage', 'label', 'source_geometry', 'label_provenance_sha256')}, 'modality': 'image',
            'label_scope': 'planned_units', 'required_families': [family(branch)], 'units': units})
    result = {'schema': engine.SCHEMA, 'policy': {'specialists': specs}, 'files': files}
    engine.audit(result)
    return result


def predict(file, artifact):
    values, combined = engine._features(file, artifact), artifact['combiners'].get(engine.pattern(file))
    if (values is None or not combined or combined['response']['status'] != 'fitted'
            or combined.get('calibration', {}).get('status') != 'fitted'):
        return None
    eta = engine._logit(combined['calibration'], [engine._logit(combined['response'], values)])
    return 1 / (1 + math.exp(-max(-700., min(700., eta))))


def matched_points(rows_by_arm, branch):
    """Same complete genuine calibration groups/reference budget for all arms.

    Only fusion_cal genuine observations of the requested format enter selection.
    Missing responses are recorded, never imputed. Match the current region
    selector's genuine source-group flags, capped at10% of common source groups.
    This is an internal operating comparison, not a public or guaranteed rate.
    """
    engine.check(branch in FORMATS and set(rows_by_arm) == set(ARMS), 'Exact supported format and four arms required')
    selected = {arm: {row['id']: row for row in rows if row['role'] == 'fusion_cal'
        and row['label'] == 0 and row['format'] == branch} for arm, rows in rows_by_arm.items()}
    first = selected['primary']
    for arm, rows in rows_by_arm.items():
        filtered = [row for row in rows if row['role'] == 'fusion_cal' and row['label'] == 0 and row['format'] == branch]
        engine.check(len(filtered) == len(selected[arm]) and set(selected[arm]) == set(first),
            'All planned genuine calibration files, including missing evidence, must remain in every arm')
        for key, row in selected[arm].items():
            expected = first[key]
            engine.check(row['group'] == expected['group'] and row['current_flag'] == expected['current_flag']
                and row['current_assessed'] == expected['current_assessed'], 'Reference source/flag differs between arms')
            engine.check(type(row['current_assessed']) is bool and (type(row['current_flag']) is bool
                if row['current_assessed'] else row['current_flag'] is None), 'Explicit reference status, never an imputed false flag')
            value = row['probability']
            engine.check(value is None or engine.finite(value) and 0 <= value <= 1, 'Invalid calibrated response')
    groups = defaultdict(list)
    for key, row in first.items():
        groups[row['group']].append(key)
    complete = {group: keys for group, keys in groups.items() if all(
        first[key]['current_assessed'] and all(selected[arm][key]['probability'] is not None for arm in ARMS)
        for key in keys)}
    count = len(complete)
    current_flags = sum(any(first[key]['current_flag'] for key in keys) for keys in complete.values())
    budget = min(current_flags, math.floor(count * MAXIMUM_CALIBRATION_FALSE_FRACTION))
    points = {}
    for arm in ARMS:
        point = {'status': 'insufficient_support', 'format': branch, 'threshold': None,
            'planned_genuine_files': len(first), 'planned_genuine_groups': len(groups),
            'common_complete_genuine_groups': count, 'excluded_incomplete_genuine_groups': len(groups) - count,
            'unassessed_genuine_files': sum(row['probability'] is None or not row['current_assessed'] for row in selected[arm].values()),
            'reference_flagged_common_genuine_groups': current_flags, 'matched_false_flag_budget': budget,
            'comparison': 'strictly_greater_than_genuine_group_maximum_order_statistic',
            'minimum_complete_groups': MINIMUM, 'source_group_cap': MAXIMUM_CALIBRATION_FALSE_FRACTION,
            'scope': 'Common complete fusion_cal genuine source groups only; no missing-score imputation or reserved-role selection.',
            'serving_admitted': False}
        if count >= MINIMUM:
            maxima = sorted(max(selected[arm][key]['probability'] for key in keys) for keys in complete.values())
            rank = count - budget
            point.update(status='fitted', threshold=maxima[rank - 1], upper_rank=rank,
                calibration_flagged_groups=sum(value > maxima[rank - 1] for value in maxima))
        points[arm] = point
    return points
