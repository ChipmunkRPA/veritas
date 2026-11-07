"""Research-only scoped consistency features for all four media.

Keep v2 byte-for-byte intact for frozen studies. This alternative preserves an
AI indication in one area when another area looks genuine. Within the exact
same target and source location it additionally records consistency of distinct
views from one dependence family. These are learned features, not independent
votes or a product of probabilities. Nothing in serving imports this module.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import math
from collections import defaultdict
from pathlib import Path

from app.bench import specialists as legacy

SCHEMA, ROLES, MEDIA = legacy.SCHEMA, legacy.ROLES, legacy.MEDIA
check, digest, finite = legacy.check, legacy.digest, legacy.finite
plan_digest, _fit, _logit = legacy.plan_digest, legacy._fit, legacy._logit
POOLING = 'positive_max_and_coincident_distinct_view_min_per_dependency_family'


def audit(manifest):
    result = legacy.audit(manifest)
    specs = {s['id']: s for s in manifest['policy']['specialists']}
    for file in manifest['files']:
        seen = set()
        for unit in file['units']:
            spec = specs[unit['specialist']]
            key = (spec['family'], spec['target'], digest(unit['location']), unit['specialist'])
            check(key not in seen, 'Duplicate specialist at the same source scope cannot manufacture consistency')
            seen.add(key)
    return result


def _features(file, artifact):
    specs = {s['id']: s for s in artifact['policy']['specialists']}
    families = defaultdict(lambda: defaultdict(list))
    for unit in file['units']:
        spec = specs[unit['specialist']]
        if spec['family'] not in file['required_families']:
            continue
        model = artifact['specialists'][unit['specialist'] + '|' + unit['context']]
        if unit['status'] != 'ok' or model['status'] != 'fitted':
            return None
        scope = (spec['target'], digest(unit['location']))
        families[spec['family']][scope].append(_logit(model, [unit['raw']]))
    if set(families) != set(file['required_families']):
        return None
    result = []
    for family in sorted(file['required_families']):
        units = list(families[family].values())
        positive = max(0., max(max(values) for values in units))
        # No corroborating second view yields a measured consistency feature of
        # zero. A missing planned response returned None above, never zero.
        consistent = max([0.] + [max(0., min(values)) for values in units if len(values) >= 2])
        result.extend((positive, consistent))
    return result


def pattern(file):
    return legacy.pattern(file)


def train(manifest, *, min_groups_per_class=25, l2=.001):
    audit(manifest)
    check(type(min_groups_per_class) is int and min_groups_per_class >= 2 and finite(l2) and l2 > 0, 'Explicit support and regularization required')
    training = [f for f in manifest['files'] if f['role'] != 'confirmation']
    artifact = {'schema': 'chipmunk.specialist-calibration-scoped/3', 'policy': copy.deepcopy(manifest['policy']),
        'feature_pooling': POOLING,
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'base_implementation_sha256': hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest(),
        'feature_order_within_family': ['positive_max_any_scope', 'positive_min_distinct_views_same_scope'],
        'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy')},
        'plan_sha256': plan_digest(manifest), 'training_sha256': digest(training),
        'minimum_groups_per_class': min_groups_per_class, 'l2': l2, 'specialists': {}, 'combiners': {},
        'serving_admitted': False, 'reference_prior': .5}
    for spec in manifest['policy']['specialists']:
        for context in spec['contexts']:
            selected = [(f, u) for f in training if f['role'] == 'specialist_cal'
                for u in f['units'] if u['specialist'] == spec['id'] and u['context'] == context
                and u['status'] == 'ok' and u.get('label') is not None]
            artifact['specialists'][spec['id'] + '|' + context] = _fit([u['raw'] for _, u in selected],
                [u['label'] for _, u in selected], [f['group'] for f, _ in selected], minimum=min_groups_per_class, l2=l2)
    for key in sorted({pattern(f) for f in training if f['role'] == 'fusion_fit'}):
        selected = [(f, _features(f, artifact)) for f in training if f['role'] == 'fusion_fit' and pattern(f) == key]
        usable = [(f, x) for f, x in selected if x is not None]
        response = _fit([x for _, x in usable], [f['label'] for f, _ in usable], [f['group'] for f, _ in usable], minimum=min_groups_per_class, l2=l2)
        entry = {'response': response, 'fitting_files': len(selected), 'fitting_unassessed': len(selected) - len(usable)}
        if response['status'] == 'fitted':
            calibration = [(f, _features(f, artifact)) for f in training if f['role'] == 'fusion_cal' and pattern(f) == key]
            available = [(f, x) for f, x in calibration if x is not None]
            entry['calibration'] = _fit([_logit(response, x) for _, x in available], [f['label'] for f, _ in available],
                [f['group'] for f, _ in available], minimum=min_groups_per_class, l2=l2)
            entry.update(calibration_files=len(calibration), calibration_unassessed=len(calibration) - len(available))
        artifact['combiners'][key] = entry
    artifact['sha256'] = digest(artifact)
    return artifact


def evaluate(manifest, artifact):
    audit(manifest)
    check(artifact.get('schema') == 'chipmunk.specialist-calibration-scoped/3'
        and artifact.get('feature_pooling') == POOLING
        and artifact.get('serving_admitted') is False
        and artifact.get('sha256') == digest({k: v for k, v in artifact.items() if k != 'sha256'}), 'Exact fitted artifact required')
    check(artifact['implementation_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        and artifact['base_implementation_sha256'] == hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest()
        and artifact['packages'] == {n: importlib.metadata.version(n) for n in ('numpy', 'scipy')}, 'Changed calibration implementation/runtime')
    check(artifact['plan_sha256'] == plan_digest(manifest)
        and artifact['policy'] == manifest['policy']
        and artifact['training_sha256'] == digest([f for f in manifest['files'] if f['role'] != 'confirmation']), 'Changed frozen plan or fitting/calibration observations')
    outcomes, local = [], []
    for file in manifest['files']:
        if file['role'] != 'confirmation':
            continue
        values, combined = _features(file, artifact), artifact['combiners'].get(pattern(file))
        row = {'id': file['id'], 'group': file['group'], 'modality': file['modality'], 'label': file['label'],
            'status': 'inconclusive', 'probability': None, 'scope': 'planned_evidence_units', 'reference_prior': .5}
        if (values is not None and combined and combined['response']['status'] == 'fitted'
                and combined.get('calibration', {}).get('status') == 'fitted'):
            eta = _logit(combined['calibration'], [_logit(combined['response'], values)])
            row.update(status='ok', probability=1 / (1 + math.exp(-max(-700., min(700., eta)))))
        else:
            row['reason'] = 'Incomplete evidence or unsupported specialist/context/combination; no originality verdict.'
        outcomes.append(row)
        for unit in file['units']:
            head = artifact['specialists'][unit['specialist'] + '|' + unit['context']]
            current = {'file_id': file['id'], 'unit_id': unit['id'], 'location': copy.deepcopy(unit['location']),
                'specialist': unit['specialist'], 'context': unit['context'], 'reference_prior': .5,
                'status': unit['status'], 'probability': None}
            if unit['status'] == 'ok' and head['status'] == 'fitted':
                eta = _logit(head, [unit['raw']])
                current['probability'] = 1 / (1 + math.exp(-max(-700., min(700., eta))))
            else:
                current.update(status='unavailable', reason=unit.get('reason') or 'Insufficient independent local calibration support.')
            local.append(current)
    summaries = {}
    for modality in MEDIA:
        selected = [r for r in outcomes if r['modality'] == modality]
        scored = [r for r in selected if r['probability'] is not None]
        by_class = {}
        for label in (0, 1):
            groups = defaultdict(list)
            for row in scored:
                if row['label'] == label:
                    groups[row['group']].append((row['probability'] - label) ** 2)
            by_class[label] = sum(sum(v) / len(v) for v in groups.values()) / len(groups) if groups else None
        summaries[modality] = {'planned_files': len(selected), 'scored_files': len(scored),
            'inconclusive_files': len(selected) - len(scored), 'genuine_files': sum(r['label'] == 0 for r in selected),
            'ai_files': sum(r['label'] == 1 for r in selected),
            'source_balanced_brier_on_scored_cases': (by_class[0] + by_class[1]) / 2
                if all(v is not None for v in by_class.values()) else None}
    return {'schema': 'chipmunk.specialist-confirmation-scoped/3', 'artifact_sha256': artifact['sha256'],
        'outcomes': outcomes, 'unit_outcomes': local, 'summaries': summaries, 'serving_admitted': False,
        'limits': 'Research scores under a50:50 reference prior; no validated customer accuracy or scope beyond planned units.'}
