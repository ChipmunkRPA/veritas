"""Research-only local-evidence calibration and dependent-evidence fusion.

No serving imports this module. A successful fit is not detection validation.
Four source-disjoint roles separate specialist calibration, fusion fitting,
fusion calibration and final confirmation. Outputs use a declared50:50 reference
prior; they are not a customer's probability of correctness or a court opinion.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = 'chipmunk.specialist-evaluation/1'
ROLES = ('specialist_cal', 'fusion_fit', 'fusion_cal', 'confirmation')
MEDIA = ('image', 'video', 'audio', 'text')
STATUSES = ('ok', 'failed', 'skipped', 'not_applicable', 'unavailable')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def token(value):
    return isinstance(value, str) and bool(value) and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_.-' for c in value)


def location(modality, span):
    check(isinstance(span, dict), 'Local evidence needs an explicit source location')
    if modality == 'image':
        keys = ('x', 'y', 'width', 'height', 'source_width', 'source_height')
        check(span.get('kind') == 'pixels' and all(type(span.get(k)) is int for k in keys), 'Image regions use integer stored-source pixels')
        check(span['x'] >= 0 and span['y'] >= 0 and min(span['width'], span['height']) > 0
            and span['x'] + span['width'] <= span['source_width']
            and span['y'] + span['height'] <= span['source_height'], 'Image region outside source')
    else:
        kind, unit = {'video': ('time', 'nanoseconds'), 'audio': ('samples', 'samples'), 'text': ('text', 'unicode_codepoints')}[modality]
        check(span.get('kind') == kind and span.get('unit') == unit, 'Wrong source coordinate unit')
        check(all(type(span.get(k)) is int for k in ('start', 'end', 'extent'))
            and 0 <= span['start'] < span['end'] <= span['extent'], 'Source interval outside extent')
        if modality == 'audio':
            check(type(span.get('sample_rate')) is int and span['sample_rate'] > 0, 'Audio sample clock required')


def audit(manifest):
    check(manifest.get('schema') == SCHEMA and isinstance(manifest.get('policy'), dict), 'Exact specialist manifest schema/policy required')
    specialists = manifest['policy'].get('specialists')
    check(isinstance(specialists, list) and specialists, 'Declared specialists required')
    specs = {}
    for spec in specialists:
        check(token(spec.get('id')) and spec['id'] not in specs, 'Unique specialist identity required')
        check(spec.get('modality') in MEDIA and token(spec.get('family'))
            and token(spec.get('target')), 'Modality, dependency family and target required')
        check(isinstance(spec.get('contexts'), list) and spec['contexts']
            and all(token(c) for c in spec['contexts'])
            and len(set(spec['contexts'])) == len(spec['contexts']), 'Explicit score-blind contexts required')
        check(is_hash(spec.get('routing_sha256')) and spec.get('higher_means_ai') is True, 'Pinned pre-score routing and response direction required')
        producer = spec.get('producer', {})
        check(producer.get('kind') in ('model', 'authored_synthetic_fixture')
            and is_hash(producer.get('implementation_sha256')), 'Exact measurement producer contract required')
        if producer['kind'] == 'model':
            check(isinstance(producer.get('weight_sha256'), list) and producer['weight_sha256']
                and all(is_hash(v) for v in producer['weight_sha256'])
                and is_hash(producer.get('commercial_license_record_sha256'))
                and is_hash(producer.get('native_receipt_sha256')), 'Model observations need pinned weights, commercial-license review and native receipt')
        bounds = spec.get('raw_range', [])
        check(len(bounds) == 2 and all(finite(v) for v in bounds) and bounds[0] < bounds[1], 'Finite declared raw response range required')
        specs[spec['id']] = spec
    files = manifest.get('files')
    check(isinstance(files, list) and files, 'Declared source files required')
    seen, ancestry, unit_count = set(), {}, 0
    for file in files:
        check(isinstance(file.get('id'), str) and file['id'] and file['id'] not in seen, 'Unique file identity required')
        seen.add(file['id'])
        check(file.get('role') in ROLES and file.get('modality') in MEDIA, 'Explicit role/modality required')
        check(type(file.get('label')) is int and file['label'] in (0, 1)
            and file.get('label_scope') == 'planned_units' and is_hash(file.get('label_provenance_sha256')), 'Explicit file truth for planned evidence required')
        check(isinstance(file.get('group'), str) and file['group'] and is_hash(file.get('source_sha256'))
            and is_hash(file.get('input_sha256')), 'Source group and exact hashes required')
        lineage = file.get('lineage', [])
        check(isinstance(lineage, list) and all(isinstance(v, str) and v for v in lineage), 'Declared connected lineage required')
        for key in [('group', file['group']), ('hash', file['source_sha256']), ('hash', file['input_sha256'])] + [('lineage', v) for v in lineage]:
            check(ancestry.setdefault(key, file['role']) == file['role'], 'Source/ancestry leakage across calibration, fusion or confirmation roles')
        families = file.get('required_families')
        allowed = {s['family'] for s in specialists if s['modality'] == file['modality']}
        check(isinstance(families, list) and families and len(set(families)) == len(families)
            and set(families) <= allowed, 'Predeclared required evidence families required')
        units = file.get('units')
        check(isinstance(units, list) and units, 'All planned evidence units must be recorded, including failures')
        ids = set()
        for row in units:
            check(isinstance(row.get('id'), str) and row['id'] and row['id'] not in ids, 'Unique local observation identity required')
            ids.add(row['id'])
            check(row.get('specialist') in specs, 'Unknown specialist')
            spec = specs[row['specialist']]
            check(spec['modality'] == file['modality'] and row.get('context') in spec['contexts'], 'Unknown/cross-modality calibration context')
            check(spec['family'] in families, 'Score-bearing observations must belong to the declared combined scope')
            check(row.get('routing_sha256') == spec['routing_sha256'] and is_hash(row.get('context_evidence_sha256')),
                'Context must bind the frozen pre-score routing measurement')
            check(is_hash(row.get('measurement_sha256')), 'Exact measurement commitment required')
            location(file['modality'], row.get('location'))
            check(row.get('status') in STATUSES, 'Explicit local observation status required')
            if row['status'] == 'ok':
                check(finite(row.get('raw')) and spec['raw_range'][0] <= row['raw'] <= spec['raw_range'][1], 'Finite declared raw response required')
            else:
                check(row.get('raw') is None and isinstance(row.get('reason'), str) and row['reason'], 'Unassessed evidence has no response and needs a reason')
            label = row.get('label')
            check(label is None or (type(label) is int and label in (0, 1)), 'Local labels are explicit0/1 or unknown')
            if label is not None:
                check(row.get('label_scope') == 'local_unit' and is_hash(row.get('label_provenance_sha256')),
                    'A whole-file AI label is not a local-unit label')
                check(not (label == 1 and file['label'] == 0), 'Local AI truth contradicts declared genuine scope')
            unit_count += 1
        check(set(families) <= {specs[u['specialist']]['family'] for u in units}, 'Missing required family must be recorded as unavailable, not omitted')
        if all(u.get('label') is not None for u in units):
            check(int(any(u['label'] == 1 for u in units)) == file['label'], 'Complete local truth contradicts the planned-scope file label')
    return {'files': len(files), 'units': unit_count, 'roles': dict(Counter(f['role'] for f in files)),
        'modalities': dict(Counter(f['modality'] for f in files)), 'metadata_only': True}


def plan_digest(manifest):
    planned = copy.deepcopy(manifest)
    for file in planned['files']:
        for row in file['units']:
            for key in ('raw', 'status', 'reason', 'measurement_sha256'):
                row.pop(key, None)
    return digest(planned)


def _fit(x, y, groups, *, minimum, l2):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    count = {v: len({g for g, truth in zip(groups, y, strict=True) if truth == v}) for v in (0, 1)}
    if min(count.values()) < minimum:
        return {'status': 'insufficient_support', 'source_groups_by_class': {str(k): v for k, v in count.items()}}
    cells = Counter(zip(groups, y, strict=True))
    weight = np.asarray([1 / (2 * count[int(t)] * cells[g, t]) for g, t in zip(groups, y, strict=True)], np.float64)
    center = np.einsum('n,nj->j', weight, x)
    scale = np.maximum(np.sqrt(np.einsum('n,nj->j', weight, (x - center) ** 2)), 1e-6)
    check(np.isfinite(center).all() and np.isfinite(scale).all(), 'Raw response units exceed the numerical calibration range')
    z = np.clip((x - center) / scale, -8., 8.)
    target = np.where(y == 1, (count[1] + 1) / (count[1] + 2), 1 / (count[0] + 2))

    def objective(theta):
        eta = np.einsum('nj,j->n', z, theta[:-1]) + theta[-1]
        residual = weight * (expit(eta) - target)
        loss = float(np.dot(weight, np.logaddexp(0., eta) - target * eta) + l2 * np.dot(theta[:-1], theta[:-1]))
        gradient = np.r_[np.einsum('n,nj->j', residual, z) + 2 * l2 * theta[:-1], residual.sum()]
        return loss, gradient

    solved = minimize(objective, np.zeros(x.shape[1] + 1), jac=True, method='L-BFGS-B',
        bounds=[(0., None)] * x.shape[1] + [(None, None)], options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-9})
    check(solved.success and np.isfinite(solved.x).all(), 'Source-balanced monotone calibration fit did not converge')
    return {'status': 'fitted', 'source_groups_by_class': {str(k): v for k, v in count.items()},
        'center': center.tolist(), 'scale': scale.tolist(), 'coefficients': solved.x[:-1].tolist(),
        'intercept': float(solved.x[-1]), 'reference_prior': .5, 'source_count_target_smoothing': True}


def _logit(model, values):
    check(model['status'] == 'fitted' and len(values) == len(model['center']), 'Applicable fitted calibration required')
    value = model['intercept']
    for raw, center, scale, coefficient in zip(values, model['center'], model['scale'], model['coefficients'], strict=True):
        check(finite(raw), 'Unassessed evidence cannot become a zero feature')
        value += coefficient * max(-8., min(8., (raw - center) / scale))
    check(math.isfinite(value), 'Finite calibrated logit required')
    return value


def _features(file, artifact):
    families = defaultdict(list)
    specs = {s['id']: s for s in artifact['policy']['specialists']}
    for row in file['units']:
        family = specs[row['specialist']]['family']
        if family not in file['required_families']:
            continue
        model = artifact['specialists'][row['specialist'] + '|' + row['context']]
        if row['status'] != 'ok' or model['status'] != 'fitted':
            return None
        families[family].append(_logit(model, [row['raw']]))
    if set(families) != set(file['required_families']):
        return None
    # Same-family windows/heads do not become repeated independent votes. The
    # maxima are scores for the separately fitted AND calibrated file combiner.
    return [max(families[key]) for key in sorted(file['required_families'])]


def pattern(file):
    return file['modality'] + '|' + ','.join(sorted(file['required_families']))


def train(manifest, *, min_groups_per_class=25, l2=.001):
    audit(manifest)
    check(type(min_groups_per_class) is int and min_groups_per_class >= 2 and finite(l2) and l2 > 0, 'Explicit support and regularization required')
    training = [f for f in manifest['files'] if f['role'] != 'confirmation']
    artifact = {'schema': 'chipmunk.specialist-calibration/1', 'policy': copy.deepcopy(manifest['policy']),
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
    check(artifact.get('schema') == 'chipmunk.specialist-calibration/1'
        and artifact.get('serving_admitted') is False
        and artifact.get('sha256') == digest({k: v for k, v in artifact.items() if k != 'sha256'}), 'Exact fitted artifact required')
    check(artifact['implementation_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
    return {'schema': 'chipmunk.specialist-confirmation/1', 'artifact_sha256': artifact['sha256'],
        'outcomes': outcomes, 'unit_outcomes': local, 'summaries': summaries, 'serving_admitted': False,
        'limits': 'Research scores under a50:50 reference prior; no validated customer accuracy or scope beyond planned units.'}
