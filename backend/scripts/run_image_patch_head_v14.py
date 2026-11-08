"""Freeze, fit and diagnose a supervised regional image head on consumed sources."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import socket
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import image_patch_drift_v8 as backbone_method
from scripts import image_patch_head_v14 as method
from scripts import image_residual_localizer_v13 as old_method
from scripts import run_image_residual_v13 as prior

ROOT = Path(__file__).resolve().parents[2]
CODE = tuple(dict.fromkeys((*prior.CODE,
    'backend/scripts/image_patch_head_v14.py', 'backend/scripts/run_image_patch_head_v14.py',
    'backend/scripts/review_image_patch_head_v14.py', 'backend/scripts/image_patch_head_runtime_v14.py',
    'backend/scripts/image_patch_drift_v8.py', 'backend/tests/test_image_patch_head_v14.py',
    'docs/image-patch-head-v14-protocol.md', 'backend/app/detectors/image/radar.py',
    'backend/app/engine/fetchers.py', 'backend/app/engine/registry.py', 'backend/app/core/config.py')))
POLICY = {
    'schema': 'chipmunk.image-supervised-patch-head/14',
    'sources': 'All consumed v13 source groups:120 fit,60 genuine-only point,108 diagnostics; no fresh-source accuracy claim.',
    'features': 'Frozen Apache-2 DINOv2 register-base;448-square bilinear RGB/ImageNet;normalized blocks4,8,12;1024x2304 float32 tokens.',
    'head': 'Concatenate token and token-minus-full-image-mean;4608->64 ReLU->1 sigmoid ranking head;295041 learned parameters.',
    'fit': 'Same840 original/lighting/JPEG55/ordinary-patch cases as v13;64 deterministic stratified cells per case;exact soft source-area labels.',
    'optimization': 'CPU float32/two threads;seed20261108;AdamW lr.001 weight_decay.01;30 epochs,batch512;balanced soft class mass;no early stopping or diagnostic selection.',
    'point': 'Maximum over genuine original and ordinary control per60 point groups;descending indexfloor(.05*n),strict greater-than;AI outcomes never select the point.',
    'mask': '32x32 median3 reflect;8-connected components>=4 cells;sourcepixel(x,y)->floor(32*x/width),floor(32*y/height).',
    'continuation': 'Higher edited meanIoU and more IoU>=.3 than current,no extra genuine flags,ordinary flags<=5%;full native/source/arithmetic audit also required.',
    'limits': 'Consumed development;not confidence,pixel authorship,unseen-generator validation or serving admission. Current ordinary-control model inference and fresh confirmation follow only a passing development result.',
    'cache': 'Lossless float32 sampled fit vectors and full-image means;full native feature commitments per case,retained ranking maps;separate full native replay required.',
    'serving_admitted': False, 'used_in_confidence': False,
}

digest, read, write = prior.digest, prior.read, prior.write


def code():
    return {path: digest(ROOT / path) for path in CODE}


def packages():
    return {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scipy', 'Pillow', 'safetensors')}


def assets():
    from app.engine.fetchers import file_path
    from app.engine.registry import get_registry

    found = []
    registry = get_registry()
    for identifier in (backbone_method.MODEL_ID, backbone_method.ARCHITECTURE_ID):
        model = registry.manifest.model(identifier)
        method.require(model.licence.class_ == 'commercial', 'Commercially admitted backbone and code required')
        for item in model.source.files:
            path = file_path(model, item)
            method.require(path.is_file() and digest(path) == item.sha256, 'Exact cached asset required; no download')
            found.append({'model': identifier, 'path': str(path), 'sha256': item.sha256,
                          'bytes': path.stat().st_size, 'license': model.licence.spdx})
    return found


def load_backbone():
    import torch

    assets()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(method.SEED)
    model = backbone_method.load(ROOT, 'cpu')
    def refuse(*_args, **_kwargs):
        raise RuntimeError('No network access in the native image study')
    socket.socket.connect = refuse
    socket.create_connection = refuse
    return model


def contract(output):
    model = load_backbone()
    y, x = np.indices((73, 101))
    rows, first = [], None
    for index in range(4):
        pixels = np.stack([(x * (index + 3)) % 256, (y * 7) % 256, ((x + y) * 5) % 256], axis=-1).astype(np.uint8)
        features, average, tensor = method.extract(model, pixels, hook=True)
        rows.append({'authored_case': index, 'features_sha256': method.sha(features),
                     'mean_sha256': method.sha(average), 'tensor_sha256': tensor, 'hook_exact': True})
        if first is None:
            first = (pixels, features)
    repeated = method.extract(model, first[0])[0]
    method.require(np.array_equal(repeated, first[1]), 'Exact authored native repeat required')
    write(output, {'policy': POLICY, 'code_sha256': code(), 'packages': packages(),
                   'assets': assets(), 'authored_cases': rows, 'backbone_forwards': 9,
                   'project_source_forwards': 0, 'exact_repeat': True})
    print(json.dumps({'authored_cases': 4, 'backbone_forwards': 9, 'project_source_forwards': 0}), flush=True)


def source_roles(entries):
    groups, byte_roles = defaultdict(set), {}
    for entry in entries:
        groups[entry['role']].add(entry['source']['group'])
        for field in ('source_sha256', 'input_sha256', 'rgb_sha256'):
            value = entry['source'][field]
            method.require(byte_roles.setdefault(value, entry['role']) == entry['role'],
                           'Identical source or decoded input bytes cannot cross source roles')
    return groups


def prepare(trial, contract_path):
    source = ROOT / 'backend/data/independent-evaluation/image-residual-v13'
    plan = prior.check(source)
    checked = read(contract_path)
    method.require(checked['code_sha256'] == code() and checked['policy'] == POLICY
                   and checked['packages'] == packages() and checked['assets'] == assets()
                   and checked['exact_repeat'], 'Current complete authored native contract required')
    entries, cases, probes = plan['entries'], [], {}
    for index, entry in enumerate(entries):
        names = ['original', 'lighting', 'jpeg55'] if entry['role'] == 'fit' else ['original']
        if entry['source']['label'] == 0:
            names.append('ordinary_patch')
        for variant in names:
            case = {'entry_index': index, 'variant': variant,
                    'id': hashlib.sha256((entry['key'] + '|' + variant).encode()).hexdigest(),
                    'role': entry['role'], 'label': entry['source']['label']}
            cases.append(case)
            probes.setdefault((entry['role'], entry['generator'], entry['source']['label']), case['id'])
    counts = Counter(c['role'] for c in cases)
    method.require(dict(counts) == {'fit': 840, 'point': 180, 'diagnostic': 324} and len(probes) == 18,
                   'Complete original source roles and all1344 cases required')
    # Source-role separation is inherited from and rechecked against the unchanged
    # complete source plan; not inferred from file names or neural outcomes.
    groups = source_roles(entries)
    method.require({k: len(v) for k, v in groups.items()} == {'fit': 120, 'point': 60, 'diagnostic': 108}
                   and len(set.union(*groups.values())) == 288, 'All source groups must remain role-disjoint')
    trial.mkdir(parents=True, exist_ok=False)
    write(trial / 'plan.json', {'source_plan': plan, 'cases': cases, 'probe_cases': list(probes.values())})
    freeze = {'policy': POLICY, 'plan_sha256': digest(trial / 'plan.json'), 'code_sha256': code(),
              'prior_plan_path': str(source / 'plan.json'), 'prior_plan_sha256': digest(source / 'plan.json'),
              'contract_path': str(contract_path), 'contract_sha256': digest(contract_path),
              'packages': packages(), 'assets': assets(), 'project_source_forwards_before_freeze': 0}
    write(trial / 'freeze.json', freeze)
    write(ROOT / 'docs/evaluation-panels/image-patch-head-v14-freeze.json', {
        'policy': POLICY, 'private_freeze_sha256': digest(trial / 'freeze.json'), 'code_sha256': code(),
        'source_groups': {k: len(v) for k, v in groups.items()}, 'cases': dict(counts),
        'authored_contract_sha256': digest(contract_path), 'authored_contract_backbone_forwards': 9,
        'assets': [{k: v for k, v in a.items() if k != 'path'} for a in freeze['assets']],
        'packages': packages(), 'project_source_forwards_before_freeze': 0,
        'fresh_confirmation': False, 'raw_sources_published': False, 'serving_admitted': False})
    print(json.dumps({'prepared_cases': len(cases), 'roles': dict(counts), 'source_groups': 288}), flush=True)


def check(trial):
    lock, plan = read(trial / 'freeze.json'), read(trial / 'plan.json')
    method.require(lock['policy'] == POLICY and lock['code_sha256'] == code()
                   and lock['packages'] == packages() and lock['assets'] == assets()
                   and lock['plan_sha256'] == digest(trial / 'plan.json')
                   and lock['prior_plan_sha256'] == digest(lock['prior_plan_path'])
                   and lock['contract_sha256'] == digest(lock['contract_path'])
                   and all(digest(p) == expected for p, expected in plan['source_plan']['bound_prior_files'].items()),
                   'Frozen code, source panel, assets, contract or runtime changed')
    return plan


def pixels_for(plan, case):
    entry = plan['source_plan']['entries'][case['entry_index']]
    rgb, truth = prior.load(plan['source_plan'], entry)
    return entry, old_method.intervene(rgb, case['variant']), truth


def point(records):
    maxima = defaultdict(list)
    for row in records:
        method.require(row['role'] == 'point' and row['label'] == 0, 'Only genuine point-role outcomes may select a point')
        maxima[row['group']].append(row['maximum'])
    method.require(len(maxima) == 60 and all(len(v) == 2 for v in maxima.values()), 'Both genuine variants of all60 groups required')
    return old_method.point([max(values) for values in maxima.values()])


def run(trial):
    from safetensors.torch import save_file

    plan, model = check(trial), load_backbone()
    private = trial / 'private'
    private.mkdir(exist_ok=False)
    samples = np.lib.format.open_memmap(private / 'fit-vectors.npy', mode='w+', dtype=np.float32,
                                        shape=(840, method.SAMPLES, method.DIMENSION))
    means = np.lib.format.open_memmap(private / 'fit-means.npy', mode='w+', dtype=np.float32,
                                      shape=(840, method.DIMENSION))
    targets = np.lib.format.open_memmap(private / 'fit-targets.npy', mode='w+', dtype=np.float32,
                                        shape=(840, method.SAMPLES))
    records, forwards, fit_count, head = [], 0, 0, None
    journal = (trial / 'native-private.jsonl').open('x')
    def process(case):
        nonlocal forwards, fit_count
        entry, rgb, truth = pixels_for(plan, case)
        probe = case['id'] in plan['probe_cases']
        values, average, tensor = method.extract(model, rgb, hook=probe)
        forwards += 1 + int(probe)
        row = {**case, 'key': entry['key'], 'group': entry['source']['group'], 'generator': entry['generator'],
               'cohort': entry['cohort'], 'input_sha256': entry['source']['input_sha256'],
               'pixels_sha256': method.sha(rgb), 'truth_sha256': method.sha(truth),
               'tensor_sha256': tensor, 'features_sha256': method.sha(values),
               'mean_sha256': method.sha(average), 'hook_exact': probe,
               'width': rgb.shape[1], 'height': rgb.shape[0]}
        if case['role'] == 'fit':
            soft = method.pool_truth(truth)
            chosen = method.indices(soft, entry['key'])
            samples[fit_count], means[fit_count], targets[fit_count] = values[chosen], average, soft.ravel()[chosen]
            row.update(fit_index=fit_count, sampled_indices_sha256=method.sha(chosen),
                       sampled_features_sha256=method.sha(values[chosen]), soft_targets_sha256=method.sha(soft.ravel()[chosen]))
            fit_count += 1
        else:
            scores = method.response(head, values, average)
            path = private / (case['id'] + '-scores.npy')
            np.save(path, scores, allow_pickle=False)
            row.update(scores_file=path.name, scores_sha256=method.sha(scores),
                       scores_file_sha256=digest(path), maximum=float(scores.max()))
        if len(records) == 0:
            repeated = method.extract(model, rgb)[0]
            forwards += 1
            method.require(np.array_equal(repeated, values), 'Exact first-source native repeat required')
            row['native_repeat_exact'] = True
        records.append(row)
        journal.write(json.dumps(row, allow_nan=False) + '\n')
        journal.flush()
        if len(records) % 80 == 0:
            samples.flush()
            means.flush()
            targets.flush()
            print(json.dumps({'completed_cases': len(records), 'planned': 1344, 'backbone_forwards': forwards}), flush=True)
    try:
        for case in plan['cases']:
            if case['role'] == 'fit':
                process(case)
        method.require(fit_count == 840, 'Complete fit role required before fitting')
        samples.flush()
        means.flush()
        targets.flush()
        head, training = method.train(samples, means, targets)
        head_path = private / 'head.safetensors'
        save_file(head.state_dict(), str(head_path))
        for case in plan['cases']:
            if case['role'] == 'point' and case['label'] == 0:
                process(case)
        chosen_point = point([row for row in records if row['role'] == 'point'])
        write(trial / 'fitted-point-private.json', {'training': training, 'point': chosen_point,
              'head_sha256': digest(head_path), 'cases_before_point': len(records),
              'point_ai_cases_scored': 0, 'diagnostic_cases_scored': 0,
              'fit_arrays_sha256': {p.name: digest(p) for p in private.glob('fit-*.npy')}})
        fixed = digest(trial / 'fitted-point-private.json')
        for case in plan['cases']:
            if case['role'] == 'diagnostic' or (case['role'] == 'point' and case['label'] == 1):
                process(case)
        method.require(fixed == digest(trial / 'fitted-point-private.json')
                       and digest(head_path) == read(trial / 'fitted-point-private.json')['head_sha256'],
                       'Head and point cannot change after diagnostics')
    finally:
        journal.close()
    method.require(len(records) == 1344 and forwards == 1363, 'All1344 cases and exact declared forwards required')
    outcomes = []
    for row in records:
        if row['role'] == 'fit':
            continue
        _, _, truth = pixels_for(plan, row)
        scores = np.load(private / row['scores_file'], allow_pickle=False)
        mask = method.selected(scores, chosen_point['value'], row['width'], row['height'])
        outcomes.append({k: row[k] for k in ('id', 'role', 'label', 'variant', 'group', 'generator', 'cohort')} |
                        {'metrics': old_method.metrics(mask, truth), 'mask_sha256': method.sha(mask)})
    write(trial / 'outcomes-private.json', outcomes)
    report = summarize(plan, outcomes)
    report.update(policy=POLICY, freeze_sha256=digest(trial / 'freeze.json'), cases=1344,
                  backbone_forwards=forwards, native_hook_probes=18, native_repeat_forwards=1,
                  head_training=training, head_inference_cases=504, point=chosen_point,
                  journal_sha256=digest(trial / 'native-private.jsonl'),
                  fit_array_bytes=sum(p.stat().st_size for p in private.glob('fit-*.npy')),
                  independent_native_audit_required=True, outside_human_reviewers=0,
                  serving_admitted=False, used_in_confidence=False)
    check(trial)
    write(trial / 'result.json', report)
    print(json.dumps({'completed': 1344, 'backbone_forwards': forwards, 'diagnostic': report['all']}), flush=True)


def summarize(plan, outcomes):
    result = {}
    for cohort in ('all', 'v11', 'consumed-v12', 'firefly', 'sd2', 'flux-fill'):
        def eligible(row, cohort=cohort):
            return row['role'] == 'diagnostic' and (cohort == 'all' or row['cohort'] == cohort or row['generator'] == cohort)
        selected = [e for e in plan['source_plan']['entries'] if eligible(e)]
        baseline = prior.summary([{'label': e['source']['label'], 'variant': 'original',
                                   'metrics': e['current']} for e in selected])
        candidate = prior.summary([r for r in outcomes if eligible(r)])
        gates = {'higher_mean_iou': candidate['edited_mean_iou'] > baseline['edited_mean_iou'],
                 'more_iou_0_3': candidate['edited_iou_at_least_0_3'] > baseline['edited_iou_at_least_0_3'],
                 'no_extra_genuine_flags': candidate['genuine_flags'] <= baseline['genuine_flags'],
                 'ordinary_at_most_5pct': candidate['ordinary_flags'] <= .05 * candidate['ordinary_inputs']}
        result[cohort] = {'current_originals_only': baseline, 'candidate': candidate,
                          'continuation_gates': gates, 'quality_rules_pass': all(gates.values())}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('contract', 'prepare', 'run'))
    parser.add_argument('trial', type=Path)
    parser.add_argument('--contract', type=Path)
    args = parser.parse_args()
    if args.action == 'contract':
        contract(args.trial)
    elif args.action == 'prepare':
        prepare(args.trial, args.contract)
    else:
        run(args.trial)
