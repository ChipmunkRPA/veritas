"""Prospective factorial study of context precision and actual region calibration."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts import diagnose_image_patch_head_v14 as previous_diagnostics
from scripts import image_region_point_v15 as method
from scripts import run_image_patch_head_v14 as old

ROOT, read, write, digest = old.ROOT, old.read, old.write, old.digest
CODE = tuple(dict.fromkeys((*old.CODE, *previous_diagnostics.CODE,
    'backend/scripts/image_region_point_v15.py', 'backend/scripts/run_image_region_point_v15.py',
    'backend/scripts/review_image_region_point_v15.py', 'backend/scripts/image_region_point_runtime_v15.py',
    'backend/tests/test_image_region_point_v15.py', 'docs/image-region-point-v15-protocol.md')))
POLICY = {
    'schema': 'chipmunk.image-region-point-factorial/15',
    'primary_candidate': method.PRIMARY, 'arms': list(method.ARMS),
    'context': 'Float64 accumulation of all1024 float32 patch vectors,rounded once to float32;refit the same295041-parameter head on fit-role means only.',
    'region_point': 'Highest8-connected superlevel with>=4 cells;process equal-valued cells together. A selected region exists iff activation>point.',
    'source': 'Unchanged consumed288-source v14 plan/all1344 cases;120 fit,60 point,108 diagnostic groups. No fresh-source claim.',
    'fit_cache': 'Reuse immutable old float32 sampled vectors/targets without copying. Recreate all840 full native features and compare every sample/target before fitting;retain new means.',
    'head': old.POLICY['head'], 'optimization': old.POLICY['optimization'],
    'points': 'Old/stable heads each have raw-maximum and region-activation points. Both use only genuine original/ordinary variants of60 point groups,descending rank3,strict greater-than.',
    'isolation': 'Freeze both fitted heads identities and all4 points after840 fit+120 genuine point cases,before new AI point/diagnostic scoring. Old data already consumed.',
    'spatial': old.POLICY['mask'],
    'continuation': 'Primary stable_regions must improve edited meanIoU and IoU>=.3 count,no extra genuine flags versus current,and ordinary flags<=5%;complete native/mean/head/source/activation audit required.',
    'limits': 'All other arms are ablations,not selectable alternatives. Current ordinary-control inference,fresh confirmation and full-app testing remain before admission.',
    'serving_admitted': False, 'used_in_confidence': False,
}


def prior_trial():
    return ROOT / 'backend/data/independent-evaluation/image-patch-head-v14'


def code():
    return {p: digest(ROOT / p) for p in CODE}


def prior_state():
    previous = prior_trial()
    plan = previous_diagnostics.check(previous)
    audit = read(previous / 'diagnostic-replay.json')
    old.method.require(audit['cases'] == 1344 and audit['native_features_and_maps_exact']
        and audit['source_truth_samples_and_projection_exact'] and not audit['original_strict_audit_passed'],
        'Complete old diagnostic replay and unamended original failure required')
    point = read(previous / 'fitted-point-private.json')
    for name, expected in point['fit_arrays_sha256'].items():
        old.method.require(digest(previous / 'private' / name) == expected, 'Immutable prior fit arrays required')
    return plan


def contract(output):
    model = old.load_backbone()
    yy, xx = np.indices((73, 101))
    rows, first = [], None
    for index in range(4):
        rgb = np.stack([(xx * (index + 3)) % 256, (yy * 7) % 256, ((xx + yy) * 5) % 256], axis=-1).astype(np.uint8)
        features, average, tensor = method.extract(model, rgb, hook=True)
        # Grouping the sum differently is separate from the primary axis-0
        # accumulator. No float32 intermediate sum is introduced.
        reference = features.astype(np.float64).reshape(16, 64, -1).sum(1).sum(0) / 1024
        difference = float(np.abs(average - reference).max())
        old.method.require(difference <= 1e-5, 'New mean must satisfy the unchanged strict reference bound')
        rows.append({'authored_case': index, 'features_sha256': old.method.sha(features),
                     'mean_sha256': old.method.sha(average), 'tensor_sha256': tensor,
                     'independent_mean_max_difference': difference, 'full_hook_exact': True})
        if first is None:
            first = (rgb, features, average)
    repeated, average, _ = method.extract(model, first[0])
    old.method.require(np.array_equal(repeated, first[1]) and np.array_equal(average, first[2]), 'Exact authored repeat required')
    write(output, {'policy': POLICY, 'code_sha256': code(), 'packages': old.packages(), 'assets': old.assets(),
                   'authored_cases': rows, 'backbone_forwards': 9, 'project_forwards': 0, 'exact_repeat': True})
    print({'authored_cases': 4, 'backbone_forwards': 9, 'project_forwards': 0}, flush=True)


def prepare(trial, contract_path):
    plan, previous = prior_state(), prior_trial()
    native_contract = read(contract_path)
    old.method.require(native_contract['policy'] == POLICY and native_contract['code_sha256'] == code()
                       and native_contract['packages'] == old.packages() and native_contract['assets'] == old.assets()
                       and native_contract['exact_repeat'], 'Current complete native contract required')
    bound = {str(previous / name): digest(previous / name) for name in (
        'plan.json', 'freeze.json', 'result.json', 'diagnostic-replay.json', 'diagnostic-freeze.json',
        'fitted-point-private.json', 'native-private.jsonl', 'outcomes-private.json',
        'private/head.safetensors', 'private/fit-vectors.npy', 'private/fit-targets.npy', 'private/fit-means.npy')}
    old.method.require(len(plan['cases']) == 1344, 'Full unchanged source plan required')
    trial.mkdir(parents=True, exist_ok=False)
    freeze = {'policy': POLICY, 'code_sha256': code(), 'packages': old.packages(), 'assets': old.assets(),
              'bound_prior_files': bound, 'contract_path': str(contract_path), 'contract_sha256': digest(contract_path),
              'project_forwards_before_freeze': 0, 'old_failure_unchanged': True}
    write(trial / 'freeze.json', freeze)
    write(ROOT / 'docs/evaluation-panels/image-region-point-v15-freeze.json', {
        'policy': POLICY, 'private_freeze_sha256': digest(trial / 'freeze.json'), 'code_sha256': code(),
        'packages': old.packages(), 'assets': [{k: v for k, v in a.items() if k != 'path'} for a in old.assets()],
        'case_roles': {'fit': 840, 'point': 180, 'diagnostic': 324},
        'source_groups': {'fit': 120, 'point': 60, 'diagnostic': 108},
        'prior_native_result_sha256': digest(previous / 'result.json'),
        'prior_complete_diagnostic_sha256': digest(previous / 'diagnostic-replay.json'),
        'native_contract_sha256': digest(contract_path), 'project_forwards_before_freeze': 0,
        'old_failure_unchanged': True, 'serving_admitted': False})
    print({'prepared_cases': 1344, 'primary_candidate': method.PRIMARY, 'copied_fit_vectors': False}, flush=True)


def check(trial):
    lock = read(trial / 'freeze.json')
    old.method.require(lock['policy'] == POLICY and lock['code_sha256'] == code()
                       and lock['packages'] == old.packages() and lock['assets'] == old.assets()
                       and lock['contract_sha256'] == digest(lock['contract_path'])
                       and all(digest(p) == expected for p, expected in lock['bound_prior_files'].items()),
                       'Frozen code, inputs, old results, arrays or contract changed')
    return old.check(prior_trial())


def old_rows():
    return {r['id']: r for r in (json.loads(line) for line in (prior_trial() / 'native-private.jsonl').read_text().splitlines())}


def old_scores(row):
    path = prior_trial() / 'private' / row['scores_file']
    scores = np.load(path, allow_pickle=False)
    old.method.require(digest(path) == row['scores_file_sha256'] and old.method.sha(scores) == row['scores_sha256'],
                       'Original bound score grid required')
    return scores


def point_records(rows, scores_for):
    selected = []
    for row in rows:
        if row['role'] == 'point' and row['label'] == 0:
            scores = scores_for(row)
            selected.append({**{k: row[k] for k in ('id', 'group', 'variant', 'role', 'label')},
                             'maximum': float(scores.max()), 'activation': method.activation(scores)})
    return selected


def summarize(plan, outcomes):
    result = {}
    for cohort in ('all', 'v11', 'consumed-v12', 'firefly', 'sd2', 'flux-fill'):
        def eligible(row, cohort=cohort):
            return row['role'] == 'diagnostic' and (cohort == 'all' or row['cohort'] == cohort or row['generator'] == cohort)
        entries = [e for e in plan['source_plan']['entries'] if eligible(e)]
        baseline = old.prior.summary([{'label': e['source']['label'], 'variant': 'original', 'metrics': e['current']}
                                      for e in entries])
        arms = {arm: old.prior.summary([r for r in outcomes if eligible(r) and r['arm'] == arm]) for arm in method.ARMS}
        gates = {}
        for arm, value in arms.items():
            rules = {'higher_mean_iou': value['edited_mean_iou'] > baseline['edited_mean_iou'],
                     'more_iou_0_3': value['edited_iou_at_least_0_3'] > baseline['edited_iou_at_least_0_3'],
                     'no_extra_genuine_flags': value['genuine_flags'] <= baseline['genuine_flags'],
                     'ordinary_at_most_5pct': value['ordinary_flags'] <= .05 * value['ordinary_inputs']}
            gates[arm] = {'rules': rules, 'pass': all(rules.values()), 'primary_candidate': arm == method.PRIMARY}
        result[cohort] = {'current_originals_only': baseline, 'arms': arms, 'quality_rules': gates}
    return result


def run(trial):
    from safetensors.torch import save_file

    plan, prior_records = check(trial), old_rows()
    private = trial / 'private'
    private.mkdir(exist_ok=False)
    samples = np.load(prior_trial() / 'private/fit-vectors.npy', mmap_mode='r', allow_pickle=False)
    targets = np.load(prior_trial() / 'private/fit-targets.npy', mmap_mode='r', allow_pickle=False)
    means = np.lib.format.open_memmap(private / 'stable-fit-means.npy', mode='w+', dtype=np.float32,
                                      shape=(840, old.method.DIMENSION))
    model, records, forwards, head = old.load_backbone(), [], 0, None
    fit_seen = set()
    journal = (trial / 'native-private.jsonl').open('x')
    def process(case):
        nonlocal forwards
        entry, rgb, truth = old.pixels_for(plan, case)
        previous = prior_records[case['id']]
        probe = case['id'] in plan['probe_cases']
        features, average, tensor = method.extract(model, rgb, hook=probe)
        forwards += 1 + int(probe)
        old.method.require(old.method.sha(features) == previous['features_sha256']
                           and tensor == previous['tensor_sha256'] and old.method.sha(rgb) == previous['pixels_sha256']
                           and old.method.sha(truth) == previous['truth_sha256'], 'Exact original source/features required')
        row = {**case, **{k: previous[k] for k in ('key', 'group', 'generator', 'cohort', 'input_sha256',
               'pixels_sha256', 'truth_sha256', 'features_sha256', 'width', 'height')},
               'tensor_sha256': tensor, 'stable_mean_sha256': old.method.sha(average), 'hook_exact': probe}
        if case['role'] == 'fit':
            fit = previous['fit_index']
            soft = old.method.pool_truth(truth)
            indices = old.method.indices(soft, entry['key'])
            old.method.require(fit not in fit_seen and np.array_equal(features[indices], samples[fit])
                               and np.array_equal(soft.ravel()[indices], targets[fit]), 'Every reused fit sample/target must reproduce')
            fit_seen.add(fit)
            means[fit] = average
            row.update(fit_index=fit, sampled_indices_sha256=old.method.sha(indices))
        else:
            scores = old.method.response(head, features, average)
            path = private / (case['id'] + '-scores.npy')
            np.save(path, scores, allow_pickle=False)
            row.update(scores_file=path.name, scores_sha256=old.method.sha(scores), scores_file_sha256=digest(path),
                       maximum=float(scores.max()), activation=method.activation(scores))
        if not records:
            repeat, repeat_mean, _ = method.extract(model, rgb)
            forwards += 1
            old.method.require(np.array_equal(features, repeat) and np.array_equal(average, repeat_mean), 'Exact first native repeat required')
            row['native_repeat_exact'] = True
        records.append(row)
        journal.write(json.dumps(row, allow_nan=False) + '\n')
        journal.flush()
        if len(records) % 80 == 0:
            means.flush()
            print({'completed_cases': len(records), 'planned': 1344, 'backbone_forwards': forwards}, flush=True)
    try:
        for case in plan['cases']:
            if case['role'] == 'fit':
                process(case)
        old.method.require(len(records) == len(fit_seen) == 840, 'All840 exact reused fit samples required before training')
        means.flush()
        head, training = old.method.train(samples, means, targets)
        save_file(head.state_dict(), str(private / 'head.safetensors'))
        for case in plan['cases']:
            if case['role'] == 'point' and case['label'] == 0:
                process(case)
        old_cal = point_records(prior_records.values(), old_scores)
        new_cal = point_records(records, lambda row: np.load(private / row['scores_file'], allow_pickle=False))
        points = {prefix + '_' + suffix: method.select_point(rows, kind)
                  for prefix, rows in (('old', old_cal), ('stable', new_cal))
                  for suffix, kind in (('maximum', 'maximum'), ('regions', 'regions'))}
        previous_point = read(prior_trial() / 'fitted-point-private.json')
        old.method.require(points['old_maximum']['value'] == previous_point['point']['value'], 'Original maximum point must remain exact')
        write(trial / 'fixed-points-private.json', {'policy': POLICY, 'training': training, 'points': points,
            'head_sha256': digest(private / 'head.safetensors'), 'stable_means_sha256': digest(private / 'stable-fit-means.npy'),
            'new_cases_before_points': len(records), 'new_ai_point_cases_scored': 0, 'new_diagnostic_cases_scored': 0})
        fixed = digest(trial / 'fixed-points-private.json')
        for case in plan['cases']:
            if case['role'] == 'diagnostic' or (case['role'] == 'point' and case['label'] == 1):
                process(case)
        old.method.require(fixed == digest(trial / 'fixed-points-private.json')
            and digest(private / 'head.safetensors') == read(trial / 'fixed-points-private.json')['head_sha256'],
            'Head or point changed after new diagnostics')
    finally:
        journal.close()
    old.method.require(len(records) == 1344 and forwards == 1363, 'Complete native case/probe/repeat counts required')
    outcomes, calibration = [], defaultdict(lambda: defaultdict(set))
    old_outcomes = {r['id']: r for r in read(prior_trial() / 'outcomes-private.json')}
    for row in records:
        if row['role'] == 'fit':
            continue
        _, _, truth = old.pixels_for(plan, row)
        for source, scores in (('old', old_scores(prior_records[row['id']])),
                               ('stable', np.load(private / row['scores_file'], allow_pickle=False))):
            level = method.activation(scores)
            for suffix in ('maximum', 'regions'):
                arm = source + '_' + suffix
                point = points[arm]['value']
                mask = old.method.selected(scores, point, row['width'], row['height'])
                old.method.require(bool(mask.any()) == (level > point), 'Activation must exactly represent the actual display decision')
                metrics = old.old_method.metrics(mask, truth)
                item = {**{k: row[k] for k in ('id', 'role', 'label', 'variant', 'group', 'generator', 'cohort')},
                        'arm': arm, 'metrics': metrics, 'mask_sha256': old.method.sha(mask)}
                if arm == 'old_maximum':
                    old.method.require(metrics == old_outcomes[row['id']]['metrics']
                                       and item['mask_sha256'] == old_outcomes[row['id']]['mask_sha256'], 'Original canonical outcome changed')
                outcomes.append(item)
                if row['role'] == 'point' and row['label'] == 0:
                    if metrics['flag']:
                        calibration[arm]['displayed_region_groups'].add(row['group'])
                    if float(scores.max()) > point:
                        calibration[arm]['raw_maximum_groups'].add(row['group'])
                    if level > point:
                        calibration[arm]['activation_groups'].add(row['group'])
    old.method.require(len(outcomes) == 2016, 'All4 arms and all504 point/diagnostic cases required')
    write(trial / 'outcomes-private.json', outcomes)
    result = summarize(plan, outcomes)
    calibrated = {arm: {k: len(calibration[arm][k]) for k in
                  ('displayed_region_groups', 'raw_maximum_groups', 'activation_groups')} for arm in method.ARMS}
    for arm in ('old_regions', 'stable_regions'):
        old.method.require(calibrated[arm].get('displayed_region_groups', 0) <= 3
            and calibrated[arm].get('displayed_region_groups', 0) == calibrated[arm].get('activation_groups', 0),
            'Region-based points must respect the genuine group budget under actual selection')
    result.update(policy=POLICY, freeze_sha256=digest(trial / 'freeze.json'), cases=1344,
                  backbone_forwards=1363, native_hook_probes=18, native_repeat_forwards=1,
                  head_training=training, head_inference_cases=504, points=points, calibration_group_counts=calibrated,
                  native_journal_sha256=digest(trial / 'native-private.jsonl'), outcome_records=2016,
                  original_canonical_outcomes_unchanged=True, fit_vectors_copied=False,
                  additional_fit_mean_bytes=(private / 'stable-fit-means.npy').stat().st_size,
                  primary_candidate=method.PRIMARY, native_audit_required=True,
                  serving_admitted=False, used_in_confidence=False, outside_human_reviewers=0)
    check(trial)
    write(trial / 'result.json', result)
    print(json.dumps({'completed_cases': 1344, 'all': result['all'], 'calibration': calibrated}), flush=True)


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
