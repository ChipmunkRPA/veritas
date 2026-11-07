"""Frozen CPU-only region calibration from exact previously consumed image maps."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from scripts import image_patch_drift_v8 as display
from scripts import image_region_calibration_v9 as method
from scripts import run_image_patch_drift_v8 as prior
from scripts.review_image_patch_drift_v8 import display_reference, gates, pixel_reference, summarize

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_FREEZE = '9229bc0afd16c4e8ff35a11602ad01bd6e022e54e59fe6b59e99b25b588ea8d3'
CODE = ('backend/scripts/image_region_calibration_v9.py', 'backend/scripts/run_image_region_calibration_v9.py',
    'backend/tests/test_image_region_calibration_v9.py', 'docs/image-region-calibration-v9-protocol.md')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def previous(main):
    return main / 'backend/data/independent-evaluation/image-v8-patch-drift-attempt3'


def code():
    return {name: sha(ROOT / name) for name in CODE}


def source_rows(main):
    origin = previous(main)
    return [json.loads(line) for role in ('calibration', 'development')
        for line in (origin / role / 'predictions.jsonl').read_text().splitlines()]


def prepare(main, trial):
    lock, panel = prior.check(main, previous(main), PREVIOUS_FREEZE)
    audit = json.loads((previous(main) / 'development-review.json').read_text())
    if audit['verified_cases'] != 312 or audit['private_native_probes'] != 12:
        raise ValueError('Complete original native/source/grid audit required')
    rows = source_rows(main)
    identities = {(r['role'], r['id'], r['condition']) for r in rows}
    if len(rows) != 312 or identities != {(r['role'], r['id'], r['condition']) for r in panel}:
        raise ValueError('Exact complete original response panel required')
    arrays = {}
    for row in rows:
        if row['status'] != 'ok':
            raise ValueError('No incomplete original response may be dropped')
        name = row['role'] + '/' + row['arrays']
        if sha(previous(main) / name) != row['arrays_sha256']:
            raise ValueError('Original native response archive changed')
        arrays[name] = row['arrays_sha256']
    trial.mkdir(exist_ok=False)
    frozen = {'schema': 'chipmunk.image-region-calibration-freeze/9', 'code_sha256': code(),
        'prior_development_freeze_sha256': PREVIOUS_FREEZE,
        'prior_native_freeze_sha256': sha(previous(main) / 'native-freeze.json'),
        'prior_complete_audit_sha256': sha(previous(main) / 'development-review.json'),
        'prior_panel_sha256': sha(previous(main) / 'development-panel.json'),
        'prior_journals_sha256': {role: sha(previous(main) / role / 'predictions.jsonl')
            for role in ('calibration', 'development')}, 'cached_archives_sha256': arrays,
        'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'opencv-python-headless', 'Pillow')},
        'component_policy': lock['component_policy'], 'arm': 'raw', 'conditions': list(prior.FORMATS),
        'planned': {'calibration': 120, 'development': 192}, 'calibration_minimum_eligible_per_format': 48,
        'point_rule': 'second-largest genuine maximum feasible connected-region level; strict greater than',
        'minimum_additional_overlapping_edits': 4, 'maximum_additional_genuine_highlights': 1,
        'minimum_mean_iou_delta': 0, 'minimum_family_overlap_delta': 0,
        'new_project_critical_levels_before_freeze': 0, 'new_neural_forwards': 0,
        'source_role': 'All312 inputs already consumed development; never fresh confirmation.',
        'serving_changed': False, 'overall_confidence_weight': 0}
    write(trial / 'freeze.json', frozen)
    print(json.dumps({'freeze_sha256': sha(trial / 'freeze.json'), 'cached_archives_verified': len(arrays),
        'project_critical_levels_computed': 0}), flush=True)


def check(main, trial, expected):
    if sha(trial / 'freeze.json') != expected:
        raise ValueError('Independently recorded new freeze SHA256 required')
    lock = json.loads((trial / 'freeze.json').read_text())
    if lock['code_sha256'] != code():
        raise ValueError('Frozen region-calibration implementation changed')
    _, panel = prior.check(main, previous(main), PREVIOUS_FREEZE)
    for role, value in lock['prior_journals_sha256'].items():
        if sha(previous(main) / role / 'predictions.jsonl') != value:
            raise ValueError('Original consumed response journal changed')
    for name, value in [('development-review.json', lock['prior_complete_audit_sha256']),
            ('development-panel.json', lock['prior_panel_sha256']), ('native-freeze.json', lock['prior_native_freeze_sha256'])]:
        if sha(previous(main) / name) != value:
            raise ValueError('Original source/native/audit contract changed')
    if {name: importlib.metadata.version(name) for name in lock['packages']} != lock['packages']:
        raise ValueError('Frozen numerical runtime changed')
    return lock, panel


def cached(main, row, lock):
    name = row['role'] + '/' + row['arrays']
    expected = lock['cached_archives_sha256'][name]
    path = previous(main) / name
    if sha(path) != expected or expected != row['arrays_sha256']:
        raise ValueError('Original response bytes changed')
    with np.load(path, allow_pickle=False) as archive:
        floating, reliable = archive['raw_screening'].copy(), archive['reliable'].copy()
        if hashlib.sha256(archive['filtered'].tobytes()).hexdigest() != row['native']['filtered_map_float32_sha256']:
            raise ValueError('Original native filtered-map commitment differs')
    if floating.shape != reliable.shape or floating.dtype != np.float32 or reliable.dtype != bool:
        raise ValueError('Unexpected original grid/reliability dtype or shape')
    return floating, reliable


def measure(floating, reliable, point, geometry, truth, policy):
    height, width = geometry[:2]
    metrics, grids, _ = display.predict_regions(floating, reliable, point, width, height, truth)
    encoded, selected = display_reference(floating, reliable, point, policy, width, height)
    np.testing.assert_array_equal(grids['display_grid'], encoded)
    np.testing.assert_array_equal(grids['selected_grid'], selected)
    if pixel_reference(selected, truth) != metrics:
        raise ValueError('Independent new display/source-pixel replay differs')
    return metrics, {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in grids.items()}


def run(main, trial, expected, role, points_sha=None):
    lock, panel = check(main, trial, expected)
    folder = trial / role
    folder.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    points = None
    if role == 'development':
        if not points_sha or sha(trial / 'points.json') != points_sha:
            raise ValueError('Published genuine-only point receipt required')
        points = json.loads((trial / 'points.json').read_text())
        if (points['freeze_sha256'] != expected or not points['passed']
                or points['calibration_journal_sha256'] != sha(trial / 'calibration/predictions.jsonl')):
            raise ValueError('Original successful calibration receipt required')
    originals = [r for r in source_rows(main) if r['role'] == role]
    items = {(r['id'], r['condition']): r for r in panel if r['role'] == role}
    with (folder / 'predictions.jsonl').open('x') as stream:
        for index, row in enumerate(originals):
            start = time.monotonic()
            item = items[row['id'], row['condition']]
            result = {k: row[k] for k in ('id', 'source_id', 'family', 'label', 'role', 'condition',
                'input_sha256', 'input_rgb_sha256', 'geometry', 'arrays_sha256')}
            result.update(status='failed', freeze_sha256=expected)
            try:
                floating, reliable = cached(main, row, lock)
                response = method.critical_level(floating, reliable, lock['component_policy']['minimum_region_fraction'])
                oracle = method.verify_critical(floating, reliable, lock['component_policy']['minimum_region_fraction'], response)
                result.update(response=response, independent_boundary=oracle,
                    screening_grid_shape=list(floating.shape),
                    raw_screening_sha256=hashlib.sha256(floating.tobytes()).hexdigest(),
                    reliable_grid_sha256=hashlib.sha256(reliable.tobytes()).hexdigest())
                if role == 'development':
                    _, _, truth = prior.source(prior.paths(main)[role], item)
                    point = points['points'][row['condition']]
                    metrics, hashes = measure(floating, reliable, point, row['geometry'], truth, lock['component_policy'])
                    region_exists = response is not None and response['critical_level'] > point
                    if region_exists != (metrics['predicted_positive_pixels'] > 0):
                        raise ValueError('Critical level and actual displayed-region eligibility differ')
                    result.update(baseline=item['baseline'], candidate=metrics, grid_sha256=hashes,
                        historical_v8_raw=row['arms']['raw']['metrics'])
                result['status'] = 'ok'
            except Exception as error:
                result.update(error_type=type(error).__name__, error=str(error)[:600])
            result['duration_s'] = time.monotonic() - start
            stream.write(json.dumps(result, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps({'role': role, 'complete': index + 1, 'planned': len(originals), 'status': result['status']}), flush=True)
    if lock['code_sha256'] != code():
        raise ValueError('Frozen implementation changed during computation')


def records(trial, role):
    return [json.loads(line) for line in (trial / role / 'predictions.jsonl').read_text().splitlines()]


def fit(main, trial, expected):
    lock, panel = check(main, trial, expected)
    cv2.setNumThreads(1)
    rows = records(trial, 'calibration')
    if (len(rows) != 120 or len({(r['id'], r['condition']) for r in rows}) != 120
            or {(r['id'], r['condition']) for r in rows} != {(r['id'], r['condition']) for r in panel if r['role'] == 'calibration'}
            or any(r['status'] != 'ok' or r['label'] != 0 or r['freeze_sha256'] != expected for r in rows)):
        raise ValueError('All120 original-only critical levels must complete before fitting')
    points = {condition: method.point_from_genuine([r['response']['critical_level'] if r['response'] else None
        for r in rows if r['condition'] == condition]) for condition in prior.FORMATS}
    old = {(r['id'], r['condition']): r for r in source_rows(main) if r['role'] == 'calibration'}
    checks, flags = [], {condition: 0 for condition in prior.FORMATS}
    for row in rows:
        floating, reliable = cached(main, old[row['id'], row['condition']], lock)
        height, width = row['geometry'][:2]
        metrics, hashes = measure(floating, reliable, points[row['condition']], row['geometry'],
            np.zeros((height, width), bool), lock['component_policy'])
        predicted = row['response'] is not None and row['response']['critical_level'] > points[row['condition']]
        if predicted != (metrics['predicted_positive_pixels'] > 0):
            raise ValueError('Genuine critical order statistic differs from displayed regions')
        flags[row['condition']] += int(predicted)
        checks.append({'id': row['id'], 'condition': row['condition'], 'metrics': metrics, 'grid_sha256': hashes})
    if any(v > 1 for v in flags.values()):
        raise ValueError('Original-only displayed-region budget exceeded')
    write(trial / 'points.json', {'schema': 'chipmunk.image-region-calibration-points/9',
        'freeze_sha256': expected, 'calibration_journal_sha256': sha(trial / 'calibration/predictions.jsonl'),
        'points': points, 'genuine_displayed_regions': flags, 'display_checks': checks,
        'edited_inputs_used': 0, 'passed': True, 'not_ai_probabilities': True})
    print(json.dumps({'points_sha256': sha(trial / 'points.json'), 'points': points, 'genuine_displayed_regions': flags}), flush=True)


def review(main, trial, expected, points_sha):
    lock, panel = check(main, trial, expected)
    cv2.setNumThreads(1)
    if sha(trial / 'points.json') != points_sha:
        raise ValueError('Original point receipt changed')
    points = json.loads((trial / 'points.json').read_text())
    if points['freeze_sha256'] != expected or points['calibration_journal_sha256'] != sha(trial / 'calibration/predictions.jsonl'):
        raise ValueError('Original successful calibration binding changed')
    old = {(r['role'], r['id'], r['condition']): r for r in source_rows(main)}
    planned = {(r['role'], r['id'], r['condition']): r for r in panel}
    checks = {(r['id'], r['condition']): r for r in points['display_checks']}
    verified, calibration_flags = 0, {c: 0 for c in prior.FORMATS}
    for role in ('calibration', 'development'):
        measured = records(trial, role)
        expected_keys = {key for key in planned if key[0] == role}
        keys = {(r['role'], r['id'], r['condition']) for r in measured}
        if len(measured) != lock['planned'][role] or keys != expected_keys or len(keys) != len(measured):
            raise ValueError('Missing/duplicate outcome in complete independent review')
        for row in measured:
            key = role, row['id'], row['condition']
            original, item = old[key], planned[key]
            if row['status'] != 'ok' or row['freeze_sha256'] != expected or any(row[k] != original[k]
                    for k in ('id', 'source_id', 'family', 'label', 'role', 'condition', 'input_sha256',
                        'input_rgb_sha256', 'geometry', 'arrays_sha256')):
                raise ValueError('Frozen source/role/response metadata differs')
            floating, reliable = cached(main, original, lock)
            if (row['screening_grid_shape'] != list(floating.shape)
                    or row['raw_screening_sha256'] != hashlib.sha256(floating.tobytes()).hexdigest()
                    or row['reliable_grid_sha256'] != hashlib.sha256(reliable.tobytes()).hexdigest()):
                raise ValueError('New statistic is not bound to the exact prior response grid')
            oracle = method.verify_critical(floating, reliable, lock['component_policy']['minimum_region_fraction'], row['response'])
            if oracle != row['independent_boundary']:
                raise ValueError('Separate plateau certificate differs')
            data, rgb, truth = prior.source(prior.paths(main)[role], item)
            variant, pixels = prior.variant(rgb, data, row['condition'])
            if (hashlib.sha256(variant).hexdigest() != row['input_sha256']
                    or hashlib.sha256(pixels.tobytes()).hexdigest() != row['input_rgb_sha256']):
                raise ValueError('Source image and previously scored tensor input differ')
            point = points['points'][row['condition']]
            encoded, selected = display_reference(floating, reliable, point, lock['component_policy'],
                pixels.shape[1], pixels.shape[0])
            independent = pixel_reference(selected, truth)
            grid_sha = {name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in
                [('display_grid', encoded), ('reliable', reliable), ('selected_grid', selected)]}
            target = checks[row['id'], row['condition']] if role == 'calibration' else row
            if target['grid_sha256'] != grid_sha or target['metrics' if role == 'calibration' else 'candidate'] != independent:
                raise ValueError('Independent component/source-pixel result differs')
            eligible = row['response'] is not None and row['response']['critical_level'] > point
            if eligible != (independent['predicted_positive_pixels'] > 0):
                raise ValueError('Critical statistic differs from retained/displayed component')
            if role == 'calibration':
                calibration_flags[row['condition']] += int(eligible)
            elif row['baseline'] != item['baseline'] or row['historical_v8_raw'] != original['arms']['raw']['metrics']:
                raise ValueError('Original current/historical comparator changed')
            verified += 1
        if role == 'calibration':
            for condition in prior.FORMATS:
                refit = method.point_from_genuine([r['response']['critical_level'] if r['response'] else None
                    for r in measured if r['condition'] == condition])
                if refit != points['points'][condition]:
                    raise ValueError('Genuine-only point refit differs')
    if calibration_flags != points['genuine_displayed_regions'] or any(v > 1 for v in calibration_flags.values()):
        raise ValueError('Genuine-only complete-region budget differs')
    rows = records(trial, 'development')
    expected_ids = {(r['id'], r['condition']) for r in panel if r['role'] == 'development'}
    complete = (len(rows) == 192 and len({(r['id'], r['condition']) for r in rows}) == 192
        and {(r['id'], r['condition']) for r in rows} == expected_ids and all(r['status'] == 'ok' for r in rows))
    profiles = {role: json.loads((trial / (role + '-profile') / 'profile.json').read_text())
        for role in ('calibration', 'development')}
    complete = complete and all(p['guard'] is None and p['child_exit_code'] == 0 for p in profiles.values())
    if not complete:
        raise ValueError('Retain incomplete trial; no reduced-denominator qualification')
    conditions = {}
    for condition in prior.FORMATS:
        subset = [r for r in rows if r['condition'] == condition]
        compatible = [{**r, 'arms': {'region': {'metrics': r['candidate']}}} for r in subset]
        baseline, candidate = summarize(compatible, 'baseline'), summarize(compatible, 'region')
        requirements = gates(baseline, candidate, complete, verified == 312)
        conditions[condition] = {'baseline': baseline, 'candidate': candidate,
            'requirements': requirements, 'passed': all(requirements.values())}
    result = {'schema': 'chipmunk.image-region-calibration-review/9', 'freeze_sha256': expected,
        'points_sha256': points_sha, 'verified_calibration_inputs': 120, 'verified_development_inputs': 192,
        'independent_plateau_and_pixel_replays': verified, 'genuine_displayed_region_refits': calibration_flags,
        'conditions': conditions, 'profiles': profiles, 'new_neural_forwards': 0,
        'source_role': 'Consumed-source geometry/calibration development only.',
        'serving_changed': False, 'overall_confidence_weight': 0}
    write(trial / 'review.json', result)
    print(json.dumps(result), flush=True)


def profile(main, trial, expected, role, points_sha=None):
    import psutil

    folder = trial / (role + '-profile')
    folder.mkdir(exist_ok=False)
    command = [sys.executable, '-m', 'scripts.run_image_region_calibration_v9', role, str(main), str(trial),
        '--freeze-sha256', expected]
    if points_sha:
        command += ['--points-sha256', points_sha]
    start, maximum, guard, errors = time.monotonic(), 0, None, 0
    with (folder / 'worker.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            tree, rss = [process], 0
            try:
                tree += process.children(recursive=True)
                rss = sum(p.memory_info().rss for p in tree)
                maximum = max(maximum, rss)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                errors += 1
            if (rss > 6 * 1024**3 or psutil.virtual_memory().available < 1536 * 1024**2
                    or shutil.disk_usage(trial).free < 1536 * 1024**2 or time.monotonic() - start > 1800):
                guard = 'memory_disk_or_time'
                for p in reversed(tree):
                    try:
                        p.terminate()
                    except psutil.NoSuchProcess:
                        errors += 1
                break
            time.sleep(.25)
        try:
            status = child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            status = child.wait()
    result = {'role': role, 'child_exit_code': status, 'guard': guard,
        'sampled_tree_rss_lower_bound_bytes': maximum, 'sample_errors': errors, 'duration_s': time.monotonic() - start}
    write(folder / 'profile.json', result)
    print(json.dumps(result), flush=True)
    if status or guard:
        raise RuntimeError('Retain original failed/guarded attempt; no silent restart')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'calibration', 'fit', 'development', 'review',
        'profile-calibration', 'profile-development'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--freeze-sha256')
    parser.add_argument('--points-sha256')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.main, args.trial)
    elif not args.freeze_sha256:
        parser.error('Independently recorded new freeze required')
    elif args.stage == 'fit':
        fit(args.main, args.trial, args.freeze_sha256)
    elif args.stage == 'review':
        review(args.main, args.trial, args.freeze_sha256, args.points_sha256)
    elif args.stage.startswith('profile-'):
        profile(args.main, args.trial, args.freeze_sha256, args.stage.removeprefix('profile-'), args.points_sha256)
    else:
        run(args.main, args.trial, args.freeze_sha256, args.stage, args.points_sha256)
