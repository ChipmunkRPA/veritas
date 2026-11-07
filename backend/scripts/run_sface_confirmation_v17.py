"""Prospective fresh-source confirmation of unchanged v16 face-transition method.

Raw clips and real face representations remain in ignored research storage.
No customer database, serving registry, payment or confidence changes.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts import run_sface_transition_v16 as frozen
from scripts import sface_transition_v16 as method

ROOT = Path(__file__).resolve().parents[2]
EXTRA_CODE = ('backend/scripts/run_sface_confirmation_v17.py',
    'backend/scripts/audit_sface_vectors_v17.py', 'backend/scripts/audit_sface_vectors_v16.py',
    'backend/scripts/prepare_face_swap_upgrade_eval.py', 'backend/scripts/prepare_video_upgrade_eval.py',
    'backend/scripts/prepare_face_interval_eval.py', 'backend/scripts/score_face_interval_eval.py',
    'backend/scripts/review_sface_transition_v16.py', 'docs/video-v17-sface-confirmation-protocol.md')
POINT = .11243900750879454


def read(path):
    return json.loads(path.read_text())


def journal(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def imports():
    # Original acquisition helpers are scripts, not installed application modules.
    sys.path.insert(0, str(ROOT / 'backend/scripts'))
    import prepare_face_swap_upgrade_eval as source
    import prepare_video_upgrade_eval as media

    return source, media


def plan(main, trial):
    source, _ = imports()
    trial.mkdir(parents=True, exist_ok=False)
    inventory, prior, hashes = {}, {}, set()
    root = main / 'backend/data/independent-evaluation'

    def visit(value, found):
        if isinstance(value, dict):
            group = value.get('group')
            if isinstance(group, str) and group.startswith('RTFS:'):
                donor = value.get('donor_filename')
                prior[group, donor] = {'group': group, 'donor_filename': donor}
                found.add(group)
            for key, part in value.items():
                if key in {'sha256', 'source_sha256'} and isinstance(part, str) and len(part) == 64:
                    hashes.add(part)
                if isinstance(part, (dict, list)):
                    visit(part, found)
        elif isinstance(value, list):
            for part in value:
                visit(part, found)

    for path in sorted({*root.rglob('*manifest*.json'), *root.rglob('excluded.json')}):
        if path.is_relative_to(trial):
            continue
        value, found = read(path), set()
        visit(value, found)
        if found:
            inventory[str(path.relative_to(main))] = method.digest(path)
    previous = sorted(prior.values(), key=lambda r: (r['group'], r['donor_filename'] or ''))
    if len({r['group'] for r in previous}) < 132:
        raise ValueError('Cumulative source exclusion inventory incomplete')
    metadata = trial / 'sources/metadata'
    metadata.mkdir(parents=True)
    old = root / 'video-v6/sources'
    old_manifest = read(old / 'manifest.json')
    if old_manifest['revision'] != source.REVISION:
        raise ValueError('Pinned original metadata revision changed')
    for name in source.METADATA:
        if method.digest(old / 'metadata' / name) != old_manifest['metadata_sha256'][name]:
            raise ValueError('Metadata cache differs from pinned acquisition')
        shutil.copyfile(old / 'metadata' / name, metadata / name)
    selected = source.plan(metadata, 30, tuple(previous))
    if (len(selected) != 90 or len({r['group'] for r in selected}) != 30
            or {r['group'] for r in selected} & {r['group'] for r in previous}
            or {r['donor_filename'] for r in selected if r['donor_filename']} &
               {r['donor_filename'] for r in previous if r['donor_filename']}):
        raise ValueError('Exact source/donor-disjoint confirmation plan required')
    method.write(trial / 'excluded.json', {'items': previous, 'research_inventory_sha256': inventory,
        'previous_research_media_sha256': sorted(hashes),
        'scope': 'Ignored independent research manifests only; no customer upload inventory.'})
    method.write(trial / 'acquisition-plan.json', {'revision': source.REVISION, 'items': selected,
        'excluded_sha256': method.digest(trial / 'excluded.json'),
        'metadata_sha256': old_manifest['metadata_sha256'],
        'fresh_source_video_ids': True, 'fresh_donor_filenames': True,
        'shared_people_or_pretraining_unknown': True})
    print('planned90 public sources /30 new source-video groups; no model inference', flush=True)


def acquire(main, trial):
    _, media = imports()
    plan = read(trial / 'acquisition-plan.json')
    previous = set(read(trial / 'excluded.json')['previous_research_media_sha256'])
    for name in ('originals', 'matched'):
        (trial / 'sources' / name).mkdir(exist_ok=False)
    items, seen = [], set()
    with ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(lambda r: media.prepare(r, trial / 'sources'), plan['items']):
            for kind in ('original', 'matched'):
                if kind in row:
                    value = row[kind]['sha256']
                    if value in previous or value in seen:
                        row.update(status='duplicate', reason='Prior research or planned source byte duplicate')
                    seen.add(value)
            items.append(row)
            print(row['id'], row['status'], flush=True)
    method.write(trial / 'sources/manifest.json', {'policy': 'chipmunk-video-face-swap/1',
        'revision': plan['revision'], 'items': items,
        'acquisition_plan_sha256': method.digest(trial / 'acquisition-plan.json'),
        'metadata_sha256': plan['metadata_sha256']})
    if len(items) != 90 or any(r['status'] != 'ready' for r in items):
        raise ValueError('Original acquisition retained; no replacement sources or silent retries')
    print('all90 sources ready; byte duplicate checks passed', flush=True)


def prepare(main, trial):
    imports()
    sys.argv = ['prepare_face_interval_eval.py', str(trial / 'sources/manifest.json'),
                str(trial / 'intervals'), '--development-groups', '0', '--hard-controls']
    runpy.run_path(str(ROOT / 'backend/scripts/prepare_face_interval_eval.py'), run_name='__main__')
    panel = read(trial / 'intervals/manifest.json')
    if len(panel['items']) != 120 or any(r['status'] != 'ready' for r in panel['items']):
        raise ValueError('All120 originally planned clips required; preserve failed/short attempts')


def source_code():
    names = set(EXTRA_CODE)
    for path in (ROOT / 'backend/app').rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.json', '.yaml', '.npy', '.j2', '.css'}:
            names.add(str(path.relative_to(ROOT)))
    return {**frozen.code(), **{n: method.digest(ROOT / n) for n in sorted(names)}}


def installed_code(main):
    from scripts.run_image_smaug_v7 import runtime_files

    site = main / 'backend/.venv/lib/python3.12/site-packages'
    return {**runtime_files(main), **{str(p.relative_to(site)): method.digest(p)
        for p in sorted((site / 'timm').rglob('*.py'))}}


def freeze(main, trial):
    frozen.configure(main, trial)
    from app.engine.build_info import config_sha256
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    registry, assets = get_registry(), {}
    for mid in ('dfdc.effnet_b7_555', 'dfdc.effnet_b7_999', 'yunet.2023mar', 'dinov2.small'):
        entry = registry.manifest.model(mid)
        if entry.licence.class_ != 'commercial':
            raise ValueError('All comparison weights require commercial admission')
        assets[mid] = {'license': entry.licence.model_dump(mode='json'),
            'files': {str(p.resolve().relative_to(main)): method.digest(p) for p in ensure_verified(entry)}}
    prior = main / 'backend/data/independent-evaluation/video-v16-sface/operating-point.json'
    if read(prior)['value'] != POINT:
        raise ValueError('Previously published genuine-only point required')
    weight = main / 'models/research/video-v15-sface/sface.onnx'
    if method.digest(weight) != frozen.WEIGHT_SHA:
        raise ValueError('Exact previously reviewed Apache-2.0 weights required')
    inputs = ('excluded.json', 'acquisition-plan.json', 'sources/manifest.json', 'intervals/manifest.json')
    method.write(trial / 'lock.json', {'schema': 'chipmunk.sface-fresh-confirmation-lock/17',
        'source_sha256': source_code(), 'input_packets_sha256': {n: method.digest(trial / n) for n in inputs},
        'installed_source_sha256': installed_code(main), 'models': assets,
        'sface_weight_sha256': frozen.WEIGHT_SHA, 'config_sha256': config_sha256(),
        'point': read(prior), 'planned': 120, 'source_groups': 30,
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'numpy', 'opencv-python-headless')},
        'tools': {n: {'sha256': method.digest(Path(shutil.which(n))), 'version': subprocess.check_output(
            [n, '-version'], text=True).splitlines()[0]} for n in ('ffmpeg', 'ffprobe')},
        'confirmation_inference_before_freeze': 0, 'serving_changed': False})
    print('frozen', method.digest(trial / 'lock.json'), flush=True)


def validate(main, trial):
    frozen.configure(main, trial)
    from app.engine.build_info import config_sha256

    lock = read(trial / 'lock.json')
    if (lock['source_sha256'] != source_code() or lock['config_sha256'] != config_sha256()
            or lock['installed_source_sha256'] != installed_code(main)
            or any(method.digest(trial / n) != h for n, h in lock['input_packets_sha256'].items())
            or any(method.digest(main / n) != h for e in lock['models'].values() for n, h in e['files'].items())
            or method.digest(main / 'models/research/video-v15-sface/sface.onnx') != lock['sface_weight_sha256']
            or any(method.digest(Path(shutil.which(n))) != info['sha256'] for n, info in lock['tools'].items())):
        raise ValueError('Frozen source/input/config/runtime/weight binding changed')
    return lock


def baseline(main, trial):
    validate(main, trial)
    imports()
    sys.argv = ['score_face_interval_eval.py', str(trial / 'intervals/manifest.json'),
                str(trial / 'baseline.jsonl'), '--split', 'test']
    runpy.run_path(str(ROOT / 'backend/scripts/score_face_interval_eval.py'), run_name='__main__')
    validate(main, trial)
    values = journal(trial / 'baseline.jsonl')[1:]
    if len(values) != 120 or any(r.get('check_status') not in {'ok', 'not_applicable'} for r in values):
        raise ValueError('Complete fresh baseline required; failures/skips retained')


def candidate(main, trial):
    lock = validate(main, trial)
    import cv2
    import torch

    from app.core.config import settings
    from app.engine.models import get_model_manager

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    if not torch.backends.mps.is_available():
        raise ValueError('Frozen current-check MPS runtime required')
    header, values = frozen.rows(trial / 'baseline.jsonl')
    planned = read(trial / 'intervals/manifest.json')['items']
    by_id = {r['id']: r for r in values}
    if (len(values) != 120 or set(by_id) != {r['id'] for r in planned}
            or header['manifest_sha256'] != method.digest(trial / 'intervals/manifest.json')
            or any(r.get('check_status') not in {'ok', 'not_applicable'} for r in values)):
        raise ValueError('Exact completed fresh baseline/source receipt required')
    out, features, items = trial / 'candidate.jsonl', trial / 'private-features-confirmation', []
    if out.exists():
        raise ValueError('Preserve original attempt; no overwrite/resume')
    features.mkdir(exist_ok=False)
    weight = str(main / 'models/research/video-v15-sface/sface.onnx')
    recognizer = cv2.FaceRecognizerSF.create(weight, '', cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
    graph = cv2.dnn.readNetFromONNX(weight)
    graph.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    graph.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    manager = get_model_manager()
    with manager.lease(('yunet.2023mar',), prefer='cpu') as cpu, manager.lease(('dinov2.small',), prefer='mps') as accelerated:
        for index, source in enumerate(planned):
            row = by_id[source['id']]
            if any(row.get(k) != v for k, v in source.items()):
                raise ValueError('Fresh baseline metadata differs from prepared source')
            item = {**source, 'role': 'confirmation', 'legacy': row, 'legacy_range_s': method.legacy_range(row),
                'source_path': str(trial / 'intervals' / source['path']), 'historical_current_windows': []}
            items.append(item)
            begin = time.monotonic()
            record = {'id': item['id'], 'source_sha256': item['sha256']}
            try:
                value = frozen.score(item, cpu.get('yunet.2023mar').obj, accelerated.get('dinov2.small').obj,
                    recognizer, graph, settings.face_min_size)
                # Only the old reference-comparison sidecar uses the placeholder;
                # it has no role in feature extraction/window/range calculations.
                # A fresh clip has no historical parity reference. Never claim one.
                value.pop('baseline_parity_passed')
                packet = features / (item['id'] + '.json')
                method.write(packet, {'source_sha256': item['sha256'], 'frame_inputs': value.pop('frame_inputs'),
                    'scope': 'Private real-face features; no public distribution approval.'})
                record.update(value, fresh_current_recomputed=True, historical_current_parity_applicable=False,
                    private_feature_packet_sha256=method.digest(packet))
            except Exception as error:
                record.update(status='failed', error_type=type(error).__name__, reason=str(error))
                traceback.print_exc()
            record['seconds'] = time.monotonic() - begin
            frozen.append(out, record)
            gc.collect()
            torch.mps.empty_cache()
            print(index + 1, '/120', item['id'], record['status'], flush=True)
    method.write(trial / 'private-confirmation-panel.json', {'items': items, 'lock_sha256': method.digest(trial / 'lock.json')})
    validate(main, trial)
    if lock['point']['value'] != POINT:
        raise ValueError('Fixed genuine-only development point changed')


def review(main, trial):
    validate(main, trial)
    from app.detectors.video.face_sequence import REVIEW_POINT
    from scripts.review_sface_transition_v16 import interval

    lock = read(trial / 'lock.json')
    items = read(trial / 'private-confirmation-panel.json')['items']
    values = journal(trial / 'candidate.jsonl')
    cases = method.outcomes(items, values, lock['point'], REVIEW_POINT)
    for row in cases:
        row.pop('baseline_parity_passed')
    current, candidate = [method.metrics(cases, arm) for arm in ('current', 'candidate')]
    profiles = {role: read(trial / ('profile-' + role) / 'profile.json') for role in ('baseline', 'candidate')}
    families = {family: {arm: method.metrics([r for r in cases if r['family'] == family], arm)
        for arm in ('current', 'candidate')} for family in sorted({r['family'] for r in cases})}
    requirements = {
        'all120_cases30_source_groups_accounted': len(cases) == 120 and len({r['group'] for r in cases}) == 30,
        'no_candidate_failure': all(r['status'] in {'ok', 'not_applicable'} for r in values),
        'complete_without_guard': all(p['child_exit_code'] == 0 and p['guard'] is None for p in profiles.values()),
        'all_current_checks_freshly_recomputed': all(r.get('fresh_current_recomputed') for r in values),
        'all_native_alignments_inputs_features_match': all(r.get('native_alignment_all_exact') for r in values),
        'more_correct_edit_localizations': candidate['localized_edits'] > current['localized_edits'],
        'genuine_total_flags_at_most_current_plus_one': candidate['genuine_review_flags'] <= current['genuine_review_flags'] + 1,
        'ordinary_edit_flags_at_most_current_plus_one': candidate['ordinary_edit_flags'] <= current['ordinary_edit_flags'] + 1,
        'mean_edit_iou_not_lower': candidate['mean_edit_iou'] >= current['mean_edit_iou'],
        'no_swap_family_localization_loss': all(v['candidate']['localized_edits'] >= v['current']['localized_edits']
            for family, v in families.items() if family in {'inswapper', 'uniface'})}
    summary = {'schema': 'chipmunk.sface-fresh-confirmation-result/17', 'lock_sha256': method.digest(trial / 'lock.json'),
        'point': lock['point'], 'current': current, 'candidate': candidate, 'families': families,
        'requirements': requirements, 'qualifies_for_local_integration_testing': all(requirements.values()),
        'paired_iou': interval(cases, 'iou', 1), 'paired_localization_rate': interval(cases, 'localized', 1),
        'paired_genuine_flag_rate': interval(cases, 'flags', 0), 'profiles': profiles,
        'fresh_source_and_donor_ids': True, 'fresh_people_or_upstream_training_proven': False,
        'serving_changed': False, 'overall_confidence_changed': False, 'outside_human_reviewers': 0,
        'scope': 'Fresh source-video IDs/donor filenames; bounded face-transition ranges and review flags, not calibrated AI/identity probability. Entire baseline rerun, unchanged v16 method and point; no historical reference parity claimed.'}
    method.write(trial / 'cases.json', cases)
    method.write(trial / 'summary.json', summary)
    print(json.dumps({'current': current, 'candidate': candidate, 'requirements': requirements}), flush=True)


def profile(main, trial, role):
    import psutil

    watched = ('scripts.run_sface_transition_v16', 'scripts.run_sface_confirmation_v17', 'scripts.run_image_smaug_v7',
               'scripts.run_audio_separation_v14', 'scripts.repeat_audio_separation_v14')
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info['cmdline'] or []
        if process.pid != os.getpid() and any(m in args for m in watched):
            raise ValueError('Heavy evaluations must run sequentially')
    folder = trial / ('profile-' + role)
    folder.mkdir(exist_ok=False)
    begin, maximum, errors, guard, tree = time.monotonic(), 0, 0, None, []
    with (folder / 'worker.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-m', 'scripts.run_sface_confirmation_v17', role, str(main), str(trial)],
            stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                tree = [process, *process.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in tree)
                maximum = max(maximum, rss)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                errors += 1
                rss = 0
            if (rss > 6 * 1024**3 or psutil.virtual_memory().available < 1536 * 1024**2
                    or shutil.disk_usage(trial).free < 1536 * 1024**2 or time.monotonic() - begin > 7200):
                guard = 'memory_disk_or_timeout'
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
    result = {'role': role, 'child_exit_code': status, 'guard': guard, 'sample_errors': errors,
        'sampled_tree_rss_lower_bound_bytes': maximum, 'duration_s': time.monotonic() - begin}
    method.write(folder / 'profile.json', result)
    print(json.dumps(result), flush=True)
    if status or guard:
        raise RuntimeError('Original incomplete/failed trial retained; no silent restart')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'acquire', 'prepare', 'freeze', 'baseline', 'candidate', 'review', 'profile'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--role', choices=('acquire', 'prepare', 'baseline', 'candidate'))
    args = parser.parse_args()
    if args.mode == 'profile':
        if args.role is None:
            parser.error('--role required')
        profile(args.main, args.trial, args.role)
    else:
        globals()[args.mode](args.main, args.trial)
