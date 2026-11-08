"""Prospectively frozen scene-context comparison on consumed video sources."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import background_context_v25 as method
from scripts import genuine_edit_materializer_v24 as ordinary
from scripts import materialize_video_specialists_v18 as media
from scripts import review_background_context_v25 as independent
from scripts import score_video_specialists_v20 as previous

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'backend/data/independent-evaluation/video-specialists-v18'
ORDINARY = ROOT / 'backend/data/independent-evaluation/video-genuine-edits-v24'
write, sha, require = ordinary.previous.inventory.write, ordinary.previous.inventory.sha, method.require


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


POLICY = {'schema': 'chipmunk.video-background-context/25',
    'point': 'Only80 genuine fusion_cal clips in40 source components; matched common-assessed group budget from fixed raw appearance point. Minimum25 groups. Strict greater-than. AI calibration clips are not planned/scored.',
    'diagnostic': 'All160 consumed confirmation clips (80 swaps,40 original,40 ordinary controls) plus all40 v24 copy-only scene substitutions. v24 pair links join confirmation sources into20 components. No fresh sources.',
    'descriptor': 'Exact640x480 RGB;12x16 cells of40x40 pixels,float64 mean divided by255. Exclude all cells intersecting any face box expanded25% per edge,common to the exact window. Require25% background cells.',
    'temporal': 'Reuse original frozen three-middle/two-flank centers and consecutive triplets. L1 change on common background;max(0,min(mid-left,mid-right)-.35*left-right).',
    'primary': 'context: cached source-bound appearance raw minus2*background change. Cosine appearance range2 and RGB change range1 motivate the fixed multiplier; no coefficient fitting or selection.',
    'arms': ['current-fixed', 'appearance-matched', 'context'],
    'control': 'appearance-matched uses own genuine-only point without context; isolate point refitting from new information. No probabilistic classifier or AI confidence.',
    'parity': 'First eligible hash-ordered calibration and confirmation clip: re-execute every recorded SFace alignment/vector and supported appearance unit exactly. Other face measurements reused under original source/journal/model commitments. No new face selection.',
    'audit': 'Every newly read native frame equals original source/cached input RGB and PTS;all new block means independently checked via integer integral sums within1e-12. Every temporal value checked by cell-wise scalar transcription within1e-12. Reconstruct saved arrays,points,masks/intervals/counts separately.',
    'continuation': 'Primary must beat both references on usefulAI intervals and meanIoU,lose no already-useful edit,add no original/ordinary false-flag components,reduce v24 scene-cut flags,and have no extra wholly unassessed files. All planned inputs/errors remain.',
    'limits': 'Inspection/localization research,not AI authorship. Consumed sources/publisher truth;no lip-sync,pixel-edit or whole-video authenticity claim. No serving admission without fresh sources and full-app/export tests.',
    'guard': {'seconds': 900, 'rss_gib': 6, 'available_mib': 1536, 'disk_mib': 1536},
    'serving_admitted': False, 'used_in_confidence': False}
PINS = ('backend/scripts/background_context_v25.py', 'backend/scripts/run_background_context_v25.py',
    'backend/scripts/review_background_context_v25.py', 'backend/scripts/background_context_runtime_v25.py',
    'backend/tests/test_background_context_v25.py', 'docs/video-background-context-v25-protocol.md',
    'backend/scripts/measure_video_specialists_v18.py', 'backend/scripts/score_video_specialists_v20.py',
    'backend/scripts/genuine_edit_materializer_v24.py', 'backend/app/detectors/video/face_appearance.py') + previous.source_audit.CODE


def identities():
    previous.source_audit.check(SOURCE)
    native = read(SOURCE / 'v20-freeze.json')
    # Do not alter/relax v20's repository-wide check. This new study binds only
    # the unchanged dependencies it actually executes plus immutable old data.
    for path in PINS:
        if path in native['code_sha256']:
            require(sha(ROOT / path) == native['code_sha256'][path], 'Used historical video dependency changed')
    for role in ('fusion_cal', 'confirmation'):
        receipt = read(SOURCE / f'{role}-v20-native.json')
        require(sha(SOURCE / f'{role}-v20-native.jsonl') == receipt['journal_sha256'], 'Original native journal changed')
        require(lines(SOURCE / f'{role}-v20-native.jsonl')[0]['freeze_sha256'] == sha(SOURCE / 'v20-freeze.json'), 'Original source freeze changed')
    require(sha(ORDINARY / 'native.jsonl') == read(ORDINARY / 'native-receipt.json')['journal_sha256'], 'Ordinary native journal changed')
    model = native['models']['opencv.sface.2021dec']
    require(model['licence']['spdx'] == 'Apache-2.0' and all(sha(ROOT / p) == h for p, h in model['files'].items()), 'Permitted original SFace asset required')
    require(all(sha(Path(shutil.which(k))) == h for k, h in native['binaries_sha256'].items()), 'Original source decoder binaries changed')
    bound = [SOURCE / n for n in ('v20-freeze.json', 'scoring-input-plan-v2.json', 'source-frame-audit-v1.jsonl',
        'fusion_cal-v20-native.json', 'fusion_cal-v20-native.jsonl', 'confirmation-v20-native.json', 'confirmation-v20-native.jsonl')]
    bound += [ORDINARY / n for n in ('source-plan.json', 'input-plan.jsonl', 'baselines.json', 'source-audit.json',
        'scoring-freeze.json', 'native-receipt.json', 'native.jsonl', 'readonly-audit-with-clock.json')]
    return {'policy': POLICY, 'code_sha256': {p: sha(ROOT / p) for p in dict.fromkeys(PINS)},
            'inputs_sha256': {str(p.relative_to(ROOT)): sha(p) for p in bound}, 'model': model,
            'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'opencv-python-headless', 'Pillow')},
            'binaries': native['binaries_sha256']}


def plans():
    rows = read(SOURCE / 'scoring-input-plan-v2.json')['files']
    selected = [r for r in rows if r['role'] == 'confirmation' or r['role'] == 'fusion_cal' and r['label'] == 0]
    require(len(selected) == 240, 'All planned80 genuine point and160 diagnostic source cases required')
    require(not ({r['scoring_component'] for r in selected if r['role'] == 'fusion_cal'} &
                 {r['scoring_component'] for r in selected if r['role'] == 'confirmation'}), 'Source components cross roles')
    return {r['id']: r for r in selected}


def prepare(trial):
    current = identities()
    selected = plans()
    trial.mkdir(exist_ok=False)
    current['case_counts'] = {'genuine_point': 80, 'diagnostic': 160, 'ordinary_scene_edits': 40}
    current['planned_ids_sha256'] = hashlib.sha256(json.dumps(sorted(selected)).encode()).hexdigest()
    write(trial / 'freeze.json', current)
    write(ROOT / 'docs/evaluation-panels/video-background-context-v25-freeze.json', {
        **{k: v for k, v in current.items() if k != 'inputs_sha256'},
        'input_commitments': {Path(p).name + ':' + hashlib.sha256(p.encode()).hexdigest()[:12]: h for p, h in current['inputs_sha256'].items()},
        'private_freeze_sha256': sha(trial / 'freeze.json'), 'new_project_pixel_measurements': 0})


def check(trial):
    lock = read(trial / 'freeze.json')
    require(all(lock[k] == v for k, v in identities().items()), 'Prospectively frozen method/assets/source records changed')
    return plans()


def cached(role):
    return {r['id']: r for r in lines(SOURCE / f'{role}-v20-native.jsonl')[1:]}


def check_points(trial):
    complete = read(trial / 'point-fit-complete.json')
    require(complete['points_sha256'] == sha(trial / 'points.json')
            and complete['journal_sha256'] == sha(trial / 'point.jsonl')
            and complete['freeze_sha256'] == sha(trial / 'freeze.json'),
            'Genuine-only fitted points or their completed source journal changed')


def measure(case, plan, old, source_hashes, private, recognizer=None):
    import cv2

    from app.detectors.video import face_appearance as appearance
    from app.detectors.video.decode import FrameTable, decode_frames
    from scripts.measure_video_specialists_v18 import fixed_change

    require(case['planned']['sha256'] == old['input_sha256'] == plan['sha256']
        and old['native_table_sha256'] == plan['source_geometry']['pts_table_sha256'], 'Cached appearance input/clock changed')
    windows = [u for u in plan['units'] if u['view'] == 'face-appearance']
    scheduled = sorted({i for u in windows for key in ('sampled_frames', 'context_frames') for i in u['location'][key]})
    original = {r['frame_index']: r for r in old['frame_inputs']}
    table, values, boxes, proofs, vectors = FrameTable.from_json(case['table']), {}, {}, [], {}
    maximum, forwards = 0., 0
    for frame in decode_frames(case['path'], table, scheduled, timeout_s=120):
        i, rgb = frame.index, frame.rgb
        require(i not in values and ordinary.digest(rgb) == source_hashes[i] == original[i]['source_rgb_sha256']
                and Fraction(table.time_base) * frame.pts == Fraction(original[i]['time_base']) * original[i]['pts'], 'Native source RGB or PTS changed')
        descriptor = method.describe(rgb)
        difference = float(np.abs(descriptor - independent.describe(rgb)).max())
        require(difference <= 1e-12, 'Independent native block arithmetic failed')
        maximum = max(maximum, difference)
        values[i] = descriptor
        proofs.append({'frame': i, 'rgb_sha256': source_hashes[i], 'pts': frame.pts, 'time_base': table.time_base,
                       'descriptor_sha256': ordinary.digest(descriptor), 'integral_max_difference': difference})
        if original[i].get('status') == 'ok':
            boxes[i] = original[i]['bbox']
        if recognizer is not None and original[i].get('appearance_status') == 'ok':
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            detection = appearance.face_row(original[i]['landmarks'])
            detection[0, :4] = original[i]['bbox']
            aligned = recognizer.alignCrop(bgr, detection)
            require(ordinary.digest(aligned) == original[i]['appearance_crop_bgr_sha256'], 'Original SFace alignment changed')
            vector = recognizer.feature(aligned).reshape(-1).copy()
            forwards += 1
            require(ordinary.digest(vector) == original[i]['appearance_feature_sha256'], 'Original SFace native feature changed')
            vectors[i] = vector
    require(set(values) == set(scheduled), 'Every planned background source frame must remain')
    units, old_units = [], {r['id']: r for r in old['units']}
    for u in windows:
        response = old_units[u['id']]
        require(response['location'] == u['location'], 'Cached temporal support changed')
        context = method.measure(u, values, boxes)
        reference = independent.measure(u, values, boxes)
        require((reference is None) == (context['raw'] is None), 'Independent background availability changed')
        if reference is not None:
            require(abs(reference['raw'] - context['raw']) <= 1e-12
                    and reference['background_cells'] == context['background_cells'], 'Independent temporal background arithmetic failed')
        if recognizer is not None and response['raw'] is not None:
            require(fixed_change(u, vectors, appearance=True) == response['raw'], 'Original native appearance window changed')
        units.append({'id': u['id'], 'location': u['location'], 'baseline': response['raw'],
                      'context': context, 'candidate': method.adjusted(response['raw'], context)})
    name = hashlib.sha256(plan['id'].encode()).hexdigest() + '.npz'
    np.savez_compressed(private / name, indices=np.array(sorted(values), np.int32),
                        descriptors=np.stack([values[i] for i in sorted(values)]))
    return {'id': plan['id'], 'group': plan.get('scoring_component', plan['group']), 'role': plan.get('role', 'ordinary'),
        'family': plan.get('family', 'ordinary-scene-edit'), 'label': plan['label'], 'truth': plan.get('known_edit'),
        'input_sha256': plan['sha256'], 'units': units, 'frames': proofs, 'boxes': boxes,
        'descriptor_file': name, 'descriptor_file_sha256': sha(private / name),
        'native_frames': len(values), 'native_integral_max_difference': maximum, 'sface_parity_forwards': forwards}


def execute(trial, phase):
    import cv2

    selected = check(trial)
    cv2.setNumThreads(1)
    private = trial / 'private'
    private.mkdir(exist_ok=True)
    journal = trial / (phase + '.jsonl')
    counts, completed = Counter(), 0
    if phase != 'point':
        check_points(trial)
    with journal.open('x') as stream:
        if phase in ('point', 'diagnostic'):
            role = 'fusion_cal' if phase == 'point' else 'confirmation'
            originals, grouped = media.inputs(ROOT, SOURCE)
            native = cached(role)
            hashes = {r['id']: r['rgb_sha256'] for r in lines(SOURCE / 'source-frame-audit-v1.jsonl')[1:]}
            recognizer = cv2.FaceRecognizerSF.create(str(ROOT / 'models/weights/face_recognition_sface_2021dec.onnx'), '')
            parity_done = False
            for group in sorted(grouped, key=previous.source.stable):
                wanted = {r['id'] for r in grouped[group] if r['id'] in selected and selected[r['id']]['role'] == role}
                if not wanted:
                    continue
                with media.group_files(ROOT, SOURCE, originals, grouped[group]) as cases:
                    for case in cases:
                        identity = case['planned']['id']
                        if identity not in wanted:
                            continue
                        eligible = not parity_done and any(u['view'] == 'face-appearance' and u['raw'] is not None for u in native[identity]['units'])
                        result = measure(case, selected[identity], native[identity], hashes[identity], private, recognizer if eligible else None)
                        parity_done |= eligible
                        stream.write(json.dumps(result, allow_nan=False) + '\n')
                        stream.flush()
                        counts.update({k: result[k] for k in ('native_frames', 'sface_parity_forwards')})
                        completed += 1
                print({'phase': phase, 'completed': completed, 'counts': dict(counts)}, flush=True)
            require(completed == (80 if phase == 'point' else 160) and parity_done, 'All planned source cases and native parity required')
        else:
            require(phase == 'ordinary', 'Known study phase required')
            source, originals, grouped, hashes = ordinary.source_inputs(ROOT, ORDINARY)
            native = {r['id']: r for r in lines(ORDINARY / 'native.jsonl')[1:] if r['kind'] == 'ordinary-edit'}
            planned = {r['id']: r for r in lines(ORDINARY / 'input-plan.jsonl')}
            for pair in source['pairs']:
                with ordinary.pair_files(ROOT, ORDINARY, pair, originals, grouped, hashes) as (_, cases):
                    for item in cases:
                        row = planned[item['plan']['id']]
                        require(row['rgb_sha256'] == item['rgb_sha256'] and item['plan'] == row['plan'], 'Ordinary source construction changed')
                        result = measure(item['case'], row['plan'], native[row['id']], item['rgb_sha256'], private)
                        stream.write(json.dumps(result, allow_nan=False) + '\n')
                        stream.flush()
                        counts.update({k: result[k] for k in ('native_frames', 'sface_parity_forwards')})
                        completed += 1
                print({'phase': phase, 'completed': completed, 'counts': dict(counts)}, flush=True)
            require(completed == 40, 'All40 ordinary source substitutions required')
    check(trial)
    write(trial / (phase + '-receipt.json'), {'phase': phase, 'cases': completed, 'counts': dict(counts),
        'journal_sha256': sha(journal), 'freeze_sha256': sha(trial / 'freeze.json'), 'new_background_model_forwards': 0})
    if phase == 'point':
        rows = []
        for r in lines(journal):
            values = {k: max((u[k] for u in r['units'] if u[k] is not None), default=None) for k in ('baseline', 'candidate')}
            rows.append({**{k: r[k] for k in ('id', 'group', 'role', 'label')}, **values})
        context = method.fit_point(rows, previous.RAW_POINTS['face-appearance'])
        grouped = {r['group']: [s for s in rows if s['group'] == r['group']] for r in rows}
        complete = {g for g, part in grouped.items() if all(s['baseline'] is not None and s['candidate'] is not None for s in part)}
        baseline = method.fit_point([{**r, 'candidate': r['baseline']} for r in rows if r['group'] in complete], previous.RAW_POINTS['face-appearance'])
        write(trial / 'points.json', {'current-fixed': {'value': previous.RAW_POINTS['face-appearance']},
            'appearance-matched': baseline, 'context': context, 'point_journal_sha256': sha(journal),
            'fit_uses_ai': False, 'diagnostic_started': False})
        write(trial / 'point-fit-complete.json', {'points_sha256': sha(trial / 'points.json'),
            'journal_sha256': sha(journal), 'freeze_sha256': sha(trial / 'freeze.json'),
            'diagnostic_measurements': 0})


def score_file(row, arm, point):
    key = 'candidate' if arm == 'context' else 'baseline'
    valid = [u for u in row['units'] if u[key] is not None]
    locations = [u['location'] for u in valid if u[key] > point]
    spans = previous.intervals(locations)
    require(spans == independent.intervals(locations), 'Independent interval reconstruction failed')
    truth = row['truth']['time_ns'] if row['truth'] else None
    overlap = previous.best_iou(spans, truth)
    require(overlap == independent.iou(spans, truth), 'Independent temporal IoU failed')
    return {'id': row['id'], 'group': row['group'], 'family': row['family'], 'label': row['label'], 'arm': arm,
        'flag': bool(locations) if valid else None, 'intervals_ns': spans, 'best_iou': overlap,
        'scored_units': len(valid), 'planned_units': len(row['units']), 'originality_verdict': None}


def summarize(rows):
    result = {}
    for arm in POLICY['arms']:
        part = [r for r in rows if r['arm'] == arm]
        positive = [r for r in part if r['label'] == 1]
        result[arm] = {'planned': len(part), 'edited': len(positive),
            'edited_flagged': sum(r['flag'] is True for r in positive),
            'useful_iou_0_3': sum(r['best_iou'] >= .3 for r in positive),
            'mean_iou_including_misses': sum(r['best_iou'] for r in positive) / len(positive),
            'unassessed': sum(r['flag'] is None for r in part), 'families': {}}
        for family in sorted({r['family'] for r in part}):
            selected = [r for r in part if r['family'] == family]
            result[arm]['families'][family] = {'planned': len(selected), 'flags': sum(r['flag'] is True for r in selected),
                'flagged_components': len({r['group'] for r in selected if r['flag'] is True}),
                'unassessed': sum(r['flag'] is None for r in selected),
                'useful_iou_0_3': sum(r['best_iou'] is not None and r['best_iou'] >= .3 for r in selected)}
    return result


def review(trial):
    selected = check(trial)
    check_points(trial)
    points = read(trial / 'points.json')
    rows, counts, native_max, temporal_max = [], Counter(), 0., 0.
    original_hashes = {r['id']: r['rgb_sha256'] for r in lines(SOURCE / 'source-frame-audit-v1.jsonl')[1:]}
    original_ordinary = {r['id']: r for r in lines(ORDINARY / 'input-plan.jsonl')}
    source_plans = {**selected, **{k: v['plan'] for k, v in original_ordinary.items()}}
    natives = {**cached('fusion_cal'), **cached('confirmation'),
        **{r['id']: r for r in lines(ORDINARY / 'native.jsonl')[1:] if r['kind'] == 'ordinary-edit'}}
    for phase, number in (('point', 80), ('diagnostic', 160), ('ordinary', 40)):
        receipt, profile = read(trial / (phase + '-receipt.json')), read(trial / (phase + '-profile/profile.json'))
        require(profile['child_exit_code'] == 0 and profile['guard'] is None and profile['seconds_limit'] == 900,
                'All original guarded phases must complete')
        part = lines(trial / (phase + '.jsonl'))
        require(len(part) == len({r['id'] for r in part}) == number and sha(trial / (phase + '.jsonl')) == receipt['journal_sha256']
                and receipt['freeze_sha256'] == sha(trial / 'freeze.json'), 'Complete original phase receipt required')
        expected = {k for k, v in selected.items() if v['role'] == ('fusion_cal' if phase == 'point' else 'confirmation')} if phase != 'ordinary' else set(original_ordinary)
        require({r['id'] for r in part} == expected, 'No planned source may be omitted or replaced')
        phase_counts = Counter()
        for row in part:
            plan, native = source_plans[row['id']], natives[row['id']]
            require(row['input_sha256'] == plan['sha256'] == native['input_sha256'] and row['label'] == plan['label']
                    and row['truth'] == plan.get('known_edit'), 'Source input/truth changed')
            path = trial / 'private' / row['descriptor_file']
            require(path.resolve().parent == (trial / 'private').resolve() and sha(path) == row['descriptor_file_sha256'], 'Private descriptor array changed')
            with np.load(path, allow_pickle=False) as data:
                require(data['indices'].dtype == np.int32 and data['descriptors'].dtype == np.float64
                        and data['descriptors'].shape == (len(data['indices']), 12, 16, 3), 'Complete descriptor array required')
                descriptors = dict(zip(data['indices'].tolist(), data['descriptors'], strict=True))
            hashes = original_ordinary[row['id']]['rgb_sha256'] if phase == 'ordinary' else original_hashes[row['id']]
            frames = {r['frame_index']: r for r in native['frame_inputs']}
            boxes = {r['frame_index']: r['bbox'] for r in native['frame_inputs'] if r['status'] == 'ok' and r['frame_index'] in descriptors}
            require({int(k): v for k, v in row['boxes'].items()} == boxes, 'Common-background mask is not from cached native boxes')
            require(len(row['frames']) == len(descriptors) == row['native_frames'], 'Every source frame proof required')
            for proof in row['frames']:
                i = proof['frame']
                require(proof['rgb_sha256'] == frames[i]['source_rgb_sha256'] == hashes[i]
                        and proof['descriptor_sha256'] == ordinary.digest(descriptors[i])
                        and Fraction(proof['time_base']) * proof['pts'] == Fraction(frames[i]['time_base']) * frames[i]['pts']
                        and proof['integral_max_difference'] <= 1e-12, 'Source pixel/descriptor/native clock proof changed')
                native_max = max(native_max, proof['integral_max_difference'])
            scheduled = [u for u in plan['units'] if u['view'] == 'face-appearance']
            require(len(scheduled) == len(row['units']), 'Every planned temporal window required')
            native_units = {u['id']: u for u in native['units']}
            for unit, value in zip(scheduled, row['units'], strict=True):
                require(unit['id'] == value['id'] and unit['location'] == value['location']
                        and value['baseline'] == native_units[unit['id']]['raw'], 'Source temporal/cached face response changed')
                actual = method.measure(unit, descriptors, boxes)
                reference = independent.measure(unit, descriptors, boxes)
                require(actual == value['context'] and (reference is None) == (actual['raw'] is None)
                        and method.adjusted(value['baseline'], actual) == value['candidate'], 'Stored context response did not replay')
                if reference is not None:
                    difference = abs(reference['raw'] - actual['raw'])
                    require(difference <= 1e-12 and reference['background_cells'] == actual['background_cells'], 'Independent temporal replay failed')
                    temporal_max = max(temporal_max, difference)
            phase_counts.update({k: row[k] for k in ('native_frames', 'sface_parity_forwards')})
        require(dict(phase_counts) == receipt['counts'], 'Native frame/model counters changed')
        counts.update(phase_counts)
        rows.extend(part)
    require(points['point_journal_sha256'] == sha(trial / 'point.jsonl') and points['fit_uses_ai'] is False, 'Genuine point freeze changed')
    genuine = [r for r in rows if r['role'] == 'fusion_cal']
    groups = {r['group']: [s for s in genuine if s['group'] == r['group']] for r in genuine}
    complete = [[{k: max((u[k] for u in r['units'] if u[k] is not None), default=None) for k in ('baseline', 'candidate')} for r in part] for part in groups.values()]
    complete = [part for part in complete if all(s[k] is not None for s in part for k in ('baseline', 'candidate'))]
    budget = sum(max(s['baseline'] for s in part) > previous.RAW_POINTS['face-appearance'] for part in complete)
    require(len(complete) >= 25 and points['current-fixed']['value'] == previous.RAW_POINTS['face-appearance'], 'Original fixed comparator/common point support changed')
    for arm, key in (('context', 'candidate'), ('appearance-matched', 'baseline')):
        expected = sorted((max(s[key] for s in part) for part in complete), reverse=True)[budget]
        require(points[arm]['value'] == expected and points[arm]['assessed_groups'] == len(complete)
                and points[arm]['baseline_flagged_groups'] == budget, 'Independent genuine-only point refit failed')
    mapping = {}
    for pair in read(ORDINARY / 'source-plan.json')['pairs']:
        for parent in pair['parents']:
            mapping[selected[parent['id']]['scoring_component']] = pair['component']
    require(len(mapping) == 40 and len(set(mapping.values())) == 20 and not set(mapping) & {r['group'] for r in genuine}, 'Ordinary-source pair links or role separation changed')
    outcomes = []
    for row in rows:
        if row['role'] == 'fusion_cal':
            continue
        for arm in POLICY['arms']:
            value = score_file(row, arm, points[arm]['value'])
            value['group'] = mapping.get(row['group'], row['group'])
            outcomes.append(value)
    require(len(outcomes) == 600, 'All200 diagnostic files in all three arms required')
    summary, rules = summarize(outcomes), {}
    primary = summary['context']
    for arm in ('current-fixed', 'appearance-matched'):
        base = summary[arm]
        wanted = {r['id'] for r in outcomes if r['arm'] == arm and r['best_iou'] is not None and r['best_iou'] >= .3}
        actual = {r['id'] for r in outcomes if r['arm'] == 'context' and r['best_iou'] is not None and r['best_iou'] >= .3}
        local = {'more_useful_ai': primary['useful_iou_0_3'] > base['useful_iou_0_3'],
                 'higher_mean_iou': primary['mean_iou_including_misses'] > base['mean_iou_including_misses'],
                 'no_lost_useful_ai': wanted <= actual, 'no_extra_unassessed': primary['unassessed'] <= base['unassessed']}
        for family in primary['families']:
            if family not in ('uniface', 'inswapper'):
                local['no_extra_false_components_' + family] = primary['families'][family]['flagged_components'] <= base['families'][family]['flagged_components']
        local['fewer_scene_cut_flags'] = primary['families']['ordinary-scene-edit']['flags'] < base['families']['ordinary-scene-edit']['flags']
        rules[arm] = {'rules': local, 'pass': all(local.values())}
    write(trial / 'outcomes-private.json', outcomes)
    result = {'schema': 'chipmunk.video-background-context-development/25', 'policy': POLICY,
        'freeze_sha256': sha(trial / 'freeze.json'), 'points': points, 'summary': summary, 'comparisons': rules,
        'primary_pass': all(r['pass'] for r in rules.values()), 'counts': dict(counts),
        'point_files': 80, 'diagnostic_files': 200, 'outcomes': 600, 'diagnostic_connected_components': 20,
        'native_integral_max_difference': native_max, 'independent_temporal_max_difference': temporal_max,
        'all_source_pixels_and_clocks_bound': True, 'all_saved_arrays_and_points_replayed': True,
        'new_background_model_forwards': 0, 'fresh_confirmation': False, 'outside_human_reviewers': 0,
        'serving_admitted': False, 'used_in_confidence': False}
    write(trial / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'point', 'diagnostic', 'ordinary', 'review'))
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.trial)
    elif args.action == 'review':
        review(args.trial)
    else:
        execute(args.trial, args.action)
