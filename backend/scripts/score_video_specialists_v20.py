"""Prospective fixed-support face evidence calibration; never serving admission.

Private native inputs and face geometry stay local. Public scalar replay may be
released separately. No identity recognition, lip-sync or whole-video claim.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from app.bench import specialists as engine
from scripts import audit_video_scoring_inputs_v18 as source_audit
from scripts import inventory_video_specialists_v18 as source
from scripts import materialize_video_specialists_v18 as media
from scripts import specialist_calibration as fitting

ROOT = Path(__file__).resolve().parents[2]
VIEWS = ('face-crop', 'face-appearance', 'face-feature-change')
MODELS = ('yunet.2023mar', 'gend.clip_l14', 'opencv.sface.2021dec', 'dinov2.small')
ARMS = {'crop': [VIEWS[0]], 'crop-appearance': list(VIEWS[:2]),
    'crop-features': [VIEWS[0], VIEWS[2]], 'all': list(VIEWS)}
RAW_POINTS = {'face-crop': 0., 'face-appearance': .11243900750879454,
    'face-feature-change': .007326911389827726}
POLICY = {'schema': 'chipmunk.video-face-specialist-policy/20', 'arms': ARMS,
    'source': 'All648 consumed, exact prepared clips in162 known source components; four source-disjoint roles. No fresh reserve used.',
    'positions': 'Frozen half-second centers and native consecutive triplets; three middle, two before and two after centers. No authenticity response chooses positions.',
    'truth': 'Audited changed RGB source-frame presence in sampled support. Publisher manipulation labels, not identified-face or pixel-authorship ground truth.',
    'views': 'Unchanged commercial GenD face-crop logit, SFace appearance transition, DINOv2 patch feature transition. One visible detected face; no identity database.',
    'native': 'YuNet and SFace CPU; GenD and DINOv2 MPSfloat32. Fixed crop batch8 and patch batch12; no adaptive precision/batch fallback.',
    'repeat': 'First eligible crop/appearance/features input in each role. SFace exact; crop absolute1e-4, patches absolute/relative1e-4. Native source RGB must equal model-free audit.',
    'combination': 'Separate monotone local heads by target/view; one dependent video-face family, positive-logit maximum, then separately fitted and calibrated combination. No multiplication of independent votes.',
    'missing': 'Every planned unit retained; any unavailable required unit withholds combined confidence. No missing response becomes zero or an original verdict.',
    'minimum_groups_per_class': 25, 'l2': .001, 'reference_prior': .5,
    'points': 'Internal evaluation only. Across all four arms use the same completely assessed genuine fusion_cal source components, minimum25. Budget equals raw GenD genuine component false flags, capped atfloor(.1*n). Strict greater than sorted component maxima at rankn-budget. Local points match each existing raw cue in the same manner on common complete genuine files.',
    'comparison': 'Raw current-code cue arithmetic on exactly these fixed supports, not full production tracking/adaptive windows or whole application fusion.',
    'localization': 'Merge flagged support spans separated by at most550ms, for every arm and raw reference. Best merged-span IoU with known edit >=.3; separately count actually sampled flagged source frames and local-unit truth. Bridged gaps are interval estimates, not examined pixels.',
    'continuation': 'Require more AI files at no more genuine false-flag components than crop, no coverage loss, and more correctly localized manipulated clips at IoU>=.3 than crop. Report ordinary controls separately. No automatic serving adoption; fresh-source and full-app tests remain required.',
    'resource_guard': {'rss_gib': 6, 'available_mib': 1536, 'disk_mib': 1536, 'seconds_per_role': 3600},
    'serving_admitted': False}
NAMES = ('backend/scripts/score_video_specialists_v20.py', 'backend/scripts/measure_video_specialists_v18.py',
    'backend/scripts/specialist_runtime_v7.py', 'backend/scripts/specialist_calibration.py',
    'backend/tests/test_video_specialist_measurements_v18.py', 'backend/tests/test_video_specialists_v20.py',
    'docs/video-face-specialists-v20-protocol.md', 'backend/scripts/score_video_specialists_v19.py',
    'backend/scripts/specialist_runtime_v6.py', 'backend/tests/test_video_specialists_v19.py',
    'docs/video-face-specialists-v19-protocol.md')


def code():
    names = set(NAMES) | set(source_audit.CODE)
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / 'backend/app').rglob('*')
        if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.npy'))
    return {n: source.sha(ROOT / n) for n in sorted(names)}


def runtime():
    return {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'torch', 'transformers',
        'tokenizers', 'safetensors', 'opencv-python-headless', 'psutil', 'Pillow')}


def configure(main, trial, role):
    # Must precede imports that capture application settings. No customer DB.
    from app.core import config

    scratch = trial / ('native-context-' + role)
    scratch.mkdir(exist_ok=True)
    config.settings = config.Settings(_env_file=main / 'backend/.env', data_dir=scratch,
        database_url='sqlite:///' + str(scratch / 'state.db'), models_dir=main / 'models',
        hf_home=main / 'models/hf', offline=True, model_memory_budget_mb='6144')


def freeze(main, trial):
    source_audit.check(trial)
    plan = json.loads((trial / 'scoring-input-plan-v2.json').read_text())
    require(len(plan['files']) == 648 and not plan['scoring_protocol_complete'], 'Complete audited source plan required')
    require(not list(trial.glob('*-v20-native.jsonl')), 'Freeze must precede native scoring')
    configure(main, trial, 'preflight')
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    models = {}
    registry = get_registry()
    for mid in MODELS:
        entry = registry.manifest.model(mid)
        require(entry.licence.class_ == 'commercial', 'Commercial weight admission required')
        models[mid] = {'licence': entry.licence.model_dump(mode='json'),
            'files': {str(p.resolve().relative_to(main)): source.sha(p) for p in ensure_verified(entry)}}
    lock = {'schema': 'chipmunk.video-face-specialist-freeze/20', 'main': str(main), 'policy': POLICY,
        'code_sha256': code(), 'runtime': runtime(), 'models': models,
        'inputs_sha256': {n: source.sha(trial / n) for n in ('scoring-input-plan-v2.json',
            'source-frame-audit-v1.jsonl', 'source-frame-audit-v1.json', 'source-frame-audit-freeze-v1.json')},
        'binaries_sha256': {n: source.sha(Path(shutil.which(n))) for n in ('ffmpeg', 'ffprobe')},
        'commercial_license_record_sha256': source.sha(ROOT / 'backend/app/models.yaml'),
        'confirmation_authenticity_forwards': 0, 'serving_admitted': False}
    source.write(trial / 'v20-freeze.json', lock)
    print(json.dumps({'freeze_sha256': source.sha(trial / 'v20-freeze.json'), 'planned_clips': 648,
        'planned_units': sum(len(r['units']) for r in plan['files']), 'new_authenticity_forwards': 0}), flush=True)


require = media.require


def check(trial):
    lock = json.loads((trial / 'v20-freeze.json').read_text())
    require(lock['policy'] == POLICY and lock['code_sha256'] == code() and lock['runtime'] == runtime(),
        'Frozen native scoring, fitting, test or runtime changed')
    require(all(source.sha(trial / n) == h for n, h in lock['inputs_sha256'].items())
        and all(source.sha(Path(shutil.which(n))) == h for n, h in lock['binaries_sha256'].items()),
        'Audited source plan, frame commitments or decoder binary changed')
    return lock, json.loads((trial / 'scoring-input-plan-v2.json').read_text())


def native_objects(cpu, accel):
    """ModelLease returns the adapter's revision-bound wrapper, not its payload."""
    from app.detectors.image.adapters.loaders import LoadedModel

    wrapped = [cpu.get(MODELS[0]), accel.get(MODELS[1]), accel.get(MODELS[3]), cpu.get(MODELS[2])]
    require(all(isinstance(item, LoadedModel) for item in wrapped), 'Revision-bound loaded model wrappers required')
    return tuple(item.obj for item in wrapped)


def native(trial, role):
    lock, plan = check(trial)
    journal, receipt = trial / (role + '-v20-native.jsonl'), trial / (role + '-v20-native.json')
    require(not journal.exists() and not receipt.exists(), 'Retain every attempt; no implicit native restart')
    if role == 'confirmation':
        verify_fit(trial)
    main = Path(lock['main'])
    configure(main, trial, role)
    import cv2
    import torch

    from app.engine.models import get_model_manager
    from scripts.measure_video_specialists_v18 import measure

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    require(torch.backends.mps.is_available(), 'Frozen MPSfloat32 execution required')
    audit_rows = {r['id']: r for r in [json.loads(s) for s in
        (trial / 'source-frame-audit-v1.jsonl').read_text().splitlines()][1:]}
    selected = {r['id']: r for r in plan['files'] if r['role'] == role}
    original, grouped = media.inputs(main, trial)
    manager = get_model_manager()
    counts, repeats, complete, started = Counter(), set(), [], time.monotonic()
    with journal.open('x') as stream:
        stream.write(json.dumps({'schema': 'chipmunk.video-face-native-journal/20', 'role': role,
            'freeze_sha256': source.sha(trial / 'v20-freeze.json')}) + '\n')
        stream.flush()
        with manager.lease((MODELS[0], MODELS[2]), prefer='cpu') as cpu, \
                manager.lease((MODELS[1], MODELS[3]), prefer='mps', activation_bytes=2400 * 1024**2) as accel:
            require(cpu.device == 'cpu' and accel.device == 'mps' and accel.dtype == 'float32', 'Frozen device/dtype changed')
            yunet, gend_model, dino, recognizer = native_objects(cpu, accel)
            for group in sorted(grouped, key=source.stable):
                if not any(r['id'] in selected for r in grouped[group]):
                    continue
                with media.group_files(main, trial, original, grouped[group]) as cases:
                    for case in cases:
                        identity = case['planned']['id']
                        if identity not in selected:
                            continue
                        result = measure(case, selected[identity], yunet, gend_model, dino, recognizer,
                            accel.device, repeat_views=set(('crop', 'appearance', 'features')) - repeats,
                            source_rgb_sha256=audit_rows[identity]['rgb_sha256'])
                        repeats.update(result['native_repeats'])
                        counts.update(result['native_input_counts'])
                        stream.write(json.dumps(result, allow_nan=False) + '\n')
                        stream.flush()
                        complete.append(identity)
                    print(json.dumps({'role': role, 'completed_clips': len(complete),
                        'planned_clips': len(selected), 'native_input_counts': counts}), flush=True)
    require(set(complete) == set(selected) and len(complete) == len(selected), 'All planned cases required')
    require(repeats == {'crop', 'appearance', 'features'}, 'All native view repeats required')
    check(trial)
    source.write(receipt, {'schema': 'chipmunk.video-face-native-receipt/20', 'role': role,
        'journal_sha256': source.sha(journal), 'clips': len(complete), 'native_input_counts': counts,
        'native_repeats': sorted(repeats), 'native_source_rgb_bound': True,
        'duration_s': time.monotonic() - started, 'serving_admitted': False})


def responses(trial, role):
    receipt = json.loads((trial / (role + '-v20-native.json')).read_text())
    path = trial / (role + '-v20-native.jsonl')
    require(source.sha(path) == receipt['journal_sha256'], 'Native journal changed')
    rows = [json.loads(s) for s in path.read_text().splitlines()][1:]
    require(len(rows) == receipt['clips'] and len({r['id'] for r in rows}) == len(rows), 'Complete unique native responses required')
    return {r['id']: r for r in rows}


def manifest(trial, arm, confirmation):
    lock, plan = check(trial)
    records = {role: responses(trial, role) for role in engine.ROLES if confirmation or role != 'confirmation'}
    views = ARMS[arm]
    route = engine.digest({'policy': POLICY, 'plan_sha256': lock['inputs_sha256']['scoring-input-plan-v2.json']})
    weights = [h for entry in lock['models'].values() for h in entry['files'].values()]
    specs = [{'id': v, 'modality': 'video', 'family': 'video-face',
        'target': 'video.sampled_frame_presence' if v == 'face-crop' else 'video.sampled_middle_presence',
        'contexts': ['single-face'], 'routing_sha256': route, 'higher_means_ai': True,
        'raw_range': [-100., 100.] if v == 'face-crop' else [-4., 4.], 'producer': {'kind': 'model',
            'implementation_sha256': engine.digest(lock['code_sha256']), 'weight_sha256': weights,
            'commercial_license_record_sha256': lock['commercial_license_record_sha256'],
            'native_receipt_sha256': source.sha(trial / 'v20-training-native.json')}} for v in views]
    files = []
    for row in plan['files']:
        response = records.get(row['role'], {}).get(row['id'])
        measured = {u['id']: u for u in response['units']} if response else {}
        units = []
        for planned in row['units']:
            if planned['view'] not in views:
                continue
            current = measured.get(planned['id'])
            if current:
                require(current['location'] == planned['location'] and current['view'] == planned['view'], 'Native local support changed')
            status = current['status'] if current else 'unavailable'
            unit = {'id': planned['id'], 'specialist': planned['view'], 'context': 'single-face',
                'routing_sha256': route, 'context_evidence_sha256': engine.digest({'location': planned['location'],
                    'single_visible_face_policy': True, 'source_sha256': row['sha256']}),
                'measurement_sha256': engine.digest(current or {'reserved': planned['id'], 'input': row['sha256']}),
                'location': planned['location'], 'raw': current['raw'] if current else None, 'status': status,
                'label': planned['label'], 'label_scope': 'local_unit',
                'label_provenance_sha256': engine.digest({'changed': row['actual_changed_ai_source_frames'],
                    'location': planned['location'], 'source_frame_audit': lock['inputs_sha256']['source-frame-audit-v1.json']})}
            if status != 'ok':
                unit['reason'] = current['reason'] if current else 'Reserved confirmation not yet measured.'
            units.append(unit)
        files.append({'id': row['id'], 'group': row['scoring_component'], 'role': row['role'], 'modality': 'video',
            'source_sha256': row['sha256'], 'input_sha256': row['sha256'],
            'lineage': ['known-source-component:' + row['scoring_component']],
            'label': int(any(u['label'] for u in units)), 'label_scope': 'planned_units',
            'label_provenance_sha256': engine.digest({'clip': row['sha256'], 'sampled_units': [(u['id'], u['label']) for u in units]}),
            'source_geometry': row['source_geometry'], 'required_families': ['video-face'], 'units': units})
    document = {'schema': engine.SCHEMA, 'policy': {'specialists': specs}, 'files': files}
    engine.audit(document)
    return document


def probability(model, values):
    return 1 / (1 + math.exp(-max(-700., min(700., engine._logit(model, values)))))


def predict(file, artifact):
    x, combined = engine._features(file, artifact), artifact['combiners'].get(engine.pattern(file))
    if x is None or not combined or combined['response']['status'] != 'fitted' or combined.get('calibration', {}).get('status') != 'fitted':
        return None
    return probability(combined['calibration'], [engine._logit(combined['response'], x)])


def matched_point(values, reference, minimum=25):
    common = set(values) & set(reference)
    result = {'status': 'insufficient_support', 'threshold': None, 'common_genuine_groups': len(common),
        'scope': 'Internal evaluation only; no public screening threshold', 'serving_admitted': False}
    if len(common) >= minimum:
        budget = min(sum(reference[g] for g in common), math.floor(.1 * len(common)))
        ordered = sorted(values[g] for g in common)
        cut = ordered[len(ordered) - budget - 1]
        result.update(status='fitted', threshold=cut, budget=budget,
            calibration_groups_flagged=sum(values[g] > cut for g in common))
    return result


def fit(trial):
    check(trial)
    require(not (trial / 'confirmation-v20-native.jsonl').exists(), 'Fit must precede all confirmation authenticity inference')
    source.write(trial / 'v20-training-native.json', {'roles': {r: source.sha(trial / (r + '-v20-native.json'))
        for r in engine.ROLES[:3]}, 'confirmation_authenticity_forwards': 0, 'serving_admitted': False})
    documents, artifacts, values = {}, {}, {}
    for arm in ARMS:
        document = manifest(trial, arm, False)
        source.write(trial / (arm + '-v20-before-confirmation.json'), document)
        lock = fitting.freeze(document, 25, .001)
        source.write(trial / (arm + '-v20-fit-freeze.json'), lock)
        artifact = fitting.fit(document, lock)
        source.write(trial / (arm + '-v20-artifact.json'), artifact)
        documents[arm], artifacts[arm] = document, artifact
        groups = {}
        for file in document['files']:
            if file['role'] == 'fusion_cal' and file['label'] == 0:
                p = predict(file, artifact)
                if p is not None:
                    groups.setdefault(file['group'], []).append(p)
        # Require every genuine file in a source component, not one convenient file.
        values[arm] = {g: max(v) for g, v in groups.items() if len(v) == sum(f['group'] == g and
            f['role'] == 'fusion_cal' and f['label'] == 0 for f in document['files'])}
    common = set.intersection(*(set(v) for v in values.values()))
    reference = {}
    for file in documents['crop']['files']:
        if file['role'] == 'fusion_cal' and file['label'] == 0 and file['group'] in common:
            reference[file['group']] = reference.get(file['group'], False) or any(u['status'] == 'ok' and u['raw'] > 0 for u in file['units'])
    for arm in ARMS:
        point = matched_point({g: p for g, p in values[arm].items() if g in common}, reference)
        local = {}
        for view in ARMS[arm]:
            model = artifacts[arm]['specialists'][view + '|single-face']
            current, old = {}, {}
            if model['status'] == 'fitted':
                for file in documents[arm]['files']:
                    if file['role'] == 'fusion_cal' and file['label'] == 0 and file['group'] in common:
                        for unit in file['units']:
                            if unit['specialist'] == view and unit['status'] == 'ok':
                                p = probability(model, [unit['raw']])
                                current[file['group']] = max(current.get(file['group'], p), p)
                                old[file['group']] = old.get(file['group'], False) or unit['raw'] > RAW_POINTS[view]
            local[view] = matched_point(current, old)
        source.write(trial / (arm + '-v20-points.json'), {'file': point, 'local': local})
    source.write(trial / 'v20-fit-complete.json', {'files_sha256': {arm + '-v20-' + suffix + '.json':
        source.sha(trial / (arm + '-v20-' + suffix + '.json')) for arm in ARMS for suffix in ('artifact', 'points')},
        'common_complete_genuine_groups': len(common), 'confirmation_authenticity_forwards': 0, 'serving_admitted': False})


def verify_fit(trial):
    receipt = json.loads((trial / 'v20-fit-complete.json').read_text())
    require(all(source.sha(trial / n) == h for n, h in receipt['files_sha256'].items()), 'Fitted artifacts/points changed before confirmation')
    return receipt


def summarize(rows, point):
    cells = {'planned': len(rows), 'assessed': 0, 'inconclusive': 0, 'ai': 0, 'genuine': 0,
        'ai_flagged': 0, 'missed_ai_including_inconclusive': 0, 'false_ai': 0}
    false_groups, loss = set(), {0: defaultdict(list), 1: defaultdict(list)}
    for row in rows:
        y, p = row['label'], row['probability']
        usable = p is not None and point['status'] == 'fitted'
        flag = usable and p > point['threshold']
        cells['ai' if y else 'genuine'] += 1
        cells['assessed' if usable else 'inconclusive'] += 1
        if y:
            cells['ai_flagged' if flag else 'missed_ai_including_inconclusive'] += 1
        elif flag:
            cells['false_ai'] += 1
            false_groups.add(row['group'])
        if p is not None:
            loss[y][row['group']].append((p - y) ** 2)
    per_class = [sum(sum(v) / len(v) for v in groups.values()) / len(groups) if groups else None for groups in loss.values()]
    return {**cells, 'false_ai_groups': len(false_groups), 'source_class_balanced_brier_on_scored_cases':
        sum(per_class) / 2 if all(v is not None for v in per_class) else None}


def intervals(locations):
    merged = []
    for start, end in sorted((s['start'], s['end']) for s in locations):
        if merged and start - merged[-1][1] <= 550_000_000:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def best_iou(spans, truth):
    if not truth:
        return None
    a, b = truth
    return max((max(0, min(b, y) - max(a, x)) / (max(b, y) - min(a, x)) for x, y in spans), default=0.)


def local_metrics(document, result, points, metadata):
    by_id = {(r['file_id'], r['unit_id']): r for r in result['unit_outcomes']}
    evidence, counts = [], Counter()
    for file in document['files']:
        if file['role'] != 'confirmation':
            continue
        original = metadata[file['id']]
        flagged, actual_frames, flags = [], set(original['actual_changed_ai_source_frames']), set()
        for unit in file['units']:
            point = points['local'][unit['specialist']]
            row = by_id[file['id'], unit['id']]
            usable = row['probability'] is not None and point['status'] == 'fitted'
            yes = usable and row['probability'] > point['threshold']
            counts['planned_units'] += 1
            counts['assessed_units' if usable else 'inconclusive_units'] += 1
            counts['ai_units' if unit['label'] else 'genuine_units'] += 1
            if yes:
                counts['ai_units_flagged' if unit['label'] else 'genuine_units_false_flagged'] += 1
                flagged.append(unit['location'])
                flags.update(unit['location']['sampled_frames'])
        spans = intervals(flagged)
        overlap = best_iou(spans, original['known_edit']['time_ns'] if original['known_edit'] else None)
        if original['whole_clip_label']:
            counts['edited_clips'] += 1
            counts['edited_clips_localized_iou_at_least_0_3'] += overlap >= .3
        else:
            counts['genuine_clips_with_flagged_spans'] += bool(spans)
        counts['flagged_sampled_source_frames'] += len(flags)
        counts['flagged_sampled_changed_source_frames'] += len(flags & actual_frames)
        evidence.append({'id': file['id'], 'source_pts_table_sha256': original['source_geometry']['pts_table_sha256'],
            'flagged_support_spans_ns': spans, 'flagged_sampled_source_frames': sorted(flags),
            'actual_changed_source_frames': sorted(actual_frames), 'best_known_edit_span_iou': overlap,
            'scope': 'Bridged support-span estimate; only listed source frames were examined. No face/pixel authorship claim.'})
    return {'counts': dict(counts), 'evidence': evidence}


def review(trial):
    _, plan = check(trial)
    verify_fit(trial)
    metadata = {r['id']: r for r in plan['files']}
    report = {'schema': 'chipmunk.video-face-specialist-result/20', 'policy': POLICY, 'arms': {}, 'raw_cue_references': {},
        'serving_admitted': False, 'outside_human_reviews': 0}
    for arm in ARMS:
        document = manifest(trial, arm, True)
        artifact = json.loads((trial / (arm + '-v20-artifact.json')).read_text())
        result = engine.evaluate(document, artifact)
        source.write(trial / (arm + '-v20-confirmation.json'), result)
        points = json.loads((trial / (arm + '-v20-points.json')).read_text())
        local = local_metrics(document, result, points, metadata)
        source.write(trial / (arm + '-v20-local-evidence.json'), local)
        report['arms'][arm] = {'all': summarize(result['outcomes'], points['file']),
            'by_source_family': {family: summarize([r for r in result['outcomes'] if metadata[r['id']]['family'] == family], points['file'])
                for family in sorted({r['family'] for r in plan['files']})},
            'local': local['counts'], 'points': points,
            'head_status': {k: v['status'] for k, v in artifact['specialists'].items()}}
        if arm == 'all':
            for view in VIEWS:
                raw_files, raw_units, current_files = [], [], []
                for file in document['files']:
                    if file['role'] != 'confirmation':
                        continue
                    units = [u for u in file['units'] if u['specialist'] == view]
                    current_files.append({**file, 'units': units})
                    complete = all(u['status'] == 'ok' for u in units)
                    raw_files.append({'id': file['id'], 'group': file['group'],
                        'label': int(any(u['label'] for u in units)),
                        'probability': float(any(u['raw'] > RAW_POINTS[view] for u in units)) if complete else None})
                    raw_units.extend({'file_id': file['id'], 'unit_id': u['id'],
                        'probability': u['raw'] if u['status'] == 'ok' else None} for u in units)
                raw_summary = summarize(raw_files, {'status': 'fitted', 'threshold': .5})
                raw_summary.pop('source_class_balanced_brier_on_scored_cases')
                report['raw_cue_references'][view] = {'all': raw_summary, 'raw_point': RAW_POINTS[view],
                    'scope': 'Same fixed supports and current raw cue arithmetic only; binary flags are not probabilities.',
                    'local': local_metrics({'files': current_files}, {'unit_outcomes': raw_units},
                        {'local': {view: {'status': 'fitted', 'threshold': RAW_POINTS[view]}}}, metadata)['counts']}
    baseline = report['arms']['crop']
    for arm, current in report['arms'].items():
        current['qualifies_for_fresh_source_followup'] = arm != 'crop' and (
            current['all']['ai_flagged'] > baseline['all']['ai_flagged']
            and current['all']['false_ai_groups'] <= baseline['all']['false_ai_groups']
            and current['all']['inconclusive'] <= baseline['all']['inconclusive']
            and current['local'].get('edited_clips_localized_iou_at_least_0_3', 0)
                > baseline['local'].get('edited_clips_localized_iou_at_least_0_3', 0))
    source.write(trial / 'v20-result.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'native', 'fit', 'review'))
    parser.add_argument('trial', type=Path)
    parser.add_argument('--main', type=Path)
    parser.add_argument('--role', choices=engine.ROLES)
    args = parser.parse_args()
    if args.stage == 'freeze':
        require(args.main is not None, 'Main checkout path required')
        freeze(args.main.resolve(), args.trial.resolve())
    elif args.stage == 'native':
        require(args.role is not None, 'Exact source role required')
        native(args.trial.resolve(), args.role)
    elif args.stage == 'fit':
        fit(args.trial.resolve())
    else:
        review(args.trial.resolve())
