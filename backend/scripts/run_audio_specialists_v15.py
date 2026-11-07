"""Frozen native scoring and source-separated noisy-voice calibration pilot."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from app.bench import specialists as engine
from scripts import prepare_audio_specialists_v15 as source
from scripts import specialist_calibration as fitting

ROOT = Path(__file__).resolve().parents[2]
ARMS = {'full-pooled': (False, False), 'full-context': (True, False),
    'combined-pooled': (False, True), 'combined-context': (True, True)}
POLICY = {'schema': 'chipmunk.audio-specialists-scoring-policy/15', 'arms': list(ARMS),
    'source_method': source.METHOD, 'l2': .001, 'minimum_groups_per_class': 25,
    'native_parity_absolute_limit': 1e-6, 'native_repeat_exact': True,
    'operating_alpha': .1, 'operating_minimum_assessed_genuine_groups': 25,
    'operating_scope': 'Genuine foreground-pair maxima of assessed fusion_cal files; unassessed files/groups explicitly counted, no response imputation. Strict greater than upper nearest rank ceil((n+1)*.9), capped at n.',
    'additional_continuation_gates': ['no_clean_ai_detection_loss', 'no_additional_false_ai_groups_in_any_condition',
        'more_ai_detected_across_strong_noise_music_speech', 'no_condition_coverage_loss'],
    'admission': 'No serving admission; continuation only. Consumed sources, limited independent background parents and external pretraining overlap require new independent-source validation.'}


def read_lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def code():
    names = ['backend/scripts/prepare_audio_specialists_v15.py', 'backend/scripts/run_audio_specialists_v15.py',
        'backend/scripts/specialist_runtime_v3.py', 'backend/scripts/specialist_calibration.py',
        'backend/app/bench/specialists.py', 'backend/scripts/audio_quality_v5.py',
        'backend/scripts/audio_calibration_v3_methods.py', 'backend/app/detectors/audio/upgraded.py',
        'backend/app/detectors/audio/decode.py', 'backend/app/detectors/audio/content.py',
        'backend/app/detectors/video/audio.py', 'backend/app/third_party/kaldi/fbank.py',
        'backend/tests/test_audio_specialists_v15.py', 'docs/audio-specialists-v15-protocol.md']
    return {name: source.sha(ROOT / name) for name in names}


def runtime():
    return {name: importlib.metadata.version(name) for name in
        ('numpy', 'scipy', 'torch', 'transformers', 'tokenizers', 'pyloudnorm', 'psutil')}


def freeze(main, trial):
    plan = json.loads((trial / 'source-plan.json').read_text())
    audit = json.loads((trial / 'signal-audit.json').read_text())
    header, *rows = read_lines(trial / 'signal-plan.jsonl')
    if (plan['method'] != source.METHOD or header['method'] != source.METHOD
            or audit['signal_plan_sha256'] != source.sha(trial / 'signal-plan.jsonl')
            or header['source_plan_sha256'] != source.sha(trial / 'source-plan.json')
            or plan['preparation_code_sha256'] != source.sha(Path(source.__file__))
            or header['preparation_code_sha256'] != plan['preparation_code_sha256']
            or len(rows) != 4004 or audit['files'] != 4004 or audit['authenticity_classifier_forwards'] != 0):
        raise ValueError('Complete prospective source/signal plan with zero authenticity forwards required')
    weights = main / 'models/weights/audio-upgrade'
    if {name: source.sha(weights / name) for name in source.ASSETS} != source.ASSETS:
        raise ValueError('Pinned commercial assets changed')
    lock = {'schema': 'chipmunk.audio-specialists-freeze/15', 'policy': POLICY, 'code_sha256': code(),
        'runtime': runtime(), 'main': str(main.resolve()), 'weights': str(weights.resolve()), 'assets': source.ASSETS,
        'source_plan_sha256': source.sha(trial / 'source-plan.json'),
        'signal_plan_sha256': source.sha(trial / 'signal-plan.jsonl'),
        'signal_audit_sha256': source.sha(trial / 'signal-audit.json'),
        'commercial_license_record_sha256': source.sha(main / 'backend/app/models.yaml'),
        'serving_calibration_sha256': source.sha(main / 'backend/app/detectors/audio/calibrations/local-speech-v3.json'),
        'new_authenticity_classifier_forwards': 0, 'serving_admitted': False}
    source.write(trial / 'freeze.json', lock)
    print(json.dumps({'freeze_sha256': source.sha(trial / 'freeze.json'), 'files': len(rows),
        'planned_windows': sum(len(r['windows']) for r in rows), 'new_authenticity_classifier_forwards': 0}), flush=True)


def check(trial):
    lock = json.loads((trial / 'freeze.json').read_text())
    if (lock['policy'] != POLICY or lock['code_sha256'] != code() or lock['runtime'] != runtime()
            or lock['source_plan_sha256'] != source.sha(trial / 'source-plan.json')
            or lock['signal_plan_sha256'] != source.sha(trial / 'signal-plan.jsonl')
            or lock['signal_audit_sha256'] != source.sha(trial / 'signal-audit.json')):
        raise ValueError('Frozen source/method/code/runtime changed')
    return lock, json.loads((trial / 'source-plan.json').read_text()), read_lines(trial / 'signal-plan.jsonl')[1:]


def load_voice(lock):
    import torch

    from app.detectors.audio.upgraded import VoiceModel

    weights = Path(lock['weights'])
    if {name: source.sha(weights / name) for name in source.ASSETS} != lock['assets']:
        raise ValueError('Exact local model bytes required')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with torch.device('meta'):
        model = VoiceModel(json.loads((weights / 'w2v-bert-config.json').read_text()))
    state = torch.load(weights / 'pellav2_detector.pt', weights_only=True, mmap=True, map_location='cpu')
    model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False).eval()
    if any(p.is_meta or p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Complete strict CPUfloat32 parameters required')
    return model


def native(trial):
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import prepare_voice_window

    lock, plan, _ = check(trial)
    main = Path(lock['main'])
    old = json.loads((main / 'backend/data/independent-evaluation/audio-v9/development/variants.json').read_text())
    originals = {r['base_id']: r for r in old['variants'] if r['condition'] == 'clean'}
    selected = []
    targets = {(role, label): sorted([r for p in plan['pairs'] if p['role'] == role
        for r in p['foregrounds'] if r['label'] == label], key=lambda r: source.stable(r['id']))
        for role in engine.ROLES[:3] for label in (0, 1)}
    allowed = {originals[r['id']]['id'] for rows in targets.values() for r in rows}
    cached = {}
    path = main / 'backend/data/independent-evaluation/audio-v14-separation/features.jsonl'
    with path.open() as stream:
        next(stream)
        for line in stream:
            row = json.loads(line)
            if row['id'] in allowed and row['status'] == 'ok' and engine.finite(row.get('raw')):
                cached[row['id']] = {k: row[k] for k in ('id', 'sha256', 'pcm_sha256', 'raw')}
    for (role, label), rows in targets.items():
        row = next((originals[r['id']] for r in rows if originals[r['id']]['id'] in cached), None)
        if row is None:
            raise ValueError('Native parity requires retained assessed input in every training role/class')
        selected.append((role, label, row, cached[row['id']]))
    model = load_voice(lock)
    results = []
    for role, label, row, expected in selected:
        path = Path(old['source_root']) / row['path']
        if source.sha(path) != expected['sha256'] or row['pcm_sha256'] != expected['pcm_sha256']:
            raise ValueError('Retained native input identity changed')
        audio = decode_audio(path, source_sha256=expected['sha256'])
        if source.pcm_sha(audio.pcm) != expected['pcm_sha256'] or len(audio.pcm) != 64000:
            raise ValueError('Retained exact4s native PCM changed')
        value = float(model.predict([audio.pcm])[0])
        repeat = float(model.predict([audio.pcm])[0])
        difference = abs(value - expected['raw'])
        if value != repeat or difference > POLICY['native_parity_absolute_limit']:
            raise ValueError('Native exact repeat or retained model-response parity failed')
        results.append({'role': role, 'label': label, 'input_sha256': row['sha256'],
            'pcm_sha256': row['pcm_sha256'], 'preprocessed_sha256': source.pcm_sha(prepare_voice_window(audio.pcm)),
            'raw': value, 'reference_raw': expected['raw'], 'absolute_difference': difference, 'repeat_exact': True})
    receipt = {'schema': 'chipmunk.audio-specialists-native-parity/15',
        'freeze_sha256': source.sha(trial / 'freeze.json'), 'old_features_sha256': source.sha(main / 'backend/data/independent-evaluation/audio-v14-separation/features.jsonl'),
        'cases': results, 'maximum_absolute_difference': max(r['absolute_difference'] for r in results),
        'new_composite_authenticity_forwards': 0, 'confirmation_authenticity_forwards': 0, 'serving_admitted': False}
    source.write(trial / 'native-parity.json', receipt)
    print(json.dumps({'native_parity_cases': len(results), 'maximum_absolute_difference': receipt['maximum_absolute_difference'],
        'new_composite_authenticity_forwards': 0}), flush=True)


def score_role(trial, role):
    from app.detectors.audio.decode import read_wav, wav_bytes
    from app.detectors.audio.upgraded import prepare_voice_window
    from scripts.audio_calibration_v3_methods import band_view

    lock, original, planned = check(trial)
    parity = json.loads((trial / 'native-parity.json').read_text())
    if parity['freeze_sha256'] != source.sha(trial / 'freeze.json') or len(parity['cases']) != 6:
        raise ValueError('Completed source-bound native parity required')
    if role == 'confirmation' and not (trial / 'fit-complete.json').exists():
        raise ValueError('Freeze every fitted arm and operating point before confirmation inference')
    path = trial / (role + '.jsonl')
    if path.exists():
        raise FileExistsError('Retain every prior authenticity attempt; no implicit resume')
    plan = {**original, 'pairs': [p for p in original['pairs'] if p['role'] == role],
        'backgrounds': [r for r in original['backgrounds'] if r['role'] == role]}
    cache = source.load_sources(plan)
    expected = {r['id']: r for r in planned if r['role'] == role}
    model = load_voice(lock)
    scored, audited, completed = {}, set(), set()
    started = time.monotonic()
    with path.open('x') as stream:
        stream.write(json.dumps({'schema': 'chipmunk.audio-specialists-responses/15', 'role': role,
            'freeze_sha256': source.sha(trial / 'freeze.json'), 'planned_files': len(expected)}) + '\n')
        for pair in plan['pairs']:
            for metadata, values in source.pair_signals(pair, plan, cache):
                row = expected[metadata['id']]
                if any(row[k] != v for k, v in metadata.items()):
                    raise ValueError('Frozen source/recipe/sample labels changed')
                raw = wav_bytes(values)
                pcm = read_wav(raw)
                filtered = band_view(pcm)
                if (hashlib.sha256(raw).hexdigest() != row['input_sha256'] or source.pcm_sha(pcm) != row['pcm_sha256']
                        or source.pcm_sha(filtered) != row['band_pcm_sha256']):
                    raise ValueError('Frozen composite bytes or whole-clip band view changed')
                for window in row['windows']:
                    start, end = window['start'], window['end']
                    for view, signal in (('full', pcm), ('band', filtered)):
                        chunk = signal[start:end]
                        key = view, window[view + '_pcm_sha256']
                        if source.pcm_sha(chunk) != key[1]:
                            raise ValueError('Native window does not match frozen source samples')
                        if not window['speech_eligible']:
                            continue
                        cell = view, window['context']
                        if key not in scored:
                            result = {'view': view, 'pcm_sha256': key[1], 'status': 'failed', 'raw': None}
                            try:
                                result.update(raw=float(model.predict([chunk])[0]), status='ok',
                                    preprocessed_sha256=source.pcm_sha(prepare_voice_window(chunk)))
                                if not engine.finite(result['raw']) or not 0 <= result['raw'] <= 1:
                                    raise ValueError('Nonfinite native response')
                            except Exception as error:
                                result.update(status='failed', raw=None, error_type=type(error).__name__)
                                stream.write(json.dumps(result) + '\n')
                                stream.flush()
                                raise
                            if cell not in audited:
                                repeated = float(model.predict([chunk])[0])
                                if repeated != result['raw']:
                                    raise ValueError('Native context/view repeat differs bitwise')
                                audited.add(cell)
                                result['native_repeat_exact'] = True
                            stream.write(json.dumps(result, allow_nan=False) + '\n')
                            stream.flush()
                            scored[key] = result
                completed.add(row['id'])
            print(json.dumps({'role': role, 'completed_files': len(completed), 'planned_files': len(expected),
                'unique_scored_windows': len(scored), 'elapsed_s': round(time.monotonic() - started, 2)}), flush=True)
    if completed != set(expected) or len(audited) != 4:
        raise ValueError('Complete frozen file plan and every view/context native repeat required')
    source.write(trial / (role + '-native.json'), {'schema': 'chipmunk.audio-specialists-role-native/15',
        'freeze_sha256': source.sha(trial / 'freeze.json'), 'journal_sha256': source.sha(path), 'role': role,
        'complete_files': len(completed), 'unique_scored_windows': len(scored),
        'exact_native_view_context_repeats': len(audited), 'serving_admitted': False})


def responses(trial, role):
    path = trial / (role + '.jsonl')
    native = json.loads((trial / (role + '-native.json')).read_text())
    header, *rows = read_lines(path)
    if (header['freeze_sha256'] != source.sha(trial / 'freeze.json') or native['journal_sha256'] != source.sha(path)
            or native['unique_scored_windows'] != len(rows) or any(r['status'] != 'ok' for r in rows)):
        raise ValueError('Complete native role receipt required; no failed observation may be omitted')
    return {(r['view'], r['pcm_sha256']): r for r in rows}


def manifest(trial, arm, confirmation):
    lock, plan, planned = check(trial)
    contextual, combined = ARMS[arm]
    views = ('full', 'band') if combined else ('full',)
    roles = engine.ROLES if confirmation else engine.ROLES[:3]
    records = {role: responses(trial, role) for role in roles}
    route = engine.digest({'source_quality_code': lock['code_sha256']['backend/scripts/audio_quality_v5.py'],
        'context_cut': source.METHOD['context_cut'], 'contextual': contextual, 'only_original_window_features': True})
    specs = [{'id': 'audio.' + view, 'modality': 'audio', 'family': 'voice', 'target': 'audio.synthetic_voice',
        'contexts': source.METHOD['contexts'] if contextual else ['pooled'], 'routing_sha256': route,
        'higher_means_ai': True, 'raw_range': [0., 1.], 'producer': {'kind': 'model',
            'implementation_sha256': engine.digest(lock['code_sha256']), 'weight_sha256': list(lock['assets'].values()),
            'commercial_license_record_sha256': lock['commercial_license_record_sha256'],
            'native_receipt_sha256': source.sha(trial / 'training-native.json')}} for view in views]
    files = []
    parents = {r['sha256']: r for p in plan['pairs'] for r in p['foregrounds']}
    parents.update({r['sha256']: r for r in plan['backgrounds']})
    for row in planned:
        lineage = list(sorted(set(row['foreground_sha256'])))
        if row['background'] is not None:
            lineage.append(row['background']['sha256'])
        for key in list(lineage):
            parent = parents[key]
            for field in ('canonical_speaker', 'transcript_sha256'):
                if parent.get(field):
                    lineage.append(field + ':' + engine.digest(parent[field]))
        lineage = sorted(set(lineage))
        units = []
        for window in row['windows']:
            provenance = engine.digest({'foreground_labels': row['labels'], 'foregrounds': row['foreground_sha256'],
                'source_sample_ranges': [[0, 64000], [64000, 128000]], 'location': [window['start'], window['end']], 'label': window['label']})
            for view in views:
                key = view, window[view + '_pcm_sha256']
                result = records.get(row['role'], {}).get(key) if window['speech_eligible'] else None
                status = 'ok' if result else ('not_applicable' if not window['speech_eligible'] else 'unavailable')
                unit = {'id': f'{view}.{window["start"]}', 'specialist': 'audio.' + view,
                    'context': window['context'] if contextual else 'pooled', 'routing_sha256': route,
                    'context_evidence_sha256': engine.digest({'quality': window['quality'], 'full_pcm_sha256': window['full_pcm_sha256']}),
                    'measurement_sha256': engine.digest(result) if result else engine.digest({'planned_pcm': key[1], 'speech_gate': window['speech_eligible']}),
                    'location': {'kind': 'samples', 'unit': 'samples', 'start': window['start'], 'end': window['end'],
                        'extent': source.EXTENT, 'sample_rate': source.SR},
                    'status': status, 'raw': result['raw'] if result else None, 'label': window['label'],
                    'label_scope': 'local_unit', 'label_provenance_sha256': provenance}
                if not result:
                    unit['reason'] = 'Original speech gate did not support this interval.' if not window['speech_eligible'] else 'Reserved confirmation response not yet scored.'
                units.append(unit)
        files.append({'id': row['id'], 'group': row['group'], 'role': row['role'], 'modality': 'audio',
            'source_sha256': row['input_sha256'], 'input_sha256': row['input_sha256'], 'lineage': lineage,
            'label': int(any(row['labels'])), 'label_scope': 'planned_units',
            'label_provenance_sha256': engine.digest({'foreground_labels': row['labels'], 'parents': row['foreground_sha256']}),
            'source_geometry': {'sample_count': source.EXTENT, 'sample_rate': source.SR},
            'required_families': ['voice'], 'units': units})
    document = {'schema': engine.SCHEMA, 'policy': {'specialists': specs}, 'files': files}
    engine.audit(document)
    return document


def predict_file(file, artifact):
    values = engine._features(file, artifact)
    combined = artifact['combiners'].get(engine.pattern(file))
    if (values is None or not combined or combined['response']['status'] != 'fitted'
            or combined.get('calibration', {}).get('status') != 'fitted'):
        return None
    eta = engine._logit(combined['calibration'], [engine._logit(combined['response'], values)])
    return 1 / (1 + math.exp(-max(-700., min(700., eta))))


def operating_point(document, artifact):
    selected = [f for f in document['files'] if f['role'] == 'fusion_cal' and f['label'] == 0]
    maxima, missing = {}, 0
    for file in selected:
        value = predict_file(file, artifact)
        if value is None:
            missing += 1
        else:
            maxima[file['group']] = max(maxima.get(file['group'], value), value)
    total = len({f['group'] for f in selected})
    point = {'status': 'insufficient_support', 'threshold': None, 'genuine_groups_total': total,
        'genuine_groups_with_assessed_files': len(maxima), 'unassessed_genuine_files': missing,
        'scope': POLICY['operating_scope'], 'serving_admitted': False}
    if len(maxima) >= 25:
        values = sorted(maxima.values())
        rank = min(len(values), math.ceil((len(values) + 1) * .9))
        point.update(status='fitted', threshold=values[rank - 1], upper_rank=rank,
            calibration_genuine_groups_flagged=sum(v > values[rank - 1] for v in values))
    return point


def fit(trial):
    check(trial)
    source.write(trial / 'training-native.json', {'roles': {role: source.sha(trial / (role + '-native.json'))
        for role in engine.ROLES[:3]}, 'native_parity_sha256': source.sha(trial / 'native-parity.json'),
        'confirmation_authenticity_forwards': 0, 'serving_admitted': False})
    for arm in ARMS:
        document = manifest(trial, arm, False)
        source.write(trial / (arm + '-manifest-before-confirmation.json'), document)
        frozen = fitting.freeze(document, 25, .001)
        source.write(trial / (arm + '-fit-freeze.json'), frozen)
        artifact = fitting.fit(document, frozen)
        source.write(trial / (arm + '-artifact.json'), artifact)
        source.write(trial / (arm + '-operating-point.json'), operating_point(document, artifact))
    source.write(trial / 'fit-complete.json', {'artifacts': {a: source.sha(trial / (a + '-artifact.json')) for a in ARMS},
        'operating_points': {a: source.sha(trial / (a + '-operating-point.json')) for a in ARMS},
        'confirmation_authenticity_forwards': 0, 'serving_admitted': False})


def summarize(rows, point):
    cells = {'planned': len(rows), 'assessed': 0, 'inconclusive': 0, 'genuine': 0, 'ai': 0,
        'false_ai': 0, 'ai_flagged': 0, 'ai_not_flagged_including_inconclusive': 0}
    genuine_groups, false_groups = set(), set()
    loss = {0: defaultdict(list), 1: defaultdict(list)}
    for row in rows:
        label, p = row['label'], row['probability']
        cells['ai' if label else 'genuine'] += 1
        if not label:
            genuine_groups.add(row['group'])
        usable = p is not None and point['status'] == 'fitted'
        if usable:
            cells['assessed'] += 1
        else:
            cells['inconclusive'] += 1
        flag = bool(usable and p > point['threshold'])
        if label:
            cells['ai_flagged' if flag else 'ai_not_flagged_including_inconclusive'] += 1
        elif flag:
            cells['false_ai'] += 1
            false_groups.add(row['group'])
        if p is not None:
            loss[label][row['group']].append((p - label)**2)
    by_class = [sum(sum(v) / len(v) for v in groups.values()) / len(groups) if groups else None for groups in loss.values()]
    cells.update(genuine_groups=len(genuine_groups), false_ai_groups=len(false_groups),
        source_class_balanced_brier_on_scored_cases=sum(by_class) / 2 if all(v is not None for v in by_class) else None)
    return cells


def review(trial):
    _, _, planned = check(trial)
    complete = json.loads((trial / 'fit-complete.json').read_text())
    metadata = {r['id']: r for r in planned}
    report = {'schema': 'chipmunk.audio-specialists-result/15', 'policy': POLICY, 'arms': {}, 'serving_admitted': False}
    for arm in ARMS:
        for suffix, key in (('artifact', 'artifacts'), ('operating-point', 'operating_points')):
            if source.sha(trial / (arm + '-' + suffix + '.json')) != complete[key][arm]:
                raise ValueError('Confirmation cannot change fitted models or operating points')
        document = manifest(trial, arm, True)
        artifact = json.loads((trial / (arm + '-artifact.json')).read_text())
        result = engine.evaluate(document, artifact)
        source.write(trial / (arm + '-confirmation.json'), result)
        point = json.loads((trial / (arm + '-operating-point.json')).read_text())
        report['arms'][arm] = {'all': summarize(result['outcomes'], point),
            'conditions': {condition: summarize([r for r in result['outcomes'] if metadata[r['id']]['condition'] == condition], point)
                for condition, _, _ in source.CONDITIONS}, 'point': point,
            'head_status': {k: v['status'] for k, v in artifact['specialists'].items()}}
    reference = report['arms']['full-pooled']
    for arm, observed in report['arms'].items():
        a, b = observed['all'], reference['all']
        conditions, old = observed['conditions'], reference['conditions']
        gates = {'more_ai_files_found': a['ai_flagged'] > b['ai_flagged'],
            'no_additional_false_ai_foreground_groups': a['false_ai_groups'] <= b['false_ai_groups'],
            'no_coverage_loss': a['assessed'] >= b['assessed'],
            'lower_source_balanced_brier': a['source_class_balanced_brier_on_scored_cases'] is not None
                and b['source_class_balanced_brier_on_scored_cases'] is not None
                and a['source_class_balanced_brier_on_scored_cases'] < b['source_class_balanced_brier_on_scored_cases'],
            'no_clean_ai_detection_loss': conditions['clean']['ai_flagged'] >= old['clean']['ai_flagged'],
            'no_additional_false_ai_groups_in_any_condition': all(conditions[c]['false_ai_groups'] <= old[c]['false_ai_groups'] for c in conditions),
            'more_strong_background_ai_files_found': sum(conditions[c]['ai_flagged'] for c in ('noise_-5', 'music_-5', 'speech_-5'))
                > sum(old[c]['ai_flagged'] for c in ('noise_-5', 'music_-5', 'speech_-5')),
            'no_condition_coverage_loss': all(conditions[c]['assessed'] >= old[c]['assessed'] for c in conditions)}
        observed.update(continuation_gates=gates, continuation_eligible=arm != 'full-pooled' and all(gates.values()))
    report['limits'] = source.METHOD['limits']
    source.write(trial / 'result.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'native', 'score-role', 'fit', 'review'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--role', choices=engine.ROLES)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.main, args.trial)
    elif args.stage == 'native':
        native(args.trial)
    elif args.stage == 'score-role':
        score_role(args.trial, args.role)
    elif args.stage == 'fit':
        fit(args.trial)
    else:
        review(args.trial)
