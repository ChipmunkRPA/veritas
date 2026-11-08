"""Prospective source-bound audio/mouth comparison; no serving imports."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts import mouth_audio_context_v14 as method
from scripts import run_mouth_temporal_v12 as prior

ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = ROOT / 'backend/data/independent-evaluation/lipsync-v11'
COUNTS = {'fit': 160, 'calibration': 24, 'development': 88}
FILES = ('backend/scripts/mouth_audio_context_v14.py', 'backend/scripts/run_mouth_audio_context_v14.py',
         'backend/scripts/review_mouth_audio_context_v14.py', 'backend/scripts/mouth_audio_context_runtime_v14.py',
         'backend/tests/test_mouth_audio_context_v14.py', 'docs/mouth-audio-context-v14-protocol.md',
         'backend/app/detectors/audio/decode.py', 'backend/app/detectors/video/decode.py',
         'docs/evaluation-panels/mouth-v12-development-review.json.gz')
require, sha, write = method.require, prior.sha, prior.write_new


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    packet_path = ROOT / prior.PACKET
    require(sha(packet_path) == prior.PACKET_SHA, 'Original public visual/source capsule changed')
    packet = json.loads(gzip.decompress(packet_path.read_bytes()))
    lock_path = OLD_ROOT / 'experiment/lock.json'
    parent = read(lock_path)
    require(sha(lock_path) == packet['original_lock_sha256'], 'Original visual/source lock changed')
    # These evolved app paths are not executed in this scalar/audio experiment.
    # The old whole-app guard remains failed; do not relax or rewrite that guard.
    evolved = {'backend/app/detectors/video/adapters/face_sequence.py',
               'backend/app/detectors/video/adapters/loaders.py', 'backend/app/models.yaml'}
    for name, expected in parent['implementation_sha256'].items():
        if name not in evolved:
            require(sha(ROOT / name) == expected, 'Executed historical dependency changed: ' + name)
    rows = {stage: prior.old.checked_rows(lock_path, OLD_ROOT / 'experiment' / (stage + '-features.jsonl'), stage, count)
            for stage, count in COUNTS.items()}
    for stage, cases in rows.items():
        require(all(all(row[k] == v for k, v in public.items())
                    for row, public in zip(cases, packet['panels'][stage]['cases'], strict=True)), 'Original complete source/feature capsule differs')
        require(all(r['source_license'] == 'ODbL 1.0 and Database Contents License 1.0' for r in cases), 'Separate original source terms required')
        for row in cases:
            for mask in row.get('masks', []):
                require(sha(Path(row['mask_root']) / mask['path']) == mask['sha256'], 'Original source truth mask changed')
    require(prior.old.training_windows(rows['fit']) == packet['fitting_windows'], 'Every original mask/time-derived fitting label required')
    groups = [{r['group'] for r in rows[s]} for s in COUNTS]
    require([len(g) for g in groups] == [16, 4, 8] and not any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)),
            'Exact source-disjoint actor roles required')
    public_control_path = ROOT / 'docs/evaluation-panels/mouth-v12-development-review.json.gz'
    public_control = json.loads(gzip.decompress(public_control_path.read_bytes()))
    control = ROOT / 'backend/data/independent-evaluation/lipsync-v12/experiment'
    require(public_control['head'] == read(control / 'head.json') and public_control['summary'] == read(control / 'summary.json'),
            'Retained visual-only control differs from published capsule')
    return parent, packet, rows


def absolute_rows(parent):
    path = Path(parent['parent_lock'])
    require(sha(path) == parent['parent_lock_sha256'], 'Original enclosing source lock changed')
    record = read(path)['prior_scores']['v8']
    require(sha(Path(record['path'])) == record['sha256'], 'Original retained absolute scores changed')
    return prior.old.load_rows(Path(record['path']))[1]


def prior_attempt():
    folder = ROOT / 'backend/data/independent-evaluation/lipsync-audio-context-v13'
    fit = [json.loads(v) for v in (folder / 'fit.jsonl').read_text().splitlines()]
    point = [json.loads(v) for v in (folder / 'calibration.jsonl').read_text().splitlines()]
    require(len(fit) == 160 and len(point) == 18 and not (folder / 'candidate-head.json').exists()
            and not (folder / 'development.jsonl').exists(), 'Original incomplete untrained attempt required')
    require(read(folder / 'fit-audio-profile/profile.json')['child_exit_code'] == 0
            and read(folder / 'point-audio-profile/profile.json')['child_exit_code'] == 1
            and 'Complete native mono16k sample blocks required' in (folder / 'point-audio-profile/worker.log').read_text(),
            'Original short-control failure must remain')
    names = ('freeze.json', 'fit.jsonl', 'calibration.jsonl', 'fit-complete.json',
             'fit-audio-profile/profile.json', 'fit-audio-profile/worker.log',
             'point-audio-profile/profile.json', 'point-audio-profile/worker.log')
    return {'file_sha256': {n: sha(folder / n) for n in names}, 'complete_fit_clips': 160,
            'persisted_point_clips': 18, 'trained_heads': 0, 'diagnostic_audio_tracks': 0,
            'native_failure': 'The intact19th control has50 frames; original envelope guard incorrectly required51.',
            'dispatch': 'Original CLI fit action also collides with training and extraction completion filenames.'}


def compare_prior_prefix(trial):
    folder = ROOT / 'backend/data/independent-evaluation/lipsync-audio-context-v13'
    for name in ('fit.jsonl', 'calibration.jsonl'):
        a = [json.loads(v) for v in (folder / name).read_text().splitlines()]
        b = [json.loads(v) for v in (trial / name).read_text().splitlines()]
        require(b[:len(a)] == a, 'All previously persisted source responses must reproduce exactly')
    return {'fit_cases_exact': 160, 'point_cases_exact': 18, 'original_failure_unchanged': True}


def identity():
    _, packet, rows = sources()
    files = {k: sha(ROOT / k) for k in sorted(set(prior.implementation()) | set(FILES))}
    inputs = {stage: [{'id': r['id'], 'sha256': sha(Path(r['mask_root']) / r['path'])} for r in values]
              for stage, values in rows.items()}
    require(all(entry['sha256'] == row['sha256'] for stage in COUNTS
                for entry, row in zip(inputs[stage], rows[stage], strict=True)), 'Every exact prepared media input required')
    control = ROOT / 'backend/data/independent-evaluation/lipsync-v12/experiment'
    return {'schema': 'chipmunk.mouth-audio-context-freeze/14', 'code_sha256': files,
            'source_packet_sha256': prior.PACKET_SHA, 'inputs': inputs,
            'control_head_sha256': sha(control / 'head.json'), 'control_result_sha256': sha(control / 'summary.json'),
            'runtime': prior.runtime(), 'installed_sha256': prior.installed(), 'counts': COUNTS,
            'new_names': list(method.NEW_NAMES), 'lags': list(method.LAGS),
            'scope': 'Consumed source-disjoint actor roles; new audio extraction, inherited audited visual measurements.',
            'prior_attempt': prior_attempt(),
            'historical_whole_app_guard': 'Initial preflight failed on evolved face_sequence.py, video loaders.py and models.yaml before a freeze or any audio extraction. Old guard/files remain unchanged; this study binds original visual data and unchanged executed historical helpers separately.',
            'serving_admitted': False, 'used_in_confidence': False}


def freeze(trial):
    require(prior.os.environ.get('OMP_NUM_THREADS') == prior.os.environ.get('OPENBLAS_NUM_THREADS') == '1',
            'One numerical/OpenMP thread required')
    lock = identity()
    trial.mkdir(exist_ok=False)
    write(trial / 'freeze.json', lock)
    write(ROOT / 'docs/evaluation-panels/mouth-audio-context-v14-freeze.json',
          {**lock, 'private_freeze_sha256': sha(trial / 'freeze.json'), 'new_audio_tracks_processed': 0})
    print({'frozen_media_inputs': 272, 'new_audio_tracks_processed': 0, 'source_roles': [16, 4, 8]}, flush=True)


def check(trial):
    require(read(trial / 'freeze.json') == identity(), 'Frozen code, source, runtime or prior result changed')
    return sources()


def check_fit(trial):
    complete = read(trial / 'head-fit-complete.json')
    names = ('candidate-head.json', 'control-head.json', 'fit.jsonl', 'calibration.jsonl', 'freeze.json')
    require(complete['sha256'] == {n: sha(trial / n) for n in names}, 'Fitted heads, points or source journals changed')


def audio_trace(row):
    from app.detectors.audio.decode import decode_audio
    from app.detectors.video.decode import probe_frames

    path = Path(row['mask_root']) / row['path']
    require(sha(path) == row['sha256'], 'Prepared exact file changed')
    table = probe_frames(path)
    require(len(table) == row['frames'] and all(table.seconds(i) == i / 25 for i in range(len(table))),
            'Native picture clock must have exact25Hz zero origin')
    decoded = decode_audio(path, source_sha256=row['sha256'])
    pcm, mapping = decoded.pcm, decoded.mapping
    require(mapping['sample_rate'] == 16000 and mapping['channels'] == 1
            and mapping['timeline'] == [{'start_sample': 0, 'end_sample': len(pcm), 'source_pts': 0}]
            and len(pcm) == row['audio_samples'] == len(table) * 640
            and mapping['pcm_sha256'] == row['pcm_sha256'], 'Complete original PCM and sample/video origin required')
    aperture = method.apertures(row['geometry_trace'], table.width, table.height, len(table))
    audio = method.envelope(pcm)
    return audio, aperture, {'input_sha256': row['sha256'], 'pcm_sha256': mapping['pcm_sha256'],
        'samples': len(pcm), 'frames': len(table), 'width': table.width, 'height': table.height,
        'pts': list(table.pts), 'time_base': table.time_base, 'mapping': mapping,
        'inherited_geometry_receipt_sha256': sha(OLD_ROOT / 'experiment' / 'development-features.jsonl')
            if row.get('role') not in ('fit', 'point') else sha(OLD_ROOT / 'experiment' /
                ('fit-features.jsonl' if row['role'] == 'fit' else 'calibration-features.jsonl'))}


def extract(trial, stage):
    _, _, rows = check(trial)
    if stage == 'development':
        check_fit(trial)
    require(stage in COUNTS and len(rows[stage]) == COUNTS[stage], 'All planned role cases required')
    count, windows, samples = 0, 0, 0
    with (trial / (stage + '.jsonl')).open('x') as stream:
        for row in rows[stage]:
            audio, aperture, proof = audio_trace(row)
            measured = []
            for index, window in enumerate(row['windows']):
                values, supports = method.features(window, audio, aperture)
                measured.append({'index': index, 'features': values, 'anchor_indices': supports,
                                 'source_window_sha256': hashlib.sha256(json.dumps(window, sort_keys=True).encode()).hexdigest()})
            record = {'id': row['id'], 'stage': stage, 'proof': proof, 'audio': audio.tolist(),
                      'aperture': aperture.tolist(), 'windows': measured}
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            count += 1
            windows += len(measured)
            samples += proof['samples']
            if count % 20 == 0:
                print({'stage': stage, 'clips': count, 'windows': windows}, flush=True)
    write(trial / (stage + '-complete.json'), {'clips': count, 'windows': windows, 'samples': samples,
          'journal_sha256': sha(trial / (stage + '.jsonl')), 'freeze_sha256': sha(trial / 'freeze.json')})
    print({'stage': stage, 'clips': count, 'windows': windows, 'samples': samples}, flush=True)


def joined(trial, stage, rows):
    path = trial / (stage + '.jsonl')
    complete = read(trial / (stage + '-complete.json'))
    require(complete['journal_sha256'] == sha(path) and complete['freeze_sha256'] == sha(trial / 'freeze.json'),
            'Complete exact source journal required')
    records = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(records) == COUNTS[stage] and [r['id'] for r in records] == [r['id'] for r in rows],
            'Every exact ordered role case required')
    output = []
    for row, new in zip(rows, records, strict=True):
        require(len(new['windows']) == len(row['windows']), 'All windows must remain')
        windows = []
        for index, (window, measured) in enumerate(zip(row['windows'], new['windows'], strict=True)):
            require(measured['index'] == index and measured['source_window_sha256'] ==
                    hashlib.sha256(json.dumps(window, sort_keys=True).encode()).hexdigest(), 'New features refer to another source window')
            require(len(measured['features']) == 20 and np.isfinite(measured['features']).all(), 'Complete finite new feature vector required')
            vector = window['features'] + measured['features']
            windows.append({**window, 'visual_features_float64_sha256': window['features_float64_sha256'],
                            'features': vector, 'features_float64_sha256':
                            hashlib.sha256(np.asarray(vector, np.float64).tobytes()).hexdigest()})
        output.append({**row, 'windows': windows})
    return output


def fit(trial):
    _, packet, rows = check(trial)
    fit_rows = joined(trial, 'fit', rows['fit'])
    point_rows = joined(trial, 'calibration', rows['calibration'])
    parity = compare_prior_prefix(trial)
    write(trial / 'prior-prefix-parity.json', parity)
    original = packet['fitting_windows']
    expanded = [w['features'] for row in fit_rows for w in row['windows']]
    require(len(expanded) == len(original) == 3560, 'Every original fitting label and source required')
    candidate_model, candidate = method.fit([{**r, 'features': v} for r, v in zip(original, expanded, strict=True)])
    control_model, control = prior.method.fit(original)
    old_control = read(ROOT / 'backend/data/independent-evaluation/lipsync-v12/experiment/head.json')
    require(all(control[k] == old_control[k] for k in control), 'Unchanged visual-only control failed exact refit parity')
    for name, head, module, point in [('candidate', candidate, method, point_rows),
                                     ('control', control, prior.method, rows['calibration'])]:
        maximum = max(float(v) for row in point for v in module.margins([w['features'] for w in row['windows']], head))
        head.update(review_point=math.nextafter(max(0., maximum), math.inf), ordinary_max_margin=maximum,
                    freeze_sha256=sha(trial / 'freeze.json'))
        if name == 'control':
            require(head['review_point'] == old_control['review_point'], 'Unchanged control point differs')
        write(trial / (name + '-head.json'), head)
    # Keep the native estimators only for a finite fit/point correspondence check.
    difference = {}
    for name, model, head, module, values in [('candidate', candidate_model, candidate, method, fit_rows + point_rows),
                                            ('control', control_model, control, prior.method, rows['fit'] + rows['calibration'])]:
        x = np.asarray([w['features'] for row in values for w in row['windows']], np.float64)
        error = float(np.max(np.abs(model.decision_function(x) - module.margins(x, head))))
        require(error <= 1e-10, 'Native exported complete fit/point margins differ')
        difference[name] = {'windows': len(x), 'maximum_difference': error}
    names = ('candidate-head.json', 'control-head.json', 'fit.jsonl', 'calibration.jsonl', 'freeze.json')
    write(trial / 'head-fit-complete.json', {'sha256': {n: sha(trial / n) for n in names},
          'fit_point_native_export': difference, 'diagnostic_processed': False})
    print({'fitted_arms': 2, 'fit_point_export': difference, 'diagnostic_processed': False}, flush=True)


def score(trial):
    parent, packet, rows = check(trial)
    check_fit(trial)
    enriched = joined(trial, 'development', rows['development'])
    candidates = {'control': (rows['development'], prior.method), 'candidate': (enriched, method)}
    absolute = absolute_rows(parent)
    require([{'id': r['id'], 'probe_windows': r['probe_windows']} for r in absolute]
            == packet['absolute_reference']['cases'], 'Every original absolute source/window required')
    absolute_point, original = packet['absolute_reference']['review_point'], packet['absolute_reference']['outcomes']
    summaries, private, differences = {}, {}, {}
    for name, (cases, module) in candidates.items():
        head = read(trial / (name + '-head.json'))
        fitting = packet['fitting_windows']
        if name == 'candidate':
            fit_rows = joined(trial, 'fit', rows['fit'])
            values = [w['features'] for r in fit_rows for w in r['windows']]
            fitting = [{**r, 'features': v} for r, v in zip(fitting, values, strict=True)]
            all_cases = fit_rows + joined(trial, 'calibration', rows['calibration']) + cases
        else:
            all_cases = rows['fit'] + rows['calibration'] + cases
        estimator, recovered = module.fit(fitting)
        require(all(recovered[k] == head[k] for k in recovered), 'Fitting-only reconstructed head differs from prediagnostic head')
        full = np.asarray([w['features'] for r in all_cases for w in r['windows']], np.float64)
        error = float(np.max(np.abs(estimator.decision_function(full) - module.margins(full, head))))
        require(len(full) == 6373 and error <= 1e-10, 'Complete native/export correspondence required')
        differences[name + '_native_export'] = {'windows': len(full), 'maximum_difference': error}
        standalone, combined, signals, scored = [], [], [], []
        for row, ref in zip(cases, absolute, strict=True):
            require(row['id'] == ref['id'], 'Original absolute source ordering required')
            windows = [{**w, 'response': float(v)} for w, v in zip(row['windows'],
                       module.margins([w['features'] for w in row['windows']], head), strict=True)]
            windows.sort(key=lambda w: (-w['response'], w['start_s'], w['track_id']))
            new = {**row, 'probe_windows': windows[:12], 'roi_bindings': {'probe': windows[0]['region']}}
            root = Path(row['mask_root'])
            standalone += prior.old.baseline.evaluate([new], 'probe', head['review_point'], root)
            signal = prior.old.method.selected_signal(ref['probe_windows'], absolute_point, new['probe_windows'], head['review_point'])
            combined += prior.old.baseline.evaluate([ref] if signal == 'absolute' else [new], 'probe',
                        absolute_point if signal == 'absolute' else head['review_point'], root)
            signals.append({'id': row['id'], 'signal': signal})
            scored.append({'id': row['id'], 'windows': windows})
        pseudo = [{'status': r['status'], 'baseline_replay_parity': all(h['baseline_replay_parity'] for h in r['heads'].values()),
                   'prior_native_input_parity': r['native_input_replay'] and all(h['prior_native_and_margin_parity'] for h in r['heads'].values())}
                  for r in rows['development']]
        gates = prior.old.paired.gates(original, combined, [r['family'] for r in rows['development']], pseudo)
        family = {f: prior.old.baseline.summary([r for r, row in zip(combined, cases, strict=True) if row['family'] == f])
                  for f in ('liveportrait', 'musetalk')}
        summaries[name] = {'standalone': prior.old.baseline.summary(standalone), 'combined': prior.old.baseline.summary(combined),
                           'gates': gates, 'families': family}
        private[name] = {'standalone': standalone, 'combined': combined, 'signals': signals, 'scored': scored}
        if name == 'control':
            previous = read(ROOT / 'backend/data/independent-evaluation/lipsync-v12/experiment/summary.json')
            require(combined == previous['cases']['combined'] and standalone == previous['cases']['temporal']
                    and scored == previous['scored_windows'], 'Unchanged visual-only diagnostic parity failed')
            differences['control_exact'] = True
    old = private['control']['combined']
    new = private['candidate']['combined']
    summaries['candidate']['gates'].update(
        no_fewer_locations_than_visual_control=sum(r['localized'] for r in new)
            >= sum(r['localized'] for r in old),
        no_more_ordinary_flags_than_visual_control=sum(not r['label'] and r['flagged'] for r in new)
            <= sum(not r['label'] and r['flagged'] for r in old))
    conditions = {}
    for condition in sorted({r['condition'] for r in rows['development']}):
        conditions[condition] = {name: prior.old.baseline.summary([r for r, row in zip(private[name]['combined'], rows['development'], strict=True)
                                                               if row['condition'] == condition]) for name in private}
    write(trial / 'private-result.json', private)
    result = {'schema': 'chipmunk.mouth-audio-context-development/14', 'baseline': prior.old.baseline.summary(original),
              'arms': summaries, 'conditions': conditions, 'control_parity': differences,
              'passes': all(summaries['candidate']['gates'].values()), 'case_counts': COUNTS,
              'all_diagnostic_cases': len(new), 'window_counts': {s: sum(len(r['windows']) for r in rows[s]) for s in COUNTS},
              'new_audio_decodes': 272, 'new_visual_neural_forwards': 0,
              'outside_human_reviewers': 0, 'serving_admitted': False, 'used_in_confidence': False,
              'private_result_sha256': sha(trial / 'private-result.json'), 'fit_complete_sha256': sha(trial / 'head-fit-complete.json')}
    write(trial / 'result.json', result)
    print({'arms': {k: v['combined'] for k, v in summaries.items()}, 'passes': result['passes']}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('freeze', 'train', 'score', *COUNTS))
    p.add_argument('trial', type=Path)
    a = p.parse_args()
    if a.action in COUNTS:
        extract(a.trial, a.action)
    else:
        {'freeze': freeze, 'train': fit, 'score': score}[a.action](a.trial)
