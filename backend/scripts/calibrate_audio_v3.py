"""Frozen, fresh multi-generator audio study. Never writes serving artifacts."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import evaluate_audio_candidates as util
import audio_calibration_v3_methods as method
from audio_calibration_v3_scoring import score_candidate

ROOT = util.ROOT
OUT = ROOT / 'data/independent-evaluation/audio-v3'
PROTOCOL = ROOT.parent / 'docs/audio-calibration-v3-protocol.md'
CONDITIONS = ['clean', 'noise_0', 'noise_5', 'noise_15', 'music_5', 'music_15']
SYSTEMS = ['old_guarded', 'new_unguarded', 'new_guarded']
save, sha, file_sha = util.save_json, util.digest, util.file_digest
BOOTSTRAP_SEED = 3101985
BOOTSTRAP_REPLICATES = 2000


def load(path):
    return json.loads(path.read_bytes())


def manifest():
    value = load(OUT / 'manifest.json')
    if sha(util.canonical({k: v for k, v in value.items() if k != 'sha256'})) != value['sha256']:
        raise ValueError('Manifest digest changed')
    return value


def serving_hashes():
    files = [ROOT / 'app/models.yaml']
    for folder in (ROOT / 'app/engine/fusion/release', ROOT / 'data/fusion'):
        if folder.exists():
            files.extend(p for p in folder.rglob('*') if p.is_file())
    return {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(files)}


def runtime_versions():
    from importlib.metadata import version
    packages = ('numpy', 'scipy', 'torch', 'transformers', 'soundfile', 'pyloudnorm', 'scikit-learn')
    return dict(python=platform.python_version(), platform=platform.platform(),
        packages={p: version(p) for p in packages},
        ffmpeg=subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True, text=True).stdout.splitlines()[0])


def lock():
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry
    m = manifest()
    if '**Status: frozen' not in PROTOCOL.read_text():
        raise ValueError('Protocol must be finalized before fresh inference')
    if (OUT / 'scores.jsonl').exists():
        raise ValueError('Fresh inference must not precede the run lock')
    prep = load(OUT / 'audits/independent-preparation-audit.json')
    selection = load(OUT / 'development/method-selection.json')
    optimization = load(OUT / 'development/optimization-equivalence.json')
    if prep.get('status') != 'verified_with_documented_text_and_background_population_limits' or prep['manifest_sha256'] != m['sha256']:
        raise ValueError('Independent preparation audit must verify this exact cohort')
    if not selection['accepted_for_fresh_evaluation'] or selection['method_code_sha256'] != file_sha(ROOT / 'scripts/audio_calibration_v3_methods.py'):
        raise ValueError('Fixed method was not accepted before fresh inference')
    if (optimization.get('status') != 'passed' or not optimization.get('all_probabilities_and_final_decisions_exact')
        or optimization['method_sha256'] != file_sha(ROOT / 'scripts/audio_calibration_v3_methods.py')
        or optimization['scoring_code_sha256'] != file_sha(ROOT / 'scripts/audio_calibration_v3_scoring.py')):
        raise ValueError('Execution equivalence must cover the exact scorer being frozen')
    ids = ['hf.gustking.wav2vec2_xlsr_deepfake', 'pella.voice_v2', 'silero.speech_activity']
    verified = {key: [{'path': str(p), 'sha256': file_sha(p)} for p in
        ensure_verified(get_registry().manifest.model(key), force=True)] for key in ids}
    files = ['scripts/calibrate_audio_v3.py', 'scripts/audio_calibration_v3_methods.py', 'scripts/audio_calibration_v3_scoring.py',
        'scripts/prepare_audio_v3_data.py', 'scripts/evaluate_audio_candidates.py',
        'app/detectors/audio/upgraded.py', 'app/detectors/audio/content.py', 'app/detectors/audio/decode.py',
        'app/detectors/video/adapters/audio.py', 'app/detectors/video/audio.py',
        'app/detectors/video/adapters/loaders.py', 'app/engine/profiles.py', 'app/third_party/kaldi/fbank.py']
    evidence = ['audits/independent-preparation-audit.json', 'development/method-selection.json',
                'development/method-parity.json', 'development/optimization-equivalence.json']
    save(OUT / 'run-lock.json', dict(schema='chipmunk.audio-validation-lock/3',
        frozen_at_unix=time.time(), manifest_sha256=m['sha256'], protocol_sha256=file_sha(PROTOCOL),
        source_hashes={p: file_sha(ROOT / p) for p in files}, verified_local_weight_files=verified,
        pre_inference_evidence={p: file_sha(OUT / p) for p in evidence},
        serving_hashes=serving_hashes(), runtime=runtime_versions(), method=method.METHOD, device='cpu', cpu_threads=4,
        profile='full', offline_execution=True, bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, deployment_eligible=False))
    print('Run locked:', file_sha(OUT / 'run-lock.json'), flush=True)


def verify_lock():
    m, frozen = manifest(), load(OUT / 'run-lock.json')
    if frozen['manifest_sha256'] != m['sha256'] or frozen['protocol_sha256'] != file_sha(PROTOCOL):
        raise ValueError('Manifest or protocol changed after freeze')
    for path, digest in frozen['source_hashes'].items():
        if file_sha(ROOT / path) != digest:
            raise ValueError('Frozen implementation changed: ' + path)
    for path, digest in frozen['pre_inference_evidence'].items():
        if file_sha(OUT / path) != digest:
            raise ValueError('Preparation or method acceptance evidence changed: ' + path)
    if frozen['serving_hashes'] != serving_hashes():
        raise ValueError('Serving configuration changed during offline validation')
    if frozen['runtime'] != runtime_versions():
        raise ValueError('Pinned execution runtime changed')
    return m, frozen


def authorize_test():
    """Freeze calibration AND the development-score prefix before any test inference."""
    verify_lock()
    calibration = load(OUT / 'calibration.json')
    if calibration['run_lock_sha256'] != file_sha(OUT / 'run-lock.json'):
        raise ValueError('Calibration refers to another experiment')
    if sha(util.canonical(read_scores({'fit', 'cal'}))) != calibration['development_rows_sha256']:
        raise ValueError('Development scores changed after calibration fitting')
    score_path, start_path = OUT / 'scores.jsonl', OUT / 'test-start.json'
    data = score_path.read_bytes()
    if start_path.exists():
        start = load(start_path)
        if start['calibration_sha256'] != file_sha(OUT / 'calibration.json'):
            raise ValueError('Calibration changed after test authorization')
        prefix = data[:start['development_prefix_bytes']]
        if sha(prefix) != start['development_prefix_sha256']:
            raise ValueError('Development prefix changed after test authorization')
    else:
        if any(json.loads(line)['split'] == 'test' for line in data.splitlines()):
            raise ValueError('Test scores already exist without a test-start lock')
        start = dict(first_test_authorized_at_unix=time.time(),
            calibration_sha256=file_sha(OUT / 'calibration.json'),
            run_lock_sha256=file_sha(OUT / 'run-lock.json'),
            development_prefix_bytes=len(data), development_prefix_sha256=sha(data),
            development_records=len(data.splitlines()))
        save(start_path, start)
    return start


def score(stage='development'):
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from app.detectors.audio import content
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import load_speech_activity, load_voice
    from app.detectors.video.adapters.audio import GustkingVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.detectors.video.adapters.loaders import load_audio_classifier
    from app.detectors.video.audio import voice_windows, VOICE_WINDOW_S, MAX_VOICE_WINDOWS
    from app.engine.detector import RunContext, StageOutputs, Profile
    from app.engine.profiles import SAMPLING
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    m, frozen = verify_lock()
    calibration = None
    if stage == 'test':
        authorize_test()
        calibration = load(OUT / 'calibration.json')
    elif (OUT / 'test-start.json').exists():
        raise ValueError('Development scoring cannot resume after test access')
    chosen = {'fit', 'cal'} if stage == 'development' else {'test'}
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    output = OUT / 'scores.jsonl'
    previous = [json.loads(s) for s in output.read_text().splitlines()] if output.exists() else []
    complete = {(r['variant_id'], r['detector']) for r in previous}
    if len(complete) != len(previous):
        raise ValueError('Duplicate score records')
    for key in ('old', 'new'):
        remaining = [r for r in m['variants'] if r['split'] in chosen and (r['id'], key) not in complete]
        if not remaining:
            continue
        ids = ['hf.gustking.wav2vec2_xlsr_deepfake'] if key == 'old' else ['pella.voice_v2', 'silero.speech_activity']
        for model_id in ids:
            actual = [{'path': str(p), 'sha256': file_sha(p)} for p in
                ensure_verified(get_registry().manifest.model(model_id), force=True)]
            if actual != frozen['verified_local_weight_files'][model_id]:
                raise ValueError('Pinned weight changed')
        print(f'Loading {key}: {len(remaining)} {stage} inputs.', flush=True)
        models = {ids[0]: load_audio_classifier(ids[0], 'cpu', 'float32')} if key == 'old' else {
            ids[0]: load_voice(ids[0], 'cpu', 'float32'),
            ids[1]: load_speech_activity(ids[1], 'cpu', 'float32')}
        lease = SimpleNamespace(get=models.__getitem__)
        detector = GustkingVoiceDetector() if key == 'old' else None
        ctx = None
        for index, row in enumerate(remaining):
            path = OUT / row['path']
            if file_sha(path) != row['sha256']:
                raise ValueError('Input bytes differ from manifest')
            decoded = decode_audio(path, source_sha256=row['sha256'])
            if decoded.mapping['pcm_sha256'] != row['pcm_sha256']:
                raise ValueError('Production PCM differs from manifest')
            gate = content.classify(decoded.pcm).as_json()
            record = dict(variant_id=row['id'], base_id=row['base_id'], split=row['split'], label=row['label'],
                speaker=row['speaker'], cluster_id=row.get('cluster_id', row['speaker']), detector=key,
                condition=row['condition'], device='cpu', seconds=row['seconds'],
                input_sha256=row['sha256'], production_input=decoded.mapping,
                recorded_at_unix=time.time(), heuristic_gate=gate)
            started = time.perf_counter()
            try:
                if key == 'new':
                    views = score_candidate(decoded.pcm, models[ids[0]].obj, models[ids[1]].obj, calibration)
                    record.update(status=views['status'], reason=views.get('probe_error'),
                        score=views['raw'], raw_unrounded=views['raw_unrounded'],
                        windows=[dict(start_sample=w['start_sample'], end_sample=w['end_sample'],
                            raw_unrounded=w['raw']) for w in views['windows']], views=views)
                    spans = views['speech_spans_s']
                else:
                    spans = gate['speech_spans']
                    track = AudioTrack('analysed', decoded.pcm, len(decoded.pcm), gate=tuple(tuple(s) for s in spans))
                    ctx = RunContext(analysis_id='audio-validation-v3', media=None, profile='full', device='cpu',
                        dtype='float32', lease=lease, outputs=StageOutputs(audio=track),
                        deadline=SimpleNamespace(check=lambda label='': None),
                        exhibits=SimpleNamespace(series=lambda *a, **k: 'offline-series'), log=None)
                    applicable = detector.applicable(ctx)
                    if not applicable.ok:
                        record.update(status='not_applicable', reason=applicable.reason,
                                      score=None, raw_unrounded=None, windows=[])
                    else:
                        result = detector.run(ctx)
                        record.update(status=result.status, reason=result.reason,
                            score=None if result.aggregate is None else round(float(result.aggregate.raw), 6),
                            raw_unrounded=None if result.aggregate is None else float(result.aggregate.raw),
                            windows=[dict(start_sample=round(u.quality['start_sample']),
                                end_sample=round(u.quality['end_sample']), raw_unrounded=float(u.raw)) for u in result.unit_scores])
                params = SAMPLING[Profile('full')]
                proposed = voice_windows(decoded.pcm, params.voice_hop_s or VOICE_WINDOW_S,
                    max_windows=params.audio_windows or MAX_VOICE_WINDOWS)
                record['applied_speech_spans_s'] = spans
                record['proposed_windows'] = [dict(start_sample=round(start * 16000),
                    end_sample=round(start * 16000) + len(chunk),
                    accepted_by_speech_gate=content.window_in_speech(spans, start, start + len(chunk) / 16000))
                    for start, chunk in proposed]
                if record['score'] is not None:
                    if not method.valid_probability(record['raw_unrounded']) or not record['windows']:
                        raise ValueError('Invalid scored aggregate')
                    if abs(np.mean([w['raw_unrounded'] for w in record['windows']]) - record['raw_unrounded']) > 1e-10:
                        raise ValueError('Aggregate differs from its source windows')
            except Exception as error:
                record.update(status='failed', reason=f'{type(error).__name__}: {error}',
                              score=None, raw_unrounded=None, windows=[])
                record.pop('views', None)
            record['latency_s'] = time.perf_counter() - started
            with output.open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
            if index % 25 == 0 or index + 1 == len(remaining):
                print(f'{key}: {index + 1}/{len(remaining)}', flush=True)
        del models, lease, detector, ctx
        gc.collect()


def read_scores(splits):
    m, _ = verify_lock()
    sources = {r['id']: r for r in m['base_files']}
    variants = {r['id']: r for r in m['variants']}
    raw = {}
    for line in (OUT / 'scores.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['variant_id'] not in variants or row['detector'] not in ('old', 'new'):
            raise ValueError('Unexpected score record outside the frozen study')
        if row['split'] != variants[row['variant_id']]['split']:
            raise ValueError('Score split differs from its frozen source')
        if row['split'] not in splits:
            continue
        key = row['variant_id'], row['detector']
        if key in raw:
            raise ValueError('Duplicate detector score')
        raw[key] = row
    result = []
    for variant in m['variants']:
        if variant['split'] not in splits:
            continue
        source = sources[variant['base_id']]
        row = dict(variant, generator=source.get('generator'), dataset=source.get('dataset'),
                   cluster_id=source.get('cluster_id', source['speaker']))
        for key in ('old', 'new'):
            record = raw[(variant['id'], key)]
            for field in ('split', 'label', 'base_id', 'speaker', 'condition'):
                if record[field] != variant[field]:
                    raise ValueError('Score metadata changed: ' + field)
            if record['input_sha256'] != variant['sha256'] or record['production_input']['pcm_sha256'] != variant['pcm_sha256']:
                raise ValueError('Scored source identity mismatch')
            row[key + '_raw'] = record['raw_unrounded'] if record['score'] is not None else None
            row[key + '_status'] = record['status']
            row[key + '_windows'] = record['windows']
            if row[key + '_raw'] is not None and not method.valid_probability(row[key + '_raw']):
                raise ValueError('Invalid persisted probability')
            if any(not 0 <= w['start_sample'] < w['end_sample'] <= variant['samples'] for w in record['windows']):
                raise ValueError('A scored window leaves its source interval')
            if key == 'new':
                views = record.get('views', {})
                row['new_band'] = views.get('band')
                row['new_raw_rounded'] = views.get('raw')
                row['new_views'] = views
        result.append(row)
    return result


def nominal_threshold(rows, key):
    """Conservative 95% rank of real calibration identity maxima; missing=+inf."""
    if any(r['split'] != 'cal' for r in rows):
        raise ValueError('Threshold calibration accepts cal only')
    groups = defaultdict(list)
    for row in rows:
        if row['label'] == 0:
            groups[row['speaker']].append(row[key])
    maxima = {identity: None if any(v is None for v in values) else max(values)
              for identity, values in sorted(groups.items())}
    n = len(maxima)
    rank = math.ceil((n + 1) * .95)
    values = sorted(float('inf') if v is None else v for v in maxima.values())
    order = values[rank - 1] if 0 < rank <= n else float('inf')
    threshold = float(np.nextafter(order, np.inf)) if math.isfinite(order) and order < 1 else None
    return dict(nominal_fpr=.05, identity_count=n, rank=rank, maxima=maxima,
                order_statistic=order if math.isfinite(order) else None, threshold=threshold,
                status='established' if threshold is not None else 'not_established', guarantee=False)


def fit():
    if (OUT / 'test-start.json').exists():
        raise ValueError('Calibration cannot change after test authorization')
    rows = read_scores({'fit', 'cal'})
    parameters, guards, thresholds = {}, {}, {}
    for key in ('old', 'new'):
        parameters[key] = method.fit_platt([r for r in rows if r['split'] == 'fit'], key + '_raw')
        cal = [dict(r, p=method.probability(r[key + '_raw'], parameters[key])) for r in rows if r['split'] == 'cal']
        guards[key] = method.calibrate_class_set(cal)
        thresholds[key] = nominal_threshold(cal, 'p')
    save(OUT / 'calibration.json', dict(schema='chipmunk.audio-calibration/3',
        frozen_at_unix=time.time(), parameters=parameters, guards=guards, thresholds=thresholds,
        method=method.METHOD, deployment_eligible=False, test_scores_accessed=False,
        run_lock_sha256=file_sha(OUT / 'run-lock.json'), development_rows_sha256=sha(util.canonical(rows))))
    print('Fresh fit/cal calibration frozen:', file_sha(OUT / 'calibration.json'), flush=True)


def predictions():
    authorize_test()
    cal = load(OUT / 'calibration.json')
    rows = read_scores({'test'})
    for row in rows:
        for key in ('old', 'new'):
            p = method.probability(row[key + '_raw'], cal['parameters'][key])
            row[key + '_p'] = p
            # Baseline uses its own identity guard with a no-op equal-view stability check.
            raw = row['new_raw_rounded'] if key == 'new' else row[key + '_raw']
            band = row['new_band'] if key == 'new' else raw
            decision = method.decide(p, raw, band, cal['guards'][key])
            row[key + '_decision_detail'] = decision
            row[key + '_probability_only'] = decision['probability_only_decision']
            row[key + '_unguarded'] = decision['identity_guard_decision']
            row[key + '_guarded'] = decision['decision']
            threshold = cal['thresholds'][key]['threshold']
            row[key + '_threshold_established'] = threshold is not None
            row[key + '_threshold_ai'] = bool(p is not None and threshold is not None and p >= threshold)
    return rows


def numeric_metrics(rows, system, multiplicity=None, *, common=False):
    """Fixed original/class weights; missing predictions stay in operational denominators."""
    if not rows:
        return {}
    y = np.array([r['label'] for r in rows])
    key = system.split('_')[0]
    p = np.array([np.nan if r[key + '_p'] is None else r[key + '_p'] for r in rows])
    decisions = np.array([r[system] for r in rows])
    counts = Counter(r['base_id'] for r in rows)
    weights = np.array([1 / counts[r['base_id']] for r in rows], dtype=float)
    if multiplicity is not None:
        weights *= np.asarray(multiplicity)
    assessed = np.isfinite(p)
    if common:
        assessed &= np.array([r['old_p'] is not None and r['new_p'] is not None for r in rows])
    real, ai = y == 0, y == 1
    ai_call, original_call = decisions == 'ai', decisions == 'original'
    decided = ai_call | original_call

    def rate(event, population):
        denominator = weights[population].sum()
        return float(weights[event & population].sum() / denominator) if denominator else None

    class_coverage = [rate(decided, y == c) for c in (0, 1)]
    result = dict(false_ai_rate=rate(ai_call, real), false_original_rate=rate(original_call, ai),
        real_coverage=class_coverage[0], ai_coverage=class_coverage[1],
        coverage=float(np.mean(class_coverage)) if None not in class_coverage else None,
        correct_ai_recall=rate(ai_call, ai), correct_original_recall=rate(original_call, real),
        ai_call_risk=rate(real, ai_call), original_call_risk=rate(ai, original_call),
        decided_risk=rate((real & ai_call) | (ai & original_call), decided),
        assessed_fraction=rate(np.isfinite(p), np.ones(len(y), bool)), brier=None, log_loss=None,
        threshold_fpr=rate(np.array([r[key + '_threshold_ai'] for r in rows]), real),
        threshold_recall=rate(np.array([r[key + '_threshold_ai'] for r in rows]), ai))
    if not all(r[key + '_threshold_established'] for r in rows):
        result['threshold_fpr'] = result['threshold_recall'] = None
    if all(weights[assessed & (y == c)].sum() > 0 for c in (0, 1)):
        probability_weights = weights.copy()
        probability_weights[~assessed] = 0
        for c in (0, 1):
            mask = assessed & (y == c)
            probability_weights[mask] *= .5 / probability_weights[mask].sum()
        p_assessed = np.clip(p[assessed], 1e-12, 1 - 1e-12)
        y_assessed, w_assessed = y[assessed], probability_weights[assessed]
        result['brier'] = float(np.sum(w_assessed * (p_assessed - y_assessed) ** 2))
        result['log_loss'] = float(np.sum(w_assessed * (-y_assessed * np.log(p_assessed)
                                      - (1 - y_assessed) * np.log1p(-p_assessed))))
    return result


def event_bound(rows, system, label, *, threshold=False):
    from scipy.stats import beta
    groups = {r['cluster_id'] for r in rows if r['label'] == label}
    if threshold and not all(r[system.split('_')[0] + '_threshold_established'] for r in rows):
        return dict(events=None, identities=len(groups), one_sided_95_upper=None, status='not_established')
    error = 'ai' if label == 0 else 'original'
    if threshold:
        events = {r['cluster_id'] for r in rows if r['label'] == label and r[system.split('_')[0] + '_threshold_ai']}
    else:
        events = {r['cluster_id'] for r in rows if r['label'] == label and r[system] == error}
    n, k = len(groups), len(events)
    return dict(events=k, identities=n,
        one_sided_95_upper=None if n == 0 else float(beta.ppf(.95, k + 1, n - k)) if k < n else 1.,
        dependence='Descriptive identity-event bound, conditional on the fixed backgrounds and text-group assumptions.')


def point_metrics(rows, system):
    from sklearn.metrics import roc_auc_score
    result = numeric_metrics(rows, system)
    key = system.split('_')[0]
    decisions = Counter(r[system] for r in rows)
    real = sum(r['label'] == 0 for r in rows)
    ai = len(rows) - real
    result.update(inputs=len(rows), real_inputs=real, ai_inputs=ai,
        original_files=len({r['base_id'] for r in rows}), identities=len({r['cluster_id'] for r in rows}),
        decisions={k: decisions[k] for k in ('ai', 'original', 'inconclusive')},
        false_ai_count=sum(r['label'] == 0 and r[system] == 'ai' for r in rows),
        false_original_count=sum(r['label'] == 1 and r[system] == 'original' for r in rows),
        status_counts=dict(Counter(r[key + '_status'] for r in rows)),
        false_ai_identity_event_bound=event_bound(rows, system, 0),
        false_original_identity_event_bound=event_bound(rows, system, 1),
        threshold_false_ai_identity_event_bound=event_bound(rows, system, 0, threshold=True))
    assessed = [r for r in rows if r[key + '_raw'] is not None]
    result['raw_auc'] = float(roc_auc_score([r['label'] for r in assessed], [r[key + '_raw'] for r in assessed],
        sample_weight=util.source_weights(assessed))) if len({r['label'] for r in assessed}) == 2 else None
    result['probability_only_decisions'] = dict(Counter(r[key + '_probability_only'] for r in rows))
    result['stability_veto_count'] = sum(r['new_decision_detail']['original_stability_veto'] for r in rows) if system == 'new_guarded' else 0
    if key == 'new':
        result['probe_status_counts'] = dict(Counter(r['new_views'].get('probe_status', 'original_scoring_failed') for r in rows))
    return result


def sparse_discordance_precision(rows, label):
    """Do not mistake a degenerate zero-discordance bootstrap for precise noninferiority.

    This is a separate identity-event precision diagnostic, not a file-rate CI.
    Each shared voice counts once. A zero new-harm count still has a positive
    exact upper bound. Background and unknown prompt dependence remain caveats.
    """
    from scipy.stats import beta
    identities = {r['cluster_id'] for r in rows if r['label'] == label}
    error = 'ai' if label == 0 else 'original'
    harmed = {r['cluster_id'] for r in rows if r['label'] == label
              and r['new_guarded'] == error and r['old_guarded'] != error}
    helped = {r['cluster_id'] for r in rows if r['label'] == label
              and r['old_guarded'] == error and r['new_guarded'] != error}
    n = len(identities)
    k = len(harmed)
    upper = None if not n else float(beta.ppf(.95, k + 1, n - k)) if k < n else 1.
    no_discordance = not harmed and not helped
    return dict(identities=n, harmed_identities=len(harmed), helped_identities=len(helped),
        zero_discordance=no_discordance, new_harm_event_one_sided_95_upper=upper,
        two_percentage_point_precision_established=not no_discordance or (upper is not None and upper <= .02),
        interpretation='When no paired harm or benefit is observed, [0,0] bootstrap differences are insufficient unless the independent-identity precision also supports 2 percentage points. This is not an exact file-rate difference bound.')


def slices(rows):
    result = {'all': rows}
    for condition in CONDITIONS:
        result[condition] = [r for r in rows if r['condition'] == condition]
    for family in sorted({r['generator'] for r in rows if r['label'] == 1}):
        result['generator:' + family] = [r for r in rows if r['label'] == 0 or r['generator'] == family]
    for name, predicate in [('vctk_matched', lambda r: r['cluster_id'].startswith('vctk:')),
        ('libri_cross_source', lambda r: r['label'] == 1 or not r['cluster_id'].startswith('vctk:')),
        ('seconds_2_to_lt3', lambda r: r['seconds'] < 3),
        ('seconds_3_to_4', lambda r: r['seconds'] >= 3)]:
        result[name] = [r for r in rows if predicate(r)]
    for bg in sorted({r['background_id'] for r in rows if r.get('background_id')}):
        for condition in CONDITIONS:
            selected = [r for r in rows if r.get('background_id') == bg and r['condition'] == condition]
            if selected:
                result['background:' + bg + ':' + condition] = selected
    return result


def bootstrap(rows):
    """Paired joint-voice clusters; crossed version also resamples whole background works."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    identity_groups = defaultdict(set)
    for row in rows:
        stratum = 'vctk_joint' if row['cluster_id'].startswith('vctk:') else 'librispeech'
        identity_groups[stratum].add(row['cluster_id'])
    strata = {k: sorted(v) for k, v in sorted(identity_groups.items())}
    backgrounds = {kind: sorted({r['background_id'] for r in rows if r['condition'].startswith(kind + '_')})
                   for kind in ('noise', 'music')}
    indices = {'all': np.arange(len(rows))}
    indices.update({condition: np.array([i for i, r in enumerate(rows) if r['condition'] == condition]) for condition in CONDITIONS})
    distributions = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        voice_counts = Counter()
        for identities in strata.values():
            voice_counts.update(rng.choice(identities, len(identities), replace=True))
        bg_counts = Counter()
        for identities in backgrounds.values():
            bg_counts.update(rng.choice(identities, len(identities), replace=True))
        conditional = np.array([voice_counts[r['cluster_id']] for r in rows], dtype=float)
        crossed = conditional * np.array([bg_counts[r['background_id']] if r.get('background_id') else 1 for r in rows])
        for kind, weight in [('voice', conditional), ('voice_background', crossed)]:
            for group, idx in indices.items():
                selected = [rows[i] for i in idx]
                values = {s: numeric_metrics(selected, s, weight[idx]) for s in SYSTEMS}
                shared = {s: numeric_metrics(selected, s, weight[idx], common=True) for s in ('old_guarded', 'new_guarded')}
                for metric in ('brier', 'log_loss', 'false_ai_rate', 'false_original_rate', 'coverage',
                               'real_coverage', 'ai_coverage', 'threshold_recall', 'threshold_fpr'):
                    use = shared if metric in ('brier', 'log_loss') else values
                    old, new = use['old_guarded'][metric], use['new_guarded'][metric]
                    distributions[(kind, group, 'new_minus_old:' + metric)].append(None if old is None or new is None else new - old)
                for metric in ('real_coverage', 'ai_coverage', 'false_ai_rate', 'false_original_rate'):
                    before, after = values['new_unguarded'][metric], values['new_guarded'][metric]
                    distributions[(kind, group, 'veto_minus_no_veto:' + metric)].append(None if before is None or after is None else after - before)
    result = {}
    for (kind, group, metric), values in distributions.items():
        finite = [v for v in values if v is not None and np.isfinite(v)]
        result.setdefault(kind, {}).setdefault(group, {})[metric] = dict(replicates=len(finite),
            undefined=BOOTSTRAP_REPLICATES - len(finite),
            percentile_95=np.quantile(finite, [.025, .975]).tolist() if finite else None,
            one_sided_95_upper=float(np.quantile(finite, .95)) if finite else None)
    return dict(seed=BOOTSTRAP_SEED, requested_replicates=BOOTSTRAP_REPLICATES,
        identity_strata={k: len(v) for k, v in strata.items()}, backgrounds={k: len(v) for k, v in backgrounds.items()},
        interpretation='Paired voice clusters preserve shared authentic/synthetic voices and all derivatives. Crossed resampling adds fixed noise/music works. Intervals condition on fitted calibration and tested generators; neither estimates unknown pretraining overlap.',
        intervals=result)


def admission(groups, uncertainty, calibration, precision):
    failures = []
    for kind in ('voice', 'voice_background'):
        for metric in ('brier', 'log_loss'):
            interval = uncertainty['intervals'][kind]['all']['new_minus_old:' + metric]
            if interval['undefined'] or interval['percentile_95'] is None or interval['percentile_95'][1] >= 0:
                failures.append(f'{kind}: common-assessed {metric} improvement is unestablished')
    for condition in CONDITIONS:
        point = groups[condition]['new_guarded']
        mild = condition in ('clean', 'noise_15', 'music_15')
        floor, loss = (.75, .10) if mild else (.50, .25)
        for label in ('real', 'ai'):
            coverage = point[label + '_coverage']
            before = groups[condition]['new_unguarded'][label + '_coverage']
            if coverage is None or coverage < floor:
                failures.append(f'{condition}: {label} decision coverage below {floor:.0%}')
            if coverage is None or before is None or before - coverage > loss + 1e-12:
                failures.append(f'{condition}: {label} coverage loss exceeds {loss:.0%}')
            if not precision[condition][label]['two_percentage_point_precision_established']:
                failures.append(f'{condition}: {label} zero-discordance identity precision cannot establish 2pp noninferiority')
        for kind in ('voice', 'voice_background'):
            for metric in ('false_ai_rate', 'false_original_rate'):
                interval = uncertainty['intervals'][kind][condition]['new_minus_old:' + metric]
                if interval['undefined'] or interval['one_sided_95_upper'] is None or interval['one_sided_95_upper'] > .02:
                    failures.append(f'{condition}/{kind}: {metric} noninferiority is unestablished')
            interval = uncertainty['intervals'][kind][condition]['new_minus_old:threshold_recall']
            if interval['undefined'] or interval['percentile_95'] is None or interval['percentile_95'][0] <= 0:
                failures.append(f'{condition}/{kind}: calibrated-threshold AI recall improvement is unestablished')
        bound = point['threshold_false_ai_identity_event_bound']['one_sided_95_upper']
        if calibration['thresholds']['new']['threshold'] is None or bound is None or bound > .075:
            failures.append(f'{condition}: calibrated 5% threshold lacks <=7.5% upper false-AI bound')
    # This study alone does not establish arbitrary backgrounds, languages, long files or video aggregation.
    return dict(statistical_gates_passed=not failures, failed_gates=failures, deployment_eligible=False,
        overall_confidence_weight=0,
        decision='Keep candidate outside overall confidence pending an independently audited, enforceable serving scope.'
                 if not failures else 'Do not integrate: the fresh validation does not establish the required reliability.',
        old_model_removal_eligible=False,
        remaining_release_requirements=['Independent post-run audit', 'Enforceable supported input scope and serving integration verification',
                                        'Separate candidate-versus-combined comparison before any old-model removal'])


def analyse():
    rows = predictions()
    calibration = load(OUT / 'calibration.json')
    groups = {name: {system: point_metrics(selected, system) for system in SYSTEMS + ['old_probability_only', 'new_probability_only']}
              for name, selected in slices(rows).items() if selected}
    precision = {condition: {label: sparse_discordance_precision([r for r in rows if r['condition'] == condition], code)
                 for label, code in [('real', 0), ('ai', 1)]} for condition in CONDITIONS}
    print('Computing 2,000 paired voice and crossed-background bootstrap replicates.', flush=True)
    uncertainty = bootstrap(rows)
    result = dict(schema='chipmunk.audio-validation-report/3',
        manifest_sha256=manifest()['sha256'], run_lock_sha256=file_sha(OUT / 'run-lock.json'),
        calibration_sha256=file_sha(OUT / 'calibration.json'), scores_sha256=file_sha(OUT / 'scores.jsonl'),
        test_start_sha256=file_sha(OUT / 'test-start.json'), method=method.METHOD,
        cohort=manifest().get('audit'), groups=groups, uncertainty=uncertainty, sparse_discordance_precision=precision,
        common_assessed={s: numeric_metrics(rows, s, common=True) for s in SYSTEMS},
        admission=admission(groups, uncertainty, calibration, precision), serving_unchanged=serving_hashes() == load(OUT / 'run-lock.json')['serving_hashes'],
        limitations=[f'Held-out English speech prefixes of {min(r["seconds"] for r in rows):g} to {max(r["seconds"] for r in rows):g} seconds; no validation of shorter/longer recordings, partial edits or arbitrary video soundtracks.',
            'Fresh relative to prior studies, not proven independent of unknown pretrained detector corpora.',
            'Controlled additive backgrounds from fixed licensed pools; no blanket claim for real rooms, codecs or multilingual speech.',
            'DFADD prompt texts are not individually mapped. Corpus-level exclusions and declared indices do not establish exact unique transcripts.',
            'Reference probabilities assume a balanced 50:50 class prior; they are not measured customer-population probabilities.',
            'A sensitivity veto changes abstention, not the probability or Brier/log-loss of the candidate.',
            'Repeated voice and background groups are correlated; original and derivative counts are reported separately.',
            'The test is now consumed. Further method selection requires another fresh confirmatory test.'])
    save(OUT / 'predictions.json', rows)
    save(OUT / 'evaluation.json', result)
    print(json.dumps(result['admission'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['lock', 'score', 'fit', 'analyse'])
    parser.add_argument('--stage', choices=['development', 'test'], default='development')
    args = parser.parse_args()
    score(args.stage) if args.command == 'score' else globals()[args.command]()
