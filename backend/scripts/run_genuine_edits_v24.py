"""Source-audited ordinary-edit stress test with fixed research-only checks."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from collections import Counter
from pathlib import Path

from app.bench import specialists as legacy
from scripts import genuine_edit_materializer_v24 as media
from scripts import prepare_video_scoring_units_v18 as scheduling
from scripts import score_video_specialists_v20 as native_base
from scripts import specialists_preserved_v4 as research

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('appearance-direct', 'appearance-positive', 'family-max', 'family-repeat', 'view-max')
POLICY = {'schema': 'chipmunk.genuine-ordinary-edit-policy/24',
    'source': 'All40 publisher-original consumed confirmation sources;20 predeclared two-source components, both directions.40 new ordinary edits and40 cached original baselines.',
    'construction': 'Unmodified native RGB frames48:96 from donor replace same target range.24fps actual zero-origin clock, source640x480. LosslessRGB encoded, every decoded frame must equal expected copied source pixels.',
    'truth': 'Benchmark no-new-AI label0 based on publisher original labels and copy-only construction. Upstream origin and identities not independently established. Different sources do not necessarily mean different people.',
    'scope': 'Visual ordinary source/scene substitution diagnostic. Consumed sources, no training/calibration/fresh-reserve or full-app assessment.',
    'scoring': 'Same source-bound scheduled face-crop/appearance/feature windows, cached commercial weights, nativeCPU/MPSfloat32, batch8/12. No model or point replacement.',
    'baselines': 'Verify exact original bytes/native tables/RGB against v18 audit, reuse v20 native responses; independently re-execute first pair originals within raw tolerance1e-4 and exact availability before reuse.',
    'repeats': 'First eligible new ordinary crop/appearance/feature inputs repeat. Crop absolute1e-4, appearance bitwise, patch absolute/relative1e-4. All source RGB and every native appearance alignment verified by original helper.',
    'inference': 'Raw appearance alerts can correctly indicate an ordinary appearance change. Only interpreting them as AI authorship is unsupported. v23 scores are frozen reference outputs in an unvalidated context, never calibrated customer confidence.',
    'points': 'Keep v20 existing raw cue constants and all v23 standard/mapped points unchanged. Never select an internal point from these new outcomes.',
    'missing': 'Retain every planned unit and case. Missing input/required view yields no numeric file score; no original-file verdict.',
    'evidence': 'Private actual source PNG anchors at36/72/108 before/during/after; native source clock and flagged ranges retained. No newAI edit mask or person identification.',
    'guard': {'rss_gib': 6, 'available_mib': 1536, 'disk_mib': 1536, 'seconds': 900},
    'serving_admitted': False}
PINS = ('backend/scripts/run_genuine_edits_v24.py', 'backend/scripts/genuine_edit_materializer_v24.py',
    'backend/scripts/specialist_runtime_v11.py', 'backend/tests/test_genuine_edit_materializer_v24.py',
    'backend/tests/test_genuine_edit_scoring_v24.py', 'docs/video-genuine-edits-v24-protocol.md')


def sha(path):
    return media.previous.inventory.sha(path)


def read(path):
    return json.loads(Path(path).read_text())


write, require = media.previous.inventory.write, media.require


def prep_identity(main, trial):
    source = main / 'backend/data/independent-evaluation/video-specialists-v18'
    names = set(PINS) | set(native_base.source_audit.CODE)
    return {'schema': 'chipmunk.ordinary-edit-preparation-freeze/24', 'policy': POLICY,
        'code_sha256': [{'path': n, 'sha256': sha(ROOT / n)} for n in sorted(names)],
        'inputs_sha256': {n: sha(source / n) for n in ('input-plan.jsonl', 'input-preparation.json',
            'input-preparation-policy.json', 'source-inventory.json', 'scoring-input-plan-v2.json', 'source-frame-audit-v1.jsonl')},
        'source_plan_sha256': sha(trial / 'source-plan.json'),
        'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'opencv-python-headless', 'Pillow')},
        'binaries_sha256': {n: sha(Path(shutil.which(n))) for n in ('ffmpeg', 'ffprobe')},
        'new_neural_forwards': 0, 'serving_admitted': False}


def check_preparation(main, trial):
    require(read(trial / 'preparation-freeze.json') == prep_identity(main, trial), 'Frozen source/construction/clock code or inputs changed')


def audit(main, trial):
    from PIL import Image

    check_preparation(main, trial)
    source, original, grouped, original_rgb = media.source_inputs(main, trial)
    anchors = trial / 'anchors'
    anchors.mkdir(exist_ok=False)
    rows, baselines, frames = [], [], 0
    with (trial / 'input-plan.jsonl').open('x') as stream:
        for index, pair in enumerate(source['pairs']):
            with media.pair_files(main, trial, pair, original, grouped, original_rgb) as (parents, composites):
                for parent in parents:
                    baselines.append({'id': parent['planned']['id'], 'sha256': parent['planned']['sha256'],
                        'component': pair['component'], 'source_geometry': parent['planned']['source_geometry']})
                for direction, item in enumerate(composites):
                    row = {'id': item['plan']['id'], 'component': pair['component'], 'pair_index': index,
                        'direction': direction, 'plan': item['plan'], 'rgb_sha256': item['rgb_sha256'],
                        'target_parent': item['target_parent'], 'donor_parent': item['donor_parent'], 'anchors': {}}
                    for frame, rgb in item['anchors'].items():
                        path = anchors / f'pair-{index:02d}-direction-{direction}-frame-{frame}.png'
                        Image.fromarray(rgb).save(path)
                        row['anchors'][frame] = {'path': str(path.relative_to(trial)), 'sha256': sha(path),
                            'source_rgb_sha256': media.digest(rgb), 'time_ns': round(int(frame) / 24 * 10**9)}
                    rows.append(row)
                    frames += len(row['rgb_sha256'])
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                print(json.dumps({'source_pairs_complete': index + 1, 'planned_pairs': 20, 'ordinary_clips': len(rows),
                    'new_neural_forwards': 0}), flush=True)
    require(len(rows) == len(baselines) == 40 and len({b['id'] for b in baselines}) == 40, 'All ordinary and original cases required')
    check_preparation(main, trial)
    write(trial / 'baselines.json', baselines)
    write(trial / 'source-audit.json', {'schema': 'chipmunk.ordinary-edit-source-audit/24',
        'source_pairs': 20, 'ordinary_clips': len(rows), 'baseline_originals': len(baselines),
        'decoded_ordinary_rgb_frames': frames, 'source_rgb_exact': True, 'actual_native_clocks_verified': True,
        'private_before_during_after_anchors': sum(len(r['anchors']) for r in rows),
        'input_plan_sha256': sha(trial / 'input-plan.jsonl'), 'baseline_sha256': sha(trial / 'baselines.json'),
        'preparation_freeze_sha256': sha(trial / 'preparation-freeze.json'), 'new_neural_forwards': 0,
        'serving_admitted': False, 'limits': POLICY['truth']})


def artifacts():
    prefix = ROOT / 'docs/evaluation-panels'
    return {arm: read(prefix / ('preserved-evidence-v23-' + arm + '-artifact.json')) for arm in ARMS}


def validate_artifact(artifact):
    require(artifact['schema'] == 'chipmunk.specialist-preserved-calibration/4'
        and artifact['serving_admitted'] is False
        and artifact['implementation_sha256'] == research.implementation()
        and artifact['packages'] == {n: importlib.metadata.version(n) for n in ('numpy', 'scipy')}
        and artifact['sha256'] == legacy.digest({k: v for k, v in artifact.items() if k != 'sha256'}),
        'Exact frozen compatible research artifact required for forward-only diagnostic')


def score_identity(main, trial):
    check_preparation(main, trial)
    source = main / 'backend/data/independent-evaluation/video-specialists-v18'
    old, _ = native_base.check(source)
    audit = read(trial / 'source-audit.json')
    require(audit['ordinary_clips'] == 40 and audit['source_pairs'] == 20
        and audit['input_plan_sha256'] == sha(trial / 'input-plan.jsonl')
        and audit['baseline_sha256'] == sha(trial / 'baselines.json'), 'Complete model-free input audit required')
    for value in artifacts().values():
        validate_artifact(value)
    selected = [f'docs/evaluation-panels/preserved-evidence-v23-{a}-{k}.json' for a in ARMS for k in ('artifact', 'point')]
    selected += ['docs/evaluation-panels/preserved-evidence-v23-appearance-mapped-point.json',
        'backend/scripts/specialists_preserved_v4.py', 'backend/scripts/monotone_newton_v4.py',
        'backend/scripts/measure_video_specialists_v18.py', 'backend/scripts/score_video_specialists_v20.py']
    return {'schema': 'chipmunk.ordinary-edit-scoring-freeze/24', 'policy': POLICY,
        'preparation_identity': prep_identity(main, trial),
        'input_sha256': {n: sha(trial / n) for n in ('input-plan.jsonl', 'baselines.json', 'source-audit.json')},
        'code_artifacts_points_sha256': [{'path': n, 'sha256': sha(ROOT / n)} for n in selected],
        'original_native_freeze_sha256': sha(source / 'v20-freeze.json'), 'models': old['models'],
        'original_confirmation_native_receipt_sha256': sha(source / 'confirmation-v20-native.json'),
        'original_confirmation_native_journal_sha256': sha(source / 'confirmation-v20-native.jsonl'),
        'runtime': native_base.runtime(), 'new_neural_forwards_before_scoring': 0, 'serving_admitted': False}


def check_score(main, trial):
    require(read(trial / 'scoring-freeze.json') == score_identity(main, trial), 'Frozen scoring implementation, models, points or audited inputs changed')


def parity(current, cached):
    require(current['input_sha256'] == cached['input_sha256'] and current['native_table_sha256'] == cached['native_table_sha256'],
        'Original native parity uses another input or clock')
    require(len(current['units']) == len(cached['units']), 'Native parity unit count changed')
    maximum = 0.
    for left, right in zip(current['units'], cached['units'], strict=True):
        require(all(left[k] == right[k] for k in ('id', 'view', 'location', 'status')), 'Native original unit/scope/availability changed')
        if left['raw'] is None or right['raw'] is None:
            require(left['raw'] is right['raw'], 'Missing original response changed')
        else:
            maximum = max(maximum, abs(left['raw'] - right['raw']))
    require(maximum <= 1e-4, 'Original model-input native parity exceeded frozen limit')
    return maximum


def native(main, trial):
    check_score(main, trial)
    native_base.configure(main, trial, 'ordinary-edits')
    import cv2
    import torch

    from app.engine.models import get_model_manager
    from scripts.measure_video_specialists_v18 import measure

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    require(torch.backends.mps.is_available(), 'Frozen MPS execution required')
    source, original, grouped, original_rgb = media.source_inputs(main, trial)
    expected = {r['id']: r for r in [json.loads(s) for s in (trial / 'input-plan.jsonl').read_text().splitlines()]}
    cached = native_base.responses(main / 'backend/data/independent-evaluation/video-specialists-v18', 'confirmation')
    repeats, counts, completed, parity_max = set(), Counter(), [], 0.
    manager = get_model_manager()
    with (trial / 'native.jsonl').open('x') as stream:
        stream.write(json.dumps({'schema': 'chipmunk.ordinary-edit-native-journal/24',
            'freeze_sha256': sha(trial / 'scoring-freeze.json')}) + '\n')
        stream.flush()
        with manager.lease((native_base.MODELS[0], native_base.MODELS[2]), prefer='cpu') as cpu, \
                manager.lease((native_base.MODELS[1], native_base.MODELS[3]), prefer='mps', activation_bytes=2400 * 1024**2) as accel:
            require(cpu.device == 'cpu' and accel.device == 'mps' and accel.dtype == 'float32', 'Frozen devices/precision changed')
            objects = native_base.native_objects(cpu, accel)
            for index, pair in enumerate(source['pairs']):
                with media.pair_files(main, trial, pair, original, grouped, original_rgb) as (parents, composites):
                    if index == 0:
                        for parent in parents:
                            plan = {**parent['planned'], **scheduling.plan_case(parent['planned'])}
                            result = measure(parent, plan, *objects, accel.device,
                                source_rgb_sha256=original_rgb[parent['planned']['id']])
                            parity_max = max(parity_max, parity(result, cached[parent['planned']['id']]))
                            counts.update(result['native_input_counts'])
                            stream.write(json.dumps({'kind': 'original-native-parity', **result}, allow_nan=False) + '\n')
                            stream.flush()
                    for item in composites:
                        commitment = expected[item['plan']['id']]
                        require(item['plan'] == commitment['plan'] and item['rgb_sha256'] == commitment['rgb_sha256'],
                            'Regenerated ordinary input bytes, units, RGB or native clock differ from pre-score audit')
                        result = measure(item['case'], item['plan'], *objects, accel.device,
                            repeat_views={'crop', 'appearance', 'features'} - repeats, source_rgb_sha256=item['rgb_sha256'])
                        repeats.update(result['native_repeats'])
                        counts.update(result['native_input_counts'])
                        stream.write(json.dumps({'kind': 'ordinary-edit', **result}, allow_nan=False) + '\n')
                        stream.flush()
                        completed.append(result['id'])
                print(json.dumps({'ordinary_clips_complete': len(completed), 'planned': 40, 'native_input_counts': counts}), flush=True)
    require(len(completed) == 40 and set(completed) == set(expected), 'Every planned ordinary clip required')
    require(repeats == {'crop', 'appearance', 'features'}, 'All three ordinary native repeats required')
    check_score(main, trial)
    write(trial / 'native-receipt.json', {'schema': 'chipmunk.ordinary-edit-native-receipt/24',
        'ordinary_clips': 40, 'original_native_parity_clips': 2, 'original_parity_maximum_raw_difference': parity_max,
        'native_input_counts': counts, 'new_input_repeats': sorted(repeats), 'journal_sha256': sha(trial / 'native.jsonl'),
        'serving_admitted': False, 'reference_scope': POLICY['inference']})


def bind(plan, result, artifact):
    """Forward-only diagnostic input binding, without changing frozen fit plans."""
    require(plan['id'] == result['id'] and plan['sha256'] == result['input_sha256']
        and plan['source_geometry']['pts_table_sha256'] == result['native_table_sha256'], 'Forward diagnostic source/clock mismatch')
    measured = {u['id']: u for u in result['units']}
    require(len(measured) == len(result['units']) == len(plan['units']), 'Every source-bound unit must be recorded exactly once')
    specs = {s['id']: s for s in artifact['policy']['specialists']}
    units = []
    for expected in plan['units']:
        current = measured[expected['id']]
        require(current['view'] == expected['view'] and current['location'] == expected['location'], 'Forward local location/view mismatch')
        if current['status'] == 'ok':
            require(legacy.finite(current['raw']), 'Finite original response required')
        else:
            require(current['raw'] is None and bool(current['reason']), 'Missing response cannot become zero')
        if current['view'] in specs:
            spec = specs[current['view']]
            require(spec['modality'] == 'video' and spec['contexts'] == ['single-face'], 'Original diagnostic routing context required')
            if current['raw'] is not None:
                require(spec['raw_range'][0] <= current['raw'] <= spec['raw_range'][1], 'Response outside frozen range')
            units.append({**current, 'specialist': current['view'], 'context': 'single-face'})
    require({u['specialist'] for u in units} == set(specs), 'Every required frozen view must be present or explicitly unassessed')
    return {'id': plan['id'], 'modality': 'video', 'required_families': ['video-face'], 'units': units}


def raw_alert(result, view):
    selected = [u for u in result['units'] if u['view'] == view]
    if not selected or any(u['status'] != 'ok' for u in selected):
        return None
    return any(u['raw'] > native_base.RAW_POINTS[view] for u in selected)


def review(main, trial):
    check_score(main, trial)
    receipt = read(trial / 'native-receipt.json')
    require(receipt['ordinary_clips'] == 40 and receipt['journal_sha256'] == sha(trial / 'native.jsonl'), 'Complete unchanged native receipt required')
    rows = [json.loads(line) for line in (trial / 'native.jsonl').read_text().splitlines()][1:]
    native_rows = {r['id']: r for r in rows if r['kind'] == 'ordinary-edit'}
    planned = {r['id']: r for r in [json.loads(s) for s in (trial / 'input-plan.jsonl').read_text().splitlines()]}
    require(set(native_rows) == set(planned) and len(native_rows) == 40, 'All forty native records required')
    source_trial = main / 'backend/data/independent-evaluation/video-specialists-v18'
    cached = native_base.responses(source_trial, 'confirmation')
    originals = {r['id']: r for r in read(source_trial / 'scoring-input-plan-v2.json')['files']}
    baseline = read(trial / 'baselines.json')
    models = artifacts()
    for model in models.values():
        validate_artifact(model)
    points = {a: read(ROOT / f'docs/evaluation-panels/preserved-evidence-v23-{a}-point.json') for a in ARMS}
    points['appearance-mapped'] = read(ROOT / 'docs/evaluation-panels/preserved-evidence-v23-appearance-mapped-point.json')
    cases = [(r, r['plan'], native_rows[r['id']], 'ordinary-edit') for r in planned.values()]
    cases += [(r, originals[r['id']], cached[r['id']], 'original-baseline') for r in baseline]
    outcomes = []
    for meta, plan, measurement, kind in cases:
        scores, flags = {}, {}
        for arm, model in models.items():
            value = research.predict(bind(plan, measurement, model), model)
            scores[arm] = value
            point = points[arm]
            flags[arm] = value > point['threshold'] if value is not None and point['status'] == 'fitted' else None
        point = points['appearance-mapped']
        value = scores['appearance-direct']
        flags['appearance-mapped'] = value > point['threshold'] if value is not None and point['status'] == 'fitted' else None
        alerts = {view: raw_alert(measurement, view) for view in native_base.VIEWS}
        require(flags['appearance-mapped'] == alerts['face-appearance'], 'Mapped raw reference changes inspection direction or scope')
        locations = [u['location'] for u in measurement['units'] if u['view'] == 'face-appearance'
            and u['status'] == 'ok' and u['raw'] > native_base.RAW_POINTS['face-appearance']]
        spans = native_base.intervals(locations)
        outcomes.append({'id': plan['id'], 'component': meta['component'], 'kind': kind, 'benchmark_no_new_ai_label': 0,
            'frozen_reference_scores_not_customer_confidence': scores, 'reference_ai_proxy_flags': flags,
            'raw_inspection_alerts_not_ai_proof': alerts, 'appearance_flagged_spans_ns': spans,
            'ordinary_insert_span_iou': native_base.best_iou(spans, [2 * 10**9, 4 * 10**9]) if kind == 'ordinary-edit' else None,
            'source_native_table_sha256': plan['source_geometry']['pts_table_sha256']})
    summary = {}
    for kind in ('original-baseline', 'ordinary-edit'):
        selected = [r for r in outcomes if r['kind'] == kind]
        summary[kind] = {'planned': len(selected), 'connected_components': len({r['component'] for r in selected}), 'raw_cues': {}, 'reference_methods': {}}
        for view in native_base.VIEWS:
            values = [r['raw_inspection_alerts_not_ai_proof'][view] for r in selected]
            summary[kind]['raw_cues'][view] = {'inspection_alerts': sum(v is True for v in values), 'inconclusive': sum(v is None for v in values)}
        for arm in (*ARMS, 'appearance-mapped'):
            summary[kind]['reference_methods'][arm] = {'unsupported_ai_proxy_flags': sum(r['reference_ai_proxy_flags'][arm] is True for r in selected),
                'inconclusive': sum(r['reference_ai_proxy_flags'][arm] is None for r in selected),
                'components_with_proxy_flags': len({r['component'] for r in selected if r['reference_ai_proxy_flags'][arm] is True})}
    write(trial / 'diagnostic-outcomes.json', {'outcomes': outcomes, 'reference_scope': POLICY['inference'], 'serving_admitted': False})
    report = {'schema': 'chipmunk.ordinary-edit-diagnostic-result/24', 'summary': summary,
        'source_components': 20, 'source_originals': 40, 'new_ordinary_inputs': 40, 'cached_original_baselines': 40,
        'benchmark_truth': POLICY['truth'], 'scope': POLICY['scope'], 'interpretation': POLICY['inference'],
        'new_calibration_or_point_fitting': False, 'serving_admitted': False, 'outside_human_reviews': 0}
    write(trial / 'result.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze-preparation', 'audit', 'freeze-scoring', 'native', 'review'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    main, trial = args.main.resolve(), args.trial.resolve()
    if args.stage == 'freeze-preparation':
        write(trial / 'preparation-freeze.json', prep_identity(main, trial))
    elif args.stage == 'audit':
        audit(main, trial)
    elif args.stage == 'freeze-scoring':
        native_base.configure(main, trial, 'asset-preflight')
        from app.engine.fetchers import ensure_verified
        from app.engine.registry import get_registry

        expected = score_identity(main, trial)
        for model, proof in expected['models'].items():
            entry = get_registry().manifest.model(model)
            require(entry.licence.class_ == 'commercial', 'Commercial-admitted cached weights required')
            require({str(p.resolve().relative_to(main)): sha(p) for p in ensure_verified(entry)} == proof['files'], 'Cached weights differ from original native contract')
        write(trial / 'scoring-freeze.json', expected)
    elif args.stage == 'native':
        native(main, trial)
    else:
        review(main, trial)
