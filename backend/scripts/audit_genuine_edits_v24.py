"""Read-only source, availability and aggregate review of the ordinary-edit run.

No models, fits, point selection or source replacements. Raw anchors, source
pairings and native face journals stay private; only aggregate receipts leave
the local trial directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import run_genuine_edits_v24 as run


def lines(path):
    import json

    return [json.loads(s) for s in path.read_text().splitlines()]


def source_proof(planned, native):
    """Check each measured frame against the completed model-free RGB audit."""
    run.require(planned['plan']['id'] == native['id']
        and planned['plan']['sha256'] == native['input_sha256']
        and planned['plan']['source_geometry']['pts_table_sha256'] == native['native_table_sha256'],
        'Native input source or clock changed')
    expected = {i for u in planned['plan']['units'] for key in ('sampled_frames', 'context_frames')
        for i in u['location'][key]}
    measured = native['frame_inputs']
    run.require(len(measured) == len(expected) and {r['frame_index'] for r in measured} == expected,
        'Every planned native source frame required exactly once')
    for row in measured:
        run.require(row['source_rgb_sha256'] == planned['rgb_sha256'][row['frame_index']],
            'Measured frame differs from audited RGB')
        run.require(Fraction(row['time_base']) * row['pts'] == Fraction(row['frame_index'], 24),
            'Measured frame PTS differs from the audited zero-origin native clock')
    counts = native['native_input_counts']
    reconstructed = {
        'face_detection_inputs': len(measured),
        'crop_classifier_inputs': sum('crop_tensor_sha256' in r for r in measured),
        'appearance_inputs': sum(r.get('appearance_status') == 'ok' for r in measured),
        'patch_feature_inputs': sum('patch_feature_sha256' in r for r in measured),
        'native_repeat_inputs': len(native['native_repeats']),
    }
    run.require(counts == reconstructed, 'Native input counts differ from recorded per-frame evidence')
    return counts


def counter_summary(outcomes):
    """Recount nullable decisions without coercing unassessed values to False."""
    result = {}
    for kind in ('original-baseline', 'ordinary-edit'):
        rows = [r for r in outcomes if r['kind'] == kind]
        run.require(len(rows) == 40 and len({r['component'] for r in rows}) == 20,
            'All cases and connected components must remain')
        result[kind] = {'planned': len(rows), 'connected_components': 20, 'raw_cues': {}, 'reference_methods': {}}
        for view in run.native_base.VIEWS:
            values = [r['raw_inspection_alerts_not_ai_proof'][view] for r in rows]
            run.require(all(type(v) is bool or v is None for v in values), 'Explicit nullable alert required')
            result[kind]['raw_cues'][view] = {
                'inspection_alerts': sum(v is True for v in values), 'inconclusive': sum(v is None for v in values)}
        for arm in (*run.ARMS, 'appearance-mapped'):
            values = [r['reference_ai_proxy_flags'][arm] for r in rows]
            run.require(all(type(v) is bool or v is None for v in values), 'Explicit nullable proxy flag required')
            result[kind]['reference_methods'][arm] = {
                'unsupported_ai_proxy_flags': sum(v is True for v in values),
                'inconclusive': sum(v is None for v in values),
                'components_with_proxy_flags': len({r['component'] for r in rows if r['reference_ai_proxy_flags'][arm] is True})}
    return result


def audit(main, trial):
    run.check_score(main, trial)
    profiles = {stage: run.read(trial / (stage + '-profile/profile.json')) for stage in ('source-audit', 'native')}
    for value in profiles.values():
        run.require(value['child_exit_code'] == 0 and value['guard'] is None
            and value['sampled_tree_rss_lower_bound_bytes'] <= 6 * 1024**3
            and value['duration_s'] <= value['seconds_limit'] <= 900, 'Both guarded stages must finish successfully')
    prepared = lines(trial / 'input-plan.jsonl')
    planned = {r['id']: r for r in prepared}
    journal = lines(trial / 'native.jsonl')
    run.require(journal[0]['freeze_sha256'] == run.sha(trial / 'scoring-freeze.json'), 'Native header is not bound to scoring freeze')
    ordinary = {r['id']: r for r in journal[1:] if r['kind'] == 'ordinary-edit'}
    parity = [r for r in journal[1:] if r['kind'] == 'original-native-parity']
    run.require(len(journal) == 43 and len(prepared) == len(planned) == len(ordinary) == 40
        and len(parity) == 2 and set(planned) == set(ordinary), 'Complete unique native run required')
    receipt = run.read(trial / 'native-receipt.json')
    run.require(receipt['journal_sha256'] == run.sha(trial / 'native.jsonl'), 'Native journal digest changed')
    source_trial = main / 'backend/data/independent-evaluation/video-specialists-v18'
    original_rgb = {r['id']: r['rgb_sha256'] for r in lines(source_trial / 'source-frame-audit-v1.jsonl')[1:]}
    old_plans = {r['id']: r for r in run.read(source_trial / 'scoring-input-plan-v2.json')['files']}
    source_plan = run.read(trial / 'source-plan.json')
    baseline = run.read(trial / 'baselines.json')
    run.require({r['id'] for r in baseline} == {p['id'] for pair in source_plan['pairs'] for p in pair['parents']}
        and len(baseline) == 40, 'All original baseline sources must remain')
    for index, pair in enumerate(source_plan['pairs']):
        for direction, (target, donor) in enumerate(pair['directions']):
            row = planned[pair['component'] + '.' + str(direction)]
            run.require(row['pair_index'] == index and row['direction'] == direction
                and row['component'] == pair['component']
                and row['target_parent'] == pair['parents'][target]['id']
                and row['donor_parent'] == pair['parents'][donor]['id'],
                'Native edit source pairing differs from prospective plan')
    counts, repeats = Counter(), Counter()
    anchors = frames = 0
    models = run.artifacts()
    for row in prepared:
        target, donor = original_rgb[row['target_parent']], original_rgb[row['donor_parent']]
        run.require(row['rgb_sha256'] == target[:48] + donor[48:96] + target[96:]
            and row['plan']['label'] == 0 and row['plan']['known_edit'] is None
            and all(u['label'] == 0 for u in row['plan']['units']),
            'Ordinary copy construction or no-new-AI label changed')
        run.require(set(row['anchors']) == {'36', '72', '108'}, 'All before/during/after anchors required')
        for index, proof in row['anchors'].items():
            path = trial / proof['path']
            run.require(path.resolve().is_relative_to((trial / 'anchors').resolve())
                and run.sha(path) == proof['sha256'], 'Private anchor path or PNG bytes changed')
            with Image.open(path) as image:
                run.require(image.mode == 'RGB' and image.size == (640, 480), 'Native source anchor geometry changed')
                rgb_hash = run.media.digest(np.asarray(image))
            run.require(rgb_hash == proof['source_rgb_sha256'] == row['rgb_sha256'][int(index)]
                and proof['time_ns'] == round(int(index) / 24 * 10**9), 'Anchor RGB/time source correspondence changed')
            anchors += 1
        frames += len(row['rgb_sha256'])
        counts.update(source_proof(row, ordinary[row['id']]))
        repeats.update(ordinary[row['id']]['native_repeats'])
        for artifact in models.values():
            run.bind(row['plan'], ordinary[row['id']], artifact)
    cached = run.native_base.responses(source_trial, 'confirmation')
    maximum = 0.
    for row in parity:
        proof = {'plan': old_plans[row['id']], 'rgb_sha256': original_rgb[row['id']]}
        counts.update(source_proof(proof, row))
        maximum = max(maximum, run.parity(row, cached[row['id']]))
    run.require(dict(counts) == receipt['native_input_counts']
        and repeats == Counter({'crop': 1, 'appearance': 1, 'features': 1})
        and maximum == receipt['original_parity_maximum_raw_difference'], 'Native totals, repeats or cached parity receipt changed')
    outcomes = run.read(trial / 'diagnostic-outcomes.json')['outcomes']
    report = run.read(trial / 'result.json')
    run.require(len(outcomes) == len({r['id'] for r in outcomes}) == 80
        and {r['id'] for r in outcomes} == set(planned) | {r['id'] for r in baseline}
        and all(r['benchmark_no_new_ai_label'] == 0 for r in outcomes), 'All diagnostic outcomes required with no-new-AI truth')
    run.require(counter_summary(outcomes) == report['summary'], 'Aggregate counts differ from all retained outcomes')
    spans, insert_overlaps, overlapping_clips = 0, 0, 0
    points = {a: run.read(run.ROOT / f'docs/evaluation-panels/preserved-evidence-v23-{a}-point.json') for a in run.ARMS}
    points['appearance-mapped'] = run.read(run.ROOT / 'docs/evaluation-panels/preserved-evidence-v23-appearance-mapped-point.json')
    for row in outcomes:
        native = ordinary.get(row['id'], cached.get(row['id']))
        plan = planned[row['id']]['plan'] if row['kind'] == 'ordinary-edit' else old_plans[row['id']]
        for arm, model in models.items():
            expected = run.research.predict(run.bind(plan, native, model), model)
            run.require(expected == row['frozen_reference_scores_not_customer_confidence'][arm],
                'Diagnostic score does not reproduce with exact frozen reference')
            point = points[arm]
            flag = expected > point['threshold'] if expected is not None and point['status'] == 'fitted' else None
            run.require(flag == row['reference_ai_proxy_flags'][arm], 'Frozen internal evaluation point changed')
        point = points['appearance-mapped']
        value = row['frozen_reference_scores_not_customer_confidence']['appearance-direct']
        mapped = value > point['threshold'] if value is not None and point['status'] == 'fitted' else None
        run.require(mapped == row['reference_ai_proxy_flags']['appearance-mapped'], 'Mapped raw inspection point changed')
        for view in run.native_base.VIEWS:
            run.require(row['raw_inspection_alerts_not_ai_proof'][view] == run.raw_alert(native, view),
                'Raw cue availability or frozen direction changed')
        expected_spans = run.native_base.intervals([u['location'] for u in native['units']
            if u['view'] == 'face-appearance' and u['status'] == 'ok'
            and u['raw'] > run.native_base.RAW_POINTS['face-appearance']])
        run.require(expected_spans == row['appearance_flagged_spans_ns'], 'Inspection spans lost source correspondence')
        if row['kind'] == 'ordinary-edit':
            spans += len(expected_spans)
            insert_overlaps += sum(max(0, min(b, 4 * 10**9) - max(a, 2 * 10**9)) > 0 for a, b in expected_spans)
            expected_iou = run.native_base.best_iou(expected_spans, [2 * 10**9, 4 * 10**9])
            run.require(expected_iou == row['ordinary_insert_span_iou'], 'Known ordinary insertion overlap changed')
            overlapping_clips += expected_iou >= .3
    return {'schema': 'chipmunk.ordinary-edit-readonly-audit/24', 'status': 'passed',
        'ordinary_clips': 40, 'cached_original_baselines': 40, 'connected_components': 20,
        'exact_copied_rgb_frames_checked': frames, 'private_png_anchors_checked': anchors,
        'native_rows_including_original_parity': 42, 'native_input_counts': dict(counts),
        'native_frame_timestamps_independently_checked': counts['face_detection_inputs'],
        'original_parity_maximum_raw_difference': maximum,
        'ordinary_appearance_spans': spans, 'spans_overlapping_ordinary_insertion': insert_overlaps,
        'ordinary_clips_with_best_span_iou_at_least_point3': overlapping_clips,
        'overlap_interpretation': 'Correspondence with an ordinary source insertion; not AI localization or pixel authorship.',
        'profiles': {k: {n: v[n] for n in ('child_exit_code', 'guard', 'sampled_tree_rss_lower_bound_bytes', 'sample_errors', 'duration_s', 'seconds_limit')} for k, v in profiles.items()},
        'private_commitments_sha256': {n: run.sha(trial / n) for n in (
            'preparation-freeze.json', 'source-audit.json', 'scoring-freeze.json', 'native-receipt.json', 'result.json')},
        'new_neural_forwards_in_review': 0, 'new_fitting_or_point_selection': False,
        'public_data': 'Aggregate receipts only. No raw frames, private pairings, face vectors or per-frame geometry/hash journals.',
        'serving_admitted': False, 'outside_human_reviews': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--output', default='readonly-audit.json', help='New private receipt name; existing receipts are never overwritten')
    args = parser.parse_args()
    run.write(args.trial / args.output, audit(args.main.resolve(), args.trial.resolve()))
    print('Read-only ordinary-edit source and aggregate audit passed.', flush=True)
