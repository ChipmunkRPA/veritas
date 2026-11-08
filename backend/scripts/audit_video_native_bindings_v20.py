"""Read-only private source/input binding review; zero detector forwards.

This checks commitments and availability, not the correctness of a neural
response or AI authorship. Geometry, alignments and input hashes stay private.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

from app.bench import specialists as engine
from scripts import inventory_video_specialists_v18 as source
from scripts import score_video_specialists_v20 as study


def inspect(plan, measured, source_frames):
    study.require(measured['id'] == plan['id'] and measured['input_sha256'] == plan['sha256']
        and measured['native_table_sha256'] == plan['source_geometry']['pts_table_sha256'], 'Wrong native source commitment')
    expected = {i for unit in plan['units'] for key in ('sampled_frames', 'context_frames') for i in unit['location'][key]}
    frames = measured['frame_inputs']
    by_frame = {f['frame_index']: f for f in frames}
    study.require(len(frames) == len(by_frame) and set(by_frame) == expected, 'Missing/extra/duplicate native model input')
    rate = Fraction(plan['source_geometry']['native_frame_rate'])
    for i, frame in by_frame.items():
        study.require(frame['source_rgb_sha256'] == source_frames[i], 'Native RGB differs from audited source')
        study.require(Fraction(frame['time_base']) * frame['pts'] == Fraction(i, 1) / rate, 'Native PTS/frame correspondence changed')
        study.require(frame['status'] in ('ok', 'not_applicable'), 'Unknown input availability')
        if frame['status'] == 'ok':
            study.require(frame['detected_faces'] == 1 and len(frame['landmarks']) == 5
                and all(len(p) == 2 and all(math.isfinite(v) for v in p) for p in frame['landmarks']), 'No single visible detected face')
            study.require(all(engine.is_hash(frame[k]) for k in ('crop_rgb_sha256', 'patch_crop_rgb_sha256', 'patch_feature_sha256')),
                'Missing source-bound native crop/feature commitment')
    counts = {'face_detection_inputs': len(frames),
        'crop_classifier_inputs': sum('crop_tensor_sha256' in f for f in frames),
        'appearance_inputs': sum('appearance_feature_sha256' in f for f in frames),
        'patch_feature_inputs': sum('patch_feature_sha256' in f for f in frames),
        'native_repeat_inputs': len(measured['native_repeats'])}
    study.require(measured['native_input_counts'] == counts, 'Native source-input/repeat counters disagree with bindings')
    responses = {u['id']: u for u in measured['units']}
    study.require(len(responses) == len(measured['units']) == len(plan['units']), 'Missing/duplicate local native unit')
    changed = set(plan['actual_changed_ai_source_frames'])
    for unit in plan['units']:
        current = responses[unit['id']]
        study.require(unit['label'] == int(bool(set(unit['location']['sampled_frames']) & changed)), 'Copied whole-file/local truth')
        study.require(current['location'] == unit['location'] and current['view'] == unit['view'], 'Native unit source position changed')
        if current['status'] == 'ok':
            study.require(engine.finite(current['raw']), 'Invalid native response')
        else:
            study.require(current['raw'] is None and bool(current['reason']), 'Unavailable unit became a numeric negative')
    return counts


def check_input_roles(frames, role, seen):
    for frame in frames:
        for field in ('crop_tensor_sha256', 'appearance_crop_bgr_sha256', 'patch_crop_rgb_sha256'):
            if field in frame:
                key = field, frame[field]
                previous = seen.setdefault(key, role)
                study.require(previous == role, 'Exactly equal eligible model input crosses source roles; do not rescue by resplitting scored data')


def audit(trial, roles, output):
    _, plan = study.check(trial)
    plans = {r['id']: r for r in plan['files']}
    source_rows = {r['id']: r for r in [json.loads(s) for s in
        (trial / 'source-frame-audit-v1.jsonl').read_text().splitlines()][1:]}
    counts, status, seen = Counter(), Counter(), {}
    processed = 0
    for role in roles:
        for identity, record in study.responses(trial, role).items():
            current = plans[identity]
            study.require(current['role'] == role, 'Native response entered wrong source role')
            counts.update(inspect(current, record, source_rows[identity]['rgb_sha256']))
            status.update((u['view'], u['status']) for u in record['units'])
            check_input_roles(record['frame_inputs'], role, seen)
            processed += 1
    study.require(processed == sum(r['role'] in roles for r in plan['files']), 'Every selected native source case required')
    source.write(output, {'schema': 'chipmunk.video-face-native-binding-review/20', 'roles': roles,
        'implementation_sha256': source.sha(Path(__file__)),
        'audited_source_plan_sha256': source.sha(trial / 'scoring-input-plan-v2.json'),
        'source_rgb_journal_sha256': source.sha(trial / 'source-frame-audit-v1.jsonl'),
        'native_role_receipt_sha256': {role: source.sha(trial / (role + '-v20-native.json')) for role in roles},
        'clips': processed, 'units': sum(status.values()), 'native_input_counts': counts,
        'unit_status': {view: {s: status[view, s] for s in ('ok', 'not_applicable')} for view in study.VIEWS},
        'source_rgb_and_native_clock_exact': True, 'local_truth_and_input_counters_exact': True,
        'exact_eligible_inputs_crossing_roles': 0, 'new_neural_forwards': 0,
        'serving_admitted': False, 'outside_human_reviews': 0,
        'limits': 'Local metadata/commitment review only; no neural re-execution, source authorship verification, new independent identity matching, fresh detection validation or outside human review.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--roles', nargs='+', choices=engine.ROLES, required=True)
    args = parser.parse_args()
    study.require(len(set(args.roles)) == len(args.roles), 'Unique native source roles required')
    audit(args.trial.resolve(), args.roles, args.output.resolve())
