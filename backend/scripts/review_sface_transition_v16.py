"""Fixed development decision for source-bound SFace transition localization."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts import sface_transition_v16 as method
from scripts.run_sface_transition_v16 import configure, validate


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def interval(cases, field, label):
    grouped = defaultdict(list)
    for row in cases:
        if row['label'] == label:
            if field == 'iou':
                delta = row['candidate_iou'] - row['current_iou']
            elif field == 'localized':
                delta = int(row['candidate_localized']) - int(row['current_localized'])
            else:
                delta = int(row['candidate_range_s'] is not None) - int(row['current_range_s'] is not None)
            grouped[row['group']].append(delta)
    means = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], np.float64)
    sample = np.random.default_rng(20261107).choice(means, (5000, len(means)), replace=True).mean(1)
    return {'source_groups': len(means), 'mean_paired_difference': float(means.mean()),
        'source_group_bootstrap_95_percentile': np.quantile(sample, [.025, .975]).tolist(), 'replicates': 5000, 'seed': 20261107}


def review(main, trial):
    configure(main, trial)
    lock, panel = validate(main, trial)
    from app.detectors.video.face_sequence import REVIEW_POINT

    calibration, development = read(trial / 'calibration.jsonl'), read(trial / 'development.jsonl')
    point = json.loads((trial / 'operating-point.json').read_text())
    if point != method.operating_point(panel['items'], calibration, REVIEW_POINT):
        raise ValueError('Genuine-only original operating point changed')
    planned = [r for r in panel['items'] if r['role'] == 'development']
    cases = method.outcomes(planned, development, point, REVIEW_POINT)
    if len(cases) != 120 or len({r['group'] for r in cases}) != 30:
        raise ValueError('Exact planned development denominator required')
    all_records = [*calibration, *development]
    for row in all_records:
        item = next(r for r in panel['items'] if r['id'] == row['id'])
        if row['source_sha256'] != item['sha256']:
            raise ValueError('Inference outcome lost frozen source binding')
        if row['status'] != 'failed':
            private = trial / ('private-features-' + item['role']) / (item['id'] + '.json')
            if method.digest(private) != row['private_feature_packet_sha256']:
                raise ValueError('Private real-face/source packet changed')
    profiles = {role: json.loads((trial / ('profile-' + role) / 'profile.json').read_text()) for role in ('calibration', 'development')}
    old, new = method.metrics(cases, 'current'), method.metrics(cases, 'candidate')
    families = {family: {arm: method.metrics([r for r in cases if r['family'] == family], arm)
        for arm in ('current', 'candidate')} for family in sorted({r['family'] for r in cases})}
    requirements = {
        'all_source_cases_accounted': len(calibration) == 60 and len(development) == 120,
        'no_candidate_failure': all(r['status'] in {'ok', 'not_applicable'} for r in all_records),
        'all_original_runs_complete_without_guard': all(p['child_exit_code'] == 0 and p['guard'] is None for p in profiles.values()),
        'all_current_consecutive_replays_match': all(r.get('baseline_parity_passed') for r in all_records),
        'all_actual_native_alignments_inputs_features_match': all(r.get('native_alignment_all_exact') for r in all_records),
        'more_correct_edit_localizations': new['localized_edits'] > old['localized_edits'],
        'genuine_total_flags_at_most_current_plus_one': new['genuine_review_flags'] <= old['genuine_review_flags'] + 1,
        'ordinary_edit_flags_at_most_current_plus_one': new['ordinary_edit_flags'] <= old['ordinary_edit_flags'] + 1,
        'mean_edit_iou_not_lower': new['mean_edit_iou'] >= old['mean_edit_iou'],
        'no_swap_family_localization_loss': all(v['candidate']['localized_edits'] >= v['current']['localized_edits']
            for name, v in families.items() if name in {'inswapper', 'uniface'})}
    count_changes = {'new_correct_locations': sum(r['label'] and r['candidate_localized'] and not r['current_localized'] for r in cases),
        'lost_correct_locations': sum(r['label'] and r['current_localized'] and not r['candidate_localized'] for r in cases),
        'new_genuine_flags': sum(not r['label'] and r['current_range_s'] is None and r['candidate_range_s'] is not None for r in cases),
        'removed_genuine_flags': sum(not r['label'] and r['candidate_range_s'] is None and r['current_range_s'] is not None for r in cases)}
    result = {'schema': 'chipmunk.sface-transition-development-result/16',
        'lock_sha256': method.digest(trial / 'lock.json'), 'point': point, 'current': old, 'candidate': new,
        'families': families, 'case_changes': count_changes, 'requirements': requirements,
        'advance_to_separately_frozen_fresh_confirmation': all(requirements.values()),
        'paired_iou': interval(cases, 'iou', 1), 'paired_localization_rate': interval(cases, 'localized', 1),
        'paired_genuine_flag_rate': interval(cases, 'flags', 0),
        'private_journal_sha256': {name: method.digest(trial / name) for name in ('calibration.jsonl', 'development.jsonl')},
        'fresh_confirmation_scored': False, 'serving_changed': False, 'outside_human_reviewers': 0,
        'scope': 'Consumed source-bound bounded-transition review cues, not AI authenticity/identity probabilities. Frozen legacy fallback/track receipts shared; current consecutive model recomputed with parity; not full current-pipeline reevaluation. Real face vectors remain private.'}
    method.write(trial / 'summary.json', result)
    method.write(trial / 'cases.json', cases)
    print(json.dumps({'current': old, 'candidate': new, 'requirements': requirements}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    review(args.main, args.trial)
