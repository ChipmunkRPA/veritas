"""Arithmetic review of frozen whole-image cue outcomes; no model fitting."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import image_smaug_comparison_v7 as method
from scripts.run_image_smaug_v7 import load_lock


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def paired_interval(rows, label):
    groups = defaultdict(list)
    for row in rows:
        if row['label'] == label:
            groups[row['group']].append(int(row['combined_ai_indication']) - int(row['baseline_ai']))
    keys = sorted(groups)
    values = np.asarray([sum(groups[k]) for k in keys], float)
    weights = np.asarray([len(groups[k]) for k in keys], float)
    rng = np.random.default_rng(20261107)
    selected = rng.integers(0, len(keys), size=(2000, len(keys)))
    scores = values[selected].sum(1) / weights[selected].sum(1)
    return {'planned_inputs': int(weights.sum()), 'source_components': len(keys),
        'paired_rate_difference': float(values.sum() / weights.sum()),
        'component_bootstrap_95_percentile': np.quantile(scores, [.025, .975]).tolist()}


def counts(rows):
    genuine = [x for x in rows if x['label'] == 0]
    ai = [x for x in rows if x['label'] == 1]
    return {'planned': len(rows), 'genuine': len(genuine), 'ai': len(ai),
        'baseline_ai_indications': sum(x['baseline_ai'] for x in ai),
        'candidate_ai_indications': sum(x['candidate_positive_cue'] for x in ai),
        'combined_ai_indications': sum(x['combined_ai_indication'] for x in ai),
        'baseline_false_alarms': sum(x['baseline_ai'] for x in genuine),
        'candidate_false_alarms': sum(x['candidate_positive_cue'] for x in genuine),
        'combined_false_alarms': sum(x['combined_ai_indication'] for x in genuine),
        'baseline_missed_ai_indications': sum(not x['baseline_ai'] for x in ai),
        'combined_missed_ai_indications': sum(not x['combined_ai_indication'] for x in ai),
        'current_false_original': sum(x['baseline_direction'] == 'original' for x in ai),
        'current_directions': dict(Counter(x['baseline_direction'] for x in rows)),
        'current_inconclusive': sum(x['baseline_direction'] == 'inconclusive' for x in rows),
        'candidate_failures': sum(x['candidate_status'] != 'ok' for x in rows),
        'baseline_failures': sum(x['baseline_status'] != 'done' for x in rows),
        'provenance_led': sum(x['provenance_led'] for x in rows),
        'new_conflicts_with_original': sum(x['new_conflict_with_original'] for x in rows),
        'candidate_original_votes': 0}


def review(trial):
    lock, panel = load_lock(trial)
    baseline = read(trial / 'baseline.jsonl')
    candidate = read(trial / 'candidate-calibration.jsonl') + read(trial / 'candidate-development.jsonl')
    gate = json.loads((trial / 'gate.json').read_text())
    if gate != method.select_gate(panel, candidate):
        raise ValueError('Gate differs from genuine-only original responses')
    rows = method.account(panel['items'], baseline, candidate, gate)
    development = [x for x in rows if x['role'] == 'development']
    audit = read(trial / 'candidate-audit.jsonl')
    profiles = {name: json.loads((trial / ('profile-' + name) / 'profile.json').read_text())
                for name in ('baseline', 'candidate-calibration', 'candidate-development', 'candidate-audit')}
    requirements = {'complete_original_model_runs': all(x['child_exit_code'] == 0 and x['guard'] is None for x in profiles.values()),
        'all_candidate_responses_complete': len(candidate) == len(panel['items']) and all(x['status'] == 'ok' for x in candidate),
        'all_baseline_publications_complete': len(baseline) == len(panel['items']) and all(x['status'] == 'done' for x in baseline),
        'native_replay_complete': len(audit) == len({tuple(x[k] for k in ('family', 'label', 'format', 'role')) for x in panel['items']})
            and all(x['status'] == 'ok' and x.get('native_replay_passed') is True for x in audit)}
    by_format, by_family, local_edit_requirements = {}, {}, {}
    for fmt in ('native', 'jpeg75'):
        subset = [x for x in development if x['format'] == fmt]
        table = counts(subset)
        table['ai_gain_interval'] = paired_interval(subset, 1)
        table['false_alarm_change_interval'] = paired_interval(subset, 0)
        by_format[fmt] = table
        requirements[fmt + '_extra_genuine_at_most_one'] = table['combined_false_alarms'] - table['baseline_false_alarms'] <= 1
        added = [x for x in subset if x['label'] == 1 and x['combined_ai_indication'] and not x['baseline_ai']]
        requirements[fmt + '_at_least_four_added_ai_sources'] = len({x['group'] for x in added}) >= 4
        requirements[fmt + '_two_added_generation_families'] = len({x['family'] for x in added if x['scope'] == 'whole_generation'}) >= 2
        edited = [x for x in added if x['scope'] == 'localized_edit']
        local_edit_requirements[fmt] = len({x['group'] for x in edited}) >= 4 and len({x['family'] for x in edited}) >= 2
        for family in sorted({x['family'] for x in subset}):
            by_family[fmt + ':' + family] = counts([x for x in subset if x['family'] == family])
    check_statuses = Counter((r['id'], r['status'], r.get('reason')) for b in baseline for r in b.get('checks', []))
    result = {'schema': 'chipmunk.image-smaug-development-result/7', 'lock_sha256': method.digest(trial / 'lock.json'),
        'input_count': len(panel['items']), 'development_inputs': len(development), 'gate': gate,
        'scope': 'Consumed development only. Union of AI indications is not a replacement calibrated overall conclusion. No candidate original vote, public threshold claim, pixel edit-mask claim or app integration.',
        'requirements': requirements, 'advance_to_separate_fresh_confirmation': all(requirements.values()),
        'local_edit_usefulness_requirements': local_edit_requirements,
        'local_edit_usefulness_established_in_development': all(local_edit_requirements.values()),
        'by_format': by_format, 'by_format_family': by_family,
        'check_statuses_all_planned_baseline_inputs': [{'detector': d, 'status': s, 'reason': r, 'count': n}
            for (d, s, r), n in sorted(check_statuses.items(), key=lambda x: str(x[0]))],
        'native_probes': len(audit), 'native_max_logit_difference': max(x.get('max_logit_difference', 0) for x in audit),
        'native_max_sigmoid_difference': max(x.get('sigmoid_difference', 0) for x in audit),
        'original_journal_sha256': {p.name: method.digest(p) for p in
            [trial / 'baseline.jsonl', trial / 'candidate-calibration.jsonl', trial / 'candidate-development.jsonl', trial / 'candidate-audit.jsonl']},
        'fresh_confirmation_scored': False, 'serving_changed': False, 'outside_human_reviewers': 0}
    method.write(trial / 'summary.json', result)
    method.write(trial / 'paired-accounting.json', rows)
    print(json.dumps({'requirements': requirements, 'by_format': by_format}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    review(parser.parse_args().trial)
