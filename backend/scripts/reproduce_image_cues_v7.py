"""Model-free, standard-library review of a public image-cue response packet.

Checks neural-response reduction, genuine-only gate, all counts and advancement
requirements. It cannot validate source pixels, pretrained inference or calibration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def export(trial, output):
    from scripts.run_image_smaug_v7 import load_lock

    load_lock(trial)
    def read(name):
        return [json.loads(x) for x in (trial / name).read_text().splitlines()]

    panel = json.loads((trial / 'panel.json').read_text())
    baseline = {x['id']: x for x in read('baseline.jsonl')}
    candidate = {x['id']: x for x in read('candidate-calibration.jsonl') + read('candidate-development.jsonl')}
    rows = []
    from app.core.canonical import canonical_hash
    from app.engine.customer_assessment import customer_assessment

    for item in panel['items']:
        b, c = baseline[item['id']], candidate[item['id']]
        if b['input_sha256'] != item['input_sha256'] or c['input_sha256'] != item['input_sha256']:
            raise ValueError('Both arms must be bound to the exact frozen input bytes')
        if b['status'] == 'done':
            retained = trial / 'baseline-results' / (item['id'] + '.json')
            if hashlib.sha256(retained.read_bytes()).hexdigest() != b['retained_packet_sha256']:
                raise ValueError('Retained complete stage/result packet changed')
            source_result = json.loads(retained.read_text())['result']
            if canonical_hash(source_result) != b['canonical_result_sha256']:
                raise ValueError('Retained canonical result differs from original published result hash')
            actual = customer_assessment({**source_result, 'kind': 'image', 'status': 'done'})
            if actual['direction'] != b['interpretation']['customer_direction']:
                raise ValueError('Current frozen interpretation differs from retained canonical result')
        rows.append({**{key: item[key] for key in ('id', 'group', 'label', 'role', 'format', 'scope', 'family', 'input_sha256')},
            'baseline_status': b['status'], 'baseline_direction': b.get('interpretation', {}).get('customer_direction', 'failed'),
            'baseline_statistical_direction': b.get('interpretation', {}).get('statistical_direction', 'inconclusive'),
            'baseline_provenance_led': b.get('interpretation', {}).get('provenance_led', False),
            'candidate_status': c['status'], 'candidate_crop_logits': c.get('crop_logits'),
            'candidate_mean_logit': c.get('mean_logit'), 'candidate_raw_sigmoid_response': c.get('raw_sigmoid_response')})
    native = read('candidate-audit.jsonl')
    from scripts.review_image_smaug_v7 import counts

    original_accounting = json.loads((trial / 'paired-accounting.json').read_text())
    scope_counts = {fmt + ':' + scope: counts([r for r in original_accounting if r['role'] == 'development'
        and r['format'] == fmt and r['scope'] == scope]) for fmt in ('native', 'jpeg75')
        for scope in ('whole_generation', 'localized_edit', 'genuine')}
    profiles = [json.loads((trial / ('profile-' + name) / 'profile.json').read_text())
                for name in ('baseline', 'candidate-calibration', 'candidate-development', 'candidate-audit')]
    write(output, {'schema': 'chipmunk.public-image-cue-arithmetic/7',
        'lock_sha256': hashlib.sha256((trial / 'lock.json').read_bytes()).hexdigest(),
        'panel_sha256': hashlib.sha256((trial / 'panel.json').read_bytes()).hexdigest(),
        'public_panel_file_sha256': hashlib.sha256((Path(__file__).resolve().parents[2] /
            'docs/evaluation-panels/image-v7-smaug-development-panel.json').read_bytes()).hexdigest(),
        'gate': json.loads((trial / 'gate.json').read_text()), 'items': rows,
        'original_journal_sha256': {name: hashlib.sha256((trial / name).read_bytes()).hexdigest() for name in
            ('baseline.jsonl', 'candidate-calibration.jsonl', 'candidate-development.jsonl', 'candidate-audit.jsonl')},
        'run_health': [{'child_exit_code': p['child_exit_code'], 'guard': p['guard']} for p in profiles],
        'native_audit': [{'id': r['id'], 'status': r['status'], 'native_replay_passed': r.get('native_replay_passed', False)} for r in native],
        'expected': json.loads((trial / 'summary.json').read_text()),
        'expected_scope_counts': scope_counts,
        'scope': '224 scalar model-response records and source/label/group identifiers only. No raw images, face vectors, model weights or customer files. Model-free arithmetic review is not native model replay or outside human review.'})


def tally(rows):
    """Separate loop-based accounting for family/scope tables, without ML packages."""
    out = {name: 0 for name in ('planned', 'ai', 'genuine', 'baseline_ai_indications', 'candidate_ai_indications',
        'combined_ai_indications', 'baseline_false_alarms', 'candidate_false_alarms', 'combined_false_alarms',
        'baseline_missed_ai_indications', 'combined_missed_ai_indications', 'current_false_original',
        'current_inconclusive', 'candidate_failures', 'baseline_failures', 'provenance_led',
        'new_conflicts_with_original', 'candidate_original_votes')}
    directions = Counter()
    for row in rows:
        out['planned'] += 1
        directions[row['baseline_direction']] += 1
        if row['label'] == 1:
            out['ai'] += 1
            out['baseline_ai_indications'] += int(row['old'])
            out['candidate_ai_indications'] += int(row['cue'])
            out['combined_ai_indications'] += int(row['both'])
            out['baseline_missed_ai_indications'] += int(not row['old'])
            out['combined_missed_ai_indications'] += int(not row['both'])
            out['current_false_original'] += int(row['baseline_direction'] == 'original')
        else:
            out['genuine'] += 1
            out['baseline_false_alarms'] += int(row['old'])
            out['candidate_false_alarms'] += int(row['cue'])
            out['combined_false_alarms'] += int(row['both'])
        out['current_inconclusive'] += int(row['baseline_direction'] == 'inconclusive')
        out['candidate_failures'] += int(row['candidate_status'] != 'ok')
        out['baseline_failures'] += int(row['baseline_status'] != 'done')
        out['provenance_led'] += int(row['baseline_provenance_led'])
        out['new_conflicts_with_original'] += int(row['cue'] and row['baseline_statistical_direction'] == 'original')
    out['current_directions'] = dict(directions)
    return out


def review(packet, public_panel):
    rows = packet['items']
    if len(rows) != 224 or len({r['id'] for r in rows}) != 224:
        raise ValueError('Exact unique planned input count required')
    if public_panel['private_panel_sha256'] != packet['panel_sha256']:
        raise ValueError('Public source panel does not bind the private source freeze')
    expected_sources = {x['id']: x for x in public_panel['record']['items']}
    if set(expected_sources) != {r['id'] for r in rows}:
        raise ValueError('Planned source IDs differ from original public freeze')
    for row in rows:
        if any(row[k] != expected_sources[row['id']][k] for k in
               ('group', 'label', 'role', 'format', 'scope', 'family', 'input_sha256')):
            raise ValueError('Input bytes, source group, truth label or role differs from original public freeze')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['group']].append(row)
        if row['candidate_status'] == 'ok':
            values = row['candidate_crop_logits']
            mean = row['candidate_mean_logit']
            if len(values) != 3 or not all(math.isfinite(x) for x in [*values, mean]):
                raise ValueError('Three finite crop logits and mean required')
            if abs(math.fsum(values) / 3 - mean) > 1e-4:
                raise ValueError('Neural outer logit reduction mismatch')
            sigmoid = math.exp(-math.log1p(math.exp(-mean))) if mean >= 0 else math.exp(mean) / (1 + math.exp(mean))
            if abs(sigmoid - row['candidate_raw_sigmoid_response']) > 1e-5:
                raise ValueError('Raw sigmoid reduction mismatch')
    if any(len({r['role'] for r in group}) != 1 for group in grouped.values()):
        raise ValueError('Source component crosses roles')
    cal = [r for r in rows if r['role'] == 'calibration']
    if len(cal) != 48 or len({r['group'] for r in cal}) != 24 or any(r['label'] != 0 or r['candidate_status'] != 'ok' for r in cal):
        raise ValueError('Genuine-only complete operating-point set required')
    gate = max(r['candidate_mean_logit'] for r in cal)
    if gate != packet['gate']['raw_mean_logit']:
        raise ValueError('Recorded point differs from genuine-only maximum')
    for r in rows:
        r['old'] = r['baseline_status'] == 'done' and r['baseline_statistical_direction'] == 'ai'
        r['cue'] = r['candidate_status'] == 'ok' and r['candidate_mean_logit'] > gate
        r['both'] = r['old'] or r['cue']
    requirements = {
        'complete_original_model_runs': len(packet['run_health']) == 4 and all(p['child_exit_code'] == 0 and p['guard'] is None for p in packet['run_health']),
        'all_candidate_responses_complete': all(r['candidate_status'] == 'ok' for r in rows),
        'all_baseline_publications_complete': all(r['baseline_status'] == 'done' for r in rows),
        'native_replay_complete': len(packet['native_audit']) == len({tuple(r[k] for k in ('family', 'label', 'format', 'role')) for r in rows})
            and all(r['status'] == 'ok' and r['native_replay_passed'] for r in packet['native_audit'])}
    comparisons = 0
    for fmt in ('native', 'jpeg75'):
        part = [r for r in rows if r['role'] == 'development' and r['format'] == fmt]
        ai, genuine = [r for r in part if r['label'] == 1], [r for r in part if r['label'] == 0]
        values = {'planned': len(part), 'ai': len(ai), 'genuine': len(genuine),
            'baseline_ai_indications': sum(r['old'] for r in ai), 'candidate_ai_indications': sum(r['cue'] for r in ai),
            'combined_ai_indications': sum(r['both'] for r in ai),
            'baseline_false_alarms': sum(r['old'] for r in genuine), 'candidate_false_alarms': sum(r['cue'] for r in genuine),
            'combined_false_alarms': sum(r['both'] for r in genuine),
            'baseline_missed_ai_indications': sum(not r['old'] for r in ai), 'combined_missed_ai_indications': sum(not r['both'] for r in ai),
            'current_false_original': sum(r['baseline_direction'] == 'original' for r in ai),
            'current_directions': dict(Counter(r['baseline_direction'] for r in part)),
            'current_inconclusive': sum(r['baseline_direction'] == 'inconclusive' for r in part),
            'candidate_failures': sum(r['candidate_status'] != 'ok' for r in part), 'baseline_failures': sum(r['baseline_status'] != 'done' for r in part),
            'provenance_led': sum(r['baseline_provenance_led'] for r in part),
            'new_conflicts_with_original': sum(r['cue'] and r['baseline_statistical_direction'] == 'original' for r in part),
            'candidate_original_votes': 0}
        expected = packet['expected']['by_format'][fmt]
        if any(expected[key] != value for key, value in values.items()):
            raise ValueError('Independent development accounting differs')
        comparisons += len(values)
        if values != tally(part):
            raise ValueError('Independent loop and comprehension tallies differ')
        for family in sorted({r['family'] for r in part}):
            actual = tally([r for r in part if r['family'] == family])
            if actual != packet['expected']['by_format_family'][fmt + ':' + family]:
                raise ValueError('Independent family accounting differs')
            comparisons += len(actual)
        for scope in ('whole_generation', 'localized_edit', 'genuine'):
            actual = tally([r for r in part if r['scope'] == scope])
            if actual != packet['expected_scope_counts'][fmt + ':' + scope]:
                raise ValueError('Independent scope accounting differs')
            comparisons += len(actual)
        added = [r for r in ai if r['both'] and not r['old']]
        requirements[fmt + '_extra_genuine_at_most_one'] = values['combined_false_alarms'] - values['baseline_false_alarms'] <= 1
        requirements[fmt + '_at_least_four_added_ai_sources'] = len({r['group'] for r in added}) >= 4
        requirements[fmt + '_two_added_generation_families'] = len({r['family'] for r in added if r['scope'] == 'whole_generation'}) >= 2
        edited = [r for r in added if r['scope'] == 'localized_edit']
        local = len({r['group'] for r in edited}) >= 4 and len({r['family'] for r in edited}) >= 2
        if local != packet['expected']['local_edit_usefulness_requirements'][fmt]:
            raise ValueError('Independent local-edit gate differs')
    if requirements != packet['expected']['requirements'] or all(requirements.values()) != packet['expected']['advance_to_separate_fresh_confirmation']:
        raise ValueError('Independent advancement requirements differ')
    return {'passed': True, 'inputs': len(rows), 'genuine_gate_exact': True, 'counts_compared': comparisons,
        'all_planned_source_byte_label_group_role_bindings_checked': True,
        'requirements_compared': len(requirements), 'raw_three_logit_and_sigmoid_reductions_checked': sum(r['candidate_status'] == 'ok' for r in rows),
        'paired_bootstrap_intervals_recomputed': False, 'native_neural_inference_recomputed': False,
        'source_pixels_recomputed': False, 'outside_human_review': False,
        'scope': 'Separate standard-library arithmetic implementation. Does not reproduce interval bootstraps, source pixels, pretrained model forwards or native audit claims.'}


def tamper(packet, panel):
    review(copy.deepcopy(packet), panel)
    results = []
    for name in ('missing', 'duplicate', 'unknown_id', 'role', 'truth', 'input_hash', 'nonfinite',
                 'crop_mean', 'sigmoid', 'relaxed_gate', 'failed_calibration', 'failed_development',
                 'changed_direction', 'forced_advance'):
        value = copy.deepcopy(packet)
        row = value['items'][0]
        if name == 'missing':
            value['items'].pop()
        elif name == 'duplicate':
            value['items'][1] = copy.deepcopy(row)
        elif name == 'unknown_id':
            row['id'] += '-unexpected'
        elif name == 'role':
            row['role'] = 'development' if row['role'] == 'calibration' else 'calibration'
        elif name == 'truth':
            row['label'] = 1 - row['label']
        elif name == 'input_hash':
            row['input_sha256'] = '0' * 64
        elif name == 'nonfinite':
            row['candidate_mean_logit'] = float('nan')
        elif name == 'crop_mean':
            row['candidate_crop_logits'][0] += 3
        elif name == 'sigmoid':
            row['candidate_raw_sigmoid_response'] += .01
        elif name == 'relaxed_gate':
            value['gate']['raw_mean_logit'] -= 1
        elif name == 'failed_calibration':
            next(r for r in value['items'] if r['role'] == 'calibration')['candidate_status'] = 'failed'
        elif name == 'failed_development':
            next(r for r in value['items'] if r['role'] == 'development')['candidate_status'] = 'failed'
        elif name == 'changed_direction':
            row = next(r for r in value['items'] if r['role'] == 'development')
            row['baseline_direction'] = 'ai' if row['baseline_direction'] != 'ai' else 'original'
        elif name == 'forced_advance':
            value['expected']['advance_to_separate_fresh_confirmation'] = not value['expected']['advance_to_separate_fresh_confirmation']
        try:
            review(value, panel)
        except (ValueError, KeyError, TypeError) as error:
            results.append({'mutation': name, 'rejected': True, 'reason': str(error)})
        else:
            results.append({'mutation': name, 'rejected': False})
    return {'passed': all(r['rejected'] for r in results), 'mutations': results,
            'scope': 'Deliberate changes to the actual public scalar response packet; no pretrained inference or source pixels are tested.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('export', 'review', 'tamper'))
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--panel', type=Path)
    args = parser.parse_args()
    if args.mode == 'export':
        export(args.input, args.output)
    else:
        if args.panel is None:
            parser.error('--panel is required for review')
        packet = json.loads(args.input.read_text())
        if hashlib.sha256(args.panel.read_bytes()).hexdigest() != packet['public_panel_file_sha256']:
            raise ValueError('Actual public source panel bytes changed')
        panel = json.loads(args.panel.read_text())
        value = review(packet, panel) if args.mode == 'review' else tamper(packet, panel)
        value['packet_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        write(args.output, value)
        print(json.dumps(value), flush=True)
        if not value['passed']:
            raise ValueError('Public response-packet review failed')
