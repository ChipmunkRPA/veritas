"""Review actual local source labels and interval errors after all frozen roles.

No model, fitting, threshold change or serving admission. The independent label
checker is separate from the rendering helper. Retained scores are replayed only
after every role's native receipt and the frozen artifact hashes are verified.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from app.bench import specialists as engine
from scripts import audit_audio_specialists_v15 as source_audit
from scripts import run_audio_specialists_v15 as frozen


def local_metrics(rows):
    losses = {0: defaultdict(list), 1: defaultdict(list)}
    count = {'planned': len(rows), 'genuine': 0, 'ai': 0, 'assessed': 0, 'inconclusive': 0,
        'ties': 0, 'false_ai': 0, 'ai_not_leaning_including_ties_and_inconclusive': 0}
    for row in rows:
        label, probability = row['label'], row['probability']
        count['ai' if label else 'genuine'] += 1
        assessed = probability is not None
        count['assessed' if assessed else 'inconclusive'] += 1
        leaning = assessed and probability > .5 + 1e-12
        if assessed:
            count['ties'] += abs(probability - .5) <= 1e-12
            losses[label][row['group']].append((probability - label)**2)
        if label:
            count['ai_not_leaning_including_ties_and_inconclusive'] += not leaning
        else:
            count['false_ai'] += leaning
    by_class = [sum(sum(v) / len(v) for v in groups.values()) / len(groups) if groups else None
        for groups in losses.values()]
    count['source_class_balanced_brier_on_scored_units'] = sum(by_class) / 2 if all(v is not None for v in by_class) else None
    return count


def bind_local(document, result, planned, *, contextual):
    """Verify each actual unit's clock, scope and local truth without renderer."""
    files = {file['id']: file for file in document['files']}
    units, bound = {}, []
    for file in document['files']:
        original = planned[file['id']]
        source_audit.require(file['group'] == original['group'] and file['role'] == original['role']
            and file['input_sha256'] == original['input_sha256'], 'Observation is not its planned source input')
        source_audit.require(file['label'] == int(any(original['labels'])), 'File union truth changed')
        for unit in file['units']:
            view, offset = unit['id'].split('.')
            offset = int(offset)
            window = next(w for w in original['windows'] if w['start'] == offset)
            expected = {'kind': 'samples', 'unit': 'samples', 'start': offset, 'end': offset + 64000,
                'extent': 128000, 'sample_rate': 16000}
            source_audit.require(unit['location'] == expected and unit['specialist'] == 'audio.' + view,
                'Local inference clock/view changed')
            label = source_audit.local_label(original['labels'], offset, offset + 64000)
            context = ('diffuse' if window['quality']['flatness'] >= .65 else 'concentrated') if contextual else 'pooled'
            source_audit.require(unit['label'] == label and unit['context'] == context, 'Local label/routing differs from source')
            if not window['speech_eligible']:
                source_audit.require(unit['status'] == 'not_applicable' and unit['raw'] is None,
                    'Rejected speech window received a cached/invented response')
            units[file['id'], unit['id']] = unit
    seen = set()
    for row in result['unit_outcomes']:
        key = row['file_id'], row['unit_id']
        source_audit.require(key not in seen and key in units, 'Repeated or unplanned confirmation unit')
        seen.add(key)
        unit, file = units[key], files[row['file_id']]
        source_audit.require(file['role'] == 'confirmation' and row['location'] == unit['location'], 'Local result moved from its source location')
        if unit['status'] != 'ok':
            source_audit.require(row['probability'] is None, 'Unassessed local unit became a probability')
        original = planned[file['id']]
        bound.append({**row, 'label': unit['label'], 'group': file['group'],
            'condition': original['condition'], 'pattern': original['pattern'], 'generator': original['generator']})
    expected_count = sum(len(file['units']) for file in document['files'] if file['role'] == 'confirmation')
    source_audit.require(len(seen) == expected_count, 'Missing confirmation local unit')
    return bound


def review(trial, output):
    _, plan, rows = frozen.check(trial)
    metadata_check = source_audit.verify(plan, rows)
    complete = json.loads((trial / 'fit-complete.json').read_text())
    pipeline = json.loads((trial / 'pipeline-state.json').read_text())
    source_audit.require(pipeline['status'] == 'completed', 'Wait for the exact sequential pipeline; do not review partial inference')
    # responses() verifies actual terminal native receipts, journal bytes and all
    # recorded failures, not merely an output filename or a stale process lock.
    for role in engine.ROLES:
        frozen.responses(trial, role)
        profile_dir = 'specialist-cal-profile' if role == 'specialist_cal' else role + '-profile'
        profile = json.loads((trial / profile_dir / 'profile.json').read_text())
        source_audit.require(profile['child_exit_code'] == 0 and profile['guard'] is None,
            'A failed/guarded attempt cannot be omitted from the review')
    planned = {row['id']: row for row in rows}
    report = {'schema': 'chipmunk.audio-specialists-local-source-review/15', 'freeze_sha256': source_audit.sha(trial / 'freeze.json'),
        'implementation_sha256': source_audit.sha(Path(__file__)), 'metadata_audit': metadata_check,
        'all_four_native_role_receipts_verified': True, 'arms': {}, 'new_model_forwards': 0, 'serving_admitted': False}
    for arm, (contextual, _) in frozen.ARMS.items():
        for suffix, key in (('artifact', 'artifacts'), ('operating-point', 'operating_points')):
            source_audit.require(source_audit.sha(trial / (arm + '-' + suffix + '.json')) == complete[key][arm],
                'A fitted arm/point changed after confirmation was unlocked')
        document = frozen.manifest(trial, arm, True)
        artifact = json.loads((trial / (arm + '-artifact.json')).read_text())
        result = engine.evaluate(document, artifact)
        expected = json.loads((trial / (arm + '-confirmation.json')).read_text())
        source_audit.require(result == expected, 'Retained confirmation differs from exact native-scalar replay')
        local = bind_local(document, result, planned, contextual=contextual)
        grouped = {}
        for field in ('condition', 'pattern', 'context', 'specialist', 'generator'):
            grouped[field] = {value: local_metrics([row for row in local if row[field] == value])
                for value in sorted({row[field] for row in local})}
        report['arms'][arm] = {'local': local_metrics(local), 'strata': grouped,
            'source_windows_s': {f'{offset / 16000:g}-{(offset + 64000) / 16000:g}':
                local_metrics([row for row in local if row['location']['start'] == offset]) for offset in (0, 32000, 64000)},
            'head_support': {key: {k: value[k] for k in ('status', 'source_groups_by_class')} for key, value in artifact['specialists'].items()}}
        for value in report['arms'][arm]['local'].values():
            source_audit.require(value is None or not isinstance(value, float) or math.isfinite(value), 'Nonfinite local metric')
    report['limits'] = ('Local tendencies use the internal0.5 readout under a50:50 reference prior; no customer threshold or accuracy guarantee. '
        'Overlapping windows, full/band views and reused nuisance parents are correlated, not independent trials. '
        'This review checks retained source coordinates/truth commitments and scalar replay, not independent waveforms or publisher authenticity.')
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'schema': report['schema'], 'arms': len(report['arms']),
        'all_four_native_role_receipts_verified': True, 'new_model_forwards': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    review(args.trial, args.output)
