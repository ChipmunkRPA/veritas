"""Independent metadata audit of the frozen audio pilot; no model forwards.

This checks coordinates, source assignment and local labels without calling the
preparation/scoring helpers. It does not independently verify waveform bytes,
speech detection or the publisher's authenticity labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROLES = ('specialist_cal', 'fusion_fit', 'fusion_cal', 'confirmation')
PATTERNS = {'human': (0, 0), 'ai': (1, 1), 'human-ai': (0, 1), 'ai-human': (1, 0)}
CONDITIONS = {'clean': (None, None), **{f'{kind}_{snr}': (kind, snr)
    for kind in ('noise', 'music', 'speech') for snr in (-5, 5)}}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def local_label(labels, start, end):
    """Half-open sample intervals; touching a boundary is not overlap."""
    require(len(labels) == 2 and all(type(v) is int and v in (0, 1) for v in labels), 'Invalid source labels')
    require(type(start) is int and type(end) is int and 0 <= start < end <= 128000, 'Invalid local sample coordinates')
    # Written separately from the rendering helper's interval iteration.
    return int((labels[0] == 1 and start < 64000) or (labels[1] == 1 and end > 64000))


def verify(plan, rows):
    pairs = {p['id']: p for p in plan['pairs']}
    require(len(pairs) == len(plan['pairs']), 'Repeated foreground pair')
    backgrounds = {p['sha256']: p for p in plan['backgrounds']}
    require(len(backgrounds) == len(plan['backgrounds']), 'Repeated background bytes')
    ancestry = {}

    def assign(kind, value, role):
        if value:
            require(ancestry.setdefault((kind, value), role) == role, 'Known ancestry crosses roles')

    for pair in pairs.values():
        require(pair['role'] in ROLES, 'Unknown role')
        require([p['label'] for p in pair['foregrounds']] == [0, 1], 'Ordered genuine/AI foregrounds required')
        for group in pair['original_groups']:
            assign('source_group', group, pair['role'])
        for source in pair['foregrounds']:
            # Source rows preserve their historical development role. The new
            # four-role assignment belongs to the connected foreground pair.
            require(source['role'] in ('fit', 'cal', 'development'), 'An unused source reserve entered the pilot')
            require(source['source_interval_samples'] == [0, 64000], 'Unexpected selected source interval')
            for key in ('sha256', 'canonical_speaker', 'transcript_sha256'):
                assign(key, source.get(key), pair['role'])
    for bg in backgrounds.values():
        require(bg['role'] in ROLES and bg['kind'] in ('noise', 'music', 'speech'), 'Unknown background role/kind')
        for key in ('sha256', 'canonical_speaker'):
            assign(key, bg.get(key), bg['role'])

    ids, cells, gains, nuisance, window_roles = set(), defaultdict(set), {}, {}, {}
    counts, support = Counter(), defaultdict(set)
    for row in rows:
        require(row['id'] not in ids, 'Repeated file identity')
        ids.add(row['id'])
        pair = pairs[row['group']]
        require(row['role'] == pair['role'] and row['generator'] == pair['generator'], 'File source/role assignment changed')
        require(row['pattern'] in PATTERNS and tuple(row['labels']) == PATTERNS[row['pattern']], 'Pattern truth differs from selected foregrounds')
        require(row['condition'] in CONDITIONS, 'Unknown nuisance condition')
        cell = row['pattern'], row['condition']
        require(cell not in cells[row['group']], 'Repeated recipe cannot replace missing recipe')
        cells[row['group']].add(cell)
        require(row['foreground_sha256'] == [pair['foregrounds'][label]['sha256'] for label in row['labels']], 'Known local foreground bytes changed')
        require(row['sample_rate'] == 16000 and row['sample_count'] == 128000, 'File sample clock changed')
        gain = row['common_peak_gain']
        require(math.isfinite(gain) and 0 < gain <= 1, 'Invalid common peak-safe gain')
        require(gains.setdefault(row['group'], gain) == gain, 'Label/condition-specific gain')
        kind, snr = CONDITIONS[row['condition']]
        bg = row['background']
        if kind is None:
            require(bg is None, 'Clean condition contains declared added background')
        else:
            require(isinstance(bg, dict) and bg['sha256'] in backgrounds, 'Unknown background parent')
            parent = backgrounds[bg['sha256']]
            require(parent['role'] == row['role'] and parent['kind'] == kind, 'Background crosses roles/kinds')
            require(bg['kind'] == kind and bg['snr_db'] == snr, 'Background recipe differs from condition')
            require(type(bg['start_sample']) is int and bg['start_sample'] >= 0
                and bg['end_sample'] - bg['start_sample'] == 64000
                and bg['end_sample'] <= parent['full_samples'], 'Background interval outside parent')
            fixed = {k: v for k, v in bg.items() if k != 'snr_db'}
            require(nuisance.setdefault((row['group'], kind), fixed) == fixed, 'Background or offset selected using class/SNR')
        require([(w['start'], w['end']) for w in row['windows']] == [(0, 64000), (32000, 96000), (64000, 128000)], 'Planned windows missing, repeated or shifted')
        for window in row['windows']:
            require(window['extent'] == 128000 and window['sample_rate'] == 16000, 'Local source clock differs')
            require(window['label'] == local_label(row['labels'], window['start'], window['end']), 'Local truth copied from file or wrong boundary')
            quality = window['quality']['flatness']
            require(math.isfinite(quality) and 0 <= quality <= 1, 'Invalid measured context feature')
            require(window['context'] == ('diffuse' if quality >= .65 else 'concentrated'), 'Context uses something other than declared original feature')
            require(type(window['speech_eligible']) is bool, 'Unsupported speech must be explicit')
            for view in ('full', 'band'):
                value = window[view + '_pcm_sha256']
                require(window_roles.setdefault(value, row['role']) == row['role'], 'Derived window bytes cross roles')
            counts[row['role'] + ('/speech_eligible' if window['speech_eligible'] else '/speech_unassessed')] += 1
            if window['speech_eligible']:
                support[row['role'], window['context'], window['label']].add(row['group'])
        counts[row['role'] + '/files'] += 1
        counts[row['role'] + '/' + row['condition'] + '/' + row['pattern']] += 1
    expected = {(pattern, condition) for pattern in PATTERNS for condition in CONDITIONS}
    require(set(cells) == set(pairs) and all(value == expected for value in cells.values()), 'Missing planned pair/pattern/nuisance denominator')
    return {'files': len(rows), 'windows': sum(len(row['windows']) for row in rows), 'foreground_pairs': len(pairs),
        'background_originals': len(backgrounds), 'counts': dict(sorted(counts.items())),
        'eligible_local_support_groups': {'/'.join(map(str, k)): len(v) for k, v in sorted(support.items())},
        'source_assignment_and_local_coordinates_passed': True,
        'waveform_bytes_independently_verified': False, 'speech_gate_independently_verified': False,
        'publisher_labels_independently_established': False, 'new_model_forwards': 0,
        'limits': 'Known source/speaker/transcript ancestry only; no independent-person, background-population or unseen-model-training claim.'}


def audit(trial, output):
    plan_path, signal_path = trial / 'source-plan.json', trial / 'signal-plan.jsonl'
    freeze = json.loads((trial / 'freeze.json').read_text())
    require(freeze['source_plan_sha256'] == sha(plan_path) and freeze['signal_plan_sha256'] == sha(signal_path), 'Frozen metadata changed')
    with signal_path.open() as stream:
        header = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    require(header['source_plan_sha256'] == sha(plan_path) and header['authenticity_classifier_forwards'] == 0, 'Expected frozen zero-forward source-plan declaration')
    require(len(rows) == 4004 and len(json.loads(plan_path.read_text())['pairs']) == 143, 'Incomplete declared study denominator')
    receipt = {'schema': 'chipmunk.audio-specialists-independent-metadata-audit/15',
        'freeze_sha256': sha(trial / 'freeze.json'), 'audit_implementation_sha256': sha(Path(__file__)),
        **verify(json.loads(plan_path.read_text()), rows)}
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: receipt[k] for k in ('schema', 'files', 'windows', 'foreground_pairs', 'background_originals', 'new_model_forwards')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.trial, args.output)
