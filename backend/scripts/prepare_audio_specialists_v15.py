"""Prepare known-boundary noisy-voice composites before authenticity scoring."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.bench.specialists import ROLES

SR, HALF, EXTENT = 16000, 64000, 128000
SOURCE_SHA = 'f3d7a7f2e20a9d2aa9092c371f5af4d2a05ed9172e4652677049a8286f61605f'
ASSETS = {'pellav2_detector.pt': 'd21aedfb52e2f87706c2753715a29665482b0f4a034800c31579df5d458559ac',
    'w2v-bert-config.json': 'f5572bd5998b68182e9c328a43127ed21fed687f6910497136b91a4e3b0e3675',
    'silero_vad.jit': 'e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720'}
PATTERNS = {'human': (0, 0), 'ai': (1, 1), 'human-ai': (0, 1), 'ai-human': (1, 0)}
CONDITIONS = [('clean', None, None), *[(f'{kind}_{snr}', kind, snr)
    for kind in ('noise', 'music', 'speech') for snr in (-5, 5)]]
METHOD = {'schema': 'chipmunk.audio-specialists-method/15',
    'source_groups': 143, 'sample_rate': SR, 'source_prefix_samples': HALF, 'composite_samples': EXTENT,
    'source_selection': 'All143 consumed AI-containing connected groups; hash-first AI source, same-group human if available, otherwise one unused hash-ordered genuine-only group. No authenticity score selection.',
    'roles': list(ROLES), 'roles_rule': 'Stratify by selected AI generator; salted hash order and globally continuing round-robin roles. All originals, speaker/transcript ancestry and backgrounds remain role-disjoint.',
    'backgrounds': 'All48 consumed licensed originals, disjoint canonical identities; hash-partition each kind into four roles; pair-hash choice and deterministic four-second offset within its role. Reused nuisance parents within a role are disclosed.',
    'patterns': {name: list(labels) for name, labels in PATTERNS.items()}, 'conditions': [list(row) for row in CONDITIONS],
    'render': 'Normalize both foreground prefixes separately to RMS0.1; add the same selected human/noise/music background to the second half at -5 or5dB; one peak-safe gain shared by all patterns/conditions of a pair; canonical mono16kHz s16 WAV roundtrip.',
    'source_truth': 'Any overlap of a planned4s window with a known AI foreground interval. Background speech is genuine. Ordinary edits/looping, noise and speaker changes alone are not AI truth.',
    'windows': 'Full-profile production0,2,4s windows on8s input; original full-clip Silero gate; every planned window retained, including gated-out evidence.',
    'band': 'Whole-clip zero-phase Butterworth100–3800Hz/order4; same source windows/gate; never treated as a second truth or a recovered original.',
    'routing': 'Original-window measured normalized spectral-flatness>=0.65: diffuse; otherwise concentrated. Generator, known SNR, recipe, label and model response are excluded from routing.',
    'context_cut': .65, 'contexts': ['concentrated', 'diffuse'], 'minimum_groups_per_class': 25,
    'calibration': 'Four source-disjoint roles; full-view pooled/context and full+band pooled/context arms. One dependence family; maximum positive calibrated logit, separately fitted/calibrated file result. L2=.001;50:50 reference prior.',
    'operating_comparison': 'Internal AI-only point on fusion_cal genuine foreground-pair maxima: strict greater than nearest-rank ceil((n+1)*.9), capped at n. All arm artifacts/points frozen before confirmation authenticity inference. No customer threshold or error guarantee.',
    'quality_gates': 'Against pooled full-view research reference: more AI files found, no additional false-AI foreground groups, no coverage loss, lower source/class-balanced Brier. Per-condition/local errors and all inconclusive denominators retained; no automatic admission.',
    'native_audit': 'Exact publisher assets; retained exact old WAV/PCM/raw parity before new authenticity inference; first window per role/view/context direct repeat; source/native coordinates audited separately.',
    'device': 'cpu', 'dtype': 'float32', 'threads': 4,
    'resource_guard': {'tree_rss_gib': 6, 'minimum_available_mib': 1536, 'minimum_disk_mib': 1536, 'maximum_seconds_per_role': 3600},
    'serving_admitted': False,
    'limits': 'Consumed foregrounds/backgrounds, unknown external model-training overlap and artificial splice boundaries.35/36 foreground-pair groups per role are not independent background recordings; each role has6 noise/3music/3speech parents. No background-authenticity, unseen-customer, fresh-independent or court claim.'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pcm_sha(values):
    return hashlib.sha256(np.asarray(values, '<f4').tobytes()).hexdigest()


def stable(value):
    return hashlib.sha256(('chipmunk-audio-specialists-v15\0' + value).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_plan(main, trial):
    path = main / 'backend/data/independent-evaluation/audio-v9/source-plan.json'
    if sha(path) != SOURCE_SHA:
        raise ValueError('Exact identity-audited consumed source plan required')
    prior = json.loads(path.read_text())
    sources = [r for r in prior['sources'] if r['role'] != 'confirmation']
    backgrounds = [r for r in prior['backgrounds'] if r['role'] != 'confirmation']
    grouped = defaultdict(list)
    for row in sources:
        grouped[row['group']].append(row)
    ai_groups = [g for g, rows in grouped.items() if any(r['label'] for r in rows)]
    human_groups = sorted([g for g, rows in grouped.items() if all(not r['label'] for r in rows)], key=stable)
    if len(ai_groups) != 143 or len(backgrounds) != 48:
        raise ValueError('Complete143 consumed AI groups and48 backgrounds required')
    chosen, used_humans = [], set()
    for group in sorted(ai_groups, key=stable):
        rows = grouped[group]
        ai = min((r for r in rows if r['label']), key=lambda r: stable(r['id']))
        pool = [r for r in rows if not r['label']]
        human_group = group
        if not pool:
            human_group = next(g for g in human_groups if g not in used_humans)
            used_humans.add(human_group)
            pool = grouped[human_group]
        human = min(pool, key=lambda r: stable(r['id']))
        chosen.append({'id': stable(group + '\0' + human_group), 'original_groups': sorted({group, human_group}),
            'foregrounds': [human, ai], 'generator': ai['generator']})
    strata = defaultdict(list)
    for pair in chosen:
        strata[pair['generator']].append(pair)
    pairs, ordinal = [], 0
    for generator, members in sorted(strata.items()):
        for pair in sorted(members, key=lambda p: stable(p['id'])):
            pair['role'] = ROLES[ordinal % 4]
            pairs.append(pair)
            ordinal += 1
    pools = defaultdict(list)
    for kind in ('noise', 'music', 'speech'):
        for index, background in enumerate(sorted((r for r in backgrounds if r['kind'] == kind), key=lambda r: stable(r['id']))):
            copied = {**background, 'role': ROLES[index % 4]}
            pools[copied['role'], kind].append(copied)
    seen = {}
    for pair in pairs:
        lineage = [('group', g) for g in pair['original_groups']]
        for row in pair['foregrounds']:
            lineage += [('bytes', row['sha256']), ('speaker', row['canonical_speaker']),
                ('transcript', row.get('transcript_sha256'))]
        for key in lineage:
            if key[1] and seen.setdefault(key, pair['role']) != pair['role']:
                raise ValueError('Known foreground ancestry crosses roles')
    for rows in pools.values():
        for row in rows:
            for key in [('bytes', row['sha256']), ('speaker', row.get('canonical_speaker'))]:
                if key[1] and seen.setdefault(key, row['role']) != row['role']:
                    raise ValueError('Known background/foreground ancestry crosses roles')
    trial.mkdir(exist_ok=False)
    body = {'schema': 'chipmunk.audio-specialists-source-plan/15', 'prior_source_plan_sha256': SOURCE_SHA,
        'preparation_code_sha256': sha(Path(__file__)), 'method': METHOD, 'pairs': pairs,
        'backgrounds': [r for rows in pools.values() for r in rows],
        'unselected_consumed_sources': len(sources) - len({r['id'] for p in pairs for r in p['foregrounds']}),
        'unused_reserve_sources_scored': 0, 'authenticity_classifier_forwards': 0,
        'known_limits': METHOD['limits']}
    write(trial / 'source-plan.json', body)
    print(json.dumps({'pairs': len(pairs), 'roles': dict(Counter(p['role'] for p in pairs)),
        'generators': len(strata), 'backgrounds': len(backgrounds), 'authenticity_classifier_forwards': 0}), flush=True)


def load_sources(plan):
    from app.detectors.audio.decode import decode_audio

    objects = {r['sha256']: r for p in plan['pairs'] for r in p['foregrounds']}
    objects.update({r['sha256']: r for r in plan['backgrounds']})
    cache = {}
    foreground = {r['sha256'] for p in plan['pairs'] for r in p['foregrounds']}
    for key, row in objects.items():
        path = Path(row['source_path'])
        if sha(path) != key:
            raise ValueError('Original source bytes changed')
        audio = decode_audio(path, source_sha256=key)
        if audio.mapping['pcm_sha256'] != row['production_pcm_sha256'] or len(audio.pcm) < HALF:
            raise ValueError('Original production decode/sample clock changed')
        values = audio.pcm[:HALF].copy() if key in foreground else audio.pcm.copy()
        if key in foreground and pcm_sha(values) != row['prefix_pcm_sha256']:
            raise ValueError('Original foreground prefix changed')
        cache[key] = values
    return cache


def pair_signals(pair, plan, cache):
    """No clipping, label-dependent level choice or hidden background re-selection."""
    foregrounds = []
    for row in pair['foregrounds']:
        values = np.asarray(cache[row['sha256']], np.float64)
        rms = float(np.sqrt(np.mean(values**2)))
        if values.shape != (HALF,) or not np.isfinite(values).all() or rms < 1e-4:
            raise ValueError('Selected foreground is silent/invalid; keep failed selection')
        foregrounds.append(values * (.1 / rms))
    selected = {}
    for kind in ('noise', 'music', 'speech'):
        pool = sorted([r for r in plan['backgrounds'] if r['role'] == pair['role'] and r['kind'] == kind], key=lambda r: stable(r['id']))
        row = pool[int(stable(pair['id'] + kind + ':parent'), 16) % len(pool)]
        full = cache[row['sha256']]
        start = int(stable(pair['id'] + kind + ':offset'), 16) % (len(full) - HALF + 1)
        values = np.asarray(full[start:start + HALF], np.float64)
        rms = float(np.sqrt(np.mean(values**2)))
        if values.shape != (HALF,) or not np.isfinite(values).all() or rms < 1e-5:
            raise ValueError('Selected background interval is silent/invalid; do not select another offset')
        selected[kind] = {'source': row, 'start': start, 'pcm_sha256': pcm_sha(values.astype(np.float32)),
            'unit_rms': values / rms}
    second = {}
    peak = max(float(np.abs(x).max()) for x in foregrounds)
    for condition, kind, snr in CONDITIONS:
        second[condition] = []
        for values in foregrounds:
            mixed = values if kind is None else values + selected[kind]['unit_rms'] * (.1 / 10**(snr / 20))
            second[condition].append(mixed)
            peak = max(peak, float(np.abs(mixed).max()))
    gain = min(1., .98 / peak)
    for condition, kind, snr in CONDITIONS:
        for pattern, labels in PATTERNS.items():
            values = np.concatenate((foregrounds[labels[0]], second[condition][labels[1]])) * gain
            if values.shape != (EXTENT,) or not np.isfinite(values).all() or np.abs(values).max() > .980001:
                raise ValueError('Unexpected composite clock or clipping')
            background = None if kind is None else selected[kind]
            yield {'id': stable(pair['id'] + '/' + condition + '/' + pattern), 'group': pair['id'],
                'role': pair['role'], 'condition': condition, 'pattern': pattern, 'labels': list(labels),
                'generator': pair['generator'], 'common_peak_gain': gain,
                'foreground_sha256': [pair['foregrounds'][i]['sha256'] for i in labels],
                'background': None if background is None else {'sha256': background['source']['sha256'],
                    'kind': kind, 'start_sample': background['start'], 'end_sample': background['start'] + HALF,
                    'interval_pcm_sha256': background['pcm_sha256'], 'snr_db': snr}}, values.astype(np.float32)


def local_truth(labels, start, end):
    if (len(labels) != 2 or any(type(v) is not int or v not in (0, 1) for v in labels)
            or not 0 <= start < end <= EXTENT):
        raise ValueError('Known source intervals and local truth required')
    return int(any(label and max(start, index * HALF) < min(end, (index + 1) * HALF)
        for index, label in enumerate(labels)))


def signal_plan(main, trial):
    import tempfile

    import torch

    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.decode import decode_audio, read_wav, wav_bytes
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_calibration_v3_methods import band_view
    from scripts.audio_quality_v5 import features

    plan = json.loads((trial / 'source-plan.json').read_text())
    if plan['method'] != METHOD or plan['preparation_code_sha256'] != sha(Path(__file__)):
        raise ValueError('Prospective source recipe changed')
    assets = main / 'models/weights/audio-upgrade'
    for name, expected in ASSETS.items():
        if sha(assets / name) != expected:
            raise ValueError('Pinned commercial speech/voice model asset changed')
    torch.set_num_threads(4)
    activity = SpeechActivity(assets / 'silero_vad.jit')
    cache = load_sources(plan)
    support, hash_roles, decoded, files = defaultdict(set), {}, set(), 0
    path = trial / 'signal-plan.jsonl'
    with path.open('x') as stream:
        stream.write(json.dumps({'schema': 'chipmunk.audio-specialists-signal-plan/15',
            'source_plan_sha256': sha(trial / 'source-plan.json'), 'preparation_code_sha256': sha(Path(__file__)),
            'method': METHOD, 'authenticity_classifier_forwards': 0, 'speech_gate_model_sha256': ASSETS['silero_vad.jit']}) + '\n')
        for pair in plan['pairs']:
            for row, values in pair_signals(pair, plan, cache):
                raw = wav_bytes(values)
                pcm = read_wav(raw)
                row.update(input_sha256=hashlib.sha256(raw).hexdigest(), pcm_sha256=pcm_sha(pcm),
                    sample_rate=SR, sample_count=EXTENT, speech_spans_s=[list(v) for v in activity.spans(pcm)])
                filtered = band_view(pcm)
                row['band_pcm_sha256'] = pcm_sha(filtered)
                cell = row['role'], row['condition'], int(any(row['labels']))
                if cell not in decoded:
                    with tempfile.TemporaryDirectory() as folder:
                        source = Path(folder) / 'input.wav'
                        source.write_bytes(raw)
                        actual = decode_audio(source, source_sha256=row['input_sha256'])
                        if not np.array_equal(actual.pcm, pcm):
                            raise ValueError('Canonical8s production decode differs from exact planned PCM')
                    decoded.add(cell)
                row['windows'] = []
                for start in (0, 32000, 64000):
                    end = start + HALF
                    original = pcm[start:end]
                    quality = features(original)
                    context = 'diffuse' if quality['flatness'] >= METHOD['context_cut'] else 'concentrated'
                    eligible = window_in_speech(row['speech_spans_s'], start / SR, end / SR)
                    truth = local_truth(row['labels'], start, end)
                    window = {'start': start, 'end': end, 'extent': EXTENT, 'sample_rate': SR,
                        'label': truth, 'quality': quality, 'context': context, 'speech_eligible': eligible,
                        'full_pcm_sha256': pcm_sha(original), 'band_pcm_sha256': pcm_sha(filtered[start:end])}
                    for view in ('full', 'band'):
                        key = window[view + '_pcm_sha256']
                        if hash_roles.setdefault(key, row['role']) != row['role']:
                            raise ValueError('Derived voice window bytes connect different roles')
                    if eligible:
                        support[row['role'], context, truth].add(row['group'])
                    row['windows'].append(window)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                files += 1
            stream.flush()
            print(json.dumps({'prepared_pairs': files // 28, 'planned_pairs': 143,
                'files': files, 'authenticity_classifier_forwards': 0}), flush=True)
    report = {'schema': 'chipmunk.audio-specialists-signal-audit/15', 'signal_plan_sha256': sha(path),
        'source_plan_sha256': sha(trial / 'source-plan.json'), 'files': files, 'windows': files * 3,
        'native_decode_probes': len(decoded), 'unique_view_pcm_hashes': len(hash_roles),
        'eligible_foreground_groups_by_role_context_class': [{'role': role, 'context': context, 'label': label,
            'groups': len(support[role, context, label])} for role in ROLES for context in METHOD['contexts'] for label in (0, 1)],
        'authenticity_classifier_forwards': 0, 'source_reserve_scored': False, 'serving_admitted': False,
        'runtime': {k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'torch', 'transformers', 'pyloudnorm')},
        'background_independence_limit': METHOD['limits']}
    write(trial / 'signal-audit.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('sources', 'signals'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    (source_plan if args.stage == 'sources' else signal_plan)(args.main, args.trial)
