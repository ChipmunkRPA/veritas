"""Independent v3 validation audit; no evaluator, preparation or metric helpers.

Writes only research audit files. Production decode is used for replay checks,
never to alter serving configuration. No detector scoring is performed here.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/independent-evaluation/audio-v3'
AUDIT = OUT / 'audits'
CONDITIONS = ('clean', 'noise_0', 'noise_5', 'noise_15', 'music_5', 'music_15')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def require(test, message):
    if not test:
        raise AssertionError(message)


def near(actual, expected, message, tolerance=1e-10):
    if expected is None or actual is None:
        require(actual is None and expected is None, message + ': null mismatch')
    else:
        require(abs(float(actual)-float(expected)) <= tolerance, message + f': {actual} != {expected}')


def save(name, report):
    AUDIT.mkdir(parents=True, exist_ok=True)
    target = AUDIT / name
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False).encode() + b'\n'
    require(not target.exists() or target.read_bytes() == encoded, f'Audit is immutable: {target}')
    target.write_bytes(encoded)


def normalized(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def source_pcm(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    values, rate = sf.read(path, dtype='float32')
    if values.ndim == 2:
        values = values.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        values = resample_poly(values, 16000 // divisor, rate // divisor).astype(np.float32)
    return values


def preaudit():
    import soundfile as sf
    from app.detectors.audio.decode import decode_audio
    m = read(OUT / 'manifest.json')
    require(digest(canonical({k: v for k, v in m.items() if k != 'sha256'})) == m['sha256'], 'Manifest hash')
    bases, variants, backgrounds = m['base_files'], m['variants'], m['backgrounds']
    require(len(bases) == 420 and len(variants) == 2520, 'Fixed original and variant counts')
    counts = {s: Counter(r['label'] for r in bases if r['split'] == s) for s in ('fit', 'cal', 'test')}
    require(counts == {'fit': Counter({0: 20, 1: 20}), 'cal': Counter({0: 40, 1: 40}),
                       'test': Counter({0: 150, 1: 150})}, 'Fixed class counts')
    old_rows, old_backgrounds, old_paths, old_bg_paths = [], [], [], []
    for directory in ('audio', 'audio-v2'):
        old = read(ROOT / f'data/independent-evaluation/{directory}/manifest.json')
        require(old['sha256'] in m['prior_manifest_hashes'], 'Prior manifest reference')
        old_rows.extend(old['base_files'])
        old_backgrounds.extend(old['backgrounds'])
        old_paths.extend((ROOT / f'data/independent-evaluation/{directory}', r) for r in old['base_files'])
        old_bg_paths.extend((ROOT / f'data/independent-evaluation/{directory}', r) for r in old['backgrounds'])
    require(len(old_rows) == 264, 'Prior consumed original count')
    fields = ('speaker', 'source_group', 'transcript_sha256', 'sha256', 'decoded_pcm_sha256')
    comparisons = {}
    for field in fields:
        previous = {r[field] for r in old_rows if r.get(field) is not None}
        sets = {s: {r[field] for r in bases if r['split'] == s and r.get(field) is not None}
                for s in ('fit', 'cal', 'test')}
        for split, values in sets.items():
            require(not values & previous, f'Prior overlap in {field}/{split}')
        for left, right in (('fit', 'cal'), ('fit', 'test'), ('cal', 'test')):
            require(not sets[left] & sets[right], f'Crosssplit overlap in {field}: {left}/{right}')
        comparisons[field] = {s: len(v) for s, v in sets.items()}
    require(len({r['sha256'] for r in bases}) == len(bases), 'Source byte duplicates')
    require(len({r['production_pcm_sha256'] for r in bases}) == len(bases), 'Production source PCM duplicates')
    require(len({r['source_prefix_pcm_sha256'] for r in bases}) == len(bases), 'Source prefix duplicates')
    current = {r['id']: r for r in bases}
    require(len(current) == len(bases), 'Source IDs unique')
    require(len({r['id'] for r in variants}) == len(variants), 'Variant IDs unique')
    prior_full, prior_prefix = set(), set()
    for directory, source in old_paths:
        require(file_hash(directory/source['path']) == source['sha256'], 'Prior source original bytes')
        d = decode_audio(directory/source['path'], source_sha256=source['sha256'])
        prior_full.add(d.mapping['pcm_sha256'])
        for samples in range(32000, 64001, 4000):
            if len(d.pcm) >= samples:
                prior_prefix.add(digest(d.pcm[:samples].astype('<f4').tobytes()))
    hash_checks, source_pcm_checks, wav_pcm_checks = 0, 0, 0
    for source in bases:
        path = OUT / source['path']
        require(file_hash(path) == source['sha256'], 'Source bytes: ' + source['id'])
        hash_checks += 1
        pcm = source_pcm(path)
        require(digest(pcm.astype('<f4').tobytes()) == source['decoded_pcm_sha256'], 'Decoded source hash: ' + source['id'])
        source_pcm_checks += 1
        d = decode_audio(path, source_sha256=source['sha256'])
        prefix_hash = digest(d.pcm[:round(source['seconds']*16000)].astype('<f4').tobytes())
        require(d.mapping['pcm_sha256'] == source['production_pcm_sha256'] and prefix_hash == source['source_prefix_pcm_sha256'], 'Independent production source decode')
        require(d.mapping['pcm_sha256'] not in prior_full and prefix_hash not in prior_prefix, 'Independent old production source/prefix exclusion')
        require(source['license'] and source['license_url'] and source['label_source'], 'Source provenance')
        if source['transcript'] is not None:
            require(digest(normalized(source['transcript']).encode()) == source['transcript_sha256'], 'Transcript normalization')
        else:
            require(source['split'] == 'test' and source['label'] == 1 and source['dataset'] == 'isjwdu/DFADD', 'Unknown text restriction')
        require(2 <= source['seconds'] <= 4, 'Bounded source duration')
        require(source['speaker'] == source['cluster_id'], 'Canonical voice grouping')
    old_bg_hash = {r['sha256'] for r in old_backgrounds}
    old_bg_ids = {r.get('source_id') for r in old_backgrounds}
    old_bg_names = {Path(r.get('source_id') or r['path']).stem.lower() for r in old_backgrounds}
    old_bg_pcm = {digest(source_pcm(directory/r['path']).astype('<f4').tobytes()) for directory, r in old_bg_paths}
    bg_map = {r['source_id']: r for r in backgrounds}
    require(len(bg_map) == len(backgrounds), 'Distinct background works')
    for background in backgrounds:
        require(background['sha256'] not in old_bg_hash and background['source_id'] not in old_bg_ids, 'Prior background overlap')
        require(Path(background['source_id']).stem.lower() not in old_bg_names, 'Canonical prior background work overlap')
        require(file_hash(OUT / background['path']) == background['sha256'], 'Background bytes')
        bg_pcm_hash = digest(source_pcm(OUT / background['path']).astype('<f4').tobytes())
        require(bg_pcm_hash == background['decoded_pcm_sha256'], 'Background PCM')
        require(bg_pcm_hash not in old_bg_pcm, 'Decoded prior background PCM overlap')
        require(background['license'] and background['license_url'], 'Background licence')
        hash_checks += 1
    by_source = defaultdict(list)
    per_background = defaultdict(Counter)
    def independent_variant_decode(row):
        # Invoke the pinned FFmpeg directly; do not approximate host-specific
        # SIMD rounding with NumPy or call the production mapping helper here.
        path = (OUT / row['path']).resolve()
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-protocol_whitelist', 'file',
            '-copyts', '-i', str(path), '-map', '0:a:0', '-vn', '-af',
            'aformat=sample_fmts=s16:channel_layouts=mono,aresample=16000,atrim=end_sample=9600000,ashowinfo,asetpts=N/SR/TB',
            '-t', '600', '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1']
        decoded = subprocess.run(command, capture_output=True, check=True, timeout=60, stdin=subprocess.DEVNULL).stdout
        pcm = np.frombuffer(decoded, dtype='<i2').astype(np.float32)/32768.
        require(len(pcm) == row['samples'], 'Independent FFmpeg sample count')
        return row['id'], digest(pcm.astype('<f4').tobytes())
    print('Source exclusion checks passed; independently decoding all 2,520 final inputs.', flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        independent_pcm = dict(pool.map(independent_variant_decode, variants))
    for row in variants:
        source = current[row['base_id']]
        by_source[row['base_id']].append(row)
        for field in ('split', 'label', 'speaker', 'cluster_id', 'generator', 'seconds'):
            require(row[field] == source[field], 'Variant source metadata: ' + field)
        require(row['source_interval_samples'] == [0, row['samples']], 'Source interval')
        require(row['samples'] == round(row['seconds'] * 16000), 'Sample duration')
        path = OUT / row['path']
        require(file_hash(path) == row['sha256'], 'Variant bytes')
        hash_checks += 1
        pcm, sr = sf.read(path, dtype='float32')
        require(pcm.ndim == 1 and sr == 16000 and len(pcm) == row['samples'], 'Variant encoding')
        require(independent_pcm[row['id']] == row['pcm_sha256'], 'Independent signed16 variant PCM: '+row['id'])
        wav_pcm_checks += 1
        if row['background_id']:
            bg = bg_map[row['background_id']]
            require(bg['split'] == row['split'] and bg['sha256'] == row['background_sha256'], 'Background split/identity')
            require(abs(row['measured_snr_db'] - row['snr_db']) < .02, 'Quantized mixture SNR')
            per_background[(row['split'], row['background_id'], row['condition'])][row['label']] += 1
        else:
            require(row['condition'] == 'clean' and row['snr_db'] is None, 'Clean background metadata')
    for source_id, rows in by_source.items():
        require(Counter(r['condition'] for r in rows) == Counter(CONDITIONS), 'Six conditions for ' + source_id)
        for kind in ('noise', 'music'):
            require(len({r['background_id'] for r in rows if r['condition'].startswith(kind)}) == 1, 'Same work across levels')
    for group, composition in per_background.items():
        require(set(composition) == {0, 1}, 'Background missing class: ' + str(group))
        require(abs(composition[0] - composition[1]) <= 1, 'Background class balance: ' + str(group))
    duration_counts = {}
    for split in ('fit', 'cal', 'test'):
        d = {label: Counter(r['seconds'] for r in bases if r['split'] == split and r['label'] == label) for label in (0, 1)}
        require(d[0] == d[1], 'Class duration matching: ' + split)
        duration_counts[split] = {str(c): dict(v) for c, v in d.items()}
    fresh_families = {r['generator'] for r in bases if r['split'] == 'test' and r['label'] == 1}
    development_families = {r['generator'] for r in bases if r['split'] != 'test' and r['label'] == 1}
    require(len(fresh_families) == 3 and not fresh_families & development_families, 'Held-out families')
    # Independent production-decoder spotchecks: one condition per split.
    replays = []
    for split in ('fit', 'cal', 'test'):
        for condition in CONDITIONS:
            row = next(r for r in variants if r['split'] == split and r['condition'] == condition)
            d = decode_audio(OUT / row['path'], source_sha256=row['sha256'])
            require(d.mapping['pcm_sha256'] == row['pcm_sha256'], 'Actual production decode replay')
            replays.append(row['id'])
    with (OUT / 'research/lj-speech-original-metadata.csv').open() as stream:
        lj_records = list(csv.reader(stream, delimiter='|'))
    require(len(lj_records) == 13100 and len({r[0] for r in lj_records}) == 13100, 'Complete LJ text corpus')
    lj_passages = {normalized(t) for r in lj_records for t in r[1:]}
    lj_fivegrams = {' '.join(t.split()[i:i+5]) for t in lj_passages for i in range(max(0, len(t.split())-4))}
    old_fivegram_flags, old_short_passage_flags = [], []
    for source in old_rows:
        text = normalized(source['transcript']); words = text.split()
        contained = [t for t in lj_passages if (' '+t+' ') in (' '+text+' ')]
        require(not [t for t in contained if len(t.split()) >= 5], 'Complete LJ passage of at least five words overlaps prior source')
        if contained: old_short_passage_flags.append(dict(id=source['id'], passages=contained))
        if any(' '.join(words[i:i+5]) in lj_fivegrams for i in range(max(0, len(words)-4))):
            old_fivegram_flags.append(source['id'])
    for source in bases:
        if source['transcript'] is not None:
            words = normalized(source['transcript']).split()
            require(not any(' '.join(words[i:i+5]) in lj_fivegrams for i in range(max(0, len(words)-4))), 'Known new transcript LJ five-word overlap')
    report = dict(schema='chipmunk.audio-v3-independent-preparation-audit/1',
        recorded_at=datetime.now(timezone.utc).isoformat(), manifest_sha256=m['sha256'],
        method='Independent source/hash/group checks and direct FFmpeg final-input decoding; no preparation/evaluation helpers imported.',
        status='verified_with_documented_text_and_background_population_limits', deployment_eligible=False,
        original_counts={s: dict(v) for s, v in counts.items()}, exclusion_group_counts=comparisons,
        byte_hash_checks=hash_checks, source_pcm_checks=source_pcm_checks, quantized_variant_pcm_checks=wav_pcm_checks,
        independently_decoded_prior_sources=len(old_paths), independently_decoded_current_sources=len(bases),
        prior_production_full_hashes=len(prior_full), prior_production_prefix_hashes=len(prior_prefix),
        production_decode_replays=replays, duration_counts=duration_counts,
        background_class_counts={str(k): dict(v) for k, v in per_background.items()},
        test_generator_families=sorted(fresh_families),
        independent_lj_corpus_checks=dict(entries=len(lj_records), old_fivegram_flags=old_fivegram_flags,
            old_short_passage_flags=old_short_passage_flags, complete_passage_at_least_five_words_overlap=[],
            known_new_fivegram_overlap=[], dfadd_per_file_mapping_verified=False),
        limitations=['DFADD actual prompt strings are unpublished; corpus-level exclusion is not exact per-file text verification.',
            'Shared generic prior five-word phrases remain flagged; no statement of exact prompt independence is made.',
            'Voice aliases are publisher identities, not cross-vendor biometric identity verification.',
            'Background counts support fixed-pool evaluation, not arbitrary-environment guarantees.'])
    save('independent-preparation-audit.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'original_counts', 'byte_hash_checks', 'source_pcm_checks', 'quantized_variant_pcm_checks')}, indent=2))


def calibrated(value, params):
    if value is None:
        return None
    value = min(max(value, 1e-6), 1-1e-6)
    log_odds = params['slope'] * (math.log(value)-math.log1p(-value)) + params['intercept']
    return 1/(1+math.exp(-log_odds)) if log_odds >= 0 else math.exp(log_odds)/(1+math.exp(log_odds))


def leaning(p, guard):
    if p is None:
        return 'inconclusive', 'inconclusive'
    basic = 'ai' if p >= .8 else 'original' if p <= 1-.8+1e-12 else 'inconclusive'
    if min(len(guard[str(c)]) for c in (0, 1)) < 9:
        return basic, 'inconclusive'
    included = []
    for label, nonconformity in ((0, p), (1, 1-p)):
        maxima = [g['maximum'] for g in guard[str(label)]]
        rank_p = (1+sum(v is None or v >= nonconformity for v in maxima))/(1+len(maxima))
        if rank_p > .1:
            included.append(label)
    guarded = basic if ((basic == 'ai' and included == [1]) or
                        (basic == 'original' and included == [0])) else 'inconclusive'
    return basic, guarded


def point_reconstruction(rows, system, common=False):
    """Direct weighted sums; independent of the evaluator's vectorized helpers."""
    from scipy.stats import beta
    key = system.split('_')[0]
    repeats = Counter(r['base_id'] for r in rows)
    w = [1/repeats[r['base_id']] for r in rows]
    def rate(event, population):
        denominator = sum(weight for r, weight in zip(rows, w) if population(r))
        return sum(weight for r, weight in zip(rows, w) if population(r) and event(r))/denominator if denominator else None
    call = lambda r: r[system] != 'inconclusive'
    real_cov = rate(call, lambda r: r['label'] == 0)
    ai_cov = rate(call, lambda r: r['label'] == 1)
    result = dict(false_ai_rate=rate(lambda r: r[system] == 'ai', lambda r: r['label'] == 0),
        false_original_rate=rate(lambda r: r[system] == 'original', lambda r: r['label'] == 1),
        real_coverage=real_cov, ai_coverage=ai_cov,
        coverage=None if real_cov is None or ai_cov is None else (real_cov+ai_cov)/2,
        correct_ai_recall=rate(lambda r: r[system] == 'ai', lambda r: r['label'] == 1),
        correct_original_recall=rate(lambda r: r[system] == 'original', lambda r: r['label'] == 0),
        ai_call_risk=rate(lambda r: r['label'] == 0, lambda r: r[system] == 'ai'),
        original_call_risk=rate(lambda r: r['label'] == 1, lambda r: r[system] == 'original'),
        decided_risk=rate(lambda r: (r['label'] == 0 and r[system] == 'ai') or (r['label'] == 1 and r[system] == 'original'), call),
        assessed_fraction=rate(lambda r: r[key+'_p'] is not None, lambda r: True))
    threshold = all(r[key+'_threshold_established'] for r in rows)
    result.update(threshold_fpr=rate(lambda r: r[key+'_threshold_ai'], lambda r: r['label'] == 0) if threshold else None,
                  threshold_recall=rate(lambda r: r[key+'_threshold_ai'], lambda r: r['label'] == 1) if threshold else None)
    assessed = lambda r: r[key+'_p'] is not None and (not common or (r['old_p'] is not None and r['new_p'] is not None))
    class_totals = {c: sum(weight for r, weight in zip(rows, w) if r['label'] == c and assessed(r)) for c in (0, 1)}
    brier, loss = 0., 0.
    for r, weight in zip(rows, w):
        if not assessed(r) or min(class_totals.values()) == 0:
            continue
        p, y = min(max(r[key+'_p'], 1e-12), 1-1e-12), r['label']
        weight = .5*weight/class_totals[y]
        brier += weight*(p-y)**2
        loss += weight*(-y*math.log(p)-(1-y)*math.log1p(-p))
    result['brier'] = brier if min(class_totals.values()) else None
    result['log_loss'] = loss if min(class_totals.values()) else None
    scored = [r for r in rows if r[key+'_raw'] is not None]
    counts_scored = Counter(r['base_id'] for r in scored)
    ranks = defaultdict(lambda: [0., 0.])
    for r in scored: ranks[r[key+'_raw']][r['label']] += 1/counts_scored[r['base_id']]
    total_real = sum(v[0] for v in ranks.values()); total_ai = sum(v[1] for v in ranks.values())
    lower_real, favorable_pairs = 0., 0.
    for raw, (real_weight, ai_weight) in sorted(ranks.items()):
        favorable_pairs += ai_weight*(lower_real+.5*real_weight)
        lower_real += real_weight
    result['raw_auc'] = favorable_pairs/(total_real*total_ai) if total_real and total_ai else None
    result.update(inputs=len(rows), real_inputs=sum(r['label'] == 0 for r in rows),
        ai_inputs=sum(r['label'] == 1 for r in rows), original_files=len(repeats),
        identities=len({r['cluster_id'] for r in rows}),
        false_ai_count=sum(r['label'] == 0 and r[system] == 'ai' for r in rows),
        false_original_count=sum(r['label'] == 1 and r[system] == 'original' for r in rows),
        decisions={s: sum(r[system] == s for r in rows) for s in ('ai', 'original', 'inconclusive')},
        status_counts=dict(Counter(r[key+'_status'] for r in rows)))
    result['stability_veto_count'] = sum(r['new_unguarded'] == 'original' and r['new_guarded'] == 'inconclusive' for r in rows) if system == 'new_guarded' else 0
    if key == 'new':
        result['probe_status_counts'] = dict(Counter(r['new_probe_status'] for r in rows))
    for label, field in ((0, 'false_ai_identity_event_bound'), (1, 'false_original_identity_event_bound'),
                         (0, 'threshold_false_ai_identity_event_bound')):
        is_threshold = field.startswith('threshold')
        identities = {r['cluster_id'] for r in rows if r['label'] == label}
        events = {r['cluster_id'] for r in rows if r['label'] == label and
                  (r[key+'_threshold_ai'] if is_threshold else r[system] == ('ai' if label == 0 else 'original'))}
        n, k = len(identities), len(events)
        upper = None if not n else 1. if n == k else float(beta.ppf(.95, k+1, n-k))
        result[field] = dict(events=k, identities=n, one_sided_95_upper=upper)
        if is_threshold and not threshold:
            result[field] = dict(events=None, identities=n, one_sided_95_upper=None)
    return result


def select_group(rows, group):
    if group == 'all': return rows
    if group in CONDITIONS: return [r for r in rows if r['condition'] == group]
    if group.startswith('generator:'): return [r for r in rows if r['label'] == 0 or r['generator'] == group.removeprefix('generator:')]
    if group == 'vctk_matched': return [r for r in rows if r['cluster_id'].startswith('vctk:')]
    if group == 'libri_cross_source': return [r for r in rows if r['label'] == 1 or not r['cluster_id'].startswith('vctk:')]
    if group == 'seconds_2_to_lt3': return [r for r in rows if r['seconds'] < 3]
    if group == 'seconds_3_to_4': return [r for r in rows if r['seconds'] >= 3]
    if group.startswith('background:'):
        bg, condition = group.removeprefix('background:').rsplit(':', 1)
        return [r for r in rows if r['background_id'] == bg and r['condition'] == condition]
    raise AssertionError('Unknown report group: ' + group)


def interval_reconstruction(rows, reported):
    """Recreate draw stream and weighted paired distributions using cached arrays."""
    expected = reported['uncertainty']
    require(expected['seed'] == 3101985 and expected['requested_replicates'] == 2000, 'Frozen bootstrap design')
    rng = np.random.default_rng(3101985)
    identity = {'librispeech': sorted({r['cluster_id'] for r in rows if not r['cluster_id'].startswith('vctk:')}),
                'vctk_joint': sorted({r['cluster_id'] for r in rows if r['cluster_id'].startswith('vctk:')})}
    background = {kind: sorted({r['background_id'] for r in rows if r['condition'].startswith(kind+'_')}) for kind in ('noise', 'music')}
    cache = {}
    for group in ('all', *CONDITIONS):
        indices = np.array([i for i, r in enumerate(rows) if group == 'all' or r['condition'] == group])
        selected = [rows[i] for i in indices]
        repeats = Counter(r['base_id'] for r in selected)
        item = dict(indices=indices, y=np.array([r['label'] for r in selected]),
                    weight=np.array([1/repeats[r['base_id']] for r in selected]))
        for key in ('old', 'new'):
            item[key+'_p'] = np.array([np.nan if r[key+'_p'] is None else r[key+'_p'] for r in selected])
            item[key+'_threshold'] = np.array([r[key+'_threshold_ai'] for r in selected])
            item[key+'_threshold_established'] = all(r[key+'_threshold_established'] for r in selected)
        for system in ('old_guarded', 'new_guarded', 'new_unguarded'):
            item[system] = np.array([{'ai': 1, 'original': 0, 'inconclusive': 2}[r[system]] for r in selected])
        cache[group] = item
    def calculations(data, multiplicity):
        w = data['weight'] * multiplicity
        y = data['y']; real, ai = y == 0, y == 1
        w0, w1 = sum(w[real]), sum(w[ai])
        def class_rate(mask, label):
            denominator = w1 if label else w0
            return float(np.sum(w[mask & (ai if label else real)]) / denominator) if denominator else None
        result = {}
        shared = np.isfinite(data['old_p']) & np.isfinite(data['new_p'])
        for system in ('old_guarded', 'new_guarded', 'new_unguarded'):
            key = system.split('_')[0]
            decisions = data[system]
            v = dict(false_ai_rate=class_rate(decisions == 1, 0), false_original_rate=class_rate(decisions == 0, 1),
                     real_coverage=class_rate(decisions != 2, 0), ai_coverage=class_rate(decisions != 2, 1))
            v['coverage'] = (v['real_coverage'] + v['ai_coverage'])/2 if w0 and w1 else None
            for c, field in ((0, 'threshold_fpr'), (1, 'threshold_recall')):
                v[field] = class_rate(data[key+'_threshold'], c) if data[key+'_threshold_established'] else None
            v['brier'], v['log_loss'] = None, None
            amounts = [float(np.sum(w[shared & (y == c)])) for c in (0, 1)]
            if min(amounts) > 0:
                p = np.clip(data[key+'_p'][shared], 1e-12, 1-1e-12)
                ys = y[shared]
                wp = w[shared]*np.where(ys == 0, .5/amounts[0], .5/amounts[1])
                v['brier'] = float(np.dot(wp, (p-ys)**2))
                v['log_loss'] = float(np.dot(wp, -ys*np.log(p)-(1-ys)*np.log1p(-p)))
            result[system] = v
        return result
    distributions = defaultdict(list)
    for _ in range(2000):
        voice_draw = Counter()
        for population in identity.values(): voice_draw.update(rng.choice(population, len(population), replace=True))
        background_draw = Counter()
        for population in background.values(): background_draw.update(rng.choice(population, len(population), replace=True))
        voice = np.array([voice_draw[r['cluster_id']] for r in rows], dtype=float)
        crossed = voice * np.array([background_draw[r['background_id']] if r.get('background_id') else 1 for r in rows])
        for kind, multiplicity in (('voice', voice), ('voice_background', crossed)):
            for group, cached in cache.items():
                values = calculations(cached, multiplicity[cached['indices']])
                for metric in ('brier', 'log_loss', 'false_ai_rate', 'false_original_rate', 'coverage',
                               'real_coverage', 'ai_coverage', 'threshold_recall', 'threshold_fpr'):
                    a, b = values['new_guarded'][metric], values['old_guarded'][metric]
                    distributions[(kind, group, 'new_minus_old:'+metric)].append(None if a is None or b is None else a-b)
                for metric in ('real_coverage', 'ai_coverage', 'false_ai_rate', 'false_original_rate'):
                    a, b = values['new_guarded'][metric], values['new_unguarded'][metric]
                    distributions[(kind, group, 'veto_minus_no_veto:'+metric)].append(None if a is None or b is None else a-b)
    matched = 0
    for (kind, group, metric), draws in distributions.items():
        actual = expected['intervals'][kind][group][metric]
        finite = [v for v in draws if v is not None and math.isfinite(v)]
        require(actual['replicates'] == len(finite) and actual['undefined'] == 2000-len(finite), 'Bootstrap undefined counts')
        if finite:
            bounds = np.percentile(finite, [2.5, 97.5])
            for a, b in zip(actual['percentile_95'], bounds): near(a, b, 'Bootstrap percentile interval', 1e-9)
            near(actual['one_sided_95_upper'], np.percentile(finite, 95), 'Bootstrap one-sided bound', 1e-9)
        else:
            require(actual['percentile_95'] is None and actual['one_sided_95_upper'] is None, 'Undefined bootstrap interval')
        matched += 1
    return dict(replicates=2000, seed=3101985, matched_interval_cells=matched,
                joint_voice_strata={k: len(v) for k, v in identity.items()}, background_groups={k: len(v) for k, v in background.items()})


def calibration_reconstruction(rows, cal):
    # Verify fit-only optimization objective/gradient and class smoothing, without refitting.
    fit_checks = {}
    for key in ('old', 'new'):
        params = cal['parameters'][key]
        require(math.isfinite(params['slope']) and params['slope'] >= 0 and math.isfinite(params['intercept']), 'Finite monotone fitted parameters')
        fitting = [r for r in rows if r['split'] == 'fit' and r[key+'_raw'] is not None]
        source_n = {c: len({r['base_id'] for r in fitting if r['label'] == c}) for c in (0, 1)}
        require(params['source_n_by_class'] == [source_n[0], source_n[1]], 'Fit-only smoothing source counts')
        repeats = Counter(r['base_id'] for r in fitting)
        objective, slope_gradient, intercept_gradient = 0., 0., 0.
        for r in fitting:
            y = r['label']; raw = min(max(r[key+'_raw'], 1e-6), 1-1e-6)
            x = math.log(raw)-math.log1p(-raw)
            z = cal['parameters'][key]['slope']*x+cal['parameters'][key]['intercept']
            target = (source_n[1]+1)/(source_n[1]+2) if y else 1/(source_n[0]+2)
            weight = .5/source_n[y]/repeats[r['base_id']]
            objective += weight*(float(np.logaddexp(0, z))-target*z)
            residual = weight*(r[key+'_p']-target)
            slope_gradient += residual*x; intercept_gradient += residual
        near(objective, cal['parameters'][key]['objective'], 'Fit objective')
        require(abs(intercept_gradient) < 1e-5, 'Fit intercept gradient')
        require(abs(slope_gradient) < 1e-5 or (cal['parameters'][key]['slope'] == 0 and slope_gradient >= -1e-5), 'Constrained fit slope gradient')
        fit_checks[key] = dict(objective=objective, slope_gradient=slope_gradient, intercept_gradient=intercept_gradient)
        calibration = [r for r in rows if r['split'] == 'cal']
        for label in (0, 1):
            groups = defaultdict(list)
            for r in calibration:
                if r['label'] == label:
                    p = r[key+'_p']
                    groups[r['speaker']].append(None if p is None else p if label == 0 else 1-p)
            recorded = {r['speaker']: r for r in cal['guards'][key][str(label)]}
            require(set(groups) == set(recorded), 'Cal-only identity groups')
            for speaker, values in groups.items():
                near(recorded[speaker]['maximum'], None if None in values else max(values), 'Identity maximum')
                require(recorded[speaker]['rows'] == len(values), 'Calibration row count')
        real_groups = defaultdict(list)
        for r in calibration:
            if r['label'] == 0: real_groups[r['speaker']].append(r[key+'_p'])
        maxima = {g: None if None in values else max(values) for g, values in real_groups.items()}
        threshold_info = cal['thresholds'][key]
        require(set(maxima) == set(threshold_info['maxima']), 'Threshold identities')
        for speaker, value in maxima.items(): near(value, threshold_info['maxima'][speaker], 'Threshold maximum')
        rank = math.ceil((len(maxima)+1)*.95)
        ordered = sorted(math.inf if p is None else p for p in maxima.values())
        order = ordered[rank-1] if rank <= len(ordered) else math.inf
        threshold = float(np.nextafter(order, math.inf)) if order < 1 else None
        near(threshold, threshold_info['threshold'], 'Cal-only order threshold')
    return fit_checks


def admission_reconstruction(rows, report, verified, calibration):
    """Independent gate accounting after metric/interval reconstruction."""
    from scipy.stats import beta
    failed = []
    for kind in ('voice', 'voice_background'):
        for metric in ('brier', 'log_loss'):
            ci = report['uncertainty']['intervals'][kind]['all']['new_minus_old:'+metric]
            if ci['undefined'] or ci['percentile_95'] is None or ci['percentile_95'][1] >= 0:
                failed.append(f'{kind}: common-assessed {metric} improvement is unestablished')
    precision = {}
    for condition in CONDITIONS:
        selected = [r for r in rows if r['condition'] == condition]
        precision[condition] = {}
        floor, loss = (.75, .1) if condition in ('clean', 'noise_15', 'music_15') else (.5, .25)
        for name, label in (('real', 0), ('ai', 1)):
            c = verified[condition]['new_guarded'][name+'_coverage']
            previous = verified[condition]['new_unguarded'][name+'_coverage']
            if c is None or c < floor: failed.append(f'{condition}: {name} decision coverage below {floor:.0%}')
            if c is None or previous is None or previous-c > loss+1e-12:
                failed.append(f'{condition}: {name} coverage loss exceeds {loss:.0%}')
            error = 'ai' if label == 0 else 'original'
            population = [r for r in selected if r['label'] == label]
            identities = {r['cluster_id'] for r in population}
            harm = {r['cluster_id'] for r in population if r['new_guarded'] == error and r['old_guarded'] != error}
            benefit = {r['cluster_id'] for r in population if r['old_guarded'] == error and r['new_guarded'] != error}
            n, k = len(identities), len(harm)
            upper = None if not n else 1. if k == n else float(beta.ppf(.95, k+1, n-k))
            no_discordance = not harm and not benefit
            sufficient = not no_discordance or (upper is not None and upper <= .02)
            computed = dict(identities=n, harmed_identities=len(harm), helped_identities=len(benefit),
                zero_discordance=no_discordance, new_harm_event_one_sided_95_upper=upper,
                two_percentage_point_precision_established=sufficient)
            for key, value in computed.items():
                near(report['sparse_discordance_precision'][condition][name][key], value, 'Independent sparse precision: '+key)
            precision[condition][name] = computed
            if not sufficient:
                failed.append(f'{condition}: {name} zero-discordance identity precision cannot establish 2pp noninferiority')
        for kind in ('voice', 'voice_background'):
            for metric in ('false_ai_rate', 'false_original_rate'):
                ci = report['uncertainty']['intervals'][kind][condition]['new_minus_old:'+metric]
                if ci['undefined'] or ci['one_sided_95_upper'] is None or ci['one_sided_95_upper'] > .02:
                    failed.append(f'{condition}/{kind}: {metric} noninferiority is unestablished')
            ci = report['uncertainty']['intervals'][kind][condition]['new_minus_old:threshold_recall']
            if ci['undefined'] or ci['percentile_95'] is None or ci['percentile_95'][0] <= 0:
                failed.append(f'{condition}/{kind}: calibrated-threshold AI recall improvement is unestablished')
        upper = verified[condition]['new_guarded']['threshold_false_ai_identity_event_bound']['one_sided_95_upper']
        if calibration['thresholds']['new']['threshold'] is None or upper is None or upper > .075:
            failed.append(f'{condition}: calibrated 5% threshold lacks <=7.5% upper false-AI bound')
    require(set(failed) == set(report['admission']['failed_gates']), 'Independent admission gate outcomes')
    require(report['admission']['statistical_gates_passed'] == (not failed), 'Independent statistical admission flag')
    require(report['admission']['overall_confidence_weight'] == 0 and not report['admission']['deployment_eligible'], 'Offline admission boundary')
    return dict(failed_gates=failed, precision=precision, statistical_gates_passed=not failed,
                unknown_exact_prompt_mapping_unresolved=True, production_admission_verified=False)


def pretestaudit():
    """Audit frozen fit/cal coefficients and exact development prefix before test access."""
    require(not (OUT/'test-start.json').exists(), 'Pre-test audit must precede holdout access')
    m, cal, lock = (read(OUT/name) for name in ('manifest.json', 'calibration.json', 'run-lock.json'))
    data = (OUT/'scores.jsonl').read_bytes()
    records = [json.loads(line) for line in data.splitlines()]
    require(len(records) == 1440 and all(r['split'] in ('fit', 'cal') for r in records), 'Exact development-only score prefix')
    index = {(r['variant_id'], r['detector']): r for r in records}
    require(len(index) == len(records), 'Distinct development score pairs')
    require(file_hash(OUT/'run-lock.json') == cal['run_lock_sha256'], 'Calibration lock identity')
    require(lock['manifest_sha256'] == m['sha256'], 'Run lock cohort')
    require(file_hash(ROOT.parent/'docs/audio-calibration-v3-protocol.md') == lock['protocol_sha256'], 'Frozen protocol remains unchanged')
    for path, expected in lock['source_hashes'].items(): require(file_hash(ROOT/path) == expected, 'Frozen implementation: '+path)
    for path, expected in lock['pre_inference_evidence'].items(): require(file_hash(OUT/path) == expected, 'Frozen evidence: '+path)
    rows, canonical_rows = [], []
    for variant in m['variants']:
        if variant['split'] not in ('fit', 'cal'): continue
        row, mapped = dict(variant), dict(variant)
        for key in ('old', 'new'):
            record = index[(variant['id'], key)]
            for field in ('split', 'label', 'base_id', 'speaker', 'condition'):
                require(record[field] == variant[field], 'Development record identity: '+field)
            require(record['input_sha256'] == variant['sha256'] and record['production_input']['pcm_sha256'] == variant['pcm_sha256'], 'Development input byte/PCM identity')
            raw = record['raw_unrounded'] if record['score'] is not None else None
            if raw is not None:
                require(record['windows'] and math.isfinite(raw) and 0 <= raw <= 1, 'Valid development score')
                near(sum(w['raw_unrounded'] for w in record['windows'])/len(record['windows']), raw, 'Development raw window mean')
                near(round(raw, 6), record['score'], 'Development rounded score')
            row[key+'_raw'], row[key+'_status'], row[key+'_windows'] = raw, record['status'], record['windows']
            if key == 'new':
                views = record.get('views', {})
                row['new_band'], row['new_raw_rounded'], row['new_views'] = views.get('band'), views.get('raw'), views
                if views:
                    require(views['probe_status'] == 'not_required_for_decision' and views['band'] is None and views['probe_error'] is None, 'Calibration uses no unnecessary probe')
            mapped[key+'_raw'] = raw
            mapped[key+'_p'] = calibrated(raw, cal['parameters'][key])
        canonical_rows.append(row); rows.append(mapped)
    require(digest(canonical(canonical_rows)) == cal['development_rows_sha256'], 'Exact fitted development row hash')
    checks = calibration_reconstruction(rows, cal)
    report = dict(schema='chipmunk.audio-v3-independent-pretest-audit/1', recorded_at=datetime.now(timezone.utc).isoformat(),
        status='frozen_calibration_and_development_prefix_verified_before_test', deployment_eligible=False,
        manifest_sha256=m['sha256'], run_lock_sha256=file_hash(OUT/'run-lock.json'),
        calibration_sha256=file_hash(OUT/'calibration.json'), development_prefix_sha256=digest(data),
        development_prefix_bytes=len(data), development_records=len(records),
        fit_objectives=checks, guard_identity_counts={k: {c: len(v) for c, v in g.items()} for k, g in cal['guards'].items()},
        thresholds={k: dict(status=v['status'], threshold=v['threshold'], identities=v['identity_count'], rank=v['rank']) for k, v in cal['thresholds'].items()},
        method='Independent fitting objective/gradient, class guard maxima, 95% order thresholds, exact canonical development score hash and freeze checks; no evaluation helpers imported.',
        permission='Only evaluation of this frozen candidate/cohort is supported; this audit grants no production admission.')
    save('independent-pretest-audit.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'development_records', 'guard_identity_counts', 'thresholds')}, indent=2))


def postaudit():
    m, cal, result = (read(OUT / name) for name in ('manifest.json', 'calibration.json', 'evaluation.json'))
    raw_bytes = (OUT / 'scores.jsonl').read_bytes()
    scores = [json.loads(line) for line in raw_bytes.splitlines()]
    by_pair = {(r['variant_id'], r['detector']): r for r in scores}
    require(len(scores) == 5040 and len(by_pair) == len(scores), 'Exact complete raw score count')
    start = read(OUT / 'test-start.json')
    prefix = raw_bytes[:start['development_prefix_bytes']]
    require(digest(prefix) == start['development_prefix_sha256'], 'Development prefix immutable')
    prefix_rows = [json.loads(line) for line in prefix.splitlines()]
    require(len(prefix_rows) == 1440 and all(r['split'] != 'test' for r in prefix_rows), 'Development-only prefix')
    require(all(r['split'] == 'test' for r in scores[1440:]), 'Test-only suffix')
    require(file_hash(OUT/'calibration.json') == start['calibration_sha256'], 'Frozen calibration')
    require(file_hash(OUT/'run-lock.json') == start['run_lock_sha256'], 'Frozen run lock')
    require(file_hash(OUT/'scores.jsonl') == result['scores_sha256'], 'Report score hash')
    rows = []
    for source in m['variants']:
        row = dict(source)
        for key in ('old', 'new'):
            record = by_pair[(source['id'], key)]
            for name in ('split', 'label', 'base_id', 'speaker', 'condition'):
                require(record[name] == source[name], 'Recorded input identity')
            require(record['input_sha256'] == source['sha256'] and record['production_input']['pcm_sha256'] == source['pcm_sha256'], 'Scored hash identity')
            raw = record['raw_unrounded'] if record['score'] is not None else None
            if raw is not None:
                require(math.isfinite(raw) and 0 <= raw <= 1, 'Valid raw probability')
                require(record['windows'], 'Assessed windows present')
                near(sum(w['raw_unrounded'] for w in record['windows'])/len(record['windows']), raw, 'Mean aggregation')
                near(round(raw, 6), record['score'], 'Six-decimal exposed aggregate')
            for window in record['windows']:
                require(0 <= window['start_sample'] < window['end_sample'] <= source['samples'], 'Bounded score window')
            if record['status'] != 'failed':
                proposed = record['proposed_windows']
                require(len(proposed) == 1 and proposed[0]['start_sample'] == 0 and proposed[0]['end_sample'] == source['samples'], 'Full short-clip window plan')
                spans = record['applied_speech_spans_s']
                duration = source['samples']/16000
                speech_seconds = sum(max(0., min(duration, b)-max(0., a)) for a, b in spans)
                accepted = speech_seconds+1e-9 >= .5*duration
                require(proposed[0]['accepted_by_speech_gate'] == accepted, 'Independent speech overlap gate')
                actual_intervals = [(w['start_sample'], w['end_sample']) for w in record['windows']]
                require(actual_intervals == ([(0, source['samples'])] if accepted else []), 'Accepted window list matches gate')
            row[key+'_raw'], row[key+'_status'] = raw, record['status']
            row[key+'_p'] = calibrated(raw, cal['parameters'][key])
            basic, guarded = leaning(row[key+'_p'], cal['guards'][key])
            row[key+'_probability_only'], row[key+'_unguarded'] = basic, guarded
            if key == 'new':
                views = record.get('views')
                row['new_probe_status'] = views.get('probe_status', 'original_scoring_failed') if views else 'original_scoring_failed'
                if views is None:
                    require(record['status'] == 'failed', 'Missing candidate views require recorded failure')
                else:
                    require(views['source_pcm_sha256'] == source['pcm_sha256'], 'Candidate source view identity')
                    near(views['raw'], record['score'], 'Inner/outer exposed mean')
                    near(views['raw_unrounded'], raw, 'Inner/outer unrounded mean')
                    require(len(views['windows']) == len(record['windows']), 'Inner/outer window count')
                    for inner, outer in zip(views['windows'], record['windows']):
                        require(inner['start_sample'] == outer['start_sample'] and inner['end_sample'] == outer['end_sample'], 'Probe source interval alignment')
                        near(inner['raw'], outer['raw_unrounded'], 'Inner/outer window score')
                    required = source['split'] == 'test' and guarded == 'original'
                    if required:
                        require(views['probe_status'] in ('computed', 'failed'), 'Required original probe cannot be skipped')
                        if views['probe_status'] == 'computed':
                            require(views['band'] is not None and views['band_pcm_sha256'], 'Computed band view has evidence')
                            require(all(w['band'] is not None and math.isfinite(w['band']) and 0 <= w['band'] <= 1 for w in views['windows']), 'Finite band window scores')
                            require(views['probe_error'] is None, 'Computed probe has no failure')
                            near(sum(w['band'] for w in views['windows'])/len(views['windows']), views['band_unrounded'], 'Probe mean')
                            near(round(views['band_unrounded'], 6), views['band'], 'Probe rounded mean')
                        else:
                            require(views['band'] is None and views['probe_error'], 'Required failed probe explicitly recorded')
                    else:
                        require(views['probe_status'] == 'not_required_for_decision' and views['band'] is None, 'Intentional unused probe recorded')
                        require(views['probe_error'] is None and views['band_pcm_sha256'] is None, 'Skipped probe has neither error nor invented evidence')
            if key == 'new' and guarded == 'original':
                views = record.get('views', {})
                original, band = views.get('raw'), views.get('band')
                if original is None or band is None:
                    guarded = 'inconclusive'
                else:
                    logit = lambda p: math.log(min(max(p, 1e-6), 1-1e-6))-math.log1p(-min(max(p, 1e-6), 1-1e-6))
                    if abs(logit(original)-logit(band)) > math.log(2): guarded = 'inconclusive'
            row[key+'_guarded'] = guarded
            threshold = cal['thresholds'][key]['threshold']
            row[key+'_threshold_established'] = threshold is not None
            row[key+'_threshold_ai'] = row[key+'_p'] is not None and threshold is not None and row[key+'_p'] >= threshold
        rows.append(row)
    fit_checks = calibration_reconstruction(rows, cal)
    test = [r for r in rows if r['split'] == 'test']
    exported = read(OUT/'predictions.json')
    require([r['id'] for r in test] == [r['id'] for r in exported], 'Prediction row order/identity')
    for expected, published in zip(test, exported):
        for key in ('old', 'new'):
            near(expected[key+'_p'], published[key+'_p'], 'Published probability')
            for suffix in ('guarded', 'unguarded', 'probability_only', 'threshold_ai', 'threshold_established'):
                require(expected[key+'_'+suffix] == published[key+'_'+suffix], 'Published decision: '+suffix)
    verified = {}
    for group, systems in result['groups'].items():
        selected = select_group(test, group)
        verified[group] = {}
        for system, published in systems.items():
            calculated = point_reconstruction(selected, system)
            for key, value in calculated.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str): require(published[key][subkey] == subvalue, 'Grouped metric')
                        else: near(published[key][subkey], subvalue, f'{group}/{system}/{key}/{subkey}')
                else: near(published[key], value, f'{group}/{system}/{key}')
            verified[group][system] = calculated
    for system, published in result['common_assessed'].items():
        shared = point_reconstruction(test, system, common=True)
        for metric in ('brier', 'log_loss'):
            near(published[metric], shared[metric], 'Common-assessed proper score: '+system+'/'+metric)
    print('Independent point metrics matched; reconstructing all 2,000 paired bootstrap draws.', flush=True)
    intervals = interval_reconstruction(test, result)
    admission = admission_reconstruction(test, result, verified, cal)
    locked = read(OUT/'run-lock.json')
    for path, expected in locked['source_hashes'].items(): require(file_hash(ROOT/path) == expected, 'Final frozen implementation')
    current_serving = {str((ROOT/'app/models.yaml').relative_to(ROOT))}
    for directory in (ROOT/'app/engine/fusion/release', ROOT/'data/fusion'):
        if directory.exists(): current_serving.update(str(p.relative_to(ROOT)) for p in directory.rglob('*') if p.is_file())
    require(current_serving == set(locked['serving_hashes']), 'Serving artifact inventory unchanged')
    for path, expected in locked['serving_hashes'].items(): require(file_hash(ROOT/path) == expected, 'Serving bytes unchanged')
    weight_checks = 0
    for receipts in locked['verified_local_weight_files'].values():
        for receipt in receipts:
            require(file_hash(Path(receipt['path'])) == receipt['sha256'], 'Pinned local weight bytes unchanged')
            weight_checks += 1
    require(result['serving_unchanged'], 'Report serving configuration unchanged')
    report = dict(schema='chipmunk.audio-v3-independent-postrun-audit/1', recorded_at=datetime.now(timezone.utc).isoformat(),
        status='point_metrics_calibration_and_bootstrap_intervals_verified', deployment_eligible=False,
        method='Raw-record probability/decision/objective/calibration/threshold/weighted-metric reconstruction without evaluator helpers.',
        manifest_sha256=m['sha256'], calibration_sha256=file_hash(OUT/'calibration.json'),
        scores_sha256=file_hash(OUT/'scores.jsonl'), evaluation_sha256=file_hash(OUT/'evaluation.json'),
        predictions_sha256=file_hash(OUT/'predictions.json'), verified_weight_files=weight_checks,
        verified_serving_files=len(current_serving),
        original_score_records=len(scores), development_records=len(prefix_rows), test_variants=len(test),
        fit_objectives=fit_checks, verified_groups=verified, interval_audit=intervals, admission_audit=admission,
        limitation='Verified numerical results do not remove documented source/voice/background dependence or establish admission by themselves.')
    save('independent-postrun-audit.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'original_score_records', 'test_variants')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['pre', 'pretest', 'post'])
    args = parser.parse_args()
    {'pre': preaudit, 'pretest': pretestaudit, 'post': postaudit}[args.command]()
