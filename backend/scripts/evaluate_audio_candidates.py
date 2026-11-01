"""Frozen source/speaker-disjoint audio evaluation. Never writes production fusion files.

prepare downloads public provenance-labelled inputs; score is resumable and has
no access to fitted candidates; analyse fits only fit/cal and evaluates the locked
test under its recorded scope. All derivatives stay with their source/speaker split.

Maintenance v2 corrects production PCM decoding. Its seed and source selection
remain identical to the consumed first-stage holdout: a new output directory
is storage isolation, NOT new independent data. New accuracy confirmation
requires new source selection/exclusions and a prospectively frozen protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data/independent-evaluation/audio'
SR = 16000
N = 4 * SR
SEED = 'chipmunk-audio-evaluation-20261101-v1'
REVISIONS = {
    'openslr/librispeech_asr': '71cacbfb7e2354c4226d01e70d77d5fca3d04ba1',
    'ductai199x/dictation-speech': '692f01ad28a5d6f26bde093b6d717e89cfae0067',
    'corypaik/musan': '76f9882cfa4475efe11508ac9aa32722f84ca5b7',
}
TARGETS = {'fit': 16, 'cal': 24, 'test': 40}
EXCLUSIONS = []
OLD_DEVICE = 'cpu'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stable(value):
    return digest(f'{SEED}:{value}'.encode())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n'
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f'Refusing to replace a frozen artifact: {path}')
    path.write_bytes(data)


def get(url, *, params=None):
    import httpx
    error = None
    for attempt in range(5):
        try:
            response = httpx.get(url, params=params, timeout=60, follow_redirects=True)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            error = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                wait = min(60, max(30, int(exc.response.headers.get('retry-after', '30'))))
                print(f'Public dataset service rate limit; waiting {wait}s before retry.', flush=True)
                time.sleep(wait)
            else:
                time.sleep(min(8, 2 ** attempt))
    raise error


def metadata_rows(dataset, config, split, offset, length=100):
    path = OUT / 'metadata' / f'{dataset.replace("/", "_")}-{config}-{split}-{offset}-{length}.json'
    if not path.exists():
        data = get('https://datasets-server.huggingface.co/rows', params=dict(
            dataset=dataset, config=config, split=split, offset=offset, length=length)).json()
        save_json(path, data)
    return json.loads(path.read_bytes())['rows']


def download_audio(row, dataset, config, split, row_index):
    """The unaltered downloaded bytes are committed before decoding or scoring."""
    import soundfile as sf

    cache = OUT / 'metadata' / f'asset-{stable(f"{dataset}:{config}:{split}:{row_index}")}.json'
    if cache.exists():
        ref = json.loads(cache.read_bytes())
        raw = (OUT / ref['path']).read_bytes()
        if digest(raw) != ref['sha256']:
            raise ValueError('Downloaded source bytes changed')
    else:
        urls = row['audio']
        url = urls[0]['src']
        if REVISIONS[dataset] not in url:
            raise ValueError('Dataset server served a different revision')
        try:
            raw = get(url).content
        except Exception:
            # Refresh expiring public CDN signatures, not source identity.
            fresh = get('https://datasets-server.huggingface.co/rows', params=dict(
                dataset=dataset, config=config, split=split, offset=row_index, length=1)).json()['rows'][0]['row']
            url = fresh['audio'][0]['src']
            if REVISIONS[dataset] not in url:
                raise ValueError('Dataset revision changed while refreshing media URL') from None
            raw = get(url).content
        sha = digest(raw)
        path = OUT / 'sources' / f'{sha}.audio'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        ref = {'sha256': sha, 'bytes': len(raw), 'path': str(path.relative_to(OUT)),
               'url': url.split('?')[0], 'dataset': dataset, 'revision': REVISIONS[dataset],
               'config': config, 'upstream_split': split, 'row_index': row_index}
        save_json(cache, ref)
    pcm, rate = sf.read(io.BytesIO(raw), dtype='float32')
    if pcm.ndim == 2:
        pcm = pcm.mean(1)
    if rate != SR:
        from scipy.signal import resample_poly
        divisor = math.gcd(rate, SR)
        pcm = resample_poly(pcm, SR // divisor, rate // divisor).astype(np.float32)
    return ref, pcm, rate


def transcript_key(value):
    return digest(re.sub(r'[^a-z0-9]+', ' ', value.lower()).strip().encode())


def real_candidates(upstream_split):
    import fsspec
    import pyarrow.parquet as pq
    repo = 'openslr/librispeech_asr'
    url = f'https://huggingface.co/datasets/{repo}/resolve/{REVISIONS[repo]}/clean/{upstream_split}/0000.parquet'
    path = OUT / 'metadata' / f'librispeech-{upstream_split}-speakers.json'
    if not path.exists():
        with fsspec.open(url, block_size=65536) as file:
            table = pq.ParquetFile(file).read(columns=['speaker_id', 'chapter_id', 'id', 'text'])
        save_json(path, {'source_url': url, 'rows': table.to_pylist()})
    rows = json.loads(path.read_bytes())['rows']
    speakers = defaultdict(list)
    for index, row in enumerate(rows):
        speakers[str(row['speaker_id'])].append((index, row))
    return speakers


def prepare_reals():
    dev, test = real_candidates('validation'), real_candidates('test')
    if set(dev).intersection(test) or len(dev) != 40 or len(test) != 40:
        raise ValueError('Expected the official disjoint 40-speaker validation/test sets')
    ordered = sorted(dev, key=stable)
    jobs = [('fit' if i < 16 else 'cal', 'validation', speaker, dev[speaker]) for i, speaker in enumerate(ordered)]
    jobs += [('test', 'test', speaker, test[speaker]) for speaker in sorted(test, key=stable)]

    def one(job):
        split, upstream, speaker, choices = job
        for index, meta in sorted(choices, key=lambda pair: stable(pair[1]['id'])):
            data = metadata_rows('openslr/librispeech_asr', 'clean', upstream, index, 1)[0]['row']
            if data['id'] != meta['id']:
                raise ValueError('Parquet and dataset-server row order differ')
            ref, pcm, rate = download_audio(data, 'openslr/librispeech_asr', 'clean', upstream, index)
            if len(pcm) >= N and np.sqrt(np.mean(pcm[:N] ** 2)) > 1e-5:
                return {**ref, 'id': 'real:' + data['id'], 'label': 0, 'split': split,
                    'speaker': 'librispeech:' + speaker, 'generator': 'human', 'source_group': 'librispeech:' + data['id'],
                    'transcript_sha256': transcript_key(data['text']), 'transcript': data['text'],
                    'license': 'CC-BY-4.0', 'license_url': 'https://www.openslr.org/12/',
                    'label_source': 'LibriSpeech publisher: human LibriVox audiobook recordings',
                    'original_sample_rate': rate, 'decoded_samples': len(pcm),
                    'decoded_pcm_sha256': digest(pcm.astype('<f4').tobytes())}
            EXCLUSIONS.append({'id': data['id'], 'reason': 'Decoded source shorter than 4 seconds or first interval silent',
                               'decoded_samples': len(pcm), 'sha256': ref['sha256']})
        raise ValueError(f'No four-second source for speaker {speaker}')
    with ThreadPoolExecutor(max_workers=6) as pool:
        values = list(pool.map(one, jobs))
    return values


def prepare_fakes():
    repo = 'ductai199x/dictation-speech'
    candidates = {}
    for config in ('kokoro', 'chatterbox'):
        rows = []
        for offset in range(0, 600, 100):
            rows.extend(metadata_rows(repo, config, 'test', offset))
        candidates[config] = rows
    voices = sorted({item['row']['tts_voice'] for item in candidates['kokoro']}, key=stable)
    if len(voices) != 12:
        raise ValueError('Expected the publisher-documented twelve Kokoro voices')
    voice_split = {voice: 'fit' if i < 6 else 'cal' for i, voice in enumerate(voices)}
    used_text, used_sources, out = set(), set(), []
    for split in ('fit', 'cal', 'test'):
        config = 'chatterbox' if split == 'test' else 'kokoro'
        eligible = [r for r in candidates[config] if r['row']['duration_s'] >= 4
            and (split == 'test' or voice_split[r['row']['tts_voice']] == split)]
        # Cycle speakers, then hash order within each speaker. No score affects selection.
        grouped = defaultdict(list)
        for item in sorted(eligible, key=lambda r: stable(r['row']['id'])):
            grouped[item['row']['tts_voice']].append(item)
        ordered = []
        while any(grouped.values()):
            for voice in sorted(grouped, key=stable):
                if grouped[voice]:
                    ordered.append(grouped[voice].pop())
        chosen = 0
        for item in ordered:
            data, index = item['row'], item['row_idx']
            key = transcript_key(data['transcript'])
            source = data['source_url'] or data['rendition_id']
            if key in used_text or source in used_sources:
                continue
            ref, pcm, rate = download_audio(data, repo, config, 'test', index)
            if len(pcm) < N or np.sqrt(np.mean(pcm[:N] ** 2)) <= 1e-5:
                EXCLUSIONS.append({'id': data['id'], 'reason': 'Decoded source shorter than 4 seconds or first interval silent',
                                   'decoded_samples': len(pcm), 'sha256': ref['sha256']})
                continue
            out.append({**ref, 'id': 'fake:' + data['id'], 'label': 1, 'split': split,
                'speaker': data['tts_engine'] + ':' + data['tts_voice'], 'generator': data['tts_engine'],
                'source_group': source, 'transcript_sha256': key, 'transcript': data['transcript'],
                'text_source_url': data['source_url'], 'text_license': data['license'],
                'license': 'CC-BY-SA-4.0', 'license_url': f'https://huggingface.co/datasets/{repo}/blob/{REVISIONS[repo]}/README.md',
                'label_source': 'Publisher documents all clips as TTS generated and provides engine/voice metadata',
                'original_sample_rate': rate, 'decoded_samples': len(pcm),
                'decoded_pcm_sha256': digest(pcm.astype('<f4').tobytes())})
            used_text.add(key)
            used_sources.add(source)
            chosen += 1
            if chosen == TARGETS[split]:
                break
        if chosen != TARGETS[split]:
            raise ValueError(f'Insufficient pre-score eligible {split} fake sources: {chosen}')
    return out


def prepare_backgrounds():
    import soundfile as sf
    backgrounds = []
    noise_rows = metadata_rows('corypaik/musan', 'noise', 'train', 0)
    licence_url = f'https://huggingface.co/datasets/corypaik/musan/raw/{REVISIONS["corypaik/musan"]}/licenses/noise/free-sound/LICENSE'
    licence = get(licence_url).text
    if 'Public Domain' not in licence:
        raise ValueError('Noise-source rights declaration changed')
    for item in noise_rows:
        row = item['row']
        if 'free-sound' not in row['path']:
            continue
        ref, pcm, rate = download_audio(row, 'corypaik/musan', 'noise', 'train', item['row_idx'])
        if len(pcm) < N or np.sqrt(np.mean(pcm[:N] ** 2)) < 1e-4:
            continue
        split = ('fit', 'cal', 'test')[len(backgrounds)]
        backgrounds.append({**ref, 'kind': 'noise', 'split': split, 'source_id': row['path'],
            'license': 'Public-domain', 'license_url': licence_url, 'license_declaration': licence,
            'original_sample_rate': rate, 'decoded_samples': len(pcm)})
        if len(backgrounds) == 3:
            break
    music = [('fit', 'piano_nostalgia.wav', 'regret-short-emotional-piano', 'Wolfgang_'),
             ('cal', 'can_be_so_beautiful.wav', 'can-be-so-beautiful', 'cinameng / James Gargette'),
             ('test', 'at_the_end_of_hope_loop.wav', 'at-the-end-of-hope', 'Emma_MA')]
    for split, filename, page, author in music:
        url = 'https://opengameart.org/sites/default/files/' + filename
        path = OUT / 'metadata' / f'music-{split}.json'
        if path.exists():
            ref = json.loads(path.read_bytes())
        else:
            raw = get(url).content
            sf.info(io.BytesIO(raw))
            sha = digest(raw)
            asset = OUT / 'sources' / f'{sha}.audio'
            asset.write_bytes(raw)
            ref = {'sha256': sha, 'bytes': len(raw), 'path': str(asset.relative_to(OUT)), 'url': url,
                   'kind': 'music', 'split': split, 'source_id': page, 'author': author,
                   'license': 'CC0-1.0', 'license_url': f'https://opengameart.org/content/{page}'}
            save_json(path, ref)
        backgrounds.append(ref)
    return backgrounds


def decode(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    pcm, rate = sf.read(path, dtype='float32')
    if pcm.ndim == 2:
        pcm = pcm.mean(1)
    divisor = math.gcd(rate, SR)
    if rate != SR:
        pcm = resample_poly(pcm, SR // divisor, rate // divisor).astype(np.float32)
    return pcm


def mix_conditions(base, noise, music):
    """Common gain after mixtures preserves SNR and avoids clipping in all variants."""
    base = base[:N].astype(np.float64)
    base = base * (.1 / np.sqrt(np.mean(base ** 2)))
    mixtures = {'clean': base}
    for kind, background in (('noise', noise), ('music', music)):
        background = background.astype(np.float64)
        background *= np.sqrt(np.mean(base ** 2)) / (np.sqrt(np.mean(background ** 2)) * 10 ** (10 / 20))
        mixtures[kind] = base + background
    gain = min(1.0, .98 / max(np.max(np.abs(value)) for value in mixtures.values()))
    return {k: (v * gain).astype(np.float32) for k, v in mixtures.items()}, gain


def audit_groups(bases):
    cross = []
    counts = {}
    for key in ('speaker', 'source_group', 'transcript_sha256', 'sha256', 'decoded_pcm_sha256'):
        splits = defaultdict(set)
        for row in bases:
            splits[row[key]].add(row['split'])
        cross.extend({'key': key, 'identity': k, 'splits': sorted(v)} for k, v in splits.items() if len(v) > 1)
    for split in TARGETS:
        selected = [r for r in bases if r['split'] == split]
        counts[split] = {'base_files': len(selected), 'labels': dict(Counter(r['label'] for r in selected)),
            'real_speakers': len({r['speaker'] for r in selected if r['label'] == 0}),
            'fake_voices': len({r['speaker'] for r in selected if r['label'] == 1}),
            'generators': sorted({r['generator'] for r in selected if r['label'] == 1})}
    if cross:
        raise ValueError(f'Cross-split identity leakage: {cross[:3]}')
    if set(counts['test']['generators']).intersection(counts['cal']['generators'] + counts['fit']['generators']):
        raise ValueError('Held-out generator family leaked into development')
    return {'counts': counts, 'cross_split_identities': cross,
        'training_overlap': 'Unknown classifier training provenance; encoder pretraining includes LibriSpeech-related speech. This is split isolation, not proof of training independence.'}


def prepare():
    import soundfile as sf

    from app.detectors.audio.decode import decode_audio
    if (OUT / 'manifest.json').exists():
        print('Frozen manifest already exists; preparation is read-only after lock.')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for repo, revision in REVISIONS.items():
        meta = get('https://huggingface.co/api/datasets/' + repo).json()
        if meta['sha'] != revision or meta.get('gated'):
            raise ValueError('Dataset changed or requires approval: ' + repo)
        save_json(OUT / 'metadata' / f'{repo.replace("/", "_")}-publisher.json', meta)
        card = get(f'https://huggingface.co/datasets/{repo}/raw/{revision}/README.md').content
        (OUT / 'metadata' / f'{repo.replace("/", "_")}-README.md').write_bytes(card)
    print('Preparing real sources...', flush=True)
    bases = prepare_reals()
    print('Preparing labelled generated sources...', flush=True)
    bases += prepare_fakes()
    print('Preparing independently licensed backgrounds...', flush=True)
    backgrounds = prepare_backgrounds()
    grouped = audit_groups(bases)
    variants = []
    for row in bases:
        selected = {b['kind']: b for b in backgrounds if b['split'] == row['split']}
        arrays, offsets = {}, {}
        for kind, ref in selected.items():
            pcm = decode(OUT / ref['path'])
            offset = int(stable(row['id'] + kind)[:12], 16) % (len(pcm) - N + 1)
            arrays[kind], offsets[kind] = pcm[offset:offset + N], offset
        conditions, gain = mix_conditions(decode(OUT / row['path']), arrays['noise'], arrays['music'])
        for kind, pcm in conditions.items():
            sha = digest(pcm.astype('<f4').tobytes())
            file = OUT / 'inputs' / f'{sha}.wav'
            file.parent.mkdir(exist_ok=True)
            sf.write(file, pcm, SR, subtype='FLOAT')
            decoded = decode_audio(file, source_sha256=file_digest(file))
            if kind == 'clean':
                production_clean = decoded.pcm
            variants.append({'id': stable(row['id'] + ':' + kind), 'base_id': row['id'], 'split': row['split'],
                'label': row['label'], 'speaker': row['speaker'], 'condition': kind, 'path': str(file.relative_to(OUT)),
                'sha256': digest(file.read_bytes()), 'pcm_sha256': decoded.mapping['pcm_sha256'],
                'mixture_float_pcm_sha256': sha, 'production_input': decoded.mapping, 'samples': N, 'sample_rate': SR,
                'source_interval_samples': [0, N], 'snr_db': None if kind == 'clean' else 10,
                'background_sha256': None if kind == 'clean' else selected[kind]['sha256'],
                'background_start_sample': None if kind == 'clean' else offsets[kind], 'common_peak_gain': gain,
                'measured_snr_db': None if kind == 'clean' else float(20 * np.log10(
                    np.sqrt(np.mean(production_clean.astype(np.float64) ** 2)) /
                    np.sqrt(np.mean((decoded.pcm.astype(np.float64) - production_clean) ** 2))))})
    by_pcm = defaultdict(set)
    for variant in variants:
        by_pcm[variant['pcm_sha256']].add(variant['split'])
    if any(len(splits) > 1 for splits in by_pcm.values()):
        raise ValueError('Normalized input identity crossed splits')
    manifest = {'schema': 'chipmunk.audio-evaluation/2', 'seed': SEED, 'deployment_eligible': False,
        'scoring_pcm_contract': 'production-s16',
        'source_selection_status': 'Same seed and selection as the consumed first-stage holdout; engineering replay, not new independent confirmation.',
        'scope': 'Production-PCM replay of consumed first-stage sources; English first-four-second source windows; Kokoro development, unseen Chatterbox holdout; clean / one noise / one music per split at 10dB full-window RMS SNR.',
        'source_normalization': '16kHz mono, first 4 seconds, -20dBFS RMS then identical peak-avoidance gain; final scoring input from production decode_audio signed-16-bit quantization.',
        'base_files': bases, 'backgrounds': backgrounds, 'variants': variants, 'group_audit': grouped,
        'pre_score_exclusions': EXCLUSIONS}
    manifest['sha256'] = digest(canonical(manifest))
    save_json(OUT / 'manifest.json', manifest)
    print(json.dumps({'manifest_sha256': manifest['sha256'], 'variants': len(variants), **grouped}, indent=2), flush=True)


def read_manifest():
    manifest = json.loads((OUT / 'manifest.json').read_bytes())
    if digest(canonical({k: v for k, v in manifest.items() if k != 'sha256'})) != manifest['sha256']:
        raise ValueError('Frozen manifest was modified')
    audit_groups(manifest['base_files'])
    return manifest


def lock():
    import torch
    import yaml
    manifest = read_manifest()
    if manifest.get('scoring_pcm_contract') != 'production-s16':
        raise ValueError('Legacy FLOAT-input experiment is frozen. Use --output with a new directory and a new prospective protocol; historical replay uses run-source.')
    if OLD_DEVICE == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('Requested production MPS device is unavailable')
    protocol = ROOT.parent / 'docs/independent-evaluation-protocol.md'
    registry = yaml.safe_load((ROOT / 'app/models.yaml').read_text())
    ids = {'hf.gustking.wav2vec2_xlsr_deepfake', 'pella.voice_v2', 'silero.speech_activity'}
    files = ['scripts/evaluate_audio_candidates.py', 'app/detectors/audio/upgraded.py',
             'app/detectors/audio/content.py', 'app/detectors/audio/decode.py', 'app/detectors/video/adapters/audio.py',
             'app/detectors/video/audio.py', 'app/third_party/kaldi/fbank.py',
             'app/detectors/video/adapters/loaders.py', 'app/engine/profiles.py',
             'app/engine/fusion/artefact.py', 'app/engine/fusion/features.py',
             'app/engine/fusion/release/fuser-audio-2.1.1.json',
             'app/engine/fusion/release/fuser-audio-2.1.1.boot.npy']
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry
    verified = {}
    for model_id in sorted(ids):
        paths = ensure_verified(get_registry().manifest.model(model_id), force=True)
        verified[model_id] = [{'path': str(path), 'sha256': file_digest(path)} for path in paths]
    record = {'schema': 'chipmunk.audio-evaluation-lock/1', 'manifest_sha256': manifest['sha256'],
        'protocol_sha256': digest(protocol.read_bytes()), 'frozen_at_unix': time.time(),
        'source_hashes': {name: digest((ROOT / name).read_bytes()) for name in files},
        'models': [entry for entry in registry['models'] if entry['id'] in ids],
        'verified_local_weight_files': verified,
        'cpu_threads': 4, 'production_profile': 'full', 'recorded_score_precision_decimals': 6,
        'scoring_pcm_contract': 'production-s16', 'detector_devices': {'old': OLD_DEVICE, 'new': 'cpu'},
        'systems': ['S0_old_service_score', 'S1_old_recalibrated', 'S2_new_calibrated', 'S3_nonnegative_combination'],
        'fit': {'combination_l2': .1, 'positive_slopes': True, 'calibration': 'class-balanced monotonic Platt',
                'effective_target_smoothing': 'counts of distinct original sources, not derivatives'},
        'missing_policy': 'Each single detector abstains on its own gate/failure; combination abstains if either missing.',
        'threshold_rule': 'Only cal real sources with all three conditions assessed; max probability per source; rank ceil((n+1)*.95); nextafter(order_stat,+inf) handles ties. Rank above n is unestablished. Nominal target, no guarantee.',
        'reliability_bins': '10 fixed equal-width bins [0,.1,...,1]',
        's0_scope': 'Frozen Fuser.apply probability_display after detector applicability; score-only, not full API conflict/eligibility conclusion.',
        'timing_scope': 'Devices pinned per detector; CPU 4 threads; model loading excluded; first-call warmup included; no concurrent heavy work intended.',
        'bootstrap': {'replicates': 2000, 'seed': 304919, 'unit': 'speaker/voice, all source conditions carried together'},
        'deployment_eligible': False}
    save_json(OUT / 'run-lock.json', record)
    print('Run locked:', digest((OUT / 'run-lock.json').read_bytes()))


def score():
    import gc
    from types import SimpleNamespace

    import torch

    from app.detectors.audio import content
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import load_speech_activity, load_voice
    from app.detectors.video.adapters.audio import GustkingVoiceDetector, PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.detectors.video.adapters.loaders import load_audio_classifier
    from app.engine.detector import RunContext, StageOutputs
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    manifest = read_manifest()
    locked = json.loads((OUT / 'run-lock.json').read_bytes())
    if locked.get('scoring_pcm_contract') != 'production-s16':
        raise ValueError('Do not mix production PCM with the historical FLOAT run. Restore frozen run-source in an isolated copy for historical replay.')
    if locked['manifest_sha256'] != manifest['sha256']:
        raise ValueError('Run lock refers to a different manifest')
    for name, sha in locked['source_hashes'].items():
        if digest((ROOT / name).read_bytes()) != sha:
            raise ValueError('Scoring implementation changed after freeze: ' + name)
    if digest((ROOT.parent / 'docs/independent-evaluation-protocol.md').read_bytes()) != locked['protocol_sha256']:
        raise ValueError('Protocol changed after freeze')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    output = OUT / 'scores.jsonl'
    previous = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    complete = {(item['variant_id'], item['detector']) for item in previous}
    deadline = SimpleNamespace(check=lambda label='': None)
    writer = SimpleNamespace(series=lambda *args, **kwargs: 'offline-series')
    for key, detector in [('old', GustkingVoiceDetector()), ('new', PellaVoiceDetector())]:
        remaining = [row for row in manifest['variants'] if (row['id'], key) not in complete]
        if not remaining:
            continue
        print(f'Loading {key} detector; {len(remaining)} inputs remain.', flush=True)
        device = locked['detector_devices'][key]
        model_ids = [detector.MODEL_ID] + (['silero.speech_activity'] if key == 'new' else [])
        for model_id in model_ids:
            entry = get_registry().manifest.model(model_id)
            actual = [{'path': str(p), 'sha256': file_digest(p)} for p in ensure_verified(entry, force=True)]
            if actual != locked['verified_local_weight_files'][model_id]:
                raise ValueError('Model pins changed after freeze: ' + model_id)
        if key == 'old':
            models = {detector.MODEL_ID: load_audio_classifier(detector.MODEL_ID, device, 'float32')}
        else:
            models = {detector.MODEL_ID: load_voice(detector.MODEL_ID, 'cpu', 'float32'),
                      'silero.speech_activity': load_speech_activity('silero.speech_activity', 'cpu', 'float32')}
        lease = SimpleNamespace(get=models.__getitem__)
        for index, row in enumerate(remaining):
            path = OUT / row['path']
            if digest(path.read_bytes()) != row['sha256']:
                raise ValueError('Input differs from frozen manifest')
            decoded = decode_audio(path, source_sha256=row['sha256'])
            pcm = decoded.pcm
            if digest(pcm.astype('<f4').tobytes()) != row['pcm_sha256']:
                raise ValueError('Decoded input differs from frozen PCM')
            gate = content.classify(pcm).as_json()
            track = AudioTrack('analysed', pcm, len(pcm), gate=tuple(tuple(span) for span in gate['speech_spans']))
            ctx = RunContext(analysis_id='offline-evaluation', media=None, profile='full', device=device, dtype='float32',
                             lease=None, outputs=StageOutputs(audio=track), deadline=deadline, exhibits=writer, log=None)
            started = time.perf_counter()
            record = {'variant_id': row['id'], 'base_id': row['base_id'], 'detector': key,
                      'split': row['split'], 'condition': row['condition'], 'label': row['label'], 'speaker': row['speaker'],
                      'input_sha256': row['sha256'], 'production_input': decoded.mapping, 'device': device,
                      'heuristic_assessable': gate['assessable'],
                      'gate': 'original acoustic heuristic' if key == 'old' else 'independent learned speech activity'}
            try:
                applicable = detector.applicable(ctx)
                if not applicable.ok:
                    record.update(status='not_applicable', reason=applicable.reason, score=None, windows=[])
                else:
                    ctx.lease = lease
                    result = detector.run(ctx)
                    record.update(status=result.status, reason=result.reason,
                        score=None if result.aggregate is None else round(float(result.aggregate.raw), 6),
                        raw_unrounded=None if result.aggregate is None else float(result.aggregate.raw),
                        windows=[{'start_sample': round(unit.quality['start_sample']), 'end_sample': round(unit.quality['end_sample']),
                                  'score': round(float(unit.raw), 6), 'raw_unrounded': float(unit.raw)} for unit in result.unit_scores])
            except Exception as error:
                record.update(status='failed', reason=f'{type(error).__name__}: {error}', score=None, windows=[])
            record['latency_s'] = time.perf_counter() - started
            with output.open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
            if index % 10 == 0 or index + 1 == len(remaining):
                print(f'{key}: {index + 1}/{len(remaining)} inputs completed; values retained for locked evaluation.', flush=True)
        del ctx, lease, models
        gc.collect()
    print('Scoring completed; holdout results have not been used to fit candidates.', flush=True)


SYSTEMS = ['S0_old_service_score', 'S1_old_recalibrated', 'S2_new_calibrated', 'S3_nonnegative_combination']


def logit(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def source_weights(rows, *, balanced=True):
    """One unit per source, regardless of available condition count; each class totals 0.5."""
    counts = Counter(row['base_id'] for row in rows)
    result = np.array([1 / counts[row['base_id']] for row in rows], dtype=float)
    labels = np.array([row['label'] for row in rows])
    if balanced:
        for label in (0, 1):
            mask = labels == label
            if mask.any():
                result[mask] *= .5 / result[mask].sum()
    return result


def fit_monotonic(features, rows, *, calibration):
    """Predeclared weighted logistic objective: Platt-smoothed targets or fixed L2=.1 stacker."""
    from scipy.optimize import minimize
    from scipy.special import expit
    x = np.asarray(features, dtype=float).reshape(len(rows), -1)
    y = np.array([row['label'] for row in rows])
    counts = [len({r['base_id'] for r in rows if r['label'] == label}) for label in (0, 1)]
    if min(counts) < 2 or not np.isfinite(x).all():
        raise ValueError('Insufficient complete source support for fitting')
    weights = source_weights(rows)
    target = np.where(y == 1, (counts[1] + 1) / (counts[1] + 2), 1 / (counts[0] + 2)) if calibration else y
    penalty = 0.0 if calibration else .1

    def objective(theta):
        eta = x @ theta[:-1] + theta[-1]
        loss = np.sum(weights * (np.logaddexp(0, eta) - target * eta)) + .5 * penalty * np.sum(theta[:-1] ** 2)
        residual = weights * (expit(eta) - target)
        gradient = np.r_[x.T @ residual + penalty * theta[:-1], residual.sum()]
        return float(loss), gradient

    fitted = minimize(objective, np.r_[np.ones(x.shape[1]), 0.0], jac=True, method='L-BFGS-B',
        bounds=[(0, None)] * x.shape[1] + [(None, None)], options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9})
    if not fitted.success:
        raise ValueError('Fixed optimizer failed: ' + fitted.message)
    return {'slopes': fitted.x[:-1].tolist(), 'intercept': float(fitted.x[-1]), 'source_n_by_class': counts,
            'condition_rows': len(rows), 'objective': float(fitted.fun), 'iterations': int(fitted.nit),
            'penalty_l2': penalty, 'calibration': calibration}


def linear(features, parameters):
    return np.asarray(features) @ np.asarray(parameters['slopes']) + parameters['intercept']


def threshold_from_cal(rows, probabilities):
    """Conservative rank using complete real originals, never the number of derivatives."""
    grouped = defaultdict(list)
    for row, value in zip(rows, probabilities, strict=True):
        if row['label'] == 0:
            grouped[row['base_id']].append((row['condition'], value))
    maxima = []
    for records in grouped.values():
        if {c for c, _ in records} == {'clean', 'noise', 'music'} and all(np.isfinite(v) for _, v in records):
            maxima.append(max(v for _, v in records))
    n = len(maxima)
    rank = math.ceil((n + 1) * .95)
    established = n > 0 and rank <= n
    order_stat = float(sorted(maxima)[rank - 1]) if established else None
    return {'threshold': float(np.nextafter(order_stat, np.inf)) if established else None,
            'order_statistic': order_stat, 'n_complete_real_sources': n, 'rank': rank,
            'n_real_sources_total': len(grouped), 'nominal_fpr': .05,
            'comparison': 'probability >= threshold; strict above order statistic, including ties',
            'guarantee': False}


def fit_candidates(rows):
    """No test access: this function rejects holdout rows even if supplied accidentally."""
    if any(r['split'] == 'test' for r in rows):
        raise ValueError('Holdout rows cannot be passed to fitting')
    result = {}
    for key, system in [('old', SYSTEMS[1]), ('new', SYSTEMS[2])]:
        selected = [r for r in rows if r['split'] == 'cal' and np.isfinite(r[key])]
        try:
            result[system] = {'platt': fit_monotonic(logit([r[key] for r in selected]), selected, calibration=True)}
        except ValueError as error:
            result[system] = {'failed': str(error)}
    train = [r for r in rows if r['split'] == 'fit' and np.isfinite(r['old']) and np.isfinite(r['new'])]
    cal = [r for r in rows if r['split'] == 'cal' and np.isfinite(r['old']) and np.isfinite(r['new'])]
    try:
        stacker = fit_monotonic(logit([[r['old'], r['new']] for r in train]), train, calibration=False)
        cal_eta = linear(logit([[r['old'], r['new']] for r in cal]), stacker)
        result[SYSTEMS[3]] = {'stacker': stacker, 'platt': fit_monotonic(cal_eta, cal, calibration=True)}
    except ValueError as error:
        result[SYSTEMS[3]] = {'failed': str(error)}
    return result


def predict_candidates(rows, fitted, fuser):
    from scipy.special import expit

    from app.engine.fusion.features import Context
    probabilities = {system: np.full(len(rows), np.nan) for system in SYSTEMS}
    unclamped = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        if np.isfinite(row['old']):
            prediction = fuser.apply({'aud.gustking_xlsr': (row['old'], None)}, Context(modality='audio'))
            probabilities[SYSTEMS[0]][i] = prediction.probability_display
            unclamped[i] = prediction.probability
        for key, system in [('old', SYSTEMS[1]), ('new', SYSTEMS[2])]:
            if np.isfinite(row[key]) and 'platt' in fitted[system]:
                probabilities[system][i] = expit(linear([float(logit(row[key]))], fitted[system]['platt']))
        parameters = fitted[SYSTEMS[3]]
        if np.isfinite(row['old']) and np.isfinite(row['new']) and 'platt' in parameters:
            eta = linear(logit([row['old'], row['new']]), parameters['stacker'])
            probabilities[SYSTEMS[3]][i] = expit(linear([eta], parameters['platt']))
    return probabilities, unclamped


def metric_values(labels, probabilities, weights, threshold=None):
    """Missing values count in operational denominators, never as negative classifications."""
    from sklearn.metrics import roc_auc_score, roc_curve
    y, p, w = np.asarray(labels), np.asarray(probabilities), np.asarray(weights)
    assessed = np.isfinite(p)
    real, fake = y == 0, y == 1
    ai, original = assessed & (p >= .8), assessed & (p <= .2)
    decided = ai | original
    def rate(mask, denominator):
        total = w[denominator].sum()
        return float(w[mask & denominator].sum() / total) if total else None
    result = {'assessed_fraction': rate(assessed, np.ones(len(y), bool)),
        'abstention_fraction': rate(~assessed, np.ones(len(y), bool)),
        'decision_coverage_80': rate(decided, np.ones(len(y), bool)),
        'false_ai_80': rate(ai, real), 'false_original_80': rate(original, fake),
        'recall_ai_80': rate(ai, fake), 'recall_original_80': rate(original, real),
        'decided_risk_80': rate((ai & real) | (original & fake), decided),
        'ai_decision_risk_80': rate(real, ai), 'original_decision_risk_80': rate(fake, original),
        'fpr_at_cal_5': None, 'tpr_at_cal_5': None}
    if threshold is not None:
        result.update(fpr_at_cal_5=rate(assessed & (p >= threshold), real), tpr_at_cal_5=rate(assessed & (p >= threshold), fake))
    if not assessed.any() or len(np.unique(y[assessed])) != 2:
        return dict(result, auc=None, eer=None, brier=None, log_loss=None, ece=None)
    ya, pa, wa = y[assessed], p[assessed], w[assessed].copy()
    # Conditional calibrated metrics use the same 50:50 reference prior for both systems.
    for label in (0, 1):
        mask = ya == label
        wa[mask] *= .5 / wa[mask].sum()
    clipped = np.clip(pa, 1e-12, 1 - 1e-12)
    fpr, tpr, _ = roc_curve(ya, pa, sample_weight=wa, drop_intermediate=False)
    difference = fpr - (1 - tpr)
    high = int(np.flatnonzero(difference >= 0)[0])
    if high == 0:
        eer = float(fpr[0])
    else:
        low = high - 1
        fraction = -difference[low] / (difference[high] - difference[low])
        eer = float(fpr[low] + fraction * (fpr[high] - fpr[low]))
    ece = 0.0
    for b in range(10):
        mask = (pa >= b / 10) & ((pa < (b + 1) / 10) if b < 9 else (pa <= 1))
        if mask.any():
            ece += abs(np.sum(wa[mask] * (pa[mask] - ya[mask])))
    return dict(result, auc=float(roc_auc_score(ya, pa, sample_weight=wa)), eer=eer,
        brier=float(np.sum(wa * (pa - ya) ** 2)),
        log_loss=float(-np.sum(wa * (ya * np.log(clipped) + (1 - ya) * np.log1p(-clipped)))), ece=float(ece))


def reliability(rows, probabilities):
    selected = [i for i, p in enumerate(probabilities) if np.isfinite(p)]
    if not selected:
        return []
    subset = [rows[i] for i in selected]
    w = source_weights(rows)[selected]
    p = np.asarray(probabilities)[selected]
    y = np.array([r['label'] for r in subset])
    for label in (0, 1):
        mask = y == label
        if mask.any():
            w[mask] *= .5 / w[mask].sum()
    bins = []
    for b in range(10):
        mask = (p >= b / 10) & ((p < (b + 1) / 10) if b < 9 else (p <= 1))
        if mask.any():
            bins.append({'bounds': [b / 10, (b + 1) / 10], 'condition_rows': int(mask.sum()),
                'base_files': len({subset[i]['base_id'] for i in np.flatnonzero(mask)}),
                'weighted_mass': float(w[mask].sum()), 'mean_probability': float(np.average(p[mask], weights=w[mask])),
                'observed_ai_fraction': float(np.average(y[mask], weights=w[mask]))})
    return bins


def grouped_indices(rows, rng):
    """Class-stratified cluster bootstrap; a whole voice carries every source and condition."""
    selected = []
    for label in (0, 1):
        groups = sorted({r['speaker'] for r in rows if r['label'] == label})
        by_group = {g: [i for i, r in enumerate(rows) if r['speaker'] == g] for g in groups}
        for group in rng.choice(groups, size=len(groups), replace=True):
            selected.extend(by_group[group])
    return np.asarray(selected, dtype=int)


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z / total
    centre = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return {'events': successes, 'n_originals': total, 'low': max(0., centre-half), 'high': min(1., centre+half)}


def summarize_cohort(rows, probabilities, thresholds, *, bootstrap=True):
    labels = np.array([r['label'] for r in rows])
    weights = source_weights(rows)
    table = {system: metric_values(labels, p, weights, thresholds[system]['threshold']) for system, p in probabilities.items()}
    counts = {}
    source_rate_ci95 = {}
    for system, p in probabilities.items():
        assessed = np.isfinite(p)
        counts[system] = {'all_condition_rows': len(rows), 'base_files': len({r['base_id'] for r in rows}),
            'groups': len({r['speaker'] for r in rows}), 'assessed': int(assessed.sum()),
            'ai_80': int((assessed & (p >= .8)).sum()), 'original_80': int((assessed & (p <= .2)).sum()),
            'inconclusive_80': int((assessed & (p > .2) & (p < .8)).sum()), 'missing': int((~assessed).sum()),
            'false_ai_80': int((assessed & (p >= .8) & (labels == 0)).sum()),
            'false_original_80': int((assessed & (p <= .2) & (labels == 1)).sum())}
        # Never apply binomial intervals to pooled dependent variants.
        if len({r['base_id'] for r in rows}) == len(rows):
            real, fake = labels == 0, labels == 1
            threshold = thresholds[system]['threshold']
            source_rate_ci95[system] = {
                'false_ai_80': wilson(counts[system]['false_ai_80'], int(real.sum())),
                'false_original_80': wilson(counts[system]['false_original_80'], int(fake.sum())),
                'fpr_at_cal_5': None if threshold is None else wilson(int((assessed & (p >= threshold) & real).sum()), int(real.sum())),
                'tpr_at_cal_5': None if threshold is None else wilson(int((assessed & (p >= threshold) & fake).sum()), int(fake.sum()))}
    pairs = [(SYSTEMS[2], SYSTEMS[1]), (SYSTEMS[3], SYSTEMS[1]), (SYSTEMS[2], SYSTEMS[3])]
    difference = {}
    for a, b in pairs:
        difference[a + '_minus_' + b] = {key: None if table[a][key] is None or table[b][key] is None else table[a][key] - table[b][key] for key in table[a]}
    intervals, pair_intervals = {}, {}
    if bootstrap:
        rng = np.random.default_rng(304919)
        distributions = {system: defaultdict(list) for system in SYSTEMS}
        pair_distributions = {a + '_minus_' + b: defaultdict(list) for a, b in pairs}
        for _ in range(2000):
            index = grouped_indices(rows, rng)
            calculated = {system: metric_values(labels[index], p[index], weights[index], thresholds[system]['threshold']) for system, p in probabilities.items()}
            for system, values in calculated.items():
                for key, value in values.items():
                    if value is not None:
                        distributions[system][key].append(value)
            for a, b in pairs:
                for key in calculated[a]:
                    if calculated[a][key] is not None and calculated[b][key] is not None:
                        pair_distributions[a + '_minus_' + b][key].append(calculated[a][key] - calculated[b][key])
        def ci(values):
            return {'low': float(np.percentile(values, 2.5)), 'high': float(np.percentile(values, 97.5)),
                    'valid_replicates': len(values), 'undefined_replicates': 2000 - len(values)} if values else None
        intervals = {system: {key: ci(dist.get(key, [])) for key in table[system]} for system, dist in distributions.items()}
        pair_intervals = {pair: {key: ci(dist.get(key, [])) for key in difference[pair]} for pair, dist in pair_distributions.items()}
    return {'counts': counts, 'metrics': table, 'ci95': intervals, 'source_rate_wilson_ci95': source_rate_ci95, 'paired_difference': difference,
            'paired_ci95': pair_intervals, 'reliability': {s: reliability(rows, p) for s, p in probabilities.items()}}


def analyse():
    from app.engine.fusion.artefact import Fuser
    manifest = read_manifest()
    locked = json.loads((OUT / 'run-lock.json').read_bytes())
    for name, sha in locked['source_hashes'].items():
        if digest((ROOT / name).read_bytes()) != sha:
            raise ValueError('Frozen implementation changed: ' + name)
    if digest((ROOT.parent / 'docs/independent-evaluation-protocol.md').read_bytes()) != locked['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    raw = [json.loads(line) for line in (OUT / 'scores.jsonl').read_text().splitlines()]
    by_key = {(r['variant_id'], r['detector']): r for r in raw}
    if len(raw) != len(by_key) or len(raw) != 2 * len(manifest['variants']):
        raise ValueError('Scoring incomplete or duplicate')
    rows = []
    for variant in manifest['variants']:
        row = dict(variant)
        for key in ('old', 'new'):
            record = by_key[(variant['id'], key)]
            if record['input_sha256'] != variant['sha256']:
                raise ValueError('A scored input is outside the manifest')
            row[key] = record['score'] if record['status'] == 'ok' and record['score'] is not None else np.nan
        rows.append(row)
    fitted = fit_candidates([r for r in rows if r['split'] != 'test'])
    fuser = Fuser.load(ROOT / 'app/engine/fusion/release/fuser-audio-2.1.1.json')
    if fuser is None:
        raise ValueError('Frozen S0 unavailable')
    probabilities, unclamped = predict_candidates(rows, fitted, fuser)
    cal_index = [i for i, row in enumerate(rows) if row['split'] == 'cal']
    thresholds = {s: threshold_from_cal([rows[i] for i in cal_index], p[cal_index]) for s, p in probabilities.items()}
    candidate = {'deployment_eligible': False, 'manifest_sha256': manifest['sha256'], 'run_lock_sha256': digest((OUT / 'run-lock.json').read_bytes()),
        'parameters': fitted, 'thresholds': thresholds, 'scope': manifest['scope'], 'reference_prior': .5}
    save_json(OUT / 'offline-calibration.json', candidate)
    access = {'manifest_sha256': manifest['sha256'], 'scores_sha256': digest((OUT / 'scores.jsonl').read_bytes()),
              'candidate_sha256': digest((OUT / 'offline-calibration.json').read_bytes()),
              'note': 'Single predeclared holdout evaluation; identical deterministic reruns allowed, no retuning.'}
    save_json(OUT / 'holdout-access.json', access)
    result = {'schema': 'chipmunk.audio-evaluation-result/1', **access, 'deployment_eligible': False,
        'scope': manifest['scope'], 'group_audit': manifest['group_audit'], 'systems': SYSTEMS,
        'thresholds': thresholds, 'cohorts': {},
        'uncertainty': '2000 class-stratified paired speaker/voice cluster bootstrap replicates, seed304919. Intervals conditional on selected background recordings. Fit/calibration uncertainty not refitted in test bootstrap.',
        'limitations': ['Known LibriSpeech source-family overlap with old encoder pretraining; candidate training provenance unknown.',
            'Only 40 real and 40 AI held-out originals, one unseen generator family, English normalized four-second inputs.',
            'S0 uses exact frozen service Fuser.apply probability_display, not full API conflict/eligibility conclusion.',
            'Four-second inputs each contain only one production window. This does not calibrate long-file aggregation.',
            'FPR threshold is an empirical nominal target, not a finite-sample guarantee or observed test rate.',
            'Shared noise/music within each split; dependent derivatives never count as extra independent originals.']}
    for condition in ('all', 'clean', 'noise', 'music'):
        index = np.array([i for i, row in enumerate(rows) if row['split'] == 'test' and (condition == 'all' or row['condition'] == condition)])
        cohort = [rows[i] for i in index]
        values = {s: p[index] for s, p in probabilities.items()}
        print('Evaluating locked holdout cohort:', condition, flush=True)
        result['cohorts'][condition] = summarize_cohort(cohort, values, thresholds)
        shared = np.array([i for i in index if np.isfinite(rows[i]['old']) and np.isfinite(rows[i]['new'])])
        result['cohorts'][condition]['shared_assessable'] = (summarize_cohort([rows[i] for i in shared], {s: p[shared] for s, p in probabilities.items()}, thresholds, bootstrap=len(shared) != len(index)) if len(shared) else None)
        if len(shared) == len(index):
            result['cohorts'][condition]['shared_assessable']['ci95'] = result['cohorts'][condition]['ci95']
            result['cohorts'][condition]['shared_assessable']['paired_ci95'] = result['cohorts'][condition]['paired_ci95']
        raw_values = {key: metric_values([r['label'] for r in cohort], [r[key] for r in cohort], source_weights(cohort)) for key in ('old', 'new')}
        result['cohorts'][condition]['raw_ranking'] = {key: {k: v for k, v in values.items() if k in ('auc', 'eer', 'assessed_fraction')} for key, values in raw_values.items()}
        result['cohorts'][condition]['failures'] = {key: dict(Counter(r['status'] + ':' + str(r['reason']) for r in raw if r['detector'] == key and r['split'] == 'test' and (condition == 'all' or r['condition'] == condition) and r['status'] != 'ok')) for key in ('old', 'new')}
    result['latency'] = {key: {'n': len([r for r in raw if r['detector'] == key]),
        'median_s': float(np.median([r['latency_s'] for r in raw if r['detector'] == key])),
        'p95_s': float(np.percentile([r['latency_s'] for r in raw if r['detector'] == key], 95)),
        'total_s': float(sum(r['latency_s'] for r in raw if r['detector'] == key))} for key in ('old', 'new')}
    predictions = [{'id': r['id'], 'base_id': r['base_id'], 'split': r['split'], 'condition': r['condition'], 'label': r['label'],
        'probabilities': {s: float(p[i]) if np.isfinite(p[i]) else None for s, p in probabilities.items()},
        's0_unclamped': float(unclamped[i]) if np.isfinite(unclamped[i]) else None} for i, r in enumerate(rows)]
    save_json(OUT / 'predictions.json', predictions)
    save_json(OUT / 'evaluation.json', result)
    print('Offline evaluation saved; no runtime model or calibration was changed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'lock', 'score', 'analyse'])
    parser.add_argument('--output', type=Path, default=OUT, help='Fresh storage only: unchanged seed/selection reuses the consumed holdout. This is not new independent data; existing artifacts are never overwritten.')
    parser.add_argument('--old-device', choices=['cpu', 'mps'], default='cpu', help='Pinned at lock; use the service device for production-path evaluation.')
    args = parser.parse_args()
    OUT = args.output.resolve()
    OLD_DEVICE = args.old_device
    {'prepare': prepare, 'lock': lock, 'score': score, 'analyse': analyse}[args.command]()
