"""Prospective, isolated audio calibration pilot; never writes a serving artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import evaluate_audio_candidates as base

ROOT = base.ROOT
OUT = ROOT / 'data/independent-evaluation/audio-v2'
OLD = ROOT / 'data/independent-evaluation/audio'
PROTOCOL = ROOT.parent / 'docs/audio-calibration-v2-protocol.md'
SEED = 'chipmunk-audio-pilot-20261101-v2'
TARGETS = {'fit': 8, 'cal': 24, 'test': 20}
REVISIONS = dict(base.REVISIONS, **{'jspaulsen/vctk': 'fb74847570d78d2b23e83193d8e55df80e6271b2'})
base.OUT, base.SEED, base.REVISIONS = OUT, SEED, REVISIONS
save, sha, stable = base.save_json, base.digest, base.stable
EXCLUSIONS = []
CONDITIONS = ['clean', 'noise_5', 'noise_15', 'music_5', 'music_15']


def old_manifest():
    return json.loads((OLD / 'manifest.json').read_bytes())


def exclusion_sets():
    old = old_manifest()
    return {key: {r[key] for r in old['base_files']} for key in
            ('speaker', 'source_group', 'transcript_sha256', 'sha256', 'decoded_pcm_sha256')}


def metadata():
    """Only metadata and rights statements; no classifier is imported or scored."""
    import fsspec
    import pyarrow.parquet as pq
    OUT.mkdir(exist_ok=True, parents=True)
    for dataset in ('jspaulsen/vctk', 'ductai199x/dictation-speech', 'corypaik/musan'):
        publisher = base.get('https://huggingface.co/api/datasets/' + dataset).json()
        if publisher['sha'] != REVISIONS[dataset] or publisher.get('gated'):
            raise ValueError('Dataset revision/permission changed')
        save(OUT / 'metadata' / (dataset.replace('/', '_') + '-publisher.json'), publisher)
        card = base.get(f'https://huggingface.co/datasets/{dataset}/raw/{publisher["sha"]}/README.md').content
        (OUT / 'metadata' / (dataset.replace('/', '_') + '-README.md')).write_bytes(card)
    ds = 'jspaulsen/vctk'
    publisher = json.loads((OUT / 'metadata/jspaulsen_vctk-publisher.json').read_bytes())
    files = [x['rfilename'] for x in publisher['siblings'] if x['rfilename'].endswith('.parquet')]

    def read_index(name):
        path = OUT / 'metadata' / (Path(name).name + '.json')
        if not path.exists():
            url = f'https://huggingface.co/datasets/{ds}/resolve/{REVISIONS[ds]}/{name}'
            with fsspec.open(url, 'rb', block_size=65536) as stream:
                table = pq.ParquetFile(stream).read(columns=['id', 'text', 'speaker_id', 'mic_id']).to_pylist()
            save(path, table)
        return json.loads(path.read_bytes())

    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, records in zip(files, pool.map(read_index, files), strict=True):
            offset = len(rows)
            rows.extend(dict(r, row_idx=offset + i) for i, r in enumerate(records))
            print('Indexed', name, flush=True)
    save(OUT / 'metadata/vctk-index.json', rows)
    for offset in range(0, 600, 100):
        # Cached metadata contains no detector score; source audio is fetched afresh.
        previous = OLD / 'metadata' / f'ductai199x_dictation-speech-chatterbox-test-{offset}-100.json'
        target = OUT / 'metadata' / previous.name
        if not target.exists():
            target.write_bytes(previous.read_bytes())
    print('Metadata ready; no model inference performed.', flush=True)


def prepare():
    import fsspec
    import zipfile
    import soundfile as sf
    from app.detectors.audio.decode import decode_audio
    if (OUT / 'manifest.json').exists():
        print('Manifest already frozen.'); return
    denied = exclusion_sets()
    used = {k: set(v) for k, v in denied.items()}
    sources = []
    (OUT / 'sources').mkdir(exist_ok=True)
    (OUT / 'inputs').mkdir(exist_ok=True)
    candidates = [r for offset in range(0, 600, 100) for r in base.metadata_rows(
        'ductai199x/dictation-speech', 'chatterbox', 'test', offset)]
    voices = sorted({r['row']['tts_engine'] + ':' + r['row']['tts_voice'] for r in candidates}
                    - denied['speaker'], key=stable)
    if len(voices) != 26:
        raise ValueError('Prospective cohort expected exactly 26 unused donated voices')
    allocation = {'fit': voices[:4], 'cal': voices[4:16], 'test': voices[16:]}
    for split, identities in allocation.items():
        for speaker in identities:
            count = 0
            for item in sorted(candidates, key=lambda x: stable(x['row']['id'])):
                row = item['row']
                if row['tts_engine'] + ':' + row['tts_voice'] != speaker:
                    continue
                seconds = 4 if count == 0 else 8
                text_key = base.transcript_key(row['transcript'])
                group = row['source_url'] or row['rendition_id']
                if text_key in used['transcript_sha256'] or group in used['source_group']:
                    EXCLUSIONS.append({'id': row['id'], 'reason': 'previous/selected source or transcript'}); continue
                if row['duration_s'] < seconds:
                    EXCLUSIONS.append({'id': row['id'], 'reason': 'too short for assigned duration'}); continue
                ref, pcm, rate = base.download_audio(row, 'ductai199x/dictation-speech', 'chatterbox', 'test', item['row_idx'])
                ph = sha(pcm.astype('<f4').tobytes())
                if len(pcm) < seconds * 16000 or ref['sha256'] in used['sha256'] or ph in used['decoded_pcm_sha256']:
                    EXCLUSIONS.append({'id': row['id'], 'reason': 'short or duplicate bytes/PCM'}); continue
                source = dict(ref, id='fake:' + row['id'], split=split, label=1, speaker=speaker,
                    source_group=group, transcript_sha256=text_key, transcript=row['transcript'],
                    decoded_pcm_sha256=ph, generator=row['tts_engine'], seconds=seconds,
                    original_sample_rate=rate, license='CC-BY-SA-4.0', text_license=row['license'],
                    license_url='https://huggingface.co/datasets/ductai199x/dictation-speech',
                    label_source='Publisher declares all speech synthetic; donated-voice Chatterbox output.')
                sources.append(source)
                for key in used:
                    if key != 'speaker': used[key].add(source[key])
                count += 1
                if count == 2: break
            if count != 2:
                raise ValueError('Insufficient eligible fresh passages for frozen voice: ' + speaker)
            print('Prepared generated voice', speaker, split, flush=True)
    index = json.loads((OUT / 'metadata/vctk-index.json').read_bytes())
    archive_url = 'https://huggingface.co/datasets/isjwdu/DFADD/resolve/dfc1eeab3cb0068db8e87a2b89a1ebd103665b1f/DATASET_VCTK_BONAFIDE.zip'
    archive_stream = fsspec.open(archive_url, 'rb', block_size=65536).open()
    archive = zipfile.ZipFile(archive_stream)
    members = {Path(name).stem: name for name in archive.namelist() if name.endswith('.wav')}
    speakers = sorted({r['speaker_id'] for r in index}, key=stable)
    cursor = 0
    for split, target in TARGETS.items():
        selected = 0
        while selected < target:
            speaker = speakers[cursor]; cursor += 1
            desired = 4 if selected % 2 == 0 else 8
            options = []
            for duration in ([8, 4] if desired == 8 else [4]):
                options.extend((r, duration) for r in sorted([r for r in index
                    if r['speaker_id'] == speaker and r['mic_id'] == 'mic1'
                    and len(r['text'].split()) >= (8 if duration == 4 else 18)], key=lambda r: stable(r['id'])))
            for row, seconds in options:
                text_key = base.transcript_key(row['text'])
                group = 'vctk:' + row['id'].rsplit('_mic', 1)[0]
                if text_key in used['transcript_sha256'] or group in used['source_group']:
                    continue
                recording_id = row['id'].rsplit('_mic', 1)[0]
                if recording_id not in members:
                    EXCLUSIONS.append({'id': row['id'], 'reason': 'VCTK bona fide archive member absent'}); continue
                cache = OUT / 'metadata' / ('vctk-archive-' + recording_id + '.json')
                if cache.exists():
                    ref = json.loads(cache.read_bytes())
                else:
                    raw = archive.read(members[recording_id])
                    path = OUT/'sources'/(sha(raw)+'.audio'); path.write_bytes(raw)
                    ref = dict(path=str(path.relative_to(OUT)), sha256=sha(raw), bytes=len(raw), url=archive_url,
                        archive_member=members[recording_id], dataset='isjwdu/DFADD',
                        revision='dfc1eeab3cb0068db8e87a2b89a1ebd103665b1f',
                        transcript_dataset='jspaulsen/vctk', transcript_revision=REVISIONS['jspaulsen/vctk'],
                        transcript_row_index=row['row_idx'], microphone='not documented by archive publisher')
                    save(cache, ref)
                pcm = base.decode(OUT/ref['path']); rate = sf.info(OUT/ref['path']).samplerate
                ph = sha(pcm.astype('<f4').tobytes())
                if len(pcm) < seconds * 16000 or ref['sha256'] in used['sha256'] or ph in used['decoded_pcm_sha256']:
                    EXCLUSIONS.append({'id': row['id'], 'reason': 'short or duplicate bytes/PCM'}); continue
                source = dict(ref, id='real:' + row['id'], split=split, label=0, speaker='vctk:' + speaker,
                    source_group=group, transcript_sha256=text_key, transcript=row['text'], decoded_pcm_sha256=ph,
                    generator=None, seconds=seconds, original_sample_rate=rate, license='CC-BY-4.0',
                    license_url='https://datashare.ed.ac.uk/handle/10283/3443',
                    label_source='CSTR VCTK bona fide recordings in DFADD archive, matched by speaker/utterance ID.')
                sources.append(source)
                for key in used: used[key].add(source[key])
                selected += 1
                print('Prepared authentic speaker', speaker, split, seconds, flush=True)
                break
            else:
                EXCLUSIONS.append({'speaker': speaker, 'reason': 'no eligible duration/text candidate'})
    archive.close(); archive_stream.close()
    backgrounds = prepare_backgrounds()
    variants = []
    for source in sources:
        n = source['seconds'] * 16000
        clean = base.decode(OUT / source['path'])[:n].astype(np.float64)
        rms = np.sqrt(np.mean(clean**2))
        if rms < 1e-4: raise ValueError('Silent selected source')
        clean *= .1 / rms
        mixtures, selected = {'clean': clean}, {}
        for kind in ('noise', 'music'):
            pool = [b for b in backgrounds if b['split'] == source['split'] and b['kind'] == kind]
            ref = pool[int(stable(source['id'] + kind)[:8], 16) % len(pool)]
            bg = base.decode(OUT / ref['path']).astype(np.float64)
            start = int(stable(source['id'] + kind + ':offset')[:8], 16) % (len(bg) - n + 1)
            bg = bg[start:start + n]
            for snr in (5, 15):
                key = kind + '_' + str(snr)
                mixtures[key] = clean + bg * (.1 / np.sqrt(np.mean(bg**2)) / 10**(snr / 20))
                selected[key] = (ref, start, snr)
        gain = min(1., .98 / max(np.max(np.abs(v)) for v in mixtures.values()))
        clean_pcm = None
        for condition, mixture in mixtures.items():
            pcm = (mixture * gain).astype(np.float32)
            path = OUT / 'inputs' / (sha(pcm.tobytes()) + '.wav')
            sf.write(path, pcm, 16000, subtype='FLOAT')
            decoded = decode_audio(path, source_sha256=base.file_digest(path))
            if condition == 'clean': clean_pcm = decoded.pcm
            background = selected.get(condition)
            variants.append(dict(id=stable(source['id'] + condition), base_id=source['id'], split=source['split'],
                label=source['label'], speaker=source['speaker'], seconds=source['seconds'], condition=condition,
                path=str(path.relative_to(OUT)), sha256=base.file_digest(path), pcm_sha256=decoded.mapping['pcm_sha256'],
                production_input=decoded.mapping, samples=n, sample_rate=16000, source_interval_samples=[0, n],
                background_sha256=background[0]['sha256'] if background else None,
                background_start_sample=background[1] if background else None, snr_db=background[2] if background else None,
                common_peak_gain=gain, measured_snr_db=None if not background else float(20 * np.log10(
                    np.sqrt(np.mean(clean_pcm.astype(float)**2)) / np.sqrt(np.mean((decoded.pcm - clean_pcm).astype(float)**2))))))
    manifest = dict(schema='chipmunk.audio-pilot/2', seed=SEED, deployment_eligible=False,
        scope='Fresh source/speaker/transcript pilot; one previously observed AI family; controlled 4/8-second English mixtures.',
        original_study_manifest_sha256=old_manifest()['sha256'], protocol_sha256=base.file_digest(PROTOCOL),
        base_files=sources, backgrounds=backgrounds, variants=variants, exclusions=EXCLUSIONS)
    manifest['audit'] = audit(manifest)
    manifest['sha256'] = sha(base.canonical(manifest))
    save(OUT / 'manifest.json', manifest)
    print(json.dumps(manifest['audit'], indent=2), flush=True)


def prepare_backgrounds():
    backgrounds = []
    excluded = {b['sha256'] for b in old_manifest()['backgrounds']}
    licence_url = f'https://huggingface.co/datasets/corypaik/musan/raw/{REVISIONS["corypaik/musan"]}/licenses/noise/free-sound/LICENSE'
    declaration = base.get(licence_url).text
    if 'Public Domain' not in declaration: raise ValueError('Noise rights declaration changed')
    (OUT / 'metadata/noise-LICENSE.txt').write_text(declaration)
    for item in base.metadata_rows('corypaik/musan', 'noise', 'train', 0):
        row = item['row']
        if 'free-sound' not in row['path']: continue
        ref, pcm, _ = base.download_audio(row, 'corypaik/musan', 'noise', 'train', item['row_idx'])
        if ref['sha256'] in excluded or len(pcm) < 8 * 16000 or np.sqrt(np.mean(pcm**2)) < 1e-4: continue
        excluded.add(ref['sha256'])
        backgrounds.append(dict(ref, kind='noise', split=list(TARGETS)[len(backgrounds)//3],
            source_id=row['path'], license='Public-domain', license_url=licence_url))
        if len(backgrounds) == 9: break
    if len(backgrounds) != 9: raise ValueError('Insufficient fresh noise backgrounds')
    for split, filename, page, author in [('fit', 'Searching.ogg', 'searching', 'yd'),
        ('cal', '439015_somepin_cavernous.mp3', 'cavern', 'Sudocolon'),
        ('test', 'piano1.3.ogg', 'happy-moments', 'Centurion_of_war')]:
        url = 'https://opengameart.org/sites/default/files/' + filename
        licence_url = 'https://opengameart.org/content/' + page
        html = base.get(licence_url).text
        if 'CC0' not in html: raise ValueError('Music CC0 declaration absent')
        (OUT / 'metadata' / (page + '-rights.html')).write_text(html)
        raw = base.get(url).content
        path = OUT / 'sources' / (sha(raw) + '.audio'); path.write_bytes(raw)
        if len(base.decode(path)) < 8 * 16000: raise ValueError('Short background')
        backgrounds.append(dict(path=str(path.relative_to(OUT)), sha256=sha(raw), bytes=len(raw), url=url,
            kind='music', split=split, source_id=page, author=author, license='CC0-1.0', license_url=licence_url))
    return backgrounds


def audit(manifest):
    excluded = exclusion_sets()
    for key in excluded:
        splits = defaultdict(set)
        for row in manifest['base_files']:
            if row[key] in excluded[key]: raise ValueError('Previous study identity reused: ' + key)
            splits[row[key]].add(row['split'])
        if any(len(v) != 1 for v in splits.values()): raise ValueError('Identity crosses split: ' + key)
    for key, rows in [('pcm_sha256', manifest['variants']), ('sha256', manifest['backgrounds'])]:
        splits = defaultdict(set)
        for row in rows: splits[row[key]].add(row['split'])
        if any(len(v) != 1 for v in splits.values()): raise ValueError('Derived/background identity crosses split')
    counts = {}
    for split, target in TARGETS.items():
        selected = [r for r in manifest['base_files'] if r['split'] == split]
        if Counter(r['label'] for r in selected) != {0: target, 1: target}: raise ValueError('Cohort size differs')
        counts[split] = {str(c): {'files': sum(r['label'] == c for r in selected),
            'speakers': len({r['speaker'] for r in selected if r['label'] == c})} for c in (0, 1)}
    return dict(counts=counts, cross_split_overlap=0, previous_study_overlap=0,
        original_files=len(manifest['base_files']), variants=len(manifest['variants']),
        duration_counts=dict(Counter(r['seconds'] for r in manifest['base_files'])),
        training_overlap='Unknown detector training corpus; VCTK may be familiar. No claim of pretraining independence.',
        scope_limit='VCTK and synthetic source domains are confounded; single AI family already seen in previous study.')


def manifest():
    value = json.loads((OUT / 'manifest.json').read_bytes())
    if sha(base.canonical({k: v for k, v in value.items() if k != 'sha256'})) != value['sha256']:
        raise ValueError('Manifest modified')
    audit(value)
    return value


def lock():
    import yaml
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry
    m = manifest()
    if m['protocol_sha256'] != base.file_digest(PROTOCOL):
        raise ValueError('Protocol changed since source preparation')
    ids = ['hf.gustking.wav2vec2_xlsr_deepfake', 'pella.voice_v2', 'silero.speech_activity']
    verified = {key: [{'path': str(p), 'sha256': base.file_digest(p)} for p in
        ensure_verified(get_registry().manifest.model(key), force=True)] for key in ids}
    files = ['scripts/calibrate_audio_v2.py', 'scripts/evaluate_audio_candidates.py', 'app/detectors/audio/upgraded.py',
        'app/detectors/audio/content.py', 'app/detectors/audio/decode.py', 'app/detectors/video/adapters/audio.py',
        'app/detectors/video/audio.py', 'app/detectors/video/adapters/loaders.py', 'app/engine/profiles.py',
        'app/third_party/kaldi/fbank.py']
    save(OUT / 'run-lock.json', dict(schema='chipmunk.audio-pilot-lock/2', frozen_at_unix=time.time(),
        manifest_sha256=m['sha256'], protocol_sha256=base.file_digest(PROTOCOL),
        source_hashes={p: base.file_digest(ROOT/p) for p in files}, verified_local_weight_files=verified,
        detector_devices={'old': 'cpu', 'new': 'cpu'}, cpu_threads=4, production_profile='full', offline_model_execution=True,
        models=[r for r in yaml.safe_load((ROOT/'app/models.yaml').read_text())['models'] if r['id'] in ids],
        calibration_fit_split='fit', rejection_calibration_split='cal', rejection_alpha=.1,
        bootstrap_replicates=2000, bootstrap_seed=749031, deployment_eligible=False))
    print('Prospective lock:', base.file_digest(OUT/'run-lock.json'), flush=True)


def verify_lock():
    m = manifest(); frozen = json.loads((OUT/'run-lock.json').read_bytes())
    if frozen['manifest_sha256'] != m['sha256'] or frozen['protocol_sha256'] != base.file_digest(PROTOCOL):
        raise ValueError('Manifest/protocol changed after lock')
    for p, digest in frozen['source_hashes'].items():
        if base.file_digest(ROOT/p) != digest: raise ValueError('Implementation changed: '+p)
    return m, frozen


def valid_probability(value):
    return value is not None and np.isfinite(value) and 0 <= value <= 1


def score(stage='development'):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import gc
    from types import SimpleNamespace
    import torch
    from app.detectors.audio import content
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import load_speech_activity, load_voice
    from app.detectors.video.adapters.audio import GustkingVoiceDetector, PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.detectors.video.adapters.loaders import load_audio_classifier
    from app.detectors.video.audio import voice_windows, VOICE_WINDOW_S, MAX_VOICE_WINDOWS
    from app.engine.detector import RunContext, StageOutputs, Profile
    from app.engine.profiles import SAMPLING
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry
    m, frozen = verify_lock()
    if stage == 'test' and not (OUT/'calibration.json').exists():
        raise ValueError('Fit and freeze the calibration before running any test inference')
    chosen = {'fit', 'cal'} if stage == 'development' else {'test'}
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(0)
    output = OUT/'scores.jsonl'
    previous = [json.loads(s) for s in output.read_text().splitlines()] if output.exists() else []
    complete = {(r['variant_id'], r['detector']) for r in previous}
    for key, detector in [('old', GustkingVoiceDetector()), ('new', PellaVoiceDetector())]:
        remaining = [r for r in m['variants'] if r['split'] in chosen and (r['id'], key) not in complete]
        if not remaining: continue
        ids = [detector.MODEL_ID] + (['silero.speech_activity'] if key == 'new' else [])
        for model_id in ids:
            actual = [{'path': str(p), 'sha256': base.file_digest(p)} for p in
                ensure_verified(get_registry().manifest.model(model_id), force=True)]
            if actual != frozen['verified_local_weight_files'][model_id]: raise ValueError('Weight changed')
        print(f'Loading {key}: {len(remaining)} {stage} inputs.', flush=True)
        models = {detector.MODEL_ID: load_audio_classifier(detector.MODEL_ID, 'cpu', 'float32')} if key == 'old' else {
            detector.MODEL_ID: load_voice(detector.MODEL_ID, 'cpu', 'float32'),
            'silero.speech_activity': load_speech_activity('silero.speech_activity', 'cpu', 'float32')}
        lease = SimpleNamespace(get=models.__getitem__)
        for index, row in enumerate(remaining):
            decoded = decode_audio(OUT/row['path'], source_sha256=row['sha256'])
            if decoded.mapping['pcm_sha256'] != row['pcm_sha256']: raise ValueError('PCM differs from manifest')
            gate = content.classify(decoded.pcm).as_json()
            track = AudioTrack('analysed', decoded.pcm, len(decoded.pcm), gate=tuple(tuple(s) for s in gate['speech_spans']))
            ctx = RunContext(analysis_id='audio-calibration-v2', media=None, profile='full', device='cpu', dtype='float32',
                lease=None, outputs=StageOutputs(audio=track), deadline=SimpleNamespace(check=lambda label='': None),
                exhibits=SimpleNamespace(series=lambda *a, **k: 'offline-series'), log=None)
            record = dict(variant_id=row['id'], base_id=row['base_id'], split=row['split'], label=row['label'],
                speaker=row['speaker'], detector=key, condition=row['condition'], device='cpu', seconds=row['seconds'],
                input_sha256=row['sha256'], production_input=decoded.mapping, heuristic_assessable=gate['assessable'])
            started = time.perf_counter()
            try:
                spans = gate['speech_spans'] if key == 'old' else models['silero.speech_activity'].obj.spans(decoded.pcm)
                params = SAMPLING[Profile('full')]
                proposed = voice_windows(decoded.pcm, params.voice_hop_s or VOICE_WINDOW_S,
                    max_windows=params.audio_windows or MAX_VOICE_WINDOWS)
                record['gate_input'] = gate
                record['applied_speech_spans_s'] = [list(s) for s in spans]
                record['proposed_windows'] = [dict(start_sample=round(start*16000),
                    end_sample=round(start*16000)+len(chunk),
                    accepted_by_speech_gate=content.window_in_speech(spans, start, start+len(chunk)/16000))
                    for start, chunk in proposed]
                record['gate_audit_latency_s'] = time.perf_counter()-started
                started = time.perf_counter()
                applicable = detector.applicable(ctx)
                if not applicable.ok:
                    record.update(status='not_applicable', reason=applicable.reason, score=None, windows=[])
                else:
                    ctx.lease = lease
                    result = detector.run(ctx)
                    if result.aggregate is not None and (not valid_probability(result.aggregate.raw)
                            or not result.unit_scores or any(not valid_probability(u.raw) for u in result.unit_scores)):
                        raise ValueError('Nonfinite or out-of-range production score')
                    record.update(status=result.status, reason=result.reason,
                        score=None if result.aggregate is None else round(float(result.aggregate.raw), 6),
                        raw_unrounded=None if result.aggregate is None else float(result.aggregate.raw),
                        windows=[{'start_sample': round(u.quality['start_sample']), 'end_sample': round(u.quality['end_sample']),
                                  'score': round(float(u.raw), 6), 'raw_unrounded': float(u.raw)} for u in result.unit_scores])
                    if result.aggregate is not None:
                        if abs(np.mean([u.raw for u in result.unit_scores])-result.aggregate.raw) > 1e-10:
                            raise ValueError('Production aggregate does not match its windows')
            except Exception as error:
                record.update(status='failed', reason=f'{type(error).__name__}: {error}', score=None,
                              raw_unrounded=None, windows=[])
            record['latency_s'] = time.perf_counter()-started
            with output.open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
            if index % 20 == 0 or index+1 == len(remaining): print(f'{key}: {index+1}/{len(remaining)}', flush=True)
        del models, lease, ctx
        gc.collect()


def probability(raw, parameters):
    from scipy.special import expit
    return float(expit(base.linear([float(base.logit(raw))], parameters))) if raw is not None else None


def calibrate_guard(rows, key):
    """Independent calibration identities, worst score; missing retains infinity."""
    groups = {0: defaultdict(list), 1: defaultdict(list)}
    for row in rows:
        if row['split'] != 'cal': raise ValueError('Rejection calibration accepts cal only')
        p = row[key]
        groups[row['label']][row['speaker']].append(None if p is None else (p if row['label'] == 0 else 1-p))
    return {str(label): [{'speaker': speaker, 'maximum': None if any(v is None for v in values) else max(values),
                          'conditions': len(values)} for speaker, values in sorted(by_identity.items())]
            for label, by_identity in groups.items()}


def decision(p, guard=None):
    if p is None or not np.isfinite(p): return 'inconclusive'
    lean = 'ai' if p >= .8 else ('original' if p <= .2 else 'inconclusive')
    if guard is None: return lean
    if min(len(guard['0']), len(guard['1'])) < 9: return 'inconclusive'
    included = []
    for label in (0, 1):
        nonconformity = p if label == 0 else 1-p
        records = guard[str(label)]
        pv = (1 + sum(r['maximum'] is None or r['maximum'] >= nonconformity for r in records))/(len(records)+1)
        if pv > .1: included.append(label)
    expected = [1] if lean == 'ai' else [0]
    return lean if lean != 'inconclusive' and included == expected else 'inconclusive'


def read_scores(split=None):
    m, _ = verify_lock()
    raw = {}
    for line in (OUT/'scores.jsonl').read_text().splitlines():
        r = json.loads(line)
        if split is not None and r['split'] not in split: continue
        key = r['variant_id'], r['detector']
        if key in raw: raise ValueError('Duplicate detector score')
        raw[key] = r
    rows = []
    for v in m['variants']:
        if split is not None and v['split'] not in split: continue
        row = dict(v)
        for key in ('old', 'new'):
            result = raw[(v['id'], key)]
            if (result['input_sha256'] != v['sha256'] or result['production_input']['pcm_sha256'] != v['pcm_sha256']
                    or result['split'] != v['split'] or result['label'] != v['label'] or result['base_id'] != v['base_id']):
                raise ValueError('Score record does not match frozen input identity')
            if any(not 0 <= w['start_sample'] < w['end_sample'] <= v['samples'] for w in result['windows']):
                raise ValueError('Scored window leaves its source sample interval')
            row[key] = result.get('raw_unrounded') if result['score'] is not None else None
            if row[key] is not None and not valid_probability(row[key]):
                raise ValueError('Invalid persisted score')
            row[key+'_windows'] = result['windows']
        rows.append(row)
    return rows


def fit():
    rows = read_scores({'fit', 'cal'})
    parameters = {}; guards = {}
    for key in ('old', 'new'):
        fit_rows = [r for r in rows if r['split'] == 'fit' and r[key] is not None]
        parameters[key] = base.fit_monotonic(base.logit([r[key] for r in fit_rows]), fit_rows, calibration=True)
        cal_rows = [dict(r, **{key: probability(r[key], parameters[key])}) for r in rows if r['split'] == 'cal']
        guards[key] = calibrate_guard(cal_rows, key)
    save(OUT/'calibration.json', dict(schema='chipmunk.audio-pilot-calibration/2',
        parameters=parameters, guards=guards, alpha=.1, deployment_eligible=False,
        manifest_sha256=manifest()['sha256'], run_lock_sha256=base.file_digest(OUT/'run-lock.json'),
        development_rows_sha256=sha(base.canonical(rows)), test_scores_accessed=False))
    print('Calibration frozen without reading test records:', base.file_digest(OUT/'calibration.json'), flush=True)


def metrics(rows, key, guarded):
    values = [r[key] for r in rows]
    w = base.source_weights(rows)
    basic = base.metric_values([r['label'] for r in rows], [np.nan if p is None else p for p in values], w)
    decisions = [r[key+('_guarded' if guarded else '_plain')] for r in rows]
    real = sum(r['label'] == 0 for r in rows); ai = len(rows)-real
    false_ai = sum(d == 'ai' and r['label'] == 0 for d, r in zip(decisions, rows))
    false_original = sum(d == 'original' and r['label'] == 1 for d, r in zip(decisions, rows))
    count = Counter(decisions); decided = count['ai'] + count['original']
    basic.update(total=len(rows), real=real, ai=ai, decisions=dict(count), false_ai_count=false_ai,
        false_original_count=false_original, false_ai_rate=false_ai/real if real else None,
        false_original_rate=false_original/ai if ai else None, coverage=decided/len(rows) if rows else None,
        decided_error=(false_ai+false_original)/decided if decided else None)
    return basic


def analyse():
    calibration = json.loads((OUT/'calibration.json').read_bytes())
    if calibration['run_lock_sha256'] != base.file_digest(OUT/'run-lock.json'): raise ValueError('Calibration lock mismatch')
    access = OUT/'holdout-access.json'
    if access.exists():
        if json.loads(access.read_bytes())['calibration_sha256'] != base.file_digest(OUT/'calibration.json'):
            raise ValueError('Cannot change calibration after accessing test')
    else:
        save(access, dict(calibration_sha256=base.file_digest(OUT/'calibration.json'),
            first_test_access_unix=time.time(), purpose='One frozen pilot evaluation; no promotion permitted'))
    rows = read_scores({'test'})
    for row in rows:
        for key in ('old', 'new'):
            row[key+'_raw'] = row[key]
            row[key] = probability(row[key], calibration['parameters'][key])
            row[key+'_plain'] = decision(row[key])
            row[key+'_guarded'] = decision(row[key], calibration['guards'][key])
    report = {'schema': 'chipmunk.audio-pilot-report/2', 'deployment_eligible': False,
        'decision': 'No promotion: bounded single-family pilot below release sample requirements.',
        'manifest_sha256': manifest()['sha256'], 'scores_sha256': base.file_digest(OUT/'scores.jsonl'),
        'calibration_sha256': base.file_digest(OUT/'calibration.json'), 'cohort': manifest()['audit'], 'groups': {}}
    from scipy.stats import beta
    for group in ['all'] + CONDITIONS + ['seconds_4', 'seconds_8']:
        selected = [r for r in rows if group == 'all' or r['condition'] == group or 'seconds_'+str(r['seconds']) == group]
        group_result = {}
        for key in ('old', 'new'):
            group_result[key] = {mode: metrics(selected, key, mode == 'guarded') for mode in ('plain', 'guarded')}
            raw_rows = [dict(r, **{key: r[key+'_raw']}) for r in selected]
            group_result[key]['raw_auc'] = metrics(raw_rows, key, False)['auc']
            # Cluster-level event bounds remain descriptive for this source population.
            for label, error_label, field in [(0, 'ai', 'false_ai'), (1, 'original', 'false_original')]:
                speakers = {r['speaker'] for r in selected if r['label'] == label}
                failures = {r['speaker'] for r in selected if r['label'] == label and r[key+'_guarded'] == error_label}
                k, n = len(failures), len(speakers)
                group_result[key][field+'_speaker_event_bound'] = {'events': k, 'speakers': n,
                    'one_sided_95_upper': None if not n else (float(beta.ppf(.95, k+1, n-k)) if k < n else 1.)}
        report['groups'][group] = group_result
    shared = [r for r in rows if r['old'] is not None and r['new'] is not None]
    report['shared_assessed'] = {key: metrics(shared, key, True) for key in ('old', 'new')}
    report['bootstrap_probability_scope'] = 'Brier/log-loss differences use only the identical jointly assessed records in each paired resample; coverage and errors use all records.'
    # Paired cluster resampling, all correlated conditions and files move together.
    rng = np.random.default_rng(749031)
    groups = {label: {s: [r for r in rows if r['speaker'] == s] for s in
                     sorted({r['speaker'] for r in rows if r['label'] == label})} for label in (0, 1)}
    distributions = defaultdict(list)
    for _ in range(2000):
        sampled = []
        for label in (0, 1):
            identities = list(groups[label])
            for index, speaker in enumerate(rng.choice(identities, len(identities), replace=True)):
                sampled.extend(dict(r, base_id=f'{label}:{index}:'+r['base_id']) for r in groups[label][speaker])
        for guarded in (False, True):
            results = {key: metrics(sampled, key, guarded) for key in ('old', 'new')}
            for metric in ('brier', 'log_loss', 'coverage', 'false_ai_rate', 'false_original_rate'):
                selected_results = results
                name = metric
                if metric in ('brier', 'log_loss'):
                    common = [r for r in sampled if r['old'] is not None and r['new'] is not None]
                    selected_results = {key: metrics(common, key, guarded) for key in ('old', 'new')}
                    name = 'shared_' + metric
                if selected_results['old'][metric] is not None and selected_results['new'][metric] is not None:
                    distributions[('guarded' if guarded else 'plain')+':'+name].append(selected_results['new'][metric]-selected_results['old'][metric])
    report['paired_cluster_bootstrap_new_minus_old'] = {key: {'replicates': len(v),
        'percentile_95': np.quantile(v, [.025, .975]).tolist()} for key, v in distributions.items()}
    report['limitations'] = ['20 test files per class; 10 synthetic and 20 authentic test identities.',
        'The 8-second test subgroup contains only one authentic source and ten generated sources; its results cannot establish a duration-specific error rate.',
        'Previously observed Chatterbox family only; no unseen-generator claim.',
        'Source class is confounded with recording collection and passage style.',
        'Controlled additive backgrounds; not natural rooms, codecs, languages or partial edits.',
        'Unknown pretraining overlap; probabilities use a balanced reference prior.',
        'Test now consumed. Any change needs another prospective test.']
    save(OUT/'predictions.json', rows); save(OUT/'evaluation.json', report)
    print(json.dumps(report['groups']['all'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['metadata', 'prepare', 'lock', 'score', 'fit', 'analyse'])
    parser.add_argument('--stage', choices=['development', 'test'], default='development')
    args = parser.parse_args()
    score(args.stage) if args.command == 'score' else globals()[args.command]()
