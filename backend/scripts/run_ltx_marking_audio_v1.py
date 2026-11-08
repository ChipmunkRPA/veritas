"""Frozen native stereo/public-payload contract, isolated from customer serving."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import ltx_marking_audio_v1 as method
from scripts import public_audio_watermark_v1 as base

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'backend/data/independent-evaluation/ltx-marking-v1-source'
CACHED = ROOT / 'backend/data/independent-evaluation/public-watermark-v2-source'
PINS = ('backend/scripts/ltx_marking_audio_v1.py', 'backend/scripts/ltx_marking_sources_v1.json',
    'backend/scripts/run_ltx_marking_audio_v1.py', 'backend/scripts/ltx_marking_runtime_v1.py',
    'backend/scripts/public_audio_watermark_v1.py', 'backend/scripts/specialist_runtime_v4.py',
    'backend/tests/test_ltx_marking_audio_v1.py', 'docs/ltx-marking-audio-v1-protocol.md')
POLICY = {'schema': 'chipmunk.ltx-public-audio-contract/1', 'rates': [24000, 48000], 'carriers': 4, 'seconds': 2,
    'conditions': ['unmarked', 'matching-public-payload', 'opposite-public-payload', 'mixed-channel-payloads', 'middle-only-mark', 'marked-plus-noise'],
    'native_cases': 48, 'file_cases': 32, 'codecs': ['pcm16-wav', 'aac128k-m4a'],
    'files': 'For each of8 rate/carrier groups, original and matching mark through both codecs. Decode first exact2seconds at original native rate, preserve stereo,bind file/decoded sample hashes and actual stream timing. AAC encoder delay is decoded by ffmpeg,not treated as a forged offset.',
    'scope': 'Authored mathematical carriers and native Tensor/file contract,not human speech/provider export accuracy,not lip-sync or video marking.',
    'upstream': 'Pinned inspected AST definitions only; omit upstream create/weight-loader/package/file-decode entry points. Reuse exact hash-matching restricted-loaded16-bit generator/detector weights.',
    'parity': 'Each native detector response compared to independent channel-presence and exact integer binomial-bit arithmetic. First generator and detector input repeat bitwise. Stereo channels remain separate mono batch items; no resampling in marker.',
    'meaning': 'Upstream audio detected is family presence,not a payload test. The returned payload field is the baked expected payload,not recovered bits. Matching public marks are copyable and cannot authenticate AI origin or stop detection.',
    'missing': 'Preserve all80 planned cases and errors. Missing/unavailable checks are not negative or original. No inference from absent marks.',
    'guard': {'seconds': 900, 'rss_gib': 6, 'available_mib': 1536, 'disk_mib': 1536},
    'used_in_confidence': False, 'early_stop_eligible': False, 'serving_admitted': False}


def identity():
    method.verify(SOURCE)
    base.verify_source(CACHED)
    return {'policy': POLICY, 'code_sha256': {p: base.digest(ROOT / p) for p in PINS},
            'source_lock_sha256': base.digest(method.LOCK), 'cached_source_review_sha256': base.digest(CACHED / 'source-review.json'),
            'cached_dependency_manifest_sha256': base.digest(CACHED / 'isolated-python-files.json'),
            'weight_sha256': base.WEIGHTS,
            'packages': {n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'scipy', 'einops', 'PyYAML', 'psutil', 'Pillow')},
            'binaries_sha256': {n: base.digest(Path(shutil.which(n))) for n in ('ffmpeg', 'ffprobe')}}


def freeze(trial):
    value = identity()
    trial.mkdir(exist_ok=False)
    base.write(trial / 'freeze.json', value)
    base.write(ROOT / 'docs/evaluation-panels/ltx-marking-audio-v1-freeze.json',
               {**value, 'private_freeze_sha256': base.digest(trial / 'freeze.json'), 'authored_media_forwards_before_freeze': 0})


def check(trial):
    base.require(base.read(trial / 'freeze.json') == identity(), 'Frozen method/source/weights changed')


def codec(waveform, rate, kind, folder):
    raw = waveform.T.astype('<f4').tobytes()
    if kind == 'pcm16-wav':
        path = folder / 'input.wav'
        pcm = np.rint(np.clip(waveform.T, -1, 1 - 1 / 32768) * 32768).astype('<i2')
        with wave.open(str(path), 'wb') as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(pcm.tobytes())
    else:
        path = folder / 'input.m4a'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'f32le', '-ar', str(rate), '-ac', '2', '-i', 'pipe:0',
            '-map_metadata', '-1', '-c:a', 'aac', '-b:a', '128k', str(path)], input=raw, check=True, timeout=45)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)], timeout=20))
    audio = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    base.require(len(audio) == 1 and audio[0]['channels'] == 2 and int(audio[0]['sample_rate']) == rate,
                 'Codec changed actual channel count or native rate')
    output = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-map', '0:a:0',
        '-t', '2', '-ar', str(rate), '-ac', '2', '-f', 'f32le', 'pipe:1'], timeout=30)
    value = np.frombuffer(output, dtype='<f4').reshape(-1, 2).T.copy()
    base.require(value.shape == waveform.shape, 'The complete declared first2-second audio extent is required')
    return value, {'file_sha256': base.digest(path), 'decoded_pcm_sha256': method.sha_array(value),
                   'native_stream': {k: audio[0].get(k) for k in ('sample_rate', 'channels', 'time_base', 'start_pts', 'start_time', 'duration_ts', 'duration')},
                   'scope': 'First two decoded source seconds; no silent mono conversion or detector resampling.'}


def run(trial):
    check(trial)
    import socket

    import torch

    torch.set_num_threads(2)
    marker, scope, receipts = method.models(SOURCE, CACHED)
    def forbidden(*a, **kw):
        raise RuntimeError('No network during native watermark contract')
    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    private = trial / 'private'
    private.mkdir(exist_ok=False)
    rows, generation_calls, repeat = [], 0, None
    with (trial / 'journal.jsonl').open('x') as stream:
        for rate in POLICY['rates']:
            for index in range(4):
                wav = torch.from_numpy(method.carrier(rate, index))
                with torch.inference_mode():
                    positive = marker.embed(method.Audio(wav, rate)).waveform
                    generation_calls += 1
                    other = scope['AudioSealMarker'](marker._generator, marker._detector, nbits=16)
                    other._payload = 1 - marker._payload
                    opposite = other.embed(method.Audio(wav, rate)).waveform
                    generation_calls += 1
                    if repeat is None:
                        repeated = marker.embed(method.Audio(wav, rate)).waveform
                        generation_calls += 1
                        base.require(torch.equal(positive, repeated), 'First marked native tensor did not repeat bitwise')
                mixed = torch.stack([positive[0], opposite[1]])
                partial = wav.clone()
                partial[:, rate // 2:3 * rate // 2] = positive[:, rate // 2:3 * rate // 2]
                rng = np.random.default_rng(771 + rate + index)
                noise = torch.from_numpy(rng.standard_normal(tuple(wav.shape)).astype(np.float32))
                rms = torch.sqrt(torch.mean(positive * positive))
                noisy = positive + noise * rms / np.sqrt(10**(.5))
                cases = dict(zip(POLICY['conditions'], (wav, positive, opposite, mixed, partial, noisy), strict=True))
                def observe(name, tensor, extra=None, rate=rate, index=index):
                    nonlocal repeat
                    value, probabilities, messages = method.observe(marker, tensor.contiguous(), rate)
                    if repeat is None:
                        actual, p, m = method.observe(marker, tensor.contiguous(), rate)
                        base.require(actual == value and np.array_equal(p, probabilities) and np.array_equal(m, messages), 'First detector input did not repeat bitwise')
                        repeat = True
                    identity = f'{rate}-{index}-{name}'
                    path = private / (identity + '.npz')
                    np.savez_compressed(path, probabilities=probabilities, messages=messages,
                                        waveform=tensor.detach().cpu().numpy())
                    row = {'id': identity, 'rate': rate, 'carrier': index, 'condition': name,
                        'pcm_sha256': method.sha_array(tensor.detach().cpu().numpy()), 'result': value,
                        'packet': path.name, 'packet_sha256': base.digest(path), **(extra or {})}
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                for name, tensor in cases.items():
                    observe(name, tensor)
                for kind in POLICY['codecs']:
                    for name, tensor in [('unmarked', wav), ('matching-public-payload', positive)]:
                        folder = private / f'codec-{rate}-{index}-{kind}-{name}'
                        folder.mkdir()
                        decoded, proof = codec(tensor.numpy(), rate, kind, folder)
                        observe(kind + '/' + name, torch.from_numpy(decoded), {'file_proof': proof})
                print({'rate': rate, 'carrier': index, 'completed_cases': len(rows), 'detector_forwards': marker._detector.calls}, flush=True)
    base.require(len(rows) == 80 and marker._detector.calls == 81 and generation_calls == 17,
                 'Complete80-case native/file contract with both exact repeats required')
    check(trial)
    summary = {}
    for condition in sorted({r['condition'] for r in rows}):
        selected = [r for r in rows if r['condition'] == condition]
        summary[condition] = {'cases': len(selected), 'family_presence': sum(r['result']['family_presence_detected'] for r in selected),
            'exact_public_payload_with_presence': sum(r['result']['state'] == 'recognized_public_payload' for r in selected),
            'bit_matches': dict(Counter(r['result']['bit_matches'] for r in selected))}
    base.write(trial / 'result.json', {'policy': POLICY, 'freeze_sha256': base.digest(trial / 'freeze.json'),
        'cases': 80, 'summary': summary, 'model_load_receipts': receipts, 'generator_forwards': generation_calls,
        'detector_forwards': marker._detector.calls, 'exact_first_generator_and_detector_repeats': True,
        'journal_sha256': base.digest(trial / 'journal.jsonl'), 'provider_authenticated': False,
        'serving_admitted': False, 'used_in_confidence': False, 'early_stop_eligible': False,
        'customer_inputs': 0, 'outside_human_reviewers': 0, 'new_weights_downloaded': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    freeze(args.trial) if args.action == 'freeze' else run(args.trial)
