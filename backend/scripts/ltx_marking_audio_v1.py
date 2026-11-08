"""Pinned LTX audio marking research; public patterns never authenticate origin."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts import public_audio_watermark_v1 as base

LOCK = Path(__file__).with_name('ltx_marking_sources_v1.json')


@dataclass
class Audio:
    waveform: object
    sampling_rate: int


def verify(root):
    record = json.loads(LOCK.read_text())
    for name, facts in record['files'].items():
        path = root / name
        base.require(path.stat().st_size == facts['bytes'] and base.digest(path) == facts['sha256'],
                     'Pinned LTX source or declaration changed')
    return record


def upstream(root):
    """Execute only inspected definitions; skip provider package/weight loaders."""
    verify(root)
    import typing

    import torch
    from einops import rearrange
    from scipy import stats

    scope = {'torch': torch, 'nn': torch.nn, 'rearrange': rearrange, 'stats': stats,
             'Audio': Audio, 'AudioMarker': object, 'NamedTuple': typing.NamedTuple}
    for name, selected in [('types.py', {'MarkDetectResult'}), ('score.py', {'score_bits'}),
                           ('audio_seal.py', {'AudioSealMarker', 'baked_audioseal_payload', 'waveform_to_channel_batch'})]:
        tree = ast.parse((root / name).read_text())
        nodes = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected:
                if isinstance(node, ast.ClassDef) and node.name == 'AudioSealMarker':
                    node.body = [part for part in node.body if not isinstance(part, ast.FunctionDef) or part.name != 'create']
                nodes.append(node)
        base.require({n.name for n in nodes} == selected, 'Exact inspected source definitions required')
        scope.update(AUDIOSEAL_NBITS=16, AUDIOSEAL_PAYLOAD_SEED=0x4C5458)
        code = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[]))
        exec(compile(code, str(root / name), 'exec'), scope)
    return scope


def interpret(presence, decoded, expected):
    base.require(presence.dtype == np.float32 and presence.ndim == 2 and presence.shape[0] == 2
        and presence.shape[1] > 0 and np.isfinite(presence).all() and ((presence >= 0) & (presence <= 1)).all(),
        'Both complete native channel/sample probabilities required')
    base.require(decoded.shape == expected.shape == (16,) and np.isin(decoded, [0, 1]).all()
        and np.isin(expected, [0, 1]).all(), 'Actual decoded and expected16-bit arrays required')
    count, total = int(np.count_nonzero(presence > .5)), int(presence.size)
    matches = int(np.count_nonzero(decoded == expected))
    p = sum(math.comb(16, k) for k in range(matches, 17)) / 2**16
    found = count * 2 >= total
    return {'family_presence_detected': found, 'positive_native_sample_fraction': count / total,
        'bit_matches': matches, 'bit_accuracy': matches / 16, 'bit_match_p_value': p,
        'public_payload_exact_match': matches == 16,
        'state': ('recognized_public_payload' if matches == 16 else 'family_signal_only') if found else 'not_detected',
        'provider_authenticated': False, 'used_in_confidence': False, 'early_stop_eligible': False,
        'scope': 'Public post-hoc watermark observation. A matching pattern is copyable, not a signature or AI-authorship probability.'}


def observe(marker, waveform, rate):
    import torch

    base.require(waveform.dtype == torch.float32 and waveform.ndim == 2 and waveform.shape[0] == 2
                 and waveform.shape[1] == 2 * rate and rate in (24000, 48000)
                 and torch.isfinite(waveform).all(), 'Exact two-second stereo native24/48kHz PCM required')
    with torch.inference_mode():
        declared = marker.detect(Audio(waveform, rate))
    actual = marker._detector.last
    detection, message, tensor = actual
    expected_tensor = waveform[:, None, :].contiguous()
    base.require(torch.equal(tensor, expected_tensor), 'Upstream input changed native samples or downmixed channels')
    decoded = (message.mean(dim=0) > .5).to(dtype=torch.int64).numpy()
    expected = marker._payload.to(dtype=torch.int64).numpy()
    result = interpret(detection[:, 1, :].numpy(), decoded, expected)
    base.require(result['family_presence_detected'] == declared.detected
        and result['bit_accuracy'] == declared.bit_accuracy and abs(result['bit_match_p_value'] - declared.p_value) <= 1e-15
        and torch.equal(declared.payload, marker._payload), 'Independent interpretation differs from inspected upstream output')
    return result, detection[:, 1, :].numpy().copy(), message.numpy().copy()


def models(source, cached):
    import torch

    scope = upstream(source)
    models, receipts = base.load_models(cached)

    class RecordingDetector(torch.nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector
            self.calls = 0
            self.last = None

        def forward(self, tensor):
            result = self.detector(tensor)
            self.calls += 1
            self.last = (result[0].detach().cpu(), result[1].detach().cpu(), tensor.detach().cpu().clone())
            return result

    recorder = RecordingDetector(models['detector'])
    marker = scope['AudioSealMarker'](models['generator'], recorder, nbits=16)
    return marker, scope, receipts


def carrier(rate, index):
    t = np.arange(2 * rate, dtype=np.float64) / rate
    envelope = .4 + .6 * np.sin(2 * np.pi * (1.1 + .2 * index) * t)**2
    left = .16 * envelope * np.sin(2 * np.pi * (151 + index * 79) * t)
    right = .13 * envelope[::-1] * np.sin(2 * np.pi * (229 + index * 67) * t)
    return np.stack([left, right]).astype(np.float32)


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
