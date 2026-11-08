"""Portable authored audio/mouth arithmetic review: NumPy only, no media or models."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCIENCE = 'fddd0241b724985b20d750de2c654d9f94f45500'
FREEZE_SHA256 = 'af82459f9aad1aa46194fe4692fb026454365f7d35a4f0e6b3d09fcfb44923e5'


def require(value, message):
    if not value:
        raise ValueError(message)


def definitions(path, names, namespace, pins):
    source = (ROOT / path).read_bytes()
    require(hashlib.sha256(source).hexdigest() == pins[path], 'Original prospective math source changed')
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names), 'Every selected original definition required')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), path, 'exec'), namespace)
    return namespace


def review(output):
    require(not output.exists(), 'Retain earlier receipts')
    frozen = (ROOT / 'docs/evaluation-panels/mouth-audio-context-v14-freeze.json').read_bytes()
    require(hashlib.sha256(frozen).hexdigest() == FREEZE_SHA256, 'Original public freeze changed')
    lock = json.loads(frozen)
    require(lock['new_audio_tracks_processed'] == 0, 'Original prospective freeze required')
    core = definitions('backend/scripts/mouth_audio_context_v14.py',
        ('require', 'envelope', 'apertures', 'correlation', 'features'),
        {'np': np, 'math': math, 'LAGS': (-8, -4, 0, 4, 8)}, lock['code_sha256'])
    independent = definitions('backend/scripts/review_mouth_audio_context_v14.py', ('correlate', 'features'),
        {'math': math, 'statistics': statistics, 'run': SimpleNamespace(require=require)}, lock['code_sha256'])
    rng, maximum, count = np.random.default_rng(14014), 0., 0
    window = {'centers': [0, 6, 12, 24, 36, 48, 60, 72, 83]}
    for lag in (-8, -4, 0, 4, 8):
        audio = rng.normal(size=84)
        mouth = np.zeros(84)
        for i in range(8, 76):
            mouth[i] = audio[i + lag]
        values, support = core['features'](window, audio, mouth)
        other, expected_support = independent['features'](window, audio.tolist(), mouth.tolist())
        maximum = max(maximum, float(np.max(np.abs(np.asarray(values) - other))))
        require(support == expected_support and all(8 <= i < 76 for part in support for i in part), 'Exact common lag support differs')
        require(abs(values[(-8, -4, 0, 4, 8).index(lag)] - 1.) < 1e-12
                and values[5] == lag / 8 and values[15] == lag / 8, 'Known source lag sign or peak differs')
        changed = mouth.copy()
        changed[:8], changed[-8:] = 1e9, -1e9
        require(core['features'](window, audio, changed) == (values, support), 'Out-of-support values wrapped into evidence')
        count += 1
    values, _ = core['features'](window, np.ones(84), np.ones(84))
    require(values == [0.] * 20, 'Flat traces cannot imply verified synchronization')
    pcm = rng.uniform(-1., 1., 50 * 640).astype(np.float32)
    expected = [math.log(max(math.sqrt(math.fsum(float(v) ** 2 for v in pcm[i:i + 640]) / 640), 1e-8))
                for i in range(0, len(pcm), 640)]
    maximum = max(maximum, float(np.max(np.abs(core['envelope'](pcm) - expected))))
    require(maximum <= 1e-12 and not any(n in sys.modules for n in ('torch', 'transformers', 'sklearn', 'scipy')),
            'Arithmetic agreement or model-independent import boundary failed')
    record = {'schema': 'chipmunk.mouth-audio-portable-math-review/14', 'science_commit': SCIENCE,
              'known_signed_lags_checked': count, 'short_clip_sample_blocks_checked': 50,
              'maximum_scalar_vector_difference': maximum, 'absolute_limit': 1e-12,
              'flat_trace_and_no_wrap_checks': True, 'python': platform.python_version(),
              'platform': platform.platform(), 'numpy': np.__version__, 'source_media_inputs': 0,
              'neural_forwards': 0, 'head_fits': 0, 'outside_human_reviewers': 0,
              'scope': 'Authored signal arithmetic only; not native media, learned-head parity or detection quality.'}
    with output.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    review(parser.parse_args().output)
