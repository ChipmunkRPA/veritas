# SPDX-License-Identifier: Apache-2.0
"""Run pinned two-track separators offline on exact 16 kHz mono evaluation buffers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

KEZEK_WHEEL = 'kezek-1.0.1-py3-none-any.whl'
KEZEK_REVISION = 'a31397b5f637e9e6512bdff1f17dfb9efd2975e2'
KEZEK_DIGESTS = {
    KEZEK_WHEEL: 'fa4997d545b4126c211b5ae8fa65e6ac1571b577cebbf4c1a2ed91f196c25b86',
    'fingerprint.pt': '31b223c28a852833263c93fc1e6e8fd4f5e45cc267a50be02fa7fea2aefa9c1d',
    'separator.pt': '7758bd83c6546de48487fb6519b72c987fd5431e6d7a086bc052ee45027400ab',
    'stage1_pairnet.pt': '43fd76fe157439358c0e31f7f8e472fd80b0e41aab1627bd038321e424df4341',
}
RATE = 16000
MAX_SECONDS = 30


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_assets(path, digests):
    if any(sha(path / name) != digest for name, digest in digests.items()):
        raise ValueError('Separator asset differs from its published checksum')


def _load_kezek(path, torch):
    verify_assets(path, KEZEK_DIGESTS)
    sys.path.insert(0, str(path / KEZEK_WHEEL))
    from kezek.pipeline import Kezek, KezekTSEL, PairNet, SpeakerNet

    # Same graph, parameters and CPU inference as the pinned release. Read tensor
    # dictionaries with PyTorch's restricted loader instead of general pickle.
    def weights(name):
        return torch.load(path / name, map_location='cpu', weights_only=True)

    stage = weights('stage1_pairnet.pt')
    speaker = weights('fingerprint.pt')
    separator = weights('separator.pt')
    model = Kezek.__new__(Kezek)
    model.pn = PairNet(stage.get('config', 'S'), front=stage.get('front', 'conv'), fp=stage.get('fp', False),
                       mel=stage.get('mel', False), ssl=stage.get('ssl', False)).eval()
    model.spk = SpeakerNet(speaker['config'], speaker.get('front', 'conv')).eval()
    model.sep = KezekTSEL('L').eval()
    model.pn.load_state_dict(stage['model'], strict=True)
    model.spk.load_state_dict(speaker['model'], strict=True)
    model.sep.load_state_dict(separator.get('model', separator), strict=True)
    model.dev, model.thr, model.refine = 'cpu', .3, 1

    def separate(mixture):
        # Silence has no voice to separate. The upstream profile constructor
        # requires at least one voice, so keep that documented boundary explicit.
        if np.max(np.abs(mixture)) <= 1e-7:
            return None, 'silent_input', None
        try:
            separated, activity, _ = model(mixture)
        except TypeError as error:
            if str(error) != "bad operand type for unary -: 'NoneType'":
                raise
            return None, 'no_active_profiles', None
        exclusive = [int(np.count_nonzero(activity[index] & ~activity[1 - index])) for index in (0, 1)]
        return separated, 'scored', exclusive

    return separate


def load(name, assets, torch):
    if name == 'kezek':
        return _load_kezek(assets, torch), KEZEK_DIGESTS, KEZEK_REVISION
    from scripts.audio_overlap_reconstruction_views import ASSETS, REVISION, load_two, run_model

    verify_assets(assets, ASSETS)
    model = load_two(assets)
    return lambda pcm: (run_model(model, pcm, 2), 'scored', None), ASSETS, REVISION


def read_pcm(row):
    pin = row['decoded_pcm']
    path = Path(pin['path'])
    size = row['samples']
    if not 0 < size <= MAX_SECONDS * RATE or row['sample_rate_hz'] != RATE or pin['samples'] != size:
        raise ValueError('Only bounded 16 kHz native mono buffers are supported')
    if sha(path) != pin['sha256'] or path.stat().st_size != 4 * size:
        raise ValueError('Prepared audio differs from the exact decoded clock or waveform')
    pcm = np.fromfile(path, dtype='<f4')
    if not np.isfinite(pcm).all():
        raise ValueError('Prepared audio must be finite')
    return pcm


def run(name, assets, manifest, output):
    from scripts.audio_strong_traffic_view import network_free

    network_free()
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    inputs = json.loads(manifest.read_bytes())
    if inputs['sample_rate_hz'] != RATE or not inputs['rows']:
        raise ValueError('An exact-clock evaluation panel is required')
    cases = [(int(row['group']), row['role'], row) for row in inputs['rows']]
    if (len({(group, role) for group, role, _ in cases}) != len(cases)
            or any(group < 1 or role not in {'edited', 'ordinary', 'background'} for group, role, _ in cases)):
        raise ValueError('The panel has ambiguous group/role assignments')
    separate, digests, revision = load(name, assets, torch)
    plan = {'schema': 'chipmunk.severe-overlap-separation-private/1', 'panel_sha256': sha(manifest),
            'code_sha256': sha(Path(__file__)), 'method': name, 'revision': revision, 'digests': digests,
            'torch': str(torch.__version__), 'threads': 2, 'total': len(cases)}
    output.mkdir(mode=0o700)
    with (output / 'measurements.jsonl').open('x') as stream:
        stream.write(json.dumps(plan, sort_keys=True) + '\n')
        for group, role, row in cases:
            pcm = read_pcm(row)
            t0 = time.monotonic()
            with torch.inference_mode():
                wave, status, exclusive = separate(pcm)
            elapsed = time.monotonic() - t0
            record = {'group': group, 'role': role, 'input_sha256': row['decoded_pcm']['sha256'], 'samples': len(pcm),
                      'status': status, 'exclusive_activity_frames': exclusive, 'seconds': round(elapsed, 6)}
            if wave is not None:
                wave = np.asarray(wave, dtype='<f4')
                if wave.shape != (2, len(pcm)) or not np.isfinite(wave).all():
                    raise ValueError('The separator did not produce finite tracks on the original sample clock')
                destination = output / f'{group:03d}-{role}.npy'
                np.save(destination, wave, allow_pickle=False)
                record['track_sha256'] = sha(destination)
            stream.write(json.dumps(record, sort_keys=True) + '\n')
            stream.flush()
            print(group, role, status, f'{elapsed:.3f}s', 'exclusive:', exclusive, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['kezek', 'sepformer'])
    for key in ('assets', 'manifest', 'output'):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    run(args.method, args.assets, args.manifest, args.output)
