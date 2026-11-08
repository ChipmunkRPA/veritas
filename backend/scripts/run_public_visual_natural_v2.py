"""Prospectively fixed natural-media qualification of the public MIT WAM family."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import prepare_public_visual_natural_v2 as sources
from scripts import public_visual_watermark_v1 as m
from scripts import run_public_visual_watermark_v1 as authored

ROOT = Path(__file__).resolve().parents[2]
CODE = authored.CODE + [
    'backend/scripts/prepare_public_visual_natural_v2.py',
    'backend/scripts/run_public_visual_natural_v2.py',
    'backend/scripts/review_public_visual_natural_v2.py',
    'backend/scripts/public_natural_visual_runtime_v2.py',
    'backend/tests/test_public_visual_natural_v2.py',
    'docs/public-visual-natural-v2-protocol.md',
]
PAYLOADS = [list(range(2)) * 16, [0] * 32, [1] * 32,
            np.unpackbits(np.frombuffer(hashlib.sha256(b'chipmunk-natural-v2-payload').digest()[:4], np.uint8)).tolist()]
POLICY = {
    'schema': 'chipmunk.natural-visual-watermark-policy/2',
    'scope': 'Public-family mark recovery/localization on source-reviewed natural media. No-added-mark controls are not certified human/original files. Not general deepfake, face-swap, speech or production-provider accuracy.',
    'images': '32 fixed source photos: 10 architecture, 10 food, 6 plants, 6 landscapes; no more than2 images per credited creator. Four states: no_added_mark/full_mark/partial_center/ordinary_rotated_center; native/JPEG70.256 cases.',
    'videos': '8 fixed source camera clips, distinct credited creators;8 sampled frames per clip with exact original PTS and center-crop mapping. No-added-mark/partial/ordinary edit; raw/H264CRF23.384 frame cases. No audio check.',
    'video_truth': 'Derived2fps sequence only: center128square on sampled frames2:6 is marked or conventionally rotated. Derived1–3s maps to recorded source samples; no examination of unsampled original frames. Codec propagation can affect outside intervals.',
    'payloads': PAYLOADS,
    'payload_assignment': 'Photo/clip numeric ID modulo4; alternating, all-zero, all-one and fixed SHA256-derived32bit patterns. Input diversity, not a detector/key/threshold change.',
    'point': 'Unchanged native presence logits strictly>0;8-connected components of at least500pixels flag inspection. Every presence logit, pixel mask and hard bit map retained. A mark does not authenticate an AI generator.',
    'preprocessing': 'Unchanged publisher ImageNet transform, native F32, scaling_w2 and JND. Full marked output unnormalized/clamped/half-up RGB8. Partial copies marked center pixels only; ordinary edits are center90degree rotation.',
    'storage': 'Every native presence F32 byte retained losslessly as transposed byte planes; original-byte SHA256 per case. All32 binary prediction maps packed losslessly. No floating precision reduction. Original sources, working arrays, full/partial/ordinary base arrays and H264 clips retained privately.',
    'source_review': 'All40 selected sources and56 previews reviewed for photographic content before scoring. Metadata dates/licenses are publisher claims; project URL inventories excluded; pretraining overlap unknown.',
    'image_cases': 256, 'video_frame_cases': 384, 'generator_forwards': 64, 'detector_forwards': 645,
    'parity': 'Four first-photo wrapper/direct comparisons and one exact repeated partial prediction. Every message compared with publisher hard readout. No score-dependent source replacement, point change or omitted case.',
    'reporting': 'All-source/state/codec totals, source-level any-frame flags and complete marked-interval misses, exact message recovery/unavailability, marked IoU>=0.5 counts, and separate payload/photo-stratum summaries. Variants/frames are correlated; no population accuracy or calibrated confidence inferred.',
    'inference': 'CPUfloat32,2Torch threads,seed20261108;900seconds,6GiB sampled treeRSS,1536MiB availableRAM/disk guard.',
    'provider_authenticated': False, 'used_in_confidence': False, 'early_stop_eligible': False,
    'serving_admitted': False, 'outside_human_reviewers': 0,
}


def byteplanes(logits):
    return logits.astype('<f4', copy=False).view(np.uint8).reshape(-1, 4).T.copy()


def evidence(logits, bits, expected):
    mask = logits > 0
    count = int(mask.sum())
    decoded = None if count == 0 else (bits[:, mask].sum(axis=1) > count / 2).astype(np.uint8).tolist()
    return {**m.spatial(logits), 'decoded_message': decoded,
            'decoded_bits_correct_of32': None if decoded is None else sum(a == b for a, b in zip(decoded, expected, strict=True))}


def aggregate(rows):
    groups = authored.aggregate(rows)
    for group in groups:
        members = [r for r in rows if (r['kind'], r.get('transform', r.get('container')), r.get('clip_state', r['state']))
                   == (group['kind'], group['alteration'], group['state'])]
        marked = [r for r in members if r['state'] in ('marked', 'partial')]
        names = {r['carrier'] for r in members}
        known = [r for r in marked if r['evidence']['decoded_message'] is not None]
        marked_names = {r['carrier'] for r in marked}
        group.update(sources=len(names), sources_with_any_flag=sum(any(r['carrier'] == n and r['evidence']['flagged_for_inspection'] for r in members) for n in names),
            marked_sources=len(marked_names), marked_sources_with_no_marked_frame_flag=sum(not any(r['carrier'] == n and r['evidence']['flagged_for_inspection'] for r in marked) for n in marked_names),
            marked_message_exact=sum(r['evidence']['decoded_bits_correct_of32'] == 32 for r in marked),
            marked_message_unavailable=len(marked) - len(known),
            marked_message_mean_bit_accuracy=None if not known else float(np.mean([r['evidence']['decoded_bits_correct_of32'] / 32 for r in known])),
            marked_cases_with_flag_and_iou_at_least_0_5=sum(r['evidence']['flagged_for_inspection'] and r['overlap']['iou'] >= .5 for r in marked))
    return groups


def check_sources(trial):
    sources.check(trial)
    acquired = m.read(trial / 'acquired-sources.json')
    m.require(acquired['source_plan_sha256'] == m.digest(trial / 'source-plan.json'), 'Selected source plan changed')
    m.require(len(acquired['sources']) == 40 and len({r['id'] for r in acquired['sources']}) == 40,
              'Complete unique natural-source panel required')
    for row in acquired['sources']:
        for field in ('asset', 'working_array'):
            path = Path(row[field])
            m.require(path.resolve().is_relative_to((trial / 'sources').resolve())
                and m.digest(path) == row[field + '_sha256'], 'Acquired source bytes changed')
        pixels = np.load(row['working_array'], allow_pickle=False)
        shape = (256, 256, 3) if row['kind'] == 'image' else (8, 256, 256, 3)
        m.require(pixels.dtype == np.uint8 and pixels.shape == shape and list(shape) == row['shape']
            and hashlib.sha256(pixels.tobytes()).hexdigest() == row['rgb_sha256'], 'Working RGB/source identity changed')
    return acquired['sources']


def freeze(trial, source):
    m.verify(ROOT, source)
    check_sources(trial)
    reviewed = m.read(trial / 'visual-review.json')
    m.require(reviewed['approved_source_ids'] == [r['id'] for r in m.read(trial / 'acquired-sources.json')['sources']]
        and reviewed['source_plan_sha256'] == m.digest(trial / 'source-plan.json')
        and reviewed['acquired_sources_sha256'] == m.digest(trial / 'acquired-sources.json')
        and reviewed['source_review_sha256'] == m.digest(trial / 'source-review.png')
        and reviewed['local_detector_forwards'] == 0, 'Pre-inference visual source review required')
    lock = {'policy': POLICY, 'source_root': str(source), 'code_sha256': {p: m.digest(ROOT / p) for p in CODE},
        'packages': {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'numpy', 'scipy', 'PyYAML', 'einops', 'opencv-python-headless', 'Pillow', 'psutil')},
        'source_assets_sha256': m.digest(m.LOCK),
        'source_plan_sha256': m.digest(trial / 'source-plan.json'),
        'acquired_sources_sha256': m.digest(trial / 'acquired-sources.json'),
        'visual_review_sha256': m.digest(trial / 'visual-review.json'), 'neural_forwards_before_freeze': 0}
    m.write(trial / 'freeze.json', lock)
    return {'frozen': True, 'cases': 640, 'native_forwards': 709}


def check(trial):
    lock = m.read(trial / 'freeze.json')
    m.require(lock['policy'] == POLICY and lock['code_sha256'] == {p: m.digest(ROOT / p) for p in CODE}, 'Frozen natural-media policy/code changed')
    m.require(lock['packages'] == {p: importlib.metadata.version(p) for p in lock['packages']}, 'Frozen environment changed')
    for name in ('source-plan', 'acquired-sources', 'visual-review'):
        m.require(m.digest(trial / (name + '.json')) == lock[name.replace('-', '_') + '_sha256'], 'Frozen source/review record changed')
    m.require(lock['source_assets_sha256'] == m.digest(m.LOCK), 'Model asset lock changed')
    m.verify(ROOT, Path(lock['source_root']))
    return lock, check_sources(trial)


def run(trial):
    lock, inputs = check(trial)
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261108)
    model, loaded = m.load_model(ROOT, Path(lock['source_root']))
    from watermark_anything.data.metrics import msg_predict_inference
    from watermark_anything.data.transforms import default_transform, unnormalize_img

    output = trial / 'private'
    output.mkdir(exist_ok=False)
    rows = []
    plan_sources = {r['id']: r for r in m.read(trial / 'source-plan.json')['sources']}
    generators = detectors = 0
    journal = trial / 'case-journal-private.jsonl'
    with journal.open('x') as ledger:
        def embed(pixels, payload):
            nonlocal generators
            tensor = default_transform(Image.fromarray(pixels))[None]
            message = torch.tensor([payload], dtype=torch.float32)
            with torch.inference_mode():
                marked = model.embed(tensor, message)['imgs_w']
            generators += 1
            values = unnormalize_img(marked)[0].permute(1, 2, 0).numpy()
            return np.floor(np.clip(values, 0, 1) * 255 + .5).astype(np.uint8)

        def detect(pixels, identity, state, payload, parity=False, repeat=False):
            nonlocal detectors
            tensor = default_transform(Image.fromarray(pixels))[None]
            with torch.inference_mode():
                pred = model.detect(tensor)['preds']
            detectors += 1
            m.require(pred.shape == (1, 33, 256, 256) and pred.dtype == torch.float32
                      and torch.isfinite(pred).all(), 'Native geometry/type/finite output required')
            if parity:
                with torch.inference_mode():
                    direct = model.detector(tensor)
                detectors += 1
                m.require(torch.equal(pred, direct), 'Native wrapper/direct mismatch')
            if repeat:
                with torch.inference_mode():
                    again = model.detect(tensor)['preds']
                detectors += 1
                m.require(torch.equal(pred, again), 'Native repeated partial prediction differs')
            logits, bits = pred[0, 0].numpy().copy(), (pred[0, 1:] > 0).numpy().copy()
            e = evidence(logits, bits, payload)
            if e['decoded_message'] is not None:
                with torch.inference_mode():
                    message = msg_predict_inference(pred[:, 1:].clone(), torch.from_numpy((logits > 0).astype(np.float32))[None, None], method='hard')
                m.require(message[0].int().tolist() == e['decoded_message'], 'Publisher hard-message readout differs')
            path = output / (identity + '.npz')
            np.savez_compressed(path, pixels=pixels, presence_byteplanes=byteplanes(logits),
                                binary_bit_maps=np.packbits(bits.reshape(32, -1), axis=1, bitorder='little'))
            return {'id': identity, 'state': state, 'payload': payload, 'private_array': str(path),
                'array_sha256': m.digest(path), 'source_rgb_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                'native_presence_f32_sha256': hashlib.sha256(logits.astype('<f4').tobytes()).hexdigest(),
                'evidence': e, 'overlap': m.overlap(logits > 0, authored.truth(state))}

        def retain(row):
            row['source_stratum'] = plan_sources[row['carrier']]['metadata_file']
            rows.append(row)
            ledger.write(json.dumps(row, allow_nan=False) + '\n')
            ledger.flush()

        for source in inputs:
            name = source['id']
            original = np.load(source['working_array'], allow_pickle=False)
            payload = PAYLOADS[int(name[1:]) % 4]
            if source['kind'] == 'image':
                marked = embed(original, payload)
                partial, ordinary = original.copy(), original.copy()
                partial[64:192, 64:192] = marked[64:192, 64:192]
                ordinary[64:192, 64:192] = np.rot90(original[64:192, 64:192])
                states = {'unmarked': original, 'marked': marked, 'partial': partial, 'ordinary_edit': ordinary}
            else:
                partial, ordinary = original.copy(), original.copy()
                marked = []
                for i in range(2, 6):
                    full = embed(original[i], payload)
                    marked.append(full)
                    partial[i, 64:192, 64:192] = full[64:192, 64:192]
                    ordinary[i, 64:192, 64:192] = np.rot90(original[i, 64:192, 64:192])
                states = {'unmarked': original, 'partial': partial, 'ordinary_edit': ordinary}
            base = output / (name + '-base.npz')
            np.savez_compressed(base, **states, **({'full_marked_frames': np.stack(marked)} if source['kind'] == 'video' else {}))
            base_sha = m.digest(base)
            for state, values in states.items():
                if source['kind'] == 'image':
                    for alteration in ('native', 'jpeg70'):
                        first = name == 'I00'
                        row = detect(authored.transform(values, alteration), f'{name}-{state}-{alteration}', state, payload,
                            parity=first and (state, alteration) in [('unmarked', 'native'), ('marked', 'native'), ('partial', 'native'), ('partial', 'jpeg70')],
                            repeat=first and (state, alteration) == ('partial', 'native'))
                        row.update(kind='image', carrier=name, transform=alteration, base_array=str(base), base_sha256=base_sha, anchor=None)
                        retain(row)
                else:
                    path = output / f'{name}-{state}.mp4'
                    decoded, derived = authored.encode_video(values, path)
                    video_sha = m.digest(path)
                    for container, frames in [('raw', values), ('h264', decoded)]:
                        for i, frame in enumerate(frames):
                            actual = state if state != 'unmarked' and 2 <= i < 6 else 'unmarked'
                            row = detect(frame, f'{name}-{state}-{container}-{i}', actual, payload)
                            row.update(kind='video', carrier=name, clip_state=state, container=container,
                                base_array=str(base), base_sha256=base_sha, anchor=source['anchors'][i],
                                derived_anchor=derived[i], video_path=str(path) if container == 'h264' else None,
                                video_sha256=video_sha if container == 'h264' else None)
                            retain(row)
            print(json.dumps({'completed_source': name, 'cases': len(rows), 'generators': generators, 'detectors': detectors}), flush=True)
    m.require(len(rows) == 640 and generators == 64 and detectors == 645, 'All prospective cases/forwards required')
    result = {'schema': 'chipmunk.natural-visual-watermark-result/2', 'policy': POLICY,
        'freeze_sha256': m.digest(trial / 'freeze.json'), 'case_count': len(rows),
        'generator_forwards': generators, 'detector_forwards': detectors, 'load_receipt': loaded,
        'groups': aggregate(rows), 'journal_sha256': m.digest(journal),
        'by_payload': {str(i): aggregate([r for r in rows if r['payload'] == p]) for i, p in enumerate(PAYLOADS)},
        'by_photo_stratum': {k: aggregate([r for r in rows if r['kind'] == 'image' and r['source_stratum'] == k]) for k in sources.QUOTAS},
        'provider_authenticated': False, 'used_in_confidence': False, 'early_stop_eligible': False,
        'serving_admitted': False, 'outside_human_reviewers': 0}
    m.write(trial / 'result.json', result)
    print(json.dumps({'complete': len(rows), 'groups': result['groups']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('freeze', 'run'))
    p.add_argument('trial', type=Path)
    p.add_argument('--source', type=Path)
    a = p.parse_args()
    if a.action == 'freeze':
        print(json.dumps(freeze(a.trial, a.source), indent=2))
    else:
        run(a.trial)
