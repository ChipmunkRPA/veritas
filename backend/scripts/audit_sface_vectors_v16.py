"""Private retained-vector/source association and outer-window arithmetic review.

This loads no detector or inference framework. It never exports real-face vectors.
The float64 arithmetic cap1e-10 is an additional diagnostic check, not a changed
model operating point, native tolerance, selection rule or advancement requirement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

TOLERANCE = 1e-10


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    shared = max(0., min(ax + aw, bx + bw) - max(ax, bx)) * max(0., min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - shared
    return shared / union if union > 0 else 0.


def cosine_distance(a, b):
    an = math.sqrt(math.fsum(float(x) ** 2 for x in a))
    bn = math.sqrt(math.fsum(float(x) ** 2 for x in b))
    if min(an, bn) <= 1e-12:
        raise ValueError('Degenerate pooled vector in independent arithmetic')
    cosine = math.fsum(float(x) * float(y) for x, y in zip(a, b, strict=True)) / (an * bn)
    return 1. - max(-1., min(1., cosine))


def replay(item, packet):
    table = item['legacy']['table']
    times = [float(int(p) * Fraction(table['time_base'])) for p in table['pts']]
    millis = Counter(round(t * 1000) for t in times)
    track_boxes = defaultdict(list)
    for exhibit in item['legacy']['exhibits']:
        if exhibit.get('kind') != 'face_track':
            continue
        tid = exhibit['track']['track_id']
        for row in exhibit['frames']:
            i = row['frame_index']
            if 0 <= i < len(times) and row['pts'] == table['pts'][i]:
                track_boxes[i].append((tid, [row['bbox'][k] for k in ('x', 'y', 'w', 'h')]))
    features, associations = {}, defaultdict(set)
    for row in packet['frame_inputs']:
        i = row['frame_index']
        if row['pts'] != table['pts'][i] or row['time_base'] != table['time_base']:
            raise ValueError('Private feature packet changed native source clock')
        if 'private_sface_feature' not in row:
            continue
        raw = np.asarray(row['private_sface_feature'], np.float32)
        if raw.shape != (128,) or hashlib.sha256(raw.tobytes()).hexdigest() != row['sface_feature_sha256']:
            raise ValueError('Retained native float32 vector bytes changed')
        vector = raw.astype(np.float64)
        norm = math.sqrt(math.fsum(float(x) ** 2 for x in vector))
        if not np.isfinite(vector).all() or norm <= 1e-12:
            raise ValueError('Nonfinite/zero private vector')
        features[i] = vector / norm
        if i in table['interpolated'] or millis[round(times[i] * 1000)] != 1:
            continue
        choices = sorted(((iou(row['bbox'], box), tid) for tid, box in track_boxes[i]), reverse=True)
        if choices and choices[0][0] >= .55 and (len(choices) == 1 or choices[0][0] - choices[1][0] >= .12):
            associations[choices[0][1]].add(i)
    centers = {r['frame_index'] for r in item['legacy']['points'] if 0 < r['frame_index'] < len(times) - 1}
    output = {}
    for tid, indices in associations.items():
        blocks = []
        for i in sorted(centers):
            if {i - 1, i, i + 1} <= indices and times[i + 1] - times[i - 1] <= .13:
                blocks.append((i, (features[i - 1] + features[i] + features[i + 1]) / 3))
        if len(blocks) < 7:
            continue
        prefix = np.vstack([np.zeros(128), np.cumsum([v for _, v in blocks], axis=0)])
        for begin in range(2, len(blocks) - 4):
            for stop in range(begin + 3, len(blocks) - 1):
                first, last = times[blocks[begin][0]], times[blocks[stop - 1][0]]
                if last - first > 2.6:
                    break
                if last - first < .52:
                    continue
                context = blocks[begin - 2:stop + 2]
                if any(times[b[0]] - times[a[0]] > .55 for a, b in zip(context, context[1:])):
                    continue
                left = (prefix[begin] - prefix[begin - 2]) / 2
                middle = (prefix[stop] - prefix[begin]) / (stop - begin)
                right = (prefix[stop + 2] - prefix[stop]) / 2
                components = [cosine_distance(middle, left), cosine_distance(middle, right), cosine_distance(left, right)]
                key = (tid, first, last, tuple(i for i, _ in context))
                output[key] = {'response': min(components[:2]) - .35 * components[2],
                    'middle_to_before': components[0], 'middle_to_after': components[1], 'flank_change': components[2]}
    return output, len(features)


def audit(trial, output):
    panel = json.loads((trial / 'panel.json').read_text())
    cases, frames, retained, maxima, profile_cases = [], 0, 0, 0., 0
    for role in ('calibration', 'development'):
        path = trial / (role + '.jsonl')
        records = {r['id']: r for r in map(json.loads, path.read_text().splitlines())}
        expected = [r for r in panel['items'] if r['role'] == role]
        if set(records) != {r['id'] for r in expected}:
            raise ValueError('Exact complete phase IDs required for private vector audit')
        for item in expected:
            row = records[item['id']]
            if row['status'] == 'failed':
                raise ValueError('Inference failure; no full private-vector audit completion')
            if digest(Path(item['source_path'])) != item['sha256']:
                raise ValueError('Frozen video bytes changed after inference')
            packet_path = trial / ('private-features-' + role) / (item['id'] + '.json')
            if digest(packet_path) != row['private_feature_packet_sha256']:
                raise ValueError('Retained private source packet changed')
            packet = json.loads(packet_path.read_text())
            if packet['source_sha256'] != item['sha256'] or row['source_sha256'] != item['sha256']:
                raise ValueError('Retained vector/source labels lost exact byte binding')
            windows, count = replay(item, packet)
            actual = row['candidate_windows']
            if len(actual) != min(12, len(windows)):
                raise ValueError('Retained window count does not match full allowed search')
            errors = []
            for w in actual:
                key = (w['track_id'], w['start_s'], w['end_s'], tuple(w['centers']))
                if key not in windows:
                    raise ValueError('A retained window crosses source/track/cadence boundaries')
                errors.extend(abs(w[k] - windows[key][k]) for k in ('response', 'middle_to_before', 'middle_to_after', 'flank_change'))
            delta = max(errors, default=0.)
            if delta > TOLERANCE:
                raise ValueError('Separate float64 pooled-distance arithmetic differs')
            if actual:
                best = sorted((w['response'] for w in windows.values()), reverse=True)
                if any(abs(w['response'] - value) > TOLERANCE for w, value in zip(actual, best, strict=False)):
                    raise ValueError('Retained maxima differ from the complete allowed search')
            cases.append({'id': item['id'], 'private_packet_sha256': row['private_feature_packet_sha256'],
                          'features_verified': count, 'windows_verified': len(actual), 'max_difference': delta})
            frames += count
            retained += len(actual)
            maxima = max(maxima, delta)
            profile_cases += int(row.get('native_alignment_all_exact', False))
    result = {'schema': 'chipmunk.sface-private-vector-arithmetic/16', 'passed': True, 'cases': len(cases),
        'native_vector_bytes_verified': frames, 'retained_windows_verified': retained,
        'all_allowed_source_track_cadence_windows_searched': True, 'native_profile_passed_cases': profile_cases,
        'float64_arithmetic_tolerance': TOLERANCE, 'max_separate_component_difference': maxima,
        'raw_vectors_exported': False, 'native_neural_inference_repeated': False, 'outside_human_review': False,
        'case_packet_digests': cases,
        'scope': 'Separate private vector-byte/source/track binding and prefix-sum/cosine arithmetic. No pretrained model inference or public biometric-vector export. Existing declared model/native/quality gates unchanged.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'case_packet_digests'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.trial, args.output)
