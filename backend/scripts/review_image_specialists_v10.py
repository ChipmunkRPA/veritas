"""Independent source-pixel and per-format review; no model loading or fitting."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def sha(data):
    return hashlib.sha256(data).hexdigest()


def brier(rows):
    values = []
    for label in (0, 1):
        groups = defaultdict(list)
        for truth, probability, group in rows:
            if truth == label and probability is not None:
                groups[group].append((probability - truth)**2)
        if not groups:
            return None
        values.append(sum(sum(v) / len(v) for v in groups.values()) / len(groups))
    return sum(values) / 2


def calls(rows):
    result = {'genuine': 0, 'ai': 0, 'false_ai': 0, 'missed_ai': 0, 'unassessed_or_tied': 0}
    for label, p, _ in rows:
        result['ai' if label else 'genuine'] += 1
        if p is None or abs(p - .5) <= 1e-12:
            result['unassessed_or_tied'] += 1
        elif label == 0 and p > .5:
            result['false_ai'] += 1
        elif label == 1 and p < .5:
            result['missed_ai'] += 1
    return {**result, 'source_class_balanced_brier': brier(rows)}


def source_audit(plan):
    files = {r['id']: r for r in plan['files']}
    verified, pixels = 0, 0
    for group in plan['groups']:
        loaded = []
        for entry in group['items']:
            item = entry['item']
            raw = (Path(entry['root']) / item['image']).read_bytes()
            if sha(raw) != item['image_sha256']:
                raise ValueError('Original image bytes changed')
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != 'PNG' or image.size != (group['width'], group['height']):
                    raise ValueError('Native source format or geometry changed')
                rgb = np.asarray(image.convert('RGB')).copy()
            loaded.append((raw, rgb, item))
        diff = np.any(loaded[0][1] != loaded[1][1], axis=-1)
        edited = loaded[1][2]
        mask = (Path(group['items'][1]['root']) / edited['mask']).read_bytes()
        if sha(mask) != edited['mask_sha256']:
            raise ValueError('Reference bytes changed')
        with Image.open(io.BytesIO(mask)) as image:
            if not np.array_equal(np.asarray(image.convert('L')) > 127, diff):
                raise ValueError('Reference does not equal exact original/edit RGB changes')
        pixels += diff.size
        for original, rgb, item in loaded:
            truth = diff if item['label'] else np.zeros(diff.shape, bool)
            for condition in ('native', 'jpeg75'):
                row = files[item['id'] + '-' + condition]
                if condition == 'native':
                    data, decoded = original, rgb
                else:
                    stream = io.BytesIO()
                    Image.fromarray(rgb).save(stream, format='JPEG', quality=75)
                    data = stream.getvalue()
                    with Image.open(io.BytesIO(data)) as image:
                        decoded = np.asarray(image.convert('RGB')).copy()
                if row['input_sha256'] != sha(data) or row['rgb_sha256'] != sha(decoded.tobytes()):
                    raise ValueError('Prospective input byte/RGB hash differs from independent reconstruction')
                width, height = group['width'], group['height']
                expected = [[0, 0, width, height], [0, 0, width // 2, height // 2],
                    [width // 2, 0, width - width // 2, height // 2],
                    [0, height // 2, width // 2, height - height // 2],
                    [width // 2, height // 2, width - width // 2, height - height // 2]]
                if [u['box'] for u in row['units']] != expected:
                    raise ValueError('Source coverage coordinates changed')
                for unit in row['units']:
                    x, y, w, h = unit['box']
                    if unit['label'] != int(truth[y:y + h, x:x + w].any()):
                        raise ValueError('File truth was copied to a genuine region or edit truth was lost')
                    verified += 1
    return {'original_pairs': len(plan['groups']), 'source_truth_pixels_verified': pixels,
        'input_files_reconstructed': len(plan['files']), 'local_truth_boxes_verified': verified}


def review(trial, output):
    plan = json.loads((trial / 'plan.json').read_text())
    audit = source_audit(plan)
    report = {'schema': 'chipmunk.image-specialists-source-condition-review/10', 'source_audit': audit,
        'plan_sha256': sha((trial / 'plan.json').read_bytes()), 'code_sha256': sha(Path(__file__).read_bytes()),
        'arms': {}, 'new_model_forwards': 0, 'serving_admitted': False}
    for arm in ('global-image', 'condition-image', 'global-regions', 'condition-regions'):
        result = json.loads((trial / f'{arm}-confirmation.json').read_text())
        original = json.loads((trial / f'{arm}-manifest-before-confirmation.json').read_text())
        rows = {f['id']: f for f in original['files']}
        local_truth = {(f['id'], u['id']): u['label'] for f in original['files'] for u in f['units']}
        report['arms'][arm] = {}
        for condition in ('native', 'jpeg75'):
            chosen = [r for r in result['outcomes'] if r['id'].endswith('-' + condition)]
            files = [(r['label'], r['probability'], r['group']) for r in chosen]
            regions = [(local_truth[r['file_id'], r['unit_id']], r['probability'], rows[r['file_id']]['group'])
                for r in result['unit_outcomes'] if r['file_id'].endswith('-' + condition)
                and r['unit_id'].startswith('quadrant.')]
            report['arms'][arm][condition] = {'files': calls(files), 'regions': calls(regions) if regions else None}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    review(args.trial, args.output)
