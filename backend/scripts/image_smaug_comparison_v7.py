"""Research-only image panel, genuine-tail operating point and paired accounting.

No probability calibration, production change or pixel-localization claim is made.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

SALT = 'chipmunk-smaug-v7-development-2026-11-07'
FAMILIES = ('firefly', 'sd2', 'flux-fill')
TGIF = (
    'localization-eval/tgif-pairs-v2', 'localization-eval/tgif-confirmation-v3',
    'localization-eval/tgif-confirmation-v4', 'localization-eval/tgif-confirmation-v5',
    'independent-evaluation/image-fused-crop-v6/confirmation',
    'independent-evaluation/rule80-v1/tgif',
)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def order(value):
    return hashlib.sha256((SALT + ':' + value).encode()).hexdigest()


def coco_id(value):
    found = re.search(r'COCO_val2014_0*(\d+)', str(value))
    return 'COCO:' + str(int(found[1])) if found else None


def pixel_identity(path):
    with Image.open(path) as image:
        upright = ImageOps.exif_transpose(image).convert('RGB')
        size = upright.size
        sha = hashlib.sha256(str(size).encode() + upright.tobytes()).hexdigest()
        small = list(upright.convert('L').resize((9, 8), Image.Resampling.LANCZOS).getdata())
        bits = [small[y * 9 + x] > small[y * 9 + x + 1] for y in range(8) for x in range(8)]
        dhash = sum(int(bit) << i for i, bit in enumerate(bits))
    return {'oriented_size': list(size), 'rgb_sha256': sha, 'dhash64': f'{dhash:016x}'}


def prepare(main, output):
    """Select roles from labels/source identities only, without model responses."""
    output.mkdir(parents=True, exist_ok=False)
    data = main / 'backend/data'
    manifests, inventory, seen = {}, [], set()
    for directory in TGIF:
        path = data / directory / 'manifest.json'
        manifests[str(path.relative_to(main))] = digest(path)
        source = json.loads(path.read_text())
        for item in source['items']:
            key = (str(item['source_id']), item['family'], item['label'], item['image_sha256'])
            if key in seen:
                continue
            seen.add(key)
            inventory.append({'source_id': f'tgif:{item["family"]}:{item["source_id"]}:{item["label"]}',
                'alias': 'COCO:' + str(int(item['source_id'])), 'family': item['family'],
                'label': item['label'], 'scope': 'localized_edit' if item['label'] else 'genuine',
                'pool': 'tgif', 'path': str((path.parent / item['image']).relative_to(main)),
                'sha256': item['image_sha256'], 'source_manifest': str(path.relative_to(main)),
                'source_manifest_id': item['id'], 'mask_sha256': item.get('mask_sha256'),
                'source_terms': 'TGIF CC-BY-SA-4.0; underlying COCO source rights remain. Evaluation only; no images redistributed.'})
    rule = data / 'independent-evaluation/rule80-v1/image-manifest.json'
    manifests[str(rule.relative_to(main))] = digest(rule)
    whole_families = {'camera', 'FLUX.1-schnell', 'PixArt-Sigma-XL-2-1024-MS', 'stable-diffusion-3-medium-diffusers'}
    for item in json.loads(rule.read_text())['items']:
        if item['family'] not in whole_families or not item['path'].startswith('image-native/'):
            continue
        inventory.append({'source_id': 'rule80:' + item['id'], 'alias': coco_id(item['group']) or item['group'],
            'family': item['family'], 'label': item['label'], 'scope': 'whole_generation' if item['label'] else 'genuine',
            'pool': 'whole', 'path': str((rule.parent / item['path']).relative_to(main)),
            'sha256': item['sha256'], 'source_manifest': str(rule.relative_to(main)), 'source_manifest_id': item['id'],
            'source_terms': item['source_terms']})
    multi = data / 'independent-evaluation/multimodal/image-manifest.json'
    manifests[str(multi.relative_to(main))] = digest(multi)
    for item in json.loads(multi.read_text())['items']:
        label = int(item['label'] == 'ai')
        inventory.append({'source_id': 'multimodal:' + item['id'], 'alias': coco_id(item['source_group']) or item['source_group'],
            'family': item.get('generator') or 'camera', 'label': label, 'scope': 'whole_generation' if label else 'genuine',
            'pool': 'whole', 'path': str((main / 'backend' / item['path']).relative_to(main)),
            'sha256': item['sha256'], 'source_manifest': str(multi.relative_to(main)), 'source_manifest_id': item['id'],
            'source_terms': item['license'] + '; evaluation only, no raw redistribution.'})
    for item in inventory:
        path = main / item['path']
        if digest(path) != item['sha256']:
            raise ValueError('Historical source bytes changed')
        item.update(pixel_identity(path))
    parent = list(range(len(inventory)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    links = Counter()
    for i, a in enumerate(inventory):
        for j in range(i):
            b = inventory[j]
            reason = ('source_alias' if a['alias'] == b['alias'] else 'exact_rgb' if a['rgb_sha256'] == b['rgb_sha256']
                      else 'near_dhash_le4' if (int(a['dhash64'], 16) ^ int(b['dhash64'], 16)).bit_count() <= 4 else None)
            if reason:
                parent[find(i)] = find(j)
                links[reason] += 1
    components = defaultdict(list)
    for i, item in enumerate(inventory):
        components[find(i)].append(item)
    groups = {}
    for rows in components.values():
        group = order('|'.join(sorted(x['source_id'] for x in rows)))
        for row in rows:
            row['group'] = group
        groups[group] = rows
    eligible = {family: [] for family in FAMILIES}
    for group, rows in groups.items():
        if {x['pool'] for x in rows} == {'tgif'} and len(rows) == 2 and {x['label'] for x in rows} == {0, 1}:
            families = {x['family'] for x in rows}
            if len(families) == 1:
                eligible[next(iter(families))].append(group)
    selected = []
    for family in FAMILIES:
        candidates = sorted(eligible[family], key=order)
        if len(candidates) < 16:
            raise ValueError('Insufficient independent connected pairs')
        for role, group_ids in [('calibration', candidates[:8]), ('development', candidates[8:16])]:
            for group in group_ids:
                selected.extend({**x, 'role': role} for x in groups[group] if role != 'calibration' or x['label'] == 0)
    # All historical whole-image cases are development, never operating-point calibration.
    selected.extend({**x, 'role': 'development'} for x in inventory if x['pool'] == 'whole')
    if len({x['group'] for x in selected if x['role'] == 'calibration'} &
           {x['group'] for x in selected if x['role'] == 'development'}):
        raise ValueError('Connected source group crosses roles')
    inputs = []
    (output / 'jpeg75').mkdir()
    for row in sorted(selected, key=lambda x: (x['role'], order(x['source_id']))):
        sid = order(row['source_id'])[:24]
        for fmt in ('native', 'jpeg75'):
            path = main / row['path']
            if fmt == 'jpeg75':
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert('RGB')
                    stream = io.BytesIO()
                    image.save(stream, format='JPEG', quality=75, subsampling=2, optimize=False, progressive=False)
                path = output / 'jpeg75' / (sid + '.jpg')
                path.write_bytes(stream.getvalue())
            inputs.append({**row, 'id': sid + '-' + fmt, 'format': fmt, 'input_path': str(path.resolve()),
                           'input_sha256': digest(path), 'input_pixels': pixel_identity(path)})
    write(output / 'panel.json', {'schema': 'chipmunk.image-smaug-panel/7', 'salt': SALT, 'source_manifests': manifests,
        'inventory_rows': len(inventory), 'connected_components': len(components), 'link_counts': dict(links),
        'eligible_pair_groups': {f: len(v) for f, v in eligible.items()},
        'selection': '8 genuine-only calibration groups and 8 paired development groups per TGIF family; all 40 consumed whole-image cases as development; connected aliases/exact-RGB/dHash distance<=4 cannot cross roles.',
        'jpeg_recipe': {'quality': 75, 'subsampling': 2, 'optimize': False, 'progressive': False, 'exif': 'removed after upright RGB conversion'},
        'freshness': 'Already-consumed development sources. Source/near-duplicate grouping does not establish absence from pretrained-model training.',
        'inventory': inventory, 'items': inputs})


def select_gate(panel, candidate_rows):
    """Genuine-only max-tail rule. No development responses may enter selection."""
    by_id = {x['id']: x for x in candidate_rows}
    calibration = [x for x in panel['items'] if x['role'] == 'calibration']
    if len(calibration) != 48 or any(x['label'] != 0 for x in calibration):
        raise ValueError('Exact 24 genuine groups in two formats required')
    for item in calibration:
        if by_id[item['id']]['status'] != 'ok' or not math.isfinite(float(by_id[item['id']]['mean_logit'])):
            raise ValueError('Incomplete calibration; no gate or development admission')
    # Strict > max creates zero calibration flags. One shared gate for both formats.
    gate = max(float(by_id[x['id']]['mean_logit']) for x in calibration)
    if not math.isfinite(gate):
        raise ValueError('Nonfinite raw operating point')
    return {'kind': 'strictly_greater_than_max_of_48_genuine_responses', 'raw_mean_logit': gate,
            'calibration_inputs': 48, 'calibration_groups': 24,
            'note': 'Empirical genuine-tail operating point, not an AI probability, accuracy guarantee or public screening threshold.'}


def direction(result, assessment):
    """Keep file credentials separate from actual statistical customer interpretation."""
    d = result.get('decision') or {}
    return {'customer_direction': assessment['direction'], 'decision_code': d.get('code'),
            'statistical_direction': assessment['direction'] if assessment['direction'] in {'ai', 'original'} else 'inconclusive',
            'provenance_led': assessment['direction'] == 'record',
            'probability_ai': d.get('probability_ai_calibrated'),
            'statistical_estimate_before_provenance': d.get('statistical_estimate_before_provenance'),
            'fusion_mode': d.get('fusion_mode'), 'fuser_sha256': d.get('fuser_sha256')}


def account(items, baseline, candidate, gate):
    """Compare additional AI *indications*, never replace calibrated overall confidence."""
    b, c = {x['id']: x for x in baseline}, {x['id']: x for x in candidate}
    if len(b) != len(baseline) or len(c) != len(candidate) or set(b) != {x['id'] for x in items} or set(c) != set(b):
        raise ValueError('Exact planned IDs and one outcome per input required')
    cases = []
    for x in items:
        br, cr = b[x['id']], c[x['id']]
        old = br.get('interpretation', {}).get('statistical_direction') == 'ai' and br['status'] == 'done'
        cue = cr['status'] == 'ok' and cr['mean_logit'] > gate['raw_mean_logit']
        cases.append({**{k: x[k] for k in ('id', 'group', 'label', 'family', 'scope', 'format', 'role')},
            'baseline_ai': old, 'candidate_positive_cue': cue, 'combined_ai_indication': old or cue,
            'baseline_direction': br.get('interpretation', {}).get('customer_direction', 'failed'),
            'baseline_status': br['status'], 'candidate_status': cr['status'],
            'new_conflict_with_original': cue and br.get('interpretation', {}).get('statistical_direction') == 'original',
            'provenance_led': br.get('interpretation', {}).get('provenance_led', False)})
    return cases
