"""Compare cached PNG production components with prior regional prototypes.

This exploratory subset includes fitting sources and is never qualification.
It loads retained native maps, source images and exact masks, not model weights.
JPEG source preparation is included in v11; this PNG cache cannot evaluate it.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import prepare_image_supported_v11 as source
from scripts.replay_image_supported_v11 import replay
from scripts.specialist_source_tail_v1 import predict

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('global-regions', 'condition-regions')
CODE = ('backend/scripts/preview_image_supported_v11.py', 'backend/scripts/prepare_image_supported_v11.py',
    'backend/scripts/replay_image_supported_v11.py', 'backend/scripts/specialist_source_tail_v1.py',
    'backend/app/bench/specialists.py', 'backend/app/detectors/image/splice.py', 'backend/app/detectors/image/fused.py',
    'backend/app/detectors/image/adapters/localisers.py', 'backend/app/detectors/image/adapters/common.py',
    'backend/app/detectors/image/localiser.py', 'backend/app/evidence/image_eligibility.py',
    'backend/scripts/localization_benchmark.py')


def summarize(rows, name):
    missing = sum(row[name + '_probability'] is None for row in rows) if name in ARMS else sum(not row['current_reliable_cells'] for row in rows)
    genuine, ai = [r for r in rows if not r['label']], [r for r in rows if r['label']]
    def flagged(row):
        return row[name + '_flag']
    return {'planned': len(rows), 'genuine': len(genuine), 'ai': len(ai), 'unassessed': missing,
        'ai_flagged': sum(flagged(row) for row in ai), 'ai_not_flagged_including_unassessed': sum(not flagged(row) for row in ai),
        'false_ai': sum(flagged(row) for row in genuine),
        'ai_groups_flagged': len({r['group'] for r in ai if flagged(r)}),
        'genuine_groups_flagged': len({r['group'] for r in genuine if flagged(r)}),
        'note': 'A missing flag is not an original-file verdict. Current unassessed count means no reliable localization cells; prototype unassessed count means no calibrated file score.'}


def preview(main, trial):
    import cv2

    from scripts.localization_benchmark import mask_metrics

    cv2.setNumThreads(1)
    plan_path = trial / 'source-plan.json'
    plan = json.loads(plan_path.read_text())
    source.require(plan['preparation_code_sha256'] == source.sha(Path(source.__file__)), 'Input preparation implementation changed')
    public = ROOT / 'docs/evaluation-panels'
    tail_release = json.loads((public / 'specialist-source-tail-v1-release.json').read_text())
    artifact_path = public / 'specialist-source-tail-v1-artifacts.json'
    source.require(source.sha(artifact_path) == tail_release['files_sha256'][str(artifact_path.relative_to(ROOT))], 'Prior frozen candidate artifacts/points changed')
    artifacts = json.loads(artifact_path.read_text())['image-specialists-v10']
    image_release = json.loads((public / 'image-specialists-v10-release.json').read_text())
    manifest_path = ROOT / image_release['manifests']
    source.require(source.sha(manifest_path) == image_release['files_sha256'][image_release['manifests']], 'Prior source/response manifests changed')
    manifests = json.loads(gzip.decompress(manifest_path.read_bytes()))
    files = {arm: {f['id']: f for f in manifests[arm]['files']} for arm in ARMS}
    entries = {entry['item']['id']: entry for group in plan['groups'] for entry in group['items']}
    selected = [row for row in plan['files'] if row['format'] == 'png' and row['item_id'] in plan['png_caches']]
    source.require(len(selected) == 144, 'Complete cached subset denominator required')
    failure_path = trial / 'cached-preview-attempt1-failure.json'
    frozen = {'schema': 'chipmunk.image-supported-cached-preview-freeze/11.2',
        'source_plan_sha256': source.sha(plan_path), 'code_sha256': {name: source.sha(ROOT / name) for name in CODE},
        'candidate_artifacts_sha256': source.sha(artifact_path), 'original_manifest_sha256': source.sha(manifest_path),
        'prior_failure_sha256': source.sha(failure_path) if failure_path.exists() else None,
        'amendment': 'Use the original float64 regional-mean contract; no tolerance relaxation, point change, model call or source replacement.',
        'scope': 'All144 cached PNG cases/72 consumed pairs, including model-calibration fitting roles. Actual current localiser postprocessing only, not full application verdict or independent validation.',
        'new_neural_forwards': 0, 'qualification_allowed': False, 'serving_admitted': False}
    source.write(trial / 'cached-preview-v2-freeze.json', frozen)
    rows = []
    for row in selected:
        entry = entries[row['item_id']]
        data, pixels, truth = source.source_item(entry)
        source.require(hashlib.sha256(data).hexdigest() == row['input_sha256'], 'Cached preview source bytes changed')
        cached = plan['png_caches'][row['item_id']]
        path = Path(cached['path'])
        source.require(source.sha(path) == cached['file_sha256'] and cached['input_sha256'] == row['input_sha256'], 'Cached native response identity changed')
        native = np.load(path, allow_pickle=False)
        boxes = [u['box'] for u in row['units'] if u['target'] == 'region']
        signal, selected_pixels = replay(data, pixels, 'png', native, boxes=boxes)
        record = {**{k: row[k] for k in ('id', 'group', 'role', 'label', 'input_sha256')},
            'native_map_sha256': signal['native_map_sha256'], 'current_flag': signal['current_region_flag'],
            'current_reliable_cells': signal['current_reliable_cells'], 'current_gate': signal['current_gate'],
            'current_region_count': signal['current_region_count'], 'current_selected_source_pixels': signal['current_selected_source_pixels'],
            'current_source_mask_sha256': hashlib.sha256(selected_pixels.tobytes()).hexdigest(),
            'current_mask_metrics': mask_metrics(selected_pixels, truth)}
        for arm in ARMS:
            file = files[arm][row['id']]
            source.require(file['input_sha256'] == row['input_sha256'] and file['group'] == row['group']
                and file['role'] == row['role'] and file['label'] == row['label'], 'Prototype source/role binding changed')
            expected = {'whole': float(np.quantile(native, .95)),
                **{f'quadrant.{i}': signal['raw_values'][f'quadrant.{i}'] for i in range(4)}}
            for unit in file['units']:
                source.require(unit['status'] == 'ok' and unit['raw'] == expected[unit['id']],
                    'Actual retained native map does not reproduce prior prototype raw readout')
            artifact, point = artifacts[arm]['artifact'], artifacts[arm]['operating_point']
            probability = predict(file, artifact)
            record[arm + '_probability'] = probability
            record[arm + '_flag'] = probability is not None and point['status'] == 'fitted' and probability > point['threshold']
        rows.append(record)
    source.write(trial / 'cached-preview-v2-cases.json', rows)
    roles = sorted({r['role'] for r in rows})
    report = {'schema': 'chipmunk.image-supported-cached-preview/11.2', 'freeze_sha256': source.sha(trial / 'cached-preview-v2-freeze.json'),
        'cases_sha256': source.sha(trial / 'cached-preview-v2-cases.json'), 'cached_cases': len(rows),
        'connected_groups': len({row['group'] for row in rows}), 'source_roles': dict(Counter(row['role'] for row in rows)),
        'all': {name: summarize(rows, name) for name in ('current', *ARMS)},
        'by_role': {role: {name: summarize([r for r in rows if r['role'] == role], name)
            for name in ('current', *ARMS)} for role in roles},
        'current_edited_mean_pixel_f1': float(np.mean([r['current_mask_metrics']['pixel_f1'] for r in rows if r['label']])),
        'current_edited_mean_pixel_iou': float(np.mean([r['current_mask_metrics']['iou'] for r in rows if r['label']])),
        'new_neural_forwards': 0, 'qualification_allowed': False, 'serving_admitted': False,
        'required_next': 'Complete source-disjoint-role comparison on all supported PNG and JPEG inputs and frozen new per-format calibration. This cache includes fitting sources and has no JPEG responses; it cannot admit or reject a final branch-calibration design.',
        'limits': frozen['scope']}
    source.write(trial / 'cached-preview-v2.json', report)
    print(json.dumps({'schema': report['schema'], 'all': report['all'], 'source_roles': report['source_roles'],
        'new_neural_forwards': 0, 'qualification_allowed': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    preview(args.main, args.trial)
