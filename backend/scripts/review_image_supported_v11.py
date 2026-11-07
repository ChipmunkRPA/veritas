"""Replay every completed image response against exact source pixels and masks.

No model weights or new neural inference. This source/postprocessing audit is
separate from numerical refitting and from outside human review.
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
from scripts import run_image_supported_v11 as frozen


def load_map(record, branch):
    path = Path(record['raw_map_path'])
    source.require(source.sha(path) == record['raw_map_file_sha256'], 'Original native map file changed')
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as stream:
            raw = np.load(stream, allow_pickle=False)
    else:
        raw = np.load(path, allow_pickle=False)
    frozen.verify_map(raw, branch)
    source.require(hashlib.sha256(raw.tobytes()).hexdigest() == record['native_map_sha256'], 'Native tensor content changed')
    return raw


def verify_case(plan_row, record, data, rgb, truth, raw):
    from scripts.localization_benchmark import mask_metrics
    from scripts.replay_image_supported_v11 import replay

    boxes = [unit['box'] for unit in plan_row['units'] if unit['target'] == 'region']
    source.require(source.units(truth, boxes) == plan_row['units'], 'Local truth does not match original changed pixels')
    packet, mask = replay(data, rgb, plan_row['format'], raw,
        primary=record['raw_values']['primary'] if plan_row['format'] == 'jpeg' else None, boxes=boxes)
    for key in ('format', 'input_sha256', 'rgb_sha256', 'native_map_sha256', 'raw_values',
            'current_region_flag', 'current_region_count', 'current_regions', 'current_gate',
            'current_grid_shape', 'current_reliable_cells', 'current_selected_source_pixels', 'source_width', 'source_height'):
        source.require(packet[key] == record[key], 'Source-bound production postprocessing changed: ' + key)
    source.require(mask_metrics(mask, truth) == record['current_mask_metrics']
        and hashlib.sha256(mask.tobytes()).hexdigest() == record['current_source_mask_sha256'],
        'Current connected-pixel selection or source-mask metrics changed')
    source.require(record['postprocessing_neural_forwards'] == 0 and record['postprocessing_weights_loaded'] is False,
        'Cached postprocessing was misreported as native model inference')
    return {'changed_truth_pixels': int(truth.sum()), 'selected_source_pixels': int(mask.sum())}


def review(trial, output):
    from unittest.mock import patch

    import cv2

    from app.detectors.image import fused, splice

    _, plan = frozen.check(trial)
    pipeline = json.loads((trial / 'pipeline-state.json').read_text())
    source.require(pipeline['status'] == 'completed' and source.sha(trial / 'result.json') == pipeline['result_sha256'],
        'Wait for complete source-bound native scoring and review')
    records, receipts = frozen.responses(trial, True)
    cv2.setNumThreads(1)
    counts = Counter()

    def forbidden(*args, **kwargs):
        raise AssertionError('No weights may be loaded by this source/postprocessing audit')

    with patch.object(fused, 'load_model', forbidden), patch.object(splice, 'load_model', forbidden):
        # Independently recompute the original pair's RGB change mask before
        # checking each format. JPEG compression never adds manipulation truth.
        for group in plan['groups']:
            items = sorted(group['items'], key=lambda item: item['item']['label'])
            original, changed = [source.source_item(entry) for entry in items]
            exact = np.any(original[1] != changed[1], axis=-1)
            source.require(np.array_equal(exact, changed[2]) and not original[2].any(), 'Original pair changed-pixel truth mismatch')
            counts['original_pair_pixels'] += exact.size
        for row in plan['files']:
            record = records[row['id']]
            data, rgb, truth = frozen.input_for(plan, row)
            raw = load_map(record, row['format'])
            verify_case(row, record, data, rgb, truth, raw)
            counts[row['format'] + '_source_bound_postprocessing_cases'] += 1
            if sum(counts[key] for key in ('png_source_bound_postprocessing_cases', 'jpeg_source_bound_postprocessing_cases')) % 120 == 0:
                print(json.dumps({'source_bound_cases': sum(counts[key] for key in
                    ('png_source_bound_postprocessing_cases', 'jpeg_source_bound_postprocessing_cases')), 'new_neural_forwards': 0}), flush=True)
    native_total = 0
    profiles = {}
    for role in frozen.engine.ROLES:
        for branch in ('png', 'jpeg'):
            name = role + '-' + branch
            receipt = json.loads((trial / (name + '-native.json')).read_text())
            native = sum(row['native_inference_forwards'] for row in records.values() if row['role'] == role and row['format'] == branch)
            source.require(receipt['model_weights_loaded'] is True and receipt['new_neural_forwards'] == native
                + 2 * len(receipt['native_reference_checks']), 'Native versus cached forward counters are inconsistent')
            native_total += receipt['new_neural_forwards']
            profile = json.loads((trial / (name + '-profile/profile.json')).read_text())
            source.require(profile['child_exit_code'] == 0 and profile['guard'] is None, 'Failed/guarded native attempt cannot disappear')
            profiles[name] = {k: v for k, v in profile.items() if k != 'arguments'}
    result = {'schema': 'chipmunk.image-supported-source-postprocessing-review/11',
        'implementation_sha256': source.sha(Path(__file__)), 'freeze_sha256': source.sha(trial / 'freeze.json'),
        'completed_result_sha256': pipeline['result_sha256'], 'actual_role_receipts_sha256': receipts,
        'original_pairs': len(plan['groups']), **counts, 'actual_native_forward_count_all_roles': native_total,
        'resource_profiles': profiles, 'this_review_new_neural_forwards': 0, 'this_review_weights_loaded': False,
        'all_actual_sources_maps_local_labels_and_component_masks_replayed_exactly': True,
        'serving_admitted': False,
        'limits': 'All consumed sources. Exact changed-pixel masks are not proof of AI authorship. Current component overlap, quarter-presence calibration, native inference, numerical refit and outside human review are distinct. No fresh or whole-app verification.'}
    source.write(output, result)
    print(json.dumps({'schema': result['schema'], **counts, 'all_cases_exact': True, 'new_neural_forwards': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    review(args.trial, args.output)
