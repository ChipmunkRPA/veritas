"""A source audit must refuse changed truth, counters, maps and masks."""
import hashlib
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from scripts.localization_benchmark import mask_metrics
from scripts.prepare_image_supported_v11 import units
from scripts.replay_image_supported_v11 import replay
from scripts.review_image_supported_v11 import load_map, verify_case
from tests.test_image_supported_v11 import pixels_and_bytes, ring


def authored_case(branch):
    pixels, data = pixels_and_bytes(branch)
    raw = ring(384 if branch == 'png' else 512)
    truth = np.zeros(pixels.shape[:2], bool)
    truth[2:12, 3:20] = True
    boxes = [[0, 0, 64, 48], [64, 0, 64, 48], [0, 48, 64, 48], [64, 48, 64, 48]]
    plan = {'format': branch, 'units': units(truth, boxes)}
    record, selected = replay(data, pixels, branch, raw, primary=.8 if branch == 'jpeg' else None, boxes=boxes)
    record['postprocessing_neural_forwards'] = record.pop('new_neural_forwards')
    record['postprocessing_weights_loaded'] = record.pop('weights_loaded')
    record['current_mask_metrics'] = mask_metrics(selected, truth)
    record['current_source_mask_sha256'] = hashlib.sha256(selected.tobytes()).hexdigest()
    return plan, record, data, pixels, truth, raw


@pytest.mark.parametrize('branch', ['png', 'jpeg'])
def test_actual_source_replay_uses_no_model_weights(branch):
    values = authored_case(branch)
    with patch('torch.load', side_effect=AssertionError('No weights may load')):
        assert verify_case(*values)['changed_truth_pixels'] == 170


@pytest.mark.parametrize('change', ['local_truth', 'map_readout', 'mask', 'neural_count'])
def test_changed_source_or_measurement_is_refused(change):
    plan, record, *rest = authored_case('png')
    if change == 'local_truth':
        plan = deepcopy(plan)
        plan['units'][-1]['label'] = 1
    elif change == 'map_readout':
        record['raw_values']['quadrant.1'] += .001
    elif change == 'mask':
        record['current_source_mask_sha256'] = '0' * 64
    else:
        record['postprocessing_neural_forwards'] = 1
    with pytest.raises(ValueError):
        verify_case(plan, record, *rest)


def test_native_file_and_tensor_hashes_are_both_required(tmp_path):
    raw = ring(384)
    path = tmp_path / 'map.npy'
    np.save(path, raw, allow_pickle=False)
    record = {'raw_map_path': str(path), 'raw_map_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'native_map_sha256': hashlib.sha256(raw.tobytes()).hexdigest()}
    assert np.array_equal(load_map(record, 'png'), raw)
    raw[0, 0] = .9
    np.save(path, raw, allow_pickle=False)
    with pytest.raises(ValueError, match='map file changed'):
        load_map(record, 'png')
    record['raw_map_file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='tensor content changed'):
        load_map(record, 'png')
