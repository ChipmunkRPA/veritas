"""Authored source-coordinate and real-postprocessing regressions; no weights."""
import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from scripts.prepare_image_supported_v11 import bind_role, encoded, route, units
from scripts.replay_image_supported_v11 import replay


def pixels_and_bytes(branch='png', flat=False):
    pixels = np.full((96, 128, 3), 128, np.uint8) if flat else np.random.default_rng(41).integers(60, 190, (96, 128, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    data, actual = encoded(pixels, stream.getvalue(), branch)
    return actual, data


def ring(side):
    values = np.full((side, side), .1, np.float32)
    values[side // 6:5 * side // 6, side // 6:5 * side // 6] = .9
    values[side // 3:2 * side // 3, side // 3:2 * side // 3] = .1
    return values


def test_byte_signature_routing_uses_real_encoded_content():
    _, png = pixels_and_bytes()
    _, jpeg = pixels_and_bytes('jpeg')
    assert route(png) == 'png' and route(jpeg) == 'jpeg'
    with pytest.raises(ValueError):
        replay(jpeg, pixels_and_bytes('jpeg')[0], 'png', ring(384))


def test_local_truth_is_not_whole_image_truth_and_jpeg_noise_is_not_ai():
    mask = np.zeros((96, 128), bool)
    mask[1, 1] = True
    boxes = [(0, 0, 64, 48), (64, 0, 64, 48), (0, 48, 64, 48), (64, 48, 64, 48)]
    assert [u['label'] for u in units(mask, boxes)] == [1, 1, 1, 0, 0, 0]
    original, _ = pixels_and_bytes()
    compressed, _ = pixels_and_bytes('jpeg')
    assert np.any(original != compressed)
    assert not any(u['label'] for u in units(np.zeros_like(mask), boxes))


def test_connected_pixels_preserve_holes_instead_of_filling_region_boxes():
    pixels, data = pixels_and_bytes()
    with patch('torch.load', side_effect=AssertionError('No model weights may load in replay')):
        packet, selected = replay(data, pixels, 'png', ring(384))
    assert packet['current_gate']['eligible'] is True and packet['current_region_count'] == 1
    assert selected.any() and not selected[48, 64]
    assert packet['new_neural_forwards'] == 0 and packet['weights_loaded'] is False


def test_jpeg_gate_does_not_erase_raw_map_or_regional_measurements():
    pixels, data = pixels_and_bytes('jpeg')
    raw = ring(512)
    with patch('torch.load', side_effect=AssertionError('No model weights may load in replay')):
        low, rejected = replay(data, pixels, 'jpeg', raw, primary=.4)
        high, accepted = replay(data, pixels, 'jpeg', raw, primary=.6)
    assert not rejected.any() and accepted.any()
    assert low['native_map_sha256'] == high['native_map_sha256']
    assert all(low['raw_values'][f'quadrant.{i}'] == high['raw_values'][f'quadrant.{i}'] for i in range(4))
    assert low['current_region_flag'] is False and high['current_region_flag'] is True


def test_genuine_flat_area_withholds_regions_without_an_originality_verdict():
    pixels, data = pixels_and_bytes(flat=True)
    packet, selected = replay(data, pixels, 'png', ring(384))
    assert packet['current_gate']['eligible'] is True
    assert packet['current_reliable_cells'] == 0 and not selected.any()
    assert packet['current_region_flag'] is False and 'verdict' not in packet


def test_replay_refuses_different_source_pixels():
    pixels, data = pixels_and_bytes()
    pixels[0, 0, 0] += 1
    with pytest.raises(ValueError, match='exact decoded source'):
        replay(data, pixels, 'png', ring(384))


@pytest.mark.parametrize('corruption', ['dtype', 'nonfinite'])
def test_replay_refuses_corrupt_native_map(corruption):
    pixels, data = pixels_and_bytes()
    raw = ring(384).astype(np.float64) if corruption == 'dtype' else ring(384)
    if corruption == 'nonfinite':
        raw[0, 0] = np.nan
    with pytest.raises(ValueError, match='native model map'):
        replay(data, pixels, 'png', raw)


def test_same_original_cannot_cross_calibration_and_confirmation():
    roles = {}
    bind_role(roles, 'same-original-bytes', 'specialist_cal')
    with pytest.raises(ValueError, match='ancestry crosses'):
        bind_role(roles, 'same-original-bytes', 'confirmation')


def test_regional_mean_preserves_constant_native_value_with_original_float64_contract():
    pixels, data = pixels_and_bytes()
    native = np.full((384, 384), .1, np.float32)
    packet, _ = replay(data, pixels, 'png', native)
    assert all(packet['raw_values'][f'quadrant.{i}'] == float(native[0, 0]) for i in range(4))
