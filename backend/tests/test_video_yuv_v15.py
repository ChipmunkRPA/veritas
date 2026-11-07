"""Native sample/clock integrity and honest renderer observations."""
import copy
import hashlib

import numpy as np
import pytest

from scripts.reproduce_video_yuv_v15 import compare_renderers, native_samples, validate_source


def metadata():
    image = {'width': 128, 'height': 64, 'pix_fmt': 'yuv420p',
             'color_space': 'bt709', 'color_range': 'tv'}
    return {'streams': [{**image, 'time_base': '1/1000'}],
            'frames': [{**image, 'best_effort_timestamp': i * 125} for i in range(12)]}


def test_all_planar_samples_have_a_fixed_commitment_and_correct_plane_order():
    raw = native_samples()
    assert len(raw) == 147456
    assert hashlib.sha256(raw).hexdigest() == '2025fc4531fbd91e0148b7b26b17181157c0e80ede8b1361d9c5a561f9a02cf4'
    assert raw[:16] == bytes([80]) * 16
    assert raw[8192:8200] == bytes([90]) * 8
    assert raw[10240:10248] == bytes([220]) * 8
    # Second frame moves exactly two luma / one chroma samples, not two chroma.
    assert raw[12288:12292] == bytes([190, 190, 80, 80])
    assert raw[12288 + 8192:12288 + 8192 + 3] == bytes([70, 90, 90])


def test_complete_native_recovery_and_all_exact_clocks_are_accepted():
    raw = native_samples()
    validate_source(raw, raw, metadata())


@pytest.mark.parametrize('change', ['one_sample', 'truncated', 'extra_frame'])
def test_native_corruption_is_never_a_complete_source(change):
    raw = native_samples()
    recovered = {'one_sample': bytes([raw[0] ^ 1]) + raw[1:],
                 'truncated': raw[:-1], 'extra_frame': raw + raw[:12288]}[change]
    with pytest.raises(ValueError, match='recover exactly'):
        validate_source(raw, recovered, metadata())


@pytest.mark.parametrize('change', ['reordered', 'rounded_clock', 'wrong_range',
                                  'wrong_matrix', 'wrong_size', 'missing_frame'])
def test_metadata_or_timeline_mismatch_cannot_be_ignored(change):
    meta = copy.deepcopy(metadata())
    if change == 'reordered':
        meta['frames'][1:3] = reversed(meta['frames'][1:3])
    elif change == 'rounded_clock':
        meta['frames'][3]['best_effort_timestamp'] += 1
    elif change == 'wrong_range':
        meta['frames'][3]['color_range'] = 'pc'
    elif change == 'wrong_matrix':
        meta['streams'][0]['color_space'] = 'smpte170m'
    elif change == 'wrong_size':
        meta['streams'][0]['width'] += 1
    else:
        meta['frames'].pop()
    raw = native_samples()
    with pytest.raises(ValueError):
        validate_source(raw, raw, meta)


def test_unknown_renderer_is_recorded_without_forcing_a_known_matrix():
    arrays = [np.full((64, 128, 3), value, np.uint8) for value in (1, 2, 3, 4)]
    result = compare_renderers(arrays[0], arrays[1:])
    assert result['opencv_matches'] == []
    assert result['metadata_default_equals_bt709'] is False
    assert result['explicit_matrices_differ'] is True


def test_different_platform_observation_is_valid_and_not_a_failed_old_hypothesis():
    a = np.ones((64, 128, 3), dtype=np.uint8)
    b = np.zeros_like(a)
    result = compare_renderers(a, [a, b, a])
    assert result['opencv_matches'] == ['metadata_default', 'bt709']
    assert result['metadata_default_equals_bt709'] is True


def test_nonbyte_or_incomplete_renderers_cannot_supply_comparison_hashes():
    a = np.ones((64, 128, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compare_renderers(a.astype('float32'), [a, a, a])
    with pytest.raises(ValueError):
        compare_renderers(a, [a, a])
