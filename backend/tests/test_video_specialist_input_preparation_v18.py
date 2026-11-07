"""Exact source-clock/truth checks; no decoding or model inference."""
from copy import deepcopy

import pytest

from scripts.prepare_video_specialists_v18 import frame_clock, truth


def table():
    return {'truncated': False, 'interpolated': [], 'pts': [i * 512 for i in range(120)],
        'width': 640, 'height': 480, 'rotation': 0, 'time_base': '1/12288'}


def test_actual_rational_source_clock_maps_to_integer_nanoseconds():
    geometry = frame_clock(table(), 120)
    assert geometry['duration_ns'] == 5_000_000_000
    assert geometry['frame_count'] == 120
    row = {'status': 'ready', 'fps': 24, 'label': 1, 'frame_count': 120,
        'inserted_frames': [30, 66], 'edit_s': [1.25, 2.75], 'exact_frames_verified': True}
    assert truth(row) == {'frames': [30, 66], 'time_ns': [1_250_000_000, 2_750_000_000]}


@pytest.mark.parametrize('change', ['truncated', 'interpolated', 'missing', 'pts_shift', 'wrong_rate', 'rotated'])
def test_inferred_or_wrong_clock_is_refused(change):
    value = deepcopy(table())
    if change == 'truncated':
        value['truncated'] = True
    elif change == 'interpolated':
        value['interpolated'] = [4]
    elif change == 'missing':
        value['pts'].pop()
    elif change == 'pts_shift':
        value['pts'][30] += 1
    elif change == 'wrong_rate':
        value['time_base'] = '1/15360'
    else:
        value['rotation'] = 90
    with pytest.raises(ValueError):
        frame_clock(value, 120)


def test_a_genuine_ordinary_edit_has_no_ai_interval():
    assert truth({'status': 'ready', 'fps': 24, 'label': 0, 'frame_count': 112,
        'inserted_frames': None, 'edit_s': None, 'exact_frames_verified': True}) is None


def test_ordinary_control_uses_actual25fps_pts_instead_of_nominal24fps():
    value = table()
    value.update(time_base='1/12800', pts=[i * 512 for i in range(141)])
    geometry = frame_clock(value, 141, expected_fps=None)
    assert geometry['native_frame_rate'] == '25'
    assert geometry['duration_ns'] == 5_640_000_000
    assert geometry['duration_ns'] != 141 * 10**9 / 24
    # An inserted swap with24fps truth must not silently accept the other clock.
    with pytest.raises(ValueError):
        frame_clock(value, 141)


def test_positive_truth_requires_exact_source_frames_and_agreeing_seconds():
    row = {'status': 'ready', 'fps': 24, 'label': 1, 'frame_count': 120,
        'inserted_frames': [30, 66], 'edit_s': [1.25, 2.75], 'exact_frames_verified': False}
    with pytest.raises(ValueError):
        truth(row)
    row.update(exact_frames_verified=True, edit_s=[1.25, 3.0])
    with pytest.raises(ValueError):
        truth(row)
