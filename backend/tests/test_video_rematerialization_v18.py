"""Byte, role-independent clock, and exact edit-boundary reconstruction guards."""
import hashlib
from copy import deepcopy

import pytest

from scripts.materialize_video_specialists_v18 import verify_case
from scripts.prepare_video_specialists_v18 import frame_clock, truth


def fixture(tmp_path, *, rate=24, label=1):
    path = tmp_path / 'authored-payload.mp4'
    path.write_bytes(b'Authored byte-binding test; not actual encoded video')
    table = {'truncated': False, 'interpolated': [], 'pts': [i * 512 for i in range(120)],
        'width': 640, 'height': 480, 'rotation': 0, 'time_base': f'1/{512 * rate}'}
    actual = {'group': 'authored-parent', 'id': 'authored-clip', 'family': 'authored-swap' if label else 'jump-cut',
        'label': label, 'status': 'ready', 'fps': 24, 'frame_count': 120,
        'inserted_frames': [24, 48] if label else None, 'edit_s': [1., 2.] if label else None,
        'exact_frames_verified': True}
    expected = {'group': actual['group'], 'source_id': actual['id'], 'family': actual['family'], 'label': label,
        'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_geometry': frame_clock(table, 120, expected_fps=None), 'known_edit': truth(actual)}
    return expected, actual, path, table


def test_exact_declared_bytes_clock_and_truth_are_preserved(tmp_path):
    verify_case(*fixture(tmp_path))


@pytest.mark.parametrize('change', ['bytes', 'identity', 'label', 'clock', 'boundary'])
def test_a_rebuilt_case_cannot_silently_change(change, tmp_path):
    expected, actual, path, table = fixture(tmp_path)
    if change == 'bytes':
        path.write_bytes(b'Different bytes with an unchanged source identifier')
    elif change == 'identity':
        actual['id'] = 'another-clip'
    elif change == 'label':
        actual['label'] = 0
    elif change == 'clock':
        table = deepcopy(table)
        table['time_base'] = '1/12800'
    else:
        actual.update(inserted_frames=[24, 60], edit_s=[1., 2.5])
    with pytest.raises(ValueError):
        verify_case(expected, actual, path, table)


def test_genuine25fps_control_keeps_actual_clock(tmp_path):
    verify_case(*fixture(tmp_path, rate=25, label=0))


def test_a_positive_cannot_borrow_a25fps_control_clock(tmp_path):
    with pytest.raises(ValueError, match='24fps clock'):
        verify_case(*fixture(tmp_path, rate=25, label=1))
