"""Tiny authored native-clock regression; no real-media detection evidence."""
import hashlib
import subprocess
from fractions import Fraction

from app.detectors.video.decode import probe_frames
from scripts.prepare_video_specialists_v18 import ordinary


def test_jump_cut_preserves24fps_and_exact_frame_count(tmp_path):
    source, target = tmp_path / 'authored.mp4', tmp_path / 'ordinary.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
        'testsrc2=size=64x64:rate=24:duration=5', '-an', '-c:v', 'libx264', '-qp', '0',
        '-threads', '1', str(source)], check=True, timeout=15)
    group = next(f'authored-{i}' for i in range(30)
        if hashlib.sha256(f'authored-{i}'.encode()).digest()[2] % 3 == 0)
    row = {'id': 'authored', 'group': group, 'label': 0, 'family': 'original_videos',
        'frame_count': 120, 'inserted_frames': [24, 60], 'edit_s': None, 'fps': 24, 'status': 'ready'}
    result = ordinary(row, source, target)
    assert result['family'] == 'jump-cut' and result['frame_count'] == 112
    table = probe_frames(target, timeout_s=15).to_json()
    assert len(table['pts']) == 112 and not table['interpolated'] and not table['truncated']
    assert all((b - a) * Fraction(table['time_base']) == Fraction(1, 24)
        for a, b in zip(table['pts'], table['pts'][1:]))
    assert result['label'] == 0 and result['edit_s'] is None
