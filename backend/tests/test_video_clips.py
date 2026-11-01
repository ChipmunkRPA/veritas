from __future__ import annotations

import hashlib
import io
import json
import struct
import subprocess
from dataclasses import replace
from copy import deepcopy

import pytest
from test_video_evidence_packets import vfr_source as vfr_source

from app.evidence.mp4 import InvalidMP4, inspect_mp4, validate_clip_binding
from app.evidence.video_clips import ClipUnavailable, build_clip, select_range
from app.detectors.video.decode import probe_frames


@pytest.fixture(scope='module')
def clip_fixture(vfr_source):
    path, table = vfr_source
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    data, asset, context = build_clip(path, table, source_sha256=sha, focus_index=2,
                                     sampled_frame_indices=[1, 2, 3], timeline_origin_s=5.0)
    return path, table, data, asset, context


def test_real_vfr_nonzero_source_origin_preserved_in_silent_clip(clip_fixture, tmp_path):
    _, table, data, asset, context = clip_fixture
    assert context['source_pts'] == list(table.pts)
    assert context['source_pts'][0] == 5000
    assert context['clip_pts'] == [pts - 5000 for pts in table.pts]
    assert len(set(b - a for a, b in zip(context['clip_pts'], context['clip_pts'][1:]))) > 1
    assert context['start_playback_s'] == 0 and context['start_stream_s'] == 5
    assert context['transform']['reencoded'] is True and context['transform']['audio'] == 'omitted'
    assert context['last_frame_duration']['basis'] == 'inferred_previous_source_interval'
    assert context['last_frame_duration']['next_source_frame_delta_ticks'] is None
    assert context['sampled_frame_indices'] == [1, 2, 3]
    parsed = inspect_mp4(io.BytesIO(data))
    assert parsed == asset
    path = tmp_path / 'context.mp4'
    path.write_bytes(data)
    probed = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_frames', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True, timeout=20)
    video = json.loads(probed.stdout)
    assert len(video['streams']) == 1 and video['streams'][0]['codec_name'] == 'h264'
    assert [frame['pts'] for frame in video['frames']] == context['clip_pts']
    assert parsed['has_audio'] is False


@pytest.mark.parametrize('bad', ['source_hash', 'interpolated', 'duplicate_pts'])
def test_untrustworthy_source_clocks_do_not_produce_clips(vfr_source, bad):
    path, table = vfr_source
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if bad == 'source_hash':
        sha = '0' * 64
    elif bad == 'interpolated':
        table = replace(table, interpolated=(2,))
    else:
        table = replace(table, pts=(table.pts[0], table.pts[0], *table.pts[2:]))
    with pytest.raises(ClipUnavailable):
        build_clip(path, table, source_sha256=sha, focus_index=2, sampled_frame_indices=[2], timeline_origin_s=5)


def test_long_high_rate_selection_is_bounded_and_centered(vfr_source):
    _, table = vfr_source
    table = replace(table, pts=tuple(range(-5000, 15000, 5)), pict_types='I' * 4000)
    start, end = select_range(table, 2000)
    assert end - start == 480 and start <= 2000 < end
    assert table.seconds(end - 1) - table.seconds(start) < 8


def test_real_rotated_source_exports_display_orientation_without_hidden_matrix(tmp_path):
    plain, rotated = tmp_path / 'plain.mp4', tmp_path / 'rotated.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=96x64:r=10:d=0.6',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(plain)], check=True, timeout=30)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-display_rotation', '90', '-i', str(plain), '-c', 'copy', str(rotated)], check=True, timeout=30)
    table = probe_frames(rotated)
    assert (table.width, table.height) == (64, 96) and abs(table.rotation) == 90
    data, asset, context = build_clip(rotated, table, source_sha256=hashlib.sha256(rotated.read_bytes()).hexdigest(),
                                      focus_index=2, sampled_frame_indices=[2], timeline_origin_s=0)
    assert (asset['width'], asset['height']) == (64, 96)
    assert inspect_mp4(data) == asset
    assert context['source_pts'] == list(table.pts)


def test_mp4_inspector_rejects_offsets_extra_tracks_and_unbounded_boxes(clip_fixture):
    _, _, data, _, _ = clip_fixture
    # A composition-time table is forbidden, even if it replaces an otherwise
    # valid box; source/clip correspondence must not depend on hidden offsets.
    if b'stss' in data:
        with pytest.raises(InvalidMP4, match='offsets'):
            inspect_mp4(data.replace(b'stss', b'ctts', 1))
    with pytest.raises(InvalidMP4):
        inspect_mp4(struct.pack('>I4sQ', 1, b'moov', 2 ** 63) + b'\0' * 32)
    with pytest.raises(InvalidMP4):
        inspect_mp4(data[:-20])
    with pytest.raises(InvalidMP4, match='track'):
        inspect_mp4(data.replace(b'vide', b'soun', 1))
    with pytest.raises(InvalidMP4):
        inspect_mp4(data.replace(b'udta', b'trak', 1))
    with pytest.raises(InvalidMP4, match='external'):
        inspect_mp4(data.replace(b'url \0\0\0\1', b'url \0\0\0\0', 1))


def test_inspector_never_reads_compressed_payload(clip_fixture):
    _, _, data, expected, _ = clip_fixture
    class MetadataOnly(io.BytesIO):
        def read(self, size=-1):
            assert 0 <= size <= 256 * 1024
            start = self.tell()
            result = super().read(size)
            # The mdat body begins after its type field. Header reads are fine.
            payload = data.index(b'mdat') + 4
            assert not payload <= start < len(data), 'Inspector read compressed video bytes'
            return result
    assert inspect_mp4(MetadataOnly(data)) == expected


def _bound_packet(clip_fixture):
    path, table, data, asset, context = clip_fixture
    packet = {'clip': deepcopy(context), 'artifacts': {'context_clip': {**deepcopy(asset), 'sha256': hashlib.sha256(data).hexdigest(), 'media_type': 'video/mp4'}},
              'frames': [{'anchor': {'frame_index': i, 'pts': table.pts[i]}} for i in (1, 2, 3)]}
    source = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'time_base': table.time_base, 'frame_count': len(table),
              'width': table.width, 'height': table.height, 'timeline_origin_s': 5.0}
    return packet, source


def test_binding_checks_actual_frame_clocks_and_keeps_legacy_absence_valid(clip_fixture):
    packet, source = _bound_packet(clip_fixture)
    validate_clip_binding(packet, source)
    validate_clip_binding({'artifacts': {}}, source)
    validate_clip_binding({'clip': {'status': 'unavailable', 'reason': 'Unverified source time'}, 'artifacts': {}}, source)


@pytest.mark.parametrize('change', ['source_clock', 'clip_clock', 'samples', 'source_hash', 'playback', 'audio', 'dimensions', 'duration_basis', 'duration_ticks', 'false_unavailable', 'bad_type', 'no_binding'])
def test_binding_rejects_tampered_clocks_coverage_and_derivation(clip_fixture, change):
    packet, source = _bound_packet(clip_fixture)
    clip, asset = packet['clip'], packet['artifacts']['context_clip']
    if change == 'source_clock': clip['source_pts'][0] += 1
    elif change == 'clip_clock': asset['pts'][1] += 1
    elif change == 'samples': clip['sampled_frame_indices'] = list(range(len(clip['source_pts'])))
    elif change == 'source_hash': clip['source_sha256'] = '0' * 64
    elif change == 'playback': clip['start_playback_s'] += 5
    elif change == 'audio': asset['has_audio'] = True
    elif change == 'dimensions': clip['transform']['width'] += 2
    elif change == 'duration_basis': clip['last_frame_duration']['basis'] = 'exact_source_duration'
    elif change == 'duration_ticks': asset['duration_ticks'] += 1
    elif change == 'false_unavailable': packet['clip'] = {'status': 'unavailable', 'reason': 'No clip'}
    elif change == 'bad_type': clip['source_pts'] = None
    else: packet.pop('clip')
    with pytest.raises(InvalidMP4): validate_clip_binding(packet, source)
