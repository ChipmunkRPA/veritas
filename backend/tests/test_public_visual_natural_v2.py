"""Lossless evidence, strict inspection point, source clocks and clip denominators."""
import hashlib
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from scripts import prepare_public_visual_natural_v2 as sources
from scripts import review_public_visual_watermark_v1 as independent
from scripts import run_public_visual_natural_v2 as study


def test_native_float_bytes_survive_plane_storage_including_signed_zero_and_subnormal():
    raw = np.tile(np.array([0, 0x80000000, 1, 0x80000001, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0x00800000], '<u4'), 8192)
    values = raw.view('<f4').reshape(256, 256)
    planes = study.byteplanes(values)
    restored = planes.T.copy().view('<f4').reshape(256, 256)
    assert values.tobytes() == restored.tobytes()
    assert hashlib.sha256(values.tobytes()).digest() == hashlib.sha256(restored.tobytes()).digest()


@pytest.mark.parametrize('pixels,flagged', [(0, False), (499, False), (500, True)])
def test_original_strict_positive_and_component_point_is_preserved(pixels, flagged):
    logits = np.zeros((256, 256), np.float32)
    if pixels:
        logits[:20, :25] = 1
        if pixels == 499:
            logits[19, 24] = 0
    payload = study.PAYLOADS[3]
    bits = np.broadcast_to(np.array(payload, bool)[:, None, None], (32, 256, 256))
    evidence = study.evidence(logits, bits, payload)
    packed = np.packbits(bits.reshape(32, -1), axis=1, bitorder='little')
    mask, regions, decoded = independent.independent(logits, packed)
    assert evidence['selected_pixels'] == mask.sum() == pixels
    assert evidence['flagged_for_inspection'] is flagged and bool(regions) is flagged
    assert evidence['decoded_message'] == decoded == (payload if pixels else None)
    assert evidence['provider_authenticated'] is evidence['early_stop_eligible'] is evidence['used_in_confidence'] is False


def test_outside_interval_flag_does_not_count_as_recovering_a_marked_interval():
    def row(state, flagged):
        logits = np.full((256, 256), -1, np.float32)
        if flagged:
            logits[:20, :25] = 1
        bits = np.zeros((32, 256, 256), bool)
        return {'kind': 'video', 'container': 'h264', 'clip_state': 'partial', 'carrier': 'V00', 'state': state,
                'evidence': study.evidence(logits, bits, study.PAYLOADS[0]),
                'overlap': study.m.overlap(logits > 0, study.authored.truth(state))}
    group = study.aggregate([row('partial', False), row('partial', False), row('unmarked', True)])[0]
    assert group['cases'] == 3 and group['sources'] == 1
    assert group['sources_with_any_flag'] == 1 and group['unmarked_flagged'] == 1
    assert group['marked_missed'] == 2 and group['marked_sources_with_no_marked_frame_flag'] == 1


def test_original_and_derived_clocks_are_distinct_and_replayable(tmp_path):
    from scripts.review_public_visual_natural_v2 import source_replay

    path = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i', 'testsrc2=s=320x240:r=25:d=5',
                    '-c:v', 'libx264', '-threads', '1', str(path)], check=True, timeout=30)
    frames, anchors, geometry = sources.video(path)
    assert frames.shape == (8, 256, 256, 3) and len({a['source_frame_index'] for a in anchors}) == 8
    assert anchors[1]['derived_time_s'] == .5 and anchors[1]['source_time_s'] == .48
    assert all(float(a['source_pts'] * Fraction(a['source_time_base'])) == a['source_time_s'] for a in anchors)
    assert np.array_equal(frames, source_replay({'kind': 'video', 'asset': str(path), 'anchors': anchors, 'geometry': geometry}))


def test_arbitrary_external_routes_are_rejected_before_any_download(tmp_path):
    with pytest.raises(ValueError, match='Wikimedia'):
        sources.download('https://example.com/private-file', tmp_path / 'unexpected', 100)
    assert not list(tmp_path.iterdir())
