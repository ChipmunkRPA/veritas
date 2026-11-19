from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.canonical import canonical_hash
from app.detectors.video import lip_shot_guard, viseme_review
from app.detectors.video.decode import FrameTable
from app.evidence import video_lip_agreement as agreement
from app.evidence import video_viseme_binding as binding


@pytest.fixture
def recording():
    pts = agreement.NATIVE25_PTS
    table = FrameTable(time_base='1/90000', pts=pts, key_frames=(0,), pict_types='I' + 'P' * 74,
                       width=360, height=288, coded_width=360, coded_height=288)
    mapping = {'source_codec': 'aac', 'source_profile': 'LC', 'source_container': 'mov,mp4,m4a,3gp,3g2,mj2',
               'source_sample_rate': 44_100, 'source_channels': 1, 'source_time_base': '1/44100', 'source_start_pts': 0,
               'source_duration_pts': 131_330, agreement.AAC_44100_CLOCK: True, 'samples': 47_648,
               'timeline_time_base': '1/16000', 'timeline': [{'start_sample': 0, 'end_sample': 47_648, 'source_pts': 0}],
               'pcm_sha256': 'd' * 64}
    return table, mapping


def _evidence(table, mapping, record, packets=()):
    return {'source': {'width': table.width, 'height': table.height, 'time_base': table.time_base,
                       'frame_count': len(table), 'timeline_origin_s': 0, binding.COMMITMENT: canonical_hash(record)},
            'audio': {'source': mapping}, binding.KEY: record, 'packets': list(packets)}


def test_guard_adds_a_persistent_change_and_rejects_a_flash_or_aligned_illumination():
    left = (np.zeros((2, 2, 3)), np.array([.74, .26]))
    right = (np.full((2, 2, 3), 20), np.array([.40, .60]))
    rng = np.random.default_rng(8)
    a, b = (rng.integers(10, 180, (41, 50), dtype=np.uint8) for _ in range(2))
    content, histogram = lip_shot_guard._difference(left, right)
    assert 13 < content < 27 and histogram < .35
    assert lip_shot_guard._supplemental([left] * 3 + [right] * 3, [a] * 3 + [b] * 3, 3, content, histogram)
    assert not lip_shot_guard._supplemental([left] * 3 + [right, left, left], [a] * 3 + [b, a, a], 3, content, histogram)
    assert not lip_shot_guard._supplemental([left] * 3 + [right] * 3, [a] * 3 + [a + 15] * 3, 3, content, histogram)


def test_native25_cut_proof_blocks_packets_and_rejects_changed_clocks(recording):
    table, mapping = recording
    cut = binding.shot_blocked(agreement.shot_guard_record([37]))
    assert list(binding.exhibits(cut)) == []
    evidence = _evidence(table, mapping, cut)
    binding.validate(evidence, timestamps=table.pts, committed=cut)
    with pytest.raises(ValueError, match='native picture profile'):
        binding.validate(evidence, timestamps=table.pts[:-1] + (table.pts[-1] + 1,))
    with pytest.raises(ValueError, match='native sound and picture clocks'):
        binding.validate(_evidence(table, mapping | {'source_duration_pts': 131_329}, cut), timestamps=table.pts)
    with pytest.raises(ValueError, match='must be withheld'):
        binding.validate(_evidence(table, mapping, cut, [{'id': 'VLT01'}]))
    malformed = binding.shot_blocked(agreement.shot_guard_record([37, 37]))
    with pytest.raises(ValueError, match='native picture profile'):
        binding.validate(_evidence(table, mapping, malformed))
    historical = cut | {'schema': binding.SCHEMA_WITH_AAC441_MOTION}
    with pytest.raises(ValueError, match='dedicated timing version'):
        binding.validate(_evidence(table, mapping, historical))


def test_guarded_motion_is_opt_in_and_never_runs_across_a_detected_cut(recording, tmp_path, monkeypatch):
    table, mapping = recording
    reference = tmp_path / 'geometry.task'
    reference.touch()
    for name in ('vision', 'shared'):
        (tmp_path / f'{name}.safetensors').touch()
    for key, value in {'video_lip_aac441_motion_review': True, 'video_lip_aac441_native25_review': False,
                       'video_lip_silent_mouth_review': True,
                       'video_lip_timing_model_dir': None, 'video_lip_timing_runtime': None,
                       'video_lip_timing_face_asset': reference, 'video_lip_motion_sound_models': tmp_path}.items():
        monkeypatch.setattr(viseme_review.settings, key, value)
    ctx = SimpleNamespace(deadline=SimpleNamespace(remaining=lambda: 50), store=SimpleNamespace(path_for=lambda _: tmp_path / 'sound.wav'))
    frames = SimpleNamespace(path=tmp_path / 'source.mp4', table=table)
    args = (ctx, frames, mapping, {'wav_sha256': '1' * 64}, [[0, len(table)]])
    monkeypatch.setattr(lip_shot_guard, 'qualified_delivery', lambda *_: True)
    checked = []

    def scan(*_):
        checked.append(True)
        return agreement.shot_guard_record([37] if len(checked) == 1 else [])

    monkeypatch.setattr(lip_shot_guard, 'scan', scan)
    assert viseme_review.assess(*args, shot_frames=len(table)) == binding.skipped('unsupported')
    assert not checked
    monkeypatch.setattr(viseme_review.settings, 'video_lip_aac441_native25_review', True)

    def do_not_compare(*_):
        pytest.fail('The motion comparison ran across a recorded cut')

    monkeypatch.setattr(viseme_review, '_observe', do_not_compare)
    assert viseme_review.assess(*args, shot_frames=len(table)) == binding.shot_blocked(agreement.shot_guard_record([37]))

    def compare(request, _):
        assert request['motion_only'] and request['shot_ranges'] is None
        assert request['checkpoint'] is None and not request['articulation']
        return {'frames': [{'index': i, 'pts': p, 'gap_units': None} for i, p in enumerate(table.pts)],
                'phones': [], agreement.AAC_44100_MOTION_STATUS: 'no_local_dip'}

    monkeypatch.setattr(viseme_review, '_observe', compare)
    record = viseme_review.assess(*args, shot_frames=len(table))
    assert record['schema'] == binding.SCHEMA_WITH_NATIVE25_AAC441_MOTION and record['status'] == 'no_match'
    assert record['affects_overall_confidence'] is False and record[agreement.NATIVE25_SHOT_KEY]['boundary_frames'] == []
    binding.validate(_evidence(table, mapping, record))
    change = deepcopy(record)
    change[agreement.NATIVE25_SHOT_KEY]['boundary_frames'].append(37)
    with pytest.raises(ValueError, match='must be withheld'):
        binding.validate(_evidence(table, mapping, change))

    legacy_shots = [[0, 37], [37, len(table)]]
    assert viseme_review.supported_shots(table, mapping, legacy_shots)
    monkeypatch.setattr(lip_shot_guard, 'scan', do_not_compare)
    monkeypatch.setattr(viseme_review.settings, 'video_lip_timing_face_asset', None)
    assert viseme_review.assess(*args[:-1], legacy_shots, shot_frames=len(table)) == binding.skipped('assets')
