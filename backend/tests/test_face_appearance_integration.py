from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest

from app.core.canonical import canonical_json
from app.core.config import settings
from app.detectors.video import face_appearance as method
from app.detectors.video.adapters.face_appearance import DETECTOR, FaceAppearanceReviewDetector, record
from app.engine.fusion.runtime import annotate
from app.engine.registry import get_registry
from app.evidence.face_interval_binding import validate_packet, validate_sequence_record
from app.evidence.image_packets import PacketIntegrityError
from app.evidence.video_packets import _sequence_observations
from tests.test_face_sequence_integration import sample


def appearance_record(response=.2):
    table, old, source = sample()
    for frame in old['frame_inputs']:
        frame['alignment_side'] = 112
    keys = sum(old['centers'].values(), [])
    value = record({'centers': keys, 'track_id': 'track:0', 'start_s': old['start_stream_s'],
        'end_s': old['end_stream_s'], 'response': response, 'middle_to_before': response + .035,
        'middle_to_after': response + .1, 'flank_change': .1}, table,
        {r['frame_index']: r for r in old['frame_inputs']})
    return table, value, source


def test_qualified_math_and_native_alignment_are_copied_without_refitting():
    root = Path(__file__).parents[1]
    actual = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(
        (root / 'app/detectors/video/face_appearance.py').read_text()).body if isinstance(n, ast.FunctionDef)}
    for name, functions in [('sface_transition_v16.py', {'unit', 'distance', 'windows'}),
                            ('sface_native_v15.py', {'validate', 'cpp_alignment', 'sum_sequential', 'face_row'})]:
        expected = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(
            (root / 'scripts' / name).read_text()).body if isinstance(n, ast.FunctionDef) and n.name in functions}
        assert {k: actual[k] for k in functions} == expected
    assert method.REVIEW_POINT == .11243900750879454 and method.FLANK_PENALTY == .35


@pytest.mark.parametrize('offset', [1e-9, 1e-8, 1e-7, .1])
def test_strict_point_survives_six_decimal_canonical_serialization(offset):
    import json

    table, value, source = appearance_record(method.REVIEW_POINT + offset)
    canonical = json.loads(canonical_json(value))
    validate_sequence_record(canonical, source, table_pts=table.pts)
    assert float.fromhex(canonical['measurement_hex']['response']) > method.REVIEW_POINT


@pytest.mark.parametrize('offset', [0., -1e-9, -.01])
def test_equal_or_lower_point_is_not_a_flag(offset):
    table, value, source = appearance_record(method.REVIEW_POINT + offset)
    validate_sequence_record(value, source, table_pts=table.pts, require_flag=False)
    with pytest.raises(ValueError, match='inconsistent arithmetic'):
        validate_sequence_record(value, source, table_pts=table.pts)


@pytest.mark.parametrize('damage', ['missing_neighbor', 'wrong_pts', 'bad_clock', 'bad_crop', 'bad_matrix',
    'wrong_math', 'wrong_hex', 'nan', 'wrong_method', 'missing_hex', 'wrong_native_size'])
def test_appearance_commitments_refuse_changed_inputs_and_arithmetic(damage):
    table, value, source = appearance_record()
    if damage == 'missing_neighbor':
        value['frame_inputs'].pop()
    elif damage == 'wrong_pts':
        value['frame_inputs'][0]['pts'] += 1
    elif damage == 'bad_clock':
        value['time_base'] = '1/30'
    elif damage == 'bad_crop':
        value['frame_inputs'][0]['aligned_rgb_sha256'] = 'invalid'
    elif damage == 'bad_matrix':
        value['frame_inputs'][0]['alignment_matrix_hex'][0][0] = 'nan'
    elif damage == 'wrong_math':
        value['measurement_hex']['middle_to_before'] = .3.hex()
        value['middle_to_before'] = .3
    elif damage == 'wrong_hex':
        value['measurement_hex']['response'] = .5.hex()
    elif damage == 'nan':
        value['response'] = float('nan')
    elif damage == 'wrong_method':
        value['method'] = 'unknown/1'
    elif damage == 'missing_hex':
        value.pop('measurement_hex')
    elif damage == 'wrong_native_size':
        value['frame_inputs'][0]['alignment_side'] = 224
    with pytest.raises((ValueError, KeyError)):
        validate_sequence_record(value, source, table_pts=table.pts)


def test_source_roles_crop_positions_and_method_are_bound_without_frame_confidence():
    table, value, source = appearance_record()
    anchors = {r['frame_index']: deepcopy(r) for r in value['frame_inputs']}
    details = {'face_sequence_binding_version': method.METHOD, 'face_sequence': value, 'face_sequence_windows': [value]}
    candidates, _ = _sequence_observations({'signals': [{'id': DETECTOR + ':global', 'status': 'ok', 'details': details}]},
        table, source, anchors)
    assert len(candidates) == 1
    rows = candidates[0]['rows']
    assert [r['interval_role'] for r in rows] == ['before', 'during', 'after']
    assert all(r['score'] is None for r in rows)
    packet = {'semantics': method.METHOD, 'evidence_type': 'anomaly', 'claim_scope': 'sampled_window',
        'face_sequence': value, 'window': candidates[0]['window'], 'frames': [
            {'interval_role': r['interval_role'], 'score': None, 'is_peak': False,
             'anchor': {k: anchors[r['frame_index']][k] for k in ('frame_index', 'pts', 'decoded_rgb_sha256')},
             'context_region': {'x': 10, 'y': 8, 'w': 32, 'h': 24}} for r in rows]}
    validate_packet(packet, source, table_pts=table.pts)
    packet['semantics'] = 'face-sequence-change/1'
    with pytest.raises(ValueError, match='misrepresents'):
        validate_packet(packet, source, table_pts=table.pts)
    details['face_sequence_binding_version'] = 'face-sequence-change/1'
    with pytest.raises(PacketIntegrityError, match='method differs'):
        _sequence_observations({'signals': [{'status': 'ok', 'details': details}]}, table, source, anchors)


def test_only_one_opted_in_method_and_commercial_assets_with_rollback(monkeypatch):
    registry = get_registry()
    for mode, sequence, appearance in [('legacy', False, False), ('sequence', True, False), ('appearance', False, True)]:
        monkeypatch.setattr(settings, 'video_face_review_mode', mode)
        assert registry.decision('vid.review.face_sequence').enabled == sequence
        assert registry.decision(DETECTOR).enabled == appearance
    entry = registry.manifest.detector(DETECTOR)
    assert not entry.hypotheses and entry.legacy_weight == 0
    assert all(registry.manifest.model(mid).licence.class_ == 'commercial' for mid in entry.models)
    assert FaceAppearanceReviewDetector().spec.family == 'face-sequence-review'


def test_inspection_response_never_acquires_probability_or_fusion_weight():
    signal = {'id': DETECTOR + ':global', 'detector_id': DETECTOR, 'raw_score': .9, 'raw_kind': 'score',
        'effective_weight': 0, 'calibrated_probability': None, 'log_lr': None}
    [value] = annotate([signal], {}, artefact_detectors=['existing.classifier'])
    assert value['effective_weight'] == 0 and value['calibrated_probability'] is None and value['log_lr'] is None
    assert value['contribution_logit'] is None and value['contribution_pp'] is None
