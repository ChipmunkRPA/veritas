from __future__ import annotations

import ast
import hashlib
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.canonical import canonical_json
from app.core.config import settings
from app.detectors.fakes import InMemoryExhibitWriter
from app.detectors.image.faces import Face
from app.detectors.video import face_appearance as method
from app.detectors.video import face_boundary
from app.detectors.video.adapters import face_appearance as adapter
from app.detectors.video.adapters.face_appearance import DETECTOR, FaceAppearanceReviewDetector, record
from app.detectors.video.adapters.face_sequence import SequenceInputs
from app.detectors.video.decode import DecodedFrame, DecodeError, FrameTable
from app.detectors.video.frame_store import FrameStore
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


def boundary_table(pts=tuple(range(145)), time_base="1/24"):
    return FrameTable(time_base, pts, (0,), "I" + "P" * (len(pts) - 1), 64, 64, 64, 64)


@pytest.mark.parametrize("pts", [
    tuple(11 + 3*i + (i % 4) for i in range(70)),
    tuple(-37 + 4 * (i // 2) for i in range(100)),
])
def test_boundary_grid_uses_native_times_and_earlier_ties(pts):
    table = boundary_table(pts, "1/100")
    times = [tick * table.tb for tick in table.pts]
    first, last = table.pts[20], table.pts[50]
    start, end = first * table.tb, last * table.tb
    expected = {}
    for position in range(int((times[-1] - times[0]) * face_boundary.CADENCE) + 1):
        target = times[0] + Fraction(position, face_boundary.CADENCE)
        nearest = min(range(len(times)), key=lambda index: (abs(target - times[index]), index))
        value = times[nearest]
        if start - face_boundary.PROTOTYPE_FAR <= value <= end + face_boundary.PROTOTYPE_FAR:
            expected[nearest] = value
    assert face_boundary.schedule(table, first, last) == expected
    assert face_boundary.schedule(table, first, first + 100)
    assert not face_boundary.schedule(table, first, first + 101)


@pytest.mark.parametrize("condition", ["swap", "original", "decode_error", "ambiguous_track", "unverified_edge"])
def test_optional_boundary_keeps_exact_coarse_result_and_only_binds_native_support(monkeypatch, tmp_path, condition):
    table = boundary_table()
    face = Face(8, 8, 48, 48, .99, [(22, 20), (41, 20), (32, 30), (24, 39), (40, 39)])
    boxes = [{"frame_index": i, "pts": p, "bbox": {"x": 8, "y": 8, "w": 48, "h": 48}}
             for i, p in enumerate(table.pts)]
    track = {"kind": "face_track", "track": {"track_id": "track:0"}, "frames": boxes}
    exhibits = (track,)
    if condition == "ambiguous_track":
        exhibits += ({**track, "track": {"track_id": "track:1"}, "frames": [boxes[64]]},)
    calls = []

    def decoder(_path, timeline, indices, *, timeout_s=None, **_):
        calls.append((indices, timeout_s))
        if timeout_s is not None and condition == "decode_error":
            raise DecodeError("unavailable additional read")
        for index in indices:
            pixels = np.full((64, 64, 3), index, dtype=np.uint8)
            yield DecodedFrame(index, timeline.pts[index], pixels, pts_verified=not (
                condition == "unverified_edge" and index == 64
            ))

    class Recognizer:
        def alignCrop(self, bgr, _detection):
            return np.full((112, 112, 3), bgr[0, 0, 0], dtype=np.uint8)

        def feature(self, crop):
            data = np.zeros(128, dtype=np.float32)
            changed = condition != "original" and 66 <= int(crop[0, 0, 0]) < 80
            data[int(changed)] = 1
            return data

    monkeypatch.setattr(adapter, "detect_faces", lambda *_args, **_kwargs: [face])
    recognizer = Recognizer()

    def run(enabled):
        monkeypatch.setattr(settings, "video_face_boundary_refinement", enabled)
        store = FrameStore(tmp_path / "source.mp4", table, max_bytes=1, decoder=decoder)
        inputs = SequenceInputs(table, store, tuple(range(6, len(table) - 1, 6)), exhibits)
        lease = SimpleNamespace(get=lambda _key: SimpleNamespace(obj=recognizer, revision="fixture"))
        ctx = SimpleNamespace(outputs=SimpleNamespace(face_review=inputs), lease=lease, deadline=None,
                              exhibits=InMemoryExhibitWriter())
        return FaceAppearanceReviewDetector().run(ctx)

    assert type(settings).model_fields["video_face_boundary_refinement"].default is False
    current = run(False)
    checked = run(True)
    refined = checked.details.pop("face_boundary", None)
    assert current.aggregate == checked.aggregate
    assert current.observed == checked.observed
    assert current.details == checked.details
    extra_reads = [indices for indices, timeout in calls if timeout is not None]
    if condition == "original":
        assert current.details["face_sequence"] is None and not extra_reads and refined is None
    elif condition != "swap":
        assert current.details["face_sequence"] is not None and len(extra_reads) == 1 and refined is None
    else:
        assert refined is not None
        assert refined["track_id"] == current.details["face_sequence"]["track_id"]
        assert (refined["start_pts"], refined["end_pts"]) == (66, 80)
        assert refined["anchors"] == {"before": 64, "during": 72, "after": 80}
        assert extra_reads == [refined["scheduled_frame_indices"]]
        assert len(extra_reads[0]) <= face_boundary.MAX_FRAMES
        for row in refined["frame_inputs"]:
            index = row["frame_index"]
            assert row["pts"] == index
            assert row["decoded_rgb_sha256"] == hashlib.sha256(bytes([index]) * (64 * 64 * 3)).hexdigest()
        for edge in refined["edges"].values():
            signed = list(map(float.fromhex, edge["signed_distance_hex"]))
            assert max(signed[:2]) < 0 < min(signed[2:])
