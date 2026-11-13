"""Video v3 joins source-bound voice/AV evidence under one immutable parent."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
import wave
import zipfile
from dataclasses import asdict, replace
from datetime import UTC, datetime
from fractions import Fraction
from types import SimpleNamespace

import cv2
import numpy as np
import pymupdf
import pytest
from sqlalchemy.orm import Session
from test_video_av_evidence import av_fixture as av_fixture
from test_video_av_evidence import av_source as av_source

from app.api import v2_evidence_packets
from app.api.public_projection import project
from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.audio.decode import SAMPLE_RATE, decode_audio
from app.detectors.video import lipsync
from app.detectors.video.adapters import face_appearance as face_adapter
from app.detectors.video.adapters.face_appearance import DETECTOR, record
from app.detectors.video.adapters.lipsync_xcorr import LandmarkXcorrDetector
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.face_appearance import METHOD, cpp_alignment
from app.detectors.video.frame_store import FrameStore
from app.engine.boundary import Deadline
from app.engine.detector import DetectorResult, UnitScore
from app.engine.stages import video
from app.engine.synthesis.engine import SynthesisInput, synthesise
from app.evidence import (
    face_boundary_binding,
    face_return_binding,
    face_return_coverage,
    face_short_carrier_binding,
    face_short_window_binding,
    face_spatial_cut_binding,
    service,
    video_audio,
    video_packets,
)
from app.evidence.audio_binding import validate_audio_channel
from app.evidence.image_packets import PacketIntegrityError
from app.reports import engine, multimedia_evidence, package, signing, snapshot, verify, verify_video_source
from app.reports.specimens import specimen
from app.storage.artefacts import register_artefact
from app.storage.db import AnalysisStage, Artefact, ReportRow
from app.storage.objects import ObjectIntegrityError


def test_one_source_bound_timing_window_discloses_missing_change_coverage_in_json_and_pdf(av_fixture, monkeypatch):
    f = av_fixture
    window = f.sync.windows[0].window
    used = {f.grid.frames[tick] for tick in window.ticks}
    track = {'track_id': 'track:0', 'exhibit_id': 'source-face'}
    inputs = replace(f.inputs, tracks=(track,), frames={'track:0': [f.by_frame[i] for i in sorted(used)]})
    exhibits = SimpleNamespace(series=lambda *args, **kwargs: 'series', segments=lambda *args, **kwargs: 'segments')
    ctx = SimpleNamespace(outputs=SimpleNamespace(av=inputs), exhibits=exhibits, deadline=None)
    monkeypatch.setattr(lipsync, 'vad', lambda pcm: f.speech)
    monkeypatch.setattr(lipsync, 'speaker_score', lambda *args: .9)
    detector = LandmarkXcorrDetector()
    result = detector.run(ctx)
    assert result.status == 'ok' and result.aggregate is None
    proof = result.details['av_input']
    assert proof['status'] == 'ready' and proof['audio_origin_pts'] == 86400
    (track_proof,) = proof['tracks']
    (window_proof,) = track_proof['windows']
    assert window_proof['start_tick'] == 25 and window_proof['end_tick'] == 50
    measured = '6.400–7.400 s on the source clock'
    assert measured in result.observed and 'Timing change unassessed' in result.observed

    entry = {**asdict(result), 'detector_id': detector.spec.id, 'version': detector.spec.version,
             'name': detector.spec.name, 'family': 'av-sync', 'modality': 'video',
             'hypotheses': ['H2_localised_manipulation']}
    stage = SimpleNamespace(state={}, outputs={'tracks': {'tracks': [track]}, 'detect': {'detectors': [entry]}})
    (signal,) = video.track_signals(stage, 1)
    (public,) = project([signal])
    assert measured in public['observed'] and 'Timing change unassessed' in public['observed']
    assert public['model_revisions'] == {}

    report = specimen('video', 'standard')
    payload = copy.deepcopy(report.result)
    findings = synthesise(SynthesisInput('video', payload['decision'], [signal]))
    (finding,) = [row for row in findings if row['family'] == 'V2_lip_sync']
    assert finding['direction'] == 'neutral' and signal['id'] in finding['evidence_ids']
    payload.update(signals=[signal], findings=findings, exhibits=[])
    rendered = engine.render_pdf(report.model_copy(update={'result': payload}))
    text = ' '.join(engine.pdf_text(rendered.pdf).split())
    assert measured in text and 'Timing change unassessed' in text


@pytest.fixture(params=['video-evidence-3', 'video-evidence-4', 'video-evidence-5', 'video-evidence-6'])
def nested_case(av_fixture, monkeypatch, request):
    f = av_fixture
    _prepare_parent(f)
    # Pin this fixture's writer independently of future versions; older sets
    # remain readable and are never retroactively enriched.
    monkeypatch.setattr(video_packets, 'VERSION', request.param)
    monkeypatch.setattr(video_packets, 'CREATOR', 'video-evidence-packets/' + request.param.rsplit('-', 1)[1])
    if request.param == 'video-evidence-4':
        from app.evidence import audio_background
        current_acoustics = audio_background.build_content
        def historical_acoustics(*args, **kwargs):
            kwargs.setdefault('version', 'mixed-acoustics-1')
            return current_acoustics(*args, **kwargs)
        monkeypatch.setattr(audio_background, 'build_content', historical_acoustics)
    if request.param == 'video-evidence-3':
        from app.evidence import audio_packets

        current_builder = audio_packets.build_content
        def historical_content(*args, **kwargs):
            content = current_builder(*args, **kwargs)
            content.pop('acoustic_context')
            return content
        monkeypatch.setattr(audio_packets, 'build_content', historical_content)
    _freeze_parent(f)
    return f


def _prepare_parent(f, *, has_audio=True):
    anchors = list(f.anchors.values())
    sampling = f.result['manifest']['sampling']
    sampling.update(frame_anchor_version=video_packets.ANCHOR_VERSION,
                    frame_anchor_sha256=canonical_hash(anchors), frame_table_sha256=canonical_hash(f.table.to_json()),
                    timeline_origin_s=5.)
    for stage, content in [('probe', {'frame_table': f.table.to_json(), 'info': {'has_audio': has_audio}}),
                           ('render', {'frame_anchor_version': video_packets.ANCHOR_VERSION, 'frame_anchors': anchors})]:
        art = register_artefact(f.db, f.store, canonical_json(content), media_type='application/json', role='stage_output',
                               created_by='av-fixture', analysis_id=f.analysis.id)
        f.db.add(AnalysisStage(analysis_id=f.analysis.id, stage=stage, status='done', output_artefact_id=art.id))
    f.result['exhibits'] = []
    f.analysis.result_hash = canonical_hash(f.result)
    f.store.put_bytes(canonical_json(f.result))
    f.db.flush()


def _freeze_parent(f):
    f.document = video_packets.build_packet_set(f.db, f.analysis, f.store, f.result)
    frozen = specimen('video', 'standard')
    f.frozen = frozen.model_copy(update={
        'result': copy.deepcopy(f.result), 'evidence': copy.deepcopy(f.document),
        'analysis': frozen.analysis.model_copy(update={'id': f.analysis.id, 'revision': f.analysis.revision}),
        'media': frozen.media.model_copy(update={'sha256': f.analysis.sha256, 'size_bytes': f.analysis.size_bytes,
                                                'filename': f.analysis.filename, 'content_type': f.analysis.content_type}),
        'report': frozen.report.model_copy(update={'template_set': engine.TEMPLATE_SET})})


def test_brief_face_coverage_without_a_signal_reaches_signed_report(av_fixture, tmp_path, monkeypatch):
    f = av_fixture
    incomplete = {'status': 'unassessed', 'reason': 'multiple_faces', 'scheduled_frames': 24,
                  'measured_face_frames': 10}
    initial = {'status': 'no_distinct_range', 'reason': None, 'scheduled_frames': 16, 'measured_face_frames': 15}
    coverage = face_return_coverage.from_worker({**initial, 'dense_fallback': incomplete}, len(f.table))
    f.result['manifest']['sampling'][face_return_coverage.KEY] = coverage
    assert not any(s.get('detector_id') == DETECTOR for s in f.result['signals'])
    _prepare_parent(f, has_audio=False)
    _freeze_parent(f)
    f.result = _report_scaffold(f).result
    f.analysis.result_hash = f.store.put_bytes(canonical_json(f.result)).sha256
    _freeze_parent(f)

    assert f.document['coverage'][face_return_coverage.KEY] == coverage
    assert not any(p.get('face_return') for p in f.document['packets'])
    assert service.load_packet_set(f.db, f.analysis, f.store)['coverage'][face_return_coverage.KEY] == coverage
    assert project(f.document)['coverage'][face_return_coverage.KEY] == coverage
    altered = copy.deepcopy(f.document)
    del altered['coverage'][face_return_coverage.KEY]
    with pytest.raises(ValueError, match='differs from the signed public result'):
        face_return_coverage.validate_commitment(altered, canonical_hash(coverage))
    with pytest.raises(ObjectIntegrityError, match='differs from the frozen analysis'):
        multimedia_evidence._verify_video(altered, f.store, {}, result=f.result)

    monkeypatch.setattr(multimedia_evidence, 'default_store', lambda: f.store)
    pdf = engine.render_pdf(f.frozen).pdf
    (tmp_path / 'brief-face-coverage.pdf').write_bytes(pdf)
    text = ' '.join(engine.pdf_text(pdf).split())
    assert 'Brief face-change check' in text
    assert 'Initial scan. No distinct brief face-change range was found in this scan.' in text
    assert 'Denser scan. More than one face was detected in a sampled picture.' in text
    assert '10 of 24 scheduled pictures.' in text and DETECTOR not in text
    assert '/coverage/brief_face_review' in text and 'measured_face_frames' not in text
    _check_public_zip(f, tmp_path, monkeypatch, pdf)


@pytest.mark.parametrize('status,reason,scheduled,measured', [
    ('not_applicable', 'native_scope', 0, 0),
    ('unassessed', 'sparse_range_outside_profile', 32, 32),
    ('no_distinct_range', None, 32, 31),
])
def test_brief_face_coverage_validates_exact_dispositions(status, reason, scheduled, measured):
    worker = {'status': status, 'reason': reason, 'scheduled_frames': scheduled, 'measured_face_frames': measured}
    coverage = face_return_coverage.from_worker(worker, 120)
    document = {'source': {'frame_count': 120}, 'packets': [], 'coverage': {face_return_coverage.KEY: coverage}}
    face_return_coverage.validate_commitment(document, canonical_hash(coverage))
    coverage['observations'] = ['No suspicious frames anywhere in the clip.']
    with pytest.raises(ValueError, match='wording differs'):
        face_return_coverage.validate(document)


def test_legacy_and_bounded_face_coverage_bind_to_separate_ranges():
    legacy = {'status': 'not_applicable', 'reason': 'native_scope', 'scheduled_frames': 0, 'measured_face_frames': 0}
    bounded = {'status': 'localized', 'reason': None, 'pixel_frames': 120, 'scheduled_frames': 35, 'measured_face_frames': 31}
    result = face_return_coverage.from_worker(legacy, 120, bounded=bounded)
    record = {'method': 'face-short-return/9', 'scene_frame_inputs': list(range(120)),
              'scheduled_frame_indices': list(range(35)), 'frame_inputs': list(range(31))}
    document = {'source': {'frame_count': 120}, 'packets': [{'face_return': record}],
                'coverage': {face_return_coverage.KEY: result}}
    face_return_coverage.validate_commitment(document, canonical_hash(result))
    assert result['observations'][0].startswith('Initial scan. This clip is outside')
    assert 'Motion inputs: 120 of 120 pictures. Comparable face inputs: 31 of 35' in result['observations'][1]

    removed = {**document, 'coverage': {}}
    with pytest.raises(ValueError, match='missing its measured coverage'):
        face_return_coverage.validate(removed)
    with pytest.raises(ValueError, match='displayed ranges'):
        face_return_coverage.validate({**document, 'packets': [{'face_return': {**record, 'method': 'face-short-return/1'}}]})
    with pytest.raises(ValueError, match='differs from its measured source inputs'):
        face_return_coverage.validate({**document, 'packets': [{'face_return': {**record, 'frame_inputs': list(range(30))}}]})

    completed = {**bounded, 'status': 'no_distinct_range'}
    no_range = face_return_coverage.from_worker(legacy, 120, bounded=completed)
    face_return_coverage.validate({'source': {'frame_count': 120}, 'packets': [], 'coverage': {face_return_coverage.KEY: no_range}})


def test_parent_builder_capture_assets_and_manifest_are_one_set(nested_case, monkeypatch):
    f = nested_case
    assert f.document['version'] in video_packets.NESTED_VERSIONS
    assert ('acoustic_context' in f.document['audio']) == (f.document['version'] != 'video-evidence-3')
    assert f.document['audio']['status'] == f.document['av']['status'] == 'ready'
    assert service.load_packet_set(f.db, f.analysis, f.store) == f.document
    assert multimedia_evidence.capture(f.db, f.analysis, result=f.result, store=f.store) == f.document
    assets = service.asset_refs(f.document)
    assert {a['media_type'] for a in assets.values()} == {'image/png', 'audio/wav', 'video/mp4'}
    manifest = multimedia_evidence.manifest(f.frozen)
    assert {a['sha256'] for a in manifest['artifacts']} == set(assets)
    assert all(f'/{f.document["id"]}/assets/' in a['url'] for a in assets.values())
    assert len(manifest['artifacts']) > len(assets)  # intentional repeated patch/clip bytes
    assert snapshot.dump(snapshot.load(snapshot.dump(f.frozen))) == snapshot.dump(f.frozen)
    monkeypatch.setattr(service, 'load_packet_set', lambda *a, **k: pytest.fail('render read mutable analysis'))
    multimedia_evidence.view(f.frozen, store=f.store)


@pytest.mark.parametrize("refine", [False, True])
def test_appearance_evidence_records_the_same_measurements_in_json_and_reports(av_fixture, refine):
    f = av_fixture
    centers = [36, 44, 52, 60, 68, 76, 84]
    indices = {center + offset for center in centers for offset in (-1, 0, 1)}
    during = {center + offset for center in centers[2:-2] for offset in (-1, 0, 1)}
    # 12 Hz grid on the native 25 fps clock. Frame 37 wins the 37.5 tie.
    dense = [37, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83]
    pixels = f.inputs.store.get_many(indices | (set(dense) if refine else set()))
    anchors = f.inputs.store.native_anchors()
    by_index = {row['frame_index']: row for row in anchors}
    inputs = {}
    for index in pixels:
        frame = f.by_frame[index]
        landmarks = copy.deepcopy(frame['landmarks'])
        if index in during:
            landmarks[:2] = [[42, 22], [54, 22]]
        alignment = cpp_alignment(landmarks)
        aligned = cv2.warpAffine(pixels[index], alignment, (112, 112), flags=cv2.INTER_LINEAR)
        inputs[index] = {
            'frame_index': index, 'pts': f.table.pts[index], 'bbox': frame['bbox'], 'landmarks': landmarks,
            'decoded_rgb_sha256': by_index[index]['decoded_rgb_sha256'],
            'aligned_rgb_sha256': hashlib.sha256(aligned.tobytes()).hexdigest(),
            'alignment_matrix': alignment.tolist(),
            'alignment_matrix_hex': [[float(value).hex() for value in row] for row in alignment],
            'alignment_side': 112, 'aligned_size': 112,
        }
    sequence = record({
        'centers': centers, 'track_id': 'track:0', 'start_s': f.table.seconds(52), 'end_s': f.table.seconds(68),
        'response': .2, 'middle_to_before': .235, 'middle_to_after': .3, 'flank_change': .1,
    }, f.table, inputs)
    details = {'face_sequence_binding_version': METHOD, 'face_sequence': sequence}
    if refine:
        details['face_boundary'] = {
            'method': 'face-appearance-boundary/1', 'time_base': f.table.time_base, 'track_id': 'track:0',
            'start_pts': f.table.pts[50], 'end_pts': f.table.pts[69],
            'start_stream_s': f.table.seconds(50), 'end_stream_s': f.table.seconds(69),
            'scheduled_frame_indices': dense, 'frame_inputs': [inputs[i] for i in dense],
            'anchors': {'before': 48, 'during': 60, 'after': 69},
            'edges': {edge: {'frame_indices': indices, 'signed_distance_hex': [float(value).hex() for value in (-.2, -.3, .3, .2)]}
                      for edge, indices in (('entry', [46, 48, 50, 52]), ('exit', [65, 67, 69, 71]))},
        }
    f.result['signals'].append({
        'id': DETECTOR + ':global', 'detector_id': DETECTOR, 'status': 'ok', 'display_id': 'S03',
        'details': details, 'effective_weight': 0.,
    })
    f.result['manifest']['sampling'].update(
        frame_anchor_version=video_packets.ANCHOR_VERSION, frame_anchor_sha256=canonical_hash(anchors),
        frame_table_sha256=canonical_hash(f.table.to_json()), timeline_origin_s=5.,
    )
    for stage, data in (
        ('probe', {'frame_table': f.table.to_json(), 'info': {'has_audio': True}}),
        ('render', {'frame_anchor_version': video_packets.ANCHOR_VERSION, 'frame_anchors': anchors}),
    ):
        asset = register_artefact(f.db, f.store, canonical_json(data), media_type='application/json',
                                  role='stage_output', created_by='appearance-fixture', analysis_id=f.analysis.id)
        f.db.add(AnalysisStage(analysis_id=f.analysis.id, stage=stage, status='done', output_artefact_id=asset.id))
    f.analysis.result_hash = canonical_hash(f.result)
    f.store.put_bytes(canonical_json(f.result))
    f.db.flush()
    document = video_packets.build_packet_set(f.db, f.analysis, f.store, f.result)
    stored = service.load_packet_set(f.db, f.analysis, f.store)
    assert stored == json.loads(canonical_json(document))
    assert stored['version'] == video_packets.VERSION
    assert stored['picture_hold']['status'] == 'unavailable'
    [face] = stored['packets']
    assert face['finding_ids'] == [] and all(frame['score'] is None for frame in face['frames'])
    tracking = [row for row in face['measurements'] if row['label'].startswith('Median eye spacing')]
    assert [row['value'] for row in tracking] == [50., 16.67, 50.]
    assert all(row['unit'] == '% of face-box width' for row in tracking)
    assert face['face_lighting']['status'] == 'measured'
    assert [row['frame_index'] for row in face['face_lighting']['frames']] == [44, 60, 76]

    frozen = specimen('video', 'standard')
    frozen = frozen.model_copy(update={
        'result': json.loads(canonical_json(f.result)), 'evidence': stored,
        'analysis': frozen.analysis.model_copy(update={'id': f.analysis.id, 'revision': f.analysis.revision}),
        'media': frozen.media.model_copy(update={'sha256': f.analysis.sha256}),
    })
    report = multimedia_evidence.view(frozen, store=f.store)
    assert report['packets'][0]['measurements'] == face['measurements']
    html = engine._env.get_template('sections/multimedia_evidence.html.j2').render(v={'evidence': report}, s=frozen)
    assert 'Median eye spacing during the range' in html and '16.67 % of face-box width' in html
    assert 'Scene outside the face brightness change from before' in html

    if refine:
        assert face['face_sequence'] == json.loads(canonical_json(sequence))
        assert face['face_boundary'] == json.loads(canonical_json(details['face_boundary']))
        assert face['affects_overall_confidence'] is False
        assert [(frame['interval_role'], frame['anchor']['frame_index']) for frame in face['face_boundary_frames']] == [
            ('before', 48), ('during', 60), ('after', 69)]
        assert 'Refined boundary estimate' in html and '7.000–7.760 s' in html
        assert 'Before refined range' in html and '7.080-7.720 s' in html
        replay = FrameStore(f.inputs.store.path, f.table)
        ctx = SimpleNamespace(outputs={'detect': {'detectors': [{'details': details}]}}, check=lambda *_: None)
        restored = video._native_render_anchors(ctx, replay)
        assert {row['frame_index']: row for row in restored} == {index: by_index[index] for index in pixels}

        def unverified_decoder(*args, **kwargs):
            for frame in decode_frames(*args, **kwargs):
                yield replace(frame, pts_verified=False) if frame.index == dense[0] else frame

        unverified = FrameStore(f.inputs.store.path, f.table, decoder=unverified_decoder)
        with pytest.raises(video.DecodeError, match='lacks a verified decoded timestamp'):
            video._native_render_anchors(ctx, unverified)
        bad_input = copy.deepcopy(details)
        # Use a dense-only index, so the stored coarse binding remains unchanged.
        dense_only = next(row for row in bad_input['face_boundary']['frame_inputs'] if row['frame_index'] not in indices)
        dense_only['decoded_rgb_sha256'] = 'f' * 64
        ctx.outputs['detect']['detectors'][0]['details'] = bad_input
        with pytest.raises(video.DecodeError, match='differs from the re-decoded source frames'):
            video._native_render_anchors(ctx, replay)
        omitted = copy.deepcopy(stored)
        altered_boundary = omitted['packets'][0]['face_boundary']
        # Dropping one outer frame preserves the observed edge support and image references.
        altered_boundary['scheduled_frame_indices'].pop(0)
        altered_boundary['frame_inputs'].pop(0)
        face_boundary_binding.validate_packet(omitted['packets'][0], omitted['source'], omitted['version'])
        bundle = {'schema': package.PUBLIC_SCHEMA, 'analysis': {'kind': 'video'},
                  'source_commitment': {'sha256': f.analysis.sha256}}
        with zipfile.ZipFile(io.BytesIO(), 'w') as archive:
            archive.writestr('evidence.json', canonical_json(omitted))
            with pytest.raises(ValueError, match='sample schedule differs from the source clock'):
                verify_video_source.check(archive, bundle, f.inputs.store.path)
        broken = copy.deepcopy(stored)
        broken['packets'][0]['face_boundary']['edges']['entry']['signed_distance_hex'][1] = .1.hex()
        broken['sha256'] = video_packets.packet_hash(broken)
        with pytest.raises(PacketIntegrityError, match='lack measured change'):
            video_packets.validate_packet_set(broken, analysis_id=f.analysis.id, revision=f.analysis.revision,
                                             original_result_hash=f.analysis.result_hash, source_sha256=f.analysis.sha256)
    else:
        assert 'face_boundary' not in face and 'Refined boundary estimate' not in html

    from app.evidence import face_lighting_context

    altered = copy.deepcopy(stored)
    changed = altered['packets'][0]
    item = changed['face_lighting']['frames'][1]
    item['scene'] = 99 if item['scene'] != 99 else 1
    changed['measurements'] = [row for row in changed['measurements'] if row['label'] not in face_lighting_context.LABELS]
    changed['measurements'].extend(face_lighting_context.measurements(changed['face_lighting']))
    altered['sha256'] = video_packets.packet_hash(altered)
    with pytest.raises(ObjectIntegrityError, match='brightness differs from the displayed source pixels'):
        multimedia_evidence.view(frozen.model_copy(update={'evidence': altered}), store=f.store)

    tracking[1]['value'] = 90.
    stored['sha256'] = video_packets.packet_hash(stored)
    with pytest.raises(PacketIntegrityError, match='Face tracking context differs'):
        video_packets.validate_packet_set(stored, analysis_id=f.analysis.id, revision=f.analysis.revision,
                                         original_result_hash=f.analysis.result_hash, source_sha256=f.analysis.sha256)


@pytest.mark.parametrize('redacted', [False, True])
def test_nested_report_section_has_input_proof_and_withholds_source(nested_case, redacted):
    f = nested_case
    frozen = f.frozen.model_copy(update={'report': f.frozen.report.model_copy(update={'redacted': redacted})})
    view = multimedia_evidence.view(frozen, store=f.store)
    assert [p['kind'] for p in view['packets']] == ['audio', 'av', 'av', 'av']
    html = engine._env.get_template('sections/multimedia_evidence.html.j2').render(v={'evidence': view}, s=frozen)
    for required in ('Speech inspection window', 'All 31 lag candidates', 'Source audio PTS', 'actual 32 by 24 inputs'):
        assert required in html
    assert ('<img src="data:image/png;base64,' in html) is not redacted
    assert 'aud.gustking_xlsr' not in html
    assert len(view['packets'][1]['measured_inputs']) == 25
    for row in view['packets']:
        for image in row['images']:
            assert image['label'].startswith(f"{row['id']} - {row['locator']} - ")
        if row['kind'] == 'av':
            for tick in row['measured_inputs']:
                assert tick['image']['label'].startswith(f"{row['id']} - tick {tick['tick']}; frame {tick['frame_index']}")


def _report_scaffold(f):
    """Labelled report specimen around real frozen source inputs, with compatible published check refs."""
    original = specimen('video','standard')
    result = copy.deepcopy(original.result)
    result['input']['sha256'] = f.analysis.sha256
    result['manifest']['sampling'] = copy.deepcopy(f.result['manifest']['sampling'])
    template = result['signals'][0]
    result['signals'] = [{**copy.deepcopy(template), **row, 'name':'Controlled engineering check'} for row in f.result['signals']]
    result['findings'] = []
    result['exhibits'] = []
    evidence = copy.deepcopy(f.document)
    evidence['original_result_hash'] = canonical_hash(result)
    evidence['sha256'] = service.packet_hash(evidence)
    return f.frozen.model_copy(update={'result':result,'evidence':evidence})


def test_overlapping_face_windows_require_matching_scope_and_shared_signal():
    binding = face_short_window_binding
    windows = [{"start_frame_index": start, "end_frame_index_exclusive": start + binding.WINDOW,
                "status": "no_distinct_range", "reason": None, "pixel_frames": binding.WINDOW,
                "scheduled_frames": 12, "measured_face_frames": 8} for start in binding.WINDOW_STARTS]
    source = {"frame_count": binding.COUNT}
    document = {"version": "video-evidence-23", "source": source,
                "coverage": {binding.KEY: binding.coverage(windows)}, "packets": []}
    face_return_binding.validate_set(document)
    with pytest.raises(ValueError, match="evidence version disagrees"):
        face_return_binding.validate_set({**document, "version": "video-evidence-11"})
    malformed = copy.deepcopy(document)
    malformed["coverage"][binding.KEY]["windows"][1]["end_frame_index_exclusive"] += 1
    with pytest.raises(ValueError, match="invalid measured coverage"):
        face_return_binding.validate_set(malformed)

    for index in range(2):
        start = binding.WINDOW_STARTS[index]
        windows[index].update(status="localized", measured_face_frames=12)
        document["packets"].append({"semantics": binding.METHOD, "signal_refs": [{"id": f"S{index}"}],
            "face_return": {"method": binding.METHOD,"window_start_frame_index": start, "source_first_pts": 0,
                            "start_pts": (start+15)*1000,"end_pts": (start+21)*1000,
                            "scene_frame_inputs": [None]*120,"scheduled_frame_indices": [None]*12,"frame_inputs": [None]*12}})
    document["coverage"][binding.KEY] = binding.coverage(windows)
    with pytest.raises(ValueError, match="same confidence-neutral review check"):
        face_return_binding.validate_set(document)


@pytest.mark.parametrize(("carrier", "encoding", "time_base", "end_pts"), [
    ("webm", ["vp9", "yuv420p", "[0][0][0][0]"], "1/1000", 7967),
    ("mp4", ["h264", "yuv420p", "avc1"], "1/30000", 239000),
])
def test_carrier_face_windows_bind_actual_ticks_and_scope_without_a_finding(carrier, encoding, time_base, end_pts):
    proof, windows = face_short_carrier_binding, face_short_window_binding
    native = [57 + proof.tick(i, Fraction(time_base), carrier) for i in range(windows.COUNT)]
    assert native[-1] - native[0] == end_pts
    if carrier == "webm":
        assert native[1:4] == [90, 124, 157]
    source = {"frame_count": windows.COUNT, "width": 1280, "height": 720, "time_base": time_base}
    clock = {"source_encoding": encoding, "source_carrier": carrier, "source_first_pts": native[0],
             "source_time_base": time_base}
    disposition = [{"start_frame_index": start, "end_frame_index_exclusive": start + windows.WINDOW,
                    "status": "no_distinct_range", "reason": None, "pixel_frames": windows.WINDOW,
                    "scheduled_frames": 12, "measured_face_frames": 8} for start in windows.WINDOW_STARTS]
    coverage = windows.coverage(disposition) | clock
    document = {"version": "video-evidence-24", "source": source, "coverage": {windows.KEY: coverage}, "packets": []}
    face_return_binding.validate_set(document)
    proof.validate_metadata(coverage, source, table_pts=native)
    native[63] += 1
    with pytest.raises(ValueError, match="measured native carrier or cadence"):
        proof.validate_metadata(coverage, source, table_pts=native)
    with pytest.raises(ValueError, match="evidence version disagrees"):
        face_return_binding.validate_set({**document, "version": "video-evidence-23"})
    with pytest.raises(ValueError, match="measured native carrier or cadence"):
        proof.validate_metadata({**coverage, "source_carrier": "mkv"}, source)


def test_overlapping_face_window_timeout_preserves_incumbent_and_reports_unassessed():
    now = [0.]
    deadline = Deadline(10., clock=lambda: now[0])

    def timeout_during_read(indices, *, timeout_s):
        assert list(indices) == list(range(240)) and 0 < timeout_s <= 9
        now[0] = 9.1
        yield 0, None

    prior = {"start_pts": 20000, "end_pts": 60000}
    result = DetectorResult(status="ok", aggregate=UnitScore("global", .7, "score"), observed="Existing review.",
                            details={"face_sequence": prior, "review_only": True})
    inputs = SimpleNamespace(table=object(), store=SimpleNamespace(iter_many=timeout_during_read))
    measured = face_adapter._with_short_windows(result, inputs, None, None, deadline)
    assert measured is result and measured.aggregate.raw == .7 and measured.details["face_sequence"] == prior
    assert measured.observed == "Existing review." and "face_short_windows" not in measured.details
    coverage = face_short_window_binding.coverage(measured.details["face_short_windows_check"])
    assert all(row["status"] == "unassessed" and row["reason"] == "time_budget" for row in coverage["windows"])
    assert all("was unassessed" in line for line in coverage["observations"])


def test_picture_cut_context_binds_both_native_edges_and_representative_frames():
    binding = face_spatial_cut_binding
    a, b, start, end = 56, 72, 63, 74
    pts = [1000 * i for i in range(120)]
    schedule, face_frames = binding.schedule(a, b)
    source = {'width': 1280, 'height': 720, 'time_base': '1/30000', 'frame_count': 120}
    coarse = {'method': binding.APPEARANCE_METHOD, 'time_base': source['time_base'],
              'start_pts': pts[a], 'end_pts': pts[b]}
    record = {'method': binding.METHOD, 'time_base': source['time_base'], 'source_first_pts': 0,
              'source_encoding': ['hevc', 'yuv420p', 'hvc1'], 'source_carrier': 'mp4',
              'start_frame_index': start, 'end_frame_index': end, 'start_pts': pts[start], 'end_pts': pts[end],
              'start_stream_s': start / 30, 'end_stream_s': end / 30,
              'coarse_start_pts': pts[a], 'coarse_end_pts': pts[b], 'score_hex': .35.hex(),
              'tiles': 10, 'shared_tiles': 5, 'entry_tiles': 5, 'exit_tiles': 5, 'sectors': 2,
              'face_inputs': [{'frame_index': i, 'bbox': {'x': 300, 'y': 180, 'w': 80, 'h': 90}}
                              for i in face_frames],
              'frame_inputs': [{'frame_index': i, 'pts': pts[i], 'decoded_rgb_sha256': '0' * 64}
                               for i in schedule],
              'anchors': {'before': 62, 'during': 68, 'after': 74}}
    binding.validate_record(record, coarse, source, table_pts=pts)
    for changes, message in [
        ({'coarse_end_pts': pts[b] + 1}, 'original appearance range'),
        ({'frame_inputs': record['frame_inputs'][1:]}, 'omits a native frame'),
        ({'anchors': {**record['anchors'], 'after': 75}}, 'wrong representative frames'),
    ]:
        with pytest.raises(ValueError, match=message):
            binding.validate_record({**record, **changes}, coarse, source, table_pts=pts)
    with pytest.raises(ValueError, match='evidence version disagrees'):
        binding.validate_set({'version': binding.SPATIAL_CUT_VERSION,
                              'packets': [{'face_spatial_cut': record}, {'face_return': {'verified': True}}]})


@pytest.mark.parametrize('width,height', [(1920, 1080), (1280, 720), (640, 360)])
def test_short_face_change_keeps_exact_source_frames_in_pdf_json_and_offline_verifier(
    av_fixture, av_source, tmp_path, monkeypatch, width, height,
):
    f = av_fixture
    path = tmp_path / 'native-face-clock.mkv'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-filter_threads', '1', '-copyts',
                    '-i', str(av_source[0]), '-vf', f'scale={width}:{height}:flags=neighbor', '-an', '-c:v', 'ffv1',
                    '-threads', '1', str(path)], check=True, timeout=60)
    obj = f.store.put_file(path)
    f.analysis.sha256, f.analysis.size_bytes = obj.sha256, obj.size_bytes
    f.table = probe_frames(path)
    f.result.update(input={'sha256': obj.sha256}, signals=[],
                    manifest={'sampling': {'video_audio_evidence': {'source_audio_present': False}}})
    frame_store = FrameStore(f.store.path_for(obj.sha256), f.table)
    clock = [(pts - f.table.pts[0]) * f.table.tb for pts in f.table.pts]
    scheduled = [min(range(len(clock)), key=lambda i: (abs(clock[i] - Fraction(tick, 8)), i))
                 for tick in range(32)]
    entry, stop = 13, 16
    start, end = scheduled[entry], scheduled[stop]
    before, during, after = scheduled[entry-4:entry], scheduled[entry:stop], scheduled[stop:stop+4]
    indices = before + during + after
    pixels = frame_store.get_many(set(indices))
    f.anchors = {item['frame_index']: item for item in frame_store.native_anchors()}
    faces = []
    for index in indices:
        landmarks = [[x * width / 96, y * height / 64] for x, y in f.by_frame[index]['landmarks']]
        matrix = cpp_alignment(landmarks)
        aligned = cv2.warpAffine(pixels[index], matrix, (112, 112), flags=cv2.INTER_LINEAR)
        faces.append({**{key: f.anchors[index][key] for key in ('frame_index', 'pts', 'decoded_rgb_sha256')},
                      'bbox': {'x': width // 8, 'y': height // 8, 'w': 3 * width // 4, 'h': 878 * height // 1080},
                      'landmarks': landmarks, 'aligned_size': 112,
                      'alignment_side': 112, 'alignment_matrix': matrix.tolist(),
                      'alignment_matrix_hex': [[float(value).hex() for value in row] for row in matrix],
                      'aligned_rgb_sha256': hashlib.sha256(aligned.tobytes()).hexdigest()})
    scene_indices = (before[-1], start, during[-1], end)
    anchors = {'before': before[-1], 'during': during[1], 'after': after[0]}
    review = {'method': face_return_binding.METHOD, 'time_base': f.table.time_base, 'start_frame_index': start,
              'end_frame_index': end, 'start_pts': f.table.pts[start], 'end_pts': f.table.pts[end],
              'start_stream_s': f.table.seconds(start), 'end_stream_s': f.table.seconds(end),
              'response': .15, 'response_hex': .15.hex(), 'scene_correlations': [.88, .91],
              'scene_correlations_hex': [.88.hex(), .91.hex()], 'anchors': anchors,
              'scheduled_frame_indices': scheduled, 'support_indices': {'before': before, 'during': during, 'after': after},
              'frame_inputs': faces,
              'scene_frame_inputs': [{key: f.anchors[i][key] for key in ('frame_index', 'pts', 'decoded_rgb_sha256')}
                                     for i in scene_indices]}
    f.result['signals'].append({'id': DETECTOR + ':global', 'detector_id': DETECTOR, 'status': 'ok', 'display_id': 'S03',
                               'details': {'face_return': review}, 'effective_weight': 0.})
    _prepare_parent(f, has_audio=False)
    _freeze_parent(f)
    f.result = _report_scaffold(f).result
    f.analysis.result_hash = f.store.put_bytes(canonical_json(f.result)).sha256
    _freeze_parent(f)

    [packet] = [row for row in f.document['packets'] if row.get('face_return')]
    assert f.document['version'] == 'video-evidence-12'
    with pytest.raises(ValueError, match='evidence version disagrees'):
        face_return_binding.validate_set({**f.document, 'version': 'video-evidence-11'})
    assert packet['face_return'] == json.loads(canonical_json(review))
    assert packet['affects_overall_confidence'] is False
    assert [(row['interval_role'], row['anchor']['frame_index']) for row in packet['frames']] == list(anchors.items())
    assert service.load_packet_set(f.db, f.analysis, f.store) == f.document
    for key, delta in [('end_pts', 1), ('start_playback_s', .2)]:
        altered = copy.deepcopy(packet)
        altered['window'][key] += delta
        with pytest.raises(ValueError, match='displayed time range'):
            face_return_binding.validate_packet(altered, f.document['source'])
    altered = copy.deepcopy(packet)
    altered['frames'][0]['interval_role'] = 'after'
    with pytest.raises(ValueError, match='native sample'):
        face_return_binding.validate_packet(altered, f.document['source'])
    for source_change, table_change in [({'width': 960}, {}), ({}, {'rotation': 180}), ({}, {'truncated': True}),
                                        ({}, {'interpolated': [0]})]:
        with pytest.raises(ValueError, match='complete native timestamps'):
            face_return_binding.validate_packet(packet, {**f.document['source'], **source_change}, **table_change)
    if height in {720, 360}:
        altered = copy.deepcopy(packet)
        altered['face_return']['frame_inputs'][0]['pts'] -= 1
        with pytest.raises(ValueError, match='exact source clock'):
            face_return_binding.validate_packet(altered, f.document['source'])

    monkeypatch.setattr(multimedia_evidence, 'default_store', lambda: f.store)
    pdf = engine.render_pdf(f.frozen).pdf
    report = ' '.join(engine.pdf_text(pdf).split())
    assert 'Brief face-change range to review' in report
    assert '6.640-7.000 s on the source clock' in report and 'with an exclusive end' in report
    assert 'before, during, and after' in report and DETECTOR not in report
    _check_public_zip(f, tmp_path, monkeypatch, pdf, source_video=path)


@pytest.mark.parametrize('profile', ['standard','forensic','summary','newsroom'])
def test_full_report_profiles_include_nested_evidence_without_internal_sampling(nested_case,monkeypatch,profile):
    f=nested_case
    frozen=_report_scaffold(f)
    frozen=frozen.model_copy(update={'report':frozen.report.model_copy(update={'profile':profile})})
    monkeypatch.setattr(multimedia_evidence,'default_store',lambda:f.store)
    html=engine.render_html(frozen,embed_fonts=False)
    assert 'All 31 lag candidates' in html
    assert 'video_audio_evidence' not in html and 'voice_checks' not in html and 'av_checks' not in html
    assert 'aud.gustking_xlsr' not in html
    assert ('<img src="data:image/png;base64,' in html) is (profile!='newsroom')


@pytest.mark.parametrize('part,change', [('audio','score'), ('audio','availability'), ('av','profile'), ('av','raw_run')])
def test_nested_capture_rejects_semantic_tampering_even_after_rehash(nested_case, part, change):
    f = nested_case
    altered = copy.deepcopy(f.document)
    if change == 'score':
        altered['audio']['packets'][0]['observations'][0]['value'] = .9
        altered['audio']['coverage']['windows'][0]['score'] = .9
    elif change == 'availability':
        from app.evidence.video_audio import unavailable
        altered['audio'] = unavailable(f.analysis.sha256, 'Invented missing input')
    elif change == 'profile':
        altered['av']['packets'][0]['measurement']['audio_curve'][0]['rms'] += .01
    else:
        altered['av']['packets'][0]['source_record']['raw_output_sha256'] = 'b'*64
    altered['sha256'] = service.packet_hash(altered)
    with pytest.raises((ObjectIntegrityError, ValueError)):
        multimedia_evidence.verify_assets(altered, f.store, result=f.result)


def test_old_video_sets_keep_their_bytes_and_remain_readable(nested_case):
    f = nested_case
    for version in ('video-evidence-1', 'video-evidence-2'):
        old = copy.deepcopy(f.document)
        old.pop('audio')
        old.pop('av')
        old['version'] = version
        old['sha256'] = service.packet_hash(old)
        before = canonical_json(old)
        video_packets.validate_packet_set(old, analysis_id=f.analysis.id, revision=f.analysis.revision,
            original_result_hash=f.analysis.result_hash, source_sha256=f.analysis.sha256)
        assert canonical_json(old) == before


def test_nested_public_zip_offline_verifier_without_runtime(nested_case, tmp_path, monkeypatch):
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((40,60), 'Controlled AV public export integrity fixture')
        pdf_bytes=pdf.tobytes()
    _check_public_zip(nested_case, tmp_path, monkeypatch, pdf_bytes)


def _check_public_zip(f, tmp_path, monkeypatch, pdf_bytes, *, source_audio=None, source_video=None):
    monkeypatch.setattr(settings, 'data_dir', tmp_path/'data')
    key = signing.instance_key(tmp_path/'key.pem')
    pdf_art=register_artefact(f.db,f.store,pdf_bytes,media_type='application/pdf',role='report',created_by='fixture',analysis_id=f.analysis.id)
    snap_art=register_artefact(f.db,f.store,snapshot.dump(f.frozen),media_type='application/json',role='stage_output',created_by='fixture',analysis_id=f.analysis.id)
    report=ReportRow(id=f.frozen.report.id,display_id=f.frozen.report.display_id,revision=1,
        analysis_id=f.analysis.id,analysis_revision=f.analysis.revision,profile='standard',redacted=False,
        review_state='machine',issuable=False,template_set=engine.TEMPLATE_SET,engine='fixture',status='done',
        snapshot_hash=snapshot.snapshot_hash(f.frozen),snapshot_artefact_id=snap_art.id,artefact_id=pdf_art.id,
        sha256=pdf_art.sha256,signature=key.sign(pdf_bytes),signing_key_id=key.key_id,
        requested_by='fixture',requested_at=datetime.now(UTC),page_count=1)
    f.db.add(report)
    f.db.flush()
    row=package.prepare(f.db,f.analysis,report,kind='draft',include_original=False,created_by='fixture',schema_id=package.PUBLIC_SCHEMA)
    built=package.build(f.db,row,store=f.store,key=key,exported_at=datetime.now(UTC))
    checked=verify.verify_package(built.path,public_key_pem=key.public_pem())
    assert checked.ok, checked.as_dict()
    with zipfile.ZipFile(built.path) as archive:
        assert json.loads(archive.read('evidence.json')) == f.document
        assert archive.read('report.pdf') == pdf_bytes
        bundle=json.loads(archive.read('bundle.json'))
        assert {a['sha256'] for a in bundle['artifacts']} == set(service.asset_refs(f.document))
        assert all(archive.read(a['file']) == f.store.read_bytes(a['sha256']) for a in bundle['artifacts'])
    trusted=tmp_path/'trusted.pem'
    trusted.write_bytes(key.public_pem())
    if source_audio is not None or source_video is not None:
        command = [sys.executable, '-m', 'app.cli', 'verify', str(built.path), '--public-key', str(trusted), '--json']
        source_args = []
        if source_audio is not None:
            source_args.extend(('--source-audio', str(source_audio)))
        if source_video is not None:
            source_args.extend(('--source-video', str(source_video)))
        checked = subprocess.run(command + source_args, capture_output=True, text=True, timeout=30)
        assert checked.returncode == 0, checked.stdout + checked.stderr
        checks = {check['name']: check for check in json.loads(checked.stdout)['checks']}
    if source_audio is not None:
        audio_check = checks['original audio samples']
        assert audio_check['ok']
        assert 'source timeline matched' in audio_check['detail'] and 'Verified 0 published' not in audio_check['detail']
        part = f.document.get('audio', f.document)
        stereo = part.get('stereo_context')
        if stereo and stereo['schema'] == 'chipmunk.stereo-channel-context/2' and stereo['stereo_pcm_sha256'] is not None:
            assert f"Recomputed stereo channel and mono-downmix evidence for {len(stereo['windows'])} windows" in audio_check['detail']
    if source_video is not None:
        video_check = checks['original video frames']
        assert video_check['ok'] and 'source frame clock matched' in video_check['detail']
        assert f"mouth patches ({25 * len(f.document['av']['packets'])})" in video_check['detail']
    if source_audio is not None and source_video is not None:
        wrong_source = tmp_path / 'unrelated-original.bin'
        wrong_source.write_bytes(b'unrelated original')
        checked = subprocess.run(command + ['--source-audio', str(wrong_source), '--source-video', str(source_video)],
                                 capture_output=True, text=True, timeout=30)
        assert checked.returncode == 1, checked.stdout + checked.stderr
        checks = {check['name']: check for check in json.loads(checked.stdout)['checks']}
        assert not checks['original audio samples']['ok'] and checks['original video frames']['ok']
    offline=subprocess.run([sys.executable,'-c', '''
import importlib.abc,json,socket,sys
from pathlib import Path
class RuntimeUnavailable(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in ('app.evidence','app.evidence.mp4','app.evidence.audio_binding','app.evidence.stereo_binding',
                        'app.evidence.stereo_listening_binding', 'app.evidence.stereo_repetition_binding',
                        'app.evidence.audio_gain_repetition_binding',
                        'app.evidence.audio_background_binding','app.evidence.audio_bandwidth_binding','app.evidence.audio_repetition_binding','app.evidence.audio_exact_repetition',
                        'app.evidence.audio_similarity_binding',
                        'app.evidence.av_binding','app.evidence.face_interval_binding','app.evidence.face_boundary_binding','app.evidence.face_boundary_clock','app.evidence.face_alignment_context',
                        'app.evidence.face_paired_edge_binding','app.evidence.face_short_proposal_binding','app.evidence.face_short_window_binding','app.evidence.face_short_carrier_binding','app.evidence.face_spatial_cut_binding',
                        'app.evidence.face_lighting_context','app.evidence.face_return_binding','app.evidence.face_return_coverage',
                        'app.evidence.video_packet_versions','app.evidence.picture_hold_binding', 'app.evidence.picture_sequence',
                        'app.evidence.scene_return_binding',
                        'app.evidence.mouth_local_hold_binding','app.evidence.source_geometry',
                        'app.evidence.image_eligibility','app.evidence.image_copy_similarity_binding','app.evidence.image_interpolation_binding',
                        'app.evidence.image_copy_affine_binding','app.evidence.image_embedded_pair_binding',
                        'app.evidence.document_scope','app.evidence.email_alternatives'):
            return None
        if fullname.startswith(('app.storage','app.core.config','app.detectors','app.evidence','app.api')):
            raise RuntimeError('Offline verifier imported runtime: '+fullname)
sys.meta_path.insert(0,RuntimeUnavailable())
def no_network(*a,**k): raise RuntimeError('network forbidden')
socket.socket=no_network
from app.reports.verify import verify_package
r=verify_package(Path(sys.argv[1]),public_key_pem=Path(sys.argv[2]).read_bytes())
print(json.dumps(r.as_dict()));sys.exit(0 if r.ok else 1)
''',str(built.path),str(trusted)],capture_output=True,text=True,timeout=20)
    assert offline.returncode==0, offline.stdout+offline.stderr


def test_video_clock_break_survives_pdf_json_and_signed_audio_replay(av_fixture, av_source, tmp_path, monkeypatch):
    f = av_fixture
    original, _, decoded, _ = av_source
    path = tmp_path / 'clock-gap.mkv'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-copyts', '-i', str(original),
                    '-map', '0:v:0', '-map', '0:a:0', '-c:v', 'copy', '-c:a', 'pcm_s16le',
                    '-af', r'asetnsamples=n=2048:p=0,asetpts=PTS+if(gte(N\,24576)\,10240\,0)',
                    '-avoid_negative_ts', 'disabled', str(path)], check=True, timeout=30)
    obj = f.store.put_file(path)
    f.analysis.sha256, f.analysis.size_bytes = obj.sha256, obj.size_bytes
    f.result['input']['sha256'] = obj.sha256
    f.table = probe_frames(path)

    def sessions():
        return Session(f.db.get_bind())

    ctx = SimpleNamespace(outputs={'ingest': {'sha256': obj.sha256}, 'probe': {
        'frame_table': f.table.to_json(), 'plan': {'legacy': {'frames': [0, 8, 16]}}, 'info': {'has_audio': True}}},
        state={'media': SimpleNamespace(path=path)}, store=f.store, check=lambda *a: None,
        sessions=sessions, analysis_id=f.analysis.id)
    stage = video.decode(ctx)['audio']
    assert stage['source_clock_break_count'] == 1 and ctx.state['audio'].clock_breaks == (24576,)
    assert stage['evidence_input']['pcm_sha256'] == decoded.mapping['pcm_sha256']
    record = f.result['manifest']['sampling']['video_audio_evidence']
    record.update(input=stage['evidence_input'], av_checks=[], voice_clock_policy='source-clock-split/1')
    check = record['voice_checks'][0]
    raw = json.loads(f.store.read_bytes(check['raw_output_sha256']))
    raw['unit_scores'][0]['quality'].update(end_sample=24576, end_s=24576 / SAMPLE_RATE)
    check['raw_output_sha256'] = f.store.put_bytes(canonical_json(raw)).sha256
    _prepare_parent(f)
    _freeze_parent(f)
    f.result = _report_scaffold(f).result
    f.analysis.result_hash = f.store.put_bytes(canonical_json(f.result)).sha256
    _freeze_parent(f)
    clock = f.document['audio']['coverage']['source_clock_context']
    assert clock['total_discontinuities'] == 1 and clock['scored_windows_crossing'] == 0
    assert clock['recording_voice_estimate_withheld'] is True
    assert clock['breaks'][0]['jump_samples'] == 10240 and clock['breaks'][0]['boundary_sample'] == 24576
    tampered = copy.deepcopy(f.document['audio'])
    tampered['coverage']['source_clock_context']['breaks'][0]['jump_samples'] += 1
    with pytest.raises(ValueError, match='Audio source clock observations differ'):
        validate_audio_channel(tampered, obj.sha256)

    monkeypatch.setattr(multimedia_evidence, 'default_store', lambda: f.store)
    pdf = engine.render_pdf(f.frozen).pdf
    report = ' '.join(engine.pdf_text(pdf).split())
    assert 'Audio source clock breaks' in report and 'Source timestamp gap of 640.000 ms' in report
    assert 'the report gives no recording-wide voice estimate' in report
    assert '/audio/coverage/source_clock_context' in report
    _check_public_zip(f, tmp_path, monkeypatch, pdf, source_audio=path)


def test_real_video_soundtrack_cue_preserves_production_voice_and_av(av_source, tmp_path, monkeypatch):
    source, _, original, face = av_source
    stereo_path = tmp_path / 'stereo.wav'
    left = np.rint(original.pcm * 32768).astype('<i2')
    right = left.copy()
    right[2 * SAMPLE_RATE:3 * SAMPLE_RATE] *= -1
    with wave.open(str(stereo_path), 'wb') as stereo_wav:
        stereo_wav.setparams((2, 2, SAMPLE_RATE, 0, 'NONE', ''))
        stereo_wav.writeframes(np.column_stack((left, right)).astype('<i2').tobytes())
    path = tmp_path / 'stereo-offset.mkv'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-copyts', '-i', str(source), '-itsoffset', '5.4',
                    '-i', str(stereo_path), '-itsoffset', '5.4', '-f', 'lavfi',
                    '-i', 'sine=frequency=711:sample_rate=16000:duration=3',
                    '-map', '0:v:0', '-map', '1:a:0', '-map', '2:a:0', '-c', 'copy',
                    '-avoid_negative_ts', 'disabled', str(path)], check=True, timeout=30)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    table = probe_frames(path)
    reference = decode_audio(path, source_sha256=sha, max_seconds=120)
    assert table.seconds(0) == 5 and reference.mapping['timeline'][0]['source_pts'] == 86400
    assert reference.mapping['audio_stream_indices'] == [1, 2]

    fixture = av_fixture.__wrapped__(tmp_path, (path, table, reference, face))
    try:
        f = next(fixture)
        voice_ref = f.result['manifest']['sampling']['video_audio_evidence']['voice_checks'][0]
        voice = json.loads(f.store.read_bytes(voice_ref['raw_output_sha256']))
        voice['unit_scores'][0]['quality'].update(end_sample=32000, end_s=2)
        voice_ref['raw_output_sha256'] = f.store.put_bytes(canonical_json(voice)).sha256
        _prepare_parent(f)
        _freeze_parent(f)
        ordinary = f.document
        assert 'stereo_context' not in ordinary['audio']

        def sessions():
            return Session(f.db.get_bind())

        ctx = SimpleNamespace(outputs={'ingest': {'sha256': sha}, 'probe': {
            'frame_table': table.to_json(), 'plan': {'legacy': {'frames': [0, 8, 16]}}, 'info': {'has_audio': True}}},
            state={'media': SimpleNamespace(path=path)}, store=f.store, check=lambda *a: None,
            sessions=sessions, analysis_id=f.analysis.id)
        ctx.outputs['decode'] = video.decode(ctx)
        stage = ctx.outputs['decode']['audio']
        binding = stage['evidence_input']
        mapping = stage['source_mapping']
        assert mapping == reference.mapping
        assert binding['mapping_sha256'] == canonical_hash(reference.mapping)
        listening_pcm_sha = binding['stereo_listening_pcm_sha256']
        assert len(f.store.read_bytes(listening_pcm_sha)) == 4 * SAMPLE_RATE
        assert binding['pcm_sha256'] == reference.mapping['pcm_sha256']
        assert f.store.read_bytes(binding['wav_sha256']) == reference.wav
        np.testing.assert_array_equal(ctx.state['audio'].pcm, reference.pcm)

        stereo_sha = stage['stereo_context_sha256']
        assert 'stereo_context_sha256' not in video_audio.published_record(ctx)
        ctx.outputs['render'] = {'audio_stereo_context_sha256': stereo_sha}
        committed = video_audio.published_record(ctx)
        assert committed['stereo_context_sha256'] == stereo_sha and committed['input'] == binding
        f.result['manifest']['sampling']['video_audio_evidence'].update(
            stereo_context_sha256=stereo_sha, input=binding)
        f.result = _report_scaffold(f).result
        f.analysis.result_hash = f.store.put_bytes(canonical_json(f.result)).sha256
        _freeze_parent(f)
        f.db.commit()
        stereo = f.document['audio']['stereo_context']
        assert f.document['audio']['source']['audio_stream_indices'] == [1, 2]
        assert canonical_hash(stereo) == stereo_sha
        assert json.loads(f.store.read_bytes(stereo_sha)) == stereo
        assert stereo['source_stream_index'] == 1 and stereo['affects_overall_confidence'] is False
        assert [w['downmix']['strong_attenuation'] for w in stereo['windows']] == [False, False, True]
        flagged = stereo['windows'][2]
        assert flagged['downmix']['mono_rms_dbfs'] == stereo['silence_floor_dbfs']
        assert flagged['source_timestamp_ranges'] == [{
            'start_sample': 32000, 'end_sample': 48000, 'source_start_pts': 118400, 'source_end_pts': 134400,
            'time_base': '1/16000',
        }]
        coverage = f.document['audio']['coverage']
        assert {k: v for k, v in coverage.items() if k != 'stereo_attenuation_overlaps'} == ordinary['audio']['coverage']
        (overlap,) = coverage['stereo_attenuation_overlaps']
        assert overlap == {'start_sample': 32000, 'end_sample': 48000, 'unscored_interval_index': 0,
                           'stereo_window_index': 2, 'source_timestamp_ranges': flagged['source_timestamp_ranges']}
        (unscored,) = coverage['unscored_intervals']
        assert (unscored['start_sample'], unscored['end_sample']) == (32000, len(reference.pcm))
        assert unscored['source_timestamp_ranges'][0]['source_start_pts'] == 118400
        assert f.document['av']['packets']
        assert [p['measurement'] for p in f.document['av']['packets']] == [p['measurement'] for p in ordinary['av']['packets']]
        assert [p['anchor'] for p in f.document['av']['packets']] == [p['anchor'] for p in ordinary['av']['packets']]
        assert 'gustking' not in json.dumps(stereo)
        listening = f.document['audio']['stereo_listening']
        assert [row['stereo_window_index'] for row in listening['windows']] == [2]
        assert listening['affects_overall_confidence'] is False

        monkeypatch.setattr(v2_evidence_packets, 'get_session', sessions)
        monkeypatch.setattr(v2_evidence_packets, 'default_store', lambda: f.store)
        response = v2_evidence_packets.image_evidence(f.analysis.id)
        assert json.loads(response.body) == f.document
        monkeypatch.setattr(multimedia_evidence, 'default_store', lambda: f.store)
        rendered = engine.render_pdf(f.frozen)
        text = engine.pdf_text(rendered.pdf)
        assert 'Audio track coverage' in text and 'The second track was unassessed.' in ' '.join(text.split())
        assert '/audio/source/audio_stream_indices' in text
        assert 'Stereo soundtrack comparison' in text and 'Mono is digitally silent' in text
        assert '[118400, 134400)' in text and '/audio/stereo_context' in text
        assert '/audio/stereo_listening/windows/0' in text
        assert 'Video playback 2.400–3.400 s' in ' '.join(text.split())
        assert 'Unscored voice intervals' in text and '/audio/coverage/unscored_intervals' in text
        assert 'Unscored voice during a mono level drop' in text and '/audio/coverage/stereo_attenuation_overlaps' in text
        assert 'Overlap 2.000–3.000 s of analyzed audio' in ' '.join(text.split())
        assert f'Video playback 2.400–{.4 + len(reference.pcm) / SAMPLE_RATE:.3f} s' in ' '.join(text.split())
        assert 'All 31 lag candidates' in text
        _check_public_zip(f, tmp_path, monkeypatch, rendered.pdf, source_audio=path, source_video=path)

        forged_tracks = copy.deepcopy(f.document)
        forged_tracks['audio']['source']['audio_stream_indices'] = [1, 3]
        forged_tracks['sha256'] = service.packet_hash(forged_tracks)
        with pytest.raises(ObjectIntegrityError, match='Audio track coverage differs'):
            multimedia_evidence.verify_assets(forged_tracks, f.store, result=f.result)
        frozen_row = f.db.get(Artefact, f.document['id'])
        original_hash = frozen_row.sha256
        try:
            frozen_row.sha256 = f.store.put_bytes(canonical_json(forged_tracks)).sha256
            f.db.flush()
            with pytest.raises(PacketIntegrityError, match='Audio track coverage differs'):
                video_packets.load_packet_set(f.db, f.analysis, f.store, set_id=f.document['id'])
        finally:
            frozen_row.sha256 = original_hash
            f.db.flush()

        forged = copy.deepcopy(f.document['audio'])
        forged['stereo_context']['windows'][2]['source_timestamp_ranges'][0]['source_start_pts'] += 1
        forged['stereo_context_sha256'] = canonical_hash(forged['stereo_context'])
        with pytest.raises(ValueError, match='source timestamps'):
            validate_audio_channel(forged, f.analysis.sha256)
        forged = copy.deepcopy(f.document['audio'])
        forged['coverage']['unscored_intervals'][0]['start_sample'] += 1
        with pytest.raises(ValueError, match='Unscored voice intervals differ'):
            validate_audio_channel(forged, f.analysis.sha256)
        forged = copy.deepcopy(f.document['audio'])
        forged['coverage']['stereo_attenuation_overlaps'][0]['source_timestamp_ranges'][0]['source_start_pts'] += 1
        with pytest.raises(ValueError, match='Unscored stereo overlaps differ'):
            validate_audio_channel(forged, f.analysis.sha256)
        omitted = copy.deepcopy(f.document)
        del omitted['audio']['stereo_context'], omitted['audio']['stereo_context_sha256']
        del omitted['audio']['stereo_listening']
        del omitted['audio']['coverage']['stereo_attenuation_overlaps']
        omitted['sha256'] = service.packet_hash(omitted)
        with pytest.raises(ObjectIntegrityError, match='differs from its published commitment'):
            multimedia_evidence.verify_assets(omitted, f.store, result=f.result)
    finally:
        fixture.close()
