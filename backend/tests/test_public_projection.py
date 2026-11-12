from app.api.public_projection import alias_id, original_id, project, project_card


def test_second_writing_run_keeps_private_checkpoint_details_out_of_public_projection():
    detector = 'txt.review.agreement'
    revision = 'a6efd4bf6ecd852f332b2569ae0f98cae13b7985'
    model = 'hf.rasbt.writing_review'
    run = {'id': 'run-1', 'detector_id': detector, 'status': 'failed', 'duration_ms': 15,
           'model_revisions': {model: revision}, 'raw_output_artefact_id': 'private-run-output',
           'reason': f'Could not load {model} revision {revision}'}
    shown, = project({'items': [run]})['items']
    assert shown['detector_id'] == alias_id(detector)
    assert shown['status'] == 'failed' and shown['duration_ms'] == 15
    assert shown['reason'] == 'The second writing response was unavailable.'
    assert shown['model_revisions'] == {} and shown['raw_output_artefact_id'] is None
    assert run['model_revisions'] == {model: revision}


def test_legacy_video_voice_scores_use_the_same_public_projection_as_recording_scores():
    from copy import deepcopy

    name = 'wav2vec2-large-xlsr-deepfake-audio-classification'
    observed = f'fused p(synthetic)=0.840 over 4 window(s) of ≤ 4 s; per model: {name}=0.84'
    generated = {'id': 'audio_synthetic_voice', 'category': 'model', 'observed': observed,
                 'details': {'per_model_chunks': {name: [0.81, 0.87]}, 'strongest': name}}
    source = {'id': 'source', 'category': 'provenance', 'observed': observed}
    payload = {'result': {'signals': [generated, source]},
               'metadata': {'observed': observed}, 'notes': [{'text': observed}]}
    before = deepcopy(payload)
    shown = project(payload)
    signal = shown['result']['signals'][0]
    expected = project({'category': 'model', 'observed': observed})['observed']
    assert signal['observed'] == expected
    assert 'per check: component.' in expected and expected.endswith('=0.84')
    assert name not in str(signal)
    alias = next(iter(signal['details']['per_model_chunks']))
    assert f'{alias}=0.84' in expected and signal['details']['strongest'] == alias
    assert signal['details']['per_model_chunks'][alias] == [0.81, 0.87]
    assert shown['result']['signals'][1] == source
    assert shown['metadata'] == payload['metadata'] and shown['notes'] == payload['notes']
    assert payload == before


def test_generated_validation_disclosures_hide_names_without_rewriting_source_text():
    payload = {'not_validated': [{'id': 'image.localisers', 'label': 'Mesorch region maps',
                                 'text': 'Mesorch and FUSED localisers are uncalibrated.'}],
               'text': 'The source document discusses Mesorch and FUSED.',
               'metadata': {'not_validated': [{'text': 'Mesorch was written in this source.'}]}}
    output = project(payload)
    assert 'Mesorch' not in str(output['not_validated']) and 'FUSED' not in str(output['not_validated'])
    assert output['text'] == payload['text'] and output['metadata'] == payload['metadata']


def test_segment_source_references_use_the_same_public_ids_as_the_signals():
    private_id = 'aud.pella_v2:global'
    segment = {'start_s': 2, 'end_s': 6, 'sources': [private_id]}
    signal = {'id': private_id, 'detector_id': 'aud.pella_v2', 'segments': [segment]}
    payload = {'items': [signal], 'metadata': {'segments': [segment]}}
    shown, = project(payload)['items']
    assert shown['id'] == alias_id(private_id) and shown['id'] != private_id
    assert shown['segments'] == [{'start_s': 2, 'end_s': 6, 'sources': [shown['id']]}]
    assert project(payload)['metadata'] == payload['metadata']
    assert segment['sources'] == [private_id]


def test_new_local_inspection_has_explicit_validation_disclosure():
    from app.engine.fusion.disclosures import for_result

    items = for_result({'kind': 'image', 'signals': [{'detector_id': 'loc.fused', 'status': 'ok'}],
                        'decision': {'fusion_mode': 'calibrated'}})
    assert any(item['id'] == 'image.localisers' for item in items)


def test_localiser_prose_hides_architectures_and_generated_observations_only():
    from copy import deepcopy

    from app.detectors.image.adapters.localisers import MesorchDetector, RadarDetector
    from app.detectors.image.adapters.regional import RegionalDetector

    signals = [{'id': f'{detector.spec.id}:global', 'detector_id': detector.spec.id,
                'name': detector.spec.name, 'reference': detector.REFERENCE,
                'observed': f'{detector.METHOD} integrity score 0.750; 2 regions at map ≥ 0.5; '
                            'source note says “Mesorch”.'}
               for detector in (MesorchDetector, RadarDetector, RegionalDetector)]
    provenance = {'id': 'img.meta.exif:global', 'detector_id': 'img.meta.exif', 'category': 'provenance',
                  'observed': 'Software=Mesorch, Depth Anything V2 Base', 'reference': 'Source says RADAR'}
    payload = {'signals': [*signals, provenance], 'metadata': {'Software': 'Mesorch'},
               'annotations': [{'body': 'RADAR was written in the source.'}]}
    before = deepcopy(payload)
    projected = project(payload)
    for signal in projected['signals'][:-1]:
        assert 'uncalibrated local editing response' in signal['reference']
        assert 'not proof of AI editing' in signal['reference']
        assert all(name not in signal['reference'] for name in
                   ('Mesorch', 'RADAR', 'ConvNeXt', 'SegFormer', 'DINOv2', 'Depth Anything', 'MiT-B3', 'ViT-B/14'))
        assert signal['observed'].startswith('a detection check integrity score 0.750; 2 regions at map ≥ 0.5;')
        assert 'source note says “Mesorch”.' in signal['observed']
    assert projected['signals'][-1] == provenance
    assert projected['metadata'] == payload['metadata'] and projected['annotations'] == payload['annotations']
    # The v1 JSON export downgrades signals to their scoped id without detector_id.
    legacy = deepcopy(payload)
    for signal in legacy['signals']:
        signal.pop('detector_id')
    public_legacy = project(legacy)
    assert public_legacy['signals'][0]['reference'] == projected['signals'][0]['reference']
    assert public_legacy['signals'][0]['observed'] == projected['signals'][0]['observed']
    assert payload == before


def test_references_join_without_changing_source_evidence():
    payload = {
        'filename': 'DDA sample.jpg',
        'input': {'filename': 'DDA sample.jpg'},
        'metadata': {'Software': 'DDA'},
        'signals': [{'id': 'img.dda:global', 'detector_id': 'img.dda',
                     'name': 'DDA (DINOv2 ViT-L/14 + LoRA, native 336 px crops)',
                     'observed': 'DDA was in the image metadata', 'reference': 'DINOv2 ViT-L/14 with LoRA'},
                    {'id': 'img.meta.exif:global', 'detector_id': 'img.meta.exif', 'category': 'provenance',
                     'name': 'Capture-device metadata', 'observed': 'Model=DDA'}],
        'findings': [{'family': 'F3_whole_image_generation', 'statement': 'DDA returned a score of 0.4',
                      'evidence_ids': ['img.dda:global'], 'sentences': [{'text': 'DDA returned 0.4'}]},
                     {'family': 'F2_self_declared_metadata', 'statement': 'The file mentions DDA',
                      'evidence_ids': ['detector:img.meta.generator']}],
        'decision': {'contributions': [{'detector_id': 'img.dda', 'label': 'DDA'}]},
    }
    result = project(payload)
    signal = result['signals'][0]
    assert signal['id'] == alias_id('img.dda:global')
    assert result['findings'][0]['evidence_ids'] == [signal['id']]
    assert result['decision']['contributions'][0]['detector_id'] == signal['detector_id']
    assert 'DDA' not in result['findings'][0]['statement']
    assert 'DDA' not in result['decision']['contributions'][0]['label']
    assert result['findings'][1]['statement'] == 'The file mentions DDA'
    assert result['metadata'] == payload['metadata'] and result['input'] == payload['input']
    assert signal['observed'] == payload['signals'][0]['observed']
    assert original_id(signal['id']) == 'img.dda:global'
    assert payload['signals'][0]['id'] == 'img.dda:global'


def test_model_card_does_not_publish_model_source_or_names():
    card = {'id': 'img.dda', 'name': 'DDA', 'adapter': 'DDADetector', 'models': ['dda.dinov2_vitl14_lora'],
            'model_entries': [{'id': 'dda.dinov2_vitl14_lora', 'source': {'repo': 'DDA/repo'},
                               'licence': {'class': 'commercial', 'notice': 'DDA'}}],
            'source_url': 'https://example.com/DDA', 'training_data': 'DDA', 'citation': 'DDA', 'client': {}}
    output = project_card(card)
    assert output['id'] == alias_id('img.dda') and output['name'] == 'Analysis check'
    assert 'DDA' not in str(output) and 'dinov2' not in str(output).lower()
    assert card['model_entries'][0]['source']['repo'] == 'DDA/repo'


def test_customer_model_routes_paginate_and_resolve_alias():
    import json
    from types import SimpleNamespace

    from app.api.v2_models import get_model, list_models

    request = SimpleNamespace(state=SimpleNamespace(principal=SimpleNamespace(satisfies=lambda _: False)))
    all_cards = json.loads(list_models(request, cursor=None, limit=200, modality=None, enabled=None).body)['items']
    ids = []
    cursor = None
    for _ in range(len(all_cards) + 1):
        body = json.loads(list_models(request, cursor=cursor, limit=1, modality=None, enabled=None).body)
        ids.extend(card['id'] for card in body['items'])
        cursor = body['next_cursor']
        if cursor is None:
            break
    assert ids == [card['id'] for card in all_cards]
    learned = next(card for card in all_cards if card['id'].startswith('check.'))
    detail = json.loads(get_model(learned['id'], request).body)
    assert detail['id'] == learned['id']
    assert 'source' not in detail['model_entries'][0]


def test_v1_export_audits_exact_delivered_bytes(monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    from app.api import routes

    class FakeAnalysis:
        status = 'done'
        result = {'signals': [{'id': 'img.dda:global', 'name': 'DDA',
                               'observed': 'user text says DDA', 'category': 'model'}],
                  'metadata': {'Software': 'DDA'}, 'artifacts': ['gradcam_dima806.jpg']}

        def to_summary(self):
            return {'id': 'fake', 'filename': 'DDA photo.jpg'}

    recorded = {}
    monkeypatch.setattr(routes, '_get', lambda _: FakeAnalysis())
    monkeypatch.setattr(routes, '_current_revision', lambda _: None)
    monkeypatch.setattr(routes, '_verified', lambda *_args: None)
    monkeypatch.setattr(routes.audit, 'record', lambda *_args, **kwargs: recorded.update(kwargs))
    request = SimpleNamespace(state=SimpleNamespace(principal=SimpleNamespace(id='local')))
    response = routes.export_json('fake', request)
    data = json.loads(response.body)
    assert recorded['after_hash'] == hashlib.sha256(response.body).hexdigest()
    assert data['result']['signals'][0]['id'] == alias_id('img.dda:global')
    assert data['result']['signals'][0]['name'] != 'DDA'
    assert data['result']['signals'][0]['observed'] == 'user text says DDA'
    assert data['result']['metadata']['Software'] == 'DDA'
    assert data['filename'] == 'DDA photo.jpg'
    assert 'dima806' not in data['result']['artifacts'][0]
    assert response.headers['x-chipmunk-view'] == 'customer-projection'


def test_event_frames_use_same_aliases_and_keep_status():
    import json

    from app.api.v2 import _frame

    payload = {'seq': 1, 'type': 'partial_signal', 'payload': {'detector_id': 'img.dda',
               'signal_id': 'img.dda:global', 'name': 'DDA (DINOv2 ViT-L/14 + LoRA, native 336 px crops)',
               'status': 'ok'}}
    frame = _frame('partial_signal', payload, 1)
    data = json.loads(next(line[6:] for line in frame.splitlines() if line.startswith('data: ')))['payload']
    assert data['detector_id'] == alias_id('img.dda')
    assert data['signal_id'] == alias_id('img.dda:global')
    assert data['status'] == 'ok' and 'DDA' not in data['name']


def test_public_cursor_round_trips_composite_detector_references():
    from app.api.public_projection import public_cursor
    from app.api.v2 import decode_cursor, encode_cursor

    cursor = encode_cursor(('img.dda:global',))
    projected = public_cursor(cursor)
    assert projected != cursor
    assert decode_cursor(projected, 1) == ['img.dda:global']


def test_audit_customer_projection_preserves_user_notes_and_original_hashes():
    import json

    from app.api.v2_audit import AuditEventOut, AuditPage, _customer_audit_page

    base = dict(seq=1, ts='2026-01-01T00:00:00Z', actor_id='system', case_id=None, media_id=None, analysis_id=None,
                report_id=None, before_hash=None, after_hash=None, prev_hash='0' * 64, event_hash='1' * 64)
    machine = AuditEventOut(**base, action='analysis_finished', object_ids=['img.dda'],
                            details={'status': 'done', 'model': 'DDA'})
    user = AuditEventOut(**{**base, 'seq': 2}, action='note_added', object_ids=[],
                         details={'note': 'DDA', 'target_type': 'signal', 'target_id': 'img.dda:global'})
    review = AuditEventOut(**{**base, 'seq': 3}, action='fuser_review_withdrawn', object_ids=[],
                           details={'kind': 'image', 'reason': 'DDA', 'artefact': 'model-DDA.json'})
    response = _customer_audit_page(AuditPage(items=[machine, user, review]))
    rows = json.loads(response.body)['items']
    assert rows[0]['object_ids'] == [alias_id('img.dda')] and rows[0]['details'] == {'status': 'done'}
    assert rows[0]['event_hash'] == '1' * 64 and rows[1]['details']['note'] == 'DDA'
    assert rows[1]['details']['target_id'] == alias_id('img.dda:global')
    assert rows[2]['details'] == {'kind': 'image'} and review.details['reason'] == 'DDA'
    assert response.headers['x-audit-hash-meaning'] == 'original-events-not-projected-rows'


def test_historical_v1_detector_signals_and_face_scores_are_projected():
    source = {'signals': [
        {'id': 'clf_dfdc_effnet_b7', 'name': 'DFDC-winner EfficientNet-B7 face-swap classifier', 'category': 'face',
         'details': {'model': 'selimsef/dfdc_deepfake_challenge ensemble', 'per_face': [0.5]}},
        {'id': 'classifier_document', 'name': 'DeBERTa-v3-large AI-text classifier (document chunks)', 'category': 'model',
         'details': {'model': 'desklib/ai-text-detector-v1.01'}},
        {'id': 'univfd_clip', 'name': 'UnivFD — CLIP ViT-L/14 universal fake detector', 'category': 'model',
         'reasoning': 'CLIP feature-space probe places this image on the synthetic side'},
        {'id': 'clf_deepfake_vs_real_image_detection', 'name': 'Classifier — dima806/deepfake_vs_real_image_detection',
         'category': 'face', 'details': {'architecture': 'ViTForImageClassification'}},
        {'id': 'frames_general', 'name': 'Whole-frame AI-image detectors (UnivFD + classifiers), averaged', 'category': 'model'},
    ], 'faces': [{'scores': {'dfdc_effnet_b7': 0.5, 'deepfake_vs_real_image_detection': 0.9}}]}
    result = project(source)
    for signal in result['signals']:
        assert signal['name'] == 'Analysis check'
        assert 'model' not in signal.get('details', {}) and 'architecture' not in signal.get('details', {})
    assert result['signals'][0]['id'] == alias_id('clf_dfdc_effnet_b7')
    assert 'CLIP' not in result['signals'][2]['reasoning']
    assert all(key.startswith('check.') for key in result['faces'][0]['scores'])
    assert source['signals'][0]['details']['model'].startswith('selimsef/')


def test_signal_annotation_projection_preserves_user_body_and_round_trips():
    import json
    from datetime import UTC, datetime

    from app.api.v2_annotations import AnnotationOut, _json

    annotation = AnnotationOut(id='a', analysis_id='b', case_id=None, author_id='local', target_type='signal',
                               target_id='img.dda:global', shape='none', geometry=None, examiner_status=None,
                               body='My note says DDA', visibility='internal', created_at=datetime.now(UTC),
                               edited_at=None, deleted_at=None, revision=1, history=[])
    projected = json.loads(_json(annotation).body)
    assert projected['target_id'] == alias_id('img.dda:global')
    assert original_id(projected['target_id']) == annotation.target_id
    assert projected['body'] == 'My note says DDA'
    assert annotation.target_id == 'img.dda:global'
def test_generated_architecture_and_fusion_identifiers_are_private_but_source_quotes_are_exact():
    from copy import deepcopy

    from app.api.public_projection import alias_id, project

    payload = {'reference': 'ViT-S/16 trained on photographs; VAE residual with fuser-video-2.1.1.',
               'description': 'CLIP and open_clip used for an analysis check; the clip lasts 4 seconds.',
               'reason': 'ViT-L/14 returned 0.7; source says “ViT-S/16 and fuser-video-2.1.1”.',
               'fuser_id': 'fuser-video-2.1.1', 'subject': 'fuser-video-2.1.1',
               'manifest': {'calibration_id': 'fuser-video-2.1.1', 'library_versions': {'open_clip': '3.3.0', 'numpy': '2.0'}},
               'text': 'Source document mentions ViT-S/16 and VAE.',
               'metadata': {'Software': 'fuser-video-2.1.1'},
               'annotations': [{'body': 'The source names CLIP.'}]}
    before = deepcopy(payload)
    public = project(payload)
    assert all(name not in public['reference'] for name in ('ViT-S/16', 'VAE', 'fuser-video-2.1.1'))
    assert public['reference'].endswith('.')
    assert 'CLIP' not in public['description'] and 'open_clip' not in public['description']
    assert 'the clip lasts 4 seconds' in public['description']
    assert '“ViT-S/16 and fuser-video-2.1.1”' in public['reason'] and 'ViT-L/14' not in public['reason']
    assert public['fuser_id'] == public['subject'] == alias_id('fuser-video-2.1.1')
    assert public['manifest']['calibration_id'] == public['fuser_id']
    assert 'open_clip' not in public['manifest']['library_versions']
    assert sorted(public['manifest']['library_versions'].values()) == ['2.0', '3.3.0']
    assert public['fuser_id'].startswith('component.')
    for key in ('text', 'metadata', 'annotations'):
        assert public[key] == payload[key]
    assert payload == before

