from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.reports.public_view import public_manifest, public_view
from app.reports.snapshot import snapshot_hash
from app.reports.specimens import specimen


def test_skipped_check_diagnostics_are_private_even_without_statistical_signals():
    original = specimen('video', 'standard')
    result = deepcopy(original.result)
    result['signals'] = []
    result['manifest']['detectors'] = [{'id': 'vid.dda', 'version': '1.0.0'}]
    snapshot = original.model_copy(update={'result': result})
    reason = 'Too few checks could run: DDA (DINOv2 ViT-L/14 + LoRA), frames below 336 px.'
    quote = 'The source says “DDA”.'
    view = {'summary': {'insufficient': [reason], 'text': reason + ' ' + quote},
            'reasons': [reason], 'notes': [{'body': 'User wrote DINOv2'}]}
    before = snapshot_hash(snapshot)
    public = public_view(snapshot, view)
    for value in [*public['summary']['insufficient'], *public['reasons']]:
        assert 'DINOv2' not in value and 'DDA' not in value and 'ViT' not in value
        assert 'frames below 336 px' in value
    assert quote in public['summary']['text']
    assert public['notes'] == view['notes'] and snapshot_hash(snapshot) == before
    assert view['summary']['insufficient'] == [reason]


@pytest.mark.parametrize('kind', ['image', 'video', 'audio', 'document'])
def test_generated_configuration_fields_hide_architectures_and_fusion_identity(kind):
    original = specimen(kind, 'forensic')
    fusion = f'fuser-{kind}-2.1.1'
    source_quote = f'Source said “ViT-S/16 and {fusion}”.'
    view = {
        'estimate': {'calibration': [{'subject': fusion, 'auc': .81}], 'statement': f'{fusion} recorded 0.81.'},
        'detectors': {'ledger': [{'id': 'S99', 'reference': 'ViT-S/16 trained on camera photographs and VAE reconstructions.',
                                  'observed': source_quote}],
                      'detectors': [{'id': 'internal', 'name': 'ViT-S/16', 'weights': 'private-weights-sha', 'version': '0.4.1'}]},
        'methods': {'method': [{'key': 'Fusion', 'value': f'Calibrated fusion ({fusion}): combine recorded responses.'}]},
        'appendix': {'rows': [{'key': 'Calibration id', 'value': fusion},
                             {'key': 'Libraries', 'value': 'open_clip 3.3.0 · torch 2.13.0'}],
                     'registry': [{'id': 'internal', 'weights': '6076002b', 'version': '0.4.1',
                        'coverage': 'not established (weights revision 6076002b, upstream code commit ee5b71d4 of 2025-10-18; no video generators in the reviewed training description)'}]},
        'notes': [{'body': source_quote}],
    }
    before = snapshot_hash(original)
    shown = public_view(original, view)
    assert shown['estimate']['calibration'][0] == {'subject': 'Combined assessment', 'auc': .81}
    generated = str({key: value for key, value in shown.items() if key not in {'notes', 'detectors'}})
    reference = shown['detectors']['ledger'][0]['reference']
    for name in (fusion, 'ViT-S/16', 'VAE', 'open_clip', '6076002b', 'ee5b71d4'):
        assert name not in generated + reference
    assert 'no video generators' in shown['appendix']['registry'][0]['coverage']
    assert shown['detectors']['detectors'][0]['weights'] == 'Withheld in public report'
    assert shown['detectors']['ledger'][0]['observed'] == source_quote
    assert shown['notes'] == view['notes'] and snapshot_hash(original) == before
    assert fusion in view['methods']['method'][0]['value']


@pytest.mark.parametrize('model_identity', ['aud.gustking_xlsr', 'wav2vec2-large-xlsr-deepfake-audio-classification'])
def test_audio_fuser_observed_scores_are_public_without_rewriting_source_values(model_identity):
    from app.api.public_projection import project

    original = specimen('audio', 'forensic')
    result = deepcopy(original.result)
    observed = f'fused p(synthetic)=0.870 over 9 speech window(s) of up to 4 s; per model: {model_identity}=0.87'
    result['signals'] = [
        {'id': 'synthetic_voice', 'display_id': 'S01', 'category': 'model', 'observed': observed},
        {'id': 'source', 'display_id': 'S02', 'category': 'provenance', 'detector_id': 'prov.metadata', 'observed': observed},
    ]
    frozen = original.model_copy(update={'result': result})
    before = snapshot_hash(frozen)
    view = {'detectors': {'ledger': [{'id': s['display_id'], 'observed': s['observed']} for s in result['signals']]}}
    shown = public_view(frozen, view)['detectors']['ledger']
    assert shown[0]['observed'] == project({'category': 'model', 'observed': observed})['observed']
    assert shown[0]['observed'] == project(result)['signals'][0]['observed']
    assert model_identity not in shown[0]['observed'] and '=0.87' in shown[0]['observed']
    assert shown[1]['observed'] == observed
    assert snapshot_hash(frozen) == before


def test_localiser_ledger_uses_neutral_reference_and_preserves_source_quotes():
    from app.detectors.image.adapters.localisers import MesorchDetector, RadarDetector

    original = specimen('image', 'forensic')
    result = deepcopy(original.result)
    result['manifest']['detectors'] = [{'id': d.spec.id, 'version': '1.0.0'}
                                       for d in (MesorchDetector, RadarDetector)]
    result['signals'] = [{'id': f'{d.spec.id}:global', 'display_id': f'S0{i}', 'detector_id': d.spec.id,
                          'name': d.spec.name, 'reference': d.REFERENCE,
                          'observed': f'{d.METHOD} integrity score 0.750; 2 regions; source says “RADAR”.'}
                         for i, d in enumerate((MesorchDetector, RadarDetector), 1)]
    result['signals'].append({'id': 'img.meta.exif:global', 'display_id': 'S03', 'detector_id': 'img.meta.exif',
                              'category': 'provenance', 'observed': 'Software=Mesorch', 'reference': 'Source RADAR'})
    snapshot = original.model_copy(update={'result': result})
    before = snapshot_hash(snapshot)
    view = {'detectors': {'ledger': [{'id': s['display_id'], 'name': s.get('name'), 'reference': s['reference'],
                                      'observed': s['observed']} for s in result['signals']]}}
    projected = public_view(snapshot, view)
    rows = projected['detectors']['ledger']
    for row in rows[:2]:
        assert 'uncalibrated local editing response' in row['reference'] and 'not proof of AI editing' in row['reference']
        assert all(name not in row['reference'] for name in
                   ('Mesorch', 'RADAR', 'ConvNeXt', 'SegFormer', 'DINOv2', 'Depth Anything', 'MiT-B3', 'ViT-B/14'))
        assert row['observed'] == 'A detection check integrity score 0.750; 2 regions; source says “RADAR”.'
    assert rows[2]['observed'] == 'Software=Mesorch' and rows[2]['reference'] == 'Source RADAR'
    assert snapshot_hash(snapshot) == before
    assert view['detectors']['ledger'][0]['reference'] == MesorchDetector.REFERENCE


def test_generated_fixed_baseline_narrative_hides_frozen_check_names_without_changing_evidence():
    from app.reports.forensic import from_fifty_line

    original = specimen('video', 'forensic')
    name = 'Gustking wav2vec2-XLSR deepfake audio classifier'
    result = deepcopy(original.result)
    result['manifest']['detectors'] = [{'id': 'aud.gustking_xlsr', 'version': '0.1'}]
    result['decision']['contributions'] = [{'kind': 'detector', 'label': name, 'baseline': 'p50'}]
    frozen = original.model_copy(update={'result': result, 'detector_names': {'aud.gustking_xlsr': name}})
    before = snapshot_hash(frozen)
    narrative = from_fifty_line(result['decision'])
    view = {'estimate': {'narrative': [narrative, f'Source note says “{name}”.'],
                         'bars': [{'kind': 'detector', 'label': name, 'value': '0.12'}]},
            'notes': [{'body': name}]}
    shown = public_view(frozen, view)
    assert shown['estimate']['narrative'][0] == (
        'A detection check has no neutral point of its own, so its bar is measured from a score of 50.')
    assert shown['estimate']['bars'][0]['label'] == 'Detection check'
    assert shown['estimate']['bars'][0]['value'] == '0.12'
    assert shown['estimate']['narrative'][1] == view['estimate']['narrative'][1]
    assert shown['notes'] == view['notes']
    assert snapshot_hash(frozen) == before and view['estimate']['narrative'][0] == narrative


def test_public_report_hides_checks_preserves_source_and_snapshot():
    original = specimen('image', 'forensic')
    result = deepcopy(original.result)
    result['manifest']['detectors'] = [{'id': 'img.dda', 'version': '1.0.0'}]
    result['signals'] = [{'id': 'img.dda:global', 'display_id': 'S1', 'detector_id': 'img.dda',
                          'name': 'DDA (DINOv2 ViT-L/14 + LoRA)', 'observed': 'source DDA',
                          'reference': 'DINOv2 ViT-L/14 with LoRA'}]
    result['findings'] = [{'id': 'F1', 'family': 'F3_whole_image_generation'},
                          {'id': 'F2', 'family': 'F2_self_declared_metadata'}]
    snapshot = original.model_copy(update={'result': result})
    before = snapshot_hash(snapshot)
    view = {'findings': [{'id': 'F1', 'title': 'DDA result',
                          'sentences': ['DDA scored 0.4; source says “DDA”.'],
                          'refs': [{'label': 'img.dda', 'href': '#det-img-dda'}]},
                         {'id': 'F2', 'title': 'Source metadata mentions DDA', 'sentences': ['Source says “DDA”.']}],
            'detectors': {'ledger': [{'id': 'S1', 'name': 'DDA', 'observed': 'source DDA',
                                      'reference': 'DINOv2 ViT-L/14 with LoRA', 'anchor': 'sig-img-dda-global'}],
                          'not_run': [], 'detectors': [{'id': 'img.dda', 'anchor': 'det-img-dda',
                                                        'name': 'DDA', 'citation': 'DDA paper'}]},
            'appendix': {'registry': [{'id': 'img.dda'}], 'rows': []},
            'exhibits': [{'kind': 'text_spans', 'title': 'DDA score',
                          'table': {'columns': ['Sentence', 'Reasons'], 'rows': [['DDA is the original text.', 'DDA scored 0.4.']]}}],
            'notes': [{'body': 'User wrote DDA'}]}
    public = public_view(snapshot, view)
    assert public['findings'][0]['sentences'] == ['A detection check scored 0.4; source says “DDA”.']
    assert public['findings'][1]['title'] == 'Source metadata mentions DDA'
    assert public['detectors']['ledger'][0]['observed'] == 'source DDA'
    assert public['detectors']['ledger'][0]['reference'] == 'A detection check'
    assert public['exhibits'][0]['table']['rows'][0] == ['DDA is the original text.', 'A detection check scored 0.4.']
    assert public['notes'][0]['body'] == 'User wrote DDA'
    assert public['findings'][0]['refs'][0]['href'] == '#det-check-1'
    assert public['detectors']['detectors'][0]['anchor'] == 'det-check-1'
    assert snapshot_hash(snapshot) == before
    assert view['findings'][0]['title'] == 'DDA result'


def test_public_attachment_is_commitment_not_private_manifest():
    snapshot = specimen('image', 'forensic')
    attachment = public_manifest(snapshot, 'weasyprint test', 'test-template')
    assert 'manifest' not in attachment
    assert attachment['snapshot_sha256'] == snapshot_hash(snapshot)
    assert 'withheld' in attachment['note']


def test_generated_localiser_limitation_hides_names_preserving_benchmark_and_source():
    from app.engine.fusion.disclosures import DISCLOSURES
    from app.reports.forensic import applied_limitations

    original = specimen('image', 'forensic')
    result = deepcopy(original.result)
    # This generated limitation discusses another localizer's evaluation even
    # when that check is absent from the snapshot's recorded detector names.
    result['manifest']['detectors'] = [{'id': 'loc.radar', 'version': '0.4.1'}]
    result['signals'] = [{'id': 'loc.radar:global', 'detector_id': 'loc.radar', 'status': 'ok', 'name': 'RADAR'}]
    snapshot = original.model_copy(update={'result': result, 'detector_names': {'loc.radar': 'RADAR'}, 'model_cards': []})
    before = snapshot_hash(snapshot)
    generated = next(d.text for d in DISCLOSURES if d.id == 'image.localisers')
    limits = applied_limitations(snapshot, {'fusion_mode': 'artefact'})
    assert generated in [row['text'] for row in limits]
    quoted = 'Source note says “Mesorch”; the source also mentions RADAR.'
    limits.append({'id': 'L-source', 'text': quoted})
    view = {'limitations': limits, 'summary': {'limitation': generated},
            'not_validated': 'Not validated for this result: SPAI and UnivFD.'}
    public = public_view(snapshot, view)
    rows = [r['text'] for r in public['limitations']]
    selected = next(text for text in rows if text.startswith('Region maps are not validated'))
    assert 'Mesorch' not in selected
    assert '36 % of genuine photographs' in selected
    assert 'not an error rate for every check or image' in selected
    assert 'not evidence of an edit on its own' in selected
    assert public['summary']['limitation'] == selected
    assert public['limitations'][-1]['text'] == quoted
    assert 'SPAI' not in public['not_validated'] and 'UnivFD' not in public['not_validated']
    assert snapshot_hash(snapshot) == before and view['summary']['limitation'] == generated


@pytest.mark.parametrize('profile', ['standard', 'forensic'])
def test_partial_document_main_limitation_uses_the_same_public_prose_as_its_list(profile):
    from app.engine.stages.document import enforce_extraction_scope
    from app.reports import engine

    # Real partial-OCR shape: measured text findings survive, but the whole-file
    # decision becomes not_assessable. Its first caveat supplies Main limitation.
    original = specimen('document', profile)
    generated = 'Text classifier that scored this document: desklib-ft-v2 (Veritas fine-tune, DeBERTa-v3-large).'
    source = 'Source note says “DeBERTa”; the author also wrote desklib-ft-v2.'
    result = deepcopy(original.result)
    result['caveats'] = [generated, source]
    result = enforce_extraction_scope(result, SimpleNamespace(meta={'unmapped_pages': [2]}))
    frozen = original.model_copy(update={'result': result})
    before = snapshot_hash(frozen)
    raw = engine.view_of(frozen, version='test')
    assert raw['summary']['limitation'] == 'Main limitation: ' + generated
    assert frozen.result['decision']['code'] == 'not_assessable'

    shown = public_view(frozen, raw)
    main = shown['summary']['limitation']
    index = next(i for i, row in enumerate(raw['limitations']) if row['text'] == generated)
    public_limit = shown['limitations'][index]['text']
    assert main.removeprefix('Main limitation: ').casefold() == public_limit.casefold()
    assert all(name not in main for name in ('desklib', 'Veritas fine-tune', 'DeBERTa'))
    assert source in [row['text'] for row in shown['limitations']]
    html = engine.render_html(frozen, embed_fonts=False)
    assert main in html and 'Main limitation: ' + generated not in html
    assert source in html
    assert snapshot_hash(frozen) == before
    assert raw['summary']['limitation'] == 'Main limitation: ' + generated


def test_historical_email_report_uses_current_customer_readout_without_changing_snapshot():
    original = specimen('document', 'forensic')
    result = deepcopy(original.result)
    result['condition_profile'] = {'slice_id': 'document/local-email-v1', 'within_validated_envelope': True}
    result['decision'].update(code='ai_supported', probability_ai_calibrated=0.7, fusion_mode='artefact',
                              provenance_record=None, contributions=[], ci90=[0.6, 0.8])
    frozen = original.model_copy(update={'result': result})
    before = snapshot_hash(frozen)
    raw = {'header_left': '', 'summary': {}, 'estimate': {'rates': [{'context': 'Held-out plain-text business email',
                                                    'threshold': 'fixed · 0.60'}]}, 'sections': []}
    shown = public_view(frozen, raw)
    assert shown['customer_assessment']['direction'] == 'inconclusive'
    assert shown['customer_assessment']['confidence_pct'] is None
    assert shown['estimate']['rates'] == []
    assert 'does not support a reliable' in shown['estimate']['none']
    assert snapshot_hash(frozen) == before and raw['estimate']['rates']


def test_calibration_component_is_private_even_without_a_run_or_manifest_entry():
    frozen = specimen('video', 'standard')
    frozen.result['calibration'] = [{'subject': 'aud.retired_private@unknown', 'method': 'platt'}]
    shown = public_view(frozen, {'estimate': {'calibration': deepcopy(frozen.result['calibration'])}})
    assert shown['estimate']['calibration'][0]['subject'].startswith('Check ')
    assert 'retired_private' not in str(shown)


def test_skipped_check_diagnostics_are_redacted_even_inside_generated_quotes():
    frozen = specimen('video', 'standard')
    frozen.result['findings'] = [{'id': 'F1', 'family': 'coverage'}]
    view = {'findings': [{'id': 'F1', 'sentences': ['Not run: “YuNet skipped: memory budget”.']}],
            'limitations': [{'text': 'A check that this analysis depends on did not run: YuNet (memory budget).'}]}
    shown = public_view(frozen, view)
    assert 'YuNet' not in str(shown)


def test_scope_and_counted_missing_check_limitations_do_not_name_models():
    frozen = specimen('video', 'standard')
    view = {'scope': {'questions': [{'how': 'No estimate: DFDC EfficientNet-B7 did not run.'}]},
            'limitations': [{'text': '1 check could not run on this server: YuNet (memory budget).'}]}
    shown = public_view(frozen, view)
    assert 'DFDC' not in str(shown) and 'YuNet' not in str(shown)
