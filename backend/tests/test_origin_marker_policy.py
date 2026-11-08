"""Copied origin markers cannot decide authenticity or contribute confidence."""
import copy
import hashlib
import json

import pytest

from app.engine.customer_assessment import customer_assessment
from app.engine.fusion import decisive, runtime
from app.engine.fusion.artefact import Fuser
from app.engine.fusion.registry import RELEASE_DIR, Selection
from app.engine.stages.video import provenance_override


def entry(detector='img.meta.generator', value=.99, raw_kind='probability'):
    return {'detector_id': detector, 'status': 'ok', 'name': 'Origin marker', 'family': 'provenance',
            'provenance_override': 'self_declared_ai', 'hypotheses': [runtime.H3],
            'aggregate': {'unit_ref': 'global', 'raw': value, 'raw_kind': raw_kind},
            'metrics': {}, 'version': 'test'}


@pytest.mark.parametrize('p', [0., .1, .5, .9, 1.])
def test_advisory_marker_never_floors_or_decides(p):
    outcome = decisive.apply(p, ['self_declared_ai'])
    assert outcome.probability == outcome.statistical == p
    assert decisive.floor_or_cap(p, 'self_declared_ai') == p
    assert outcome.kind is None
    assert outcome.not_applied == ('self_declared_ai',)
    assert decisive.ADVISORY_CAVEAT in outcome.caveats
    assert decisive.resolve(['self_declared_ai'], historical=True) == 'self_declared_ai'


@pytest.mark.parametrize('kind,target', [('decisive_ai', .97), ('trusted_capture', .10)])
def test_documentary_precedence_is_retained(kind, target):
    result = decisive.apply(.5, ['self_declared_ai', kind])
    assert result.kind == kind and result.probability == target


@pytest.mark.parametrize('raw_kind,value', [('probability', .99), ('logit', 10.), ('llr', 10.)])
def test_raw_marker_response_cannot_become_an_origin_probability(raw_kind, value):
    row = entry(raw_kind=raw_kind, value=value)
    assert runtime.score_of(row) is None
    assert not runtime.unusable_score(row)  # Policy exclusion is not a failed measurement.
    signal = {'id': 'marker', 'detector_id': row['detector_id'], 'provenance_override': 'self_declared_ai',
              'raw_score': value, 'effective_weight': 1., 'legacy': {'direction': 'supports_ai'}}
    annotated = runtime.annotate([signal], {'marker': {'calibrated_probability': .99, 'contribution_logit': 10.}},
                                 artefact_detectors=[row['detector_id']])[0]
    assert annotated['raw_score'] == value
    assert annotated['effective_weight'] == 0 and annotated['calibrated_probability'] is None
    assert annotated['contribution_logit'] is None and annotated['legacy']['direction'] == 'inconclusive'


def selection(retired_input=False):
    path = RELEASE_DIR / 'fuser-image-2.1.1.json'
    data = path.read_bytes()
    if retired_input:
        data = data.replace(b'img.dda', b'img.meta.generator')
    fuser = Fuser(json.loads(data))
    selected = Selection('image', 'artefact', fuser, hashlib.sha256(data).hexdigest())
    rows = [{'detector_id': d, 'name': 'Recorded content check', 'status': 'ok', 'family': 'learned-whole-image',
             'hypotheses': [runtime.H1], 'aggregate': {'unit_ref': 'global', 'raw': .15, 'raw_kind': 'probability'}}
            for d in fuser.detectors if d not in decisive.MARKER_DETECTORS]
    return selected, rows


@pytest.mark.parametrize('detector', ['img.meta.generator', 'img.watermark.dwtdct', 'prov.c2pa',
                                     'prov.c2pa.video', 'prov.c2pa.audio'])
def test_real_packaged_calculation_unchanged_with_marker(detector):
    sel, rows = selection()
    condition = {'context': {'min_side': 512, 'jpeg_q': 95}}
    baseline = runtime.fuse_with_artefact(sel, rows, condition, kind='image')
    marked = runtime.fuse_with_artefact(sel, [*rows, entry(detector)], condition, kind='image')
    assert marked['decision'] == baseline['decision']
    assert marked['signal_fields'] == baseline['signal_fields']
    assert decisive.ADVISORY_CAVEAT in marked['caveats']


@pytest.mark.parametrize('present', [False, True])
def test_calibration_that_depended_on_retired_marker_cannot_publish_confidence(present):
    sel, rows = selection(retired_input=True)
    result = runtime.fuse_with_artefact(sel, [*rows, *([entry()] if present else [])],
                                      {'context': {'min_side': 512}}, kind='image')
    assert result['decision']['code'] == 'inconclusive'
    assert result['decision']['probability_ai_calibrated'] is None
    assert any('needs revalidation' in c for c in result['caveats'])


def test_video_marker_leaves_content_decision_unchanged():
    decision = {'code': 'inconclusive', 'probability_ai_uncalibrated': None}
    before = copy.deepcopy(decision)
    got, caveats = provenance_override(decision, [], {**entry('prov.c2pa.video'), 'id': 'marker'})
    assert got == before == decision and caveats == [decisive.ADVISORY_CAVEAT]


@pytest.mark.parametrize('kind', ['image', 'video', 'audio', 'document'])
@pytest.mark.parametrize('record_field', ['provenance_record', 'stored_record', 'legacy_signals'])
def test_retired_marker_interpretation_preserves_frozen_calculation(kind, record_field):
    decision = {'code': 'ai_supported', 'probability_ai_uncalibrated': .85, 'fusion_mode': 'fallback',
                'driving_hypothesis': runtime.H3}
    result = {'kind': kind, 'status': 'done', 'decision': decision, 'signals': [entry()]}
    record = {'kind': 'self_declared_ai', 'marker': 'metadata', 'signal_ids': []}
    if record_field == 'provenance_record':
        decision['provenance_record'] = record
    elif record_field == 'stored_record':
        result['stored_record'] = record
    before = copy.deepcopy(result)
    got = customer_assessment(result)
    assert got['applies'] and got['direction'] == 'inconclusive'
    assert got['confidence_pct'] is None and got['probability_ai_pct'] is None
    assert 'retired rule' in got['meaning'] and result == before
