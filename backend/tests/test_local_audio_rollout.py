"""Serving calibration and rollback checks; no benchmark fitting or network calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.config import settings
from app.detectors.audio import calibration as c
from app.engine.customer_assessment import customer_assessment
from app.engine.fusion.registry import fusion_status, select_for_registry
from app.engine.registry import get_registry


def test_frozen_bundle_and_saved_holdout_parity():
    # Consumed holdout: engineering parity only, never a fresh accuracy estimate.
    path = Path(__file__).parents[1] / 'data/independent-evaluation/audio-v3/predictions.json'
    if not path.exists():
        pytest.skip('Local frozen evaluation is not distributed with the source')
    bundle = c.load()
    rows = json.loads(path.read_text())
    for row in rows:
        p = c.probability(row['new_raw'], bundle)
        assert p == pytest.approx(row['new_p'], abs=1e-12)
        direction = c.initial_decision(p, bundle)
        if direction == 'original':
            probe = row.get('new_band')
            if not c.valid(probe) or abs(c.logit(round(row['new_raw'], 6))-c.logit(round(probe, 6))) > np.log(2):
                direction = 'inconclusive'
        assert direction == row['new_guarded']


def test_bundle_tampering_fails_closed(tmp_path, monkeypatch):
    p = tmp_path / 'bundle.json'
    p.write_bytes(c.PATH.read_bytes() + b' ')
    monkeypatch.setattr(c, 'PATH', p)
    with pytest.raises(ValueError, match='integrity'):
        c.load()


@pytest.mark.parametrize('raw', [None, True, float('nan'), float('inf'), -1, 2])
def test_invalid_raw_not_probability(raw):
    assert c.probability(raw) is None


def test_long_audio_has_no_whole_recording_probability():
    pcm = np.zeros(128000, np.float32)
    result = c.evaluate(.999, pcm, [(0., pcm[:64000]), (4., pcm[64000:])], None,
                        SimpleNamespace(check=lambda *a: None))
    assert result['decision'] == 'inconclusive' and result['probability_ai'] is None
    assert result['reference_estimate_ai'] > .8


def test_failed_stability_probe_withholds_original():
    pcm = np.zeros(64000, np.float32)
    result = c.evaluate(.00001, pcm, [(0., pcm)], SimpleNamespace(predict=lambda _: [None]),
                        SimpleNamespace(check=lambda *a: None))
    assert result['decision'] == 'inconclusive'
    assert result['original_stability_veto'] and result['probe_status'] == 'unavailable'


@pytest.mark.parametrize('code,p,direction,confidence', [('ai_supported', .92, 'ai', 92.),
    ('no_ai_indicators', .08, 'original', 92.), ('inconclusive', .92, 'inconclusive', None),
    ('inconclusive', .08, 'inconclusive', None), ('inconclusive', None, 'inconclusive', None)])
def test_customer_result_uses_saved_calibration_and_guards(code, p, direction, confidence):
    result = customer_assessment({'kind': 'audio', 'status': 'done',
        'decision': {'fuser_sha256': c.SHA256, 'code': code, 'verbal': 'Speech assessment',
                     'probability_ai_calibrated': p, 'reason': 'Uncertainty check'},
        'condition_profile': {'within_validated_envelope': False}})
    assert result['direction'] == direction and result['confidence_pct'] == confidence
    assert result['interval_pct'] is None


@pytest.mark.parametrize('scope, p, guarded, veto, direction, confidence', [
    (True, .921, 'ai', False, 'ai', 92.1),
    (True, .08, 'original', False, 'original', 92.),
    (True, .92, 'inconclusive', False, 'inconclusive', None),
    (True, .08, 'original', True, 'inconclusive', None),
    (False, .92, 'ai', False, 'inconclusive', None),
])
def test_video_soundtrack_uses_only_eligible_saved_speech(scope, p, guarded, veto, direction, confidence):
    screen = {'calibration_sha256': c.SHA256, 'scope_supported': scope, 'probability_ai': p,
              'decision': guarded, 'original_stability_veto': veto}
    signal = {'status': 'ok', 'scope': 'global', 'modality': 'audio', 'id': 'S02',
              'calibration_ref': c.SHA256, 'details': {'audio_screening': screen}, 'effective_weight': 0.}
    baseline = {'kind': 'video', 'status': 'done', 'signals': [signal]}
    result = customer_assessment(baseline)
    assert result['direction'] == 'unavailable' and result['confidence_pct'] is None
    assert result['soundtrack']['direction'] == direction and result['soundtrack']['confidence_pct'] == confidence
    signal['calibration_ref'] = 'different-reference'
    assert 'soundtrack' not in customer_assessment(baseline)


def test_candidate_switch_and_legacy_rollback(monkeypatch):
    registry = get_registry()
    monkeypatch.setattr(settings, 'audio_detector_mode', 'candidate')
    assert not registry.decision('aud.gustking_xlsr').enabled
    assert registry.decision('aud.pella_v2').enabled
    assert fusion_status(registry, 'full')['audio'] == 'artefact'
    video = select_for_registry(registry, 'video', 'full')
    assert video.mode == 'artefact'
    assert video.intentionally_missing == ('aud.gustking_xlsr',)
    monkeypatch.setattr(settings, 'audio_detector_mode', 'legacy')
    assert registry.decision('aud.gustking_xlsr').enabled
    assert select_for_registry(registry, 'audio', 'full').mode == 'artefact'


def test_video_missing_feature_cannot_bypass_other_compatibility_checks():
    import copy

    from app.engine.fusion.registry import SCHEMA, compatibility
    doc = {'schema': SCHEMA, 'modality': 'video', 'reviewed_by': 'test',
           'detectors': [{'id': 'old-voice', 'weights_sha12': 'abc'}, {'id': 'picture', 'weights_sha12': 'def'}],
           'coef': {'miss:old-voice': -.1}}
    assert compatibility(doc, 'video', {'picture': 'def'}, permitted_missing={'old-voice': 'abc'}) is None
    assert compatibility(doc, 'video', {'picture': 'def'}) is not None
    assert compatibility(doc, 'video', {'picture': 'def'}, permitted_missing={'old-voice': 'wrong'}) is not None
    assert compatibility(doc, 'video', {}, permitted_missing={'old-voice': 'abc'}) is not None
    bad = copy.deepcopy(doc)
    bad['coef'] = {}
    assert compatibility(bad, 'video', {'picture': 'def'}, permitted_missing={'old-voice': 'abc'}) is not None


def test_new_report_never_reuses_old_audio_error_rate():
    from app.reports.view import _guidance
    for code in ['no_ai_indicators', 'no_ai_indicators_strong']:
        decision = {'code': code, 'fuser_sha256': c.SHA256}
        assert not any('1 of 89' in x for group in _guidance('audio', decision, 'statistical') for x in group)
        decision['fuser_sha256'] = 'legacy'
        assert any('1 of 89' in x for group in _guidance('audio', decision, 'statistical') for x in group)


def test_old_inspection_record_cannot_be_relabelled_as_calibrated(monkeypatch):
    from app.engine.stages import audio_screening
    monkeypatch.setattr(audio_screening.video, 'synthesis_signals', lambda *a: [
        {'id': 'voice', 'detector_id': 'aud.pella_v2', 'details': {}}])
    monkeypatch.setattr(audio_screening.video, 'attach_temporal', lambda *a: None)
    with pytest.raises(ValueError, match='no candidate calibration'):
        audio_screening.fuse(SimpleNamespace())
