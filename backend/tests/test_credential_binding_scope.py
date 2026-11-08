"""Trust/date failures must not be described as content tampering."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.detectors.image.adapters.provenance import C2PADetector, c2pa_result
from app.engine.provenance_gate import decide
from app.engine.stages.common import detector_entry
from app.provenance import c2pa


def code(value, level='success', scope='active manifest'):
    return c2pa.ValidationCode(level, value, scope, '', '')


@pytest.mark.parametrize('binding', sorted(c2pa.HARD_BINDING_MATCH))
def test_ingredient_hash_never_binds_the_active_file(binding):
    report = c2pa.C2PAReport(state='trusted', codes=[code(binding, scope='ingredient example')])
    assert report.binding_status == 'not_verified' and not report.covers_bytes
    assert 'not verified' in report.binding_description


def test_expired_or_untrusted_certificate_does_not_turn_a_matching_hash_into_mismatch():
    report = c2pa.C2PAReport(state='invalid', codes=[code('assertion.dataHash.match'),
        code('signingCredential.expired', 'failure'), code('signingCredential.untrusted', 'failure')])
    assert report.binding_status == 'matched' and not report.covers_bytes
    assert 'matches the current file bytes' in report.binding_description
    assert 'fail other validation checks' in report.binding_description


def test_explicit_active_binding_mismatch_takes_precedence_over_success():
    report = c2pa.C2PAReport(state='trusted', codes=[code('assertion.dataHash.match'),
        code('assertion.dataHash.mismatch', 'failure')])
    assert report.binding_status == 'mismatch' and not report.covers_bytes


@pytest.mark.parametrize('failures,has_match,expected', [
    (['signingCredential.expired', 'signingCredential.untrusted'], True, 'matched'),
    (['claimSignature.mismatch'], False, 'not_verified'),
    (['assertion.dataHash.mismatch'], False, 'mismatch'),
])
def test_report_separates_credential_failures_from_tampering(failures, has_match, expected, monkeypatch):
    report = c2pa.C2PAReport(state='invalid', codes=[*(
        [code('assertion.dataHash.match')] if has_match else []), *[code(v, 'failure') for v in failures]])
    monkeypatch.setattr(c2pa, 'read', lambda *args, **kwargs: report)
    table = Mock(return_value='table')
    ctx = SimpleNamespace(exhibits=SimpleNamespace(table=table, image=Mock()))
    result = c2pa_result(ctx, b'\x89PNG\r\n\x1a\n' + bytes(20))
    assert result.provenance_override is None
    assert result.metrics['covers_bytes'] == 0.
    assert result.metrics['binding_' + ('not_verified' if expected == 'not_verified' else expected)] == 1.
    assert ('changed after signing' not in result.reasoning)
    if has_match:
        assert 'expired' in result.reasoning and 'not trusted' in result.reasoning
        assert 'does not match' not in result.reasoning
    entry = detector_entry(C2PADetector.spec, result, 'authored-state-validation')
    assert decide({'detectors': [entry]}, 'image')['action'] == 'continue_detection'


@pytest.mark.parametrize('kind,detector', [('image', 'prov.c2pa'), ('video', 'prov.c2pa.video'),
                                        ('audio', 'prov.c2pa.audio')])
def test_findings_and_all_report_profiles_explain_expiration_without_inventing_tampering(kind, detector):
    from copy import deepcopy

    from app.engine.synthesis import SynthesisInput, synthesise
    from app.reports import engine
    from app.reports.specimens import specimen
    from app.schemas.evidence import Decision, SignalV2

    signal = SignalV2(id=detector + ':global', display_id='S01', detector_id=detector,
        detector_version='authored-test', name='Content Credentials', family='provenance',
        modality=kind, category='provenance', hypothesis='H3_provenance_integrity', status='ok', scope='global',
        raw_score=0., raw_kind='score', effective_weight=0., observed='Credentials were not accepted.',
        metrics={'invalid': 1., 'trusted': 0., 'valid': 0., 'binding_matched': 1., 'binding_mismatch': 0.,
                 'signing_certificate_expired': 1., 'signer_untrusted': 1.}).model_dump(mode='json')
    decision = Decision(code='not_assessable', verbal='No content estimate was formed', band='uncertain',
        confidence='low', reason='No statistical content checks ran.', fusion_mode='fallback').model_dump(mode='json')
    findings = synthesise(SynthesisInput(kind=kind, decision=decision, signals=[signal]))
    finding = next(f for f in findings if signal['id'] in f['evidence_ids'])
    assert finding['title'] == 'Content Credentials that do not validate'
    assert 'The signing certificate is expired' in finding['statement']
    assert 'hash matches the current file bytes' in finding['statement']
    assert 'not trusted' in finding['statement'] and 'changed after signing' not in finding['statement']
    assert finding['probability'] is None
    for profile in ('summary', 'standard', 'forensic', 'newsroom'):
        snapshot = specimen(kind, profile)
        result = deepcopy(snapshot.result)
        result.update(signals=[signal], findings=findings, decision=decision)
        result.setdefault('manifest', {}).setdefault('sampling', {})['source_labels'] = {
            'version': 'source-labels/2', 'source_sha256': snapshot.media.sha256, 'status': 'checked',
            'records': [], 'checked_metadata': ['PNG XMP metadata'], 'issues': [],
            'notice': 'Unverified metadata is outside confidence.',
            'provider_private_watermarks_checked': False, 'used_in_confidence': False}
        html = engine.render_html(snapshot.model_copy(update={'result': result}), embed_fonts=False)
        assert 'changed after signing' not in html
        assert 'checked_metadata' not in html and 'provider_private_watermarks_checked' not in html
        if profile in ('standard', 'forensic'):
            assert 'credential' in html.lower()
            assert 'Credentials were not accepted.' in html
            assert 'Content Credentials were not checked' not in html
