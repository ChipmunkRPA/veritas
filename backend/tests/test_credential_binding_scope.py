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
