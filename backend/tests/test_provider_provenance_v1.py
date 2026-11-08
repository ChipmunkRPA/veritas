"""A missing check, HTTP failure or malformed response cannot verify AI origin."""
import json

import pytest

from scripts.provider_provenance_v1 import summarize


def bria(present=False, trusted=False, state=None, mark=False):
    credential = {'present': present, 'is_trusted': trusted}
    if state is not None:
        credential['validation_state'] = state
    result = {'verified': trusted or mark is True, 'c2pa': credential}
    if mark is not None:
        result['invisible_watermark_detected'] = mark
    return {'result': result, 'request_id': 'authored-contract-example'}


@pytest.mark.parametrize('state,trusted', [('Trusted', True), ('Valid', False), ('Invalid', False)])
def test_credential_trust_and_watermark_are_separate(state, trusted):
    result = summarize('bria', 'image/png', 200, json.dumps(bria(True, trusted, state)).encode())
    assert result['status'] == 'completed' and result['c2pa'] == state
    assert result['provider_reported_match'] is trusted and result['early_stop_eligible'] is False


@pytest.mark.parametrize('mime', ['image/webp', 'image/gif'])
def test_inapplicable_watermark_is_omitted_and_not_a_negative(mime):
    result = summarize('bria', mime, 200, json.dumps(bria(mark=None)).encode())
    assert result['status'] == 'completed' and result['watermark'] == 'not_run'


@pytest.mark.parametrize('body', [b'{}', b'[]', b'<html>not JSON</html>',
    b'{"result":{},"result":{}}', b'{"result":NaN}',
    json.dumps(bria(mark=None)).encode(),
    b'{"request_id":"example","result":{"verified":"true","c2pa":{"present":false,"is_trusted":false}}}',
    json.dumps({**bria(), 'result': {**bria()['result'], 'verified': True}}).encode(),
    json.dumps(bria(False, True, 'Trusted')).encode()])
def test_incomplete_contradictory_or_coerced_responses_remain_inconclusive(body):
    result = summarize('bria', 'image/png', 200, body)
    assert result['status'] == 'inconclusive' and result['provider_reported_match'] is None


@pytest.mark.parametrize('http', [None, 202, 301, 400, 401, 403, 429, 500])
def test_http_failure_or_pending_is_never_a_negative_or_positive(http):
    result = summarize('suno', 'audio/wav', http, b'{"verdict":"verified_suno"}')
    assert result['status'] == 'unavailable' and result['provider_reported_match'] is None


@pytest.mark.parametrize('verdict,matched', [('verified_suno', True), ('no_suno_provenance', False),
                                          ('inconclusive', None)])
def test_suno_verdicts_are_not_ai_confidence_scores(verdict, matched):
    result = summarize('suno', 'audio/wav', 200, json.dumps({'verdict': verdict}).encode())
    assert result['provider_reported_match'] is matched
    assert result['confidence_pct'] is None and result['used_in_confidence'] is False
    assert result['early_stop_eligible'] is False
