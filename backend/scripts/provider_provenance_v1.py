"""Strict research-only summaries of provider verification responses.

No HTTP, credentials, serving registry, fusion or model imports. Provider
responses remain documentary context until the route is separately qualified.
"""
from __future__ import annotations

import json

VERSION = 'provider-provenance-qualification/1'


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate response field')
        result[key] = value
    return result


def _constant(_):
    raise ValueError('Non-finite JSON constant')


def summarize(provider: str, mime: str, http_status: int | None, body: bytes) -> dict:
    result = {'version': VERSION, 'provider': provider, 'http_status': http_status,
        'status': 'unavailable', 'provider_reported_match': None, 'c2pa': 'not_assessed',
        'watermark': 'not_assessed', 'confidence_pct': None, 'used_in_confidence': False,
        'early_stop_eligible': False, 'scope': 'Research-only provider response; not a complete authorship finding.'}
    if http_status != 200:
        return {**result, 'reason': 'The provider did not return a successful verification response.'}
    try:
        if len(body) > 65536:
            raise ValueError('Oversized response')
        document = json.loads(body, object_pairs_hook=_pairs, parse_constant=_constant)
        if not isinstance(document, dict):
            raise ValueError('Expected response object')
        if provider == 'suno':
            verdict = document.get('verdict')
            if verdict not in ('verified_suno', 'no_suno_provenance', 'inconclusive'):
                raise ValueError('Unrecognized Suno verdict')
            return {**result, 'status': 'inconclusive' if verdict == 'inconclusive' else 'completed',
                'provider_reported_match': {'verified_suno': True, 'no_suno_provenance': False,
                                           'inconclusive': None}[verdict], 'c2pa': verdict,
                'reason': 'Only the Suno credentials verdict is recorded; absence does not establish originality.'}
        if provider != 'bria':
            raise ValueError('Unsupported verifier')
        if not isinstance(document.get('request_id'), str) or not 0 < len(document['request_id']) <= 256:
            raise ValueError('Missing request identifier')
        record = document.get('result')
        if not isinstance(record, dict) or type(record.get('verified')) is not bool:
            raise ValueError('Missing result.verified boolean')
        credentials = record.get('c2pa')
        if not isinstance(credentials, dict) or any(type(credentials.get(k)) is not bool
                for k in ('present', 'is_trusted')):
            raise ValueError('Missing C2PA booleans')
        present, trusted, state = (credentials.get(k) for k in ('present', 'is_trusted', 'validation_state'))
        if (present and state not in ('Trusted', 'Valid', 'Invalid')) or (not present and state is not None):
            raise ValueError('Inconsistent C2PA presence/state')
        if trusted != (present and state == 'Trusted'):
            raise ValueError('Inconsistent C2PA trust/state')
        mark_expected = mime in ('image/png', 'image/jpeg', 'video/mp4', 'video/quicktime', 'video/x-m4v')
        mark = record.get('invisible_watermark_detected')
        if mark_expected and type(mark) is not bool:
            raise ValueError('Applicable watermark check is missing')
        if not mark_expected and ('invisible_watermark_detected' in record or mime not in ('image/gif', 'image/webp')):
            raise ValueError('Unexpected watermark/format contract')
        if record['verified'] != (trusted or mark is True):
            raise ValueError('Inconsistent verification signals')
        return {**result, 'status': 'completed', 'provider_reported_match': record['verified'],
            'c2pa': state if present else 'absent',
            'watermark': ('detected' if mark else 'not_detected') if mark_expected else 'not_run',
            'reason': 'Bria reports provenance signals. Signed-by-Bria alone does not supply an explicit AI declaration; '
                      'missing credentials/marks do not establish originality.'}
    except (ValueError, TypeError, RecursionError, UnicodeError):
        return {**result, 'status': 'inconclusive', 'reason': 'The response does not meet the reviewed verifier contract.'}
