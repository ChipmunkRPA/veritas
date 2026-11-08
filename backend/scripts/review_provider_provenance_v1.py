"""Separate offline byte/response review of the completed provider study."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from collections import Counter
from pathlib import Path

from app.provenance import c2pa

PLAN_SHA256 = '1775e69325de94c55755aa2e232561d39af521616d3b7945ce23dd0bf25a5516'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def review(folder):
    plan_bytes = (folder / 'plan.json').read_bytes()
    assert sha(plan_bytes) == PLAN_SHA256
    plan = json.loads(plan_bytes)
    root = Path(__file__).resolve().parents[2]
    for relative, digest in plan['code'].items():
        assert sha((root / relative).read_bytes()) == digest
    rows = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
    complete = json.loads((folder / 'complete.json').read_text())
    assert len(rows) == len(plan['cases']) == complete['cases_recorded'] == 9
    assert [row['id'] for row in rows] == [case['id'] for case in plan['cases']]
    assert len(set(row['id'] for row in rows)) == 9
    counts = {}
    discrepancies = Counter()
    for case, row in zip(plan['cases'], rows, strict=True):
        data = (folder / case['path']).read_bytes()
        assert sha(data) == case['sha256'] == row['source_sha256']
        assert len(data) == case['bytes']
        if case['provider'] == 'bria':
            body = json.dumps({'image': base64.b64encode(data).decode()}, separators=(',', ':')).encode()
            assert base64.b64decode(json.loads(body)['image'], validate=True) == data
        else:
            boundary = 'chipmunk-qualification-' + sha(data)[:24]
            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="authored.wav"\r\n'
                    f'Content-Type: {case["mime"]}\r\n\r\n').encode() + data + f'\r\n--{boundary}--\r\n'.encode()
            assert body.split(b'\r\n\r\n', 1)[1][:-len(f'\r\n--{boundary}--\r\n'.encode())] == data
        assert sha(body) == row['request_body_sha256']
        response = (folder / (case['id'] + '.response')).read_bytes()
        assert sha(response) == row['response_sha256'] and len(response) == row['response_bytes']
        raw = json.loads(response)
        summary = row['summary']
        assert summary['confidence_pct'] is None
        assert summary['early_stop_eligible'] is summary['used_in_confidence'] is False
        facts = counts.setdefault(case['provider'], Counter())
        facts['cases'] += 1
        facts['request_attempts'] += int(row['attempted'])
        facts['http_200'] += int(row['http_status'] == 200)
        facts['transport_errors'] += int(row['error'] is not None)
        facts['summary_' + summary['status']] += 1
        if case['provider'] == 'bria':
            result = raw['result']
            credential = result['c2pa']
            raw_mark = result.get('invisible_watermark_detected')
            assert type(result['verified']) is bool
            assert result['verified'] == (credential['is_trusted'] or raw_mark is True)
            facts['provider_reported_matches'] += int(result['verified'])
            facts['credentials_' + (credential['validation_state'] if credential['present'] else 'absent')] += 1
            if case['mime'] == 'image/webp':
                assert 'invisible_watermark_detected' in result and raw_mark is False
                assert summary['status'] == 'inconclusive' and summary['provider_reported_match'] is None
                discrepancies['webp_unexpected_watermark_boolean'] += 1
            else:
                assert type(raw_mark) is bool and summary['status'] == 'completed'
                assert summary['provider_reported_match'] == result['verified']
        else:
            assert raw['verdict'] == 'no_suno_provenance'
            assert summary['status'] == 'completed' and summary['provider_reported_match'] is False
            facts['provider_reported_matches'] += 0
        native = c2pa.read(data, case['mime'], bundle=c2pa.load_bundle('c2pa'), thumbnails=False)
        replay = {'state': native.state, 'binding_status': native.binding_status, 'covers_bytes': native.covers_bytes,
                  'ai_declared': native.ai_declared, 'failure_codes': native.failure_codes}
        assert replay == row['local']
    assert complete['request_attempts'] == sum(row['attempted'] for row in rows) == 9
    originals = [r['local'] for r in rows if r['id'].startswith('provider_source_')]
    assert all(r['state'] == 'invalid' and r['binding_status'] == 'matched' and not r['covers_bytes']
               and set(r['failure_codes']) == {'signingCredential.expired', 'signingCredential.untrusted'}
               for r in originals)
    copied = next(r['local'] for r in rows if r['id'] == 'transplanted')
    assert copied['binding_status'] == 'mismatch' and not copied['covers_bytes']
    result = {'schema': 'provider-provenance-qualification-result/1', 'plan_sha256': PLAN_SHA256,
        'freeze_commit': '04ba62d2048ee60ce509032da39e9d9c9309c560',
        'counts': {k: dict(v) for k, v in counts.items()}, 'contract_discrepancies': dict(discrepancies),
        'local_content_matching_but_invalid_provider_originals': len(originals),
        'local_copied_credential_hash_mismatch_rejected': True,
        'offline_review': {'frozen_code_commitments': len(plan['code']),
                           'source_commitments': 9, 'request_bodies_reconstructed': 9,
                           'reply_commitments': 9, 'local_sdk_results_replayed': 9, 'all_passed': True,
                           'new_network_requests': 0, 'model_forwards': 0},
        'native_seconds': complete['seconds'], 'customer_files_uploaded': 0,
        'provider_positive_matches_observed': 0, 'genuine_suno_credential_positive_tested': False,
        'current_bria_credential_positive_tested': False,
        'external_serving_admission': False, 'statistical_calibration_changed': False,
        'limitation': 'Small contract study on authored controls and historical public provider files; not generator-wide detection accuracy.'}
    (folder / 'review.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    print(json.dumps(review(parser.parse_args().folder), indent=2))
