"""Actual public runtime metadata exceptions must preserve credential detection."""
import hashlib
import json
from pathlib import Path

import pytest
from test_secret_scan_metadata import SCANNER, scan

from scripts.writing_modernbert_v4 import BASELINE_ASSETS, CANDIDATE_ASSETS

PATHS = tuple('docs/evaluation-panels/writing-v4-modernbert-' + suffix + '.json'
              for suffix in ('freeze', 'synthetic-runtime'))
PAIRS = [*(CANDIDATE_ASSETS.items()), *(BASELINE_ASSETS.items())]
PAIRS = [(name, value) for name, value in PAIRS
         if name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')]
RECEIPT = json.loads((Path(__file__).resolve().parents[2] /
    'docs/evaluation-panels/writing-v4-modernbert-synthetic-runtime.json').read_text())
PAIRS += [(name, RECEIPT['installed_sha256'][name]) for name in
          ('transformers.tokenization_utils_tokenizers', 'transformers.tokenization_utils_base', 'tokenizers.tokenizers')]
PAIRS += [('token_ids_sha256', row['token_ids_sha256']) for row in RECEIPT['coverage']]
pytestmark = pytest.mark.skipif(SCANNER is None, reason='Optional publication scanner not installed')


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize(('field', 'value'), PAIRS)
def test_reviewed_metadata_does_not_exempt_a_credential_in_the_same_file(tmp_path, path, field, value):
    body = json.dumps({field: value, 'api_key': hashlib.sha256(b'synthetic unreviewed key').hexdigest()}, indent=2)
    findings = scan(tmp_path, path, body)
    assert [r['RuleID'] for r in findings] == ['generic-api-key']
    assert findings[0]['StartLine'] == 3


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize(('field', 'value'), PAIRS)
@pytest.mark.parametrize('fault', ['path', 'field', 'value'])
def test_unreviewed_path_field_or_value_is_still_detected(tmp_path, path, field, value, fault):
    if fault == 'path':
        path = 'docs/evaluation-panels/unreviewed-writing-freeze.json'
    elif fault == 'field':
        field = 'api_key'
    else:
        value = hashlib.sha256(b'unreviewed metadata').hexdigest()
    assert any(r['RuleID'] == 'generic-api-key' for r in scan(tmp_path, path, json.dumps({field: value})))


@pytest.mark.parametrize('path', PATHS)
def test_provider_rule_still_detects_a_key_in_either_public_receipt(tmp_path, path):
    text = json.dumps({'token_ids_sha256': PAIRS[-1][1], 'aws_access_key': 'AKIA' + 'Q7T4M2R6S5V3W2XY'})
    assert any(r['RuleID'] == 'aws-access-token' for r in scan(tmp_path, path, text))
