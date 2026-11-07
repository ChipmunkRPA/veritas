"""Exact v4 public digests must not exempt their files or other credentials."""
import hashlib
import json

import pytest
from test_secret_scan_metadata import SCANNER, scan

from scripts.writing_modernbert_v4 import CANDIDATE_ASSETS

PATH = 'backend/scripts/writing_modernbert_v4.py'
FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')
pytestmark = pytest.mark.skipif(SCANNER is None, reason='Optional publication scanner not installed')


@pytest.mark.parametrize('name', FILES)
def test_only_verified_field_value_is_allowed_with_real_generic_detection_preserved(tmp_path, name):
    text = json.dumps({name: CANDIDATE_ASSETS[name], 'api_key': hashlib.sha256(b'v4 synthetic key').hexdigest()}, indent=2)
    findings = scan(tmp_path, PATH, text)
    assert [r['RuleID'] for r in findings] == ['generic-api-key']
    assert findings[0]['StartLine'] == 3


@pytest.mark.parametrize('name', FILES)
@pytest.mark.parametrize('fault', ['path', 'field', 'value'])
def test_wrong_path_field_or_value_remains_flagged(tmp_path, name, fault):
    path, field, value = PATH, name, CANDIDATE_ASSETS[name]
    if fault == 'path':
        path = 'unrelated.py'
    elif fault == 'field':
        field = 'api_key'
    else:
        value = hashlib.sha256(b'not a verified public asset').hexdigest()
    assert any(r['RuleID'] == 'generic-api-key' for r in scan(tmp_path, path, json.dumps({field: value})))


def test_provider_key_rule_still_scans_the_same_metadata_source(tmp_path):
    text = json.dumps({'tokenizer.json': CANDIDATE_ASSETS['tokenizer.json'],
                       'aws_access_key': 'AKIA' + 'Q7T4M2R6S5V3W2XY'})
    assert any(r['RuleID'] == 'aws-access-token' for r in scan(tmp_path, PATH, text))
