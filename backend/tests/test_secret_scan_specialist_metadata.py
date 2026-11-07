"""Exact source-hash exceptions must preserve scanning within each receipt."""
import hashlib
import json
from pathlib import Path

import pytest
from test_secret_scan_metadata import SCANNER, scan

ROOT = Path(__file__).resolve().parents[2]
PATHS = (
    'docs/evaluation-panels/image-v8-patch-drift-native-freeze.json',
    'docs/evaluation-panels/image-v8-patch-drift-development-freeze.json',
    'docs/evaluation-panels/image-v7-smaug-development-freeze.json',
    'docs/evaluation-panels/video-v16-sface-freeze.json',
    'docs/evaluation-panels/video-v17-sface-freeze.json',
)
pytestmark = pytest.mark.skipif(SCANNER is None, reason='Optional publication scanner not installed')


def pair(path):
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('tokenizers/tokenizers.abi3.so', 'backend/app/api/auth.py'):
                    return key, item
                found = walk(item)
                if found:
                    return found
        if isinstance(value, list):
            for item in value:
                found = walk(item)
                if found:
                    return found
        return None

    return walk(json.loads((ROOT / path).read_text()))


@pytest.mark.parametrize('path', PATHS)
def test_reviewed_hash_and_other_secrets_are_scanned_in_the_same_receipt(tmp_path, path):
    key, value = pair(path)
    body = {key: value, 'api_key': hashlib.sha256(b'authored fake secret for scanning test').hexdigest()}
    findings = scan(tmp_path, path, json.dumps(body, indent=2))
    assert [f['RuleID'] for f in findings] == ['generic-api-key']
    assert findings[0]['StartLine'] == 3


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('fault', ('path', 'field', 'value'))
def test_metadata_exception_requires_exact_path_field_and_verified_value(tmp_path, path, fault):
    key, value = pair(path)
    if fault == 'path':
        path = 'unreviewed/source.json'
    elif fault == 'field':
        key = 'api_key'
    else:
        value = hashlib.sha256(b'unverified code commitment').hexdigest()
    assert any(f['RuleID'] == 'generic-api-key' for f in scan(tmp_path, path, json.dumps({key: value})))
