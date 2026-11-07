"""Metadata exceptions must not hide other credentials in the same file."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCANNER = shutil.which("gitleaks")
CONFIG = Path(__file__).resolve().parents[2] / ".gitleaks.toml"
pytestmark = pytest.mark.skipif(SCANNER is None, reason="Optional publication scanner not installed")
VERIFIED_DIGEST = "2c070ff353c36f85e20919da787a30e1e8e52b626d0e026e044f373378f6d0bf"
METADATA_PATHS = ["backend/scripts/writing_focus_v3.py", "docs/evaluation-panels/writing-v3-focus-freeze.json"]


def scan(tmp_path, relative, text):
    source = tmp_path / "source"
    path = source / relative
    path.parent.mkdir(parents=True)
    path.write_text(text)
    report = tmp_path / "report.json"
    result = subprocess.run([SCANNER, "dir", str(source), "--config", str(CONFIG), "--no-banner",
                             "--report-format", "json", "--report-path", str(report), "--redact=100"],
                            capture_output=True, text=True, check=False)
    assert result.returncode in (0, 1), result.stderr
    findings = json.loads(report.read_text())
    assert bool(findings) == (result.returncode == 1)
    return findings


@pytest.mark.parametrize("relative", METADATA_PATHS)
def test_verified_checksum_is_allowed_but_generic_secret_in_same_file_is_detected(tmp_path, relative):
    synthetic = hashlib.sha256(b"publication regression synthetic credential").hexdigest()
    text = json.dumps({"tokenizer.json": VERIFIED_DIGEST, "api_key": synthetic}, indent=2)
    findings = scan(tmp_path, relative, text)
    assert [f["RuleID"] for f in findings] == ["generic-api-key"]
    assert findings[0]["StartLine"] == 3


@pytest.mark.parametrize("relative", METADATA_PATHS)
def test_other_provider_rules_still_scan_metadata_file(tmp_path, relative):
    synthetic = "AKIA" + "Q7T4M2R6S5V3W2XY"
    findings = scan(tmp_path, relative, json.dumps({"tokenizer.json": VERIFIED_DIGEST, "aws_access_key": synthetic}, indent=2))
    assert any(f["RuleID"] == "aws-access-token" for f in findings)


@pytest.mark.parametrize("relative,field,value", [
    ("unrelated/source.py", "tokenizer.json", VERIFIED_DIGEST),
    (METADATA_PATHS[0], "api_key", VERIFIED_DIGEST),
    (METADATA_PATHS[1], "tokenizer.json", hashlib.sha256(b"unverified asset").hexdigest()),
])
def test_exception_requires_correct_path_field_and_verified_value(tmp_path, relative, field, value):
    findings = scan(tmp_path, relative, json.dumps({field: value}, indent=2))
    assert any(f["RuleID"] == "generic-api-key" for f in findings)


@pytest.mark.parametrize("relative", METADATA_PATHS)
def test_only_verified_metadata_has_no_finding(tmp_path, relative):
    assert scan(tmp_path, relative, json.dumps({"tokenizer.json": VERIFIED_DIGEST}, indent=2)) == []


@pytest.mark.parametrize("relative", ["backend/app/engine/stages/video.py", "docs/runbook.md",
    "frontend/src/demo/steps.ts", "docs/evaluation-panels/mouth-v8-head.json",
    "docs/evaluation-panels/mouth-v9-head.json", "docs/evaluation-panels/audio-v8-source-plan.json"])
def test_existing_exceptions_also_preserve_generic_credential_detection(tmp_path, relative):
    synthetic = hashlib.sha256(b"legacy path synthetic credential").hexdigest()
    findings = scan(tmp_path, relative, json.dumps({"api_key": synthetic}, indent=2))
    assert any(f["RuleID"] == "generic-api-key" for f in findings)
