import hashlib
import json

import pytest

from scripts import summarize_rule_validation as summary
from scripts.summarize_rule_validation import rates


def test_errors_abstentions_and_failures_keep_the_planned_denominators():
    rows = [{"label": label, "prediction": outcome, "group": str(i)} for i,(label,outcome) in enumerate([
        (0,"ai"), (0,"original"), (0,"inconclusive"), (0,"failed"),
        (1,"ai"), (1,"original"), (1,"inconclusive"), (1,"failed"), (1,"record"),
    ])]
    result = rates(rows)
    assert result["false_alarms"]["count"] == 1 and result["false_alarms"]["denominator"] == 4
    assert result["false_original"]["count"] == 1 and result["false_original"]["denominator"] == 5
    assert result["missed_ai"]["count"] == 4 and result["missed_ai"]["denominator"] == 5
    assert result["inconclusive"]["count"] == 2 and result["inconclusive"]["denominator"] == 9
    assert result["failures"]["count"] == 2 and result["provenance_led"]["count"] == 1
    assert result["false_alarms"]["wilson95"][0] < .25 < result["false_alarms"]["wilson95"][1]


def test_report_rejects_tampered_published_bytes_and_unsupported_logged_verdict(tmp_path, monkeypatch):
    from app.engine.customer_assessment import customer_assessment

    monkeypatch.setattr(summary, "TRIAL", tmp_path)
    result = {"decision": {"code": "inconclusive"}}
    content = json.dumps(result).encode()
    sha = hashlib.sha256(content).hexdigest()
    path = tmp_path/"runs-document/objects/sha256"/sha[:2]/sha[2:4]/sha
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    row = {"status": "done", "result_sha256": sha, "decision_code": "inconclusive",
           "assessment": customer_assessment({**result, "status": "done", "kind": "document"}),
           "prediction": "inconclusive"}
    summary.verify_published_result("document", row)
    with pytest.raises(ValueError, match="prediction differs"):
        summary.verify_published_result("document", {**row, "prediction": "original"})
    path.write_bytes(content+b" ")
    with pytest.raises(ValueError, match="bytes do not match"):
        summary.verify_published_result("document", row)


def test_provenance_alert_is_separate_from_the_frozen_statistical_rule():
    rows = [{"label": 1, "group": "source", "prediction": "record", "decision_code": "ai_supported"}]
    result = rates(rows)
    assert result["missed_ai"]["count"] == 1
    assert result["ai_provenance_declarations"]["count"] == 1
    assert result["ai_without_any_ai_alert"]["count"] == 0
