from copy import deepcopy
from unittest.mock import Mock

import pytest

from app.detectors.text.adapters import agreement
from app.detectors.text.extract import text_hash
from app.engine.boundary import Cancelled, Deadline
from app.engine.detector import Profile
from app.engine.runner import StageContext
from app.engine.stages import common
from app.evidence import document_matched_context, document_scope


def context(deadline):
    return StageContext("test", "detect", Profile.FULL, deadline, None, None, {}, {}, None, Mock())


def source():
    text = "First part. Second part."
    spans = [{"start": 0, "end": 11}, {"start": 12, "end": 24}]
    window = {"focus": spans[0], "spans": spans, "input_sha256": text_hash(text), "score": .8,
              "token_coverage": {"tokens": 5, "token_windows": 1}, "unit_ref": "sentence:0"}
    evidence = {"windows": [window]}
    primary = {"detector_id": document_matched_context.DETECTOR, "status": "ok", "details": {"text_evidence": evidence}}
    return text, window, evidence, {"detectors": [primary]}


def test_review_timeout_preserves_a_parent_budget_and_primary_response(monkeypatch):
    text, window, evidence, detected = source()
    monkeypatch.setattr(agreement, "candidates", lambda *_: [window])
    clock = [0.0]
    deadline = Deadline(45, clock=lambda: clock[0])
    ctx = context(deadline)

    def time_out(optional, *args, **kwargs):
        assert optional.deadline.remaining() == 15
        clock[0] = 16
        optional.check("forced test")

    monkeypatch.setattr(common, "run_detectors", time_out)
    agreement.record(ctx, detected, {"text": text})
    assert window["score"] == .8 and evidence[agreement.KEY]["ranges"][0]["state"] == "unavailable"
    assert ctx.deadline.remaining() == 29


def test_review_does_not_start_when_only_the_finish_reserve_remains(monkeypatch):
    text, window, evidence, detected = source()
    monkeypatch.setattr(agreement, "candidates", lambda *_: [window])
    runner = Mock()
    monkeypatch.setattr(common, "run_detectors", runner)
    agreement.record(context(Deadline(29)), detected, {"text": text})
    assert evidence[agreement.KEY]["ranges"][0]["state"] == "unavailable"
    runner.assert_not_called()


def test_parent_cancellation_still_propagates(monkeypatch):
    deadline = Deadline(60)

    def cancelled(optional, *args, **kwargs):
        deadline.lose_lease()
        optional.check("forced test")

    monkeypatch.setattr(common, "run_detectors", cancelled)
    with pytest.raises(Cancelled):
        agreement._review(context(deadline), "source", [{}])
    assert deadline.lease_lost


def test_invalid_second_response_is_unavailable_and_cannot_be_rebound(monkeypatch):
    text, window, evidence, detected = source()
    monkeypatch.setattr(agreement, "candidates", lambda *_: [window])
    monkeypatch.setattr(agreement, "_review", lambda *_: {
        "sentence:0": {"input_sha256": window["input_sha256"], "response_exact_hex": float(2).hex(), "tokens": 5},
    })
    agreement.record(context(Deadline(60)), detected, {"text": text})
    result = {"signals": detected["detectors"]}
    document = {"coverage": {agreement.KEY: deepcopy(evidence[agreement.KEY])}, "source": {"chars": len(text)}}
    row = document["coverage"][agreement.KEY]["ranges"][0]
    assert row["state"] == "unavailable" and "response_exact_hex" not in row
    document_scope.validate_checkpoint_agreement(document, text, result=result)
    row["input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different source input"):
        document_scope.validate_checkpoint_agreement(document, text)
    with pytest.raises(ValueError, match="differs from the frozen result"):
        document_scope.validate_checkpoint_agreement(document, text, result=result)
