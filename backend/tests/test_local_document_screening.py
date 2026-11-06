from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.detectors.text.extract import from_string
from app.engine import customer_assessment
from app.engine.stages import document_screening as screening
from app.schemas.evidence import SignalV2

TEXT = "I spoke with the planning team this morning. We reviewed the revised calendar and discussed the upcoming deadline. " * 5


def response(raw, artifact):
    return SignalV2(
        id="document:global", detector_id=screening.DETECTOR, detector_version=artifact["detector_version"],
        model_revisions={screening.MODEL: artifact["checkpoint_revision"]}, name="Text authorship check",
        family="text", modality="text", category="model", raw_score=raw,
    ).model_dump(mode="json")


def test_email_calibration_cannot_supply_article_confidence():
    _, artifact = screening.load()
    fallback = {"signals": [response(0.999, artifact)], "caveats": [], "v1": {"verdict": {}}}
    ctx = SimpleNamespace(state={})
    assert screening.apply(ctx, from_string(TEXT), fallback) is fallback
    ctx.state["fuser_pin"] = screening.EMAIL_SHA256
    with pytest.raises(ValueError, match="original calibrated text response and extraction"):
        screening.apply(ctx, from_string(TEXT), fallback)


def test_email_fit_is_one_sided_and_bound_to_the_message_body(monkeypatch):
    _, artifact = screening.load()
    monkeypatch.setattr(screening, "_commit", lambda *args: {"sha256": screening.EMAIL_SHA256})
    doc = from_string(TEXT)
    doc.meta.update(method="email", email={"body_content_type": "text/plain"})
    doc.source_map[0]["quote_role"] = "body"
    ctx = SimpleNamespace(state={})
    for raw, expected in ((0.99999, "ai"), (0.00001, "inconclusive")):
        fallback = {"signals": [response(raw, artifact)], "caveats": [], "v1": {"verdict": {}}}
        calibrated = screening.apply(ctx, doc, fallback)
        assert calibrated["decision"]["fuser_sha256"] == screening.EMAIL_SHA256
        public = customer_assessment.customer_assessment({**calibrated, "kind": "document"})
        assert public["direction"] == expected
        assert (public["confidence_pct"] is None) == (expected == "inconclusive")
        assert (calibrated["decision"]["probability_ai_calibrated"] is None) == (expected == "inconclusive")
        if expected == "ai":
            assert public["interval_pct"][0] < public["confidence_pct"] < public["interval_pct"][1]
            rate = customer_assessment.text_reference_rate(
                "document", calibrated["condition_profile"], calibrated["calibration"],
            )
            # Stored 80% reference counts do not establish the new 60% service rule.
            assert rate is None
        else:
            assert public["headline"] == "Inconclusive: no reliable AI-writing tendency"
    doc.source_map[0]["quote_role"] = "quoted"
    assert screening.apply(ctx, doc, fallback) is fallback
