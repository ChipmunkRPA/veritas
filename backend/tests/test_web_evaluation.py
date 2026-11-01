"""An abstention or a metadata declaration must not inflate content detection recall."""
from scripts.run_web_evaluation import outcome, summarize


def test_inconclusive_and_failed_samples_stay_in_denominator():
    item = {"id": "sample", "kind": "audio", "label": "ai", "source_group": "speaker1", "source_url": "https://example.com"}
    abstained = outcome(item, {"id": "a", "status": "done", "customer_assessment": {"direction": "inconclusive"}})
    failed = outcome(item, {"id": "b", "status": "failed"})
    flagged = outcome(item, {"id": "c", "status": "done", "customer_assessment": {"direction": "ai"}})
    assert summarize([abstained, failed, flagged])["groups"]["audio/ai"] == {
        "n": 3, "correct_direction": 1, "wrong_direction": 0, "inconclusive": 1, "record_led": 0, "failed": 1}


def test_provenance_is_separate_from_content_detection():
    item = {"id": "sample", "kind": "image", "label": "ai", "source_group": "source1", "source_url": "https://example.com"}
    row = outcome(item, {"id": "a", "status": "done", "customer_assessment": {"direction": "ai", "applies": False}})
    assert row["prediction"] == "record_led"
    assert row["correct_direction"] is False


def test_visual_only_tendency_keeps_scope():
    item = {"id": "sample", "kind": "video", "label": "real", "source_group": "source1", "source_url": "https://example.com"}
    row = outcome(item, {"id": "a", "status": "done", "customer_assessment": {
        "direction": "original", "scope": "visual_content", "confidence_pct": 87.3}})
    assert row["correct_direction"] is True
    assert row["scope"] == "visual_content"
