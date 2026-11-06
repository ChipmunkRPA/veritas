import gzip
import hashlib
import json
from pathlib import Path

import pytest

from scripts.review_mouth_learning_v8 import review

ROOT = Path(__file__).resolve().parents[2]
RESPONSES = ROOT / "docs/evaluation-panels/mouth-v8-review-responses.json.gz"
SUMMARY = ROOT / "docs/mouth-learning-v8-evaluation.json"
HEAD = ROOT / "docs/evaluation-panels/mouth-v8-head.json"


def test_public_window_arithmetic_reproduces_complete_published_outcomes():
    result = review(RESPONSES, SUMMARY, HEAD)
    assert result["outcomes_checked"] == 176
    assert result["counts"]["probe"]["localized_ai"] == 14
    assert result["counts"]["probe"]["non_ai_flags"] == 0
    assert "Does not verify model inference" in result["scope"]


def test_changed_retained_frame_margin_is_detected(tmp_path):
    data = json.loads(gzip.decompress(RESPONSES.read_bytes()))
    row = data["cases"][0]
    index = row["probe_windows"][0]["centers"][2]
    next(r for r in row["model_inputs"] if r["frame_index"] == index)["margin"] += 0.5
    corrupted = tmp_path / "changed.json.gz"
    corrupted.write_bytes(gzip.compress(json.dumps(data).encode()))
    with pytest.raises(ValueError, match="arithmetic"):
        review(corrupted, SUMMARY, HEAD)


def test_dropped_or_repeated_source_case_cannot_shrink_denominator(tmp_path):
    data = json.loads(gzip.decompress(RESPONSES.read_bytes()))
    data["cases"][-1] = data["cases"][0]
    corrupted = tmp_path / "repeated.json.gz"
    corrupted.write_bytes(gzip.compress(json.dumps(data).encode()))
    with pytest.raises(ValueError, match="88 cases"):
        review(corrupted, SUMMARY, HEAD)


def test_exact_point_response_uses_same_inclusive_rule_as_detector(tmp_path):
    head = tmp_path / "head.json"
    head.write_text(json.dumps({"review_point": 0.0}))
    cases, evaluated, baseline = [], [], []
    centers = [1, 4, 7, 10, 13, 16, 19]
    window = {
        "track_id": "t0",
        "centers": centers,
        "response": 0.0,
        "before_margin": 0.0,
        "after_margin": 0.0,
        "start_s": 0.28,
        "end_s": 0.52,
    }
    for i in range(88):
        label = int(i < 32)
        identity = str(i)
        cases.append(
            {
                "id": identity,
                "group": str(i % 8),
                "label": label,
                "base_windows": [],
                "probe_windows": [window],
                "edit_time_range_s": [0.28, 0.52],
                "model_inputs": [
                    {"frame_index": k, "time_s": k / 25, "margin": 0.0}
                    for k in sorted({c + d for c in centers for d in (-1, 0, 1)})
                ],
            }
        )
        evaluated.append(
            {
                "id": identity,
                "label": label,
                "assessed": True,
                "flagged": True,
                "localized": bool(label),
                "temporal_iou": 1.0,
            }
        )
        baseline.append(
            {
                "id": identity,
                "label": label,
                "assessed": False,
                "flagged": False,
                "localized": False,
                "temporal_iou": 0.0,
            }
        )
    payload = {
        "original_receipt_sha256": "receipt",
        "header": {"head_sha256": hashlib.sha256(head.read_bytes()).hexdigest(), "review_point": 0.0},
        "current_review_point": 1.0,
        "cases": cases,
    }
    responses = tmp_path / "responses.json.gz"
    responses.write_bytes(gzip.compress(json.dumps(payload).encode()))
    common = {"planned": 88, "ai_edits": 32, "non_ai_controls": 56}
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "score_receipt_sha256": "receipt",
                "cases": {"baseline": baseline, "candidate": evaluated},
                "baseline": {
                    **common,
                    "assessed": 0,
                    "localized_ai": 0,
                    "non_ai_flags": 0,
                    "ai_without_flag": 32,
                },
                "candidate": {
                    **common,
                    "assessed": 88,
                    "localized_ai": 32,
                    "non_ai_flags": 56,
                    "ai_without_flag": 0,
                },
            }
        )
    )
    assert review(responses, summary, head)["counts"]["probe"]["non_ai_flags"] == 56
