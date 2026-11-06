import gzip
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
