import copy
import gzip
import json
from pathlib import Path

import pytest

from scripts.review_mouth_context_v10 import reconstruct, review

DOCS = Path(__file__).resolve().parents[2] / "docs"


def first_head():
    data = json.loads(
        gzip.decompress((DOCS / "evaluation-panels/mouth-v10-development-responses.json.gz").read_bytes())
    )
    return copy.deepcopy(data["cases"][0]["heads"]["v9"])


def test_every_context_window_and_ordinary_point_is_publicly_reconstructible():
    result = review(DOCS)
    assert result["passed"]
    assert result["calibration_cases"] == 24
    assert result["development_cases"] == 88
    assert result["all_eligible_context_windows_checked"] > 112 * 2 * 12
    assert "Does not replay neural inference" in result["scope"]


def test_omitting_unhighlighted_window_is_detected_even_if_top12_unchanged():
    head = first_head()
    assert len(head["context_windows_all"]) > 12
    head["context_windows_all"].pop()
    with pytest.raises(ValueError, match="omitted"):
        reconstruct(head)


def test_changed_bound_raw_frame_margin_invalidates_window_arithmetic():
    head = first_head()
    i = head["probe_windows"][0]["centers"][2]
    next(r for r in head["model_inputs"] if r["frame_index"] == i)["margin"] += 0.3
    with pytest.raises(ValueError, match="arithmetic"):
        reconstruct(head)
