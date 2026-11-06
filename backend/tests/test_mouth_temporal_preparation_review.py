import gzip
import json
from pathlib import Path

import pytest

from scripts.review_mouth_temporal_preparation_v11 import review


def capsule():
    path = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/mouth-v11-preparation.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))


def test_complete_public_preparation_is_reviewable_without_models():
    result = review(capsule())
    assert result["passed"]
    assert result["cases"] == 160
    assert result["neural_input_bindings"] == 1280


@pytest.mark.parametrize("change", ["offset", "mask_time", "role", "missing_condition", "seed"])
def test_review_refuses_mismatched_bindings(change):
    body = capsule()
    if change == "offset":
        body["inputs"]["items"][0]["original_audio_mapping"]["timeline"][0]["source_pts"] += 640
    elif change == "mask_time":
        body["inputs"]["items"][6]["masks"][0]["source_time_s"] += 0.04
    elif change == "role":
        body["plan"]["sources"][0]["role"] = "point"
    elif change == "missing_condition":
        del body["inputs"]["items"][5]
    else:
        body["inputs"]["items"][8]["masks"][0]["model_proof"]["seed"] += 1
    with pytest.raises(ValueError):
        review(body)
