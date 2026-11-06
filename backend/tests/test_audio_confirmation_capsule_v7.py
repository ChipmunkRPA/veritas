import copy
import gzip
import json
from pathlib import Path

import pytest

from scripts.reproduce_audio_confirmation_v7 import review, same_tree


def test_discrete_gates_and_counts_do_not_use_float_or_boolean_coercion():
    assert not same_tree({"passes": True}, {"passes": 1})
    assert not same_tree({"assessed": 1080}, {"assessed": 1080.0})
    assert same_tree({"brier": 0.1}, {"brier": 0.1 + 1e-12})


@pytest.fixture(scope="module")
def published_inputs():
    root = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
    body = json.loads(gzip.decompress((root / "audio-v7-confirmation-inputs.json.gz").read_bytes()))
    return body, json.loads((root / "audio-v7-fresh-sources.json").read_text())


def test_published_preparation_reproduces_without_audio_or_neural_weights(published_inputs):
    body, sources = published_inputs
    result = review(body, sources)
    assert result["passed"] and result["inputs"] == 1080
    assert result["source_groups"] == 110
    assert result["head_artifact_sha256_verified"]
    assert not result["detector_outcomes_reviewed"]


@pytest.mark.parametrize("change", ["head", "label", "missing", "clock", "background"])
def test_public_reviewer_rejects_changed_evidence_bindings(published_inputs, change):
    original, sources = published_inputs
    body = copy.deepcopy(original)
    if change == "head":
        body["head"]["fitted"]["coefficients"][0] += 0.05
    elif change == "label":
        body["variants"][0]["label"] = 1 - body["variants"][0]["label"]
    elif change == "missing":
        body["variants"].pop()
    elif change == "clock":
        body["variants"][0]["samples"] = 63999
    elif change == "background":
        row = next(r for r in body["variants"] if r["background_id"] is not None)
        row["background_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        review(body, sources)
