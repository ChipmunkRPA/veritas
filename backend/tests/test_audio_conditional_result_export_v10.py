import copy
import hashlib
import json

import pytest

from scripts.publish_audio_conditional_results_v10 import comparable, exact_artifact, review


def test_fitted_artifact_check_rejects_a_numerically_different_original():
    original = {"fitted": {"value": 0.125}, "deployment_eligible": False}
    digest = hashlib.sha256((json.dumps(original, indent=2) + "\n").encode()).hexdigest()
    exact_artifact(original, digest)
    changed = copy.deepcopy(original)
    changed["fitted"]["value"] = 0.126
    with pytest.raises(ValueError, match="Exact original"):
        exact_artifact(changed, digest)


def test_summary_comparison_preserves_counts_and_intervals_while_correcting_scope_text():
    original = {
        "nonlinear": {
            "schema": "original",
            "complete_inputs": 1449,
            "paired_differences": {"all": {"interpretation": "original text", "interval": [-0.1, -0.01]}},
        }
    }
    copy_with_scope = copy.deepcopy(original)
    copy_with_scope["nonlinear"]["paired_differences"]["all"]["interpretation"] = "Consumed development"
    assert comparable(original) == comparable(copy_with_scope)
    copy_with_scope["nonlinear"]["complete_inputs"] -= 1
    assert comparable(original) != comparable(copy_with_scope)
    assert original["nonlinear"]["complete_inputs"] == 1449


def test_public_review_refuses_private_nested_paths_before_loading_references(tmp_path):
    with pytest.raises(ValueError, match="Local paths"):
        review({"nested": [{"value": "/Users/example/private.wav"}]}, tmp_path)
