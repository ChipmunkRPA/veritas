import numpy as np
import pytest

from scripts.audio_guard_v7 import initial as runtime_initial
from scripts.review_audio_guard_v7 import initial, metric, probability


def test_independent_guard_arithmetic_matches_inclusive_and_missing_cases():
    g = {str(c): [{"maximum": (0.2 if i else None)} for i in range(12)] for c in (0, 1)}
    for p in [None, 0.0, 0.2, 0.8, 1.0, *np.linspace(0.01, 0.99, 99)]:
        assert initial(p, g) == runtime_initial(p, g)


def test_proper_scores_do_not_inflate_repeated_source_weight():
    rows = [{"base_id": "real", "label": 0, "p": 0.1, "decision": "original"} for _ in range(2)]
    rows += [
        {"base_id": "ai", "label": 1, "p": 0.6, "decision": "inconclusive"},
        {"base_id": "unassessed", "label": 1, "p": None, "decision": "inconclusive"},
    ]
    result = metric(rows, "p", "decision")
    assert result["brier"] == pytest.approx(0.085)
    assert result["unassessed_inputs"] == 1
    assert result["decisions"] == {"original": 2, "inconclusive": 2}
    assert result["false_original"] == 0


def test_zero_residual_coefficients_reconstruct_original_only_head():
    transform = {
        "unit_head": [1.0] + [0.0] * 1023,
        "center": [0.0] * 1024,
        "directions": np.eye(16, 1024).tolist(),
        "scale": [1.0] * 16,
    }
    head = {"transform": transform, "coefficients": [1.0] + [0.0] * 17}
    for p in (0.01, 0.2, 0.5, 0.8, 0.99):
        assert probability({"raw": p, "embedding": [0.1] * 1024}, head) == pytest.approx(p)
    assert probability({"raw": None}, head) is None
