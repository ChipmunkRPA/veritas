from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.writing_token_curvature import compare, conditional_moments, focus_values, target_token_indices


def test_conditional_moments_use_observed_token_and_distribution_variance():
    # Equal conditional distributions survive a constant shift in all logits.
    probabilities = (0.25, 0.75)
    scores = np.array([[0, math.log(3)], [30, 30 + math.log(3)]])
    row1, row2 = conditional_moments(scores, [0, 1])
    logs = tuple(math.log(p) for p in probabilities)
    mean = sum(p * value for p, value in zip(probabilities, logs, strict=True))
    var = sum(p * (value - mean) ** 2 for p, value in zip(probabilities, logs, strict=True))
    assert row1 == pytest.approx((logs[0], mean, var))
    assert row2 == pytest.approx((logs[1], mean, var))
    values = focus_values([row1, row2])
    assert values["curvature"] == pytest.approx((sum(logs) - 2 * mean) / math.sqrt(2 * var))
    assert values["mean_log_probability"] == pytest.approx(sum(logs) / 2)


def test_exact_source_token_coverage_handles_unicode_and_leading_whitespace():
    text = "A. café! B."
    focus = (3, 8)
    # Two vocabulary tokens can have the same character span for a split Unicode value.
    offsets = [(0, 1), (1, 2), (2, 6), (6, 7), (6, 7), (7, 8), (8, 10), (10, 11)]
    assert target_token_indices(text, offsets, focus) == [2, 3, 4, 5]
    with pytest.raises(ValueError, match="complete causal token coverage"):
        target_token_indices(text, [span for span in offsets if span != (6, 7)], focus)
    with pytest.raises(ValueError, match="unrelated source text"):
        target_token_indices(text, [*offsets[:2], (1, 6), *offsets[3:]], focus)


def test_invalid_distributions_and_nonfinite_comparisons_are_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        conditional_moments(np.array([[1, float("inf")]]), [0])
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        conditional_moments(np.array([[1, 2]]), [2])
    with pytest.raises(ValueError, match="valid conditional variance"):
        focus_values([(-0.4, -0.3, 0)])
    with pytest.raises(ValueError, match="finite responses"):
        compare(float("nan"), 0)
