from copy import deepcopy

import numpy as np
import pytest

from scripts.audio_residual_head_v6 import cv, design, fit, predict, projection
from scripts.review_audio_noise_v4 import fit as original_fit


def panel():
    rng = np.random.default_rng(12)
    head = np.r_[1.0, np.zeros(23)]
    rows = []
    for fold in range(5):
        for label in (0, 1):
            for n in range(6):
                e = rng.normal(size=24)
                e[2] += 1.5 * label
                rows.append(
                    {
                        "base_id": f"{fold}-{label}-{n}",
                        "group": f"g-{fold}-{label}-{n}",
                        "fold": fold,
                        "label": label,
                        "raw": float(0.2 + 0.6 * label + 0.005 * n),
                        "embedding": e.tolist(),
                    }
                )
    return rows, head


def test_projection_removes_original_head_and_keeps_fixed_rank():
    rows, head = panel()
    p = projection(rows, head)
    directions = np.array(p["directions"])
    assert directions.shape == (16, 24)
    np.testing.assert_allclose(directions @ head, 0, atol=1e-12)
    np.testing.assert_allclose(directions @ directions.T, np.eye(16), atol=1e-12)


def test_heldout_representation_does_not_enter_its_own_pca_or_classifier():
    rows, head = panel()
    before, fits_before = cv(rows, head)
    altered = deepcopy(rows)
    for row in altered:
        if row["fold"] == 0:
            row["label"] = 1 - row["label"]
            row["embedding"] = (np.asarray(row["embedding"]) + 100).tolist()
    after, fits_after = cv(altered, head)
    assert [p for p in fits_before if p["fold"] == 0] == [p for p in fits_after if p["fold"] == 0]
    for a, b in zip(before, after, strict=True):
        if a["fold"] == 0:
            assert a["original_p"] == b["original_p"]


def test_original_comparator_uses_the_existing_objective():
    rows, _ = panel()
    a, b = fit(rows), original_fit(rows, ["raw"])
    np.testing.assert_allclose(a["coefficients"], b["slopes"] + [b["intercept"]], atol=1e-6)


def test_source_metadata_does_not_enter_prediction():
    rows, head = panel()
    p = fit(rows, head)
    row = rows[0]
    other = {**row, "label": 1, "dataset": "different", "condition": "noise_0", "generator": "unknown"}
    assert predict(row, p) == predict(other, p)


def test_missing_representation_or_score_cannot_clear_an_original():
    rows, head = panel()
    p = fit(rows, head)
    assert predict({**rows[0], "embedding": None}, p) is None
    assert predict({**rows[0], "raw": None}, p) is None


def test_rank_deficient_representation_is_not_silently_padded():
    rows, head = panel()
    for row in rows:
        row["embedding"] = np.zeros(24).tolist()
    with pytest.raises(ValueError, match="rank"):
        projection(rows, head)


def test_connected_identity_cannot_cross_folds():
    rows, head = panel()
    rows[0]["group"] = rows[-1]["group"]
    with pytest.raises(ValueError, match="crossed development folds"):
        cv(rows, head)


def test_original_logit_slope_cannot_reverse_at_a_fixed_representation():
    rows, head = panel()
    p = fit(rows, head)
    values = [predict({**rows[0], "raw": float(raw)}, p) for raw in np.linspace(0, 1, 30)]
    assert p["coefficients"][0] >= 0 and np.min(np.diff(values)) >= 0


def test_invalid_heldout_representation_is_rejected():
    rows, head = panel()
    p = projection(rows, head)
    with pytest.raises(ValueError, match="Invalid held-out representation"):
        design({**rows[0], "embedding": [float("nan")] * 24}, p)
