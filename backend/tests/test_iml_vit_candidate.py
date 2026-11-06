from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts import freeze_iml_vit as freeze
from scripts import iml_vit_candidate as candidate


def test_padding_retains_top_left_geometry_and_rgb_normalization():
    rgb = np.zeros((3, 7, 3), dtype=np.uint8)
    rgb[1, 5] = [255, 128, 64]
    actual = candidate.preprocess(rgb).numpy()[0]
    assert actual.shape == (3, 1024, 1024)
    np.testing.assert_allclose(actual[:, 1, 5], (rgb[1, 5] / np.float32(255) - candidate.MEAN) / candidate.STD)
    np.testing.assert_allclose(actual[:, 3, 7], -candidate.MEAN / candidate.STD)
    assert rgb[1, 5].tolist() == [255, 128, 64]


@pytest.mark.parametrize("rgb", [np.zeros((1025, 3, 3), np.uint8), np.zeros((3, 3), np.uint8),
                                np.zeros((3, 3, 3), np.float32), np.zeros((0, 3, 3), np.uint8)])
def test_unsupported_native_inputs_are_not_silently_cropped_or_resized(rgb):
    with pytest.raises(ValueError):
        candidate.preprocess(rgb)


def test_selected_map_is_unpadded_on_the_original_coordinate_grid():
    class Model:
        def __call__(self, pixels):
            result = torch.zeros((1, 1, 1024, 1024))
            result[0, 0, 1, 5] = .75
            result[0, 0, 900, 900] = 1  # Prediction in padding must never become source evidence.
            return result
    result = candidate.predict(Model(), np.zeros((3, 7, 3), np.uint8))
    assert result.shape == (3, 7) and result[1, 5] == .75 and np.count_nonzero(result) == 1


def development_rows():
    return [{"source_id": str(i), "label": label, "condition": condition, "status": "ok",
             "active_fraction": (i + 1) / 1000 if not label else .1,
             "ungated": {"predicted_positive_pixels": 100, "intersection": 5 if label else 0, "iou": .2}}
            for i in range(24) for label in (0, 1) for condition in ("native", "jpeg75")]


def test_gate_is_fitted_only_to_originals_and_ties_do_not_exceed_budget():
    rows = development_rows()
    rules = freeze.fit(rows)
    assert rules["native"]["false_alert_budget"] == rules["native"]["originals_with_regions"] == 1
    assert .023 < rules["native"]["minimum_active_fraction"] < .024
    assert rules["native"]["edits_with_overlap"] == 24
    for row in rows:
        if row["label"]:
            row["active_fraction"] = .0001
    changed = freeze.fit(rows)
    assert changed["native"]["minimum_active_fraction"] == rules["native"]["minimum_active_fraction"]
    assert changed["native"]["edits_with_overlap"] == 0


def test_incomplete_development_cannot_produce_a_gate():
    rows = development_rows()
    rows[0]["status"] = "failed"
    with pytest.raises(ValueError, match="coverage"):
        freeze.fit(rows)
