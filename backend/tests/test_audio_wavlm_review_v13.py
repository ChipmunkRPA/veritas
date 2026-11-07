"""Independent predictor's hand-calculated and invalid-input boundaries."""
import copy

import numpy as np
import pytest
from scipy.special import expit

from scripts import review_audio_wavlm_v13 as review


def example(arm):
    vector = [2., 3.] + [0.] * 14
    coefficients = [.5, -.25] + [0.] * 14 + [-1.]
    if arm == "complement":
        coefficients = [2.] + coefficients
    fitted = {"arm": arm, "coefficients": coefficients,
              "transform": {"center": [0.] * 16, "scale": [1.] * 16, "variance_factor": 1.}}
    return {"raw": .75, "wavlm": vector}, fitted


@pytest.mark.parametrize("arm", ["standalone", "complement"])
def test_known_log_odds_and_source_score_independence(monkeypatch, arm):
    monkeypatch.setattr(review.features, "DIMENSIONS", 16)
    row, fitted = example(arm)
    margin = -.75 if arm == "standalone" else 2 * np.log(3) - .75
    assert review.probability(row, fitted) == pytest.approx(float(expit(margin)), abs=1e-14)
    if arm == "standalone":
        assert review.probability({**row, "raw": .01}, fitted) == review.probability(row, fitted)


@pytest.mark.parametrize("fault", ["negative_scale", "missing_coordinate", "nonfinite", "negative_raw_slope", "variance_budget"])
def test_altered_fitted_mapping_is_refused(monkeypatch, fault):
    monkeypatch.setattr(review.features, "DIMENSIONS", 16)
    row, fitted = example("complement")
    fitted = copy.deepcopy(fitted)
    if fault == "negative_scale":
        fitted["transform"]["scale"][0] = -.1
    elif fault == "missing_coordinate":
        row["wavlm"].pop()
    elif fault == "nonfinite":
        fitted["coefficients"][2] = float("nan")
    elif fault == "negative_raw_slope":
        fitted["coefficients"][0] = -1.
    else:
        fitted["transform"]["variance_factor"] = 2.
    with pytest.raises(ValueError):
        review.probability(row, fitted)


def test_unassessed_has_no_probability_or_features():
    assert review.probability({"raw": None, "wavlm": None}, {}) is None
    with pytest.raises(ValueError):
        review.probability({"raw": None, "wavlm": [0.]}, {})
