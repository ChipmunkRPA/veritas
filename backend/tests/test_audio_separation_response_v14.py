"""Unordered source bags, weak mixture labels and stable analytic gradients."""
import copy

import numpy as np
import pytest
from scipy.special import expit

from scripts import audio_separation_response_v14 as method


def test_any_source_probability_is_not_mean_or_two_copied_ai_labels():
    logits = np.array([[np.log(.8 / .2), np.log(.1 / .9)]])
    value, derivative = method.bag_logit(logits)
    assert expit(value)[0] == pytest.approx(.82)
    assert derivative.shape == (1, 2)
    assert expit(method.bag_logit(logits[:, ::-1])[0])[0] == pytest.approx(.82)


@pytest.mark.parametrize("value", [-1000., 1000.])
def test_extreme_source_logits_remain_finite(value):
    eta, derivative = method.bag_logit(np.array([[value, value]]))
    assert np.isfinite(eta).all() and np.isfinite(derivative).all()


@pytest.mark.parametrize("arm", method.ARMS)
def test_gradient_matches_independent_central_finite_differences(arm):
    rng = np.random.default_rng(17)
    x = rng.normal(size=(7, 2, 4))
    mixture = rng.normal(size=7)
    targets = np.array([.1, .9, .1, .9, .1, .9, .1])
    weights = np.array([.1, .2, .15, .15, .05, .2, .15])
    size = 4 + (2 if arm == "complement" else 0)
    beta, ridge = rng.normal(size=size) / 4, np.full(size, .05)
    _, derivative = method.objective(beta, x, mixture, targets, weights, ridge, arm)
    numerical = []
    for j in range(size):
        delta = np.zeros(size)
        delta[j] = 1e-5
        numerical.append((method.objective(beta + delta, x, mixture, targets, weights, ridge, arm)[0]
                          - method.objective(beta - delta, x, mixture, targets, weights, ridge, arm)[0]) / 2e-5)
    np.testing.assert_allclose(derivative, numerical, atol=1e-9, rtol=1e-8)
    swapped = method.objective(beta, x[:, ::-1], mixture, targets, weights, ridge, arm)
    assert swapped[0] == pytest.approx(method.objective(beta, x, mixture, targets, weights, ridge, arm)[0])
    np.testing.assert_allclose(swapped[1], derivative, atol=1e-12)


def source_row():
    return {"raw": .6, "stems": [{"raw": .8, "embedding": [0.] * 1024},
                                  {"raw": .1, "embedding": [1.] * 1024}]}


def test_missing_incomplete_or_invalid_stems_cannot_be_scored_as_original():
    assert method.features({"raw": None, "stems": None}) is None
    bad = [source_row(), source_row(), source_row(), source_row()]
    bad[0]["stems"].pop()
    bad[1]["stems"][0]["embedding"][0] = float("nan")
    bad[2]["stems"][1]["raw"] = True
    bad[3]["raw"] = None
    for row in bad:
        with pytest.raises(ValueError):
            method.features(row)


def test_predictor_is_symmetric_and_does_not_read_source_metadata():
    row = source_row()
    fitted = {"arm": "separated", "method": method.METHOD, "transform": {"center": [0.] * 1024, "scale": [1.] * 1024,
               "variance_factor": .125}, "coefficients": [1., *[0.] * 1024, 0.]}
    assert method.predict(row, fitted) == pytest.approx(.82)
    altered = copy.deepcopy(row)
    altered["stems"].reverse()
    altered.update(condition="pretend_clean", generator="anything", label=0, group="unrelated")
    assert method.predict(altered, fitted) == pytest.approx(method.predict(row, fitted), abs=1e-15)


def test_non_fitting_roles_refused_before_learning():
    with pytest.raises(ValueError):
        method.fit([{**source_row(), "role": "development"}], "separated")
    with pytest.raises(ValueError):
        method.calibrate([{**source_row(), "role": "fit", "label": 1}], {})


@pytest.mark.parametrize("arm", method.ARMS)
def test_synthetic_bag_fit_and_separate_group_calibration_keep_missing_groups(arm):
    rows = []
    for label in (0, 1):
        for group in range(2):
            for variant in range(2):
                pattern = np.linspace(-.02, .02, 1024) + .005 * (group + variant)
                rows.append({"role": "fit", "label": label, "generator": str(label),
                    "group": f"fit-{label}-{group}", "base_id": f"source-{label}-{group}",
                    "raw": .8 if label else .2,
                    "stems": [{"raw": .9 if label else .1, "embedding": (pattern + label).tolist()},
                              {"raw": .1, "embedding": pattern.tolist()}]})
    fitted = method.fit(rows, arm)
    assert fitted["fit_groups_by_class"] == [2, 2]
    assert len(fitted["coefficients"]) == (1026 if arm == "separated" else 1028)
    assert fitted["coefficients"][0] >= 0
    assert method.predict(rows[-1], fitted) > method.predict(rows[0], fitted)
    calibration = []
    for label in (0, 1):
        for group in range(9):
            row = copy.deepcopy(rows[4 if label else 0])
            row.update(role="cal", group=f"cal-{label}-{group}")
            calibration.append(row)
    calibration[0].update(raw=None, stems=None)
    pools = method.calibrate(calibration, fitted)
    assert len(pools["0"]) == len(pools["1"]) == 9
    assert pools["0"][0]["maximum"] is None


@pytest.mark.parametrize("change", ["scale", "center", "budget", "coefficient_shape", "negative_slope",
                                    "nonfinite_coefficient", "arm", "method"])
def test_invalid_fitted_mapping_does_not_silently_produce_an_inconclusive_probability(change):
    fitted = {"arm": "separated", "method": method.METHOD,
        "transform": {"center": [0.] * 1024, "scale": [1.] * 1024, "variance_factor": .125},
        "coefficients": [1., *[0.] * 1024, 0.]}
    method.validate_fitted(fitted)
    if change == "scale":
        fitted["transform"]["scale"][0] = 0
    elif change == "center":
        fitted["transform"]["center"][0] = float("nan")
    elif change == "budget":
        fitted["transform"]["variance_factor"] = 1.
    elif change == "coefficient_shape":
        fitted["coefficients"].pop()
    elif change == "negative_slope":
        fitted["coefficients"][0] = -1.
    elif change == "nonfinite_coefficient":
        fitted["coefficients"][1] = float("nan")
    elif change == "arm":
        fitted["arm"] = "unknown"
    else:
        fitted["method"] = {}
    with pytest.raises(ValueError):
        method.predict(source_row(), fitted)
