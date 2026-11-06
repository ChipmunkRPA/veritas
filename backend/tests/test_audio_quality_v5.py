from copy import deepcopy

import numpy as np
import pytest

from scripts.audio_quality_v5 import cv, design, features, fit, predict
from scripts.review_audio_noise_v4 import fit as original_fit


def signal(samples=64000):
    t = np.arange(samples) / 16000
    return (0.1 * np.sin(2 * np.pi * 200 * t) + 0.005 * np.random.default_rng(8).normal(size=samples)).astype(
        np.float32
    )


def test_descriptors_keep_the_waveform_and_are_gain_invariant():
    pcm = signal()
    saved = pcm.copy()
    a, b = features(pcm), features(pcm * np.float32(0.05))
    np.testing.assert_array_equal(pcm, saved)
    for key in a:
        assert 0 <= a[key] <= 1
        assert a[key] == pytest.approx(b[key], abs=1e-8)


@pytest.mark.parametrize(
    "pcm",
    [
        np.zeros(64000, np.float32),
        signal(39999),
        signal(64001),
        np.full(64000, np.nan, np.float32),
        np.ones((64000, 2), np.float32),
        np.ones(64000, np.float64),
    ],
)
def test_invalid_or_silent_audio_cannot_receive_quality_features(pcm):
    with pytest.raises(ValueError):
        features(pcm)


def rows():
    return [
        {
            "base_id": f"{fold}-{label}-{n}",
            "group": f"voice-{fold}-{label}-{n}",
            "fold": fold,
            "label": label,
            "raw": 0.15 + 0.68 * label + 0.01 * n,
            "quality": {"flatness": 0.1 + 0.1 * n, "floor": 0.8 - 0.15 * n},
        }
        for fold in range(5)
        for label in (0, 1)
        for n in range(3)
    ]


def test_candidate_uses_no_label_source_or_condition_to_predict():
    row = rows()[0]
    x = design(row, "quality")
    other = {
        **row,
        "label": 1,
        "base_id": "another",
        "condition": "clean",
        "generator": "unknown",
        "fold": 99,
    }
    np.testing.assert_array_equal(design(other, "quality"), x)


def test_fitted_mapping_is_monotone_across_the_quality_box():
    pars = fit(rows(), "quality")
    for a in np.linspace(0, 1, 7):
        for b in np.linspace(0, 1, 7):
            p = [
                predict({"raw": float(raw), "quality": {"flatness": float(a), "floor": float(b)}}, pars)
                for raw in np.linspace(0, 1, 20)
            ]
            assert np.min(np.diff(p)) >= -1e-8


def test_refit_original_comparator_matches_the_existing_objective():
    data = rows()
    old, new = original_fit(data, ["raw"]), fit(data, "original")
    np.testing.assert_allclose(new["coefficients"], old["slopes"] + [old["intercept"]], atol=1e-5)


def test_heldout_labels_never_change_their_own_predictions():
    data = rows()
    before, _ = cv(data)
    altered = deepcopy(data)
    for row in altered:
        if row["fold"] == 0:
            row["label"] = 1 - row["label"]
    after, _ = cv(altered)
    for a, b in zip(before, after, strict=True):
        if a["fold"] == 0:
            assert {k: v for k, v in a.items() if k.endswith("_p")} == {
                k: v for k, v in b.items() if k.endswith("_p")
            }


def test_connected_identity_leakage_is_refused():
    data = rows()
    data[0]["group"] = data[-1]["group"]
    with pytest.raises(ValueError, match="crossed development folds"):
        cv(data)


@pytest.mark.parametrize("raw", [None, True, float("nan")])
def test_missing_original_response_cannot_be_replaced_by_quality(raw):
    for arm in ("original", "quality", "quality_only"):
        assert design({"raw": raw, "quality": {"flatness": 0.5, "floor": 0.5}}, arm) is None


def test_missing_quality_does_not_receive_a_default_probability():
    assert predict({"raw": 0.2, "quality": None}, fit(rows(), "quality")) is None
