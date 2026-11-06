import copy

import numpy as np
import pytest

from scripts import audio_full_representation_v11 as method


def rows():
    result = []
    for c in (0, 1):
        for i in range(12):
            embedding = np.zeros(1024)
            embedding[777] = 2 * c - 1
            result.append(
                {
                    "role": "fit",
                    "raw": 0.5,
                    "embedding": embedding.tolist(),
                    "label": c,
                    "generator": "human" if c == 0 else "ai",
                    "group": f"{c}:{i}",
                    "base_id": f"{c}:{i}",
                }
            )
    return result


@pytest.fixture(scope="module")
def fitted():
    return method.fit(rows(), [1.0] + [0.0] * 1023)


def test_supervised_feature_beyond_sixteen_is_retained_with_fitting_normalization(fitted):
    sources = rows()
    assert len(fitted["coefficients"]) == 1026
    assert fitted["coefficients"][778] > 0
    assert method.predict(sources[0], fitted) < 0.5 < method.predict(sources[-1], fitted)
    assert fitted["transform"]["center"][777] == pytest.approx(0.0)
    assert fitted["transform"]["scale"][777] == pytest.approx(1.0)
    assert fitted["transform"]["scale"][0] == 1e-6
    assert fitted["transform"]["variance_factor"] == 0.125
    assert fitted["fit_groups_by_class"] == [12, 12]


@pytest.mark.parametrize("role", ["cal", "development", "confirmation"])
def test_held_roles_cannot_learn_normalization_or_coefficients(role):
    sources = rows()
    sources[-1]["role"] = role
    with pytest.raises(ValueError, match="Only fitting"):
        method.fit(sources, [1.0] + [0.0] * 1023)


def test_prediction_ignores_source_labels_and_preserves_unassessed(fitted):
    row = rows()[0]
    changed = {
        **row,
        "label": 1,
        "generator": "altered",
        "group": "altered",
        "base_id": "altered",
        "condition": "altered",
    }
    assert method.predict(row, fitted) == method.predict(changed, fitted)
    assert method.predict({"raw": None, "embedding": None}, fitted) is None
    changed["embedding"] = [0.0] * 1023
    with pytest.raises(ValueError, match="1024-vector"):
        method.predict(changed, fitted)


def test_nonfinite_fitting_feature_is_refused():
    sources = rows()
    sources[0]["embedding"][777] = float("nan")
    with pytest.raises(ValueError, match="Finite full"):
        method.fit(sources, [1.0] + [0.0] * 1023)


def test_independent_guards_keep_unassessed_variants_and_paired_truth_groups(fitted):
    sources = [{**r, "role": "cal", "group": str(i % 12)} for i, r in enumerate(rows())]
    sources.append({**sources[0], "raw": None, "embedding": None})
    guards = method.calibrate(sources, fitted)
    assert len(guards["0"]) == len(guards["1"]) == 12
    assert guards["0"][0] == {"group": "0", "maximum": None, "rows": 2}
    assert guards["1"][0]["maximum"] is not None
    sources[0]["role"] = "development"
    with pytest.raises(ValueError, match="Only independent calibration"):
        method.calibrate(sources, fitted)


def test_prediction_does_not_mutate_a_frozen_fit(fitted):
    before = copy.deepcopy(fitted)
    method.predict(rows()[0], fitted)
    assert fitted == before
