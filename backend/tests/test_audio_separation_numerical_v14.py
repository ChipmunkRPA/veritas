"""Independent bag arithmetic, stable extremes and separate normalization use."""
import copy

import pytest

from scripts import audio_separation_response_v14 as method
from scripts.review_audio_separation_v14 import probability, sigmoid


def fitted(arm):
    return {"arm": arm, "method": method.METHOD,
        "transform": {"center": [0.] * 1024, "scale": [1.] * 1024, "variance_factor": .125},
        "coefficients": [1., *[0.] * 1024, 0., *([0., 0.] if arm == "complement" else [])]}


@pytest.mark.parametrize("arm", method.ARMS)
def test_hand_computed_any_source_probability_and_permutation(arm):
    row = {"raw": .5, "stems": [{"raw": .8, "embedding": [0.] * 1024},
                                  {"raw": .1, "embedding": [0.] * 1024}]}
    pars = fitted(arm)
    assert probability(row, pars) == pytest.approx(.82)
    assert probability(row, pars) == pytest.approx(method.predict(row, pars), abs=1e-14)
    row["stems"].reverse()
    assert probability(row, pars) == pytest.approx(.82)
    assert probability({"raw": None, "stems": None}, pars) is None


def test_embedding_standardization_and_intercept_have_correct_slots():
    row = {"raw": .5, "stems": [{"raw": .5, "embedding": [9.] * 1024},
                                  {"raw": .5, "embedding": [9.] * 1024}]}
    pars = fitted("separated")
    pars["transform"]["center"] = [1.] * 1024
    pars["transform"]["scale"] = [2.] * 1024
    pars["coefficients"][0] = 0
    pars["coefficients"][1] = 2.
    pars["coefficients"][1025] = -1.
    # Each source margin is2*(9-1)/2/8-1=0, hence noisy-OR0.75.
    assert probability(row, pars) == pytest.approx(.75)
    assert probability(row, pars) == pytest.approx(method.predict(row, pars), abs=1e-14)


@pytest.mark.parametrize("value,expected", [(-1000, 0.), (1000, 1.), (0., .5)])
def test_stable_sigmoid_extremes(value, expected):
    assert sigmoid(value) == expected


def test_invalid_mapping_or_unassessed_estimate_is_refused():
    row = {"raw": .5, "stems": [{"raw": .5, "embedding": [0.] * 1024}] * 2}
    pars = fitted("complement")
    for bad in ("shape", "scale", "slope"):
        altered = copy.deepcopy(pars)
        if bad == "shape":
            altered["coefficients"].pop()
        elif bad == "scale":
            altered["transform"]["scale"][0] = 0
        else:
            altered["coefficients"][-2] = -1
        with pytest.raises(ValueError):
            probability(row, altered)
    with pytest.raises(ValueError):
        probability({"raw": None, "stems": row["stems"]}, pars)
