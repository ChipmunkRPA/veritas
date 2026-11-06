import copy

import numpy as np
import pytest

from scripts import audio_layer_response_v12 as method


def rows():
    result = []
    for c in (0, 1):
        for i in range(12):
            layers = np.zeros((3, 1024))
            layers[1, 777] = 2 * c - 1
            result.append(
                {
                    "role": "fit",
                    "raw": 0.5,
                    "embedding": [0.0] * 1024,
                    "layer_ids": [8, 16, 24],
                    "layer_embeddings": layers.tolist(),
                    "label": c,
                    "generator": "human" if c == 0 else "ai",
                    "group": f"{c}:{i}",
                    "base_id": f"{c}:{i}",
                }
            )
    return result


@pytest.fixture(scope="module")
def fitted():
    return {arm: method.fit(rows(), arm) for arm in method.ARMS}


def test_fixed_middle_layer_can_supply_signal_absent_from_final_control(fitted):
    a, b = rows()[0], rows()[-1]
    assert method.predict(a, fitted["final_control"]) == method.predict(b, fitted["final_control"])
    assert method.predict(a, fitted["layers"]) < 0.5 < method.predict(b, fitted["layers"])
    assert len(fitted["layers"]["coefficients"]) == 3074


@pytest.mark.parametrize("role", ["cal", "development", "confirmation"])
def test_held_roles_cannot_learn_any_normalization(role):
    values = rows()
    values[0]["role"] = role
    with pytest.raises(ValueError, match="Only fitting"):
        method.fit(values, "layers")


def test_layer_order_cannot_change_and_unassessed_does_not_become_original(fitted):
    a = rows()[0]
    a["layer_ids"] = [24, 16, 8]
    with pytest.raises(ValueError, match="preselected layer order"):
        method.predict(a, fitted["layers"])
    assert method.predict({"raw": None}, fitted["layers"]) is None


def test_truth_and_source_metadata_do_not_change_predictions(fitted):
    a = rows()[0]
    b = {**a, "label": 1, "group": "changed", "generator": "changed", "base_id": "changed"}
    assert method.predict(a, fitted["layers"]) == method.predict(b, fitted["layers"])
    before = copy.deepcopy(fitted)
    method.predict(a, fitted["layers"])
    assert fitted == before


def test_failed_arm_cannot_be_selected_merely_for_more_correct_conclusions():
    summary = {a: {"advance_to_fresh_confirmation": False} for a in method.ARMS}
    assert method.select(summary) is None


def test_calibration_retains_unassessed_variants_and_refuses_development_sources(fitted):
    sources = [{**r, "role": "cal"} for r in rows()]
    sources.append({**sources[0], "raw": None, "embedding": None, "layer_embeddings": None})
    result = method.calibrate(sources, fitted["layers"])
    assert len(result["0"]) == len(result["1"]) == 12
    group = next(r for r in result["0"] if r["group"] == sources[0]["group"])
    assert group == {"group": sources[0]["group"], "maximum": None, "rows": 2}
    sources[0]["role"] = "development"
    with pytest.raises(ValueError, match="Only independent calibration"):
        method.calibrate(sources, fitted["layers"])
