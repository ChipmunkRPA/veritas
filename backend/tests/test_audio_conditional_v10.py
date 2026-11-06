import copy
import json

import numpy as np
import pytest

from scripts import audio_conditional_v10 as method


def fixture():
    rng = np.random.default_rng(40)
    rows = []
    transform = {
        "unit_head": [1.0] + [0.0] * 19,
        "center": [0.0] * 20,
        "directions": np.eye(20)[1:17].tolist(),
        "scale": [1.0] * 16,
        "fit_sources": 160,
    }
    for i in range(160):
        raw = 0.1 if i % 2 == 0 else 0.9
        rows.append(
            {
                "id": str(i),
                "base_id": str(i),
                "group": str(i // 2),
                "role": "fit",
                "generator": "human" if i % 2 == 0 else "fixture-ai",
                "label": i % 2,
                "raw": raw,
                "embedding": rng.normal(size=20).tolist(),
                "quality": {"flatness": float(rng.random()), "floor": float(rng.random())},
            }
        )
    return rows, transform


@pytest.fixture(scope="module")
def fitted():
    rows, transform = fixture()
    return method.fit(rows, transform)


def test_json_prediction_matches_native_and_uses_all_fixed_iterations(fitted):
    _, parity = fitted
    assert set(parity) == set(method.ARMS)
    assert all(r["trees"] == 200 and r["max_absolute_error"] <= 1e-12 for r in parity.values())


def test_labels_generator_ids_and_condition_cannot_change_a_prediction(fitted):
    head, _ = fitted
    rows, _ = fixture()
    first = copy.deepcopy(rows[0])
    second = {
        **first,
        "label": 1,
        "generator": "another",
        "condition": "secret_snr",
        "role": "confirmation",
        "id": "changed",
    }
    for arm in method.ARMS:
        assert method.predict_rows([first], head, arm) == method.predict_rows([second], head, arm)


def test_fitted_json_roundtrip_preserves_all_predictor_outputs(fitted):
    head, _ = fitted
    restored = json.loads(json.dumps(head, allow_nan=False))
    rows, _ = fixture()
    for arm in method.ARMS:
        assert method.predict_rows(rows, head, arm) == method.predict_rows(rows, restored, arm)


def test_unassessed_input_does_not_acquire_a_quality_only_original_verdict(fitted):
    head, _ = fitted
    row = {"raw": None, "embedding": None, "quality": {"flatness": 0.5, "floor": 0.5}}
    for arm in method.ARMS:
        assert method.predict_rows([row], head, arm) == [None]


def test_original_raw_coordinate_stays_monotone(fitted):
    head, _ = fitted
    rows, _ = fixture()
    for arm in method.ELIGIBLE:
        views = [{**rows[0], "raw": float(p)} for p in np.linspace(0.001, 0.999, 100)]
        assert np.min(np.diff(method.predict_rows(views, head, arm))) >= -1e-12


def test_fit_cannot_use_calibration_or_development_sources():
    rows, transform = fixture()
    rows[0]["role"] = "cal"
    with pytest.raises(ValueError, match="Only fitting"):
        method.fit(rows, transform)


def test_quality_missing_or_nonfinite_cannot_be_treated_as_a_normal_signal():
    rows, transform = fixture()
    rows[0]["quality"]["flatness"] = float("nan")
    with pytest.raises(ValueError, match="finite original-waveform"):
        method.inputs(rows[0], transform, "quality")


def test_missing_calibration_variants_remain_worst_case_and_classes_share_groups(fitted):
    head, _ = fitted
    rows, _ = fixture()
    rows = [{**r, "role": "cal"} for r in rows]
    rows.append({**rows[0], "raw": None, "embedding": None})
    guards = method.calibrate(rows, head, "quality")
    assert guards["0"][0]["maximum"] is None
    assert len(guards["0"]) == len(guards["1"]) == 80
    rows[0]["role"] = "development"
    with pytest.raises(ValueError, match="independent calibration"):
        method.calibrate(rows, head, "quality")


def test_quality_nuisance_cannot_advance_and_predeclared_tie_rule_prefers_control():
    metric = {"correct_ai": 100, "correct_original": 100, "false_ai": 0, "false_original": 0, "brier": 0.1}
    summary = {"advance_to_fresh_confirmation": True, "groups": {"all": {"candidate": metric}}}
    result = {arm: copy.deepcopy(summary) for arm in method.ARMS}
    assert method.select(result) == "nonlinear"
    result["quality"]["groups"]["all"]["candidate"]["correct_ai"] += 5
    assert method.select(result) == "quality"
    for arm in method.ELIGIBLE:
        result[arm]["advance_to_fresh_confirmation"] = False
    assert method.select(result) is None
