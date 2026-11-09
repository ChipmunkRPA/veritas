import copy

import numpy as np
import pytest

from scripts.audio_score_fusion_v83 import calibrate, fit, flatness, predict, predictors, weights


def authored(role):
    result = []
    for label in (0, 1):
        for number in range(8):
            p = (0.15 + number / 100) if label == 0 else (0.75 + number / 100)
            result.append({"role": role, "status": "ok", "raw": p,
                           "candidates": {"complement": {"p": p + 0.03}}, "quality_flatness": number / 8,
                           "kind": "genuine" if label == 0 else "stock", "label": label,
                           "base_id": f"{role}-{label}-{number}", "group": f"{role}-{label}-{number}"})
    return result


def test_separate_global_and_context_calibrations_predict_authored_sources():
    for arm in ("global", "flatness_context"):
        fitted = fit(authored("fusion_fit"), arm)
        ready = calibrate(fitted, authored("confidence_cal"))
        assert all(value["status"] == "fitted" for value in fitted["branches"].values())
        assert all(value["status"] == "calibrated" for value in ready["branches"].values())
        for row in authored("confirmation"):
            result = predict(row, ready)
            assert result["decision"] == ("ai" if row["label"] else "original")
            changed = {**row, "label": 1 - row["label"], "kind": "unknown", "role": "unknown", "group": "changed"}
            assert predict(changed, ready) == result


def test_fitting_roles_and_calibration_roles_cannot_be_interchanged():
    with pytest.raises(ValueError, match="fusion fitting"):
        fit(authored("confirmation"), "global")
    fitted = fit(authored("fusion_fit"), "global")
    with pytest.raises(ValueError, match="separate confidence"):
        calibrate(fitted, authored("fusion_fit"))
    calibrated = calibrate(fitted, authored("confidence_cal"))
    with pytest.raises(ValueError, match="cannot be refitted"):
        calibrate(calibrated, authored("confidence_cal"))


def test_relabeling_the_role_cannot_hide_fitting_source_or_group_reuse():
    rows = authored("fusion_fit")
    fitted = fit(rows, "global")
    for name in ("group", "base_id"):
        bad = authored("confidence_cal")
        bad[0][name] = rows[0][name]
        with pytest.raises(ValueError, match="must be disjoint"):
            calibrate(fitted, bad)


def test_missing_check_and_unsupported_branch_cannot_supply_original_verdict():
    fitted = fit(authored("fusion_fit"), "global")
    row = authored("confirmation")[0]
    assert predict(row, fitted)["decision"] == "unassessed"
    ready = calibrate(fitted, authored("confidence_cal"))
    assert predict({"status": "not_applicable"}, ready)["probability_ai"] is None
    bad = copy.deepcopy(row)
    bad["candidates"]["complement"]["p"] = None
    with pytest.raises(ValueError, match="Both actual finite"):
        predict(bad, ready)


def test_duplicate_variants_do_not_increase_source_or_class_weight():
    rows = authored("fusion_fit")
    w = weights(rows)
    extended = [*rows, *[copy.deepcopy(rows[0]) for _ in range(10)]]
    more = weights(extended)
    assert more[[0, *range(len(rows), len(extended))]].sum() == pytest.approx(w[0])
    assert more[1:len(rows)] == pytest.approx(w[1:])
    assert sum(value for row, value in zip(extended, more, strict=True) if row["label"] == 0) == pytest.approx(0.5)


def test_flatness_is_measured_from_samples_and_requires_complete_finite_window():
    t = np.arange(64000) / 16000
    tone = 0.1 * np.sin(2 * np.pi * 500 * t)
    noise = np.random.default_rng(8307).normal(0, 0.1, 64000)
    assert flatness(noise) > flatness(tone)
    with pytest.raises(ValueError, match="Exact finite"):
        flatness(noise[:-1])
    noise[0] = np.nan
    with pytest.raises(ValueError, match="Exact finite"):
        flatness(noise)


def test_failed_extraction_and_boolean_scores_are_never_interpreted_as_probabilities():
    with pytest.raises(ValueError, match="Failed extraction"):
        predictors({"status": "failed"})
    row = authored("fusion_fit")[0]
    row["raw"] = False
    with pytest.raises(ValueError, match="Both actual finite"):
        predictors(row)


def test_context_without_its_own_calibration_remains_unassessed():
    fitted = fit(authored("fusion_fit"), "flatness_context")
    low_only = [row for row in authored("confidence_cal") if row["quality_flatness"] <= fitted["routing_cutoff"]]
    ready = calibrate(fitted, low_only)
    assert ready["branches"]["low"]["status"] == "calibrated"
    assert ready["branches"]["high"]["status"] == "unsupported_calibration"
    high = next(row for row in authored("confirmation") if row["quality_flatness"] > ready["routing_cutoff"])
    assert predict(high, ready)["decision"] == "unassessed"
    assert predict(high, ready)["probability_ai"] is None


def test_positive_slope_fusion_and_own_calibration_remain_monotone():
    ready = calibrate(fit(authored("fusion_fit"), "global"), authored("confidence_cal"))
    row = authored("confirmation")[0]
    values = []
    for score in (0.1, 0.3, 0.5, 0.7, 0.9):
        values.append(predict({**row, "raw": score, "candidates": {"complement": {"p": score}}}, ready)["probability_ai"])
    assert values == sorted(values)


def test_measured_context_can_learn_complementary_specialists_without_oracle_labels():
    def cases(role):
        result = []
        for context in (0, 1):
            for label in (0, 1):
                for number in range(8):
                    good = 0.85 if label else 0.15
                    unrelated = 0.85 if number % 2 else 0.15
                    key = f"{role}-{context}-{label}-{number}"
                    result.append({"role": role, "status": "ok", "label": label,
                                   "kind": "genuine" if label == 0 else "stock", "group": key, "base_id": key,
                                   "raw": good if context == 0 else unrelated,
                                   "candidates": {"complement": {"p": unrelated if context == 0 else good}},
                                   "quality_flatness": 0.1 if context == 0 else 0.9})
        return result

    fitted = fit(cases("fusion_fit"), "flatness_context")
    ready = calibrate(fitted, cases("confidence_cal"))
    assert ready["branches"]["low"]["coefficients"][0] > ready["branches"]["low"]["coefficients"][1]
    assert ready["branches"]["high"]["coefficients"][1] > ready["branches"]["high"]["coefficients"][0]
    for row in cases("confirmation"):
        expected = "ai" if row["label"] else "original"
        inference_only = {key: row[key] for key in ("status", "raw", "candidates", "quality_flatness")}
        assert predict(inference_only, ready)["decision"] == expected
