import numpy as np
import pytest

from scripts import audio_score_fusion_v83 as fusion
from scripts.audio_combinations_v52 import METHODS
from scripts.audio_fusion_evaluation_v84 import choose, evaluate, probability_metrics, tendencies


def rows():
    return [{"label": 0}, {"label": 0}, {"label": 1}, {"label": 1}]


def test_known_probability_losses_and_auc_include_ties_correctly():
    result = probability_metrics(rows(), [0.1, 0.2, 0.8, 0.9], np.ones(4))
    assert result["brier"] == pytest.approx(0.025)
    assert result["weighted_brier"] == pytest.approx(0.025)
    assert result["auroc"] == 1.0
    tied = probability_metrics(rows(), [0.5] * 4, np.ones(4))
    assert tied["auroc"] == 0.5
    assert tied["brier"] == 0.25
    assert sum(row["samples"] for row in tied["bins"]) == 4


def test_missing_score_keeps_coverage_and_neutral_ranking_loss_without_a_prediction():
    p = [None, 0.2, 0.8, None]
    result = probability_metrics(rows(), p, np.ones(4))
    assert result["scored"] == result["unscored"] == 2
    assert result["weighted_score_coverage"] == 0.5
    assert result["weighted_brier_with_unassessed_neutral_loss"] == pytest.approx(0.145)
    assert p == [None, 0.2, 0.8, None]
    assert tendencies(p, 0.8) == ["unassessed", "original", "ai", "unassessed"]
    assert probability_metrics(rows(), [None] * 4, np.ones(4))["brier"] is None


def test_single_truth_stratum_has_loss_but_no_auc_and_invalid_score_is_rejected():
    assert probability_metrics(rows()[:2], [0.1, 0.2], np.ones(2))["auroc"] is None
    for p in ([False, 0.1, 0.8, 0.9], [np.nan, 0.1, 0.8, 0.9]):
        with pytest.raises(ValueError, match="Actual finite"):
            probability_metrics(rows(), p, np.ones(4))


def test_selection_uses_fixed_calibration_loss_margin_without_zero_error_gate():
    artifacts = {name: {"branches": {"all": {"status": "calibrated"}}} for name in ("global", "flatness_context")}
    quality = {"fusion_global": {"weighted_brier_with_unassessed_neutral_loss": 0.2},
               "fusion_flatness_context": {"weighted_brier_with_unassessed_neutral_loss": 0.197}}
    assert choose({"probability_quality": quality}, artifacts) == "global"
    quality["fusion_flatness_context"]["weighted_brier_with_unassessed_neutral_loss"] = 0.19
    assert choose({"probability_quality": quality}, artifacts) == "flatness_context"
    artifacts["flatness_context"]["branches"]["all"]["status"] = "unsupported_calibration"
    assert choose({"probability_quality": quality}, artifacts) == "global"


def test_full_authored_fit_calibration_and_evaluation_keep_all_methods_and_skips():
    def examples(role):
        result = []
        for label in (0, 1):
            for number in range(8):
                p = 0.15 + number / 100 if label == 0 else 0.75 + number / 100
                key = f"{role}-{label}-{number}"
                decision = "ai" if label else "original"
                result.append({"role": role, "label": label, "kind": "stock" if label else "genuine",
                               "base_id": key, "group": key, "status": "ok", "raw": p,
                               "quality_flatness": number / 8, "condition": "clean" if number % 2 else "noise_0",
                               "original_p": p, "candidate_p": p,
                               "candidates": {name: {"p": p} for name in ("standalone", "complement")},
                               "decisions": {name: decision for name in [*METHODS, "v7"]}})
        return result

    artifacts = {arm: fusion.calibrate(fusion.fit(examples("fusion_fit"), arm), examples("confidence_cal"))
                 for arm in fusion.METHOD["arms"]}
    cases = examples("selection")
    missing = cases[0]
    missing.update(status="not_applicable", raw=None, original_p=None, candidate_p=None,
                   candidates={name: {"p": None} for name in ("standalone", "complement")},
                   decisions={name: "unassessed" for name in [*METHODS, "v7"]})
    measured = evaluate(cases, artifacts)
    assert len(measured["totals"]) == 13
    assert all(value["inputs"] == 16 and value["unassessed"] == 1 for value in measured["totals"].values())
    assert all(value["scored"] == 15 and value["unscored"] == 1 for value in measured["probability_quality"].values())
    assert measured["strata"]["kind"]["genuine"]["probability_quality"]["fusion_global"]["auroc"] is None
    assert choose(measured, artifacts) in {"global", "flatness_context"}
