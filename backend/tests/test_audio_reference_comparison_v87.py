import copy

from scripts import audio_common_prior_context_v87 as common
from scripts import audio_score_fusion_v83 as base
from scripts.audio_combinations_v52 import METHODS
from scripts.audio_reference_comparison_v87 import choose, evaluate


def test_parallel_pool_choice_preserves_global_margin_and_existing_context_tie():
    artifact = {"branches": {"all": {"status": "calibrated"}}}
    old = {name: artifact for name in ("global", "flatness_context")}
    values = {name: {"weighted_brier_with_unassessed_neutral_loss": loss} for name, loss in
              (("fusion_global", 0.2), ("fusion_flatness_context", 0.197), ("fusion_common_reference_context", 0.196))}
    result = {"probability_quality": values}
    assert choose(result, old, artifact) == "global"
    values["fusion_common_reference_context"]["weighted_brier_with_unassessed_neutral_loss"] = 0.18
    assert choose(result, old, artifact) == "common_reference_context"
    values["fusion_flatness_context"]["weighted_brier_with_unassessed_neutral_loss"] = 0.18
    assert choose(result, old, artifact) == "flatness_context"


def test_no_supported_branch_cannot_win_the_parallel_choice():
    unsupported = {"branches": {"all": {"status": "unsupported_calibration"}}}
    old = {name: unsupported for name in ("global", "flatness_context")}
    assert choose({"probability_quality": {}}, old, unsupported) is None


def test_full_parallel_report_preserves_primary_counts_and_exposes_prior_only_shortcut():
    def rows(role):
        result = []
        for context, counts in ((0.1, (9, 3)), (0.9, (3, 9))):
            for label, count in enumerate(counts):
                for index in range(count):
                    key = f"{role}-{context}-{label}-{index}"
                    result.append({"role": role, "status": "ok", "raw": 0.5, "original_p": 0.5, "candidate_p": 0.5,
                                   "candidates": {name: {"p": 0.5} for name in ("standalone", "complement")},
                                   "quality_flatness": context, "label": label, "kind": "genuine" if label == 0 else "stock",
                                   "base_id": key, "group": key, "condition": "noise_0",
                                   "decisions": {name: "inconclusive" for name in [*METHODS, "v7"]}})
        return result

    originals = {arm: base.calibrate(base.fit(rows("fusion_fit"), arm), rows("confidence_cal")) for arm in base.METHOD["arms"]}
    snapshot = copy.deepcopy(originals)
    additional = common.calibrate(originals["flatness_context"], rows("confidence_cal"))
    result = evaluate(rows("selection"), originals, additional)
    assert originals == snapshot
    assert len(result["totals"]) == 14
    assert len(result["joint_condition_family_diagnostics"]["conditions"]) == 9
    assert result["context_prior_only_control"]["totals"]["ai_found"] == result["totals"]["fusion_common_reference_context"]["ai_found"]
    assert result["context_prior_only_control"]["auroc"] == 0.75
    assert choose(result, originals, additional) == "common_reference_context"
