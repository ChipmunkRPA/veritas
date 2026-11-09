"""Parallel reference-prior ablation on the same locked study responses."""
from __future__ import annotations

from scripts import audio_common_prior_context_v87 as common
from scripts import audio_fusion_evaluation_v84 as original
from scripts import audio_score_fusion_v83 as base
from scripts.audio_combinations_v52 import changes, summarize
from scripts.audio_fusion_diagnostics_v86 import joint_table

NAME = "fusion_common_reference_context"
CHOICE = {"pool": ["global", "flatness_context", "common_reference_context"],
          "loss": "Same complete-role balanced Brier with neutral unscored loss as primary v84",
          "context_improvement_required_over_global": 0.005,
          "context_tie_preference": "flatness_context",
          "scope": "Parallel three-way calibration comparison, fixed before selection outputs; primary v84 choice remains unchanged"}


def evaluate(rows, original_artifacts, added_artifact):
    result = original.evaluate(rows, original_artifacts)
    predictions = {name: [row["decisions"][name] for row in rows] for name in rows[0]["decisions"]}
    for arm, artifact in original_artifacts.items():
        predictions["fusion_" + arm] = [base.predict(row, artifact)["decision"] for row in rows]
    scored = [common.predict(row, added_artifact) for row in rows]
    decisions, probabilities = [row["decision"] for row in scored], [row["probability_ai"] for row in scored]
    predictions[NAME] = decisions
    weights = base.weights(rows)
    result["totals"][NAME] = summarize(rows, decisions)
    result["changes_from_current"][NAME] = changes(rows, predictions["current"], decisions)
    result["probability_quality"][NAME] = original.probability_metrics(rows, probabilities, weights)
    result["research_operating_points"][NAME] = {str(point): summarize(rows, original.tendencies(probabilities, point))
                                                 for point in original.OPERATING_POINTS}
    for field, strata in result["strata"].items():
        for value, cell in strata.items():
            indices = [index for index, row in enumerate(rows) if row[field] == value]
            subset = [rows[index] for index in indices]
            cell["totals"][NAME] = summarize(subset, [decisions[index] for index in indices])
            cell["probability_quality"][NAME] = original.probability_metrics(subset, [probabilities[index] for index in indices], weights[indices])
    result["joint_condition_family_diagnostics"] = joint_table(rows, predictions)
    control_p = [None if value["probability_ai"] is None else
                 added_artifact["branches"][value["branch"]]["conditional_ai_reference_prior"] for value in scored]
    control_choices = ["unassessed" if p is None else "ai" if p > 0.5 else "original" if p < 0.5 else "inconclusive"
                       for p in control_p]
    result["context_prior_only_control"] = {
        "totals": summarize(rows, control_choices),
        "auroc": original.probability_metrics(rows, control_p, weights)["auroc"],
        "scope": "Nuisance/shortcut diagnostic using branch reference priors alone; not eligible for selection or customer use. Branch prior values and fitted parameters stay private."}
    return result


def choose(result, original_artifacts, added_artifact):
    artifacts = {**original_artifacts, "common_reference_context": added_artifact}
    eligible = [name for name in CHOICE["pool"]
                if any(detail["status"] == "calibrated" for detail in artifacts[name]["branches"].values())]
    if not eligible:
        return None

    def loss(name):
        return result["probability_quality"]["fusion_" + name]["weighted_brier_with_unassessed_neutral_loss"]

    contexts = [name for name in eligible if name != "global"]
    winner = min(contexts, key=lambda name: (loss(name), CHOICE["pool"].index(name))) if contexts else None
    if "global" not in eligible:
        return winner
    return winner if winner is not None and loss(winner) < loss("global") - CHOICE["context_improvement_required_over_global"] else "global"
