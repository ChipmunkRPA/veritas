"""Fixed score-quality, coverage and selection arithmetic; no model extraction."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from scripts import audio_score_fusion_v83 as fusion
from scripts.audio_combinations_v52 import changes, summarize

OPERATING_POINTS = (0.6, 0.8, 0.9)
SELECTION = {
    "primary": "Balanced truth/family/group/source/variant Brier; unavailable score contributes neutral 0.25 loss only to ranking, never a prediction",
    "context_minimum_improvement": 0.005,
    "tie_preference": "global",
    "eligibility": "At least one independently calibrated branch; no accuracy/no-loss/zero-error gate",
    "confirmation": "Publish fixed selection before any final-role neural forward; report all predeclared methods without retuning",
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def probability_metrics(rows, probabilities, weights):
    require(len(rows) == len(probabilities) == len(weights) and bool(rows), "Complete score/weight alignment required")
    y = np.asarray([row["label"] for row in rows], np.float64)
    require(all(type(row["label"]) is int and row["label"] in {0, 1} for row in rows), "Known integer truth labels required")
    w = np.asarray(weights, np.float64)
    require(np.isfinite(w).all() and bool((w >= 0).all()) and w.sum() > 0, "Finite nonnegative metric weights required")
    w = w / w.sum()
    valid = np.asarray([p is not None for p in probabilities])
    require(all(p is None or (isinstance(p, (int, float)) and not isinstance(p, bool)
                             and np.isfinite(p) and 0 <= p <= 1) for p in probabilities), "Actual finite scores or explicit None required")
    # Neutral loss is solely a ranking reference. Missing outputs remain None.
    losses = np.full(len(rows), 0.25)
    result = {"scored": int(valid.sum()), "unscored": int((~valid).sum()),
              "weighted_score_coverage": float(w[valid].sum()), "bins": []}
    if valid.any():
        p = np.asarray([value for value in probabilities if value is not None], np.float64)
        actual, scored_w = y[valid], w[valid] / w[valid].sum()
        losses[valid] = (p - actual) ** 2
        clipped = np.clip(p, 1e-12, 1 - 1e-12)
        logs = -(actual * np.log(clipped) + (1 - actual) * np.log1p(-clipped))
        result.update(brier=float(np.mean((p - actual) ** 2)), weighted_brier=float(scored_w @ (p - actual) ** 2),
                      log_loss=float(logs.mean()), weighted_log_loss=float(scored_w @ logs))
        counts = [int((actual == label).sum()) for label in (0, 1)]
        result["auroc"] = (float((rankdata(p)[actual == 1].sum() - counts[1] * (counts[1] + 1) / 2)
                                 / (counts[0] * counts[1])) if min(counts) else None)
        assigned = np.minimum((p * 10).astype(int), 9)
        ece = 0.0
        for index in range(10):
            chosen = assigned == index
            mass = float(scored_w[chosen].sum())
            mean_score = float(np.mean(p[chosen])) if chosen.any() else None
            ai_fraction = float(np.mean(actual[chosen])) if chosen.any() else None
            weighted_mean = float(scored_w[chosen] @ p[chosen] / mass) if mass > 0 else None
            weighted_fraction = float(scored_w[chosen] @ actual[chosen] / mass) if mass > 0 else None
            if mass > 0:
                ece += mass * abs(weighted_mean - weighted_fraction)
            result["bins"].append({"lower": index / 10, "upper": (index + 1) / 10, "samples": int(chosen.sum()),
                                   "mean_score": mean_score, "ai_fraction": ai_fraction,
                                   "scored_weight_fraction": mass, "weighted_mean_score": weighted_mean,
                                   "weighted_ai_fraction": weighted_fraction})
        result["weighted_ece_10_equal_width_bins"] = ece
    else:
        result.update(brier=None, weighted_brier=None, log_loss=None, weighted_log_loss=None, auroc=None,
                      weighted_ece_10_equal_width_bins=None)
    result["weighted_brier_with_unassessed_neutral_loss"] = float(w @ losses)
    return result


def tendencies(probabilities, point):
    require(0.5 < point < 1, "Fixed symmetric research operating point required")
    return ["unassessed" if p is None else "ai" if p >= point else "original" if 1 - p >= point
            else "inconclusive" for p in probabilities]


def evaluate(rows, artifacts):
    require(bool(rows) and len({row["role"] for row in rows}) == 1
            and rows[0]["role"] in {"selection", "confirmation"}, "Assigned selection or confirmation role required")
    names = list(rows[0]["decisions"])
    require(all(set(row["decisions"]) == set(names) for row in rows), "All frozen baseline decisions required")
    predictions = {name: [row["decisions"][name] for row in rows] for name in names}
    probabilities = {
        "current": [row["original_p"] for row in rows],
        "v7": [row["candidate_p"] for row in rows],
        "new_features_only": [row["candidates"]["standalone"]["p"] for row in rows],
        "features_plus_current": [row["candidates"]["complement"]["p"] for row in rows],
    }
    for arm, artifact in artifacts.items():
        results = [fusion.predict(row, artifact) for row in rows]
        name = "fusion_" + arm
        predictions[name] = [value["decision"] for value in results]
        probabilities[name] = [value["probability_ai"] for value in results]
    weights = fusion.weights(rows)
    result = {"totals": {name: summarize(rows, values) for name, values in predictions.items()},
              "changes_from_current": {name: changes(rows, predictions["current"], values) for name, values in predictions.items()},
              "probability_quality": {name: probability_metrics(rows, values, weights) for name, values in probabilities.items()},
              "research_operating_points": {name: {str(point): summarize(rows, tendencies(values, point)) for point in OPERATING_POINTS}
                                            for name, values in probabilities.items()}, "strata": {}}
    for field in ("condition", "kind"):
        result["strata"][field] = {}
        for value in sorted({row[field] for row in rows}):
            indices = [index for index, row in enumerate(rows) if row[field] == value]
            subset = [rows[index] for index in indices]
            result["strata"][field][value] = {
                "totals": {name: summarize(subset, [values[index] for index in indices]) for name, values in predictions.items()},
                "probability_quality": {name: probability_metrics(subset, [values[index] for index in indices], weights[indices])
                                        for name, values in probabilities.items()}}
    return result


def choose(evaluation, artifacts):
    eligible = [arm for arm, artifact in artifacts.items()
                if any(branch["status"] == "calibrated" for branch in artifact["branches"].values())]
    if not eligible:
        return None
    if len(eligible) == 1:
        return eligible[0]
    scores = evaluation["probability_quality"]
    global_loss = scores["fusion_global"]["weighted_brier_with_unassessed_neutral_loss"]
    context_loss = scores["fusion_flatness_context"]["weighted_brier_with_unassessed_neutral_loss"]
    return "flatness_context" if context_loss < global_loss - SELECTION["context_minimum_improvement"] else "global"
