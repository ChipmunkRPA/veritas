"""Compact nonlinear frozen-feature response mappings; research, never serving.

Only fitting sources learn trees. Separate calibration sources set unchanged
class guards. JSON prediction is independent of sklearn's fitted-object loader.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.audio_multisource_v8 import balanced_weights
from scripts.audio_residual_head_v6 import design as residual_design
from scripts.review_audio_noise_v4 import valid

ARMS = ("nonlinear", "quality", "quality_only")
ELIGIBLE = ("nonlinear", "quality")
PARAMETERS = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 4,
    "max_depth": 2,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "max_features": 1.0,
    "max_bins": 255,
    "categorical_features": None,
    "interaction_cst": None,
    "early_stopping": False,
    "validation_fraction": None,
    "warm_start": False,
    "random_state": 461110,
    "class_weight": None,
}
METHOD = {
    "version": "audio-conditional-response/10.0",
    "pretrained_parameters": "All frozen; reuse exact v9 source/feature/fit-only PCA16 artifacts",
    "arms": {
        "nonlinear": "raw clipped logit plus16 frozen head-orthogonal residual coordinates",
        "quality": "same17 inputs plus unchanged v5 waveform flatness/floor coordinates",
        "quality_only": "two quality coordinates only; nuisance diagnostic, never eligible",
    },
    "learner": "Numerical histogram gradient boosting, binary log loss, fixed200 depth2 trees; no tuning or early stopping",
    "parameters": PARAMETERS,
    "weighting": "Unchanged equal truth/family/group/source/assessed-variant weights, rescaled to sum n for tree regularization",
    "targets": "Original integer binary labels; no soft-target fitting",
    "monotonicity": "Raw-logit coordinate nondecreasing, all other coordinates unconstrained; nuisance has no raw coordinate",
    "reference_prior": 0.5,
    "guard": "Unchanged separate-calibration worst nonconformity per canonical group and truth class, retaining unassessed values",
    "stability": "Unchanged six-decimal raw and mapped logit change limit ln2; quality is recomputed on the filtered probe waveform",
    "new_pretrained_checkpoint": False,
    "deployment_eligible": False,
}


def inputs(row, transform, arm):
    if arm not in ARMS:
        raise ValueError("Unknown fixed arm")
    if not valid(row.get("raw")):
        return None
    if arm == "nonlinear":
        return residual_design(row, transform)[:-1]
    q = row["quality"]
    if set(q) != {"flatness", "floor"} or not all(valid(v) for v in q.values()):
        raise ValueError("Complete finite original-waveform quality coordinates required")
    values = np.array([q["flatness"], q["floor"]], np.float64)
    return values if arm == "quality_only" else np.r_[residual_design(row, transform)[:-1], values]


def export_forest(model):
    if model.classes_.tolist() != [0, 1] or model._baseline_prediction.shape != (1, 1):
        raise ValueError("Exact binary response convention required")
    trees = []
    for stage in model._predictors:
        if len(stage) != 1 or stage[0].nodes["is_categorical"].any():
            raise ValueError("Numerical binary trees only")
        nodes = stage[0].nodes
        trees.append(
            {
                "value": nodes["value"].tolist(),
                "feature": nodes["feature_idx"].astype(int).tolist(),
                "threshold": nodes["num_threshold"].tolist(),
                "left": nodes["left"].astype(int).tolist(),
                "right": nodes["right"].astype(int).tolist(),
                "leaf": nodes["is_leaf"].astype(bool).tolist(),
            }
        )
    return {
        "baseline_logit": float(model._baseline_prediction[0, 0]),
        "n_features": model.n_features_in_,
        "max_depth": PARAMETERS["max_depth"],
        "trees": trees,
    }


def forest_predict(matrix, forest):
    x = np.asarray(matrix, np.float64)
    if x.ndim != 2 or x.shape[1] != forest["n_features"] or not np.isfinite(x).all():
        raise ValueError("Complete finite numerical predictor inputs required")
    logits = np.full(len(x), forest["baseline_logit"], np.float64)
    for tree in forest["trees"]:
        arrays = {k: np.asarray(v) for k, v in tree.items()}
        sizes = {len(v) for v in arrays.values()}
        if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
            raise ValueError("Complete tree nodes required")
        index = np.zeros(len(x), dtype=int)
        for _ in range(forest["max_depth"] + 1):
            active = ~arrays["leaf"][index]
            if not active.any():
                break
            rows = np.flatnonzero(active)
            current = index[rows]
            column = arrays["feature"][current]
            left = x[rows, column] <= arrays["threshold"][current]
            index[rows] = np.where(left, arrays["left"][current], arrays["right"][current])
        if not arrays["leaf"][index].all():
            raise ValueError("Tree depth/cycle differs from the fixed numerical learner")
        logits += arrays["value"][index]
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite learned response")
    return expit(logits)


def predict_rows(rows, fitted, arm):
    values = [inputs(row, fitted["transform"], arm) for row in rows]
    assessed = [i for i, value in enumerate(values) if value is not None]
    result = [None] * len(rows)
    if assessed:
        predicted = forest_predict([values[i] for i in assessed], fitted["arms"][arm])
        for index, p in zip(assessed, predicted, strict=True):
            result[index] = float(p)
    return result


def fit(rows, transform, *, parity_rows=None):
    if not rows or any(r["role"] != "fit" for r in rows):
        raise ValueError("Only fitting roles may learn the conditional response")
    selected = [r for r in rows if valid(r.get("raw"))]
    weights = balanced_weights(selected) * len(selected)
    labels = np.array([r["label"] for r in selected], int)
    fitted = {
        "method": METHOD,
        "transform": transform,
        "arms": {},
        "fit_groups_by_class": [len({r["group"] for r in selected if r["label"] == c}) for c in (0, 1)],
    }
    parity = {}
    for arm in ARMS:
        x = np.asarray([inputs(row, transform, arm) for row in selected], np.float64)
        monotonic = [0] * x.shape[1]
        if arm != "quality_only":
            monotonic[0] = 1
        native = HistGradientBoostingClassifier(**PARAMETERS, monotonic_cst=monotonic).fit(
            x, labels, sample_weight=weights
        )
        forest = export_forest(native)
        parity_inputs = [
            inputs(row, transform, arm) for row in (parity_rows if parity_rows is not None else selected)
        ]
        parity_x = np.asarray([value for value in parity_inputs if value is not None], np.float64)
        copied, actual = forest_predict(parity_x, forest), native.predict_proba(parity_x)[:, 1]
        error = float(np.max(np.abs(copied - actual)))
        if error > 1e-12 or native.n_iter_ != PARAMETERS["max_iter"]:
            raise ValueError("JSON/native forest parity or fixed iteration count differs")
        fitted["arms"][arm] = forest
        parity[arm] = {
            "rows": len(parity_x),
            "max_absolute_error": error,
            "trees": len(forest["trees"]),
            "native_parameters": native.get_params(),
        }
    return fitted, parity


def calibrate(rows, fitted, arm):
    if not rows or any(r["role"] != "cal" for r in rows):
        raise ValueError("Only independent calibration roles can set guards")
    pools = {0: defaultdict(list), 1: defaultdict(list)}
    for row, p in zip(rows, predict_rows(rows, fitted, arm), strict=True):
        pools[row["label"]][row["group"]].append(None if p is None else p if row["label"] == 0 else 1 - p)
    result = {
        str(c): [
            {
                "group": group,
                "maximum": None if any(v is None for v in values) else max(values),
                "rows": len(values),
            }
            for group, values in sorted(pools[c].items())
        ]
        for c in (0, 1)
    }
    if min(len(result[str(c)]) for c in (0, 1)) < 9:
        raise ValueError("At least nine independent calibration groups per truth class required")
    return result


def select(summaries):
    passing = [a for a in ELIGIBLE if summaries[a]["advance_to_fresh_confirmation"]]
    if len(passing) < 2:
        return passing[0] if passing else None
    a, b = (summaries[arm]["groups"]["all"]["candidate"] for arm in ELIGIBLE)
    gain = b["correct_ai"] + b["correct_original"] - a["correct_ai"] - a["correct_original"]
    return (
        "quality"
        if gain >= 5
        and b["false_ai"] <= a["false_ai"]
        and b["false_original"] <= a["false_original"]
        and b["brier"] <= a["brier"]
        else "nonlinear"
    )
