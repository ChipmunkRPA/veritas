"""Source-grouped development comparison, never a deployment calibration."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

RIDGE = 0.001
BOOTSTRAPS = 2000
SEED = 461109
ARMS = {"original": ["raw"], "enhanced": ["enhanced_raw"], "combined": ["raw", "enhanced_raw"]}
CONDITIONS = ["all", "clean", "noise_0", "noise_5", "noise_15", "music_5", "music_15"]


def valid(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        and 0 <= value <= 1
    )


def logit(values):
    p = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def weights(rows, multiplicities=None):
    counts = Counter(r["base_id"] for r in rows)
    w = np.array([(multiplicities or {}).get(r["group"], 1.0) / counts[r["base_id"]] for r in rows])
    y = np.array([r["label"] for r in rows])
    for label in (0, 1):
        mask = y == label
        total = w[mask].sum()
        if total <= 0:
            return None
        w[mask] *= 0.5 / total
    return w


def fit(rows, fields):
    selected = [r for r in rows if all(valid(r.get(key)) for key in fields)]
    if not selected:
        raise ValueError("No assessed fitting rows")
    w = weights(selected)
    if w is None:
        raise ValueError("A fit requires both source classes")
    x = logit(np.array([[r[key] for key in fields] for r in selected]))
    y = np.array([r["label"] for r in selected])
    ns = [len({r["base_id"] for r in selected if r["label"] == c}) for c in (0, 1)]
    target = np.where(y == 1, (ns[1] + 1) / (ns[1] + 2), 1 / (ns[0] + 2))

    def objective(p):
        z = x @ p[:-1] + p[-1]
        residual = w * (expit(z) - target)
        return (
            float(np.sum(w * (np.logaddexp(0, z) - target * z)) + 0.5 * RIDGE * np.sum(p[:-1] ** 2)),
            np.r_[x.T @ residual + RIDGE * p[:-1], residual.sum()],
        )

    result = minimize(
        objective,
        np.r_[np.ones(len(fields)), 0.0],
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * len(fields) + [(None, None)],
        options={"ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Development calibration did not converge: " + result.message)
    return {
        "fields": fields,
        "slopes": result.x[:-1].tolist(),
        "intercept": float(result.x[-1]),
        "fit_originals_by_class": ns,
        "reference_prior": 0.5,
        "ridge": RIDGE,
    }


def predict(row, parameters):
    fields = parameters["fields"]
    if not all(valid(row.get(key)) for key in fields):
        return None
    return float(
        expit(logit(np.array([row[key] for key in fields])) @ parameters["slopes"] + parameters["intercept"])
    )


def cv(rows):
    out = [dict(row) for row in rows]
    parameters = []
    for fold in sorted({row["fold"] for row in rows}):
        train = [r for r in rows if r["fold"] != fold]
        heldout = [r for r in out if r["fold"] == fold]
        if {r["group"] for r in train} & {r["group"] for r in heldout}:
            raise ValueError("A connected source group crossed development folds")
        for arm, fields in ARMS.items():
            pars = fit(train, fields)
            parameters.append({"fold": fold, "arm": arm, **pars})
            for row in heldout:
                row[arm + "_p"] = predict(row, pars)
    return out, parameters


def decision(p):
    # Fixed research diagnostics, not a new customer screening policy.
    return (
        "inconclusive" if not valid(p) else "ai" if p >= 0.8 else "original" if p <= 0.2 else "inconclusive"
    )


def metrics(rows, key, decision_key=None):
    selected = [r for r in rows if valid(r.get(key))]
    denominators = Counter(r["label"] for r in rows)
    calls = [
        (r["label"], r.get(decision_key, "inconclusive") if decision_key else decision(r.get(key)))
        for r in rows
    ]
    out = {
        "planned_inputs": len(rows),
        "originals": len({r["base_id"] for r in rows}),
        "connected_groups": len({r["group"] for r in rows}),
        "assessed_inputs": len(selected),
        "unassessed_inputs": len(rows) - len(selected),
        "decisions": dict(Counter(d for _, d in calls)),
        "false_ai": sum(y == 0 and d == "ai" for y, d in calls),
        "false_original": sum(y == 1 and d == "original" for y, d in calls),
        "correct_ai": sum(y == 1 and d == "ai" for y, d in calls),
        "correct_original": sum(y == 0 and d == "original" for y, d in calls),
        "real_inputs": denominators[0],
        "ai_inputs": denominators[1],
    }
    w = weights(selected) if selected else None
    out.update(brier=None, log_loss=None, auc=None)
    if w is not None:
        y = np.array([r["label"] for r in selected])
        p = np.array([r[key] for r in selected])
        out.update(
            brier=float(w @ ((p - y) ** 2)),
            log_loss=float(
                w @ (-y * np.log(np.clip(p, 1e-12, 1)) - (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
            ),
            auc=float(roc_auc_score(y, p, sample_weight=w)),
        )
    return out


def paired(rows, arm):
    selected = [r for r in rows if valid(r.get("original_p")) and valid(r.get(arm + "_p"))]
    w = weights(selected) if selected else None
    if w is None:
        return {"brier_difference": None, "log_loss_difference": None, "common_inputs": len(selected)}
    y = np.array([r["label"] for r in selected])
    a = np.array([r["original_p"] for r in selected])
    b = np.array([r[arm + "_p"] for r in selected])
    brier = (b - y) ** 2 - (a - y) ** 2
    def loss(p):
        return -y * np.log(np.clip(p, 1e-12, 1)) - (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))
    logs = loss(b) - loss(a)
    rng = np.random.default_rng(SEED)
    kinds = {r["group"]: "libri" for r in selected}
    for row in selected:
        if row["dataset"] == "isjwdu/DFADD":
            kinds[row["group"]] = "joint"
    strata = {kind: sorted(g for g, k in kinds.items() if k == kind) for kind in set(kinds.values())}
    draws = []
    for _ in range(BOOTSTRAPS):
        mult = {g: 0 for g in kinds}
        for groups in strata.values():
            for g in rng.choice(groups, size=len(groups), replace=True):
                mult[g] += 1
        current = weights(selected, mult)
        if current is not None:
            draws.append((float(current @ brier), float(current @ logs)))
    draws = np.array(draws)
    return {
        "common_inputs": len(selected),
        "brier_difference": float(w @ brier),
        "log_loss_difference": float(w @ logs),
        "brier_difference_ci95": np.quantile(draws[:, 0], [0.025, 0.975]).tolist(),
        "log_loss_difference_ci95": np.quantile(draws[:, 1], [0.025, 0.975]).tolist(),
        "bootstrap_defined": len(draws),
        "bootstrap_requested": BOOTSTRAPS,
        "interpretation": "Consumed identity/prompt-connected CV diagnostic, conditional on frozen fold fits and backgrounds; not fresh-test or calibration uncertainty.",
    }


def review(manifest_path, scores_path, output):
    import hashlib

    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    manifest = json.loads(manifest_path.read_text())
    with scores_path.open() as stream:
        header, *results = [json.loads(line) for line in stream]
    if header["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Scores use a different frozen manifest")
    if len(results) != len(manifest["variants"]) or [r["id"] for r in results] != [
        r["id"] for r in manifest["variants"]
    ]:
        raise ValueError("Every planned development input needs a terminal receipt")
    rows = []
    for expected, actual in zip(manifest["variants"], results, strict=True):
        if actual["source_sha256"] != expected["sha256"]:
            raise ValueError("Source hash mismatch")
        rows.append(
            {**expected, "enhanced_raw": actual.get("enhanced_raw"), "execution_status": actual["status"]}
        )
    predictions, parameters = cv(rows)
    groups = {
        condition: {
            arm: metrics(
                [r for r in predictions if condition == "all" or r["condition"] == condition], arm + "_p"
            )
            for arm in ARMS
        }
        for condition in CONDITIONS
    }
    differences = {
        condition: {
            arm: paired([r for r in predictions if condition == "all" or r["condition"] == condition], arm)
            for arm in ("enhanced", "combined")
        }
        for condition in CONDITIONS
    }
    admissible = []
    for arm in ("enhanced", "combined"):
        delta = differences["all"][arm]
        gates = {
            "overall_brier_gain": delta["brier_difference"] <= -0.005,
            "overall_log_loss_gain": delta["log_loss_difference"] <= -0.01,
            "brier_interval_below_zero": delta["brier_difference_ci95"][1] < 0,
            "strong_noise_and_music_proper_scores": all(
                differences[c][arm]["brier_difference"] < 0 for c in ("noise_0", "music_5")
            ),
            "no_material_false_call_increase": all(
                groups[c][arm][key] <= groups[c]["original"][key] + 1
                for c in CONDITIONS[1:]
                for key in ("false_ai", "false_original")
            ),
            "clean_correct_call_retention": all(
                groups["clean"][arm][key] >= groups["clean"]["original"][key] - 2
                for key in ("correct_ai", "correct_original")
            ),
            "strong_condition_usefulness": all(
                sum(groups[c][arm][k] - groups[c]["original"][k] for k in ("correct_ai", "correct_original"))
                >= 5
                for c in ("noise_0", "music_5")
            ),
            "no_new_execution_failures": all(
                r["execution_status"] in ("ok", "not_applicable") for r in predictions
            ),
        }
        differences["all"][arm]["development_gates"] = gates
        if all(gates.values()):
            admissible.append(arm)
    primary = (
        min(admissible, key=lambda arm: (groups["all"][arm]["brier"], groups["all"][arm]["log_loss"], arm))
        if admissible
        else None
    )
    report = {
        "schema": "chipmunk.audio-noise-development-summary/4",
        "manifest_sha256": digest(manifest_path),
        "score_receipt_sha256": digest(scores_path),
        "primary_for_fresh_confirmation": primary,
        "deployment_eligible": False,
        "execution_status": dict(Counter(r["execution_status"] for r in predictions)),
        "groups": groups,
        "paired": differences,
        "cross_validation_parameters": parameters,
        "operational_reference": {
            c: metrics(
                [r for r in predictions if c == "all" or r["condition"] == c],
                "serving_p",
                decision_key="serving_decision",
            )
            for c in CONDITIONS
        },
        "limitations": manifest["limitations"]
        + [
            "Development CV probabilities are not service calibrations.",
            "No outside reviewer has independently validated these outcomes.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with output.with_suffix(".predictions.json").open("x") as stream:
        json.dump(predictions, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"primary_for_fresh_confirmation": primary, "all": groups["all"], "paired": differences["all"]}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    review(args.manifest, args.scores, args.output)
