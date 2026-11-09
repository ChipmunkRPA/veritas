"""Small monotone score fusion with independent calibration and measured-input routing."""
from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

METHOD = {
    "version": "audio-score-fusion/83", "predictors": ["original_raw_ai_score", "frozen_complement_ai_score"],
    "arms": ["global", "flatness_context"], "logit_clip": 8.0,
    "routing": "Median input spectral flatness on assessed fitting windows only; no label/provider/condition at prediction",
    "flatness": "Median frame power-spectrum geometric/arithmetic mean ratio; demeaned 512-sample Hann frames, hop256, power floor1e-12",
    "weighting": "Equal truth, observed generator family, connected group, source and assessed variant within each fitting/calibration branch",
    "reference_prior": 0.5, "minimum_groups_per_truth_per_branch": 3,
    "fusion_ridge": 0.02, "calibration_ridge": 0.005,
    "bounds": {"nonnegative_slopes": [0.0, 5.0], "intercept": [-10.0, 10.0]},
    "solver": {"method": "L-BFGS-B", "ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000},
    "calibration": "Separate positive-slope logistic map fitted only on confidence_cal role for each supported branch",
    "unsupported": "Explicit unassessed; no borrowed branch calibration, missing-to-zero conversion or original verdict",
    "scope": "Research scores under a balanced reference prior, not population probabilities, statistical confidence intervals or a serving change",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def flatness(pcm):
    values = np.asarray(pcm, np.float64)
    require(values.shape == (64000,) and np.isfinite(values).all(), "Exact finite four-second input required")
    frames = np.lib.stride_tricks.sliding_window_view(values, 512)[::256]
    frames = (frames - frames.mean(axis=1, keepdims=True)) * np.hanning(512)
    power = np.maximum(np.abs(np.fft.rfft(frames, axis=1)) ** 2, 1e-12)
    result = float(np.median(np.exp(np.log(power).mean(axis=1)) / power.mean(axis=1)))
    require(np.isfinite(result) and 0 <= result <= 1.00000001, "Finite bounded measured flatness required")
    return min(1.0, result)


def predictors(row):
    if row["status"] == "not_applicable":
        return None
    require(row["status"] == "ok", "Failed extraction cannot supply a conclusion")
    values = [row["raw"], row["candidates"]["complement"]["p"]]
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) and 0 <= value <= 1 for value in values),
            "Both actual finite component scores required")
    return np.clip(logit(np.clip(values, 1e-6, 1 - 1e-6)), -8.0, 8.0)


def branch(row, artifact):
    if artifact["arm"] == "global":
        return "all"
    value = row["quality_flatness"]
    require(not isinstance(value, bool) and np.isfinite(value) and 0 <= value <= 1, "Measured bounded input flatness required")
    return "low" if value <= artifact["routing_cutoff"] else "high"


def weights(rows):
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for index, row in enumerate(rows):
        require(type(row["label"]) is int and row["label"] in {0, 1}
                and row["kind"] in {"genuine", "clone", "expressive", "stock"}
                and (row["kind"] == "genuine") == (row["label"] == 0), "Known source truth/family required for fitting only")
        groups[row["label"]][row["kind"]][row["group"]][row["base_id"]].append(index)
    require(set(groups) == {0, 1}, "Both truth classes required")
    result = np.zeros(len(rows), dtype=np.float64)
    for families in groups.values():
        for connected in families.values():
            for sources in connected.values():
                for indices in sources.values():
                    result[indices] = 1 / (2 * len(families) * len(connected) * len(sources) * len(indices))
    require(np.isclose(result.sum(), 1) and bool((result > 0).all()), "Complete balanced fitting weights required")
    return result


def support(rows):
    return {str(label): len({row["group"] for row in rows if row["label"] == label}) for label in (0, 1)}


def solve(matrix, rows, ridge):
    x = np.column_stack([matrix, np.ones(len(rows))])
    y, w = np.asarray([row["label"] for row in rows], np.float64), weights(rows)
    penalty = np.r_[np.full(x.shape[1] - 1, ridge), 0.0]

    def objective(beta):
        z = x @ beta
        return (float(w @ (np.logaddexp(0, z) - y * z) + 0.5 * np.dot(penalty, beta ** 2)),
                x.T @ (w * (expit(z) - y)) + penalty * beta)

    solver = METHOD["solver"]
    fitted = minimize(objective, np.r_[np.ones(x.shape[1] - 1), 0.0], jac=True,
                      method=solver["method"], bounds=[(0.0, 5.0)] * (x.shape[1] - 1) + [(-10.0, 10.0)],
                      options={key: solver[key] for key in ("ftol", "gtol", "maxiter")})
    require(fitted.success and np.isfinite(fitted.x).all(), "Fixed monotone score fit did not converge")
    return fitted.x.tolist()


def fit(rows, arm):
    require(bool(rows) and all(row["role"] == "fusion_fit" for row in rows)
            and arm in METHOD["arms"], "Only assigned fusion fitting sources and a fixed arm may train")
    selected = [row for row in rows if predictors(row) is not None]
    require(bool(selected), "Assessed fitting inputs required")
    artifact = {"method": copy.deepcopy(METHOD), "arm": arm, "branches": {},
                "fit_group_ids": sorted({row["group"] for row in rows}),
                "fit_source_ids": sorted({row["base_id"] for row in rows})}
    if arm == "flatness_context":
        values = [row["quality_flatness"] for row in selected]
        require(all(not isinstance(value, bool) and np.isfinite(value) and 0 <= value <= 1 for value in values),
                "Measured fitting flatness required")
        artifact["routing_cutoff"] = float(np.median(values))
    for name in ("all",) if arm == "global" else ("low", "high"):
        subset = [row for row in selected if branch(row, artifact) == name]
        counts = support(subset)
        detail = {"status": "unsupported_fit", "fit_groups_by_truth": counts,
                  "fit_sources": len({row["base_id"] for row in subset}), "fit_windows": len(subset)}
        if min(counts.values()) >= METHOD["minimum_groups_per_truth_per_branch"]:
            detail.update(status="fitted", coefficients=solve(np.asarray([predictors(row) for row in subset]),
                                                             subset, METHOD["fusion_ridge"]))
        artifact["branches"][name] = detail
    return artifact


def calibrate(artifact, rows):
    require(bool(rows) and all(row["role"] == "confidence_cal" for row in rows),
            "Only separate confidence calibration sources may calibrate")
    result = copy.deepcopy(artifact)
    require(result["method"] == METHOD, "Exact fixed fusion method required")
    require(not (set(result["fit_group_ids"]) & {row["group"] for row in rows})
            and not (set(result["fit_source_ids"]) & {row["base_id"] for row in rows}),
            "Fitting and calibration sources/groups must be disjoint")
    result["calibration_group_ids"] = sorted({row["group"] for row in rows})
    result["calibration_source_ids"] = sorted({row["base_id"] for row in rows})
    selected = [row for row in rows if predictors(row) is not None]
    for name, detail in result["branches"].items():
        require(detail["status"] in {"fitted", "unsupported_fit"}, "A calibration cannot be refitted or borrowed")
        if detail["status"] == "unsupported_fit":
            continue
        subset = [row for row in selected if branch(row, result) == name]
        counts = support(subset)
        detail.update(status="unsupported_calibration", calibration_groups_by_truth=counts,
                      calibration_windows=len(subset))
        if min(counts.values()) >= METHOD["minimum_groups_per_truth_per_branch"]:
            beta = np.asarray(detail["coefficients"])
            z = np.asarray([np.r_[predictors(row), 1.0] @ beta for row in subset])[:, None]
            detail.update(status="calibrated", calibration=solve(z, subset, METHOD["calibration_ridge"]))
    return result


def predict(row, artifact):
    require(artifact["method"] == METHOD, "Exact fixed fusion artifact required")
    x = predictors(row)
    if x is None:
        return {"status": "unassessed", "probability_ai": None, "decision": "unassessed", "reason": "no_assessed_speech_window"}
    name = branch(row, artifact)
    detail = artifact["branches"][name]
    if detail["status"] != "calibrated":
        return {"status": "unassessed", "probability_ai": None, "decision": "unassessed", "reason": detail["status"], "branch": name}
    z = float(np.r_[x, 1.0] @ np.asarray(detail["coefficients"]))
    p = float(expit(np.r_[z, 1.0] @ np.asarray(detail["calibration"])))
    return {"status": "ok", "probability_ai": p, "branch": name,
            "decision": "ai" if p > 0.5 else "original" if p < 0.5 else "inconclusive",
            "scope": "Balanced-reference-prior score tendency; no population accuracy or confidence interval guarantee"}
