"""Fit calibration on the frozen calibration split and compare paired video holdout results."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from prepare_video_upgrade_eval import sha
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

THRESHOLD = 0.8


def load_run(path: Path) -> tuple[dict, dict]:
    with path.open() as file:
        header = json.loads(next(file))["run"]
        rows = {}
        for line in file:
            row = json.loads(line)
            key = (row["id"], row["variant"])
            if key in rows:
                raise ValueError(f"duplicate result row: {key}")
            rows[key] = row
    return header, rows


def fit(logits: list[float], labels: list[int]) -> tuple[float, float]:
    x = np.clip(logits, -20, 20)
    y = np.asarray(labels)
    if set(y) != {0, 1}:
        raise ValueError("both classes are required for calibration")
    # Balance the classes at the reference prior, and weakly regularize slope.
    weights = np.where(y == 1, 0.5 / sum(y == 1), 0.5 / sum(y == 0))

    def loss(coef):
        z = coef[0] * x + coef[1]
        return float(np.dot(weights, np.logaddexp(0, z) - y * z) + 0.01 * (coef[0] - 1) ** 2)

    fitted = minimize(loss, (1, 0), method="L-BFGS-B", bounds=((0, None), (None, None)))
    if not fitted.success:
        raise RuntimeError(f"calibration fit failed: {fitted.message}")
    return tuple(map(float, fitted.x))


def calibrated(values, coef):
    return expit(np.clip(values, -20, 20) * coef[0] + coef[1])


def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -y * np.log(p) - (1 - y) * np.log1p(-p)


def metrics(y, p) -> dict:
    return {
        "n": len(y),
        "real": int(sum(y == 0)),
        "ai": int(sum(y == 1)),
        "auc": roc_auc_score(y, p) if len(set(y)) == 2 else None,
        "average_precision": average_precision_score(y, p) if len(set(y)) == 2 else None,
        "brier": float(np.mean((y - p) ** 2)),
        "log_loss": float(np.mean(logloss(y, p))),
    }


def tendencies(labels, directions) -> dict:
    return {
        side: dict(Counter(d for y, d in zip(labels, directions, strict=True) if y == label))
        for label, side in ((0, "real"), (1, "ai"))
    }


def direction(p: float) -> str:
    if p >= THRESHOLD:
        return "ai"
    if p <= 1 - THRESHOLD + 1e-12:
        return "original"
    return "inconclusive"


def paired_delta(groups, values) -> dict:
    per_group = np.asarray([np.mean(values[groups == g]) for g in sorted(set(groups))])
    rng = np.random.default_rng(901)
    sampled = rng.choice(per_group, size=(2000, len(per_group)), replace=True).mean(axis=1)
    return {"mean": float(per_group.mean()), "group_bootstrap_95": np.quantile(sampled, [0.025, 0.975]).tolist()}


def summarize(manifest_path: Path, current_path: Path, candidate_path: Path) -> dict:
    current_header, current = load_run(current_path)
    candidate_header, candidate = load_run(candidate_path)
    digest = sha(manifest_path)
    if current_header["system"] not in {"current", "current-synthesis-lane"} or candidate_header["system"] != "candidate":
        raise ValueError("incorrect run system")
    if digest != current_header["manifest_sha256"] or digest != candidate_header["manifest_sha256"]:
        raise ValueError("runs use different source manifests")
    variants = set(current_header["variants"])
    if not variants <= set(candidate_header["variants"]):
        raise ValueError("candidate run omits current input variants")
    candidate = {k: row for k, row in candidate.items() if k[1] in variants}
    manifest = json.loads(manifest_path.read_text())
    expected = {(row["id"], v): row for row in manifest["items"] for v in current_header["variants"]}
    if set(current) != set(expected) or set(candidate) != set(expected):
        raise ValueError("incomplete run")
    for key, item in expected.items():
        meta = item.get(key[1]) or {}
        for row in (current[key], candidate[key]):
            if row["sha256"] != meta.get("sha256") or any(
                row[k] != item[k] for k in ("group", "label", "split", "family")
            ):
                raise ValueError(f"source identity mismatch: {key}")
    if current_header["system"] == "current-synthesis-lane":
        # A check that cannot assess the input contributes no vote. Requiring
        # both native-crop checks would exclude every recompressed calibration
        # real here, since their short sides are under the larger check's limit.
        for row in current.values():
            scores = [check["prob"] for check in row.get("checks", {}).values() if check.get("prob") is not None]
            row["score"] = sum(scores) / len(scores) if scores else None
    paired = [key for key in expected if current[key].get("score") is not None and candidate[key].get("logit") is not None]
    cal = [key for key in paired if current[key]["split"] == "calibration"]
    # Both systems fit the same reference prior and common calibration rows.
    candidate_coef = fit([candidate[k]["logit"] for k in cal], [candidate[k]["label"] for k in cal])
    current_coef = fit(
        logit(np.clip([current[k]["score"] for k in cal], 1e-7, 1 - 1e-7)),
        [current[k]["label"] for k in cal],
    )
    comparisons = {}
    for variant in current_header["variants"]:
        all_test = [k for k in expected if k[1] == variant and current[k]["split"] == "test"]
        test = [k for k in all_test if k in paired]
        y = np.asarray([current[k]["label"] for k in test])
        if set(y) != {0, 1}:
            raise ValueError(f"paired holdout lacks one class for {variant}")
        old = np.asarray([current[k]["score"] for k in test])
        new = np.asarray([candidate[k]["score"] for k in test])
        old_cal = calibrated(logit(np.clip(old, 1e-7, 1 - 1e-7)), current_coef)
        new_cal = calibrated([candidate[k]["logit"] for k in test], candidate_coef)
        groups = np.asarray([current[k]["group"] for k in test])
        all_y = [expected[k]["label"] for k in all_test]
        current_decisions = {
            key: direction(float(p)) for key, p in zip(
                [k for k in all_test if current[k].get("score") is not None],
                calibrated(
                    logit(np.clip([current[k]["score"] for k in all_test if current[k].get("score") is not None], 1e-7, 1 - 1e-7)),
                    current_coef,
                ),
                strict=True,
            )
        }
        candidate_decisions = {
            key: direction(float(p)) for key, p in zip(
                [k for k in all_test if candidate[k].get("logit") is not None],
                calibrated([candidate[k]["logit"] for k in all_test if candidate[k].get("logit") is not None], candidate_coef),
                strict=True,
            )
        }
        comparisons[variant] = {
            "all_test_sources": len(all_test),
            "paired_groups": len(set(groups)),
            "current_raw": metrics(y, old),
            "candidate_raw": metrics(y, new),
            "current_recalibrated": metrics(y, old_cal),
            "candidate_calibrated": metrics(y, new_cal),
            "delta_candidate_minus_current_recalibrated_brier": paired_delta(groups, (y - new_cal) ** 2 - (y - old_cal) ** 2),
            "delta_candidate_minus_current_recalibrated_log_loss": paired_delta(groups, logloss(y, new_cal) - logloss(y, old_cal)),
            "current_recalibrated_tendencies_at_80": tendencies(
                all_y, [current_decisions.get(k, current[k]["status"]) for k in all_test]
            ),
            "candidate_tendencies_at_80": tendencies(all_y, [candidate_decisions.get(k, candidate[k]["status"]) for k in all_test]),
        }
        if current_header["system"] == "current":
            comparisons[variant]["current_customer_tendencies"] = tendencies(
                all_y, [current[k].get("customer_direction", current[k]["status"]) for k in all_test]
            )
        else:
            check_metrics = {}
            for check in current_header["checks"]:
                keys = [k for k in all_test if current[k].get("checks", {}).get(check, {}).get("prob") is not None]
                if keys:
                    check_metrics[check] = metrics(
                        np.asarray([current[k]["label"] for k in keys]),
                        np.asarray([current[k]["checks"][check]["prob"] for k in keys]),
                    )
            comparisons[variant]["current_individual_check_scores"] = check_metrics
    return {
        "manifest_sha256": digest,
        "current_result_sha256": sha(current_path),
        "candidate_result_sha256": sha(candidate_path),
        "current_system": current_header["system"],
        "calibration_rows": len(cal),
        "calibration_source_groups": len({current[k]["group"] for k in cal}),
        "calibration_reference_prior": 0.5,
        "current_calibration_slope_intercept": current_coef,
        "candidate_calibration_slope_intercept": candidate_coef,
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "current", "candidate", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = summarize(args.manifest, args.current, args.candidate)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
