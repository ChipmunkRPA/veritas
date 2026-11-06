"""CPU reconstruction of published source roles, fitting, guards and decisions.

Fitting reuses the declared learning implementation. Guard/decision arithmetic
is reconstructed here. No source decoding, neural inference or outside review.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

from scripts.audio_guard_v7 import source_roles
from scripts.audio_residual_head_v6 import digest_embedding, fit


def close(a, b):
    if a is None or b is None:
        if a != b:
            raise ValueError("Missing probability mismatch")
    elif not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Probability/guard arithmetic mismatch")


def logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p) - math.log1p(-p)


def probability(row, head):
    if row["raw"] is None:
        return None
    transform = head["transform"]
    e = np.asarray(row["embedding"], np.float64)
    h = np.asarray(transform["unit_head"], np.float64)
    e = e - float(e @ h) * h
    reduced = ((e - np.asarray(transform["center"])) @ np.asarray(transform["directions"]).T) / np.asarray(
        transform["scale"]
    )
    return float(expit(np.r_[logit(row["raw"]), reduced, 1.0] @ np.asarray(head["coefficients"])))


def guards(rows, fitted):
    pools = {0: defaultdict(list), 1: defaultdict(list)}
    for row in rows:
        p = probability(row, fitted)
        pools[row["label"]][row["group"]].append(None if p is None else p if row["label"] == 0 else 1 - p)
    return {
        str(c): [
            {
                "speaker": g,
                "maximum": None if any(v is None for v in values) else max(values),
                "rows": len(values),
            }
            for g, values in sorted(pools[c].items())
        ]
        for c in (0, 1)
    }


def initial(p, g):
    if p is None or min(len(g[str(c)]) for c in (0, 1)) < 9:
        return "inconclusive"
    included = []
    for c in (0, 1):
        nc = p if c == 0 else 1 - p
        rank = (1 + sum(r["maximum"] is None or r["maximum"] >= nc for r in g[str(c)])) / (len(g[str(c)]) + 1)
        if rank > 0.1:
            included.append(c)
    lean = "ai" if p >= 0.8 else "original" if p <= 0.2 + 1e-12 else "inconclusive"
    return lean if lean != "inconclusive" and included == ([1] if lean == "ai" else [0]) else "inconclusive"


def metric(rows, key, decision_key):
    calls = [(r["label"], r[decision_key]) for r in rows]
    selected = [r for r in rows if r[key] is not None]
    result = {
        "planned_inputs": len(rows),
        "assessed_inputs": len(selected),
        "unassessed_inputs": len(rows) - len(selected),
        "false_ai": sum(y == 0 and d == "ai" for y, d in calls),
        "false_original": sum(y == 1 and d == "original" for y, d in calls),
        "correct_ai": sum(y == 1 and d == "ai" for y, d in calls),
        "correct_original": sum(y == 0 and d == "original" for y, d in calls),
        "decisions": dict(Counter(d for _, d in calls)),
    }
    counts = Counter(r["base_id"] for r in selected)
    weight = [1 / counts[r["base_id"]] for r in selected]
    total = {c: sum(w for w, r in zip(weight, selected, strict=True) if r["label"] == c) for c in (0, 1)}
    if min(total.values()) <= 0:
        return result
    weight = [w * 0.5 / total[r["label"]] for w, r in zip(weight, selected, strict=True)]
    result["brier"] = sum(w * (r[key] - r["label"]) ** 2 for w, r in zip(weight, selected, strict=True))
    result["log_loss"] = sum(
        w
        * (
            -r["label"] * math.log(max(1e-12, min(1.0, r[key])))
            - (1 - r["label"]) * math.log(max(1e-12, min(1.0, 1 - r[key])))
        )
        for w, r in zip(weight, selected, strict=True)
    )
    return result


def review(body, refit=False):
    rows = body["rows"]
    if (
        Counter(r["role"] for r in rows) != {"fit": 240, "cal": 480, "development": 720}
        or len({r["id"] for r in rows}) != 1440
    ):
        raise ValueError("All complete fitting/calibration/development inputs required")
    sources = [
        {
            **r,
            "transcript": r["role_text_sha256"],
            "dataset": "isjwdu/DFADD" if "dfadd:" in r["id"] else "other",
        }
        for r in body["sources"]
    ]
    assignments, source_groups = source_roles(sources)
    if source_groups != body["groups"] or any(
        r["group"] != assignments[r["base_id"]]["group"] or r["role"] != assignments[r["base_id"]]["role"]
        for r in rows
    ):
        raise ValueError("Source group/role leakage or altered assignments")
    for base in assignments:
        current = [r for r in rows if r["base_id"] == base]
        if len(current) != 6 or {r["condition"] for r in current} != {
            "clean",
            "noise_0",
            "noise_5",
            "noise_15",
            "music_5",
            "music_15",
        }:
            raise ValueError("Missing source condition")
    fitted = body["head"]["fitted"]
    if set(body["head"]["fit_groups"]) != {r["group"] for r in rows if r["role"] == "fit"} or set(
        body["head"]["cal_groups"]
    ) != {r["group"] for r in rows if r["role"] == "cal"}:
        raise ValueError("Fitted/calibration group declarations differ")
    cal = [r for r in rows if r["role"] == "cal"]
    expected = guards(cal, fitted)
    actual = body["head"]["guards"]
    for c in ("0", "1"):
        if [(r["speaker"], r["rows"]) for r in actual[c]] != [(r["speaker"], r["rows"]) for r in expected[c]]:
            raise ValueError("Incomplete independent guard population")
        for a, b in zip(actual[c], expected[c], strict=True):
            close(a["maximum"], b["maximum"])
    development = [r for r in rows if r["role"] == "development"]
    results = body["results"]
    if [r["id"] for r in results] != [r["id"] for r in development]:
        raise ValueError("Missing development outcome")
    bundle = body["serving_bundle"]
    baseline_guards = {str(c): [{"maximum": v} for v in bundle["guards"][str(c)]] for c in (0, 1)}
    ref = {r["id"]: r for r in body["reference"]}
    original_direction = np.asarray(body["representation_header"]["head_weight"], np.float64)
    original_bias = body["representation_header"]["head_bias"]
    for row in rows:
        if row["raw"] is not None:
            reconstructed = float(
                expit(np.asarray(row["embedding"], np.float64) @ original_direction + original_bias)
            )
            if abs(reconstructed - row["raw"]) > 1e-6:
                raise ValueError("Representation does not reconstruct original raw response")
    probe_count = 0
    for source, row in zip(development, results, strict=True):
        if any(row[k] != source[k] for k in ("sha256", "pcm_sha256", "group", "label", "condition")):
            raise ValueError("Result bound to different source evidence")
        p = probability(source, fitted)
        close(row["candidate_p"], p)
        proposed = initial(p, actual)
        probe = row["probe"]
        band_p = None
        if proposed == "original":
            if probe["status"] == "computed":
                if (
                    digest_embedding(probe["embedding"]) != probe["embedding_sha256"]
                    or probe["samples"] != source["samples"]
                ):
                    raise ValueError("Probe representation/sample clock differs")
                band_p = probability({**source, "raw": probe["raw"], "embedding": probe["embedding"]}, fitted)
                close(band_p, probe["p"])
                probe_count += 1
            elif probe["status"] != "failed":
                raise ValueError("Required original stability probe is missing")
        elif probe["status"] != "not_required":
            raise ValueError("Unexpected probe execution")
        delta = (
            abs(logit(round(p, 6)) - logit(round(band_p, 6)))
            if proposed == "original" and band_p is not None
            else None
        )
        decision = (
            "inconclusive" if proposed == "original" and (delta is None or delta > math.log(2)) else proposed
        )
        close(delta, row["candidate_detail"]["stability_log_odds_change"])
        if row["candidate_decision"] != decision or row["candidate_detail"]["proposed"] != proposed:
            raise ValueError("Guard/stability conclusion differs")
        baseline_p = (
            None
            if source["raw"] is None
            else float(
                expit(
                    bundle["parameters"]["slope"] * logit(source["raw"]) + bundle["parameters"]["intercept"]
                )
            )
        )
        close(baseline_p, row["original_p"])
        baseline = initial(baseline_p, baseline_guards)
        if baseline == "original":
            band = ref[row["id"]]["saved_band_response"]
            if band is None or abs(logit(round(source["raw"], 6)) - logit(round(band, 6))) > math.log(2):
                baseline = "inconclusive"
        if row["original_decision"] != baseline:
            raise ValueError("Actual-service baseline differs")
    panels = {
        c: [r for r in results if c == "all" or r["condition"] == c]
        for c in ("all", "clean", "noise_0", "noise_5", "noise_15", "music_5", "music_15")
    }
    panels["joint_vctk"] = [r for r in results if r["dataset"] == "isjwdu/DFADD"]
    summary = body["summary"]
    for name, panel in panels.items():
        for arm, key, decision in (
            ("service", "original_p", "original_decision"),
            ("candidate", "candidate_p", "candidate_decision"),
        ):
            expected_metric = metric(panel, key, decision)
            for field, value in expected_metric.items():
                actual_metric = summary["groups"][name][arm][field]
                if field in ("brier", "log_loss"):
                    close(value, actual_metric)
                elif value != actual_metric:
                    raise ValueError("Complete operational counts differ")
        close(
            summary["differences"][name]["brier_difference"],
            summary["groups"][name]["candidate"]["brier"] - summary["groups"][name]["service"]["brier"],
        )
    groups = summary["groups"]
    differences = summary["differences"]

    def errors(c, k):
        return groups[c]["candidate"][k] <= groups[c]["service"][k] + 1

    expected_gates = {
        "brier_gain": differences["all"]["brier_difference"] <= -0.005,
        "log_loss_gain": differences["all"]["log_loss_difference"] <= -0.01,
        "brier_interval_below_zero": differences["all"]["brier_difference_ci95"][1] < 0,
        "strong_condition_brier": all(differences[c]["brier_difference"] < 0 for c in ("noise_0", "music_5")),
        "joint_voice_brier": differences["joint_vctk"]["brier_difference"] < 0,
        "total_false_calls": all(errors("all", k) for k in ("false_ai", "false_original")),
        "condition_false_calls": all(
            errors(c, k)
            for c in panels
            if c not in ("all", "joint_vctk")
            for k in ("false_ai", "false_original")
        ),
        "clean_retention": all(
            groups["clean"]["candidate"][k] >= groups["clean"]["service"][k] - 2
            for k in ("correct_ai", "correct_original")
        ),
        "strong_condition_usefulness": all(
            sum(
                groups[c]["candidate"][k] - groups[c]["service"][k]
                for k in ("correct_ai", "correct_original")
            )
            >= 5
            for c in ("noise_0", "music_5")
        ),
        "same_assessed": groups["all"]["service"]["assessed_inputs"]
        == groups["all"]["candidate"]["assessed_inputs"],
        "complete_cases": len(results) == 720 and len(rows) == 1440,
        "no_probe_failures": all(r["probe"]["status"] != "failed" for r in results),
    }
    if summary["gates"] != expected_gates or summary["passes"] != all(expected_gates.values()):
        raise ValueError("Frozen advancement gates differ")
    refit_error = None
    if refit:
        rebuilt = fit([r for r in rows if r["role"] == "fit"], body["representation_header"]["head_weight"])
        # Identical pinned numerical runtime should reconstruct the saved orientation.
        directions = np.asarray(rebuilt["transform"]["directions"])
        signs = np.where(
            np.sum(directions * np.asarray(fitted["transform"]["directions"]), axis=1) < 0, -1.0, 1.0
        )
        rebuilt["transform"]["directions"] = (directions * signs[:, None]).tolist()
        coef = np.asarray(rebuilt["coefficients"])
        coef[1:17] *= signs
        rebuilt["coefficients"] = coef.tolist()
        refit_error = float(np.max(np.abs(coef - np.asarray(fitted["coefficients"]))))
        for field in ("unit_head", "center", "directions", "scale"):
            if not np.allclose(rebuilt["transform"][field], fitted["transform"][field], rtol=1e-9, atol=1e-9):
                raise ValueError("CPU projection refit differs")
        if refit_error > 1e-7:
            raise ValueError("CPU decision-head refit differs")
        if os.environ.get("PYTHONHASHSEED") != body["statistical_runtime"]["python_hash_seed"]:
            raise ValueError(
                "Set PYTHONHASHSEED to the published statistical runtime before bootstrap replay"
            )
        from scripts.review_audio_noise_v4 import paired

        for name, panel in panels.items():
            repeated = paired(panel, "candidate")
            for field in ("brier_difference", "log_loss_difference"):
                close(repeated[field], differences[name][field])
            for a, b in zip(
                repeated["brier_difference_ci95"], differences[name]["brier_difference_ci95"], strict=True
            ):
                close(a, b)
    return {
        "passed": True,
        "source_inputs": 1440,
        "development_outcomes": 720,
        "guard_groups": {k: len(v) for k, v in actual.items()},
        "computed_probes": probe_count,
        "refit_max_coefficient_error": refit_error,
        "scope": "CPU source-role/fitting/probability/guard/baseline/stability/operational-count/gate reconstruction. Refit optionally repeats shared fitter/bootstrap. No waveforms, neural re-execution, population guarantee or outside reviewer.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    result = review(json.loads(gzip.decompress(args.capsule.read_bytes())), args.refit)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
