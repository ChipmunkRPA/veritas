"""Reconstruct both layer mappings, independent guards/stability and exact refits.

No neural inference or reserved confirmation occurs. The predictor arithmetic,
normalization and operational counts are checked separately from the runner.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import expit

from scripts import audio_layer_response_v12 as method
from scripts import run_audio_layer_v12 as execution
from scripts.audio_multisource_v8 import balanced_weights
from scripts.audio_source_identity_v9 import digest
from scripts.evaluate_audio_multisource_v8 import summarize
from scripts.prepare_audio_guard_v7 import write
from scripts.publish_audio_identity_results_v9 import comparable
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.review_audio_conditional_v10 import guards
from scripts.review_audio_guard_v7 import close, initial, logit, metric
from scripts.review_audio_identity_numerics_v9 import decisions


def probability(row, fitted):
    if row["raw"] is None:
        return None
    if (isinstance(row["raw"], bool) or not isinstance(row["raw"], (float, int, np.number))
            or not math.isfinite(row["raw"]) or not 0 <= row["raw"] <= 1):
        raise ValueError("Finite original probability required")
    arm = fitted["arm"]
    if arm not in method.ARMS:
        raise ValueError("Unknown response arm")
    values = np.asarray(row["layer_embeddings"] if arm == "layers" else row["embedding"], np.float64)
    if values.shape != ((3, 1024) if arm == "layers" else (1024,)):
        raise ValueError("Changed original representation shape")
    vector = values.reshape(-1)
    t = fitted["transform"]
    center, scale, coefficients = [np.asarray(v, np.float64) for v in (t["center"], t["scale"], fitted["coefficients"])]
    if (vector.shape != center.shape or scale.shape != center.shape or len(coefficients) != len(vector) + 2
            or not np.isfinite(vector).all() or not np.isfinite(coefficients).all()
            or not np.isfinite(center).all() or not np.isfinite(scale).all()
            or (scale < 1e-6).any() or coefficients[0] < 0):
        raise ValueError("Changed finite representation/normalization/coefficient contract")
    standardized = (vector - center) / scale * t["variance_factor"]
    z = logit(row["raw"]) * coefficients[0] + standardized @ coefficients[1:-1] + coefficients[-1]
    return float(expit(z))


def normalization(rows, fitted):
    if not rows or any(r["role"] != "fit" for r in rows):
        raise ValueError("Normalization must use fitting sources only")
    selected = [r for r in rows if r["raw"] is not None]
    arm = fitted["arm"]
    w = balanced_weights(selected)
    matrix = np.asarray([np.asarray(r["layer_embeddings"] if arm == "layers" else r["embedding"], np.float64).reshape(-1)
                         for r in selected])
    center = w @ matrix
    scale = np.maximum(np.sqrt(w @ (matrix - center) ** 2), 1e-6)
    expected = {"center": center.tolist(), "scale": scale.tolist(),
                "variance_factor": float(math.sqrt(16 / matrix.shape[1])),
                "fit_sources": len({r["base_id"] for r in selected})}
    if not same_tree(expected, fitted["transform"]):
        raise ValueError("Fitting-only center, scale or fixed variance budget differs")


def reconstruct(head, rows, baseline, responses, *, refit=False):
    if set(head["fitted"]) != set(method.ARMS) or set(head["guards"]) != set(method.ARMS):
        raise ValueError("Both frozen response arms are required")
    fit = [r for r in rows if r["role"] == "fit"]
    cal = [r for r in rows if r["role"] == "cal"]
    development = [r for r in rows if r["role"] == "development"]
    if [r["id"] for r in responses] != [r["id"] for r in development]:
        raise ValueError("Complete development order and denominator required")
    audits, summaries = {}, {}
    for arm in method.ARMS:
        fitted, guard = head["fitted"][arm], head["guards"][arm]
        if fitted["method"] != method.METHOD or fitted["arm"] != arm:
            raise ValueError("Unchanged learner and arm definition required")
        normalization(fit, fitted)
        rebuilt_guards = guards(cal, [probability(r, fitted) for r in cal])
        if not same_tree(rebuilt_guards, guard):
            raise ValueError("Independent worst-group calibration differs")
        if refit:
            rebuilt = method.fit(fit, arm)
            if rebuilt != fitted:
                raise ValueError("Original fitting-only refit did not reproduce the exact complete object")
        flat = []
        for row, source in zip(responses, development, strict=True):
            if any(row[k] != source[k] for k in execution.META):
                raise ValueError("Outcome source/truth/group binding differs")
            candidate = row["candidates"][arm]
            if row["status"] == "failed":
                if candidate["p"] is not None or candidate["decision"] != "inconclusive":
                    raise ValueError("Failures cannot establish directional conclusions")
            else:
                old = baseline[row["id"]]
                close(row["original_p"], old["original_p"])
                if row["original_decision"] != old["original_decision"] or row["status"] != source["status"]:
                    raise ValueError("Original service or assessment status differs")
                p = probability(source, fitted)
                close(candidate["p"], p)
                # Need for a probe is shared by both arms, not chosen by this arm alone.
                needed = any(initial(probability(source, head["fitted"][a]), head["guards"][a]) == "original"
                             for a in method.ARMS)
                probe = row["probe"]
                if probe["status"] not in (("computed", "failed") if needed else ("not_required",)):
                    raise ValueError("Missing or unexpected shared stability probe")
                band = probability(probe, fitted) if probe["status"] == "computed" else None
                close(probe["p"][arm], band)
                detail = decisions(p, band, source["raw"], probe["raw"], guard)
                if not same_tree(detail, candidate["detail"]) or detail["decision"] != candidate["decision"]:
                    raise ValueError("Independent raw/mapped stability or conclusion differs")
            flat.append({**row, "candidate_p": candidate["p"], "candidate_decision": candidate["decision"]})
        rebuilt_summary = summarize(flat, 161)
        for condition in ("all", *execution.prior.original.CONDITIONS):
            panel = [r for r in flat if condition == "all" or r["condition"] == condition]
            for score_key, decision_key, name in (("original_p", "original_decision", "service"),
                                                   ("candidate_p", "candidate_decision", "candidate")):
                for field, value in metric(panel, score_key, decision_key).items():
                    actual = rebuilt_summary["groups"][condition][name][field]
                    if field in ("brier", "log_loss"):
                        close(value, actual)
                    elif value != actual:
                        raise ValueError("Independent full-denominator count differs")
        summaries[arm] = rebuilt_summary
        audits[arm] = {"normalization_rebuilt": True, "independent_guards_rebuilt": True,
                       "all_dual_stability_decisions_rebuilt": True, "operational_counts_rebuilt": True,
                       "refit_performed": refit, "exact_refit_object_matched": True if refit else None}
    return summaries, audits


def review(lock, receipt, head_path, responses_path, summary_path, output, *, refit=False):
    _, (_, _, _, baseline, _), rows, head, responses = execution.bound_responses(
        lock, receipt, head_path, responses_path)
    original = json.loads(summary_path.read_text())
    rebuilt, audits = reconstruct(head, rows, baseline, responses, refit=refit)
    if (original["lock_sha256"] != digest(lock) or original["receipt_sha256"] != digest(receipt)
            or original["head_sha256"] != digest(head_path) or original["responses_sha256"] != digest(responses_path)
            or not same_tree({a: comparable(s) for a, s in rebuilt.items()},
                             {a: comparable(s) for a, s in original["arms"].items()})
            or method.select(rebuilt) != original["selected_arm_for_possible_confirmation"]):
        raise ValueError("Original summary, all gates or arm selection differs")
    result = {"schema": "chipmunk.audio-layer-numerical-review/12", "passed": True,
              "lock_sha256": digest(lock), "receipt_sha256": digest(receipt), "head_sha256": digest(head_path),
              "responses_sha256": digest(responses_path), "summary_sha256": digest(summary_path),
              "review_code_sha256": digest(Path(__file__)), "inputs": len(rows), "outcomes": len(responses),
              "computed_probes": sum(r["probe"]["status"] == "computed" for r in responses), "arms": audits,
              "selected_arm_for_possible_confirmation": method.select(rebuilt), "deployment_eligible": False,
              "scope": "Independent linear arithmetic, fitting-only normalization, group guards, dual stability/counts, "
                       "shared metric/bootstrap gates and optional exact original-learner refit. No neural inference, "
                       "reserved confirmation or outside human validation."}
    write(output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lock", type=Path)
    p.add_argument("receipt", type=Path)
    p.add_argument("head", type=Path)
    p.add_argument("responses", type=Path)
    p.add_argument("summary", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--refit", action="store_true")
    a = p.parse_args()
    print(json.dumps(review(a.lock, a.receipt, a.head, a.responses, a.summary, a.output, refit=a.refit)), flush=True)
