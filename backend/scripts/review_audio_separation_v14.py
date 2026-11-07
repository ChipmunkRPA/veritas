"""Additive arithmetic review; cannot change the frozen source experiment.

Uses separate source/bag probability, normalization and group-calibration code,
the existing independent guard/stability reviewer, and the declared shared
statistics/gate implementation. Exact refit uses the original fixed learner.
No model forward or outside human review occurs here.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import run_audio_separation_v14 as run
from scripts.audio_multisource_v8 import balanced_weights
from scripts.audio_source_identity_v9 import digest
from scripts.prepare_audio_guard_v7 import write
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.review_audio_guard_v7 import close, logit
from scripts.review_audio_identity_numerics_v9 import decisions
from scripts.run_audio_guard_v7 import load_rows


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def probability(row, fitted):
    if row["raw"] is None:
        if row["stems"] is not None:
            raise ValueError("Unassessed mixture cannot have assessed source scores")
        return None
    transform = fitted["transform"]
    center, scale, beta = [np.asarray(x, np.float64) for x in
                           (transform["center"], transform["scale"], fitted["coefficients"])]
    arm = fitted["arm"]
    if (arm not in ("separated", "complement") or center.shape != (1024,) or scale.shape != (1024,)
            or beta.shape != (1026 + (2 if arm == "complement" else 0),)
            or not np.isfinite(center).all() or not np.isfinite(scale).all() or not np.isfinite(beta).all()
            or (scale < 1e-6).any() or transform["variance_factor"] != .125 or beta[0] < 0
            or (arm == "complement" and beta[-2] < 0) or len(row["stems"]) != 2):
        raise ValueError("Complete finite declared source/bag mapping required")
    z = []
    for stem in row["stems"]:
        vector = np.asarray(stem["embedding"], np.float64)
        if vector.shape != (1024,) or not np.isfinite(vector).all():
            raise ValueError("Complete finite source embedding required")
        normalized = (vector - center) / scale / 8
        value = math.fsum([beta[0] * logit(stem["raw"]),
                          *(normalized * beta[1:1025]).tolist(), beta[1025]])
        z.append(value)
    terms = [z[0], z[1], z[0] + z[1]]
    peak = max(terms)
    margin = peak + math.log(math.fsum(math.exp(v - peak) for v in terms))
    if arm == "complement":
        margin += beta[-2] * logit(row["raw"]) + beta[-1]
    return sigmoid(margin)


def numerical(lock_path, receipt, head_path, responses_path, summary_path, output, refit):
    _, (_, _, _, baseline, _), rows = run.feature_rows(lock_path, receipt)
    head = run.read_head(lock_path, receipt, head_path)
    meta, records = load_rows(responses_path)
    summary = json.loads(summary_path.read_text())
    sources = [r for r in rows if r["role"] == "development"]
    if (meta != run.response_header(lock_path, receipt, head_path)
            or summary["lock_sha256"] != digest(lock_path) or summary["receipt_sha256"] != digest(receipt)
            or summary["head_sha256"] != digest(head_path) or summary["responses_sha256"] != digest(responses_path)
            or [r["id"] for r in records] != [r["id"] for r in sources]):
        raise ValueError("Complete original1449 outcomes and source/fitting lineage required")
    fitting = [r for r in rows if r["role"] == "fit" and r["raw"] is not None]
    calibration = [r for r in rows if r["role"] == "cal"]
    # Separate flatten-and-weight construction, rather than the learner's einsum.
    weights = np.repeat(balanced_weights(fitting) / 2, 2)
    matrix = np.asarray([s["embedding"] for r in fitting for s in r["stems"]], np.float64)
    center = weights @ matrix
    scale = np.maximum(np.sqrt(weights @ (matrix - center)**2), 1e-6)
    audit, rebuilt_summaries = {}, {}
    for arm in run.method.ARMS:
        fitted = head["fitted"][arm]
        expected_transform = {"center": center.tolist(), "scale": scale.tolist(), "variance_factor": .125,
                              "fit_sources": len({r["base_id"] for r in fitting})}
        if not same_tree(fitted["transform"], expected_transform):
            raise ValueError("Shared fitting-only source normalization differs")
        if refit and run.method.fit([r for r in rows if r["role"] == "fit"], arm) != fitted:
            raise ValueError("Exact complete fitting-only bag refit differs")
        guards = {}
        for label in (0, 1):
            groups = []
            for group in sorted({r["group"] for r in calibration if r["label"] == label}):
                values = [probability(r, fitted) for r in calibration if r["label"] == label and r["group"] == group]
                maximum = None if None in values else max(v if label == 0 else 1 - v for v in values)
                groups.append({"group": group, "maximum": maximum, "rows": len(values)})
            guards[str(label)] = groups
        if not same_tree(guards, head["guards"][arm]):
            raise ValueError("Separate group/truth calibration differs")
        counts, flat = Counter(), []
        for record, source in zip(records, sources, strict=True):
            if record["status"] != source["status"] or any(record[k] != source[k] for k in run.prior.META):
                raise ValueError("Role, truth, group, input or assessment binding differs")
            original = baseline[source["id"]]
            if any(record[k] != original[k] for k in ("original_p", "original_decision")):
                raise ValueError("Actual service control differs")
            p = probability(source, fitted)
            probe = record["probe"]
            band = probability(probe, fitted) if probe["status"] == "computed" else None
            expected = decisions(p, band, source["raw"], probe["raw"], guards)
            reported = record["candidates"][arm]
            close(reported["p"], p)
            if reported["decision"] != expected["decision"] or not same_tree(reported["detail"], expected):
                raise ValueError("Separate bag/guard/raw-stability outcome differs")
            counts[expected["decision"]] += 1
            if expected["decision"] == "ai":
                counts["correct_ai" if source["label"] == 1 else "false_ai"] += 1
            if expected["decision"] == "original":
                counts["correct_original" if source["label"] == 0 else "false_original"] += 1
            flat.append({**record, "candidate_p": p, "candidate_decision": expected["decision"]})
        reported = summary["arms"][arm]["groups"]["all"]["candidate"]
        if (any(counts[k] != reported[k] for k in ("correct_ai", "correct_original", "false_ai", "false_original"))
                or any(counts[k] != reported["decisions"].get(k, 0) for k in ("ai", "original", "inconclusive"))):
            raise ValueError("Complete independent operational counts differ")
        rebuilt_summaries[arm] = run.summarize(flat, 161)
        audit[arm] = {"normalization_rebuilt": True, "calibration_rebuilt": True,
            "bag_probability_stability_counts_rebuilt": True, "refit_performed": refit,
            "exact_refit_matched": True if refit else None}
    if (not same_tree(rebuilt_summaries, summary["arms"])
            or run.method.select(rebuilt_summaries) != summary["selected_arm_for_possible_confirmation"]):
        raise ValueError("Complete declared statistics/gates/selection differ")
    result = {"schema": "chipmunk.audio-separation-numerical/14", "passed": True,
        "lock_sha256": digest(lock_path), "receipt_sha256": digest(receipt), "head_sha256": digest(head_path),
        "responses_sha256": digest(responses_path), "summary_sha256": digest(summary_path),
        "review_code_sha256": digest(Path(__file__)), "inputs": len(rows), "outcomes": len(records), "arms": audit,
        "scope": "Separate normalization, source/bag predictor, group ranks, existing independent guard/stability and counts; shared declared statistics/gates and exact refit learner. No source forward or outside human review."}
    write(output, result)
    print(json.dumps({"passed": True, "outcomes": len(records), "exact_refits": refit}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "receipt", "head", "responses", "summary", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--refit", action="store_true")
    a = p.parse_args()
    numerical(a.lock, a.receipt, a.head, a.responses, a.summary, a.output, a.refit)
