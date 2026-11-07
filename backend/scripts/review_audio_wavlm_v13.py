"""Additive native/numerical audit; cannot change the frozen experiment."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit

from scripts import audio_wavlm_features_v13 as features
from scripts import run_audio_wavlm_v13 as run
from scripts.audio_multisource_v8 import balanced_weights
from scripts.audio_source_identity_v9 import digest
from scripts.audit_audio_identity_forwards_v9 import same_native
from scripts.prepare_audio_guard_v7 import write
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.review_audio_guard_v7 import close, logit
from scripts.review_audio_identity_numerics_v9 import decisions
from scripts.run_audio_guard_v7 import capture, load_rows
from scripts.run_audio_multisource_v8 import input_pcm


def probability(row, fitted):
    if row["raw"] is None:
        if row.get("wavlm") is not None:
            raise ValueError("Unassessed source cannot supply a candidate probability")
        return None
    if isinstance(row["raw"], bool) or not math.isfinite(row["raw"]) or not 0 <= row["raw"] <= 1:
        raise ValueError("Finite original raw probability required")
    vector = np.asarray(row["wavlm"], np.float64)
    t = fitted["transform"]
    center, scale, coefficients = [np.asarray(x, np.float64) for x in
                                   (t["center"], t["scale"], fitted["coefficients"])]
    arm = fitted["arm"]
    if arm not in run.method.ARMS:
        raise ValueError("Both declared arms only")
    dimension = features.DIMENSIONS
    expected = dimension + (2 if arm == "complement" else 1)
    if (vector.shape != (dimension,) or center.shape != vector.shape or scale.shape != vector.shape
            or coefficients.shape != (expected,) or not np.isfinite(vector).all()
            or not np.isfinite(center).all() or not np.isfinite(scale).all() or not np.isfinite(coefficients).all()
            or (scale < 1e-6).any() or t["variance_factor"] != math.sqrt(16 / dimension)):
        raise ValueError("Complete finite fixed-dimensional fitted mapping required")
    standardized = (vector - center) / scale * t["variance_factor"]
    offset = 1 if arm == "complement" else 0
    margin = standardized @ coefficients[offset:-1] + coefficients[-1]
    if offset:
        if coefficients[0] < 0:
            raise ValueError("Explicit original coefficient cannot be negative")
        margin += logit(row["raw"]) * coefficients[0]
    return float(expit(margin))


def numerical(lock_path, receipt, head_path, responses_path, summary_path, output, refit):
    _, (_, _, _, baseline, _), rows = run.feature_rows(lock_path, receipt)
    head = run.read_head(lock_path, receipt, head_path)
    header, responses = load_rows(responses_path)
    summary = json.loads(summary_path.read_text())
    development = [r for r in rows if r["role"] == "development"]
    if (header["lock_sha256"] != digest(lock_path) or header["receipt_sha256"] != digest(receipt)
            or header["head_sha256"] != digest(head_path) or summary["responses_sha256"] != digest(responses_path)
            or summary["head_sha256"] != digest(head_path) or summary["receipt_sha256"] != digest(receipt)
            or summary["lock_sha256"] != digest(lock_path)
            or [r["id"] for r in responses] != [r["id"] for r in development]):
        raise ValueError("All1449 bound original outcomes and fitting lineage required")
    fitting = [r for r in rows if r["role"] == "fit" and r["raw"] is not None]
    calibration = [r for r in rows if r["role"] == "cal"]
    weights = balanced_weights(fitting)
    matrix = np.asarray([r["wavlm"] for r in fitting], np.float64)
    center = weights @ matrix
    scale = np.maximum(np.sqrt(weights @ (matrix - center)**2), 1e-6)
    audit, rebuilt_summaries = {}, {}
    for arm in run.method.ARMS:
        fitted = head["fitted"][arm]
        if not same_tree(fitted["transform"], {"center": center.tolist(), "scale": scale.tolist(),
                            "variance_factor": math.sqrt(16 / matrix.shape[1]),
                            "fit_sources": len({r["base_id"] for r in fitting})}):
            raise ValueError("Fitting-only representation normalization differs")
        if refit and run.method.fit([r for r in rows if r["role"] == "fit"], arm) != fitted:
            raise ValueError("Exact fitting-only object refit differs")
        # Independently group each truth pool; missing variants contaminate the whole group.
        rebuilt = {}
        for label in (0, 1):
            pool = []
            for group in sorted({r["group"] for r in calibration if r["label"] == label}):
                values = [probability(r, fitted) for r in calibration if r["label"] == label and r["group"] == group]
                maximum = None if None in values else max(v if label == 0 else 1-v for v in values)
                pool.append({"group": group, "maximum": maximum, "rows": len(values)})
            rebuilt[str(label)] = pool
        if not same_tree(rebuilt, head["guards"][arm]):
            raise ValueError("Independent calibration group/truth ranks differ")
        counts, flat = Counter(), []
        for row, source in zip(responses, development, strict=True):
            if any(row[k] != source[k] for k in run.prior.META) or row["status"] != source["status"]:
                raise ValueError("Source, role, truth, group or assessment binding differs")
            old = baseline[source["id"]]
            if any(row[k] != old[k] for k in ("original_p", "original_decision")):
                raise ValueError("Actual service baseline differs")
            p = probability(source, fitted)
            probe = row["probe"]
            band = probability(probe, fitted) if probe["status"] == "computed" else None
            expected = decisions(p, band, source["raw"], probe["raw"], rebuilt)
            candidate = row["candidates"][arm]
            close(candidate["p"], p)
            if candidate["decision"] != expected["decision"] or not same_tree(candidate["detail"], expected):
                raise ValueError("Independent raw/mapped stability or outcome differs")
            counts[candidate["decision"]] += 1
            if candidate["decision"] == "ai":
                counts["correct_ai" if source["label"] == 1 else "false_ai"] += 1
            if candidate["decision"] == "original":
                counts["correct_original" if source["label"] == 0 else "false_original"] += 1
            flat.append({**row, "candidate_p": p, "candidate_decision": expected["decision"]})
        reported = summary["arms"][arm]["groups"]["all"]["candidate"]
        if (any(counts[k] != reported[k] for k in ("correct_ai", "correct_original", "false_ai", "false_original"))
                or any(counts[k] != reported["decisions"].get(k, 0) for k in ("ai", "original", "inconclusive"))):
            raise ValueError("Complete independent operational counts differ")
        rebuilt_summaries[arm] = run.summarize(flat, 161)
        audit[arm] = {"normalization_rebuilt": True, "calibration_rebuilt": True,
                      "probability_stability_counts_rebuilt": True, "refit_performed": refit,
                      "exact_refit_matched": True if refit else None}
    if (not same_tree(rebuilt_summaries, summary["arms"])
            or run.method.select(rebuilt_summaries) != summary["selected_arm_for_possible_confirmation"]):
        raise ValueError("Complete statistics, unchanged gates or selected arm differ")
    write(output, {"schema": "chipmunk.wavlm-v13-numerical-review/1", "passed": True,
                   "lock_sha256": digest(lock_path), "receipt_sha256": digest(receipt),
                   "head_sha256": digest(head_path), "responses_sha256": digest(responses_path),
                   "summary_sha256": digest(summary_path), "review_code_sha256": digest(Path(__file__)),
                   "inputs": len(rows), "outcomes": len(responses), "arms": audit,
                   "scope": "Separate predictor/normalization/group ranks/stability/counts; shared declared statistics/bootstrap/gates and exact refit learner. No neural inference or outside human review."})
    print(json.dumps({"passed": True, "outcomes": len(responses), "exact_refits": refit}), flush=True)


def native(lock_path, receipt, head_path, responses_path, output):
    lock, (body, reference, _, _, _), rows = run.feature_rows(lock_path, receipt)
    run.read_head(lock_path, receipt, head_path)
    header, responses = load_rows(responses_path)
    if (header["lock_sha256"] != digest(lock_path) or header["head_sha256"] != digest(head_path)
            or header["receipt_sha256"] != digest(receipt)
            or [r["id"] for r in responses] != [r["id"] for r in rows if r["role"] == "development"]):
        raise ValueError("Bound complete original development responses required")
    probes = {r["id"]: r["probe"] for r in responses if r["probe"]["status"] == "computed"}
    ids = sorted(set(lock["native_ids"]) | set(probes))
    source = {r["id"]: r for r in rows}
    candidate = features.load(lock["weights_root"])
    original, activity, captured, hook = run.prior.load_models(reference)
    cases = []
    try:
        for index, identifier in enumerate(ids):
            row = source[identifier]
            pcm = input_pcm(body, row)
            old = run.prior.prior.original.extract_one(pcm, original, activity, captured)
            if not same_native(old, row):
                raise ValueError("Original native status/raw/embedding/VAD/window changed")
            if row["status"] == "ok":
                replay = features.extract(pcm, candidate)
                features.validate({**replay, "status": "ok"}, row)
                if any(replay[k] != row[k] for k in replay):
                    raise ValueError("Native original-source WavLM feature replay differs")
            if identifier in probes:
                band = run.band_view(pcm)
                replay = features.extract(band, candidate)
                if any(replay[k] != probes[identifier][k] for k in replay):
                    raise ValueError("Native filtered candidate feature replay differs")
                raw, _ = capture(original, band, captured)
                if raw != probes[identifier]["raw"]:
                    raise ValueError("Native original filtered raw replay differs")
            cases.append({"id": identifier, "sha256": row["sha256"], "pcm_sha256": row["pcm_sha256"],
                          "predetermined_cell": identifier in lock["native_ids"], "computed_probe": identifier in probes,
                          "passed": True})
            if (index + 1) % 25 == 0 or index + 1 == len(ids):
                print(json.dumps({"native_completed": index + 1, "planned": len(ids)}), flush=True)
    finally:
        hook.remove()
    write(output, {"schema": "chipmunk.wavlm-v13-native-review/1", "passed": True,
                   "lock_sha256": digest(lock_path), "receipt_sha256": digest(receipt),
                   "responses_sha256": digest(responses_path), "review_code_sha256": digest(Path(__file__)),
                   "predetermined_cells": 99, "computed_probes": len(probes), "unique_cases": len(ids), "cases": cases,
                   "scope": "Actual frozen CPU source and every required filtered-probe replay; not fresh accuracy or outside review"})


if __name__ == "__main__":
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("native", "numerical"))
    p.add_argument("lock", type=Path)
    p.add_argument("receipt", type=Path)
    p.add_argument("head", type=Path)
    p.add_argument("responses", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--summary", type=Path)
    p.add_argument("--refit", action="store_true")
    a = p.parse_args()
    if a.stage == "native":
        native(a.lock, a.receipt, a.head, a.responses, a.output)
    else:
        numerical(a.lock, a.receipt, a.head, a.responses, a.summary, a.output, a.refit)
