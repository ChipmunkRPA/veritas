"""Reconstruct fresh source bindings, decision arithmetic, complete counts and gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import expit

from scripts.audio_confirmation_v7 import CONDITIONS, HEAD_SHA256, summarize
from scripts.audio_residual_head_v6 import digest_embedding
from scripts.prepare_audio_confirmation_v7 import digest
from scripts.prepare_audio_guard_v7 import write
from scripts.review_audio_guard_v7 import close, initial, logit, metric, probability


def reconstruct(lock, head, variants, rows):
    if lock["head_sha256"] != HEAD_SHA256 or len(variants) != 1080 or len(rows) != 1080:
        raise ValueError("Fixed head and every planned outcome required")
    if [r["id"] for r in rows] != [r["id"] for r in variants] or len({r["id"] for r in rows}) != 1080:
        raise ValueError("Missing/duplicated/out-of-order confirmation outcomes")
    bundle = lock["serving_bundle"]
    baseline_guard = {str(c): [{"maximum": value} for value in bundle["guards"][str(c)]] for c in (0, 1)}
    raw_direction = np.asarray(lock["representation_header"]["head_weight"], np.float64)
    bias = lock["representation_header"]["head_bias"]
    raw_errors = []
    for expected, row in zip(variants, rows, strict=True):
        for key, value in expected.items():
            if key not in ("path", "production_input") and row.get(key) != value:
                raise ValueError("Outcome source clock/hash/label/condition differs")
        if row["status"] == "failed":
            if any(row[k] is not None for k in ("raw", "embedding", "candidate_p", "original_p")) or any(
                row[k] != "inconclusive" for k in ("candidate_decision", "original_decision")
            ):
                raise ValueError("A failed check cannot establish an original/AI tendency")
            continue
        if row["status"] not in ("ok", "not_applicable"):
            raise ValueError("Invalid terminal source status")
        assessed = row["status"] == "ok"
        expected_windows = [{"start_sample": 0, "end_sample": 64000}] if assessed else []
        if row["windows"] != expected_windows:
            raise ValueError("Applicability source window changed")
        for raw, embedding, embedding_sha in [(row["raw"], row["embedding"], row["embedding_sha256"])]:
            if assessed:
                if digest_embedding(embedding) != embedding_sha:
                    raise ValueError("Original representation digest differs")
                error = abs(float(expit(np.asarray(embedding) @ raw_direction + bias)) - raw)
                raw_errors.append(error)
                if error > 1e-6:
                    raise ValueError("Representation does not reconstruct pretrained response")
            elif raw is not None or embedding is not None:
                raise ValueError("Skipped source cannot have a fabricated representation")
        new_p = probability(row, head["fitted"])
        close(new_p, row["candidate_p"])
        baseline_p = (
            None
            if row["raw"] is None
            else float(
                expit(bundle["parameters"]["slope"] * logit(row["raw"]) + bundle["parameters"]["intercept"])
            )
        )
        close(baseline_p, row["original_p"])
        original_initial = initial(baseline_p, baseline_guard)
        candidate_initial = initial(new_p, head["guards"])
        probe = row["probe"]
        required = original_initial == "original" or candidate_initial == "original"
        allowed_probe_status = ("computed", "failed") if required else ("not_required",)
        if probe["status"] not in allowed_probe_status:
            raise ValueError("Missing or unexpected stability-probe execution")
        band_p = None
        if probe["status"] == "computed":
            if probe["samples"] != 64000 or digest_embedding(probe["embedding"]) != probe["embedding_sha256"]:
                raise ValueError("Probe representation/sample clock differs")
            error = abs(float(expit(np.asarray(probe["embedding"]) @ raw_direction + bias)) - probe["raw"])
            raw_errors.append(error)
            if error > 1e-6:
                raise ValueError("Probe representation does not reconstruct original response")
            band_p = probability({"raw": probe["raw"], "embedding": probe["embedding"]}, head["fitted"])
            close(band_p, probe["p"])
        old_delta = (
            abs(logit(round(row["raw"], 6)) - logit(round(probe["raw"], 6)))
            if original_initial == "original" and probe["status"] == "computed"
            else None
        )
        old_decision = (
            "inconclusive"
            if original_initial == "original" and (old_delta is None or old_delta > math.log(2))
            else original_initial
        )
        new_delta = (
            abs(logit(round(new_p, 6)) - logit(round(band_p, 6)))
            if candidate_initial == "original" and band_p is not None
            else None
        )
        new_decision = (
            "inconclusive"
            if candidate_initial == "original" and (new_delta is None or new_delta > math.log(2))
            else candidate_initial
        )
        if row["original_decision"] != old_decision or row["candidate_decision"] != new_decision:
            raise ValueError("Guard/stability decision arithmetic differs")
        close(row["baseline_detail"]["probability_ai"], baseline_p)
        close(row["baseline_detail"]["stability_log_odds_change"], old_delta)
        close(row["candidate_detail"]["stability_log_odds_change"], new_delta)
        if (
            row["baseline_detail"]["decision"] != old_decision
            or row["candidate_detail"]["proposed"] != candidate_initial
        ):
            raise ValueError("Stored explanation differs from reconstructed outcome")
        if row["baseline_detail"]["scope_supported"] != assessed:
            raise ValueError("Stored scope differs from source applicability")
    result = summarize(rows)
    # Independent complete-denominator arithmetic, separate from shared proper
    # score/bootstrap implementation. This is still local numerical review.
    for condition in ("all", *CONDITIONS):
        panel = [r for r in rows if condition == "all" or r["condition"] == condition]
        for arm, p, decision in (
            ("service", "original_p", "original_decision"),
            ("candidate", "candidate_p", "candidate_decision"),
        ):
            for key, value in metric(panel, p, decision).items():
                actual = result["groups"][condition][arm][key]
                if key in ("brier", "log_loss"):
                    close(value, actual)
                elif actual != value:
                    raise ValueError("Independent operational counts differ")
    result["numerical_review"] = {
        "passed": True,
        "outcomes": len(rows),
        "max_raw_reconstruction_error": max(raw_errors, default=0),
        "scope": "Separate source/representation/probability/guard/stability and count reconstruction; shared declared proper-score/bootstrap/gate implementation. No neural re-execution or outside reviewer.",
    }
    return result


def review(lock_path, scores_path, output):
    from scripts.score_audio_confirmation_v7 import check

    lock = check(lock_path)
    header, *rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
    if header["lock_sha256"] != digest(lock_path) or header["weights"] != lock["weights"]:
        raise ValueError("Response receipt does not match frozen inputs/weights")
    variants = json.loads(Path(lock["variants_path"]).read_text())["variants"]
    head = json.loads(Path(lock["head_path"]).read_text())
    result = reconstruct(lock, head, variants, rows)
    result.update(
        schema="chipmunk.audio-fresh-confirmation-summary/7",
        lock_sha256=digest(lock_path),
        scores_sha256=digest(scores_path),
        limitations=[
            "One fresh short-window comparison, not whole recordings or interval localization.",
            "Known metadata/source disjointness does not establish biometric identity or unknown pretrained overlap.",
            "Conditional intervals exclude fitting/calibration and background-population uncertainty.",
            "No overall-confidence, court-evidence or outside-validation claim.",
        ],
    )
    write(output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("gates", "passes", "execution_status", "probe_status", "numerical_review")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    review(args.lock, args.scores, args.output)
