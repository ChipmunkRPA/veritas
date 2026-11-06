"""Reconstruct full-vector calibration/stability/counts and fitting-only refit.

Shares the published predictor and statistical gates, separately checks class
guards and both stability rules against the frozen original service journal.
This does not re-execute the neural backbone or reserved confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import audio_full_representation_v11 as method
from scripts import run_audio_full_representation_v11 as execution
from scripts.audio_guard_v7 import initial
from scripts.audio_source_identity_v9 import digest
from scripts.evaluate_audio_multisource_v8 import summarize
from scripts.prepare_audio_guard_v7 import write
from scripts.publish_audio_identity_results_v9 import comparable
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.review_audio_conditional_v10 import guards
from scripts.review_audio_guard_v7 import close, metric
from scripts.review_audio_identity_numerics_v9 import decisions
from scripts.run_audio_guard_v7 import load_rows


def reconstruct(head, all_rows, baseline, cached_probes, reference, responses, *, refit=False):
    rows = [r for r in all_rows if r["role"] == "development"]
    if [r["id"] for r in responses] != [r["id"] for r in rows]:
        raise ValueError("Every original development outcome in order required")
    cal = [r for r in all_rows if r["role"] == "cal"]
    if not same_tree(guards(cal, [method.predict(r, head["fitted"]) for r in cal]), head["guards"]):
        raise ValueError("Independent group/class calibration differs")
    computed = []
    for row, source in zip(responses, rows, strict=True):
        if any(
            row[k] != source[k]
            for k in (
                "id",
                "base_id",
                "group",
                "role",
                "label",
                "generator",
                "dataset",
                "condition",
                "sha256",
                "pcm_sha256",
                "background_id",
                "raw",
            )
        ):
            raise ValueError("Original source/truth/clock/raw binding differs")
        if row["status"] == "failed":
            if (
                row["candidate_p"] is not None
                or row["candidate_decision"] != "inconclusive"
                or row["original_p"] is not None
                or row["original_decision"] != "inconclusive"
            ):
                raise ValueError("Failed inputs cannot establish a directional verdict")
            continue
        old = baseline[row["id"]]
        close(row["original_p"], old["original_p"])
        if row["original_decision"] != old["original_decision"] or row["status"] != source["status"]:
            raise ValueError("Original service verdict or assessment status differs")
        p = method.predict(source, head["fitted"])
        close(row["candidate_p"], p)
        needed = initial(p, head["guards"]) == "original"
        probe = row["probe"]
        if probe["status"] not in (("computed", "failed") if needed else ("not_required",)):
            raise ValueError("Missing or unexpected original-stability probe")
        band_p = None
        if probe["status"] == "computed":
            execution.original.validate_feature(
                {**probe, "status": "ok", "windows": [{"start_sample": 0, "end_sample": 64000}]}, reference
            )
            if probe["reused_exact_probe"]:
                cached = cached_probes.get(row["id"])
                if cached is None or any(
                    probe[k] != cached[k]
                    for k in ("raw", "embedding", "embedding_sha256", "filtered_pcm_sha256")
                ):
                    raise ValueError("Exact original probe cache binding differs")
            band_p = method.predict(probe, head["fitted"])
            computed.append(probe)
        close(probe["p"], band_p)
        detail = decisions(p, band_p, source["raw"], probe["raw"], head["guards"])
        if not same_tree(detail, row["candidate_detail"]) or detail["decision"] != row["candidate_decision"]:
            raise ValueError("Independent calibrated/raw stability decision differs")
    refit_difference = None
    if refit:
        fitted = method.fit([r for r in all_rows if r["role"] == "fit"], reference["head_weight"])
        refit_difference = {
            "coefficients": float(
                np.max(np.abs(np.asarray(fitted["coefficients"]) - head["fitted"]["coefficients"]))
            )
        }
        for key in ("unit_head", "center", "scale"):
            refit_difference[key] = float(
                np.max(np.abs(np.asarray(fitted["transform"][key]) - head["fitted"]["transform"][key]))
            )
        if fitted != head["fitted"]:
            raise ValueError("Fitting-only refit must reproduce the complete original object exactly")
    summary = summarize(responses, 161)
    for condition in ("all", *execution.original.CONDITIONS):
        panel = [r for r in responses if condition == "all" or r["condition"] == condition]
        for key, decision, name in (
            ("original_p", "original_decision", "service"),
            ("candidate_p", "candidate_decision", "candidate"),
        ):
            for field, value in metric(panel, key, decision).items():
                actual = summary["groups"][condition][name][field]
                if field in ("brier", "log_loss"):
                    close(value, actual)
                elif value != actual:
                    raise ValueError("Complete-denominator operational count differs")
    return summary, {
        "passed": True,
        "inputs": len(all_rows),
        "outcomes": len(responses),
        "computed_probes": len(computed),
        "new_probes": sum(r["probe"].get("reused_exact_probe") is False for r in responses),
        "refit_performed": refit,
        "refit_max_absolute_difference": refit_difference,
        "scope": "Complete numerical calibration/dual-stability/count reconstruction and optional fitting-only refit. Shared predictor/statistical gates, no new neural forward, reserved confirmation or outside human validation.",
    }


def review(lock_path, head_path, response_path, summary_path, output, refit):
    _, (_, reference, all_rows, baseline, cached) = execution.check(lock_path)
    head = execution.read_head(lock_path, head_path)
    header, responses = load_rows(response_path)
    summary = json.loads(summary_path.read_text())
    if (
        header["lock_sha256"] != digest(lock_path)
        or header["head_sha256"] != digest(head_path)
        or summary["lock_sha256"] != digest(lock_path)
        or summary["head_sha256"] != digest(head_path)
        or summary["responses_sha256"] != digest(response_path)
    ):
        raise ValueError("Complete original head/response/summary lineage required")
    rebuilt, result = reconstruct(head, all_rows, baseline, cached, reference, responses, refit=refit)
    if not same_tree(comparable(rebuilt), comparable(summary)):
        raise ValueError("Every summary statistic and fixed gate must agree")
    result.update(
        schema="chipmunk.audio-full-representation-numerical-review/11",
        lock_sha256=digest(lock_path),
        head_sha256=digest(head_path),
        responses_sha256=digest(response_path),
        summary_sha256=digest(summary_path),
        review_code_sha256=digest(Path(__file__)),
        gates=summary["gates"],
        advance_to_fresh_confirmation=summary["advance_to_fresh_confirmation"],
    )
    write(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "head", "responses", "summary", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--refit", action="store_true")
    a = p.parse_args()
    review(a.lock, a.head, a.responses, a.summary, a.output, a.refit)
