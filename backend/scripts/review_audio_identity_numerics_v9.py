"""Reconstruct fitted guards, both decisions and complete development arithmetic.

Uses the declared frozen learner for optional CPU refitting. Probability, guard,
stability and operational-count arithmetic are separately reconstructed here.
No neural inference, new selection or acceptance-rule change occurs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import audio_multisource_v8 as method
from scripts import run_audio_identity_v9 as execution
from scripts.audio_source_identity_v9 import digest
from scripts.evaluate_audio_multisource_v8 import summarize
from scripts.prepare_audio_guard_v7 import write
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.review_audio_guard_v7 import close, initial, logit, metric, probability
from scripts.run_audio_guard_v7 import load_rows


def guards(rows, fitted):
    if not rows or any(r["role"] != "cal" for r in rows):
        raise ValueError("Only separate calibration sources can set guard ranks")
    pools = {0: defaultdict(list), 1: defaultdict(list)}
    for row in rows:
        p = probability(row, fitted)
        pools[row["label"]][row["group"]].append(None if p is None else p if row["label"] == 0 else 1 - p)
    return {
        str(c): [
            {
                "group": group,
                "maximum": None if any(p is None for p in values) else max(values),
                "rows": len(values),
            }
            for group, values in sorted(pools[c].items())
        ]
        for c in (0, 1)
    }


def decisions(p, band_p, raw, band_raw, candidate_guards):
    proposed = initial(p, candidate_guards)
    delta_p, delta_raw = None, None
    if proposed == "original":
        if p is not None and band_p is not None:
            delta_p = abs(logit(round(p, 6)) - logit(round(band_p, 6)))
        if raw is not None and band_raw is not None:
            delta_raw = abs(logit(round(raw, 6)) - logit(round(band_raw, 6)))
    veto = proposed == "original" and (
        delta_p is None or delta_raw is None or max(delta_p, delta_raw) > math.log(2)
    )
    return {
        "proposed": proposed,
        "decision": "inconclusive" if veto else proposed,
        "calibrated_stability_log_odds_change": delta_p,
        "raw_stability_log_odds_change": delta_raw,
        "original_stability_veto": veto,
    }


def reconstruct(reference, head, rows, outcomes, bundle, *, refit=False, draws=2000):
    if Counter(r["role"] for r in rows) != execution.COUNTS or len({r["id"] for r in rows}) != 4248:
        raise ValueError("Every corrected consumed representation required")
    if head["fitted"]["method"] != method.METHOD:
        raise ValueError("Unchanged declared learner required")
    for row in rows:
        execution.validate_feature(row, reference)
    expected_guards = guards([r for r in rows if r["role"] == "cal"], head["fitted"])
    if not same_tree(expected_guards, head["guards"]):
        raise ValueError("Independent calibration population or guard ranks differ")
    refit_error = None
    if refit:
        fitted = method.fit([r for r in rows if r["role"] == "fit"], reference["head_weight"])
        if not same_tree(fitted, head["fitted"]):
            raise ValueError("CPU refit differs from the original fitting-only artifact")
        refit_error = float(
            np.max(np.abs(np.asarray(fitted["coefficients"]) - np.asarray(head["fitted"]["coefficients"])))
        )
    development = [r for r in rows if r["role"] == "development"]
    if [r["id"] for r in outcomes] != [r["id"] for r in development]:
        raise ValueError("Every development outcome must remain in original order")
    baseline_guards = {c: [{"maximum": p} for p in bundle["guards"][c]] for c in ("0", "1")}
    for expected, row in zip(development, outcomes, strict=True):
        for key in (
            "sha256",
            "pcm_sha256",
            "role",
            "base_id",
            "group",
            "condition",
            "generator",
            "dataset",
            "label",
            "background_id",
        ):
            if row[key] != expected[key]:
                raise ValueError("Outcome source/truth/group/background binding differs")
        if row["status"] == "failed":
            if any(row[k] is not None for k in ("candidate_p", "original_p")) or any(
                row[k] != "inconclusive" for k in ("candidate_decision", "original_decision")
            ):
                raise ValueError("A failed outcome cannot establish a directional conclusion")
            continue
        if row["status"] != expected["status"] or row["raw"] != expected["raw"]:
            raise ValueError("Assessed outcome status or original response changed")
        p = probability(expected, head["fitted"])
        baseline_p = (
            None
            if row["raw"] is None
            else float(
                1
                / (
                    1
                    + math.exp(
                        -bundle["parameters"]["slope"] * logit(row["raw"]) - bundle["parameters"]["intercept"]
                    )
                )
            )
        )
        close(p, row["candidate_p"])
        close(baseline_p, row["original_p"])
        original_initial, candidate_initial = initial(baseline_p, baseline_guards), initial(p, head["guards"])
        required = original_initial == "original" or candidate_initial == "original"
        probe = row["probe"]
        if probe["status"] not in (("computed", "failed") if required else ("not_required",)):
            raise ValueError("Missing or unexpected original-stability probe")
        band_p = None
        if probe["status"] == "computed":
            execution.validate_feature(
                {**probe, "status": "ok", "windows": [{"start_sample": 0, "end_sample": 64000}]}, reference
            )
            band_p = probability(probe, head["fitted"])
            close(band_p, probe["p"])
        candidate = decisions(p, band_p, row["raw"], probe["raw"], head["guards"])
        if (
            not same_tree(candidate, row["candidate_detail"])
            or candidate["decision"] != row["candidate_decision"]
        ):
            raise ValueError("Candidate dual-stability conclusion or explanation differs")
        delta = (
            abs(logit(round(row["raw"], 6)) - logit(round(probe["raw"], 6)))
            if original_initial == "original" and probe["status"] == "computed"
            else None
        )
        baseline_decision = (
            "inconclusive"
            if original_initial == "original" and (delta is None or delta > math.log(2))
            else original_initial
        )
        if (
            row["original_decision"] != baseline_decision
            or row["baseline_detail"]["decision"] != baseline_decision
        ):
            raise ValueError("Actual-service raw-stability conclusion differs")
        close(row["baseline_detail"]["probability_ai"], baseline_p)
        close(row["baseline_detail"]["stability_log_odds_change"], delta)
        if row["baseline_detail"]["scope_supported"] != (expected["status"] == "ok"):
            raise ValueError("Unassessed speech cannot become supported evidence")
        if row["probe_needed_by_candidate"] != (candidate_initial == "original"):
            raise ValueError("Candidate probe requirement differs")
    summary = summarize(outcomes, 161, draws=draws)
    for condition in ("all", *execution.CONDITIONS):
        panel = [r for r in outcomes if condition == "all" or r["condition"] == condition]
        for arm, key, decision_key in (
            ("service", "original_p", "original_decision"),
            ("candidate", "candidate_p", "candidate_decision"),
        ):
            for field, value in metric(panel, key, decision_key).items():
                actual = summary["groups"][condition][arm][field]
                if field in ("brier", "log_loss"):
                    close(value, actual)
                elif value != actual:
                    raise ValueError("Separate complete-denominator operational counts differ")
    return summary, {
        "passed": True,
        "representations": len(rows),
        "outcomes": len(outcomes),
        "refit_performed": refit,
        "refit_max_coefficient_error": refit_error,
        "scope": "Separate probability/guard/dual-stability/count reconstruction; declared learner shared for optional CPU refit, declared proper-score/bootstrap/gates shared. No neural forward or outside reviewer.",
    }


def review(lock_path, receipt, head_path, responses, summary_path, output, refit):
    lock, _ = execution.check(lock_path)
    _, reference, rows = execution.feature_rows(lock_path, receipt)
    head = json.loads(head_path.read_text())
    fit_record = json.loads((head_path.parent / "fit.json").read_text())
    header, outcomes = load_rows(responses)
    summary = json.loads(summary_path.read_text())
    if (
        header["lock_sha256"] != digest(lock_path)
        or header["receipt_sha256"] != digest(receipt)
        or header["head_sha256"] != digest(head_path)
        or head["lock_sha256"] != digest(lock_path)
        or head["receipt_sha256"] != digest(receipt)
        or head["fit_sha256"] != digest(head_path.parent / "fit.json")
        or summary["responses_sha256"] != digest(responses)
        or summary["head_sha256"] != digest(head_path)
        or summary["receipt_sha256"] != digest(receipt)
        or summary["lock_sha256"] != digest(lock_path)
        or fit_record["lock_sha256"] != digest(lock_path)
        or fit_record["receipt_sha256"] != digest(receipt)
        or not same_tree(fit_record["fitted"], head["fitted"])
    ):
        raise ValueError("Original execution, fitting, outcome or summary lineage differs")
    rebuilt, result = reconstruct(reference, head, rows, outcomes, lock["serving_bundle"], refit=refit)
    if summary["schema"] != "chipmunk.audio-identity-development-comparison/9" or any(
        not same_tree(rebuilt[k], summary[k]) for k in rebuilt if k != "schema"
    ):
        raise ValueError("Complete published statistical summary differs")
    result.update(
        schema="chipmunk.audio-identity-numerical-review/9",
        review_code_sha256=digest(Path(__file__)),
        lock_sha256=digest(lock_path),
        receipt_sha256=digest(receipt),
        head_sha256=digest(head_path),
        responses_sha256=digest(responses),
        summary_sha256=digest(summary_path),
        gates_passed=sum(rebuilt["gates"].values()),
        advance_to_fresh_confirmation=rebuilt["advance_to_fresh_confirmation"],
    )
    write(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("responses", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    review(args.lock, args.receipt, args.head, args.responses, args.summary, args.output, args.refit)
