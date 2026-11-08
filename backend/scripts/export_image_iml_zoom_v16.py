"""Publish the rejected image-crop result with its strict audit failure intact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCIENCE = "d1165ffe510eabb38355d0511e52388bea0097a1"
COUNT_REVIEW = "910a9502179f625d7861a23216bc8e84eab99ac4"
PUBLIC_FREEZE = "202c8a51784d86519322499f9941dafa68b9579ab3207b7ad08a57146e2be522"
COUNT_REVIEW_SHA = "003728fca2b2f597436e6ca263620e1a321a8a08"
BOUNDARY_FREEZE = "b4a7fce06fe113216f5b3848b283ebbd172a871f496dea8b44eed141461b173e"
FORMATS = ("native", "jpeg75")
FAMILIES = ("firefly", "sd2", "flux-fill")


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def export(trial):
    public_path = ROOT / "docs/evaluation-panels/image-iml-zoom-v16-freeze.json"
    require(sha(public_path) == PUBLIC_FREEZE, "Published prospective freeze changed")
    public, frozen = read(public_path), read(trial / "freeze.json")
    require(sha(trial / "freeze.json") == public["private_freeze_sha256"], "Private freeze changed")
    for name, value in frozen["code_sha256"].items():
        require(sha(ROOT / name) == value, "Frozen scientific code changed: " + name)
    count_path = ROOT / "backend/scripts/review_image_iml_zoom_counts_v16.py"
    # Git blob hashing binds the separate audit published before diagnostics.
    data = count_path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    require(blob == COUNT_REVIEW_SHA, "Prospectively published count audit changed")
    seal = read(trial / "point-seal.json")
    require(seal["diagnostic_processed"] is False, "Genuine points were not prospectively sealed")
    for name, value in seal["sha256"].items():
        require(sha(trial / name) == value, "Point seal changed")

    phases = ["point-" + fmt for fmt in FORMATS]
    phases += [f"diag-{family}-{fmt}" for family in FAMILIES for fmt in FORMATS]
    calls, cases, grid_count = 0, 0, 0
    reductions = []
    repeats = 0
    for phase in phases:
        complete = read(trial / (phase + "-complete.json"))
        journal = trial / (phase + ".jsonl")
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        expected_calls = 219 if phase.startswith("point-") else 211
        require(
            complete["planned"] == complete["complete"] == len(rows) == 48
            and complete["failed"] == 0
            and complete["attempted_native_forwards"]
            == complete["successful_native_forwards"]
            == expected_calls
            and complete["journal_sha256"] == sha(journal)
            and complete["freeze_sha256"] == sha(trial / "freeze.json"),
            "Incomplete or changed native phase: " + phase,
        )
        require(
            len({r["id"] for r in rows}) == 48 and all(r["status"] == "ok" for r in rows),
            "Missing, duplicate or failed input",
        )
        require(all(len(r["tile_proofs"]) == 4 for r in rows), "Missing crop forward proof")
        require(
            rows[0]["full_proof"]["native_exact_repeat_and_recorded_grid_parity"] is True,
            "First-input native repeat/parity is missing",
        )
        reductions.extend(r["reduction_max_difference"] for r in rows)
        repeats += 1
        calls += complete["successful_native_forwards"]
        cases += len(rows)
        grid_count += len(rows) * 5

    reference_calls = 0
    for fmt in FORMATS:
        phase = "reference-" + fmt
        complete = read(trial / (phase + "-complete.json"))
        journal = trial / (phase + ".jsonl")
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        require(
            complete["planned"] == complete["complete"] == len(rows) == 48
            and complete["successful_native_forwards"] == 49
            and complete["journal_sha256"] == sha(journal)
            and complete["freeze_sha256"] == sha(trial / "freeze.json"),
            "Complete actual ordinary-reference phase required",
        )
        require(
            len({r["id"] for r in rows}) == 48 and all(r["status"] == "ok" for r in rows),
            "Reference input missing or failed",
        )
        reference_calls += complete["successful_native_forwards"]

    result = read(trial / "result.json")
    review = read(trial / "boundary-sensitivity-review.json")
    counts = read(trial / "separate-count-review.json")
    result_sha = sha(trial / "result.json")
    require(
        review["result_sha256"] == counts["result_sha256"] == result_sha, "Audits must bind the exact result"
    )
    require(
        review["source_file_cases"] == cases == 384
        and review["recorded_uint8_grids"] == grid_count == 1920
        and review["canonical_diagnostic_arm_metrics_exact"] == counts["arm_metric_records"] == 864
        and counts["file_cases"] == result["file_cases"] == 288
        and counts["source_groups"] == result["source_groups"] == 48,
        "Complete audit denominators required",
    )
    require(
        review["all_canonical_components_and_metrics_exact"] is True
        and review["all_sensitivity_replacement_decisions_rejected"] is True
        and review["original_strict_reconstruction"] == "failed; unchanged"
        and counts["all_counts_families_gates_and_bootstraps_match"] is True,
        "Canonical/count replay must pass while the original strict failure remains explicit",
    )
    require(
        counts["reviewer_sha256"] == sha(count_path)
        and review["reviewer_sha256"] == sha(ROOT / "backend/scripts/review_image_iml_zoom_boundaries_v16.py")
        and review["diagnostic_freeze_sha256"] == BOUNDARY_FREEZE
        and sha(ROOT / "docs/evaluation-panels/image-iml-zoom-v16-boundary-audit-freeze.json")
        == BOUNDARY_FREEZE,
        "Review code differs",
    )
    require(
        max(reductions) <= 1e-12 and review["maximum_independent_continuous_map_difference"] <= 1e-12,
        "Declared arithmetic bound failed",
    )
    require(
        result["serving_admitted"] is result["used_in_confidence"] is False,
        "Research result cannot imply release",
    )
    require(
        not any(v["qualifies_fresh_confirmation"] for v in result["formats"].values()),
        "A qualifying candidate cannot be published through a failed-strict-audit rejection exporter",
    )
    require(not (trial / "complete-review.json").exists(), "Original strict failure must not be replaced")
    require(
        review["original_strict_profile_sha256"] == sha(trial / "review-profile/profile.json")
        and review["original_strict_log_sha256"] == sha(trial / "review-profile/worker.log"),
        "Original failed audit changed",
    )
    require(
        sha(trial / "private-outcomes.json") == result["private_outcomes_sha256"],
        "Original complete scalar packet changed",
    )

    profiles = {}
    for phase in [
        *phases,
        "seal",
        "reference-native",
        "reference-jpeg75",
        "summary",
        "review",
        "boundary-review",
    ]:
        profile = read(trial / (phase + "-profile") / "profile.json")
        require(
            profile["child_exit_code"] == (1 if phase == "review" else 0) and profile["guard"] is None,
            "Failed/guarded execution cannot be hidden: " + phase,
        )
        profiles[phase] = {
            k: profile[k]
            for k in (
                "child_exit_code",
                "guard",
                "duration_s",
                "sampled_tree_rss_lower_bound_bytes",
                "sample_errors",
                "seconds_limit",
            )
        }
    linux = read(ROOT / "docs/evaluation-panels/image-iml-zoom-v16-linux-math-review.json")
    require(
        linux["workflow_head_sha"] == "28d8a56253db82aa2689999f475c53598ab323f8"
        and linux["run_conclusion"] == linux["job_conclusion"] == "success"
        and linux["review"]["science_commit"] == SCIENCE,
        "Observed separate-host arithmetic receipt required",
    )
    return {
        **result,
        "science_commit": SCIENCE,
        "separate_count_review_commit": COUNT_REVIEW,
        "public_freeze_sha256": PUBLIC_FREEZE,
        "private_result_sha256": result_sha,
        "publication_state": "rejected_development_with_failed_strict_reconstruction",
        "strict_reconstruction_passed": False,
        "private_complete_review_sha256": None,
        "private_boundary_sensitivity_review_sha256": sha(trial / "boundary-sensitivity-review.json"),
        "private_separate_count_review_sha256": sha(trial / "separate-count-review.json"),
        "development_only": True,
        "fresh_confirmation": False,
        "point_source_groups": 24,
        "all_source_groups": 72,
        "total_candidate_file_cases": cases,
        "recorded_grids": grid_count,
        "actual_native_iml_forwards": calls,
        "actual_current_reference_forwards": reference_calls,
        "first_input_native_repeat_and_old_grid_parities": repeats,
        "maximum_inline_reduction_difference": max(reductions),
        "genuine_only_points": read(trial / "points.json"),
        "complete_review": None,
        "post_result_boundary_sensitivity_review": review,
        "separate_count_review": counts,
        "separate_host_math_review": linux,
        "resource_profiles": profiles,
        "resource_scope": "Sampled process-tree RSS is a lower bound on peak use; every sampling error is retained.",
        "private_data_published": False,
        "public_scope": "Methods, commitments and aggregate counts only. No photos, source pairings, grids or per-case outcomes.",
        "interpretation": "Both formats rejected on consumed-source development. Original exact independent-mask audit remains failed; post-result canonical replay and numerical sensitivity do not clear it. No fresh accuracy, customer confidence or deployment claim.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = export(args.trial)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "written": str(args.output),
                "formats": {k: v["qualifies_fresh_confirmation"] for k, v in result["formats"].items()},
            }
        )
    )
