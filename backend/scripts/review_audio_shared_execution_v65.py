"""Additive execution, repeatability and bootstrap review; no neural imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(value, reason):
    if not value:
        raise ValueError(reason)


def bootstrap_review(rows, configuration):
    groups = sorted({row["group"] for row in rows})
    matrix = []
    for row in rows:
        truth = "ai" if row["label"] else "original"
        matrix.append({name: int(value == truth) - int(row["decisions"]["current"] == truth)
                       for name, value in row["decisions"].items()})
    group_totals = {group: {name: sum(matrix[index][name] for index, row in enumerate(rows) if row["group"] == group)
                            for name in matrix[0]} for group in groups}
    rng = np.random.default_rng(configuration["seed"])
    draws = rng.integers(len(groups), size=(configuration["replicates"], len(groups)))
    # Independent histogram-weighted implementation, rather than indexing and
    # summing the first review's vector of sampled group deltas.
    multiplicities = np.asarray([np.bincount(draw, minlength=len(groups)) for draw in draws])
    return {name: np.quantile(multiplicities @ [group_totals[group][name] for group in groups], [.025, .975]).tolist()
            for name in matrix[0]}


def review(trial):
    root = Path(__file__).resolve().parents[2]
    frozen = read(trial / "freeze.json")
    require(all(sha(root / path) == expected for path, expected in frozen["code_sha256"].items()),
            "Frozen code changed")
    require(all(sha(root / path) == expected for path, expected in frozen["artifacts_sha256"].items()),
            "Frozen source/fitting artifact changed")
    result = read(trial / "result.json")
    set_review = read(trial / "set-review.json")
    require(set_review["all_totals_strata_and_actual_gains_losses_exact"] is True
            and set_review["result_sha256"] == sha(trial / "result.json"), "Bound exact set-algebra review required")
    profiles = {"prepare": trial.parent / (trial.name + "-preparation-profile/profile.json"),
                **{f"chunk-{n:02d}": trial / f"chunk-{n:02d}-profile/profile.json" for n in range(12)},
                "evaluate": trial / "evaluation-profile/profile.json"}
    observed = {}
    for name, path in profiles.items():
        profile = read(path)
        require(profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0
                and not profile["resource_limit_observations"], "Every explicit bounded stage must pass")
        require(sha(path.parent / "resource-samples.jsonl") == profile["resource_journal_sha256"],
                "Resource sample journal changed")
        observed[name] = profile
    rows = []
    for n in range(12):
        path = trial / f"chunk-{n:02d}.jsonl"
        require(sha(path) == result["journal_sha256"][path.name], "Scoring journal changed")
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    require(len(rows) == len({row["id"] for row in rows}) == 1080, "Complete original case ordering required")
    intervals = bootstrap_review(rows, frozen["bootstrap"])
    require(intervals == result["paired_group_bootstrap_95pct_net_correct_count_change"],
            "Paired group bootstrap differs")
    old = trial.parent / "audio-shared-confirmation-v62"
    require(sha(old / "chunk-00.jsonl") == frozen["original_failed_chunk_sha256"]
            and sha(old / "chunk-00-profile/profile.json") == frozen["original_failed_profile_sha256"],
            "Original stopped attempt changed")
    prefix = [json.loads(line) for line in (old / "chunk-00.jsonl").read_text().splitlines()]
    require(len(prefix) == frozen["prior_processed_cases"] == 33, "Complete original prefix required")
    for original, repeated in zip(prefix, rows, strict=False):
        require({key: value for key, value in original.items() if key != "runtime_s"}
                == {key: value for key, value in repeated.items() if key != "runtime_s"},
                "Repeated original prefix score/feature/decision changed")
    calibration = root / "backend/app/detectors/audio/calibrations/local-speech-v3.json"
    require(sha(calibration) == "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf",
            "Protected service calibration changed")
    return {"schema": "chipmunk.audio-shared-execution-review/65", "status": "complete",
            "freeze_sha256": sha(trial / "freeze.json"), "result_sha256": sha(trial / "result.json"),
            "set_review_sha256": sha(trial / "set-review.json"), "reviewer_sha256": sha(Path(__file__)),
            "code_bindings_reverified": len(frozen["code_sha256"]), "inputs": len(rows),
            "passed_bounded_stages": len(observed), "profiles_sha256": {key: sha(path) for key, path in profiles.items()},
            "all_histogram_bootstrap_intervals_exact": True,
            "original_stopped_prefix_exact_except_runtime": len(prefix),
            "original_attempt_remains_failed": True, "possible_in_flight_original_inputs": 1,
            "sampled_tree_rss_lower_bound_bytes": max(p["sampled_tree_rss_lower_bound_bytes"] for p in observed.values()),
            "minimum_observed_available_ram_bytes": min(p["minimum_observed_available_ram_bytes"] for p in observed.values()),
            "minimum_observed_free_disk_bytes": min(p["minimum_observed_free_disk_bytes"] for p in observed.values()),
            "total_bounded_stage_duration_s": sum(p["duration_s"] for p in observed.values()),
            "verified_descendant_exit_events": sum(event["classification"].startswith("verified_descendant_exit")
                                                    for p in observed.values() for event in p["terminal_observations"]),
            "protected_calibration_sha256": sha(calibration), "new_model_calls": 0,
            "scope": "Independent local arithmetic/execution review, not an outside reviewer. Technical repetition of a panel fixed before its first calls, not a new fresh cohort. Shared backgrounds limit group intervals; sampled RSS is not a true peak."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = review(args.trial)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)
