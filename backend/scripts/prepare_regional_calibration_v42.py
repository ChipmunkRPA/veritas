"""Audit the real v26-prefix/v41-suffix chain before unchanged v28 calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import prepare_regional_calibration_v29 as original
from scripts.image_regional_continuation_v41 import audit_continuation, check
from scripts.image_regional_inference_v28 import calibration_tasks, read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
PREPARATION = ROOT / "docs/evaluation-panels/image-regional-calibration-v42-preparation.json"
READOUT30 = ROOT / "docs/evaluation-panels/image-regional-calibration-readout-v30-preparation.json"
ASSETS = original.ASSETS
runtime_versions = original.runtime_versions
OWN_FILES = (
    "backend/scripts/prepare_regional_calibration_v42.py",
    "backend/scripts/regional_calibration_runtime_v42.py",
    "backend/scripts/replay_regional_calibration_v43.py",
    "backend/tests/test_prepare_regional_calibration_v42.py",
    "backend/tests/test_replay_regional_calibration_v43.py",
    "docs/image-regional-calibration-v42-v43-protocol.md",
)


def implementations(study):
    plan = check(study)
    frozen = read(study / "freeze.json")
    packet = read(PREPARATION)
    require(packet["continuation_plan_sha256"] == sha(study / "plan.json") and packet["continuation_freeze_sha256"] == sha(study / "freeze.json"), "Preparation belongs to another continuation")
    mappings = (
        original.implementations(Path(plan["original_study"])),
        frozen["code_sha256"],
        read(READOUT30)["implementation_sha256"],
        packet["implementation_sha256"],
    )
    require(set(mappings[-1]) == set(OWN_FILES), "Complete new audit/monitor/readout implementation set required")
    expected = {}
    for mapping in mappings:
        for path, digest in mapping.items():
            require(path not in expected or expected[path] == digest, "Conflicting frozen implementation hashes")
            require(sha(ROOT / path) == digest, "Frozen comparison implementation changed")
            expected[path] = digest
    expected.update({str(path.relative_to(ROOT)): sha(path) for path in (PREPARATION, READOUT30)})
    return expected


def prepare(study, trial, raw_map_root):
    study, trial, raw_map_root = study.resolve(), trial.resolve(), raw_map_root.resolve()
    require(not trial.exists(), "A new audit/supplement destination is required")
    require(
        raw_map_root.is_dir() and not any(raw_map_root.iterdir()),
        "A private empty map destination must already exist",
    )
    # Fast refusal avoids reading gigabytes of old checkpoints during training.
    require(
        (study / "phase-039/completed.json").is_file()
        and (study / "phase-039-profile/profile.json").is_file(),
        "Final training phase has not completed; no reserved inference is admitted",
    )
    code = implementations(study)
    chain = audit_continuation(study)
    require(
        chain["completed_updates"] == 1278 and len(chain["phases"]) == 40,
        "Complete fixed training chain required",
    )
    plan = read(study / "plan.json")
    tasks = calibration_tasks(plan)
    require(len(tasks) == 156, "Every reserved calibration view required")
    require(
        all(sha(ROOT / path) == value for path, value in ASSETS.items()), "A pinned comparison asset changed"
    )
    versions = runtime_versions()
    trial.mkdir(parents=True, exist_ok=False)
    paths = {
        "study": str(study),
        "raw_map_root": str(raw_map_root),
        "preparation_profile": str(trial.parent / (trial.name + "-preparation-profile/profile.json")),
    }
    write(trial / "private-paths.json", paths)
    write(trial / "training-chain.json", chain)
    write(trial / "calibration-tasks.json", tasks)
    frozen = {
        "schema": "chipmunk.regional-calibration-execution-supplement/42",
        "implementation_sha256": code,
        "asset_sha256": ASSETS,
        "runtime_versions": versions,
        "training_plan_sha256": sha(study / "plan.json"),
        "training_freeze_sha256": sha(study / "freeze.json"),
        "training_chain_sha256": sha(trial / "training-chain.json"),
        "private_paths_sha256": sha(trial / "private-paths.json"),
        "calibration_tasks_sha256": sha(trial / "calibration-tasks.json"),
        "complete_training_updates": 1278,
        "accepted_original_prefix_phases": 33,
        "accepted_continuation_suffix_phases": 7,
        "discarded_original_attempt_updates": 32,
        "guarded_original_checkpoint_used": False,
        "complete_training_phases": 40,
        "final_checkpoint_sha256": chain["final_checkpoint_sha256"],
        "reserved_source_forwards_at_freeze": 0,
        "unique_calibration_views_per_arm_format": 156,
        "unique_model_calls_planned": 624,
        "repeat_calls_planned": 4,
        "inference_requires_remote_verified_execution_start": True,
        "serving_admitted": False,
        "scope": "Exact final-state prospective binding. This worker performs no neural inference. A successful bounded audit profile and verified publication are required before score execution.",
    }
    write(trial / "freeze.json", frozen)
    print(
        "Prepared complete final-checkpoint supplement; publication and terminal audit profile still required",
        flush=True,
    )
    return frozen


def check_score_admission(trial):
    frozen, start, paths = (
        read(trial / "freeze.json"),
        read(trial / "execution-start.json"),
        read(trial / "private-paths.json"),
    )
    require(
        start["public_freeze_sha256"] == sha(trial / "freeze.json")
        and start["remote_verified_before_first_reserved_forward"] is True
        and len(start["remote_freeze_blob_sha"]) == 40,
        "Exact remotely verified execution supplement required",
    )
    require(
        sha(trial / "private-paths.json") == frozen["private_paths_sha256"],
        "Private audit/output paths changed",
    )
    require(
        runtime_versions() == frozen["runtime_versions"],
        "Runtime changed after preparation; do not substitute a new runtime",
    )
    require(
        all(sha(ROOT / path) == value for path, value in frozen["implementation_sha256"].items()),
        "Frozen inference/audit implementation changed",
    )
    profile_path = Path(paths["preparation_profile"])
    require(sha(profile_path) == start["preparation_profile_sha256"], "Terminal preparation profile changed")
    profile = read(profile_path)
    require(
        profile["module"] == "scripts.prepare_regional_calibration_v42"
        and profile["arguments"] == ["prepare", paths["study"], str(trial.resolve()), paths["raw_map_root"]],
        "Preparation resource proof belongs to another audit",
    )
    require(
        profile["monitor_implementation_sha256"] == sha(ROOT / "backend/scripts/regional_calibration_runtime_v42.py")
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and not profile["resource_limit_observations"],
        "Failed/guarded preparation cannot authorize inference",
    )
    require(
        sha(profile_path.parent / "resource-samples.jsonl") == profile["resource_journal_sha256"],
        "Preparation resource observations changed",
    )
    require(profile["limits"]["seconds"] == 900 and profile["limits"]["sampled_tree_rss_bytes"] == 6 * 1024**3 and 0 < profile["duration_s"] <= 915, "Bounded preparation resource limits required")
    return frozen


def score(trial, arm, fmt, number):
    from scripts.regional_calibration_runtime_v42 import run

    check_score_admission(trial)
    return run("score", [str(trial.resolve()), arm, fmt, str(number)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("study", type=Path)
    p.add_argument("trial", type=Path)
    p.add_argument("raw_map_root", type=Path)
    p = commands.add_parser("score")
    p.add_argument("trial", type=Path)
    p.add_argument("arm", choices=("candidate", "baseline"))
    p.add_argument("format", choices=("native", "jpeg75"))
    p.add_argument("number", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.study, args.trial, args.raw_map_root)
    else:
        score(args.trial, args.arm, args.format, args.number)
