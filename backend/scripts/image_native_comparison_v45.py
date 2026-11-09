"""Frozen three-path comparison on the previously unused point source role."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import image_regional_inference_v28 as original
from scripts.prepare_regional_calibration_v42 import ASSETS, check_score_admission, runtime_versions

ROOT = Path(__file__).resolve().parents[2]
read, write, sha, require = original.read, original.write, original.sha, original.require
phase_tasks, encoded_view = original.phase_tasks, original.encoded_view
ARMS = ("candidate", "baseline", "fused-reference")
POLICIES = ("p0.90-a0.01000", "p0.60-a0.01000", "p0.40-a0.00100")
OWN = (
    "backend/scripts/image_native_comparison_v45.py",
    "backend/scripts/image_native_comparison_readout_v45.py",
    "backend/scripts/image_native_comparison_runtime_v45.py",
    "backend/tests/test_image_native_comparison_v45.py",
    "docs/image-native-comparison-v45-protocol.md",
)


def comparison_tasks(plan):
    cases = sorted(
        (r for r in plan["rows"] if r["role"] == "point"), key=lambda r: (r["component"], r["source_id"])
    )
    require(
        len(cases) == 27 and len({r["component"] for r in cases}) == 24,
        "Complete 27-photo/24-component point role required",
    )
    require(len({r["source_id"] for r in cases}) == 27, "Unique point photos required")
    components = {r["component"] for r in cases}
    require(
        not any(r["component"] in components and r["role"] != "point" for r in plan["rows"]),
        "Source components cannot cross roles",
    )
    return [
        {"source_id": r["source_id"], "component": r["component"], "role": "point", "kind": kind}
        for r in cases
        for kind in original.KINDS
    ]


def prepare(parent, trial, raw_root):
    parent, trial, raw_root = parent.resolve(), trial.resolve(), raw_root.resolve()
    require(not trial.exists(), "New comparison study required; retain failed attempts")
    require(raw_root.is_dir() and not any(raw_root.iterdir()), "New private empty map folder required")
    check_score_admission(parent)
    parent_freeze, chain, plan = original.lease(parent)
    result_path = ROOT / "docs/evaluation-panels/image-regional-calibration-v43-results.json"
    require(
        sha(parent / "calibration-v43-summary.json")
        == sha(result_path)
        == "eeee0ec58f300c015fd48df677b42cd5730f8abe5a1e86e0d34c6f0a53f7278a",
        "Completed unchanged calibration comparison required",
    )
    execution = read(ROOT / "docs/evaluation-panels/image-regional-calibration-v43-execution.json")
    require(
        sha(ROOT / "docs/evaluation-panels/image-regional-calibration-v43-execution.json")
        == "90fa0577bd3c7551341680d221462758e4c7371c6e2d8819f183c3bfd42bbab7",
        "Published completed calibration receipt changed",
    )
    require(
        execution["total_model_calls"] == 628 and execution["point_diagnostic_confirmation_model_calls"] == 0,
        "Point sources must not have been used by calibration",
    )
    for entry in execution["profiles"]:
        require(
            sha(parent / (entry["phase"] + "-profile") / "profile.json") == entry["profile_sha256"],
            "Calibration terminal profile changed",
        )
    code = dict(parent_freeze["implementation_sha256"])
    for name in (
        *OWN,
        "backend/scripts/evaluation_tradeoffs_v44.py",
        "docs/evaluation-panels/model-decision-review-v44.json",
        "docs/evaluation-panels/image-regional-calibration-v43-results.json",
        "docs/evaluation-panels/image-regional-calibration-v43-execution.json",
    ):
        code[name] = sha(ROOT / name)
    require(
        all(sha(ROOT / path) == value for path, value in ASSETS.items()),
        "Pinned permitted model asset changed",
    )
    tasks = comparison_tasks(plan)
    trial.mkdir(parents=True, exist_ok=False)
    write(trial / "private-paths.json", {"parent_calibration": str(parent), "raw_map_root": str(raw_root)})
    write(trial / "tasks.json", tasks)
    frozen = {
        "schema": "chipmunk.native-multimodel-comparison/45",
        "code_sha256": code,
        "asset_sha256": ASSETS,
        "runtime_versions": runtime_versions(),
        "parent_freeze_sha256": sha(parent / "freeze.json"),
        "training_plan_sha256": parent_freeze["training_plan_sha256"],
        "final_checkpoint_sha256": chain["final_checkpoint_sha256"],
        "private_paths_sha256": sha(trial / "private-paths.json"),
        "tasks_sha256": sha(trial / "tasks.json"),
        "source_role": "point",
        "photos": 27,
        "components": 24,
        "views_per_arm": 108,
        "arms": list(ARMS),
        "candidate_policies": list(POLICIES),
        "unique_model_calls": 324,
        "repeat_model_calls": 3,
        "source_forwards_at_freeze": 0,
        "confirmation_source_forwards": 0,
        "serving_admitted": False,
        "scope": "Unused-source development/selection comparison, not final validation. Three native detector paths and seven predeclared readouts. No automatic rejection from one worse error type. Source/creator aliases and pretraining overlap remain limitations.",
    }
    write(trial / "freeze.json", frozen)
    print(
        "Prepared exact comparison freeze; remotely verified publication and terminal preparation profile required",
        flush=True,
    )
    return frozen


def lease(trial):
    frozen, start, paths = (
        read(trial / "freeze.json"),
        read(trial / "execution-start.json"),
        read(trial / "private-paths.json"),
    )
    require(
        start["public_freeze_sha256"] == sha(trial / "freeze.json")
        and start["remote_verified_before_first_source_forward"] is True
        and len(start["remote_freeze_blob_sha"]) == 40,
        "Exact published point comparison required",
    )
    require(
        sha(trial / "private-paths.json") == frozen["private_paths_sha256"]
        and sha(trial / "tasks.json") == frozen["tasks_sha256"],
        "Private study paths/tasks changed",
    )
    require(
        all(sha(ROOT / name) == digest for name, digest in frozen["code_sha256"].items()),
        "Frozen comparison implementation changed",
    )
    require(runtime_versions() == frozen["runtime_versions"], "Runtime changed after freeze")
    profile_path = trial.parent / (trial.name + "-preparation-profile/profile.json")
    require(sha(profile_path) == start["preparation_profile_sha256"], "Preparation resource proof changed")
    profile = read(profile_path)
    require(
        profile["monitor_implementation_sha256"]
        == sha(ROOT / "backend/scripts/image_native_comparison_runtime_v45.py")
        and profile["limits"]["seconds"] == 900
        and profile["limits"]["sampled_tree_rss_bytes"] == 6 * 1024**3
        and 0 < profile["duration_s"] <= 915,
        "Bounded correct preparation observer required",
    )
    require(
        profile["module"] == "scripts.image_native_comparison_v45"
        and profile["arguments"]
        == ["prepare", paths["parent_calibration"], str(trial.resolve()), paths["raw_map_root"]],
        "Resource proof belongs to another preparation",
    )
    require(
        profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and not profile["resource_limit_observations"],
        "Failed/guarded preparation cannot admit model calls",
    )
    require(
        sha(profile_path.parent / "resource-samples.jsonl") == profile["resource_journal_sha256"],
        "Preparation resource observations changed",
    )
    parent = Path(paths["parent_calibration"])
    require(sha(parent / "freeze.json") == frozen["parent_freeze_sha256"], "Parent comparison changed")
    parent_freeze, chain, plan = original.lease(parent)
    require(
        chain["final_checkpoint_sha256"] == frozen["final_checkpoint_sha256"]
        and parent_freeze["training_plan_sha256"] == frozen["training_plan_sha256"],
        "Training state/roles changed",
    )
    require(comparison_tasks(plan) == read(trial / "tasks.json"), "Source support changed")
    return frozen, chain, plan


def load_predictor(arm, fmt, chain):
    require(arm in ARMS and fmt == "native", "Explicit native comparison path required")
    return original.load_predictor(
        "baseline" if arm == "fused-reference" else arm,
        "jpeg75" if arm == "fused-reference" else "native",
        chain,
    )


def score(trial, arm, fmt, number):
    from scripts.image_regional_training_v26 import PAIRS, source_views

    require(
        arm in {"candidate", "baseline", "fused-reference"} and fmt in {"native", "jpeg75"},
        "Explicit arm and format required",
    )
    frozen, chain, plan = lease(trial)
    selected = phase_tasks(comparison_tasks(plan), number)
    name = f"{arm}-{fmt}-{number:03d}"
    if number > 0:
        previous = trial / f"{arm}-{fmt}-{number - 1:03d}"
        previous_result = read(previous / "completed.json")
        previous_profile = read(trial / f"{arm}-{fmt}-{number - 1:03d}-profile/profile.json")
        require(
            previous_profile["child_exit_code"] == 0
            and previous_profile["guard"] is None
            and (previous_profile["sample_errors"] == 0)
            and (sha(previous / "rows.jsonl") == previous_result["rows_sha256"]),
            "A missing or failed previous inference phase cannot be skipped",
        )
    out = trial / name
    out.mkdir(exist_ok=False)
    raw_root = Path(read(trial / "private-paths.json")["raw_map_root"]) / name
    raw_root.mkdir(exist_ok=False)
    attempted = completed = repeats = 0

    def event(stage, **details):
        with (out / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "stage": stage,
                        "attempted_model_forwards": attempted,
                        "completed_model_forwards": completed,
                        **details,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()

    event("setup_begin")
    try:
        predict = load_predictor(arm, fmt, chain)
    except BaseException as error:
        with (out / "rows.jsonl").open("x") as journal:
            for item in selected:
                journal.write(
                    json.dumps(
                        {
                            **item,
                            "format": fmt,
                            "arm": arm,
                            "status": "unavailable",
                            "error_type": type(error).__name__,
                        }
                    )
                    + "\n"
                )
        event("setup_failed", error_type=type(error).__name__)
        raise
    cases = {r["source_id"]: r for r in plan["rows"]}
    qualified = {
        r["source_id"]: r
        for r in (json.loads(line) for line in (PAIRS / "qualification/pairs.jsonl").read_text().splitlines())
    }
    previous_id, views = (None, None)
    with (out / "rows.jsonl").open("x") as journal:
        for index, item in enumerate(selected):
            try:
                require(cases[item["source_id"]]["role"] == "point", "Only point sources admitted")
                if previous_id != item["source_id"]:
                    edited, controls, truth = source_views(
                        cases[item["source_id"]], qualified[item["source_id"]]
                    )
                    views = {"ai": edited, **controls}
                    previous_id = item["source_id"]
                data, image = encoded_view(views[item["kind"]], fmt)
                attempted += 1
                event("forward_begin", source_id=item["source_id"], kind=item["kind"])
                raw, details = predict(image)
                completed += 1
                require(
                    raw.ndim == 2
                    and raw.dtype == np.float32
                    and np.isfinite(raw).all()
                    and (raw.min() >= 0)
                    and (raw.max() <= 1),
                    "Finite float32 response field required",
                )
                if number == 0 and index == 0:
                    attempted += 1
                    repeat, repeat_details = predict(image)
                    completed += 1
                    require(
                        np.array_equal(raw, repeat) and details == repeat_details,
                        "First input must repeat exact raw response and input bindings",
                    )
                    repeats += 1
                path = raw_root / f"{index:03d}.npy"
                with path.open("xb") as stream:
                    np.save(stream, raw, allow_pickle=False)
                require(
                    np.array_equal(np.load(path, allow_pickle=False), raw), "Stored map failed exact readback"
                )
                row = {
                    **item,
                    "format": fmt,
                    "arm": arm,
                    "status": "ok",
                    "input_sha256": hashlib.sha256(data).hexdigest(),
                    "source_rgb_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                    "source_size": list(image.size),
                    "positive_truth_png_sha256": qualified[item["source_id"]]["truth_png_sha256"],
                    "negative_truth_is_zero": item["kind"] != "ai",
                    "raw_map_file": str(path),
                    "raw_map_sha256": sha(path),
                    "raw_map_shape": list(raw.shape),
                    **details,
                }
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                event("source_complete", source_id=item["source_id"], kind=item["kind"])
            except BaseException as error:
                journal.write(
                    json.dumps(
                        {
                            **item,
                            "format": fmt,
                            "arm": arm,
                            "status": "failed",
                            "error_type": type(error).__name__,
                        }
                    )
                    + "\n"
                )
                journal.flush()
                event(
                    "failed", source_id=item["source_id"], kind=item["kind"], error_type=type(error).__name__
                )
                raise
    write(
        out / "completed.json",
        {
            "schema": "chipmunk.regional-point-producer/45",
            "arm": arm,
            "format": fmt,
            "phase": number,
            "selected_views": len(selected),
            "attempted_model_forwards": attempted,
            "completed_model_forwards": completed,
            "exact_repeat_extra_forwards": repeats,
            "rows_sha256": sha(out / "rows.jsonl"),
            "execution_freeze_sha256": sha(trial / "freeze.json"),
            "serving_admitted": False,
        },
    )
    event("completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("parent", type=Path)
    p.add_argument("trial", type=Path)
    p.add_argument("raw_root", type=Path)
    p = subs.add_parser("score")
    p.add_argument("trial", type=Path)
    p.add_argument("arm", choices=ARMS)
    p.add_argument("format", choices=("native",))
    p.add_argument("number", type=int, choices=range(5))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.parent, args.trial, args.raw_root)
    else:
        score(args.trial, args.arm, args.format, args.number)
