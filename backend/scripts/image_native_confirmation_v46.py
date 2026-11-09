"""Fixed native image confirmation on reserved source groups; no refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import image_native_comparison_v45 as previous
from scripts import image_regional_inference_v28 as original
from scripts.prepare_regional_calibration_v42 import runtime_versions

ROOT = Path(__file__).resolve().parents[2]
read, write, sha, require = original.read, original.write, original.sha, original.require
phase_tasks, encoded_view = original.phase_tasks, original.encoded_view
ARMS = ("candidate", "baseline")
POLICIES = ("p0.90-a0.01000",)
OWN = (
    "backend/scripts/image_native_confirmation_v46.py",
    "backend/scripts/image_native_confirmation_readout_v46.py",
    "backend/scripts/image_native_confirmation_runtime_v46.py",
    "backend/tests/test_image_native_confirmation_v46.py",
    "docs/image-native-confirmation-v46-protocol.md",
)
ASSETS = {
    path: digest
    for path, digest in previous.ASSETS.items()
    if "image-only.pth" in path or "tidal-v30.1.pt" in path
}


def comparison_tasks(plan):
    cases = sorted(
        (r for r in plan["rows"] if r["role"] == "confirmation"),
        key=lambda r: (r["component"], r["source_id"]),
    )
    require(
        len(cases) == 59
        and len({r["component"] for r in cases}) == 48
        and len({r["source_id"] for r in cases}) == 59,
        "Complete unique 59-photo/48-component confirmation role required",
    )
    components = {r["component"] for r in cases}
    require(
        not any(r["component"] in components and r["role"] != "confirmation" for r in plan["rows"]),
        "Source components cannot cross roles",
    )
    return [
        {"source_id": r["source_id"], "component": r["component"], "role": "confirmation", "kind": kind}
        for r in cases
        for kind in original.KINDS
    ]


def check_selection(parent):
    selection = read(parent / "selected-followup.json")
    require(
        selection["primary_next_readout"] == POLICIES[0]
        and selection["secondary_next_readout"] == "current-and-strict"
        and selection["comparator"] == "baseline",
        "Previously selected methods required",
    )
    require(
        selection["selection_precedes_confirmation_forwards"] is True
        and selection["confirmation_forwards_at_selection"] == 0
        and selection["next_source_role"] == "confirmation"
        and selection["next_source_photos"] == 59
        and selection["next_source_components"] == 48,
        "Prospective confirmation selection required",
    )
    require(
        selection["completed_summary_sha256"]
        == sha(parent / "comparison-summary.json")
        == "7f2bc695455ba5786c59d0a7c63d3483d7d9b1d039860e51bb8c2e273493be63",
        "Completed unchanged comparison required",
    )
    return selection


def prepare(parent, trial, raw_root):
    parent, trial, raw_root = parent.resolve(), trial.resolve(), raw_root.resolve()
    require(
        not trial.exists() and raw_root.is_dir() and not any(raw_root.iterdir()),
        "New study and empty private map folder required",
    )
    parent_freeze, chain, plan = previous.lease(parent)
    check_selection(parent)
    result = ROOT / "docs/evaluation-panels/image-native-comparison-v45-result.json"
    execution = ROOT / "docs/evaluation-panels/image-native-comparison-v45-execution.json"
    require(
        sha(result) == "7f2bc695455ba5786c59d0a7c63d3483d7d9b1d039860e51bb8c2e273493be63"
        and sha(execution) == "540a38f9724e4ec526d70ae5b1642c86085325b17e4a185d1104fbd6a85e6f52",
        "Published comparison/selection evidence changed",
    )
    prior = read(execution)
    require(
        prior["completed_neural_calls"] == 327 and prior["final_confirmation_used"] is False,
        "Comparison must not consume confirmation",
    )
    for entry in prior["profiles"]:
        require(
            sha(parent / (entry["phase"] + "-profile") / "profile.json") == entry["profile_sha256"],
            "Prior terminal proof changed",
        )
    code = dict(parent_freeze["code_sha256"])
    for name in (
        *OWN,
        str(result.relative_to(ROOT)),
        str(execution.relative_to(ROOT)),
        "docs/image-native-comparison-v45-result.md",
    ):
        code[name] = sha(ROOT / name)
    require(
        all(sha(ROOT / path) == value for path, value in ASSETS.items()), "Pinned native model assets changed"
    )
    tasks = comparison_tasks(plan)
    trial.mkdir(parents=True, exist_ok=False)
    write(trial / "private-paths.json", {"previous_comparison": str(parent), "raw_map_root": str(raw_root)})
    write(trial / "tasks.json", tasks)
    frozen = {
        "schema": "chipmunk.native-confirmation/46",
        "code_sha256": code,
        "asset_sha256": ASSETS,
        "runtime_versions": runtime_versions(),
        "previous_freeze_sha256": sha(parent / "freeze.json"),
        "selection_sha256": sha(parent / "selected-followup.json"),
        "training_plan_sha256": parent_freeze["training_plan_sha256"],
        "final_checkpoint_sha256": chain["final_checkpoint_sha256"],
        "private_paths_sha256": sha(trial / "private-paths.json"),
        "tasks_sha256": sha(trial / "tasks.json"),
        "source_role": "confirmation",
        "photos": 59,
        "components": 48,
        "views_per_arm": 236,
        "arms": list(ARMS),
        "candidate_policies": list(POLICIES),
        "readouts": ["baseline", POLICIES[0], "current-and-strict"],
        "unique_model_calls": 472,
        "repeat_model_calls": 2,
        "source_forwards_at_freeze": 0,
        "parameters_or_calibration_refitted": False,
        "serving_admitted": False,
        "scope": "Source-group confirmation of two previously selected native readouts with current comparator. Same editing family; not an unseen-generator or population guarantee. No automatic rejection for one worse error type. No tuning or selection using confirmation labels.",
    }
    write(trial / "freeze.json", frozen)
    print(
        "Prepared fixed confirmation; remote publication and terminal preparation proof required", flush=True
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
        "Exact remotely verified confirmation freeze required",
    )
    require(
        sha(trial / "private-paths.json") == frozen["private_paths_sha256"]
        and sha(trial / "tasks.json") == frozen["tasks_sha256"],
        "Study paths/tasks changed",
    )
    require(
        all(sha(ROOT / name) == digest for name, digest in frozen["code_sha256"].items())
        and runtime_versions() == frozen["runtime_versions"],
        "Frozen code/runtime changed",
    )
    profile_path = trial.parent / (trial.name + "-preparation-profile/profile.json")
    require(sha(profile_path) == start["preparation_profile_sha256"], "Preparation proof changed")
    profile = read(profile_path)
    require(
        profile["module"] == "scripts.image_native_confirmation_v46"
        and profile["arguments"]
        == ["prepare", paths["previous_comparison"], str(trial.resolve()), paths["raw_map_root"]],
        "Preparation belongs to another study",
    )
    require(
        profile["monitor_implementation_sha256"]
        == sha(ROOT / "backend/scripts/image_native_confirmation_runtime_v46.py")
        and profile["limits"]["seconds"] == 900
        and profile["limits"]["sampled_tree_rss_bytes"] == 6 * 1024**3
        and 0 < profile["duration_s"] <= 915,
        "Bounded correct observer required",
    )
    require(
        profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and not profile["resource_limit_observations"]
        and sha(profile_path.parent / "resource-samples.jsonl") == profile["resource_journal_sha256"],
        "Failed/guarded preparation cannot admit inference",
    )
    parent = Path(paths["previous_comparison"])
    require(
        sha(parent / "freeze.json") == frozen["previous_freeze_sha256"]
        and sha(parent / "selected-followup.json") == frozen["selection_sha256"],
        "Previous selection changed",
    )
    check_selection(parent)
    parent_freeze, chain, plan = previous.lease(parent)
    require(
        chain["final_checkpoint_sha256"] == frozen["final_checkpoint_sha256"]
        and parent_freeze["training_plan_sha256"] == frozen["training_plan_sha256"]
        and comparison_tasks(plan) == read(trial / "tasks.json"),
        "Fixed model/role support changed",
    )
    return frozen, chain, plan


def load_predictor(arm, fmt, chain):
    require(arm in ARMS and fmt == "native", "Only the two selected native paths are admitted")
    return original.load_predictor(arm, "native", chain)


def score(trial, arm, fmt, number):
    from scripts.image_regional_training_v26 import PAIRS, source_views

    require(
        arm in {"candidate", "baseline"} and fmt in {"native", "jpeg75"}, "Explicit arm and format required"
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
                require(
                    cases[item["source_id"]]["role"] == "confirmation", "Only confirmation sources admitted"
                )
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
            "schema": "chipmunk.regional-confirmation-producer/46",
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
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("parent", type=Path)
    p.add_argument("trial", type=Path)
    p.add_argument("raw_root", type=Path)
    p = sub.add_parser("score")
    p.add_argument("trial", type=Path)
    p.add_argument("arm", choices=ARMS)
    p.add_argument("format", choices=("native",))
    p.add_argument("number", type=int, choices=range(10))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.parent, args.trial, args.raw_root)
    else:
        score(args.trial, args.arm, args.format, args.number)
