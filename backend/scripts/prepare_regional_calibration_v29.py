"""Complete-chain audit and publication preparation; never performs inference."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path

from scripts.image_regional_inference_v28 import audit_training, calibration_tasks, read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
PREP28 = ROOT / "docs/evaluation-panels/image-regional-inference-v28-preparation.json"
READOUT27 = ROOT / "docs/evaluation-panels/image-regional-readout-v27-freeze.json"
PREP29 = ROOT / "docs/evaluation-panels/image-regional-calibration-v29-preparation.json"
ASSETS = {
    "models/research/image-v18-relayformer-v2/image-only.pth": "00a0f145ae4a98e66cad95aa79d2ce470d77821ee4262d6b803b3705c11c2090",
    "models/weights/splice/tidal-v30.1.pt": "ca3aa144e6ca08f2bb2eae93ac45d76c27a2922770e028bf90ef76ce974ee061",
    "models/weights/fused/fused_opensdi.pth": "7e527d49080a773641cf96915b4b3fa12d571c5b87f35c527f13ab82f148364c",
    "models/weights/fused/convnext.safetensors": "1ddf6dbc7f0753fc4dfec5db5fa474acfd1ac6f67f466d6a976780c30b3a1b23",
}
OWN_FILES = (
    "backend/scripts/prepare_regional_calibration_v29.py",
    "backend/scripts/regional_calibration_preparation_runtime_v29.py",
    "backend/tests/test_prepare_regional_calibration_v29.py",
    "docs/image-regional-calibration-execution-v29-protocol.md",
)


def runtime_versions():
    distributions = (
        "torch",
        "torchvision",
        "timm",
        "numpy",
        "Pillow",
        "opencv-python-headless",
        "opencv-python",
        "opencv-contrib-python",
    )
    result = {"python": platform.python_version(), "machine": platform.machine()}
    for name in distributions:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    require(
        all(result[name] is not None for name in ("torch", "torchvision", "timm", "numpy", "Pillow")),
        "Required offline runtime distribution unavailable",
    )
    require(
        any(result[name] is not None for name in distributions[-3:]), "Pinned OpenCV distribution required"
    )
    return result


def implementations(study):
    training, prep, readout = read(study / "freeze.json"), read(PREP28), read(READOUT27)
    require(
        sha(study / "plan.json") == prep["training_plan_sha256"]
        and sha(study / "freeze.json") == prep["training_freeze_sha256"]
        and sha(READOUT27) == prep["readout_v27_freeze_sha256"],
        "Prospective roles or upstream readout commitment changed",
    )
    expected = {
        **training["code_sha256"],
        **readout["code_and_protocol_sha256"],
        **prep["implementation_sha256"],
    }
    require(
        all(sha(ROOT / path) == value for path, value in expected.items()),
        "A frozen implementation changed; explicit successor required",
    )
    own = read(PREP29)["implementation_sha256"]
    require(
        set(own) == set(OWN_FILES) and all(sha(ROOT / path) == value for path, value in own.items()),
        "Published audit/preparation implementation changed",
    )
    expected.update(own)
    expected.update({str(path.relative_to(ROOT)): sha(path) for path in (PREP28, READOUT27, PREP29)})
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
    chain = audit_training(study)
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
        "schema": "chipmunk.regional-calibration-execution-supplement/29",
        "implementation_sha256": code,
        "asset_sha256": ASSETS,
        "runtime_versions": versions,
        "training_plan_sha256": sha(study / "plan.json"),
        "training_freeze_sha256": sha(study / "freeze.json"),
        "training_chain_sha256": sha(trial / "training-chain.json"),
        "private_paths_sha256": sha(trial / "private-paths.json"),
        "calibration_tasks_sha256": sha(trial / "calibration-tasks.json"),
        "complete_training_updates": 1278,
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
        profile["module"] == "scripts.prepare_regional_calibration_v29"
        and profile["arguments"] == ["prepare", paths["study"], str(trial.resolve()), paths["raw_map_root"]],
        "Preparation resource proof belongs to another audit",
    )
    require(
        profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and not profile["resource_limit_observations"],
        "Failed/guarded preparation cannot authorize inference",
    )
    require(
        sha(profile_path.parent / "resource-samples.jsonl") == profile["resource_journal_sha256"],
        "Preparation resource observations changed",
    )
    return frozen


def score(trial, arm, fmt, number):
    from scripts.image_regional_inference_runtime_v28 import run

    check_score_admission(trial)
    return run(trial, arm, fmt, number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("study", type=Path)
    prepare_parser.add_argument("trial", type=Path)
    prepare_parser.add_argument("raw_map_root", type=Path)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("trial", type=Path)
    score_parser.add_argument("arm", choices=("candidate", "baseline"))
    score_parser.add_argument("format", choices=("native", "jpeg75"))
    score_parser.add_argument("number", type=int)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments.study, arguments.trial, arguments.raw_map_root)
    else:
        score(arguments.trial, arguments.arm, arguments.format, arguments.number)
