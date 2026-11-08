"""Reserved calibration producer; requires complete training and remote freeze.

This module cannot score point, diagnostic or confirmation cohorts. Their
selected-policy execution needs a later explicit supplement after calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 24
KINDS = ("ai", "original", "tone", "blur")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def audit_training(study):
    """Verify every completed resource/checkpoint link before granting a lease."""
    plan, frozen = read(study / "plan.json"), read(study / "freeze.json")
    require(sha(study / "plan.json") == frozen["private_plan_sha256"], "Frozen training plan changed")
    require(
        all(sha(ROOT / path) == value for path, value in frozen["code_sha256"].items()),
        "Frozen training implementation changed",
    )
    require(len(plan["schedule"]) == 1278, "Complete six-epoch prescribed training required")
    count = (len(plan["schedule"]) + 31) // 32
    previous, previous_tensors, phases = None, None, []
    for number in range(count):
        part = study / f"phase-{number:03d}"
        require(
            (part / "completed.json").is_file(), "All training phases must finish before reserved inference"
        )
        result = read(part / "completed.json")
        profile_path = study / f"phase-{number:03d}-profile/profile.json"
        require(profile_path.is_file(), "Every training phase needs a terminal resource receipt")
        profile = read(profile_path)
        require(
            profile["module"] == "scripts.image_regional_training_v26"
            and profile["phase"] == number
            and profile["arguments"] == ["train", str(study), str(number)],
            "Resource receipt belongs to another worker or phase",
        )
        require(
            profile["child_exit_code"] == 0
            and profile["guard"] is None
            and profile["sample_errors"] == 0
            and not profile["resource_limit_observations"],
            "Failed/guarded training cannot authorize reserved inference",
        )
        require(
            profile["limits"]["seconds"] == 900
            and profile["limits"]["sampled_tree_rss_bytes"] == 6 * 1024**3
            and 0 < profile["duration_s"] <= 915,
            "Unbounded training phase cannot authorize inference",
        )
        require(
            sha(profile_path.parent / "resource-samples.jsonl") == profile["resource_journal_sha256"],
            "Training resource journal changed",
        )
        start, end = number * 32, min((number + 1) * 32, len(plan["schedule"]))
        require(
            result["phase"] == number
            and result["start_update"] == start
            and result["end_update"] == end
            and result["plan_sha256"] == sha(study / "plan.json"),
            "Training phase identity or completed range differs",
        )
        require(
            result["predecessor_sha256"] == previous
            and (previous_tensors is None or result["initial_parameter_sha256"] == previous_tensors),
            "Learned-parameter chain was reset or replaced",
        )
        require(
            result["completed_optimizer_updates"] == end - start
            and result["completed_source_forwards"] == result["completed_backwards"] == 2 * (end - start),
            "Actual training completion counters differ",
        )
        require(
            result["dense_base_and_fixed_frequencies_unchanged"]
            and result["local_global_and_decoder_changed"]
            and result["checkpoint_parameter_readback_exact"]
            and result["held_out_source_forwards"] == 0,
            "Training preservation/readback or held-out separation failed",
        )
        journal = part / "updates.jsonl"
        require(sha(journal) == result["updates_sha256"], "Training update journal changed")
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        require(
            len(rows) == end - start and [r["update"] for r in rows] == list(range(start, end)),
            "Training updates are missing or duplicated",
        )
        roles = {r["source_id"]: r["role"] for r in plan["rows"]}
        for row in rows:
            require(
                {k: row[k] for k in plan["schedule"][row["update"]]} == plan["schedule"][row["update"]]
                and roles[row["source_id"]] == "fit",
                "A held-out source or changed training view entered fitting",
            )
        checkpoint = Path(plan["checkpoint_root"]) / f"phase-{number:03d}.pt"
        require(
            sha(checkpoint) == result["checkpoint_sha256"], "A retained private training checkpoint changed"
        )
        phases.append(
            {
                "number": number,
                "checkpoint_sha256": result["checkpoint_sha256"],
                "result_sha256": sha(part / "completed.json"),
                "profile_sha256": sha(profile_path),
            }
        )
        previous, previous_tensors = result["checkpoint_sha256"], result["final_parameter_sha256"]
    return {
        "phases": phases,
        "completed_updates": len(plan["schedule"]),
        "final_checkpoint": str(checkpoint),
        "final_checkpoint_sha256": previous,
        "final_parameter_sha256": previous_tensors,
    }


def calibration_tasks(plan):
    cases = sorted(
        (r for r in plan["rows"] if r["role"] == "calibration"),
        key=lambda r: (r["component"], r["source_id"]),
    )
    require(
        len(cases) == 39 and len({r["component"] for r in cases}) == 32,
        "All 32 calibration groups and 39 photos required",
    )
    return [
        {"source_id": case["source_id"], "component": case["component"], "role": "calibration", "kind": kind}
        for case in cases
        for kind in KINDS
    ]


def phase_tasks(tasks, number):
    require(type(number) is int and number >= 0, "Explicit nonnegative phase required")
    selected = tasks[number * CHUNK : (number + 1) * CHUNK]
    require(selected, "No prescribed calibration views in this phase")
    return selected


def encoded_view(image, fmt):
    require(
        fmt in {"native", "jpeg75"} and image.mode == "RGB", "Already oriented RGB and known format required"
    )
    buffer = io.BytesIO()
    if fmt == "jpeg75":
        image.save(buffer, format="JPEG", quality=75)
    else:
        image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as decoded:
        result = decoded.convert("RGB").copy()
    require(result.size == image.size, "Encoding changed source coordinates")
    return payload, result


def lease(trial):
    """An unfinished checkpoint or unpublished/mutated producer cannot run."""
    frozen, admission = read(trial / "freeze.json"), read(trial / "execution-start.json")
    require(
        admission["public_freeze_sha256"] == sha(trial / "freeze.json")
        and admission["remote_verified_before_first_reserved_forward"] is True
        and len(admission["remote_freeze_blob_sha"]) == 40,
        "Remotely verified reserved-inference supplement required",
    )
    require(
        frozen["complete_training_updates"] == 1278 and frozen["reserved_source_forwards_at_freeze"] == 0,
        "Completed training and untouched reserved sources required",
    )
    require(
        all(sha(ROOT / p) == value for p, value in frozen["implementation_sha256"].items()),
        "Frozen producer/readout changed",
    )
    require(
        sha(trial / "private-paths.json") == frozen["private_paths_sha256"],
        "Private study/map destinations changed",
    )
    study = Path(read(trial / "private-paths.json")["study"])
    require(
        sha(study / "plan.json") == frozen["training_plan_sha256"]
        and sha(trial / "training-chain.json") == frozen["training_chain_sha256"],
        "Training roles or complete-chain proof changed",
    )
    chain = read(trial / "training-chain.json")
    require(
        chain["completed_updates"] == 1278
        and sha(Path(chain["final_checkpoint"]))
        == chain["final_checkpoint_sha256"]
        == frozen["final_checkpoint_sha256"],
        "Final learned state differs from the prospective supplement",
    )
    return frozen, chain, read(study / "plan.json")


def load_predictor(arm, fmt, chain):
    import torch

    from scripts.relayformer_learning_v23 import sha_state, trainable_name
    from scripts.relayformer_native_v18 import Native, prepare

    require(
        arm in {"candidate", "baseline"} and fmt in {"native", "jpeg75"}, "Explicit arm and format required"
    )
    torch.set_num_threads(2)
    torch.manual_seed(20261108)
    if arm == "candidate":
        native = Native()
        state = torch.load(chain["final_checkpoint"], map_location="cpu", weights_only=True)
        parameters = {n: p for n, p in native.model.named_parameters() if trainable_name(n)}
        require(
            len(parameters) == 246
            and set(parameters) == set(state["parameters"])
            and state["completed_updates"] == 1278,
            "Complete prescribed learned adaptation required",
        )
        with torch.no_grad():
            for name, p in parameters.items():
                value = state["parameters"][name]
                require(
                    value.dtype == p.dtype and value.shape == p.shape and bool(torch.isfinite(value).all()),
                    "Invalid learned tensor",
                )
                p.copy_(value)
        require(sha_state(parameters) == chain["final_parameter_sha256"], "Learned tensor readback differs")
        del state

        def predict(image):
            tensor, geometry = prepare(image)
            raw, binding = native.predict(tensor, geometry)
            return raw, {"geometry": geometry, "binding": binding, "image_gate": None, "image_score": None}

        return predict
    if fmt == "native":
        from app.detectors.image import splice

        model = splice.load_model(ROOT / "models/weights/splice" / splice.CHECKPOINT_FILE)

        def predict(image):
            raw = splice.predict(model, image)
            gate = int(np.count_nonzero(raw > splice.PIXEL_THRESHOLD)) >= splice.MINIMUM_ACTIVE_CELLS
            return raw, {"image_gate": gate, "image_score": None, "pixel_threshold": splice.PIXEL_THRESHOLD}

        return predict
    from app.detectors.image import fused

    require(
        torch.backends.mps.is_available(), "Frozen current JPEG baseline requires MPS; no CPU substitution"
    )
    model = fused.load_model(
        ROOT / "models/weights/fused" / fused.CHECKPOINT_FILE,
        ROOT / "models/weights/fused" / fused.BACKBONE_FILE,
        "mps",
    )

    def predict(image):
        prediction = model.predict(image)
        return prediction.raw_mask, {
            "image_gate": prediction.candidate_eligibility["eligible"],
            "image_score": prediction.image_score,
            "pixel_threshold": 0.5,
        }

    return predict


def score(trial, arm, fmt, number):
    from scripts.image_regional_training_v26 import PAIRS, source_views

    require(
        arm in {"candidate", "baseline"} and fmt in {"native", "jpeg75"}, "Explicit arm and format required"
    )
    frozen, chain, plan = lease(trial)
    selected = phase_tasks(calibration_tasks(plan), number)
    name = f"{arm}-{fmt}-{number:03d}"
    if number > 0:
        previous = trial / f"{arm}-{fmt}-{number - 1:03d}"
        previous_result = read(previous / "completed.json")
        previous_profile = read(trial / f"{arm}-{fmt}-{number - 1:03d}-profile/profile.json")
        require(
            previous_profile["child_exit_code"] == 0
            and previous_profile["guard"] is None
            and previous_profile["sample_errors"] == 0
            and sha(previous / "rows.jsonl") == previous_result["rows_sha256"],
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
    previous_id, views = None, None
    with (out / "rows.jsonl").open("x") as journal:
        for index, item in enumerate(selected):
            try:
                require(
                    cases[item["source_id"]]["role"] == "calibration", "Only calibration sources admitted"
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
                    and raw.min() >= 0
                    and raw.max() <= 1,
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
            "schema": "chipmunk.regional-calibration-producer/28",
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
    parser.add_argument("trial", type=Path)
    parser.add_argument("arm", choices=("candidate", "baseline"))
    parser.add_argument("format", choices=("native", "jpeg75"))
    parser.add_argument("number", type=int)
    args = parser.parse_args()
    score(args.trial, args.arm, args.format, args.number)
