"""Explicit continuation from the last successful v26 checkpoint after monitor failure.

The failed original batch stays quarantined. Frozen training mathematics and
source roles are reused; only the unaccepted suffix is fitted in a new study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts import image_regional_training_v26 as original
from scripts.image_regional_training_v26 import (
    CODE as ORIGINAL_CODE,
)
from scripts.image_regional_training_v26 import (
    PAIRS,
    PHASE_UPDATES,
    ROOT,
    SEED,
    VOICE,
    VOICE_SHA,
    read,
    sha,
    source_views,
    training_view,
    write,
)
from scripts.relayformer_checkpointing_v24 import install
from scripts.relayformer_learning_v23 import configure, logits, need, regional_loss, sha_state
from scripts.relayformer_native_v18 import Native, tensor_sha

FIRST_PHASE, PREFIX_UPDATES = 33, 1056
ORIGINAL_PLAN_SHA = "f38ec72f525edf0000f772c4bb75c36b031a796b1f0c2bebbe2260bde147a14a"
PREFIX_CHECKPOINT_SHA = "981828309f496ad8a54b228f4987c89f029dd1e2843be797e19085c65b67ab98"
CODE = tuple(dict.fromkeys((*ORIGINAL_CODE, "backend/scripts/image_regional_inference_v28.py",
    "backend/scripts/image_regional_continuation_v41.py", "backend/scripts/image_regional_continuation_runtime_v41.py",
    "backend/tests/test_image_regional_continuation_v41.py")))
require = need


def audit_prefix(study):
    """Verify every completed resource/checkpoint link before granting a lease."""
    plan, frozen = read(study / "plan.json"), read(study / "freeze.json")
    require(sha(study / "plan.json") == frozen["private_plan_sha256"], "Frozen training plan changed")
    require(
        all(sha(ROOT / path) == value for path, value in frozen["code_sha256"].items()),
        "Frozen training implementation changed",
    )
    require(len(plan["schedule"]) == 1278, "Complete six-epoch prescribed training required")
    count = FIRST_PHASE
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
        "completed_updates": PREFIX_UPDATES,
        "final_checkpoint": str(checkpoint),
        "final_checkpoint_sha256": previous,
        "final_parameter_sha256": previous_tensors,
    }


def failed_attempt(study):
    """Preserve the completed but guarded batch without accepting its state."""
    profile_path = study / "phase-033-profile/profile.json"
    result_path = study / "phase-033/completed.json"
    profile, result = read(profile_path), read(result_path)
    need(profile["child_exit_code"] == 0 and profile["guard"] == "live_sampling_unavailable" and profile["sample_errors"] == 1 and not profile["resource_limit_observations"], "Exact retained guarded original batch required")
    need(profile["sampling_error_events"] == [{"type": "NoSuchProcess", "classification": "unresolved_live_sampling_failure"}], "Preserve original unresolved sampling error")
    need(sha(profile_path.parent / "resource-samples.jsonl") == profile["resource_journal_sha256"], "Guarded resource journal changed")
    plan = read(study / "plan.json")
    need(result["start_update"] == 1056 and result["end_update"] == 1088 and result["held_out_source_forwards"] == 0, "Original guarded range differs")
    need(result["completed_optimizer_updates"] == 32 and result["completed_source_forwards"] == result["completed_backwards"] == 64, "Guarded attempt completion accounting differs")
    need(result["plan_sha256"] == sha(study / "plan.json") and result["predecessor_sha256"] == PREFIX_CHECKPOINT_SHA, "Guarded result does not follow the verified prefix")
    journal = study / "phase-033/updates.jsonl"
    need(sha(journal) == result["updates_sha256"], "Guarded update journal changed")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    need(len(rows) == 32 and [row["update"] for row in rows] == list(range(1056, 1088)), "Complete guarded source-exposure accounting required")
    roles = {row["source_id"]: row["role"] for row in plan["rows"]}
    for row in rows:
        need({key: row[key] for key in plan["schedule"][row["update"]]} == plan["schedule"][row["update"]] and roles[row["source_id"]] == "fit", "Guarded attempt touched a held-out source or changed view")
    checkpoint = Path(plan["checkpoint_root"]) / "phase-033.pt"
    need(sha(checkpoint) == result["checkpoint_sha256"], "Quarantined original checkpoint changed")
    need(not (study / "phase-034").exists() and not (study / "phase-034-profile").exists(), "Original fitting continued after guarded batch")
    return {"profile_sha256": sha(profile_path), "result_sha256": sha(result_path), "checkpoint_sha256": result["checkpoint_sha256"], "updates_sha256": sha(journal), "optimizer_updates": 32, "source_forwards": 64, "accepted_as_predecessor": False}


def prepare(previous, folder, checkpoint_root):
    plan = original.check(previous)
    need(sha(previous / "plan.json") == ORIGINAL_PLAN_SHA, "Exact prospective original study required")
    prefix = audit_prefix(previous)
    need(prefix["completed_updates"] == PREFIX_UPDATES and prefix["final_checkpoint_sha256"] == PREFIX_CHECKPOINT_SHA, "Last successful checkpoint differs")
    quarantine = failed_attempt(previous)
    need(Path("/Volumes/SEAGATE").is_mount(), "Seagate checkpoint volume must be mounted")
    need(checkpoint_root.parent.parent.resolve() == Path(plan["checkpoint_root"]).parent.parent.resolve() and not checkpoint_root.parent.exists(), "New separate checkpoint directory under authorized archive required")
    folder.mkdir(parents=True, exist_ok=False)
    checkpoint_root.parent.mkdir(exist_ok=False)
    checkpoint_root.mkdir(exist_ok=False)
    write(folder / "prefix-chain.json", prefix)
    continuation = {**plan, "schema": "chipmunk.regional-continuation-plan/41", "checkpoint_root": str(checkpoint_root.resolve()), "original_study": str(previous.resolve()), "original_plan_sha256": ORIGINAL_PLAN_SHA, "prefix_chain_sha256": sha(folder / "prefix-chain.json"), "accepted_prefix_updates": PREFIX_UPDATES, "first_continuation_phase": FIRST_PHASE, "quarantined_original_batch": quarantine}
    write(folder / "plan.json", continuation)
    freeze = {
        "schema": "chipmunk.regional-continuation-freeze/41", "private_plan_sha256": sha(folder / "plan.json"),
        "original_plan_sha256": ORIGINAL_PLAN_SHA, "original_freeze_sha256": sha(previous / "freeze.json"),
        "prefix_chain_sha256": sha(folder / "prefix-chain.json"), "prefix_checkpoint_sha256": PREFIX_CHECKPOINT_SHA,
        "accepted_prefix_phases": 33, "accepted_prefix_updates": PREFIX_UPDATES, "remaining_accepted_updates": 222,
        "continuation_phases": list(range(33, 40)), "total_accepted_training_updates": 1278,
        "discarded_original_attempt_updates": 32, "total_optimizer_updates_including_discarded_attempt": 1310,
        "new_continuation_source_forwards_at_freeze": 0, "held_out_source_forwards_at_freeze": 0,
        "quarantined_original_batch": quarantine, "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "protected_voice_sha256": VOICE_SHA, "serving_admitted": False,
        "selection": "Original six epochs, source roles, order, losses, optimizer and only-final-state selection unchanged. Resume from successful phase32; explicitly repeat unaccepted phase33 in a separate study. Never use or relabel the guarded original checkpoint. No held-out selection.",
    }
    write(folder / "freeze.json", freeze)
    print(json.dumps({"accepted_prefix_updates": PREFIX_UPDATES, "remaining_updates": 222, "freeze_sha256": sha(folder / "freeze.json")}), flush=True)


def check(folder):
    plan, frozen, start = read(folder / "plan.json"), read(folder / "freeze.json"), read(folder / "execution-start.json")
    need(start["public_freeze_sha256"] == sha(folder / "freeze.json") and start["remote_verified_before_first_continuation_forward"] is True and len(start["remote_freeze_blob_sha"]) == 40, "Remotely verified explicit continuation required")
    need(sha(folder / "plan.json") == frozen["private_plan_sha256"] and frozen["code_sha256"] == {path: sha(ROOT / path) for path in CODE}, "Frozen continuation plan or implementation changed")
    previous = Path(plan["original_study"])
    old = original.check(previous)
    need(sha(previous / "plan.json") == ORIGINAL_PLAN_SHA and plan["original_plan_sha256"] == ORIGINAL_PLAN_SHA, "Original prospective plan changed")
    need(plan["rows"] == old["rows"] and plan["groups"] == old["groups"] and plan["schedule"] == old["schedule"], "Source roles, groups or complete schedule changed")
    need(sha(folder / "prefix-chain.json") == plan["prefix_chain_sha256"] == frozen["prefix_chain_sha256"], "Accepted prefix chain changed")
    need(failed_attempt(previous) == plan["quarantined_original_batch"] == frozen["quarantined_original_batch"], "Failed original attempt changed or relabelled")
    return plan


def expected_predecessor_plan_sha(folder, number):
    return ORIGINAL_PLAN_SHA if number == FIRST_PHASE else sha(folder / "plan.json")


def checkpoint_predecessor(folder, plan, number):
    need(type(number) is int and FIRST_PHASE <= number < 40, "Explicit continuation phase33..39 required")
    origin = Path(plan["original_study"]) if number == FIRST_PHASE else folder
    previous = origin / f"phase-{number - 1:03d}"
    receipt = read(previous / "completed.json")
    profile_path = origin / f"phase-{number - 1:03d}-profile/profile.json"
    profile = read(profile_path)
    need(profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0 and not profile["resource_limit_observations"], "Only a successfully monitored predecessor may resume")
    need(profile["module"] == ("scripts.image_regional_training_v26" if number == FIRST_PHASE else "scripts.image_regional_continuation_v41") and profile["arguments"] == ["train", str(origin), str(number - 1)], "Predecessor worker identity differs")
    need(sha(profile_path.parent / "resource-samples.jsonl") == profile["resource_journal_sha256"], "Predecessor resource journal changed")
    need(receipt["plan_sha256"] == expected_predecessor_plan_sha(folder, number) and receipt["phase"] == number - 1, "Predecessor plan or phase differs")
    need(sha(previous / "updates.jsonl") == receipt["updates_sha256"] and receipt["end_update"] == number * PHASE_UPDATES, "Predecessor update range or journal changed")
    path = Path(read(origin / "plan.json")["checkpoint_root"]) / f"phase-{number - 1:03d}.pt"
    need(sha(path) == receipt["checkpoint_sha256"], "Successful predecessor checkpoint changed")
    if number == FIRST_PHASE:
        need(sha(path) == PREFIX_CHECKPOINT_SHA, "Guarded original state must never replace successful phase32")
    return path, receipt["end_update"]


def train(folder, number):
    plan = check(folder)
    previous, start = checkpoint_predecessor(folder, plan, number)
    end = min(start + PHASE_UPDATES, len(plan["schedule"]))
    need(start < end, "No scheduled updates remain")
    out = folder / f"phase-{number:03d}"
    out.mkdir(exist_ok=False)
    checkpoint_root = Path(plan["checkpoint_root"])
    need(checkpoint_root.is_dir(), "Prepared private checkpoint destination must be mounted")
    destination = checkpoint_root / f"phase-{number:03d}.pt"
    need(not destination.exists(), "Learned checkpoints are never overwritten")
    torch.manual_seed(SEED)
    model = Native().model
    parameters = configure(model)
    need(len(parameters) == 246, "Exact native adaptation parameter set required")
    frozen = {n: p for n, p in model.named_parameters() if n not in parameters}
    frozen_hashes = sha_state(frozen)
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=1e-4, weight_decay=0.01)
    if previous is not None:
        state = torch.load(previous, map_location="cpu", weights_only=True)
        need(
            state["plan_sha256"] == expected_predecessor_plan_sha(folder, number)
            and state["completed_updates"] == start
            and set(state["parameters"]) == set(parameters),
            "Bound complete predecessor state required",
        )
        with torch.no_grad():
            for name, p in parameters.items():
                value = state["parameters"][name]
                need(
                    value.shape == p.shape and value.dtype == p.dtype and bool(torch.isfinite(value).all()),
                    "Finite exact-shape learned tensors required",
                )
                p.copy_(value)
        optimizer.load_state_dict(state["optimizer"])
        del state
    initial_parameters = sha_state(parameters)
    trace = install(model)
    cases = {r["source_id"]: r for r in plan["rows"]}
    qualified = {
        r["source_id"]: r
        for r in (json.loads(line) for line in (PAIRS / "qualification/pairs.jsonl").read_text().splitlines())
    }
    counters = {"completed_source_forwards": 0, "completed_backwards": 0, "completed_optimizer_updates": 0}

    def event(stage, **extra):
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": stage, **counters, **extra}, allow_nan=False) + "\n")
            stream.flush()

    event("model_ready", start_update=start, end_update=end)
    with (out / "updates.jsonl").open("x") as journal:
        for index in range(start, end):
            item = plan["schedule"][index]
            case = cases[item["source_id"]]
            need(case["role"] == "fit", "Calibration/point/diagnostic/confirmation must never enter fitting")
            edited, controls, truth = source_views(case, qualified[case["source_id"]])
            optimizer.zero_grad(set_to_none=True)
            observations = []
            for positive, image in ((True, edited), (False, controls[item["negative"]])):
                tensor, target, geometry = training_view(image, truth, item["format"], positive=positive)
                event("forward_begin", update=index, positive=positive)
                values, binding = logits(model, tensor, geometry)
                loss = regional_loss(values, target) / 2
                counters["completed_source_forwards"] += 1
                event("forward_complete", update=index, positive=positive)
                loss.backward()
                counters["completed_backwards"] += 1
                event("backward_complete", update=index, positive=positive)
                observations.append(
                    {
                        "positive": positive,
                        "half_weighted_loss": float(loss.detach()),
                        "geometry": geometry,
                        "binding": binding,
                        "target_float32_sha256": tensor_sha(target),
                    }
                )
                del tensor, target, values, loss
            missing = [n for n, p in parameters.items() if p.grad is None]
            need(
                not missing and all(bool(torch.isfinite(p.grad).all()) for p in parameters.values()),
                "Every selected tensor needs a finite accumulated gradient",
            )
            need(all(p.grad is None for p in frozen.values()), "Frozen base received a gradient")
            norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1, error_if_nonfinite=True)
            optimizer.step()
            counters["completed_optimizer_updates"] += 1
            row = {
                "update": index,
                **item,
                "component": case["component"],
                "gradient_norm_before_clipping": float(norm),
                "loss": sum(r["half_weighted_loss"] for r in observations),
                "views": observations,
            }
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            event("update_complete", update=index, loss=row["loss"])
            print(
                json.dumps({"update_complete": index, "loss": row["loss"], "epoch": item["epoch"]}),
                flush=True,
            )
    need(sha_state(frozen) == frozen_hashes, "Dense base or fixed frequencies changed")
    after = sha_state(parameters)
    need(
        after != initial_parameters and all(bool(torch.isfinite(p).all()) for p in parameters.values()),
        "Actual finite learned adaptation required",
    )
    changed_names = {n for n in parameters if after[n] != initial_parameters[n]}
    need(
        all(any(part in n for n in changed_names) for part in ("_loc.", "_glo.", "mask_decoder.")),
        "Local/global backbone adaptations and decoder must all actually change",
    )
    need(sha(VOICE) == VOICE_SHA, "Serving calibration changed")
    state = {
        "schema": "chipmunk.private-regional-checkpoint/41",
        "plan_sha256": sha(folder / "plan.json"),
        "completed_updates": end,
        "parameters": {n: p.detach().clone() for n, p in parameters.items()},
        "optimizer": optimizer.state_dict(),
    }
    # Exclusive creation retains partial writes as failed evidence; never replaces.
    with destination.open("xb") as stream:
        torch.save(state, stream)
    roundtrip = torch.load(destination, map_location="cpu", weights_only=True)
    need(sha_state(roundtrip["parameters"]) == after, "Saved learned tensors failed exact readback")
    del roundtrip
    result = {
        "schema": "chipmunk.regional-supervised-phase/41",
        "phase": number,
        "plan_sha256": sha(folder / "plan.json"),
        "freeze_sha256": sha(folder / "freeze.json"),
        "predecessor_sha256": sha(previous) if previous else None,
        "checkpoint_sha256": sha(destination),
        "updates_sha256": sha(out / "updates.jsonl"),
        "start_update": start,
        "end_update": end,
        "scheduled_updates": len(plan["schedule"]),
        **counters,
        "selected_tensors": len(parameters),
        "selected_parameters": sum(p.numel() for p in parameters.values()),
        "changed_tensors": len(changed_names),
        "local_global_and_decoder_changed": True,
        "checkpoint_parameter_readback_exact": True,
        "initial_parameter_sha256": initial_parameters,
        "final_parameter_sha256": after,
        "dense_base_and_fixed_frequencies_unchanged": True,
        "checkpoint_trace": dict(trace),
        "held_out_source_forwards": 0,
        "serving_admitted": False,
    }
    write(out / "completed.json", result)
    event("completed")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"initial_parameter_sha256", "final_parameter_sha256"}
            }
        ),
        flush=True,
    )


def audit_continuation(study):
    """Verify every completed resource/checkpoint link before granting a lease."""
    check(study)
    plan, frozen = read(study / "plan.json"), read(study / "freeze.json")
    require(sha(study / "plan.json") == frozen["private_plan_sha256"], "Frozen training plan changed")
    require(
        all(sha(ROOT / path) == value for path, value in frozen["code_sha256"].items()),
        "Frozen training implementation changed",
    )
    require(len(plan["schedule"]) == 1278, "Complete six-epoch prescribed training required")
    count = (len(plan["schedule"]) + 31) // 32
    prefix = audit_prefix(Path(plan["original_study"]))
    previous, previous_tensors, phases = prefix["final_checkpoint_sha256"], prefix["final_parameter_sha256"], list(prefix["phases"])
    for number in range(FIRST_PHASE, count):
        part = study / f"phase-{number:03d}"
        require(
            (part / "completed.json").is_file(), "All training phases must finish before reserved inference"
        )
        result = read(part / "completed.json")
        profile_path = study / f"phase-{number:03d}-profile/profile.json"
        require(profile_path.is_file(), "Every training phase needs a terminal resource receipt")
        profile = read(profile_path)
        require(
            profile["module"] == "scripts.image_regional_continuation_v41"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("previous", type=Path)
    p.add_argument("folder", type=Path)
    p.add_argument("checkpoint_root", type=Path)
    p = commands.add_parser("train")
    p.add_argument("folder", type=Path)
    p.add_argument("number", type=int)
    p = commands.add_parser("audit")
    p.add_argument("folder", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.previous.resolve(), args.folder.resolve(), args.checkpoint_root.resolve())
    elif args.command == "train":
        train(args.folder.resolve(), args.number)
    else:
        write(args.output, audit_continuation(args.folder.resolve()))
