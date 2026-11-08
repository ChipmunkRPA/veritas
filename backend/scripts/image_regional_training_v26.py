"""Prospectively grouped, full-resolution regional adaptation; private research only.

No serving import, downloads, role reassignment or implicit restart is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from scripts.relayformer_checkpointing_v24 import install
from scripts.relayformer_learning_v23 import configure, logits, need, regional_loss, sha_state
from scripts.relayformer_native_v18 import Native, prepare, tensor_sha

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
GROUPS = DATA / "image-legacy-authors-v25/private-groups.json"
PAIRS = DATA / "image-positive-pairs-v21-cache"
VOICE = ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json"
VOICE_SHA = "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf"
GROUPS_SHA = "17575bf2662b33ec3fcf9faeaf031286f596b753d7b30f2bcaea2d3a1d801b7a"
PAIR_PLAN_SHA = "e7fdb365bbae713c89eb27101bf8578defee695aeba29a5e59c41ce271c8a47a"
PAIR_ROWS_SHA = "c4ae1c532f4db1f26eb5165754c3591a87fc2c82a6ecb421c9cee9c523028489"
EPOCHS, PHASE_UPDATES, SEED = 6, 32, 20261108
CODE = (
    "backend/scripts/image_regional_training_v26.py",
    "backend/scripts/image_regional_training_runtime_v26.py",
    "backend/scripts/relayformer_native_v18.py",
    "backend/scripts/inspect_relayformer_v18.py",
    "backend/scripts/prepare_relayformer_v18.py",
    "backend/scripts/relayformer_learning_v23.py",
    "backend/scripts/relayformer_checkpointing_v24.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/data_resource_guard_v21.py",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def grouped_roles(groups):
    """Creator/near-duplicate components are indivisible; old-linked stays in fit."""
    sources = {r["key"]: r for r in groups["sources"]}
    need(len(sources) == len(groups["sources"]), "Duplicate source keys")
    flattened = [k for c in groups["components"] for k in c]
    need(
        len(flattened) == len(set(flattened)) and set(flattened) == set(sources),
        "Complete disjoint components required",
    )
    components = []
    for keys in groups["components"]:
        new = [sources[k] for k in keys if k.startswith("new/")]
        if not new:
            continue
        linked = any(sources[k]["previously_consumed"] for k in keys)
        components.append(
            {
                "component": digest(sorted(keys)),
                "source_ids": sorted(r["source_id"] for r in new),
                "old_linked": linked,
                "publisher_nontraining": any(r["publisher_role"] in {"validation", "testing"} for r in new),
            }
        )

    def rank(r):
        return digest(["chipmunk-regional-v26-role", r["component"]])

    confirmation = sorted(
        (r for r in components if not r["old_linked"] and r["publisher_nontraining"]), key=rank
    )[:48]
    need(len(confirmation) == 48, "48 eligible confirmation groups required")
    assigned = {r["component"]: "confirmation" for r in confirmation}
    pool = sorted((r for r in components if not r["old_linked"] and r["component"] not in assigned), key=rank)
    for role, quota in (("calibration", 32), ("point", 24), ("diagnostic", 48)):
        selected, pool = pool[:quota], pool[quota:]
        need(len(selected) == quota, "Insufficient independent source groups")
        assigned.update({r["component"]: role for r in selected})
    return sorted(
        ({**r, "role": assigned.get(r["component"], "fit")} for r in components), key=lambda r: r["component"]
    )


def schedule(rows):
    fit = [r for r in rows if r["role"] == "fit"]
    need(fit and len({r["source_id"] for r in fit}) == len(fit), "Unique fitting sources required")
    result = []
    for epoch in range(EPOCHS):
        ordered = sorted(fit, key=lambda r: digest(["chipmunk-regional-v26-order", epoch, r["source_id"]]))
        for r in ordered:
            result.append(
                {
                    "source_id": r["source_id"],
                    "epoch": epoch,
                    "format": "native" if epoch % 2 == 0 else "jpeg75",
                    "negative": ("original", "tone", "blur")[epoch // 2],
                }
            )
    return result


def prepare_study(folder, checkpoint_root):
    need(
        sha(GROUPS) == GROUPS_SHA
        and sha(PAIRS / "plan.json") == PAIR_PLAN_SHA
        and sha(PAIRS / "qualification/pairs.jsonl") == PAIR_ROWS_SHA,
        "Audited inputs changed",
    )
    need(sha(VOICE) == VOICE_SHA, "Protected serving audio calibration changed")
    groups = grouped_roles(read(GROUPS))
    need(len(groups) == 320 and sum(r["old_linked"] for r in groups) == 12, "Audited grouping counts changed")
    by_source = {sid: r for r in groups for sid in r["source_ids"]}
    rows = [
        {
            **r,
            "component": by_source[r["source_id"]]["component"],
            "role": by_source[r["source_id"]]["role"],
            "old_linked": by_source[r["source_id"]]["old_linked"],
        }
        for r in read(PAIRS / "plan.json")["cases"]
    ]
    need(len(rows) == len(by_source) == 397, "All 397 audited pairs required")
    folder.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": "chipmunk.regional-supervised-plan/26",
        "groups_sha256": GROUPS_SHA,
        "pair_plan_sha256": PAIR_PLAN_SHA,
        "pair_rows_sha256": PAIR_ROWS_SHA,
        "groups": groups,
        "rows": rows,
        "schedule": schedule(rows),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "fitting_epochs": EPOCHS,
        "phase_updates": PHASE_UPDATES,
        "seed": SEED,
        "serving_admitted": False,
    }
    write(folder / "plan.json", plan)
    public = {
        "schema": "chipmunk.regional-supervised-freeze/26",
        "private_plan_sha256": sha(folder / "plan.json"),
        "source_groups_sha256": GROUPS_SHA,
        "pair_plan_sha256": PAIR_PLAN_SHA,
        "pair_rows_sha256": PAIR_ROWS_SHA,
        "groups_by_role": dict(Counter(r["role"] for r in groups)),
        "photos_by_role": dict(Counter(r["role"] for r in rows)),
        "fit_updates": len(plan["schedule"]),
        "phases": (len(plan["schedule"]) + PHASE_UPDATES - 1) // PHASE_UPDATES,
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "protected_voice_sha256": VOICE_SHA,
        "project_source_neural_forwards_at_freeze": 0,
        "serving_admitted": False,
        "selection": "Use only the final six-epoch checkpoint; no best-epoch selection on held-out outcomes. Calibration and evaluation require a separate frozen readout before their first inference.",
    }
    write(folder / "freeze.json", public)
    print(json.dumps(public), flush=True)
    return public


def check(folder):
    freeze, plan = read(folder / "freeze.json"), read(folder / "plan.json")
    authorization = read(folder / "execution-start.json")
    need(
        authorization["public_freeze_sha256"] == sha(folder / "freeze.json")
        and len(authorization["remote_freeze_blob_sha"]) == 40
        and authorization["remote_verified_before_first_source_forward"] is True,
        "Remotely verified prospective freeze required",
    )
    need(sha(folder / "plan.json") == freeze["private_plan_sha256"], "Frozen roles or order changed")
    need(
        freeze["code_sha256"] == {p: sha(ROOT / p) for p in CODE},
        "Frozen training code changed; explicit successor required",
    )
    need(
        sha(GROUPS) == GROUPS_SHA
        and sha(PAIRS / "plan.json") == PAIR_PLAN_SHA
        and sha(PAIRS / "qualification/pairs.jsonl") == PAIR_ROWS_SHA,
        "Frozen source lineage changed",
    )
    need(sha(VOICE) == VOICE_SHA, "Protected audio calibration changed")
    need(plan["schedule"] == schedule(plan["rows"]), "Complete prescribed training schedule differs")
    return plan


def source_views(case, qualification):
    """Reproduce audited controls; bind truth before any compression."""
    original_path = ROOT / case["original"]
    edit = qualification["edited_image"]
    mask = qualification["compositing_mask"]
    edited_path, mask_path = ROOT / edit["root"] / edit["file"], ROOT / mask["root"] / mask["file"]
    truth_path = PAIRS / "qualification" / qualification["truth_mask"]
    for path, expected in (
        (original_path, case["original_sha256"]),
        (edited_path, edit["sha256"]),
        (mask_path, mask["sha256"]),
        (truth_path, qualification["truth_png_sha256"]),
    ):
        need(sha(path) == expected, "Source/edit/mask/truth bytes changed")
    with Image.open(original_path) as im:
        original = im.convert("RGB").copy()
    with Image.open(edited_path) as im:
        edited = im.convert("RGB").copy()
    with Image.open(mask_path) as im:
        alpha = im.convert("L").copy()
    with Image.open(truth_path) as im:
        truth = np.asarray(im.convert("L")).copy()
    rgb = np.asarray(original)
    need(
        hashlib.sha256(rgb.tobytes()).hexdigest() == case["source_rgb_sha256"], "Original RGB lineage changed"
    )
    changed = np.any(rgb != np.asarray(edited), axis=2)
    need(
        original.size == edited.size == alpha.size and truth.shape == rgb.shape[:2],
        "Source truth geometry differs",
    )
    need(
        np.array_equal(truth > 0, changed) and set(np.unique(truth)) <= {0, 255},
        "Every positive truth pixel must match the retained source-relative difference",
    )
    need(
        hashlib.sha256(changed.tobytes()).hexdigest() == qualification["proof"]["truth_mask_sha256"],
        "Truth response binding differs",
    )
    tone = np.clip(
        np.rint((rgb.astype(np.float64) - 127.5) * 1.12 + 127.5 + np.asarray([3, -2, 1])), 0, 255
    ).astype(np.uint8)
    controls = {
        "original": original,
        "tone": Image.composite(Image.fromarray(tone), original, alpha),
        "blur": Image.composite(original.filter(ImageFilter.GaussianBlur(1.25)), original, alpha),
    }
    for name in ("tone", "blur"):
        need(
            hashlib.sha256(np.asarray(controls[name]).tobytes()).hexdigest()
            == qualification["proof"]["ordinary_controls"][name]["rgb_sha256"],
            "Audited ordinary retouch control changed",
        )
    return edited, controls, truth


def compressed(image, fmt):
    need(fmt in {"native", "jpeg75"}, "Explicit native/JPEG75 format required")
    if fmt == "native":
        return image
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=75)
    with Image.open(io.BytesIO(encoded.getvalue())) as im:
        return im.convert("RGB").copy()


def training_view(image, truth, fmt, *, positive):
    tensor, geometry = prepare(compressed(image, fmt))
    fw, fh = geometry["fit_size"]
    target_image = Image.fromarray(truth).resize((fw, fh), Image.Resampling.NEAREST)
    target = (
        torch.from_numpy((np.asarray(target_image) > 0).astype(np.float32)[None, None])
        if positive
        else torch.zeros((1, 1, fh, fw), dtype=torch.float32)
    )
    if positive:
        need(
            bool(target.any()) and bool((target == 0).any()),
            "All source views must retain positive and background pixels",
        )
    return tensor, target, geometry


def checkpoint_predecessor(folder, plan, number):
    need(type(number) is int and number >= 0, "Nonnegative phase number required")
    if number == 0:
        return None, 0
    previous = folder / f"phase-{number - 1:03d}"
    receipt = read(previous / "completed.json")
    profile = read(folder / f"phase-{number - 1:03d}-profile/profile.json")
    need(
        profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0,
        "Previous phase needs a successful resource profile",
    )
    need(
        receipt["plan_sha256"] == sha(folder / "plan.json") and receipt["phase"] == number - 1,
        "Checkpoint belongs to another plan/phase",
    )
    need(sha(previous / "updates.jsonl") == receipt["updates_sha256"], "Previous update journal changed")
    path = Path(plan["checkpoint_root"]) / f"phase-{number - 1:03d}.pt"
    need(sha(path) == receipt["checkpoint_sha256"], "Previous private learned checkpoint changed")
    need(
        receipt["end_update"] == min(number * PHASE_UPDATES, len(plan["schedule"])),
        "Incomplete preceding phase",
    )
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
            state["plan_sha256"] == sha(folder / "plan.json")
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
        "schema": "chipmunk.private-regional-checkpoint/26",
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
        "schema": "chipmunk.regional-supervised-phase/26",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("folder", type=Path)
    p.add_argument("checkpoint_root", type=Path)
    p = sub.add_parser("train")
    p.add_argument("folder", type=Path)
    p.add_argument("number", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_study(args.folder, args.checkpoint_root)
    else:
        train(args.folder, args.number)
