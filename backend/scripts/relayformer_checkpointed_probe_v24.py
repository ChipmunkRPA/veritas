"""Separate full-resolution authored arms; neither arm analyzes project images."""

import argparse
import json
from pathlib import Path

import torch

from scripts.relayformer_checkpointing_v24 import install, saved_activation_payload
from scripts.relayformer_learning_v23 import (
    Native,
    authored_image,
    configure,
    logits,
    need,
    prepare,
    regional_loss,
    sha_state,
)

MATH_KEYS = ("loss", "gradient_norm_before_clipping", "binding", "gradient_sha256", "updated_sha256")


def same_math(a, b):
    return all(a[k] == b[k] for k in MATH_KEYS)


def probe(arm, folder):
    need(arm in {"baseline", "checkpointed"}, "Explicit authored arm required")
    folder.mkdir(parents=True, exist_ok=False)
    counters = {"completed_authored_forwards": 0, "completed_backward_passes": 0, "completed_optimizer_updates": 0}

    def event(stage, **details):
        row = {"stage": stage, **counters, **details}
        with (folder / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

    event("starting", arm=arm)
    torch.manual_seed(20261108)
    model = Native().model
    parameters = configure(model)
    frozen = {n: p for n, p in model.named_parameters() if n not in parameters}
    initial = {n: p.detach().clone() for n, p in parameters.items()}
    initial_hashes, frozen_hashes = sha_state(initial), sha_state(frozen)
    trace = install(model) if arm == "checkpointed" else None
    image, target = authored_image()
    tensor, geometry = prepare(image)
    event("model_ready", selected_tensors=len(parameters))
    passes = []
    for repeat in range(2):
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(initial[name])
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=1e-4, weight_decay=0.01)
        event("forward_begin", repeat=repeat)
        with saved_activation_payload(model) as payload:
            values, binding = logits(model, tensor, geometry)
            loss = regional_loss(values, target)
        counters["completed_authored_forwards"] += 1
        event("forward_complete", repeat=repeat)
        with torch.no_grad():
            canonical = model(image=tensor, origin_shape=torch.tensor([[1024, 1024]]), clip_len=torch.tensor([1]))
            counters["completed_authored_forwards"] += 1
            parity = torch.equal(values.sigmoid(), canonical)
            event("canonical_complete", repeat=repeat, native_sigmoid_parity_exact=parity)
            need(parity, "Native inference output differs")
        event("backward_begin", repeat=repeat)
        loss.backward()
        counters["completed_backward_passes"] += 1
        gradients = {n: p.grad for n, p in parameters.items() if p.grad is not None}
        missing = sorted(set(parameters) - set(gradients))
        nonfinite = [n for n, g in gradients.items() if not bool(torch.isfinite(g).all())]
        event("backward_complete", repeat=repeat, missing=missing, nonfinite=nonfinite)
        need(not missing and not nonfinite and all(p.grad is None for p in frozen.values()), "Selected/frozen gradients differ")
        nonzero = {n for n, g in gradients.items() if bool(g.ne(0).any())}
        need(all(any(test in n for n in nonzero) for test in ("_loc.", "_glo.", "mask_decoder.")), "Local/global/decoder paths need actual gradients")
        gradient_hashes = sha_state(gradients)
        norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1, error_if_nonfinite=True)
        event("optimizer_begin", repeat=repeat)
        optimizer.step()
        counters["completed_optimizer_updates"] += 1
        event("optimizer_complete", repeat=repeat)
        after = sha_state(parameters)
        need(any(after[n] != initial_hashes[n] for n in nonzero), "No selected parameter changed")
        need(sha_state(frozen) == frozen_hashes, "A frozen dense/position parameter changed")
        passes.append({"loss": float(loss.detach()), "gradient_norm_before_clipping": float(norm), "binding": binding, "gradient_sha256": gradient_hashes, "updated_sha256": after, "saved_activation_payload": dict(payload)})
        event("repeat_complete", repeat=repeat, loss=float(loss.detach()), nonzero_gradient_tensors=len(nonzero))
        del optimizer, values, canonical, loss, gradients
    need(same_math(passes[0], passes[1]), "Reset output/gradient/update bits must repeat exactly")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(initial[name])
    need(sha_state(parameters) == initial_hashes and sha_state(frozen) == frozen_hashes, "Starting tensors were not restored")
    result = {"schema": "chipmunk.authored-regional-checkpoint-comparison/24", "arm": arm, **counters, "source_geometry": geometry, "initial_selected_sha256": initial_hashes, "frozen_state_sha256": frozen_hashes, "passes": passes, "checkpoint_trace": dict(trace) if trace is not None else None, "exact_repeat": True, "starting_tensors_restored": True, "native_sigmoid_parity_exact": True, "project_neural_forwards": 0, "project_fitting_steps": 0, "serving_admitted": False, "scope": "Full1024 authored mathematical target only. Payload counts exclude parameter storage but are not physical memory or peak RSS. Interrupted event counters are recorded completion lower bounds, not exact unfinished-operation counts. Accuracy and source-study admission are unassessed."}
    with (folder / "probe.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    event("completed")
    print(json.dumps({k: v for k, v in result.items() if k not in {"passes", "initial_selected_sha256", "frozen_state_sha256"}}), flush=True)


def compare(baseline, candidate, output):
    a, b = json.loads(baseline.read_text()), json.loads(candidate.read_text())
    need(a["arm"] == "baseline" and b["arm"] == "checkpointed", "Both complete explicitly named arms required")
    for row in (a, b):
        need(row["schema"] == "chipmunk.authored-regional-checkpoint-comparison/24" and row["completed_authored_forwards"] == 4 and row["completed_backward_passes"] == row["completed_optimizer_updates"] == 2, "Complete actual authored counters required")
        need(row["project_neural_forwards"] == row["project_fitting_steps"] == 0 and row["serving_admitted"] is False, "Authored comparison scope differs")
        need(row["source_geometry"]["source_size"] == row["source_geometry"]["fit_size"] == [1024, 1024], "Full-resolution authored source required")
        need(len(row["passes"]) == 2 and same_math(row["passes"][0], row["passes"][1]), "Both complete reset repeats must independently agree")
        names = set(row["initial_selected_sha256"])
        need(len(names) == 246 and all(set(p["gradient_sha256"]) == set(p["updated_sha256"]) == names and p["binding"]["actual_patch_count"] == 4 for p in row["passes"]), "Full selected gradient/update and four-patch correspondence required")
        need(all(type(p["saved_activation_payload"]["packed_nonparameter_bytes"]) is int and p["saved_activation_payload"]["packed_nonparameter_bytes"] >= 0 for p in row["passes"]), "Nonnegative actual saved-payload counts required")
    need(all(r["exact_repeat"] and r["starting_tensors_restored"] and r["native_sigmoid_parity_exact"] for r in (a, b)), "Each arm must independently pass")
    need(a["source_geometry"] == b["source_geometry"], "Authored source or normalization differs")
    need(b["checkpoint_trace"]["checkpointed_block_calls"] > 0 and b["checkpoint_trace"]["body_entries"] > b["checkpoint_trace"]["checkpointed_block_calls"], "Actual backward recomputation evidence required")
    need(a["initial_selected_sha256"] == b["initial_selected_sha256"] and a["frozen_state_sha256"] == b["frozen_state_sha256"], "Starting model states differ")
    need(len(a["passes"]) == len(b["passes"]) == 2 and all(same_math(x, y) for x, y in zip(a["passes"], b["passes"], strict=True)), "Complete cross-arm logits, gradients or updates differ")
    result = {"schema": "chipmunk.authored-regional-checkpoint-parity/24", "complete_cross_arm_math_exact": True, "payload_bytes_baseline": [p["saved_activation_payload"]["packed_nonparameter_bytes"] for p in a["passes"]], "payload_bytes_checkpointed": [p["saved_activation_payload"]["packed_nonparameter_bytes"] for p in b["passes"]], "project_neural_forwards": 0, "project_fitting_steps": 0, "scope": "Exact math comparison only; sampled process profiles and true detection quality require separate assessment."}
    result["packed_activation_payload_reduced_in_both_repeats"] = all(y < x for x, y in zip(result["payload_bytes_baseline"], result["payload_bytes_checkpointed"], strict=True))
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for arm in ("baseline", "checkpointed"):
        commands.add_parser(arm).add_argument("folder", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "compare":
        compare(args.baseline, args.candidate, args.output)
    else:
        probe(args.command, args.folder)
