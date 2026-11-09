"""Prospectively fixed serving parity on consumed confirmation sources; not a new accuracy study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.image_native_confirmation_readout_v46 import producer_rows
from scripts.image_native_confirmation_v46 import comparison_tasks, encoded_view, lease, phase_tasks
from scripts.image_regional_training_v26 import PAIRS, ROOT, read, sha, source_views, write

DATA = ROOT / "backend/data/independent-evaluation"
TRIAL = DATA / "native-regional-serving-v47"
PUBLIC = ROOT / "docs/evaluation-panels/native-regional-serving-v47-freeze.json"
OWN = (
    "backend/app/detectors/image/regional.py",
    "backend/app/detectors/image/regional_support.py",
    "backend/app/detectors/image/adapters/regional.py",
    "backend/scripts/native_regional_serving_v47.py",
    "backend/scripts/native_regional_serving_runtime_v47.py",
    "backend/tests/test_native_regional_serving_v47.py",
    "docs/native-regional-serving-v47-protocol.md",
)


def need(value, reason):
    if not value:
        raise ValueError(reason)


def prepare():
    from app.detectors.image import regional
    from scripts.prepare_regional_calibration_v42 import runtime_versions

    parent = DATA / "image-native-confirmation-v46"
    frozen, _chain, plan = lease(parent)
    tasks = comparison_tasks(plan)
    references = []
    for n in range(10):
        selected = phase_tasks(tasks, n)
        a = producer_rows(parent, "candidate", n, selected)
        b = producer_rows(parent, "baseline", n, selected)
        references += [{"candidate": x, "baseline": y} for x, y in zip(a, b, strict=True)]
    checkpoint = ROOT / "models/weights/native-regional-v1" / regional.CHECKPOINT_FILE
    need(checkpoint.stat().st_size == regional.CHECKPOINT_BYTES and sha(checkpoint) == regional.CHECKPOINT_SHA256,
         "Verified complete safe state required")
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "plan.json", plan)
    write(TRIAL / "references.json", references)
    graph = ROOT / "backend/app/detectors/image/regional_graph"
    code = {**frozen["code_sha256"], **{p: sha(ROOT / p) for p in OWN},
            **{str(p.relative_to(ROOT)): sha(p) for p in graph.iterdir() if p.is_file()}}
    freeze = {
        "schema": "chipmunk.native-regional-serving-parity/47", "code_sha256": code,
        "checkpoint_sha256": regional.CHECKPOINT_SHA256, "checkpoint_bytes": regional.CHECKPOINT_BYTES,
        "parent_freeze_sha256": sha(parent / "freeze.json"),
        "parent_result_sha256": sha(parent / "comparison-summary.json"),
        "plan_sha256": sha(TRIAL / "plan.json"), "references_sha256": sha(TRIAL / "references.json"),
        "runtime_versions": runtime_versions(), "inputs": 236, "photos": 59, "components": 48,
        "source_role": "previously consumed confirmation; engineering parity only",
        "cpu_threads": 2, "dtype": "float32", "deterministic_algorithms": False,
        "absolute_map_difference_bound": 1e-5, "selected_masks_must_match_exactly": True,
        "gates_must_match_exactly": True, "source_input_and_geometry_must_match_exactly": True,
        "first_input_extra_repeat": True, "new_accuracy_or_calibration_claim": False,
        "source_nn_calls_before_freeze": 0, "serving_default_changed": False,
        "preparation_no_nn": True,
    }
    write(TRIAL / "freeze.json", freeze)
    write(PUBLIC, freeze)
    print(json.dumps({"prepared_inputs": len(references), "freeze_sha256": sha(PUBLIC), "neural_calls": 0}))


def admission():
    from scripts.prepare_regional_calibration_v42 import runtime_versions

    frozen = read(TRIAL / "freeze.json")
    start = read(TRIAL / "execution-start.json")
    need(sha(PUBLIC) == sha(TRIAL / "freeze.json") == start["public_freeze_sha256"]
         and start["remote_verified_before_first_source_forward"] is True
         and len(start["remote_freeze_blob_sha"]) == 40, "Exact remotely reviewed freeze required")
    need(all(sha(ROOT / p) == digest for p, digest in frozen["code_sha256"].items()), "Frozen code changed")
    need(runtime_versions() == frozen["runtime_versions"], "Frozen runtime changed")
    need(sha(TRIAL / "plan.json") == frozen["plan_sha256"]
         and sha(TRIAL / "references.json") == frozen["references_sha256"], "Private input binding changed")
    profile_path = TRIAL.parent / (TRIAL.name + "-preparation-profile/profile.json")
    need(sha(profile_path) == start["preparation_profile_sha256"], "Preparation receipt changed")
    profile = read(profile_path)
    need(profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0
         and not profile["resource_limit_observations"], "Successful bounded preparation required")
    return frozen


def phase(number):
    import torch
    from PIL import Image

    from app.detectors.image import regional, splice
    from app.detectors.image.regional_support import readout, shared_pixels
    from scripts.image_native_comparison_readout_v45 import source_mask
    from scripts.replay_regional_calibration_v43 import load_map

    frozen = admission()
    need(0 <= number < 10, "One explicit phase required")
    if number:
        prior = TRIAL / f"phase-{number - 1:03d}"
        need(read(prior / "completed.json")["status"] == "complete", "Previous successful phase required")
        profile = read(TRIAL / f"phase-{number - 1:03d}-profile/profile.json")
        need(profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0,
             "Previous bounded success required")
    out = TRIAL / f"phase-{number:03d}"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)  # owned offline worker startup, never a request
    torch.use_deterministic_algorithms(False)
    model = regional.load_model(ROOT / "models/weights/native-regional-v1" / regional.CHECKPOINT_FILE)
    references = read(TRIAL / "references.json")[number * 24:(number + 1) * 24]
    cases = {r["source_id"]: r for r in read(TRIAL / "plan.json")["rows"]}
    qualified = {r["source_id"]: r for r in map(json.loads, (PAIRS / "qualification/pairs.jsonl").read_text().splitlines())}
    raw_root = Path(read(DATA / "image-native-confirmation-v46/private-paths.json")["raw_map_root"])
    attempted = completed = 0
    maximum = 0.
    with (out / "rows.jsonl").open("x") as journal:
        for index, pair in enumerate(references):
            a, b = pair["candidate"], pair["baseline"]
            item = {k: a[k] for k in ("source_id", "kind", "component")}
            try:
                edited, controls, _truth = source_views(cases[a["source_id"]], qualified[a["source_id"]])
                data, image = encoded_view({"ai": edited, **controls}[a["kind"]], "native")
                need(hashlib.sha256(data).hexdigest() == a["input_sha256"], "Exact saved input required")
                old = load_map(a, raw_root, image.size[::-1])
                baseline = load_map(b, raw_root, (384, 384))
                attempted += 1
                prediction = model.predict(image)
                completed += 1
                need(prediction.geometry == a["geometry"], "Native geometry or input binding differs")
                need(prediction.binding["actual_patch_float32_sha256"] == a["binding"]["actual_patch_float32_sha256"],
                     "Actual native patches changed")
                delta = float(np.max(np.abs(prediction.raw_mask - old)))
                maximum = max(maximum, delta)
                need(delta <= frozen["absolute_map_difference_bound"], "Serving raw-map parity failed")
                old_gate = int(np.count_nonzero(old > .90)) >= int(np.ceil(.01 * old.size))
                need(prediction.candidate_eligibility["eligible"] == old_gate, "Frozen selection gate differs")
                pixels = np.asarray(image)
                measured = readout(prediction.raw_mask, old_gate, .90, data, pixels)
                current_gate = int(np.count_nonzero(baseline > splice.PIXEL_THRESHOLD)) >= splice.MINIMUM_ACTIVE_CELLS
                current = readout(baseline, current_gate, splice.PIXEL_THRESHOLD, data, pixels)
                expected = source_mask(old, old_gate, .90, data, pixels, np.zeros(old.shape, bool))
                expected_current = source_mask(baseline, current_gate, splice.PIXEL_THRESHOLD, data, pixels,
                                               np.zeros(old.shape, bool))
                need(np.array_equal(measured.source_mask(*image.size), expected), "Selected regional pixels differ")
                need(np.array_equal(current.source_mask(*image.size), expected_current), "Baseline retained pixels differ")
                shared = shared_pixels(measured, current)
                shared_source = np.asarray(Image.fromarray(shared.astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)) > 0
                need(np.array_equal(shared_source, expected & expected_current), "Actual shared pixels differ")
                if number == index == 0:
                    attempted += 1
                    repeat = model.predict(image)
                    completed += 1
                    need(np.array_equal(prediction.raw_mask, repeat.raw_mask) and prediction.binding == repeat.binding,
                         "First serving input must repeat exactly")
                row = {**item, "status": "ok", "maximum_map_difference": delta,
                       "exact_raw_map": bool(np.array_equal(prediction.raw_mask, old)),
                       "geometry_exact": True, "gate_exact": True, "selected_pixels_exact": True,
                       "baseline_pixels_exact": True, "shared_pixels_exact": True}
            except BaseException as error:
                journal.write(json.dumps({**item, "status": "failed", "error_type": type(error).__name__,
                                          "attempted_forwards": attempted, "completed_forwards": completed}) + "\n")
                journal.flush()
                raise
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            print(json.dumps({"phase": number, "inputs_completed": index + 1, "maximum_difference": maximum}), flush=True)
    result = {"status": "complete", "phase": number, "inputs": len(references),
              "attempted_forwards": attempted, "completed_forwards": completed,
              "maximum_map_difference": maximum, "rows_sha256": sha(out / "rows.jsonl"),
              "freeze_sha256": sha(TRIAL / "freeze.json")}
    write(out / "completed.json", result)


def seal():
    admission()
    rows, profiles, calls = [], [], 0
    for n in range(10):
        part = TRIAL / f"phase-{n:03d}"
        result = read(part / "completed.json")
        profile_path = TRIAL / f"phase-{n:03d}-profile/profile.json"
        profile = read(profile_path)
        need(result["status"] == "complete" and sha(part / "rows.jsonl") == result["rows_sha256"], "Complete inputs required")
        need(profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0
             and not profile["resource_limit_observations"], "Every phase must succeed within resources")
        need(result["attempted_forwards"] == result["completed_forwards"] == result["inputs"] + (n == 0),
             "Neural call counts differ")
        calls += result["completed_forwards"]
        rows += list(map(json.loads, (part / "rows.jsonl").read_text().splitlines()))
        profiles.append({"phase": n, "profile_sha256": sha(profile_path), "duration_s": profile["duration_s"],
                         "sampled_tree_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"]})
    need(len(rows) == 236 and calls == 237 and all(r["status"] == "ok" for r in rows), "Full fixed parity cohort required")
    summary = {"schema": "chipmunk.native-regional-serving-parity-result/47", "status": "complete",
               "inputs": 236, "photos": 59, "source_groups": 48, "completed_neural_calls": calls,
               "exact_raw_maps": sum(r["exact_raw_map"] for r in rows),
               "maximum_map_difference": max(r["maximum_map_difference"] for r in rows),
               "exact_geometry_gate_and_selected_masks": True, "exact_baseline_and_shared_masks": True,
               "freeze_sha256": sha(TRIAL / "freeze.json"), "profiles": profiles,
               "serving_default_changed": False, "authorship_confidence_contribution": 0,
               "new_accuracy_or_calibration_claim": False, "github_workflows_started": 0}
    write(TRIAL / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "phase", "seal"))
    parser.add_argument("number", type=int, nargs="?")
    args = parser.parse_args()
    {"prepare": prepare, "phase": lambda: phase(args.number), "seal": seal}[args.command]()
