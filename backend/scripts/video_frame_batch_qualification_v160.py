"""Measure the real batch-one video adapter against its batch-eight reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import types
from dataclasses import asdict
from pathlib import Path

from scripts import grid_serving_baseline_v156 as serving
from scripts.video_appearance_corrected_v112 import scoped

baseline = serving.baseline
ROOT, BASE, PANEL = serving.ROOT, serving.BASE, serving.PANEL
TRIAL = BASE / "video-frame-batch-qualification-v160"
WORK = BASE / "video-frame-batch-work-v160"
PUBLIC = PANEL / "video-frame-batch-qualification-freeze-v160.json"
INVENTORY = baseline.TRIAL / "private-inventory.json"
MODULE = "scripts.video_frame_batch_qualification_v160"
MODEL_IDS = ("dfdc.effnet_b7_555", "dfdc.effnet_b7_999")
ARMS = {"reference": 8, "batch-one": 1}
SAMPLE_COUNT = 15  # Eight inputs and a partial tail on the reference path.
LOGIT_ATOL, PROBABILITY_ATOL = 1e-3, 1e-4
CODE = (
    "backend/scripts/video_frame_batch_qualification_v160.py",
    "backend/scripts/video_frame_check_batch_one.py",
    "backend/tests/test_video_frame_batch_qualification_v160.py",
    "backend/tests/test_video_frame_check_batch_one.py",
    "docs/video-frame-batch-qualification-v160-method.md",
)
read, write, sha, require = serving.read, serving.write, serving.sha, serving.require


def original(rows):
    matches = [row for row in rows if row["condition"] == "original"]
    require(len(matches) == 1, "Exactly one frozen original required")
    return matches[0]


def exact_inputs(case, row):
    observed = row["sampled_input"]
    sampled = observed["sampled_frames"]
    faces = row["detector_stage_outputs"]["faces"]["faces"]
    require(
        case["status"] == "ready"
        and row["status"] == "done"
        and case["file_sha256"] == row["input_file_sha256"]
        and observed["clock_exact"] is observed["sampled_rgb_exact"] is True
        and len(sampled) == len(faces) == 16
        and [frame["index"] for frame in sampled] == list(range(16))
        and all(faces[:SAMPLE_COUNT]),
        "Expected exact ordered source frames and largest faces from serving",
    )
    return {
        "archive_case_file": str(Path(case["file"]).relative_to(baseline.clips.CACHE)),
        "file_sha256": case["file_sha256"],
        "frame_table": observed["probe_frame_table"],
        "frames": sampled[:SAMPLE_COUNT],
        "faces": faces[:SAMPLE_COUNT],
    }


def identity(inputs):
    old = read(baseline.PUBLIC)
    require(old["asset_inventory_sha256"] == sha(INVENTORY), "Original pinned asset inventory required")
    return {
        "schema": "chipmunk.video-frame-batch-qualification-freeze/160",
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "serving_freeze_sha256": sha(serving.PUBLIC),
        "baseline_freeze_sha256": sha(baseline.PUBLIC),
        "private_inputs_sha256": sha(inputs),
        "manifest_sha256": sha(baseline.MANIFEST),
        "sampled_face_crops": SAMPLE_COUNT,
        "batch_sizes": ARMS,
        "logit_absolute_tolerance": LOGIT_ATOL,
        "probability_absolute_tolerance": PROBABILITY_ATOL,
        "scope": "Same-device adapter parity on observed original crops and a partial reference batch.",
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every qualification attempt")
    rows = serving.load_completed(0)
    cases, _ = baseline.case_binding(0)
    inputs = exact_inputs(original(cases), original(rows))
    inputs["device"] = read(serving.stage_folder(0) / "output/result.json")["resolved_device"]
    require(inputs["device"] in {"cpu", "mps"}, "Observed local CPU or MPS required")
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "private-inputs.json", inputs)
    body = identity(TRIAL / "private-inputs.json")
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)


def verify():
    frozen = identity(TRIAL / "private-inputs.json")
    require(frozen == read(PUBLIC) == read(TRIAL / "freeze.json"), "Exact frozen qualification required")
    return frozen


def available_assets(inventory):
    models = [row for row in inventory["model_files"] if row["model"] in MODEL_IDS]
    require(
        len(models) == len(MODEL_IDS)
        and {row["model"] for row in models} == set(MODEL_IDS)
        and all(row["status"] == "available" for row in models),
        "Both original pinned checkpoints required",
    )
    return {"model_files": models, "local_fuser_files": []}


def expected_batches(batch):
    return [min(batch, SAMPLE_COUNT - start) for start in range(0, SAMPLE_COUNT, batch)]


def compare(reference, candidate):
    require(reference["device"] == candidate["device"], "Same-device reference required")
    require(reference["dtype"] == candidate["dtype"] == "float32", "Float32 comparison required")
    require(
        reference["model_revisions"] == candidate["model_revisions"],
        "Identical checkpoint revisions required",
    )
    require(
        len(reference["frames"]) == len(candidate["frames"]) == SAMPLE_COUNT,
        "Complete frame evidence required",
    )
    frame_identity = reference["frames"] == candidate["frames"]
    evidence = reference["evidence"] == candidate["evidence"]
    logit_error, crop_identity = [], True
    for model in MODEL_IDS:
        left, right = reference["members"][model], candidate["members"][model]
        require(
            left["batches"] == expected_batches(8)
            and right["batches"] == expected_batches(1)
            and len(left["logits"]) == len(right["logits"]) == SAMPLE_COUNT
            and len(left["crop_sha256"]) == len(right["crop_sha256"]) == SAMPLE_COUNT,
            "Exact complete member forwards required",
        )
        crop_identity = crop_identity and left["crop_sha256"] == right["crop_sha256"]
        logit_error.extend(abs(x - y) for x, y in zip(left["logits"], right["logits"], strict=True))
    lp, rp = reference["probabilities"], candidate["probabilities"]
    require(
        len(lp) == len(rp) == SAMPLE_COUNT
        and all(0 <= x <= 1 for x in [*lp, *rp, reference["aggregate"], candidate["aggregate"]]),
        "Complete ordered per-frame probabilities required",
    )
    probability_error = [abs(x - y) for x, y in zip(lp, rp, strict=True)]
    aggregate_error = abs(reference["aggregate"] - candidate["aggregate"])
    boundaries = [x >= 0.5 for x in [*lp, reference["aggregate"]]] == [
        x >= 0.5 for x in [*rp, candidate["aggregate"]]
    ]
    highest_frame = max(range(SAMPLE_COUNT), key=lp.__getitem__) == max(
        range(SAMPLE_COUNT), key=rp.__getitem__
    )
    require(
        all(math.isfinite(x) for x in [*logit_error, *probability_error, aggregate_error]),
        "Finite comparison required",
    )
    passed = all(
        (
            frame_identity,
            crop_identity,
            evidence,
            boundaries,
            highest_frame,
            max(logit_error) <= LOGIT_ATOL,
            max(probability_error) <= PROBABILITY_ATOL,
            aggregate_error <= PROBABILITY_ATOL,
        )
    )
    return {
        "paired_frames": SAMPLE_COUNT,
        "same_ordered_frames": frame_identity,
        "same_actual_float32_crops": crop_identity,
        "same_evidence_scope_and_times": evidence,
        "same_raw_boundary_decisions": boundaries,
        "same_highest_frame": highest_frame,
        "maximum_absolute_logit_error": max(logit_error),
        "maximum_absolute_probability_error": max(probability_error),
        "absolute_aggregate_error": aggregate_error,
        "parity_within_frozen_tolerances": passed,
    }


class EvidenceCapture:
    def __init__(self):
        self.metadata = []

    def series(self, series, **metadata):
        self.metadata.append(
            {
                **metadata,
                "series": [
                    {"name": row["name"], "time_s": [point[0] for point in row["points"]]} for row in series
                ],
            }
        )
        return "qualification-series"


def digest_array(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def worker(arm, output):
    frozen = verify()
    stage = TRIAL / arm / "authored"
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and not (stage / "profile.json").exists(),
        "Active owned resource monitor required",
    )
    inputs = read(TRIAL / "private-inputs.json")
    folder = WORK / arm
    scoped(serving.apfs.work_volume, {"WORK": WORK, "TRIAL": TRIAL})()
    folder.mkdir(parents=True, mode=0o700, exist_ok=False)
    output.mkdir(exist_ok=False)
    serving.initialize(folder, inputs["device"])
    (folder / "tmp").mkdir()
    baseline.isolate_filesystem(folder, output, baseline.checked_programs(read(baseline.PUBLIC)))
    baseline.install_assets(available_assets(read(INVENTORY)), folder)

    import torch

    from app.core.device import DEVICE
    from app.detectors.image.adapters.loaders import load_dfdc_b7
    from app.detectors.video.adapters.common import SampledFrame
    from app.detectors.video.adapters.frames import FrameDFDCB7Detector
    from app.detectors.video.decode import FrameTable, decode_frames
    from app.engine.boundary import Deadline
    from app.engine.detector import Profile, RunContext, StageOutputs
    from app.engine.fetchers import NoRecord, ensure_verified
    from app.engine.manifest import load_manifest
    from scripts.video_frame_check_batch_one import FrameDFDCB7BatchOneDetector

    require(str(DEVICE) == inputs["device"], "Observed serving device required")
    path = baseline.clips.owned_file(baseline.clips.CACHE / inputs["archive_case_file"], baseline.clips.CACHE)
    require(sha(path) == inputs["file_sha256"], "Exact source media bytes required")
    table = FrameTable.from_json(inputs["frame_table"])
    frames = []
    decoded = decode_frames(path, table, [row["frame_index"] for row in inputs["frames"]])
    for observed, frame in zip(inputs["frames"], decoded, strict=True):
        require(
            (frame.index, frame.pts, digest_array(frame.rgb))
            == (observed["frame_index"], observed["pts"], observed["sha256"]),
            "Exact frame order, time and bytes required",
        )
        frames.append(
            SampledFrame(
                observed["index"],
                frame.index,
                observed["time_s"],
                observed["pts_ms"],
                frame.rgb,
                frame.pts,
                table.time_base,
            )
        )

    manifest = load_manifest(baseline.MANIFEST)
    models, members = {}, {}
    for model in MODEL_IDS:
        entry = manifest.model(model)
        require(entry.licence.class_ == "commercial", "Commercial checkpoint required")
        ensure_verified(entry, record=NoRecord(), force=True)
        models[model] = load_dfdc_b7(model, str(DEVICE), "float32")
        members[model] = {"batches": [], "crop_sha256": [], "logits": []}

    def capture(model):
        def hook(_module, args, logits):
            tensor = args[0]
            require(tensor.dtype == torch.float32, "Actual float32 input required")
            row = members[model]
            row["batches"].append(tensor.shape[0])
            row["crop_sha256"].extend(digest_array(crop) for crop in tensor.detach().cpu().numpy())
            row["logits"].extend(logits.detach().float().cpu().flatten().tolist())

        return hook

    detector = FrameDFDCB7Detector() if arm == "reference" else FrameDFDCB7BatchOneDetector()
    exhibits = EvidenceCapture()
    ctx = RunContext(
        "qualification",
        None,
        Profile.FULL,
        DEVICE,
        torch.float32,
        types.SimpleNamespace(get=models.__getitem__),
        StageOutputs(frames=frames, faces=inputs["faces"]),
        Deadline(120),
        exhibits,
        None,
    )
    handles = [loaded.obj.register_forward_hook(capture(model)) for model, loaded in models.items()]
    try:
        result = detector.run(ctx)
    finally:
        for handle in handles:
            handle.remove()
    require(
        result.status == "ok" and result.aggregate is not None, "Complete actual detector result required"
    )
    require(
        all(row["batches"] == expected_batches(ARMS[arm]) for row in members.values()),
        "Unexpected actual batches",
    )
    receipt = {
        "status": "complete",
        "arm": arm,
        "freeze_sha256": sha(PUBLIC),
        "device": str(DEVICE),
        "dtype": "float32",
        "model_revisions": result.model_revisions,
        "members": members,
        "frames": [{**asdict(unit), "raw": None} for unit in result.unit_scores],
        "probabilities": [unit.raw for unit in result.unit_scores],
        "aggregate": result.aggregate.raw,
        "evidence": exhibits.metadata,
        "new_neural_calls": sum(len(row["batches"]) for row in members.values()),
        "private_inputs_sha256": frozen["private_inputs_sha256"],
        "mps_driver_allocated_bytes_end": torch.mps.driver_allocated_memory()
        if DEVICE.type == "mps"
        else None,
    }
    write(output / "result.json", receipt)


def run(arm):
    from scripts import video_appearance_runtime_v108 as runtime

    folder = TRIAL / arm
    folder.mkdir(exist_ok=True)

    def popen(argv, **kwargs):
        output = folder / "authored/output"
        require(
            argv[3:] == ["scripts.video_appearance_native_v108", "authored", str(output)],
            "Exact worker required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "_worker", arm, str(output)], **kwargs)

    scoped(
        runtime.run,
        {
            "TRIAL": folder,
            "verify": verify,
            "prerequisites": lambda *_: True,
            "subprocess": types.SimpleNamespace(
                Popen=popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
            ),
            "__file__": str(Path(__file__)),
        },
    )("authored")


def readout():
    verify()
    receipts, resources = {}, {}
    for arm, batch in ARMS.items():
        stage = TRIAL / arm / "authored"
        expected_calls = len(MODEL_IDS) * len(expected_batches(batch))
        receipt = baseline.speech.qualification_profile(stage, expected_calls, {}, Path(__file__).name)
        require(
            receipt["arm"] == arm and receipt["freeze_sha256"] == sha(PUBLIC), "Exact guarded arm required"
        )
        receipts[arm] = receipt
        profile = read(stage / "profile.json")
        resources[arm] = {
            "sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
            "worker_duration_seconds": profile["duration_s"],
            "maximum_forward_batch": max(max(row["batches"]) for row in receipt["members"].values()),
            "mps_driver_allocated_bytes_end": receipt["mps_driver_allocated_bytes_end"],
            "private_profile_sha256": sha(stage / "profile.json"),
            "private_worker_receipt_sha256": sha(stage / "output/result.json"),
        }
    return {
        "freeze_sha256": sha(PUBLIC),
        "device": receipts["reference"]["device"],
        "arms": resources,
        **compare(receipts["reference"], receipts["batch-one"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("freeze")
    actions.add_parser("readout")
    actions.add_parser("run").add_argument("arm", choices=ARMS)
    worker_parser = actions.add_parser("_worker")
    worker_parser.add_argument("arm", choices=ARMS)
    worker_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "readout":
        print(json.dumps(readout(), allow_nan=False))
    elif args.action == "run":
        run(args.arm)
    else:
        worker(args.arm, args.output)


if __name__ == "__main__":
    main()
