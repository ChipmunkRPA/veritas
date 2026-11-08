"""Frozen complete regional image comparison against retained current selectors."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import image_relayformer_v18 as method
from scripts import run_image_iml_zoom_v16 as prior
from scripts.relayformer_native_v18 import Native, prepare

ROOT = prior.ROOT
OLD = ROOT / "backend/data/independent-evaluation/image-iml-zoom-v16"
TRIAL = ROOT / "backend/data/independent-evaluation/image-relayformer-v18"
CODE = tuple(
    dict.fromkeys(
        (
            *prior.FILES,
            "backend/scripts/image_relayformer_v18.py",
            "backend/scripts/run_relayformer_v18.py",
            "backend/scripts/review_relayformer_v18.py",
            "backend/scripts/relayformer_study_runtime_v18.py",
            "backend/scripts/relayformer_native_v18.py",
            "backend/scripts/inspect_relayformer_v18.py",
            "backend/scripts/prepare_relayformer_v18.py",
            "backend/tests/test_image_relayformer_study_v18.py",
            "backend/tests/test_relayformer_v18.py",
            "docs/image-relayformer-v18-protocol.md",
        )
    )
)
RULES = {
    "cases": 384,
    "point_cases": 96,
    "diagnostic_cases": 288,
    "point_groups": 24,
    "diagnostic_groups": 48,
    "canonical": "CPUfloat32,unmerged LoRA,2threads;every native patch input bound;full source float32 maps retained losslessly",
    "mask": "raw float32>=.5;8-connected;minimum ceil(sourcepixels/1000);cap2 by exact float32 mean,area,y,x;no uint8 quantization",
    "point": "Separate formats;source maximum across original/ordinary;strictly above second-largest of24 exact rational maxima and >=1/1000 source area;no AI/diagnostic point inference",
    "gates": "More useful IoU>=.3 edits and higher mean IoU in each format;preserve every current useful edit;at most1 extra original and1 extra ordinary flagged in each format;all cases and full replay complete",
    "repeats": "First source for every role/family/kind/format repeated exactly",
    "audit": "Reconstruct all384 inputs;re-infer all384 source maps exactly;separate OpenCV/IEEE-bit native component selection and independent mask counts;refit both genuine points;recompute every aggregate/gate",
    "bootstrap": "Paired48 diagnostic source groups,2000 draws,seed20261108;formats never independent extra sources",
    "admission": "Consumed-source development only;success permits separate fresh confirmation,never serving/confidence admission",
    "resources": "Sequential phases;900seconds;6GiB sampled treeRSS;minimum1536MiB available RAM/disk;failures retained;no overwrite/restart",
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def journal(role):
    return [json.loads(line) for line in (TRIAL / (role + ".jsonl")).read_text().splitlines()]


def runtime():
    names = ("torch", "timm", "numpy", "Pillow", "scipy", "einops")
    files = {
        name: prior.sha(Path(importlib.import_module(name).__file__))
        for name in (
            "torch.nn.functional",
            "timm.models.vision_transformer",
            "timm.layers.patch_embed",
            "timm.layers.pos_embed",
            "torch.nn.modules.activation",
            "scipy.ndimage",
        )
    }
    return {
        "packages": {name: importlib.metadata.version(name) for name in names},
        "implementations": files,
        "rotary_source": prior.sha(
            ROOT
            / "backend/data/independent-evaluation/relayformer-v18-runtime/packages/rotary_embedding_torch/rotary_embedding_torch.py"
        ),
    }


def identity():
    return {
        "code_sha256": {p: prior.sha(ROOT / p) for p in CODE},
        "runtime": runtime(),
        "rules": RULES,
        "old_freeze_sha256": prior.sha(OLD / "freeze.json"),
        "old_outcomes_sha256": prior.sha(OLD / "private-outcomes.json"),
        "checkpoint_acquisition_sha256": prior.sha(
            ROOT / "docs/evaluation-panels/relayformer-v18-acquisition-v2.json"
        ),
        "authored_native_sha256": prior.sha(
            ROOT / "docs/evaluation-panels/relayformer-v18-authored-native.json"
        ),
    }


def freeze():
    plan = prior.check(OLD)
    require = method.require
    require(not TRIAL.exists(), "New prospective study folder required")
    old = {r["id"]: r for r in read(OLD / "private-outcomes.json")}
    rows = []
    originals = {"point": set(), "diagnostic": set()}
    rgb_roles = {}
    for case in plan:
        data, rgb, truth = prior.case_source(case)
        hashed = prior.digest(rgb)
        earlier = rgb_roles.setdefault(hashed, (case["role"], case["source_id"]))
        require(earlier == (case["role"], case["source_id"]), "Exact rendered RGB crosses source/role groups")
        if case["kind"] == "original" and case["format"] == "native":
            originals[case["role"]].add(hashed)
        row = {
            **case,
            "input_sha256": hashlib.sha256(data).hexdigest(),
            "rgb_sha256": hashed,
            "truth_sha256": prior.digest(truth),
            "source_size": [rgb.shape[1], rgb.shape[0]],
        }
        if case["role"] == "diagnostic":
            current = prior.current_record(OLD, case, data, rgb, truth)
            require(
                current == old[case["id"]]["metrics"]["current"],
                "Retained canonical current reference changed",
            )
            row["current"] = current
        rows.append(row)
    require(
        not originals["point"] & originals["diagnostic"]
        and len(originals["point"]) == 24
        and len(originals["diagnostic"]) == 48,
        "Complete source-disjoint role support required",
    )
    smoke = read(ROOT / "docs/evaluation-panels/relayformer-v18-authored-native.json")
    require(
        smoke["implementation_sha256"] == prior.sha(ROOT / "backend/scripts/relayformer_native_v18.py")
        and smoke["completed_model_forwards"] == 6
        and smoke["project_media_forwards"] == 0,
        "Exact complete authored native contract required",
    )
    TRIAL.mkdir()
    write(TRIAL / "plan.json", rows)
    value = {
        "schema": "chipmunk.relayformer-freeze/18",
        **identity(),
        "plan_sha256": prior.sha(TRIAL / "plan.json"),
        "new_project_forwards": 0,
        "serving_admitted": False,
        "used_in_confidence": False,
    }
    write(TRIAL / "freeze.json", value)
    write(
        ROOT / "docs/evaluation-panels/relayformer-v18-freeze.json",
        {**value, "private_freeze_sha256": prior.sha(TRIAL / "freeze.json")},
    )
    print(
        json.dumps({"planned": 384, "point_inputs": 96, "diagnostic_inputs": 288, "new_project_forwards": 0}),
        flush=True,
    )


def check():
    lock = read(TRIAL / "freeze.json")
    method.require(
        all(lock[k] == v for k, v in identity().items())
        and prior.sha(TRIAL / "plan.json") == lock["plan_sha256"],
        "Frozen method/runtime/weights/source reference changed",
    )
    prior.check(OLD)
    return read(TRIAL / "plan.json")


def source(case):
    data, rgb, truth = prior.case_source(case)
    method.require(
        hashlib.sha256(data).hexdigest() == case["input_sha256"]
        and prior.digest(rgb) == case["rgb_sha256"]
        and prior.digest(truth) == case["truth_sha256"],
        "Frozen actual source pixels/mask changed",
    )
    return data, rgb, truth


def points():
    seal = read(TRIAL / "point-seal.json")
    method.require(
        all(prior.sha(TRIAL / n) == s for n, s in seal["files"].items()), "Genuine-only point seal changed"
    )
    return read(TRIAL / "points.json")


def run(role):
    plan = [r for r in check() if r["role"] == role]
    method.require(
        role in ("point", "diagnostic") and len(plan) == RULES[role + "_cases"], "Complete role required"
    )
    output = TRIAL / (role + ".jsonl")
    method.require(
        not output.exists() and not (TRIAL / (role + ".json")).exists(), "No phase overwrite/restart"
    )
    policies = points() if role == "diagnostic" else None
    private = TRIAL / "private"
    private.mkdir(exist_ok=True)
    native = None
    started = time.monotonic()
    receipt = {"role": role, "status": "failed", "freeze_sha256": prior.sha(TRIAL / "freeze.json")}
    rows = []
    seen = set()
    try:
        native = Native()
        with output.open("x") as stream:
            for i, case in enumerate(plan):
                row = {k: case[k] for k in ("id", "source_id", "role", "family", "format", "kind", "label")}
                row["status"] = "failed"
                try:
                    _, rgb, truth = source(case)
                    tensor, geometry = prepare(Image.fromarray(rgb))
                    raw, proof = native.predict(tensor, geometry)
                    key = case["family"], case["format"], case["kind"]
                    if key not in seen:
                        repeat, proof2 = native.predict(tensor, geometry)
                        method.require(
                            np.array_equal(raw, repeat) and proof == proof2,
                            "First context canonical repeat differs",
                        )
                        seen.add(key)
                    mask, facts = method.native_mask(raw)
                    measured = method.counts(mask, truth)
                    name = hashlib.sha256(case["id"].encode()).hexdigest() + ".npz"
                    path = private / name
                    method.require(not path.exists(), "Owned map artifact already exists")
                    np.savez_compressed(path, raw=raw, mask=np.packbits(mask))
                    row.update(
                        status="ok",
                        geometry=geometry,
                        native_proof=proof,
                        regions=facts,
                        raw_map_file=name,
                        raw_map_sha256=prior.sha(path),
                        raw_float32_sha256=prior.digest(raw),
                        source_mask_sha256=prior.digest(mask),
                        ungated={"candidate": measured},
                    )
                    if policies is not None:
                        selected = (
                            mask
                            if method.eligible(measured, policies[case["format"]])
                            else np.zeros(truth.shape, bool)
                        )
                        row["metrics"] = {
                            "candidate": method.counts(selected, truth),
                            "current": case["current"],
                        }
                except Exception as error:
                    row["reason"] = f"{type(error).__name__}: {error}"
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                if (i + 1) % 16 == 0:
                    print(
                        json.dumps(
                            {
                                "role": role,
                                "completed": i + 1,
                                "successful": sum(r["status"] == "ok" for r in rows),
                                "actual_model_forwards": native.completed_forwards,
                            }
                        ),
                        flush=True,
                    )
        good = sum(r["status"] == "ok" for r in rows)
        receipt.update(planned=len(plan), complete=good, failed=len(plan) - good)
        method.require(good == len(plan), "Incomplete phase retained; no skipped cases or silent retry")
        if role == "point":
            fitted = {
                fmt: method.point([r for r in rows if r["format"] == fmt], "candidate")
                for fmt in prior.FORMATS
            }
            write(TRIAL / "points.json", fitted)
        receipt["status"] = "complete"
    except BaseException:
        receipt["failure"] = traceback.format_exc()
        raise
    finally:
        receipt["duration_s"] = time.monotonic() - started
        if output.exists():
            receipt["journal_sha256"] = prior.sha(output)
        if native is not None:
            receipt.update(
                attempted_model_forwards=native.attempted_forwards,
                completed_model_forwards=native.completed_forwards,
            )
        write(TRIAL / (role + ".json"), receipt)
    if role == "point":
        write(
            TRIAL / "point-seal.json",
            {
                "files": {
                    name: prior.sha(TRIAL / name)
                    for name in ("freeze.json", "plan.json", "point.json", "point.jsonl", "points.json")
                },
                "diagnostic_started": False,
            },
        )
    print(json.dumps(receipt), flush=True)


def summary():
    check()
    points()
    rows = journal("diagnostic")
    record = read(TRIAL / "diagnostic.json")
    method.require(
        record["status"] == "complete"
        and record["journal_sha256"] == prior.sha(TRIAL / "diagnostic.jsonl")
        and all(r["status"] == "ok" for r in rows),
        "Every diagnostic outcome required",
    )
    result = method.aggregate(rows)
    write(TRIAL / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "point", "diagnostic", "summary"))
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "summary":
        summary()
    else:
        run(args.phase)
