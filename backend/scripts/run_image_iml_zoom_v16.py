"""Source-frozen whole-versus-magnified image comparison, research only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.detectors.image import fused, splice
from scripts import image_iml_zoom_v16 as method
from scripts import iml_vit_candidate as iml
from scripts import review_image_iml_zoom_v16 as independent
from scripts.compare_image_localizers import measure
from scripts.evaluate_iml_vit import input_bytes
from scripts.localization_benchmark import mask_metrics, read_manifest, source
from scripts.score_splice_localizers import score_map

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "backend/data/localization-eval/iml-vit-candidate"
POINT = ROOT / "backend/data/localization-eval/tgif-pairs-v2"
DIAG = ROOT / "backend/data/localization-eval/tgif-confirmation-v4"
FAMILIES, FORMATS = ("firefly", "sd2", "flux-fill"), ("native", "jpeg75")
FILES = (
    "backend/scripts/image_iml_zoom_v16.py",
    "backend/scripts/run_image_iml_zoom_v16.py",
    "backend/scripts/review_image_iml_zoom_v16.py",
    "backend/scripts/image_iml_zoom_runtime_v16.py",
    "backend/tests/test_image_iml_zoom_v16.py",
    "docs/image-iml-zoom-v16-protocol.md",
    "backend/scripts/iml_vit_candidate.py",
    "backend/scripts/evaluate_iml_vit.py",
    "backend/scripts/score_splice_localizers.py",
    "backend/scripts/audit_fused_production_grid.py",
    "backend/scripts/compare_image_localizers.py",
    "backend/scripts/localization_benchmark.py",
    "backend/app/detectors/image/localiser.py",
    "backend/app/detectors/image/jpeg_bitstream.py",
    "backend/app/detectors/image/fused.py",
    "backend/app/detectors/image/splice.py",
    "backend/app/evidence/image_packets.py",
    "backend/scripts/specialist_runtime_v4.py",
)
require = method.require


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def lines(path):
    return [json.loads(v) for v in Path(path).read_text().splitlines()]


def plans():
    point, point_sha = read_manifest(POINT)
    diag, diag_sha = read_manifest(DIAG)
    old = read(ASSETS / "development/policy.json")
    require(
        old["manifest_sha256"] == point_sha
        and read(ASSETS / "confirmation/policy.json")["manifest_sha256"] == diag_sha,
        "Original exact source manifests required",
    )
    chosen = set(old["source_groups"])
    require(
        len(chosen) == 24 and not chosen & {r["source_id"] for r in diag["items"]},
        "Disjoint24/48 source roles required",
    )
    output = []
    for role, root, items in [
        ("point", POINT, [r for r in point["items"] if r["source_id"] in chosen and r["label"] == 0]),
        ("diagnostic", DIAG, diag["items"]),
    ]:
        for item in sorted(items, key=lambda r: (r["source_id"], r["label"])):
            kinds = ["original", "ordinary"] if not item["label"] else ["ai"]
            for kind in kinds:
                for fmt in FORMATS:
                    output.append(
                        {
                            "id": f"{role}-{item['family']}-{item['source_id']}-{kind}-{fmt}",
                            "role": role,
                            "family": item["family"],
                            "source_id": item["source_id"],
                            "kind": kind,
                            "format": fmt,
                            "label": int(kind == "ai"),
                            "root": str(root.relative_to(ROOT)),
                            "item": item,
                        }
                    )
    require(len(output) == len({r["id"] for r in output}) == 384, "Complete unique384-file plan required")
    return output


def identity():
    plan = plans()
    old_point = read(ASSETS / "development/policy.json")
    require(
        sha(ROOT / "backend/scripts/iml_vit_candidate.py") == old_point["runtime_sha256"],
        "Original candidate math changed",
    )
    for fmt in FORMATS:
        policy = read(ASSETS / ("current-native" if fmt == "native" else "current-jpeg75") / "policy.json")
        module = splice if fmt == "native" else fused
        require(policy["runtime_sha256"] == sha(Path(module.__file__)), "Current reference runtime changed")
    assets = {
        str(p.relative_to(ROOT)): sha(p)
        for p in [
            ASSETS / "iml-vit_checkpoint.pth",
            ASSETS / "source/modules/window_attention_ViT.py",
            ROOT / "models/weights/splice" / splice.CHECKPOINT_FILE,
            ROOT / "models/weights/fused" / fused.CHECKPOINT_FILE,
            ROOT / "models/weights/fused" / fused.BACKBONE_FILE,
        ]
    }
    expected = [
        iml.CHECKPOINT_SHA256,
        iml.ENCODER_SHA256,
        splice.CHECKPOINT_SHA256,
        fused.CHECKPOINT_SHA256,
        fused.BACKBONE_SHA256,
    ]
    require(list(assets.values()) == expected, "Exact permitted inference assets required")
    code = {name: sha(ROOT / name) for name in FILES}
    deps = {str(p.relative_to(ASSETS)): sha(p) for p in sorted((ASSETS / "deps").rglob("*.py"))}
    cached = {
        str(p.relative_to(ASSETS)): sha(p)
        for name in ("development", "confirmation", "current-native", "current-jpeg75")
        for p in [ASSETS / name / "policy.json", ASSETS / name / "predictions.jsonl"]
    }
    source_files = {
        str((ROOT / r["root"] / r["item"]["image"]).relative_to(ROOT)): r["item"]["image_sha256"]
        for r in plan
    }
    source_files.update(
        {
            str((ROOT / r["root"] / r["item"]["mask"]).relative_to(ROOT)): r["item"]["mask_sha256"]
            for r in plan
            if r["item"]["label"]
        }
    )
    require(
        all(sha(ROOT / name) == value for name, value in source_files.items()),
        "Every original source/mask must remain",
    )
    return {
        "schema": "chipmunk.image-iml-zoom-freeze/16",
        "code_sha256": code,
        "assets_sha256": assets,
        "isolated_python_source_sha256": deps,
        "cached_journals_sha256": cached,
        "source_files_sha256": source_files,
        "plan": plan,
        "runtime": {
            name: importlib.metadata.version(name) for name in ("torch", "numpy", "Pillow", "scipy", "timm")
        },
        "cv2": cv2.__version__,
        "device": "mps",
        "dtype": "float32",
        "torch_cpu_threads": 2,
        "serving_admitted": False,
        "used_in_confidence": False,
    }


def freeze(trial):
    value = identity()
    trial.mkdir(exist_ok=False)
    write(trial / "freeze.json", value)
    public = {k: v for k, v in value.items() if k not in ("source_files_sha256", "plan")}
    public.update(
        private_freeze_sha256=sha(trial / "freeze.json"),
        planned_file_cases=384,
        point_source_groups=24,
        diagnostic_source_groups=48,
        source_files_commitment=hashlib.sha256(
            json.dumps(value["source_files_sha256"], sort_keys=True).encode()
        ).hexdigest(),
        plan_commitment=hashlib.sha256(json.dumps(value["plan"], sort_keys=True).encode()).hexdigest(),
        new_native_forwards=0,
    )
    write(ROOT / "docs/evaluation-panels/image-iml-zoom-v16-freeze.json", public)
    print({"planned_cases": 384, "new_native_forwards": 0}, flush=True)


def check(trial):
    lock = read(trial / "freeze.json")
    require(lock == identity(), "Prospective code, weights, inputs, runtime or policy changed")
    return lock["plan"]


def case_source(case):
    data, rgb, truth = source(ROOT / case["root"], case["item"])
    if case["kind"] == "ordinary":
        rgb = method.ordinary(rgb)
        data = method.image_bytes(rgb)
        truth = np.zeros(rgb.shape[:2], bool)
    data, rgb = input_bytes(rgb, data, case["format"])
    return data, rgb, truth


def cached(case, current=False):
    if current:
        folder = ASSETS / ("current-native" if case["format"] == "native" else "current-jpeg75")
    else:
        folder = ASSETS / ("development" if case["role"] == "point" else "confirmation")
    matches = [
        r
        for r in lines(folder / "predictions.jsonl")
        if r["id"] == case["item"]["id"] and r["condition"] == case["format"]
    ]
    require(len(matches) == 1 and matches[0]["status"] == "ok", "Original complete cached response required")
    row = matches[0]
    require(
        all(
            row[k] == case["item"][k] for k in ("source_id", "family", "label", "image_sha256", "mask_sha256")
        ),
        "Cached source identity changed",
    )
    path = folder / row["raw_map"]
    require(path.parent == folder and sha(path) == row["raw_map_sha256"], "Cached native map changed")
    return np.load(path, allow_pickle=False), row


def check_points(trial):
    record = read(trial / "point-seal.json")
    require(
        record["sha256"]
        == {
            n: sha(trial / n)
            for n in ("points.json", "point-native.jsonl", "point-jpeg75.jsonl", "freeze.json")
        },
        "Genuine-only points or source journals changed",
    )


def reduce(raw):
    result = method.encode(raw)
    error = float(np.max(np.abs(method.resize(raw, 256, 256) - independent.resize(raw, 256, 256))))
    require(error <= 1e-12, "Independent complete native-map reduction differs")
    return result, error


def native(trial, phase):
    plan = check(trial)
    point = phase.startswith("point-")
    fmt = phase.split("-")[-1]
    family = None if point else phase[5:].rsplit("-", 1)[0]
    if not point:
        check_points(trial)
    selected = [
        r
        for r in plan
        if r["role"] == ("point" if point else "diagnostic")
        and r["format"] == fmt
        and (family is None or r["family"] == family)
    ]
    require(len(selected) == 48, "Every planned48 phase inputs required")
    private = trial / "private"
    private.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    model = iml.load(ASSETS, "mps")
    attempted, successful, complete = 0, 0, 0

    def predict(rgb):
        nonlocal attempted, successful
        wanted = digest(iml.preprocess(rgb).numpy())
        bound = []

        def capture(_module, args):
            actual = args[0].detach().cpu().contiguous().numpy()
            require(
                actual.dtype == np.float32
                and actual.shape == (1, 3, 1024, 1024)
                and digest(actual) == wanted,
                "Actual normalized model input differs from source/crop",
            )
            bound.append(wanted)

        hook = model.register_forward_pre_hook(capture)
        attempted += 1
        try:
            raw = iml.predict(model, rgb, "mps").copy()
        finally:
            hook.remove()
        require(bound == [wanted], "Exactly one bound forward required")
        successful += 1
        return raw, wanted

    with (trial / (phase + ".jsonl")).open("x") as journal:
        for index, case in enumerate(selected):
            record = {k: case[k] for k in ("id", "role", "family", "source_id", "kind", "format", "label")}
            record["status"] = "failed"
            try:
                data, rgb, truth = case_source(case)
                h, w = rgb.shape[:2]
                require(max(h, w) <= 1024, "Whole source exceeds original model scope")
                proofs, grids, errors = [], [], []
                if case["kind"] == "ordinary":
                    full, bound = predict(rgb)
                    full_proof = {
                        "source": "new_native",
                        "model_input_sha256": bound,
                        "raw_float32_sha256": digest(full),
                    }
                else:
                    full, old = cached(case)
                    require(
                        old["input_sha256"] == hashlib.sha256(data).hexdigest(),
                        "Whole-map bytes differ from exact new input",
                    )
                    full_proof = {
                        "source": "retained_native",
                        "old_map_sha256": old["raw_map_sha256"],
                        "raw_float32_sha256": digest(full),
                    }
                require(full.shape == (h, w), "Whole map source geometry differs")
                q, e = reduce(full)
                grids.append(q)
                errors.append(e)
                if index == 0:
                    a, _ = predict(rgb)
                    b, _ = predict(rgb)
                    require(
                        np.array_equal(a, b)
                        and np.max(np.abs(a - full)) <= 1e-4
                        and np.array_equal(method.encode(a), q),
                        "First complete native/full repeat or old-map parity failed",
                    )
                    full_proof["native_exact_repeat_and_recorded_grid_parity"] = True
                for tile_index, box in enumerate(method.boxes(w, h)):
                    pixels = method.zoom(rgb, box)
                    raw, bound = predict(pixels)
                    if index == 0 and tile_index == 0:
                        repeat, _ = predict(pixels)
                        require(
                            np.array_equal(raw, repeat), "First native zoom prediction failed exact repeat"
                        )
                    q, e = reduce(raw)
                    grids.append(q)
                    errors.append(e)
                    proofs.append(
                        {
                            "box": box,
                            "resized_shape": list(pixels.shape),
                            "rgb_sha256": digest(pixels),
                            "model_input_sha256": bound,
                            "raw_float32_sha256": digest(raw),
                        }
                    )
                grids = np.stack(grids)
                maps = method.reconstruct(grids, w, h)
                responses = {
                    arm: {
                        "active_fraction": method.fraction(value),
                        "ungated": score_map(value, True, 0.5, data, rgb, truth)["ungated"],
                    }
                    for arm, value in maps.items()
                }
                name = hashlib.sha256(case["id"].encode()).hexdigest() + ".npz"
                np.savez_compressed(private / name, grids=grids)
                record.update(
                    status="ok",
                    input_sha256=hashlib.sha256(data).hexdigest(),
                    rgb_sha256=digest(rgb),
                    width=w,
                    height=h,
                    truth_positive_pixels=int(truth.sum()),
                    grids_file=name,
                    grids_sha256=sha(private / name),
                    full_proof=full_proof,
                    tile_proofs=proofs,
                    reduction_max_difference=max(errors),
                    responses=responses,
                )
            except Exception as error:
                record["reason"] = f"{type(error).__name__}: {error}"
            journal.write(json.dumps(record, allow_nan=False) + "\n")
            journal.flush()
            complete += record["status"] == "ok"
            if index == 0 or (index + 1) % 8 == 0:
                print(
                    {
                        "phase": phase,
                        "inputs": index + 1,
                        "complete": complete,
                        "successful_forwards": successful,
                    },
                    flush=True,
                )
    write(
        trial / (phase + "-complete.json"),
        {
            "planned": 48,
            "complete": complete,
            "failed": 48 - complete,
            "attempted_native_forwards": attempted,
            "successful_native_forwards": successful,
            "journal_sha256": sha(trial / (phase + ".jsonl")),
            "freeze_sha256": sha(trial / "freeze.json"),
        },
    )
    require(complete == 48, "Incomplete phase retained; do not continue or silently restart")


def seal(trial):
    check(trial)
    points = {}
    for fmt in FORMATS:
        phase = "point-" + fmt
        rows = lines(trial / (phase + ".jsonl"))
        complete = read(trial / (phase + "-complete.json"))
        require(
            complete["complete"] == len(rows) == 48
            and complete["failed"] == 0
            and complete["journal_sha256"] == sha(trial / (phase + ".jsonl")),
            "Complete genuine-only source phase required",
        )
        points[fmt] = method.point(rows)
    write(trial / "points.json", points)
    write(
        trial / "point-seal.json",
        {
            "sha256": {
                n: sha(trial / n)
                for n in ("points.json", "point-native.jsonl", "point-jpeg75.jsonl", "freeze.json")
            },
            "diagnostic_processed": False,
        },
    )
    print({"points_locked": points, "diagnostic_processed": False}, flush=True)


def reference(trial, phase):
    plan = check(trial)
    check_points(trial)
    fmt = phase.split("-")[-1]
    selected = [
        r for r in plan if r["role"] == "diagnostic" and r["kind"] == "ordinary" and r["format"] == fmt
    ]
    require(len(selected) == 48, "All ordinary diagnostic reference inputs required")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    if fmt == "native":
        model = splice.load_model(ROOT / "models/weights/splice" / splice.CHECKPOINT_FILE)
    else:
        model = fused.load_model(
            ROOT / "models/weights/fused" / fused.CHECKPOINT_FILE,
            ROOT / "models/weights/fused" / fused.BACKBONE_FILE,
            "mps",
        )
    count, calls = 0, 0
    private = trial / "private"
    with (trial / (phase + ".jsonl")).open("x") as journal:
        for i, case in enumerate(selected):
            data, rgb, truth = case_source(case)
            if fmt == "native":
                raw = splice.predict(model, Image.fromarray(rgb))
                eligible = int((raw > splice.PIXEL_THRESHOLD).sum()) >= splice.MINIMUM_ACTIVE_CELLS
                metric = score_map(raw, eligible, splice.PIXEL_THRESHOLD, data, rgb, truth)
            else:
                prediction = model.predict(Image.fromarray(rgb))
                raw, eligible = prediction.raw_mask, prediction.candidate_eligibility["eligible"]
                metric = measure(prediction, data, rgb, truth)
            calls += 1
            if i == 0:
                repeat = (
                    splice.predict(model, Image.fromarray(rgb))
                    if fmt == "native"
                    else model.predict(Image.fromarray(rgb)).raw_mask
                )
                calls += 1
                require(np.array_equal(raw, repeat), "First current reference prediction failed exact repeat")
            name = hashlib.sha256(("reference-" + case["id"]).encode()).hexdigest() + ".npz"
            np.savez_compressed(private / name, raw=raw)
            journal.write(
                json.dumps(
                    {
                        "id": case["id"],
                        "input_sha256": hashlib.sha256(data).hexdigest(),
                        "status": "ok",
                        "gate": eligible,
                        "raw_file": name,
                        "raw_file_sha256": sha(private / name),
                        "metrics": metric["metrics"],
                        "ungated": metric["ungated"],
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            journal.flush()
            count += 1
            if count % 12 == 0:
                print({"phase": phase, "complete": count}, flush=True)
    write(
        trial / (phase + "-complete.json"),
        {
            "planned": 48,
            "complete": count,
            "successful_native_forwards": calls,
            "journal_sha256": sha(trial / (phase + ".jsonl")),
            "freeze_sha256": sha(trial / "freeze.json"),
        },
    )


def phase_rows(trial):
    rows = []
    for family in FAMILIES:
        for fmt in FORMATS:
            phase = f"diag-{family}-{fmt}"
            complete = read(trial / (phase + "-complete.json"))
            part = lines(trial / (phase + ".jsonl"))
            require(
                len(part) == complete["complete"] == complete["planned"] == 48
                and complete["journal_sha256"] == sha(trial / (phase + ".jsonl")),
                "Complete diagnostic phase required",
            )
            rows += part
    return rows


def current_record(trial, case, data, rgb, truth):
    if case["kind"] == "ordinary":
        matches = [
            r for r in lines(trial / ("reference-" + case["format"] + ".jsonl")) if r["id"] == case["id"]
        ]
        require(len(matches) == 1, "Actual ordinary reference required")
        row = matches[0]
        path = trial / "private" / row["raw_file"]
        require(sha(path) == row["raw_file_sha256"], "Current ordinary response changed")
        with np.load(path, allow_pickle=False) as saved:
            raw = saved["raw"].copy()
    else:
        raw, row = cached(case, current=True)
    require(row["input_sha256"] == hashlib.sha256(data).hexdigest(), "Actual current reference input differs")
    metric = score_map(
        raw, row["gate"], splice.PIXEL_THRESHOLD if case["format"] == "native" else 0.5, data, rgb, truth
    )
    require(
        metric["metrics"] == row["metrics"] and metric["ungated"] == row["ungated"],
        "Retained/current reference display replay differs",
    )
    return metric["metrics"]


def summarize(trial):
    plan = check(trial)
    check_points(trial)
    measured = {r["id"]: r for r in phase_rows(trial)}
    points, outcomes = read(trial / "points.json"), []
    for case in plan:
        if case["role"] != "diagnostic":
            continue
        row = measured[case["id"]]
        data, rgb, truth = case_source(case)
        metric = {"current": current_record(trial, case, data, rgb, truth)}
        for arm in ("whole", "zoom"):
            response = row["responses"][arm]
            eligible = response["active_fraction"] >= points[case["format"]][arm]["minimum_active_fraction"]
            metric[arm] = (
                response["ungated"] if eligible else mask_metrics(np.zeros(truth.shape, bool), truth)
            )
        outcomes.append(
            {
                **{k: case[k] for k in ("id", "source_id", "family", "kind", "format", "label")},
                "metrics": metric,
            }
        )
    write(trial / "private-outcomes.json", outcomes)
    result = independent.aggregate(outcomes)
    result.update(
        schema="chipmunk.image-iml-zoom-result/16",
        source_groups=48,
        file_cases=288,
        private_outcomes_sha256=sha(trial / "private-outcomes.json"),
        new_weights=0,
        serving_admitted=False,
        used_in_confidence=False,
        outside_human_reviewers=0,
    )
    write(trial / "result.json", result)
    print(result["formats"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = (
        "freeze",
        "seal",
        "summary",
        *[f"point-{f}" for f in FORMATS],
        *[f"reference-{f}" for f in FORMATS],
        *[f"diag-{family}-{fmt}" for family in FAMILIES for fmt in FORMATS],
    )
    parser.add_argument("phase", choices=phases)
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    if args.phase in ("freeze", "seal", "summary"):
        {"freeze": freeze, "seal": seal, "summary": summarize}[args.phase](args.trial)
    elif args.phase.startswith("reference-"):
        reference(args.trial, args.phase)
    else:
        native(args.trial, args.phase)
