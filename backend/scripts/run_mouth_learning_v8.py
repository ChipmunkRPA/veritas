"""Freeze sources, cache prefix tokens, fit and compare a two-block mouth learner."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import cv2
import torch
from safetensors.torch import load_file, save_file

from app.core.config import settings
from app.detectors.image import gend
from app.detectors.image.adapters.loaders import load_yunet
from app.detectors.image.faces import detect_faces
from app.detectors.video.adapters.loaders import load_dinov2_small
from app.detectors.video.face_sequence import REVIEW_POINT
from app.engine.fetchers import ensure_verified
from app.engine.registry import get_registry
from scripts import compare_mouth_probe_v6 as comparison
from scripts import compare_mouth_sequence_v4 as baseline
from scripts import mouth_learner_v8 as method
from scripts.liveportrait_mouth_benchmark import digest
from scripts.mouth_probe_v6 import margins, mouth_chip, preprocessed
from scripts.prepare_mouth_learning_v8 import validate_parent


def write(path, body):
    with path.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")


def pins():
    root = Path(__file__).resolve().parents[2]
    names = [
        "backend/scripts/run_mouth_learning_v8.py",
        "backend/scripts/mouth_learner_v8.py",
        "backend/scripts/mouth_probe_v6.py",
        "backend/scripts/compare_mouth_probe_v6.py",
        "backend/scripts/compare_mouth_sequence_v4.py",
        "backend/scripts/compare_mouth_artifact_v5.py",
        "backend/app/detectors/image/gend.py",
        "backend/app/detectors/image/faces.py",
        "backend/app/detectors/video/frame_store.py",
        "backend/app/detectors/video/face_sequence.py",
        "backend/app/detectors/video/adapters/face_sequence.py",
        "backend/app/detectors/video/adapters/loaders.py",
        "backend/app/detectors/video/adapters/tracks.py",
        "backend/app/detectors/video/tracking.py",
        "backend/app/models.yaml",
        "docs/mouth-learning-v8-development.md",
    ]
    return {p: digest(root / p) for p in names}


def check_lock(path):
    lock = json.loads(path.read_text())
    if lock["implementation_sha256"] != pins():
        raise ValueError("Frozen learning or comparison implementation changed")
    for descriptor in lock["development"]:
        for key in ("manifest", "receipt"):
            if digest(Path(descriptor[key])) != descriptor[key + "_sha256"]:
                raise ValueError("Immutable development baseline changed")
    return lock


def freeze(data_root, output):
    output.mkdir(parents=True, exist_ok=False)
    v6 = data_root / "lipsync-v6"
    v8 = data_root / "lipsync-v8"
    recipe = json.loads((v6 / "training-plan.json").read_text())
    parent = v6 / "training-inputs/manifest.json"
    extension = v8 / "training-inputs/manifest.json"
    old = json.loads(parent.read_text())
    new = json.loads(extension.read_text())
    validate_parent(recipe, old["items"])
    if new["plan_sha256"] != digest(v8 / "training-plan.json") or len(new["items"]) != 800:
        raise ValueError("Complete frozen generator extension required")
    expected = [
        (f"{s['id']}:{i}:{c}", s["group"], s["role"], int(c.startswith("muse-ai-")))
        for s in recipe["sources"]
        for i in s["selected_frames"]
        for c in ("muse-resample", "local-blur-2", "local-blur-4", "muse-ai-aligned", "muse-ai-earlier")
    ]
    if [(r["id"], r["group"], r["role"], r["label"]) for r in new["items"]] != expected:
        raise ValueError("Extension source roles, selection or labels changed")
    rows = []
    for path, family in ((parent, "liveportrait"), (extension, "musetalk")):
        for item in json.loads(path.read_text())["items"]:
            if item["status"] != "ready":
                raise ValueError("No partial fitting panel is permitted")
            for key, sha in (("path", "sha256"), ("mask_path", "mask_sha256")):
                if digest(path.parent / item[key]) != item[sha]:
                    raise ValueError("Prepared PNG/mask changed")
            rows.append(
                {
                    **item,
                    "path": str((path.parent / item["path"]).resolve()),
                    "mask_path": str((path.parent / item["mask_path"]).resolve()),
                    "family": family,
                }
            )
    if len({r["id"] for r in rows}) != 1600:
        raise ValueError("Fitting IDs must be unique across families")
    training_path = output / "training.json"
    write(
        training_path,
        {
            "schema": "chipmunk.mouth-learning-inputs/8",
            "items": rows,
            "parent_sha256": digest(parent),
            "extension_sha256": digest(extension),
        },
    )
    development = []
    for manifest, receipt, count, family in (
        (
            data_root / "lipsync-v4/development/inputs-v2/manifest.json",
            data_root / "lipsync-v4/development/scores.jsonl",
            48,
            "liveportrait",
        ),
        (
            data_root / "lipsync-v7/development-inputs/manifest.json",
            data_root / "lipsync-v7/current-check-scores.jsonl",
            40,
            "musetalk",
        ),
    ):
        cases = json.loads(manifest.read_text())["items"]
        header, *references = [json.loads(line) for line in receipt.read_text().splitlines()]
        if (
            len(cases) != count
            or header["manifest_sha256"] != digest(manifest)
            or [r["id"] for r in cases] != [r["id"] for r in references]
        ):
            raise ValueError("Complete exact development receipts required")
        if {r["group"] for r in cases} != set(recipe["excluded_development_groups"]):
            raise ValueError("Development source actors changed")
        development.append(
            {
                "manifest": str(manifest.resolve()),
                "manifest_sha256": digest(manifest),
                "receipt": str(receipt.resolve()),
                "receipt_sha256": digest(receipt),
                "family": family,
                "planned": count,
            }
        )
    lock = {
        "schema": "chipmunk.mouth-learning-lock/8",
        "method": method.METHOD,
        "training_manifest": str(training_path.resolve()),
        "training_manifest_sha256": digest(training_path),
        "implementation_sha256": pins(),
        "development": development,
        "deployment_eligible": False,
    }
    write(output / "lock.json", lock)
    print(
        json.dumps(
            {
                "lock_sha256": digest(output / "lock.json"),
                "fit_stills": 1280,
                "point_stills": 320,
                "development_clips": 88,
            }
        )
    )


def setup():
    if not torch.backends.mps.is_available():
        raise ValueError("Frozen experiment requires MPS; no fallback")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(method.SEED)
    torch.mps.set_per_process_memory_fraction(0.65)
    weights = {}
    for mid in ("yunet.2023mar", "dinov2.small"):
        spec = get_registry().manifest.model(mid)
        if spec.licence.class_ != "commercial":
            raise ValueError("Commercial checkpoint declaration required")
        weights[mid] = [{"name": p.name, "sha256": digest(p)} for p in ensure_verified(spec, force=True)]
    return weights


def cache(lock_path, output):
    lock = check_lock(lock_path)
    manifest = Path(lock["training_manifest"])
    if digest(manifest) != lock["training_manifest_sha256"]:
        raise ValueError("Frozen training manifest changed")
    rows = json.loads(manifest.read_text())["items"]
    output.mkdir(parents=True, exist_ok=False)
    weights = setup()
    detector = load_yunet("yunet.2023mar", "cpu", "float32").obj
    model = load_dinov2_small("dinov2.small", "mps", "float32").obj.model
    learner = method.MouthLearner(model).to("mps")
    tokens, bindings = [], []
    offset, parity = 0, None
    started = time.monotonic()
    for index, row in enumerate(rows):
        path = Path(row["path"])
        if digest(path) != row["sha256"]:
            raise ValueError("Training image changed")
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        if hashlib.sha256(rgb.tobytes()).hexdigest() != row["native_rgb_sha256"]:
            raise ValueError("Training native pixels changed")
        faces = detect_faces(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=settings.face_min_size
        )
        transform = gend.align_transform(faces[0].landmarks) if faces else None
        if transform is None:
            raise ValueError("A fitting frame lacks eligible source geometry")
        aligned = gend.warp(rgb, *transform)
        chip, bounds = mouth_chip(aligned)
        variants = [method.view(chip, k) for k in (range(4) if row["role"] == "fit" else [0])]
        pixels = preprocessed(variants)
        with torch.no_grad():
            x = torch.from_numpy(pixels).to("mps")
            prefix = learner.prefix(x)
            if parity is None:
                full = method.pool(model(pixel_values=x).last_hidden_state)
                split = learner.representation(prefix)
                error = float((full - split).abs().max().cpu())
                if error > 1e-6:
                    raise ValueError("Cached prefix does not match ordinary full-model inference")
                parity = {"max_absolute_difference": error, "batch_views": len(variants)}
            prefix = prefix.detach().cpu().contiguous()
        if prefix.shape != (len(variants), 65, 384) or not torch.isfinite(prefix).all():
            raise ValueError("Unexpected frozen prefix shape/values")
        binding = {
            "id": row["id"],
            "alignment_side": transform[1],
            "alignment_matrix_hex": [[float(v).hex() for v in line] for line in transform[0]],
            "aligned_rgb_sha256": hashlib.sha256(aligned.tobytes()).hexdigest(),
            "chip_bounds": bounds,
            "mouth_chip_rgb_sha256": hashlib.sha256(chip.tobytes()).hexdigest(),
            "views": [
                {
                    "view": k,
                    "offset": offset + k,
                    "view_rgb_sha256": hashlib.sha256(variants[k].tobytes()).hexdigest(),
                    "input_f32_sha256": hashlib.sha256(pixels[k].tobytes()).hexdigest(),
                    "prefix_f32_sha256": method.sha_tensor(prefix[k]),
                }
                for k in range(len(variants))
            ],
        }
        tokens.append(prefix)
        bindings.append(binding)
        offset += len(variants)
        if (index + 1) % 80 == 0:
            print(
                json.dumps(
                    {"cached_frames": index + 1, "total": 1600, "seconds": time.monotonic() - started}
                ),
                flush=True,
            )
    target = output / "prefix.safetensors"
    save_file({"prefix": torch.cat(tokens)}, target)
    write(
        output / "receipt.json",
        {
            "schema": "chipmunk.mouth-learning-cache/8",
            "lock_sha256": digest(lock_path),
            "manifest_sha256": digest(manifest),
            "tensor_sha256": digest(target),
            "tokens": offset,
            "weights": weights,
            "prefix_full_forward_parity": parity,
            "bindings": bindings,
            "device": "mps",
            "torch_version": torch.__version__,
            "opencv_version": cv2.__version__,
        },
    )
    print(
        json.dumps(
            {
                "cached_stills": len(bindings),
                "cached_views": offset,
                "receipt_sha256": digest(output / "receipt.json"),
            }
        ),
        flush=True,
    )


def load_cache(lock_path, cache_path):
    lock = check_lock(lock_path)
    manifest = Path(lock["training_manifest"])
    receipt = json.loads((cache_path / "receipt.json").read_text())
    target = cache_path / "prefix.safetensors"
    if (
        digest(manifest) != lock["training_manifest_sha256"]
        or receipt["manifest_sha256"] != digest(manifest)
        or receipt["lock_sha256"] != digest(lock_path)
        or receipt["tensor_sha256"] != digest(target)
    ):
        raise ValueError("Frozen learning inputs/cache changed")
    rows = json.loads(manifest.read_text())["items"]
    method.fit_positions(rows, receipt["bindings"])
    tokens = load_file(target)["prefix"]
    if tokens.shape != (receipt["tokens"], 65, 384):
        raise ValueError("Cached tensor shape changed")
    for binding in receipt["bindings"]:
        for v in binding["views"]:
            if method.sha_tensor(tokens[v["offset"]]) != v["prefix_f32_sha256"]:
                raise ValueError("A source-bound prefix token changed")
    return lock, rows, receipt, tokens


def train(lock_path, cache_path, output):
    _, rows, receipt, tokens = load_cache(lock_path, cache_path)
    output.mkdir(parents=True, exist_ok=False)
    weights = setup()
    if weights != receipt["weights"]:
        raise ValueError("Learning uses a different backbone checkpoint")
    learner = method.MouthLearner(load_dinov2_small("dinov2.small", "mps", "float32").obj.model).to("mps")
    result = method.optimize(learner, tokens, rows, receipt["bindings"])
    head = method.linear_head(learner)
    point_rows = [r for r in rows if r["role"] == "point"]
    selected = [
        b["views"][0]["offset"]
        for r, b in zip(rows, receipt["bindings"], strict=True)
        if r["role"] == "point"
    ]
    values = []
    with torch.no_grad():
        for start in range(0, len(selected), method.BATCH):
            values.extend(
                learner.representation(tokens[selected[start : start + method.BATCH]].to("mps")).cpu().numpy()
            )
    responses = margins(values, head)
    review_point = method.choose_point(point_rows, responses)
    target = output / "learned.safetensors"
    save_file(
        {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in learner.named_parameters()
            if parameter.requires_grad
        },
        target,
    )
    write(
        output / "head.json",
        {
            "schema": "chipmunk.mouth-learning-head/8",
            "method": method.METHOD,
            "lock_sha256": digest(lock_path),
            "cache_receipt_sha256": digest(cache_path / "receipt.json"),
            "learned_sha256": digest(target),
            "weights": weights,
            **head,
            "review_point": review_point,
            "fit_groups": sorted({r["group"] for r in rows if r["role"] == "fit"}),
            "point_groups": sorted({r["group"] for r in point_rows}),
            "fit_stills": 1280,
            "point_stills": 320,
            "point_negative_stills": sum(r["label"] == 0 for r in point_rows),
            "learning": result,
            "point_responses": [
                {"id": r["id"], "label": r["label"], "margin": float(v)}
                for r, v in zip(point_rows, responses, strict=True)
            ],
            "response_kind": "raw signed margin; not calibrated AI probability",
            "deployment_eligible": False,
            "torch_version": torch.__version__,
            "device": "mps",
        },
    )
    print(
        json.dumps(
            {
                "head_sha256": digest(output / "head.json"),
                "learned_sha256": digest(target),
                "review_point": review_point,
                "trainable_parameters": sum(result["trainable_parameters"].values()),
            }
        ),
        flush=True,
    )


def loaded_candidate(head_path):
    head = json.loads(head_path.read_text())
    target = head_path.parent / "learned.safetensors"
    if digest(target) != head["learned_sha256"]:
        raise ValueError("Learned parameter file changed")
    loaded = load_dinov2_small("dinov2.small", "mps", "float32")
    learner = method.MouthLearner(loaded.obj.model).to("mps")
    state = load_file(target)
    expected = {name for name, p in learner.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Learned checkpoint changed frozen/unknown parameter keys")
    missing, unexpected = learner.load_state_dict(state, strict=False)
    if unexpected or set(missing) != set(learner.state_dict()) - expected:
        raise ValueError("Partial learning checkpoint did not load exactly as declared")
    if not all(torch.equal(learner.state_dict()[key].detach().cpu(), value) for key, value in state.items()):
        raise ValueError("A learned tensor changed on load")
    if method.frozen_identity(learner) != head["learning"]["frozen_parameter_identity"]:
        raise ValueError("Frozen backbone changed on reload")
    if method.linear_head(learner) != {k: head[k] for k in ("coef", "intercept", "mean", "scale")}:
        raise ValueError("Learned linear parameters differ from published raw-margin calculation")
    return loaded


def score(lock_path, head_path, output):
    lock = check_lock(lock_path)
    head = json.loads(head_path.read_text())
    if head["lock_sha256"] != digest(lock_path):
        raise ValueError("Learned model is from a different experiment")
    weights = setup()
    if weights != head["weights"]:
        raise ValueError("Pretrained weights changed")
    models = {
        "yunet.2023mar": load_yunet("yunet.2023mar", "cpu", "float32"),
        "dinov2.small": load_dinov2_small("dinov2.small", "cpu", "float32"),
        "probe.dino": loaded_candidate(head_path),
    }
    expected = []
    for panel in lock["development"]:
        manifest = Path(panel["manifest"])
        _, *refs = [json.loads(line) for line in Path(panel["receipt"]).read_text().splitlines()]
        for case, reference in zip(json.loads(manifest.read_text())["items"], refs, strict=True):
            expected.append((case, reference, manifest.parent, panel["family"]))
    header = {
        "schema": "chipmunk.mouth-learning-scores/8",
        "lock_sha256": digest(lock_path),
        "head_sha256": digest(head_path),
        "review_point": head["review_point"],
        "planned": 88,
        "weights": weights,
        "candidate_device": "mps",
        "baseline_device": "cpu",
        "torch_version": torch.__version__,
    }
    complete = []
    if output.exists():
        old, *complete = [json.loads(line) for line in output.read_text().splitlines()]
        if old != header or [r["id"] for r in complete] != [r[0]["id"] for r in expected[: len(complete)]]:
            raise ValueError("Cannot resume changed comparison")
    else:
        with output.open("x") as stream:
            stream.write(json.dumps(header) + "\n")
    with output.open("a") as stream:
        for case, reference, root, family in expected[len(complete) :]:
            started = time.monotonic()
            row = {
                **case,
                "family": family,
                "mask_root": str(root),
                "status": "failed",
                "base_windows": reference["base_windows"],
                "probe_windows": [],
            }
            try:
                if case["group"] in set(head["fit_groups"] + head["point_groups"]):
                    raise ValueError("Development actor leaked into learning/point roles")
                path = root / case["path"]
                if case["status"] != "ready" or digest(path) != case["sha256"]:
                    raise ValueError("Prepared development input changed")
                with patch.object(comparison, "features", method.features):
                    row.update(comparison.score_case(path, case, reference, models, head))
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            row["runtime_s"] = time.monotonic() - started
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "status": row["status"],
                        "reason": row.get("reason"),
                        "seconds": row["runtime_s"],
                    }
                ),
                flush=True,
            )


def advancement(a, b, families, rows):
    current, candidate = baseline.summary(a), baseline.summary(b)
    gains = sum(not x["localized"] and y["localized"] for x, y in zip(a, b, strict=True))
    losses = sum(x["localized"] and not y["localized"] for x, y in zip(a, b, strict=True))
    controls = sorted({r["condition"] for r in a if not r["label"]})
    gates = {
        "additional_locations": gains - losses >= 6,
        "both_families": all(
            sum(
                y["localized"] - x["localized"] for x, y, f in zip(a, b, families, strict=True) if f == family
            )
            >= 2
            for family in ("liveportrait", "musetalk")
        ),
        "retained_locations": losses <= 1,
        "total_ordinary_flags": candidate["non_ai_flags"] <= current["non_ai_flags"] + 1,
        "condition_ordinary_flags": all(
            sum(r["flagged"] for r in b if r["condition"] == c)
            <= sum(r["flagged"] for r in a if r["condition"] == c) + 1
            for c in controls
        ),
        "spatial_gain": candidate["mean_ai_spatial_iou"] >= current["mean_ai_spatial_iou"] + 0.02,
        "coverage": all(not x["assessed"] or y["assessed"] for x, y in zip(a, b, strict=True)),
        "all_replays": len(rows) == 88
        and all(r["status"] == "ok" and r.get("baseline_replay_parity") for r in rows),
    }
    return current, candidate, gates


def review(lock_path, receipt, output):
    lock = check_lock(lock_path)
    header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    expected = [
        r["id"] for p in lock["development"] for r in json.loads(Path(p["manifest"]).read_text())["items"]
    ]
    if header["lock_sha256"] != digest(lock_path) or [r["id"] for r in rows] != expected:
        raise ValueError("Every planned development outcome is required")
    a, b = [], []
    for row in rows:
        root = Path(row["mask_root"])
        a += baseline.evaluate([row], "base", REVIEW_POINT, root)
        b += baseline.evaluate([row], "probe", header["review_point"], root)
    families = [r["family"] for r in rows]
    current, candidate, gates = advancement(a, b, families, rows)
    body = {
        "schema": "chipmunk.mouth-learning-summary/8",
        "lock_sha256": digest(lock_path),
        "score_receipt_sha256": digest(receipt),
        "head_sha256": header["head_sha256"],
        "review_point": header["review_point"],
        "baseline": current,
        "candidate": candidate,
        "gates": gates,
        "advance_to_confirmation_design": all(gates.values()),
        "deployment_eligible": False,
        "terminal_statuses": dict(Counter(r["status"] for r in rows)),
        "families": {
            f: {
                "baseline": baseline.summary([r for r, name in zip(a, families, strict=True) if name == f]),
                "candidate": baseline.summary([r for r, name in zip(b, families, strict=True) if name == f]),
            }
            for f in sorted(set(families))
        },
        "conditions": {
            c: {
                "baseline": baseline.summary([r for r in a if r["condition"] == c]),
                "candidate": baseline.summary([r for r in b if r["condition"] == c]),
            }
            for c in sorted({r["condition"] for r in rows})
        },
        "cases": {"baseline": a, "candidate": b},
        "limitations": [
            "Consumed development: 16 fit actors, 4 point actors, 8 video actors, four old sentence families.",
            "Two controlled editing families; pretrained overlap unknown; not unknown-media accuracy.",
            "Learned margins and measured regions are not calibrated probabilities or causal pixel proof.",
            "No outside independent reviewer has participated; no customer detector/confidence change.",
        ],
    }
    write(output, body)
    print(
        json.dumps(
            {k: v for k, v in body.items() if k not in ("cases", "conditions", "families", "limitations")}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "cache", "train", "score", "review"])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.input, args.output)
    elif args.stage == "cache":
        cache(args.input, args.output)
    elif args.stage == "train" and args.cache:
        train(args.input, args.cache, args.output)
    elif args.stage == "score" and args.head:
        score(args.input, args.head, args.output)
    elif args.stage == "review" and args.receipt:
        review(args.input, args.receipt, args.output)
    else:
        parser.error("train requires --cache, score requires --head, review requires --receipt")
