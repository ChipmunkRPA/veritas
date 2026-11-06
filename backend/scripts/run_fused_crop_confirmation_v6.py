"""Freeze, replay consumed controls, score and review fresh crop localization.

Only the explicit run stage may forward fresh images. It requires the published
freeze hash and the hash of a completed consumed-control parity receipt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detectors.image import fused, splice  # noqa: E402
from scripts import fused_crop_confirmation_v6 as protocol  # noqa: E402
from scripts.evaluate_iml_vit import input_bytes, write_new  # noqa: E402
from scripts.fused_crop_v6 import Crop, boxes, complement, predict, validate  # noqa: E402
from scripts.localization_benchmark import sha, source  # noqa: E402
from scripts.review_fused_crop_v6 import extra, summarize  # noqa: E402
from scripts.score_fused_crop_v6 import selected  # noqa: E402
from scripts.score_splice_localizers import score_map  # noqa: E402


def native_plan(items: list[dict]) -> list[dict]:
    chosen = []
    for family in ("firefly", "sd2", "flux-fill"):
        group = min({i["source_id"] for i in items if i["family"] == family}, key=lambda s: sha(s.encode()))
        chosen.extend(i for i in items if i["family"] == family and i["source_id"] == group)
    if len(chosen) != 6 or len({(i["source_id"], i["label"]) for i in chosen}) != 6:
        raise ValueError("Consumed parity plan must contain one original/edit pair per family")
    return chosen


def parity(actual: dict, arrays: dict, old: dict, expected: dict, gate: float) -> dict:
    if actual["status"] != "ok" or old["status"] != "ok" or set(arrays) != set(expected):
        return {"passed": False, "reason": "Missing native forward or response array"}
    if any(arrays[n].shape != expected[n].shape or arrays[n].dtype != np.float32
           or expected[n].dtype != np.float32 for n in arrays):
        return {"passed": False, "reason": "Changed native map geometry"}
    error = {n: float(np.max(np.abs(arrays[n] - expected[n]))) for n in arrays}
    crops, previous = actual.get("crops", []), old["crops"]
    if len(crops) != len(previous) or len(crops) != 4:
        return {"passed": False, "reason": "Changed native crop denominator"}
    score_error = max(abs(a["image_score"] - b["image_score"]) for a, b in zip(crops, previous, strict=True))
    same_gate = [c["image_score"] >= gate for c in crops] == [c["image_score"] >= gate for c in previous]
    same_baseline_gate = (actual.get("baseline_image_score", 0) >= 0.5) == (old.get("baseline_image_score", 0) >= 0.5)
    passed = (all(v <= 2e-6 for v in error.values()) and score_error <= 2e-6 and same_gate and same_baseline_gate
              and actual["input_sha256"] == old["input_sha256"] and actual["baseline"] == old["baseline"]
              and [(tuple(c["box"]), c["rgb_sha256"]) for c in crops] == [
                  (tuple(c["box"]), c["rgb_sha256"]) for c in previous])
    return {"passed": passed, "max_map_error": max(error.values()), "max_score_error": score_error,
            "same_crop_gate_decisions": same_gate, "same_baseline_gate": same_baseline_gate}


def models(assets: Path):
    torch.set_num_threads(2)
    return (fused.load_model(assets / "fused" / fused.CHECKPOINT_FILE,
                             assets / "fused" / fused.BACKBONE_FILE, "mps"),
            splice.load_model(assets / "splice" / splice.CHECKPOINT_FILE))


def extract(root: Path, item: dict, condition: str, model, baseline_model) -> tuple[dict, dict]:
    original, pixels, truth = source(root, item)
    data, rgb = input_bytes(pixels, original, condition)
    row = {k: item[k] for k in ("id", "source_id", "family", "label", "image_sha256", "mask_sha256")}
    row.update(condition=condition, input_sha256=sha(data), status="failed", baseline=None)
    arrays = {}
    try:
        image = Image.fromarray(rgb)
        if condition == "native":
            baseline = splice.predict(baseline_model, image)
            gate = int((baseline > splice.PIXEL_THRESHOLD).sum()) >= splice.MINIMUM_ACTIVE_CELLS
            threshold = splice.PIXEL_THRESHOLD
        else:
            result = model.predict(image)
            baseline, gate, threshold = result.raw_mask, result.candidate_eligibility["eligible"], 0.5
            row["baseline_image_score"] = result.image_score
        arrays["baseline"] = baseline
        row["baseline"] = score_map(baseline, gate, threshold, data, rgb, truth)["metrics"]
        crops = predict(model, image)
        arrays.update({f"crop{i}": c.response for i, c in enumerate(crops)})
        row["crops"] = [{"box": c.box, "image_score": c.image_score, "rgb_sha256": c.rgb_sha256} for c in crops]
        if crops:
            row["status"] = "ok"
        else:
            row.update(status="not_applicable", reason="Source below fixed320-pixel minimum")
    except Exception as error:
        row.update(error_type=type(error).__name__)
    return row, arrays


def native(lock_path: Path, root: Path, freeze_hash: str, previous_root: Path,
           selection: Path, previous_maps: Path, assets: Path, output: Path) -> dict:
    lock, _ = protocol.check(lock_path, root, freeze_hash)
    if sha((previous_maps / "predictions.jsonl").read_bytes()) != protocol.DEVELOPMENT_PREDICTIONS_SHA256:
        raise ValueError("Changed consumed neural reference")
    _, _, items = selected(previous_root, selection)
    saved = [json.loads(l) for l in (previous_maps / "predictions.jsonl").read_text().splitlines()]
    lookup = {(r["id"], r["condition"]): r for r in saved}
    chosen = native_plan(items)
    model, baseline_model = models(assets)
    cases = []
    with torch.inference_mode():
        for item in chosen:
            for condition in ("native", "jpeg75"):
                actual, arrays = extract(previous_root, item, condition, model, baseline_model)
                old = lookup[item["id"], condition]
                path = previous_maps / old["arrays"]
                if sha(path.read_bytes()) != old["arrays_sha256"]:
                    raise ValueError("Changed original consumed-control arrays")
                with np.load(path, allow_pickle=False) as expected:
                    measured = parity(actual, arrays, old, {n: expected[n] for n in expected.files},
                                      lock["rules"][condition]["image_gate"])
                cases.append({"id": item["id"], "family": item["family"], "label": item["label"],
                              "condition": condition, **measured})
    result = {"schema": "chipmunk.fused-crop-v6-native-parity/1", "freeze_sha256": freeze_hash,
              "development_predictions_sha256": protocol.DEVELOPMENT_PREDICTIONS_SHA256,
              "cases": cases, "expected_cases": 12, "fresh_inputs_scored": 0,
              "passed": len(cases) == 12 and all(c["passed"] for c in cases)}
    write_new(output, result)
    if not result["passed"]:
        raise ValueError("Consumed-control parity failed; retain receipt and stop before fresh inference")
    return result


def require_native(path: Path, expected_hash: str, freeze_hash: str) -> None:
    if sha(path.read_bytes()) != expected_hash:
        raise ValueError("Changed native parity receipt")
    d = json.loads(path.read_text())
    if (d.get("schema") != "chipmunk.fused-crop-v6-native-parity/1" or d["freeze_sha256"] != freeze_hash
            or d["development_predictions_sha256"] != protocol.DEVELOPMENT_PREDICTIONS_SHA256
            or d["passed"] is not True or d["fresh_inputs_scored"] != 0 or d["expected_cases"] != 12
            or len(d["cases"]) != 12 or not all(c["passed"] for c in d["cases"])
            or len({(c["id"], c["condition"]) for c in d["cases"]}) != 12
            or {f: sum(c["family"] == f for c in d["cases"]) for f in ("firefly", "sd2", "flux-fill")}
            != {"firefly": 4, "sd2": 4, "flux-fill": 4}):
        raise ValueError("Completed consumed-control parity is required before fresh inference")


def run(lock_path: Path, root: Path, freeze_hash: str, native_path: Path,
        native_hash: str, assets: Path, output: Path) -> None:
    lock, manifest = protocol.check(lock_path, root, freeze_hash)
    require_native(native_path, native_hash, freeze_hash)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "policy.json", {"schema": protocol.SCHEMA, "section": "confirmation",
                                       "freeze_sha256": freeze_hash, "native_receipt_sha256": native_hash,
                                       "manifest_sha256": lock["source_audit"]["manifest_sha256"],
                                       "planned_inputs": lock["planned_inputs"]})
    model, baseline_model = models(assets)
    with (output / "predictions.jsonl").open("x") as log, torch.inference_mode():
        for index, item in enumerate(manifest["items"]):
            for condition in lock["conditions"]:
                row, arrays = extract(root, item, condition, model, baseline_model)
                if arrays:
                    name = f"arrays-{index:03d}-{condition}.npz"
                    np.savez_compressed(output / name, **arrays)
                    row.update(arrays=name, arrays_sha256=sha((output / name).read_bytes()))
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps({"completed": index * 2 + (condition == "jpeg75") + 1,
                                  "planned": 192, "status": row["status"]}), flush=True)


def review(lock_path: Path, root: Path, freeze_hash: str, native_path: Path,
           native_hash: str, maps: Path, output: Path) -> dict:
    lock, manifest = protocol.check(lock_path, root, freeze_hash)
    require_native(native_path, native_hash, freeze_hash)
    policy = json.loads((maps / "policy.json").read_text())
    rows = [json.loads(l) for l in (maps / "predictions.jsonl").read_text().splitlines()]
    expected = {(i["id"], c): i for i in manifest["items"] for c in lock["conditions"]}
    if (policy["freeze_sha256"] != freeze_hash or policy["section"] != "confirmation"
            or policy["native_receipt_sha256"] != native_hash
            or policy["manifest_sha256"] != lock["source_audit"]["manifest_sha256"]
            or len(rows) != 192 or {(r["id"], r["condition"]) for r in rows} != expected.keys()):
        raise ValueError("Changed method or incomplete exact confirmation denominator")
    cases = []
    for row in rows:
        item = expected[row["id"], row["condition"]]
        if any(row[k] != item[k] for k in ("source_id", "family", "label", "image_sha256", "mask_sha256")):
            raise ValueError("Changed confirmation labels or source identity")
        original, pixels, truth = source(root, item)
        data, rgb = input_bytes(pixels, original, row["condition"])
        if row["input_sha256"] != sha(data) or row["status"] not in ("ok", "failed", "not_applicable"):
            raise ValueError("Changed evaluated bytes or result status")
        prior, candidate, used = None, None, False
        if row.get("arrays"):
            name = row["arrays"]
            if Path(name).name != name or sha((maps / name).read_bytes()) != row["arrays_sha256"]:
                raise ValueError("Changed confirmation native arrays")
            with np.load(maps / name, allow_pickle=False) as saved:
                baseline = saved["baseline"]
                if not np.isfinite(baseline).all():
                    raise ValueError("Non-finite baseline array")
                if row["condition"] == "native":
                    gate = int((baseline > splice.PIXEL_THRESHOLD).sum()) >= splice.MINIMUM_ACTIVE_CELLS
                    threshold = splice.PIXEL_THRESHOLD
                else:
                    gate = fused.candidate_eligibility(row["baseline_image_score"])["eligible"]
                    threshold = 0.5
                prior = score_map(baseline, gate, threshold, data, rgb, truth)["metrics"]
                if prior != row["baseline"]:
                    raise ValueError("Reconstructed current component differs from recorded metrics")
                candidate = prior
                if row["status"] == "ok":
                    if set(saved.files) != {"baseline", "crop0", "crop1", "crop2", "crop3"}:
                        raise ValueError("Incomplete crop evidence")
                    crops = [Crop(tuple(c["box"]), saved[f"crop{i}"], c["image_score"], c["rgb_sha256"])
                             for i, c in enumerate(row["crops"])]
                    validate(crops, rgb.shape[1], rgb.shape[0])
                    image = Image.fromarray(rgb)
                    if any(sha(np.asarray(image.crop(c.box)).tobytes()) != c.rgb_sha256 for c in crops):
                        raise ValueError("Scored crop does not correspond to its native source coordinates")
                    extra_metric = extra({"crops": crops, "rgb": rgb, "data": data, "truth": truth},
                                         lock["rules"][row["condition"]]["image_gate"])
                    candidate, used = complement(prior, extra_metric)
                elif row["status"] == "not_applicable" and boxes(rgb.shape[1], rgb.shape[0]):
                    raise ValueError("Unexplained crop scope exclusion")
        if row["status"] == "ok" and prior is None:
            raise ValueError("Successful input lacks stored evidence")
        cases.append({**{k: row[k] for k in ("id", "source_id", "family", "label", "condition")},
                      "baseline": prior, "candidate": candidate, "used_crops": used, "crop_status": row["status"]})
    def fresh_summary(selected_rows, *, qualify=False):
        summary = summarize(selected_rows)
        summary["meets_fresh_count_tradeoff"] = summary.pop("development_count_tradeoff")
        if qualify:
            summary["qualifies_for_local_integration"] = (summary["meets_fresh_count_tradeoff"]
                                                           and summary["crop_failures"] == 0)
        return summary

    conditions = {c: fresh_summary([r for r in cases if r["condition"] == c], qualify=True) for c in lock["conditions"]}
    result = {"schema": "chipmunk.fused-crop-confirmation-review/6", "section": "confirmation",
              "freeze_sha256": freeze_hash, "predictions_sha256": sha((maps / "predictions.jsonl").read_bytes()),
              "conditions": conditions, "cases": cases,
              "families": {f: {c: fresh_summary([r for r in cases if r["family"] == f and r["condition"] == c])
                                for c in lock["conditions"]} for f in sorted({r["family"] for r in cases})},
              "may_enable_service": False, "overall_confidence_weight": 0,
              "limits": lock["scope"] + " Native inference parity and source separation are local checks, not outside review. "
                        "Qualifying localization still requires UI/PDF/JSON integration verification; no automatic default change."}
    write_new(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "native", "run", "review"))
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--previous-root", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--native-sha256")
    parser.add_argument("--maps", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        protocol.freeze(args.root, args.historical_root, args.development, args.output)
    elif not args.lock or not args.freeze_sha256:
        parser.error("Published lock and its independently recorded SHA-256 are required")
    elif args.stage == "native":
        native(args.lock, args.root, args.freeze_sha256, args.previous_root, args.selection,
               args.development / "development", args.assets, args.output)
    elif args.stage == "run":
        if not args.native or not args.native_sha256:
            parser.error("Completed native parity receipt and its SHA-256 are required")
        run(args.lock, args.root, args.freeze_sha256, args.native, args.native_sha256, args.assets, args.output)
    else:
        if not args.native or not args.native_sha256:
            parser.error("Native parity receipt and its SHA-256 are required for review")
        print(json.dumps(review(args.lock, args.root, args.freeze_sha256, args.native,
                                args.native_sha256, args.maps, args.output)["conditions"], indent=2))
