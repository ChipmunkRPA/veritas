"""Freeze a simple TIDAL region selection rule on a separate development split."""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detectors.image import fused  # noqa: E402
from scripts.compare_image_localizers import WEIGHTS, measure  # noqa: E402
from scripts.localization_benchmark import read_manifest, sha, source  # noqa: E402
from scripts.score_splice_localizers import PUBLISHER_FILES, score_map, tidal  # noqa: E402

METHODS = ("native", "mirror_consensus")
PIXEL_THRESHOLD = 0.7
PUBLISHER_MIN_AREA = 400
TARGET_FALSE_RATE = 0.05


def write_new(path: Path, document: dict) -> None:
    with path.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def partition(root: Path, output: Path) -> None:
    manifest, manifest_sha = read_manifest(root)
    plan = (root / "tgif-plan.json").read_bytes()
    if sha(plan) != manifest["plan_sha256"]:
        raise ValueError("Benchmark source selection changed after acquisition")
    protocol = json.loads(plan)
    source_ids = {i["source_id"] for i in manifest["items"]}
    if source_ids & set(protocol["excluded_coco_ids"]):
        raise ValueError("Confirmation source overlaps the prior benchmark")
    development = set()
    for group in protocol["groups"].values():
        for variation in ("bbox", "segm"):
            selected = [pair["source_id"] for pair in group if pair["mask_variant"] == variation][:4]
            if len(selected) != 4:
                raise ValueError("Each group needs four development sources per shape")
            development.update(selected)
    if len(development) != 24 or len(source_ids - development) != 48:
        raise ValueError("Unexpected paired source counts")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "split.json", {"schema": "chipmunk.image-splice-split/1", "manifest_sha256": manifest_sha,
              "development": sorted(development), "confirmation": sorted(source_ids - development),
              "rule": "First four originals per generator and edit shape in frozen hashed plan order form development. "
                      "All other originals and their paired edits are reserved for confirmation.",
              "eligible_methods": list(METHODS), "pixel_threshold": PIXEL_THRESHOLD,
              "selection": "Choose the method with greatest mean edit IoU at its development false-alert budget. "
                           "Each method uses the lowest integer native area threshold at or above the published 400 "
                           "pixel minimum that permits at most floor(0.05 * authentic_count) false highlights."})


def infer(root: Path, split: dict, section: str, rule: dict | None = None, *,
          jpeg_quality: int | None = None) -> list[dict]:
    manifest, digest = read_manifest(root)
    if digest != split["manifest_sha256"]:
        raise ValueError("Source manifest differs from split")
    ids = set(split[section])
    selected = [item for item in manifest["items"] if item["source_id"] in ids]
    expected = {(id_, label): 1 for id_ in ids for label in (0, 1)}
    if Counter((i["source_id"], i["label"]) for i in selected) != expected:
        raise ValueError("Every source must have its exact authentic and edited pair")
    methods = METHODS if rule is None else (rule["method"],)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    predict, threshold = tidal()
    if threshold != PIXEL_THRESHOLD:
        raise ValueError("Publisher pixel policy changed")
    rows = []
    with torch.inference_mode():
        for n, item in enumerate(selected, 1):
            image_bytes, pixels, truth = source(root, item)
            image = Image.fromarray(pixels)
            if jpeg_quality is not None:
                with io.BytesIO() as buffer:
                    image.save(buffer, format="JPEG", quality=jpeg_quality)
                    image_bytes = buffer.getvalue()
                    buffer.seek(0)
                    with Image.open(buffer) as encoded:
                        image = encoded.convert("RGB")
                pixels = np.asarray(image)
            native, _, _ = predict(image)
            maps = {"native": native}
            if "mirror_consensus" in methods:
                mirrored, _, _ = predict(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
                maps["mirror_consensus"] = np.minimum(native, mirrored[:, ::-1])
            row = {k: item[k] for k in ("id", "source_id", "family", "label", "image_sha256", "mask_sha256")}
            if jpeg_quality is not None:
                row["encoded_image_sha256"] = sha(image_bytes)
            for method in methods:
                raw = maps[method]
                area = int((raw > threshold).sum())
                metrics = score_map(raw, True, threshold, image_bytes, pixels, truth)["ungated"]
                row[method] = {"pixels_above_threshold": area, "metrics": metrics,
                               "map_bytes_sha256": sha(raw.tobytes())}
            rows.append(row)
            if n == 1 or n % 12 == 0 or n == len(selected):
                print(f"{section}: {n}/{len(selected)}", flush=True)
    return rows


def summarize(rows: list[dict], method: str, minimum_area: int) -> dict:
    authentic = [row[method] for row in rows if not row["label"]]
    edited = [row[method] for row in rows if row["label"]]
    retained = [row["metrics"] for row in edited if row["pixels_above_threshold"] >= minimum_area]
    return {"authentic_n": len(authentic), "edited_n": len(edited),
            "authentic_with_regions": sum(row["pixels_above_threshold"] >= minimum_area
                                          and row["metrics"]["predicted_positive_pixels"] > 0 for row in authentic),
            "edited_with_overlap": sum(row["intersection"] > 0 for row in retained),
            "edited_mean_iou": sum(row["iou"] for row in retained) / len(edited),
            "edited_mean_f1": sum(row["pixel_f1"] for row in retained) / len(edited)}


def fit(rows: list[dict]) -> dict:
    authentic = [row for row in rows if not row["label"]]
    budget = math.floor(TARGET_FALSE_RATE * len(authentic))
    candidates = []
    for method in METHODS:
        false_areas = sorted((row[method]["pixels_above_threshold"] for row in authentic
                              if row[method]["metrics"]["predicted_positive_pixels"] > 0), reverse=True)
        minimum_area = max(PUBLISHER_MIN_AREA, false_areas[budget] + 1 if len(false_areas) > budget else 0)
        candidates.append({"method": method, "minimum_area_pixels": minimum_area,
                           **summarize(rows, method, minimum_area)})
    chosen = max(candidates, key=lambda row: (row["edited_mean_iou"], row["edited_with_overlap"],
                                              -row["authentic_with_regions"]))
    return {"method": chosen["method"], "minimum_area_pixels": chosen["minimum_area_pixels"],
            "pixel_threshold": PIXEL_THRESHOLD, "development_candidates": candidates,
            "development_false_alert_budget": budget, "checkpoint_sha256": PUBLISHER_FILES["tidal-v30.1.pt"]}


def run(root: Path, output: Path, section: str, *, jpeg_quality: int | None = None) -> None:
    split_path = output / "split.json"
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    receipt = section if jpeg_quality is None else f"{section}-jpeg{jpeg_quality}"
    result = output / f"{receipt}.json"
    rule_path = output / "frozen-region-rule.json"
    if result.exists():
        raise FileExistsError(f"The {section} scores are already frozen")
    rule = None
    if section == "confirmation":
        rule = json.loads(rule_path.read_bytes())
        if rule["split_sha256"] != sha(split_bytes) or rule["checkpoint_sha256"] != PUBLISHER_FILES["tidal-v30.1.pt"]:
            raise ValueError("The frozen rule belongs to a different split or checkpoint")
    rows = infer(root, split, section, rule, jpeg_quality=jpeg_quality)
    write_new(result, {"split_sha256": sha(split_bytes), "jpeg_quality": jpeg_quality, "rows": rows})
    if rule is None:
        rule = {**fit(rows), "split_sha256": sha(split_bytes), "development_sha256": sha(result.read_bytes())}
        write_new(rule_path, rule)
        print(json.dumps(rule, indent=2), flush=True)
    else:
        summary = {"combined": summarize(rows, rule["method"], rule["minimum_area_pixels"]),
                   "groups": {family: summarize([row for row in rows if row["family"] == family],
                                                 rule["method"], rule["minimum_area_pixels"])
                              for family in sorted({row["family"] for row in rows})},
                   "rule_sha256": sha(rule_path.read_bytes()), "scores_sha256": sha(result.read_bytes())}
        write_new(output / f"{receipt}-summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)


def incumbent(root: Path, output: Path, device: str) -> None:
    path = output / "incumbent-confirmation.json"
    if path.exists():
        raise FileExistsError("Incumbent confirmation is already frozen")
    split_bytes = (output / "split.json").read_bytes()
    split = json.loads(split_bytes)
    manifest, digest = read_manifest(root)
    if split["manifest_sha256"] != digest:
        raise ValueError("Confirmation split belongs to a different source manifest")
    confirmation = set(split["confirmation"])
    selected = [i for i in manifest["items"] if i["source_id"] in confirmation]
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    model = fused.load_model(WEIGHTS / fused.CHECKPOINT_FILE, WEIGHTS / fused.BACKBONE_FILE, device)
    rows = []
    for index, item in enumerate(selected, 1):
        image_bytes, pixels, truth = source(root, item)
        prediction = model.predict(Image.fromarray(pixels))
        measurement = measure(prediction, image_bytes, pixels, truth)
        rows.append({**{k: item[k] for k in ("id", "source_id", "family", "label", "image_sha256", "mask_sha256")},
                     **measurement, "map_bytes_sha256": sha(prediction.raw_mask.tobytes())})
        if index == 1 or index % 12 == 0 or index == len(selected):
            print(f"incumbent: {index}/{len(selected)}", flush=True)
    write_new(path, {"split_sha256": sha(split_bytes), "checkpoint_sha256": fused.CHECKPOINT_SHA256,
                     "device": device, "rows": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("partition", "development", "confirmation", "incumbent"))
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps", help="Incumbent evaluation device")
    parser.add_argument("--jpeg-quality", type=int, choices=(75, 95), help="Confirmation robustness transform")
    args = parser.parse_args()
    if args.jpeg_quality is not None and args.action != "confirmation":
        parser.error("--jpeg-quality applies only to confirmation")
    if args.action == "partition":
        partition(args.manifest_root, args.output)
    elif args.action == "incumbent":
        incumbent(args.manifest_root, args.output, args.device)
    else:
        run(args.manifest_root, args.output, args.action, jpeg_quality=args.jpeg_quality)
