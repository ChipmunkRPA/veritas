"""Measure the current PNG/JPEG localizers on the same frozen paired image inputs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detectors.image import fused, splice  # noqa: E402
from scripts.compare_image_localizers import WEIGHTS, measure  # noqa: E402
from scripts.evaluate_iml_vit import input_bytes, write_new  # noqa: E402
from scripts.localization_benchmark import read_manifest, sha, source  # noqa: E402
from scripts.score_splice_localizers import score_map  # noqa: E402


def score(root: Path, output: Path, condition: str, device: str) -> None:
    manifest, digest = read_manifest(root)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    if condition == "native":
        model = splice.load_model(WEIGHTS.parent / "splice" / splice.CHECKPOINT_FILE)
        weights = [splice.CHECKPOINT_SHA256]
    else:
        model = fused.load_model(WEIGHTS / fused.CHECKPOINT_FILE, WEIGHTS / fused.BACKBONE_FILE, device)
        weights = [fused.CHECKPOINT_SHA256, fused.BACKBONE_SHA256]
    policy = {"schema": "chipmunk.current-localization-evaluation/1", "manifest_sha256": digest,
              "condition": condition, "device": "cpu" if condition == "native" else device,
              "weight_sha256": weights, "script_sha256": sha(Path(__file__).read_bytes()),
              "runtime_sha256": sha(Path(splice.__file__ if condition == "native" else fused.__file__).read_bytes()),
              "gate": {"minimum_active_cells": splice.MINIMUM_ACTIVE_CELLS, "pixel_threshold": splice.PIXEL_THRESHOLD}
                       if condition == "native" else {"image_score": .5, "pixel_threshold": .5},
              "scope": "Current format-selected localization component; same reliability filtering and per-map display cap. "
                       "Does not include competing forensic maps, total UI packet caps, or whole-file confidence."}
    write_new(output / "policy.json", policy)
    with (output / "predictions.jsonl").open("x") as log:
        for index, item in enumerate(manifest["items"]):
            original, pixels, truth = source(root, item)
            data, rgb = input_bytes(pixels, original, condition)
            row = {k: item[k] for k in ("id", "source_id", "family", "label", "image_sha256", "mask_sha256")}
            row.update(condition=condition, input_sha256=sha(data), status="failed")
            started = time.perf_counter()
            try:
                if condition == "native":
                    response = splice.predict(model, Image.fromarray(rgb))
                    area = int((response > splice.PIXEL_THRESHOLD).sum())
                    eligible = area >= splice.MINIMUM_ACTIVE_CELLS
                    result = score_map(response, eligible, splice.PIXEL_THRESHOLD, data, rgb, truth)
                    row.update(gate=eligible, active_cells=area, **result)
                else:
                    prediction = model.predict(Image.fromarray(rgb))
                    response = prediction.raw_mask
                    row.update(**measure(prediction, data, rgb, truth))
                path = output / f"map-{index:04d}.npy"
                np.save(path, response, allow_pickle=False)
                row.update(status="ok", raw_map=path.name, raw_map_sha256=sha(path.read_bytes()))
            except Exception as error:
                row.update(reason=f"{type(error).__name__}: {error}")
            row["runtime_s"] = time.perf_counter() - started
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if index == 0 or (index+1) % 12 == 0 or index == len(manifest["items"])-1:
                print(f"Current {condition}: {index+1}/{len(manifest['items'])}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--condition", choices=("native", "jpeg75"), required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="cpu")
    args = parser.parse_args()
    score(args.manifest_root, args.output, args.condition, args.device)
