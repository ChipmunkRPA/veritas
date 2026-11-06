"""Extract crop responses on the exact already-consumed 24-source development panel.

This entrypoint cannot run fresh confirmation. It keeps every source and error.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detectors.image import fused, splice  # noqa: E402
from scripts.evaluate_iml_vit import input_bytes, write_new  # noqa: E402
from scripts.fused_crop_v6 import predict  # noqa: E402
from scripts.localization_benchmark import read_manifest, sha, source  # noqa: E402
from scripts.score_splice_localizers import score_map  # noqa: E402

DEVELOPMENT_MANIFEST_SHA256 = "4ec157d240f88fce43e554d5dee0bd2da9ac388345d3f0a188c93826629c8230"
SELECTION_SHA256 = "36a7c32af569d912b0d3737c2486092d4cf37469c2a652e8bd5acfa1cf55e04d"


def code_files() -> list[Path]:
    backend = Path(__file__).resolve().parents[1]
    return [backend / p for p in (
        "scripts/score_fused_crop_v6.py", "scripts/fused_crop_v6.py", "scripts/review_fused_crop_v6.py",
        "scripts/score_splice_localizers.py", "scripts/audit_fused_production_grid.py",
        "scripts/localization_benchmark.py", "scripts/evaluate_iml_vit.py",
        "app/detectors/image/fused.py", "app/detectors/image/splice.py",
        "app/detectors/image/localiser.py", "app/detectors/image/jpeg_bitstream.py",
        "tests/test_fused_crop_v6.py",
    )] + [backend.parent / "docs/fused-crop-v6-protocol.md"]


def code_pins() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {str(p.relative_to(root)): sha(p.read_bytes()) for p in code_files()}


def selected(root: Path, selection: Path) -> tuple[dict, str, list[dict]]:
    manifest, digest = read_manifest(root)
    previous = json.loads(selection.read_text())
    if (sha(selection.read_bytes()) != SELECTION_SHA256
            or digest != DEVELOPMENT_MANIFEST_SHA256 or previous.get("schema") != "chipmunk.iml-vit-evaluation/1"
            or previous.get("section") != "development" or previous.get("manifest_sha256") != digest):
        raise ValueError("Crop development must reuse the consumed IML development panel")
    groups = previous["source_groups"]
    if len(groups) != 24 or len(set(groups)) != 24:
        raise ValueError("Development source set differs from the original 24 groups")
    rows = [r for r in manifest["items"] if r["source_id"] in groups]
    if (len(rows) != 48 or len({(r["source_id"], r["label"]) for r in rows}) != 48
            or {f: sum(r["family"] == f and r["label"] == 0 for r in rows)
                for f in ("firefly", "sd2", "flux-fill")} != {"firefly": 8, "sd2": 8, "flux-fill": 8}):
        raise ValueError("Incomplete or changed source-paired development selection")
    return manifest, digest, rows


def score(root: Path, selection: Path, assets: Path, output: Path, device: str) -> None:
    _, digest, rows = selected(root, selection)
    output.mkdir(parents=True, exist_ok=False)
    policy = {"schema": "chipmunk.fused-crop-development/6", "section": "development",
              "manifest_sha256": digest, "selection_sha256": sha(selection.read_bytes()),
              "source_groups": sorted({r["source_id"] for r in rows}),
              "item_ids": [r["id"] for r in rows], "device": device, "dtype": "float32",
              "weights": {"fused": fused.CHECKPOINT_SHA256, "semantic": fused.BACKBONE_SHA256,
                          "native_baseline": splice.CHECKPOINT_SHA256},
              "code_sha256": code_pins(),
              "conditions": ["native", "jpeg75"], "crop_share": 0.6, "internal_margin": 0.06,
              "pixel_threshold": 0.5, "minimum_source_side": 320,
              "scope": "Consumed development only. Current PNG splice and JPEG inpainting components reproduced. "
                       "Four native-coordinate RGB crops use the unchanged FUSED checkpoint and semantic backbone. "
                       "No service, authorship probability, whole-file confidence or fresh-validation claim."}
    write_new(output / "policy.json", policy)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    model = fused.load_model(assets / "fused" / fused.CHECKPOINT_FILE,
                             assets / "fused" / fused.BACKBONE_FILE, device)
    native_model = splice.load_model(assets / "splice" / splice.CHECKPOINT_FILE)
    print(json.dumps({"loaded": True, "planned_inputs": 96, "crops_per_input": 4}), flush=True)
    with (output / "predictions.jsonl").open("x") as log, torch.inference_mode():
        for index, item in enumerate(rows):
            original, pixels, truth = source(root, item)
            for condition in policy["conditions"]:
                data, rgb = input_bytes(pixels, original, condition)
                row = {k: item[k] for k in ("id", "source_id", "family", "label", "image_sha256", "mask_sha256")}
                row.update(condition=condition, input_sha256=sha(data), status="failed", baseline=None)
                started = time.perf_counter()
                name = f"maps-{index:03d}-{condition}.npz"
                try:
                    image = Image.fromarray(rgb)
                    if condition == "native":
                        baseline = splice.predict(native_model, image)
                        gate = int((baseline > splice.PIXEL_THRESHOLD).sum()) >= splice.MINIMUM_ACTIVE_CELLS
                        threshold = splice.PIXEL_THRESHOLD
                    else:
                        result = model.predict(image)
                        baseline, gate, threshold = result.raw_mask, result.candidate_eligibility["eligible"], 0.5
                        row["baseline_image_score"] = result.image_score
                    row["baseline"] = score_map(baseline, gate, threshold, data, rgb, truth)["metrics"]
                    crops = predict(model, image)
                    if not crops:
                        row.update(status="not_applicable", reason="Source is below the fixed 320-pixel minimum")
                    else:
                        np.savez_compressed(output / name, baseline=baseline,
                                            **{f"crop{i}": c.response for i, c in enumerate(crops)})
                        row.update(status="ok", arrays=name, arrays_sha256=sha((output / name).read_bytes()),
                                   crops=[{"box": c.box, "image_score": c.image_score,
                                           "rgb_sha256": c.rgb_sha256} for c in crops])
                except Exception as error:
                    row.update(reason=f"{type(error).__name__}: {error}")
                row["runtime_s"] = time.perf_counter() - started
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps({"completed": index * 2 + (condition == "jpeg75") + 1,
                                  "planned": 96, "status": row["status"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("consumed_selection", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    score(args.manifest_root, args.consumed_selection, args.assets, args.output, args.device)
