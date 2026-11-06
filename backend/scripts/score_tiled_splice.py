"""Save exact offline tile responses for a predeclared paired-image experiment."""
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
from app.detectors.image import splice  # noqa: E402
from scripts.evaluate_iml_vit import input_bytes, write_new  # noqa: E402
from scripts.localization_benchmark import read_manifest, sha, source  # noqa: E402
from scripts.score_splice_localizers import score_map  # noqa: E402
from scripts.tiled_splice_candidate import INNER_MARGIN, MIN_SIDE, TILE_SHARE, predict  # noqa: E402


def score(root: Path, output: Path, *, section: str, lock: Path | None, normalization: str = "global") -> None:
    manifest, digest = read_manifest(root)
    runtime = Path(__file__).with_name("tiled_splice_candidate.py")
    predictor = predict
    if normalization == "local":
        from scripts.tiled_splice_localnorm_candidate import predict as local_predict
        predictor = local_predict
        runtime = Path(__file__).with_name("tiled_splice_localnorm_candidate.py")
    elif normalization != "global":
        raise ValueError("Unrecognized feature-plane normalization")
    if section == "confirmation":
        if lock is None:
            raise ValueError("Confirmation requires frozen development rules")
        frozen = json.loads(lock.read_text())
        if (frozen["confirmation_plan_sha256"] != sha((root / "tgif-plan.json").read_bytes())
                or frozen["runtime_sha256"] != sha(runtime.read_bytes())
                or frozen["scorer_sha256"] != sha(Path(__file__).read_bytes())
                or frozen["checkpoint_sha256"] != splice.CHECKPOINT_SHA256
                or set(frozen["development_sources"]) & {r["source_id"] for r in manifest["items"]}):
            raise ValueError("Confirmation differs from its frozen tile experiment")
    output.mkdir(parents=True, exist_ok=False)
    policy = {"schema": "chipmunk.splice-tiles/1", "section": section, "manifest_sha256": digest,
              "runtime_sha256": sha(runtime.read_bytes()), "scorer_sha256": sha(Path(__file__).read_bytes()),
              "normalization": normalization,
              "common_tile_runtime_sha256": sha(Path(__file__).with_name("tiled_splice_candidate.py").read_bytes()),
              "checkpoint_sha256": splice.CHECKPOINT_SHA256,
              "conditions": ["native", "jpeg75"], "tile_share": TILE_SHARE,
              "internal_margin": INNER_MARGIN, "minimum_source_side": MIN_SIDE,
              "source_groups": sorted({r["source_id"] for r in manifest["items"]}),
              "lock_sha256": sha(lock.read_bytes()) if lock else None,
              "hypothesis": "Complement the current map only when it has no displayed region; inspect four overlapping "
                            "feature-plane tiles to recover small edits. Same checkpoint and global residual measurements. "
                            "No confidence fusion or service selection changes."}
    write_new(output / "policy.json", policy)
    torch.set_num_threads(2)
    model = splice.load_model(Path(__file__).resolve().parents[2] / "models/weights/splice" / splice.CHECKPOINT_FILE)
    with (output / "predictions.jsonl").open("x") as log:
        for index, item in enumerate(manifest["items"]):
            original, pixels, truth = source(root, item)
            for condition in policy["conditions"]:
                data, rgb = input_bytes(pixels, original, condition)
                row = {k: item[k] for k in ("id", "source_id", "family", "label", "image_sha256", "mask_sha256")}
                row.update(condition=condition, input_sha256=sha(data), status="failed")
                started = time.perf_counter()
                try:
                    image = Image.fromarray(rgb)
                    tiles = predictor(model, image)
                    arrays = {f"tile{i}": tile.response for i, tile in enumerate(tiles)}
                    name = f"tiles-{index:04d}-{condition}.npz"
                    np.savez_compressed(output / name, **arrays)
                    row.update(status="ok", tile_maps=name, tile_maps_sha256=sha((output / name).read_bytes()),
                               tiles=[{"box": tile.box, "active_fraction": tile.active_fraction} for tile in tiles],
                               max_active_fraction=max((t.active_fraction for t in tiles), default=0.))
                    # Native current check is reproduced using the unchanged full-image path.
                    if condition == "native":
                        full = splice.predict(model, image)
                        eligible = int((full > splice.PIXEL_THRESHOLD).sum()) >= splice.MINIMUM_ACTIVE_CELLS
                        row["current_native"] = score_map(full, eligible, splice.PIXEL_THRESHOLD, data, rgb, truth)["metrics"]
                except Exception as error:
                    row.update(reason=f"{type(error).__name__}: {error}")
                row["runtime_s"] = time.perf_counter()-started
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
            if index == 0 or (index+1) % 12 == 0 or index == len(manifest["items"])-1:
                print(f"Tiles {section}: {index+1}/{len(manifest['items'])}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--section", choices=("development", "confirmation"), required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--normalization", choices=("global", "local"), default="global")
    args = parser.parse_args()
    score(args.manifest_root, args.output, section=args.section, lock=args.lock, normalization=args.normalization)
