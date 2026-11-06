"""Score native/JPEG paired sources without changing the production model selection."""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.iml_vit_candidate import CHECKPOINT_SHA256, ENCODER_SHA256, load, predict  # noqa: E402
from scripts.localization_benchmark import read_manifest, sha, source  # noqa: E402
from scripts.score_splice_localizers import score_map  # noqa: E402


def write_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def input_bytes(pixels: np.ndarray, original: bytes, condition: str) -> tuple[bytes, np.ndarray]:
    if condition == "native":
        return original, pixels
    if condition != "jpeg75":
        raise ValueError("Unknown image condition")
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="JPEG", quality=75)
    data = encoded.getvalue()
    with Image.open(io.BytesIO(data)) as image:
        return data, np.asarray(image.convert("RGB")).copy()


def score(root: Path, assets: Path, output: Path, *, section: str, device: str, limit: int | None,
          lock: Path | None = None) -> None:
    manifest, digest = read_manifest(root)
    rows = manifest["items"]
    if limit is not None:
        # Only use this option for consumed development data, never held-out data.
        if section != "development" or limit < 1:
            raise ValueError("Source limits apply only to development")
        selected: dict[str, set[str]] = {}
        for row in rows:
            family = row["family"]
            ids = selected.setdefault(family, set())
            if len(ids) < limit:
                ids.add(row["source_id"])
        rows = [r for r in rows if r["source_id"] in selected[r["family"]]]
    if len({(r["source_id"], r["label"]) for r in rows}) != len(rows):
        raise ValueError("Ambiguous paired source identity")
    if section == "confirmation":
        if lock is None:
            raise ValueError("Confirmation inference requires the frozen development decision")
        frozen = json.loads(lock.read_text())
        if (frozen.get("schema") != "chipmunk.iml-vit-decision/1"
                or frozen["confirmation_plan_sha256"] != sha((root / "tgif-plan.json").read_bytes())
                or set(frozen["development_source_groups"]) & {r["source_id"] for r in rows}
                or frozen["checkpoint_sha256"] != CHECKPOINT_SHA256
                or frozen["encoder_sha256"] != ENCODER_SHA256
                or frozen["runtime_sha256"] != sha(Path(__file__).with_name("iml_vit_candidate.py").read_bytes())
                or frozen["evaluation_sha256"] != sha(Path(__file__).read_bytes())):
            raise ValueError("Confirmation inputs or candidate differ from the frozen decision")
    output.mkdir(parents=True, exist_ok=False)
    policy = {"schema": "chipmunk.iml-vit-evaluation/1", "section": section,
              "manifest_sha256": digest, "source_groups": sorted({r["source_id"] for r in rows}),
              "script_sha256": sha(Path(__file__).read_bytes()),
              "runtime_sha256": sha(Path(__file__).with_name("iml_vit_candidate.py").read_bytes()),
              "checkpoint_sha256": CHECKPOINT_SHA256, "encoder_sha256": ENCODER_SHA256,
              "device": device, "dtype": "float32", "pixel_threshold": .5,
              "conditions": ["native", "jpeg75"],
              "decision_lock_sha256": sha(lock.read_bytes()) if lock else None,
              "scope": "Native uint8 RGB at most 1024 pixels per side; top-left zero padding, ImageNet normalization, "
                       "publisher ViT/FPN, independent compatible head, crop padding away. No resized or cropped-away sources.",
              "reference": "Original paired edit mask coordinates preserved after JPEG; global compression changes are not edit labels.",
              "display": "Current reliability exclusions, grid quantization, connected-component filtering, and two-region map cap. "
                         "This is a localization-component comparison, not a full UI packet-cap replay."}
    write_new(output / "policy.json", policy)
    torch.set_num_threads(2)
    model = load(assets, device)
    with (output / "predictions.jsonl").open("x") as log:
        for index, item in enumerate(rows):
            original, pixels, truth = source(root, item)
            for condition in policy["conditions"]:
                data, rgb = input_bytes(pixels, original, condition)
                result = {k: item[k] for k in ("id", "source_id", "family", "label", "image_sha256", "mask_sha256")}
                result.update(condition=condition, input_sha256=sha(data), status="failed")
                started = time.perf_counter()
                try:
                    if max(rgb.shape[:2]) > 1024:
                        result.update(status="not_applicable", reason="Native dimensions exceed the pinned 1024-pixel input scope")
                    else:
                        response = predict(model, rgb, device)
                        path = output / f"map-{index:04d}-{condition}.npy"
                        np.save(path, response, allow_pickle=False)
                        result.update(status="ok", raw_map=path.name, raw_map_sha256=sha(path.read_bytes()),
                                      active_fraction=float((response > .5).mean()),
                                      **score_map(response, True, .5, data, rgb, truth))
                except Exception as error:
                    result.update(reason=f"{type(error).__name__}: {error}")
                result["runtime_s"] = time.perf_counter() - started
                log.write(json.dumps(result, allow_nan=False) + "\n")
                log.flush()
            if index == 0 or (index + 1) % 8 == 0 or index == len(rows) - 1:
                print(f"{section}: {index+1}/{len(rows)} source images, both formats", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--section", choices=("development", "confirmation"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--sources-per-family", type=int)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    score(args.manifest_root, args.assets, args.output, section=args.section,
          device=args.device, limit=args.sources_per_family, lock=args.lock)
