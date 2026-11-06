"""Compare pinned localization checkpoints on identical source images and masks."""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detectors.image import fused, jpeg_bitstream  # noqa: E402
from app.detectors.image import localiser as loc  # noqa: E402
from scripts.audit_fused_production_grid import component_masks, probability, to_source  # noqa: E402
from scripts.localization_benchmark import mask_metrics, read_manifest, sha, source  # noqa: E402

SOFAKE_FILE = "fused_sofake.pth"
SOFAKE_SHA256 = "0f9273ba9d48b1a6de0d2b602c7a26cdd3624967f465ff0683668ef9c4359ad3"
WEIGHTS = Path(__file__).resolve().parents[2] / "models/weights/fused"
OPENSDI_BASELINE_RESULTS_SHA256 = "e4f51e95830204776800be498ff6f066ec17520efe55be35e7525e63656132a2"


def sofake_state(path: Path) -> dict[str, torch.Tensor]:
    # The older config names these eight learned pooling queries
    # attention_branches. The released tensor names/shapes are identical.
    expected = dict(fused.EXPECTED_CONFIG)
    expected["attention_branches"] = expected.pop("num_latents")
    expected.update(pooling="mha", final_pooling="attn", seg_global_bias=False)
    fused._check_hash(path, SOFAKE_SHA256)
    if torch.serialization.get_unsafe_globals_in_checkpoint(path):
        raise ValueError("Unexpected checkpoint globals")
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(saved, dict) or set(saved) != {"state_dict", "cfg"} or saved["cfg"]["model"] != expected:
        raise ValueError("Unexpected So-Fake checkpoint architecture")
    return fused._validated_state({"cfg": {"model": fused.EXPECTED_CONFIG}, "state_dict": saved["state_dict"]})


def measure(prediction: fused.FusedPrediction, image_bytes: bytes, pixels: np.ndarray, truth: np.ndarray) -> dict:
    height, width = truth.shape
    grid = loc.grid_shape(width, height, loc.GRID_MAX_SIDE["screening"])
    floating = cv2.resize(prediction.raw_mask, grid, interpolation=cv2.INTER_LINEAR)
    values = np.rint(np.clip(floating.astype(np.float64), 0, 1) * 255) / 255.0
    structure = jpeg_bitstream.read(image_bytes)
    excluded, _ = loc.conditions_mask(pixels, structure.quality_luma if structure else None)
    reliable = loc.reliable_grid(excluded, grid)
    masks, _ = component_masks(values, reliable, width, height)
    selected = masks["per_map_display_cap"] & prediction.candidate_eligibility["eligible"]
    source_selected = to_source(selected, width, height)
    return {"score": prediction.image_score, "gate": prediction.candidate_eligibility["eligible"],
            "metrics": mask_metrics(source_selected, truth),
            "ungated": mask_metrics(to_source(masks["per_map_display_cap"], width, height), truth)}


def compare_rows(rows: list[dict], *, measurement: str = "metrics") -> dict:
    result = {}
    for label in (0, 1):
        samples = [row for row in rows if row["label"] == label]
        if not samples:
            continue
        metric = "iou" if label else "predicted_positive_pixels"
        pairs = np.array([[row[check][measurement][metric] for check in ("opensdi", "sofake")] for row in samples])
        if not label:
            pairs = (pairs > 0).astype(float)
        differences = pairs[:, 1] - pairs[:, 0]
        rng = np.random.default_rng(20261104)
        bootstrap = rng.choice(differences, size=(5_000, len(samples)), replace=True).mean(axis=1)
        interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        result["edited" if label else "authentic"] = {
            "n": len(samples), "paired_metric": "mean IoU" if label else "fraction with any selected region",
            "candidate_minus_baseline": float(differences.mean()), "paired_bootstrap_95pct_interval": interval,
            "better_for_candidate": int((differences > 0 if label else differences < 0).sum()),
            "worse_for_candidate": int((differences < 0 if label else differences > 0).sum()),
            **{check: {"primary_metric": float(pairs[:, index].mean()),
                       "image_gate_positive": sum(row[check]["gate"] for row in samples),
                       "any_retained_region": sum(row[check][measurement]["predicted_positive_pixels"] > 0 for row in samples),
                       **({"zero_overlap": sum(row[check][measurement]["intersection"] == 0 for row in samples),
                           "mean_f1": float(np.mean([row[check][measurement]["pixel_f1"] for row in samples]))}
                          if label else {})}
               for index, check in enumerate(("opensdi", "sofake"))},
        }
    return result


def evaluate(root: Path, output: Path, device: str, *, baseline_cache: Path | None = None) -> None:
    cv2.setNumThreads(1)
    manifest, digest = read_manifest(root)
    cached = {}
    if baseline_cache:
        saved = baseline_cache.read_bytes()
        if sha(saved) != OPENSDI_BASELINE_RESULTS_SHA256:
            raise ValueError("The baseline cache must be the committed OpenSDI regression predictions")
        committed = json.loads(saved)
        if (committed["manifest_sha256"] != digest or committed["device"] != device
                or committed["selected_items"] != [item["id"] for item in manifest["items"]]):
            raise ValueError("The baseline cache belongs to a different manifest or device")
        cached = {row["id"]: row for row in committed["rows"]}
    output.mkdir(parents=True, exist_ok=False)
    policy = {"schema": "chipmunk.image-localizer-comparison/1", "manifest_sha256": digest,
              "source": str(root.resolve()), "dataset": manifest["dataset"], "device": device, "dtype": "float32",
              "opensdi": fused.CHECKPOINT_SHA256, "sofake": SOFAKE_SHA256, "backbone": fused.BACKBONE_SHA256,
              "comparison_script_sha256": sha(Path(__file__).read_bytes()),
              "inference_script_sha256": sha(Path(fused.__file__).read_bytes()),
              "baseline_cache_sha256": OPENSDI_BASELINE_RESULTS_SHA256 if cached else None,
              "selection_rule": "Both unchanged 0.5 thresholds and existing production region constraints; no fitting"}
    (output / "policy.json").write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n")
    candidate = sofake_state(WEIGHTS / SOFAKE_FILE)
    model = fused.load_model(WEIGHTS / fused.CHECKPOINT_FILE, WEIGHTS / fused.BACKBONE_FILE, device)
    print(f"Loaded common runtime on {device}; {len(manifest['items'])} sources per checkpoint", flush=True)
    rows = {item["id"]: {key: item[key] for key in ("id", "family", "label", "image_sha256", "mask_sha256")}
            for item in manifest["items"]}
    for check in ("opensdi", "sofake"):
        if check == "sofake":
            model.ops = fused.TensorEvaluator({key: value.detach().to(device=model.device, dtype=torch.float32)
                                                for key, value in candidate.items()})
        with (output / f"{check}.jsonl").open("x") as log:
            for index, item in enumerate(manifest["items"], 1):
                image_bytes, pixels, truth = source(root, item)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    if image.getexif().get(274, 1) != 1:
                        raise ValueError(f"EXIF orientation differs from reference mask at {item['id']}")
                if check == "opensdi" and baseline_cache:
                    record = cached[item["id"]]
                    native = probability(baseline_cache.parent, record["probabilities"]["native"], (512, 512))
                    prediction = fused.FusedPrediction(native, record["score"], tuple(pixels.shape[1::-1]))
                else:
                    prediction = model.predict(Image.fromarray(pixels))
                path = output / f"{check}-{index:04d}.npy"
                np.save(path, prediction.raw_mask, allow_pickle=False)
                record = {**measure(prediction, image_bytes, pixels, truth),
                          "raw_map": path.name, "raw_map_sha256": sha(path.read_bytes())}
                rows[item["id"]][check] = record
                log.write(json.dumps({"id": item["id"], **record}, allow_nan=False) + "\n")
                log.flush()
                if index == 1 or index % 10 == 0 or index == len(rows):
                    print(f"{check}: {index}/{len(rows)}", flush=True)
    groups = defaultdict(list)
    for row in rows.values():
        groups[row["family"]].append(row)
    result = {"policy": policy, "combined": compare_rows(list(rows.values())),
              "ungated_diagnostic": compare_rows(list(rows.values()), measurement="ungated"),
              "groups": {family: compare_rows(group) for family, group in groups.items()}, "items": list(rows.values())}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["combined"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--baseline-cache", type=Path, help="Replay the committed OpenSDI holdout predictions")
    args = parser.parse_args()
    evaluate(args.manifest_root, args.output, args.device, baseline_cache=args.baseline_cache)
