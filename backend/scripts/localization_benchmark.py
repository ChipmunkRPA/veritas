"""Local image-localization evaluation on pinned, original-byte OpenSDI samples.

This bounded smoke benchmark does not establish product accuracy. It never selects
a threshold on the test samples and never re-encodes inputs before inference.
Downloaded data and outputs stay under ignored data/localization-eval/.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1] / "data" / "localization-eval"
REPO = "nebula/OpenSDI_test"
REVISION = "7e233eaf98fcfee4c74c788f0e34d06feb7ad0df"
# Each pair is a fake and authentic partial-edit shard. Original embedded bytes
# are read from Parquet, not the dataset viewer's rendered/resaved image URLs.
SHARDS = {
    "sd15": ("sd15-00010-of-00019", "sd15-00015-of-00019"),
    "sd2": ("sd2-00004-of-00007", "sd2-00006-of-00007"),
    "sdxl": ("sdxl-00005-of-00009", "sdxl-00007-of-00009"),
    "sd3": ("sd3-00005-of-00010", "sd3-00008-of-00010"),
    "flux": ("flux-00004-of-00007", "flux-00006-of-00007"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(count: int, root: Path = ROOT):
    import fsspec
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError("Use a fresh --root; an existing sample manifest must not be overwritten")
    manifest = {"dataset": REPO, "revision": REVISION, "source_url": f"https://huggingface.co/datasets/{REPO}",
                "license": "CC-BY-SA-4.0; authors describe academic use; evaluation only, not redistributed",
                "selection": f"First {count} unique partial rows in row group zero of each listed shard; no score-based selection",
                "items": []}
    seen = set()
    for family, shards in SHARDS.items():
        for shard in shards:
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{shard}.parquet"
            with fsspec.open(url, "rb", block_size=2**20).open() as stream:
                table = pq.ParquetFile(stream).read_row_group(0)
            accepted = 0
            for row in table.to_pylist():
                if not row["key"].startswith("partial/"):
                    continue
                image = row["image"]["bytes"]
                mask = (row.get("mask") or {}).get("bytes")
                digest = sha(image)
                if digest in seen:
                    continue
                label = int(row["label"])
                if label and not mask:
                    raise ValueError("Edited benchmark item is missing a reference mask")
                item_id = f"{family}-{label}-{accepted:03d}-{digest[:10]}"
                image_path, mask_path = root / (item_id + ".image"), root / (item_id + ".mask")
                image_path.write_bytes(image)
                if mask:
                    mask_path.write_bytes(mask)
                manifest["items"].append({"id": item_id, "family": family, "key": row["key"], "label": label,
                                          "shard": shard, "row_group": 0, "image": image_path.name,
                                          "image_sha256": digest, "mask": mask_path.name if mask else None,
                                          "mask_sha256": sha(mask) if mask else None})
                seen.add(digest)
                accepted += 1
                if accepted >= count:
                    break
            print(f"Prepared {family}/{shard}: {accepted} original-byte samples", flush=True)
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(manifest['items'])} unique images; {root / 'manifest.json'}", flush=True)


def mask_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    if predicted.dtype != bool or truth.dtype != bool or predicted.shape != truth.shape or predicted.ndim != 2:
        raise ValueError("Pixel metrics require equally sized two-dimensional Boolean masks")
    tp, fp, fn = int((predicted & truth).sum()), int((predicted & ~truth).sum()), int((~predicted & truth).sum())
    return {"intersection": tp, "false_positive_pixels": fp, "false_negative_pixels": fn,
            "pixels": int(truth.size), "reference_positive_pixels": tp + fn, "predicted_positive_pixels": tp + fp,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "pixel_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "flagged_area_fraction": float(predicted.mean())}


MODES = ("raw_threshold", "area_retained_components", "localiser_region_cap", "per_map_display_cap")


def policy() -> dict:
    from app.detectors.image import localiser as loc
    from app.evidence.image_packets import MAX_PER_MAP

    return {"threshold": loc.THRESHOLD, "comparison": "greater_or_equal", "connectivity": 8,
            "minimum_region_fraction": loc.MIN_REGION_FRACTION, "localiser_max_regions": loc.MAX_REGIONS,
            "per_map_display_max_regions": MAX_PER_MAP, "screening_grid_max_side": loc.GRID_MAX_SIDE["screening"],
            "component_order": "mean score rounded to 4dp desc; area share rounded to 4dp desc; native bbox y then x asc",
            "projection": "nearest-neighbour from recorded grid to stored image pixels; no EXIF transpose",
            "display_cap_scope": "Component cap only: does not establish packet eligibility, competing maps, or total packet cap"}


def component_masks(binary: np.ndarray, *, width: int, height: int, rule: dict,
                    values: np.ndarray | None = None) -> tuple[dict[str, np.ndarray | None], dict]:
    """Never infer mean-score ranking from a binary mask; never fill component bounding boxes."""
    if binary.ndim != 2 or binary.dtype != bool:
        raise ValueError("A threshold mask must be a two-dimensional Boolean grid")
    if values is not None and (values.shape != binary.shape or not np.isfinite(values).all()):
        raise ValueError("Scalar grid does not match its threshold mask")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=rule["connectivity"])
    minimum = max(1, math.ceil(rule["minimum_region_fraction"] * binary.size))
    ids = [i for i in range(1, n) if int(stats[i, cv2.CC_STAT_AREA]) >= minimum]
    if values is not None:
        gh, gw = binary.shape

        def key(i):
            x, y, _, _, area = (int(v) for v in stats[i])
            return (-round(float(values[labels == i].mean()), 4), -round(area / binary.size, 4),
                    round(y * height / gh, 1), round(x * width / gw, 1))

        ids.sort(key=key)
    masks: dict[str, np.ndarray | None] = {"raw_threshold": binary,
                                         "area_retained_components": np.isin(labels, ids)}
    for mode, cap in (("localiser_region_cap", rule["localiser_max_regions"]),
                      ("per_map_display_cap", min(rule["per_map_display_max_regions"], rule["localiser_max_regions"]))):
        masks[mode] = None if values is None and len(ids) > cap else np.isin(labels, ids[:cap])
    return masks, {"threshold_component_count": n - 1, "qualifying_component_count": len(ids),
                   "minimum_component_cells": minimum, "component_ranking_available": values is not None}


def measure(binary: np.ndarray, truth: np.ndarray, rule: dict, *, values: np.ndarray | None = None) -> dict:
    height, width = truth.shape
    masks, facts = component_masks(binary, width=width, height=height, rule=rule, values=values)
    metrics = {}
    for mode, mask in masks.items():
        if mask is None:
            metrics[mode] = {"status": "unavailable", "reason": "Binary predictions discard the mean scores needed to rank components"}
        else:
            projected = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            metrics[mode] = {"status": "exact", **mask_metrics(projected, truth)}
    return {**facts, "metrics": metrics}


def summary(rows: list[dict]) -> dict:
    out: dict[str, Any] = {"edited_n": sum(bool(r["label"]) for r in rows),
                           "authentic_n": sum(not r["label"] for r in rows), "metrics": {}}
    for mode in MODES:
        edited = [r["metrics"][mode] for r in rows if r["label"] and r["metrics"][mode]["status"] == "exact"]
        controls = [r["metrics"][mode] for r in rows if not r["label"] and r["metrics"][mode]["status"] == "exact"]
        total = sum(r["intersection"] + r["false_positive_pixels"] + r["false_negative_pixels"] for r in edited)
        out["metrics"][mode] = {
            "edited_measurable_n": len(edited), "authentic_measurable_n": len(controls),
            "unavailable_n": sum(r["metrics"][mode]["status"] != "exact" for r in rows),
            "edited_mean_iou": float(np.mean([r["iou"] for r in edited])) if edited else None,
            "edited_mean_f1": float(np.mean([r["pixel_f1"] for r in edited])) if edited else None,
            "edited_micro_iou": sum(r["intersection"] for r in edited) / total if total else None,
            "edited_zero_overlap": sum(r["intersection"] == 0 for r in edited),
            "authentic_with_positive_pixels": sum(r["predicted_positive_pixels"] > 0 for r in controls),
            "authentic_mean_flagged_area": float(np.mean([r["flagged_area_fraction"] for r in controls])) if controls else None,
        }
    return out


def read_manifest(root: Path) -> tuple[dict, str]:
    data = (root / "manifest.json").read_bytes()
    manifest = json.loads(data)
    ids = [item["id"] for item in manifest["items"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("The sample manifest is empty or contains duplicate identities")
    return manifest, sha(data)


def source(root: Path, item: dict) -> tuple[bytes, np.ndarray, np.ndarray]:
    data = (root / item["image"]).read_bytes()
    if sha(data) != item["image_sha256"]:
        raise ValueError(f"Source image checksum mismatch: {item['id']}")
    with Image.open(io.BytesIO(data)) as image:
        pixels = np.asarray(image.convert("RGB"))
    height, width = pixels.shape[:2]
    truth = np.zeros((height, width), bool)
    if item["label"] not in (0, 1):
        raise ValueError("Reference labels must be zero or one")
    if item["label"]:
        mask_data = (root / item["mask"]).read_bytes()
        if sha(mask_data) != item["mask_sha256"]:
            raise ValueError(f"Reference mask checksum mismatch: {item['id']}")
        with Image.open(io.BytesIO(mask_data)) as mask:
            if mask.size != (width, height):
                raise ValueError(f"Reference mask dimensions mismatch: {item['id']}")
            truth = np.asarray(mask.convert("L")) > 127
        if not truth.any():
            raise ValueError(f"Edited reference mask is empty: {item['id']}")
    return data, pixels, truth


def _grid(root: Path, record: dict, *, binary: bool) -> np.ndarray:
    data = (root / record["path"]).read_bytes()
    if record.get("sha256") is not None and sha(data) != record["sha256"]:
        raise ValueError(f"Saved prediction checksum mismatch: {record['path']}")
    with Image.open(io.BytesIO(data)) as image:
        if image.mode != "L":
            raise ValueError("Saved prediction must be an 8-bit grayscale PNG")
        grid = np.asarray(image)
    if binary and not np.isin(grid, [0, 255]).all():
        raise ValueError("A saved binary prediction contains intermediate values")
    return grid > 0 if binary else grid.astype(np.float32) / 255.0


def _record(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha(path.read_bytes())}


def _write_result(path: Path, *, rows: list[dict], rule: dict, manifest_sha: str, **metadata) -> dict:
    result = {"schema": "chipmunk.localization-smoke-metrics/2", "manifest_sha256": manifest_sha,
              "postprocessing": rule, "protocol": "screening; persisted uint8 grid; conditions mask; fixed threshold; no tuning",
              "limitation": "Deterministic smoke sample; not representative product validation or model admission. No accuracy claim.",
              "metric_conventions": {"macro": "unweighted mean over measurable edited images; no authentic empty-mask IoU included",
                                     "micro_iou": "pooled edited-image intersection / pooled union",
                                     "empty_masks": "IoU and F1 are 1 only when both masks are empty; undefined precision/recall are null",
                                     "missing": "Excluded from means and explicitly counted; subset means must not be compared as full-set estimates"},
              "measurement_script_sha256": sha(Path(__file__).read_bytes()),
              "overall": summary(rows), "groups": {family: summary([r for r in rows if r["family"] == family])
                                                    for family in sorted({r["family"] for r in rows})},
              "rows": rows, **metadata}
    with path.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["overall"], indent=2), flush=True)
    print(path, flush=True)
    return result


def remeasure(root: Path = ROOT, *, source_results: Path | None = None, output: Path | None = None) -> dict:
    """Recompute mask metrics only; no model imports, network calls, or changes to frozen inputs/results."""
    from app.detectors.image import localiser as loc

    path = source_results or root / "mesorch-smoke-results.json"
    output = output or root / "mesorch-smoke-metrics-v2.json"
    if output.exists():
        raise FileExistsError(f"Choose a new --output; refusing to overwrite {output}")
    frozen_bytes = path.read_bytes()
    frozen = json.loads(frozen_bytes)
    manifest, manifest_sha = read_manifest(root)
    if frozen.get("manifest_sha256") != manifest_sha:
        raise ValueError("Saved inference results do not match this sample manifest")
    prior = {row["id"]: row for row in frozen["rows"]}
    if len(prior) != len(frozen["rows"]) or set(prior) != {item["id"] for item in manifest["items"]}:
        raise ValueError("Saved inference rows do not exactly cover the manifest")
    rule = frozen.get("postprocessing") or policy()
    if frozen.get("threshold", rule["threshold"]) != rule["threshold"]:
        raise ValueError("Saved predictions used a different threshold")
    rows = []
    for item in manifest["items"]:
        _, pixels, truth = source(root, item)
        height, width = truth.shape
        old = prior[item["id"]]
        if (old["family"], old["label"]) != (item["family"], item["label"]):
            raise ValueError("Saved inference label or family differs from its manifest")
        saved = old.get("predictions") or {"threshold": {"path": item["id"] + ".prediction.png"}}
        binary = _grid(root, saved["threshold"], binary=True)
        if binary.shape[::-1] != loc.grid_shape(width, height, rule["screening_grid_max_side"]):
            raise ValueError("Saved prediction dimensions differ from the recorded screening policy")
        values = None
        if "scalar" in saved and "reliable" in saved:
            values, reliable = _grid(root, saved["scalar"], binary=False), _grid(root, saved["reliable"], binary=True)
            if values.shape != binary.shape or reliable.shape != binary.shape or not np.array_equal(binary, (values >= rule["threshold"]) & reliable):
                raise ValueError("Saved scalar/reliability grids do not reproduce the threshold mask")
        measured = measure(binary, truth, rule, values=values)
        raw_prior = old["metrics"]["raw_threshold"] if "metrics" in old else old
        for key in ("intersection", "false_positive_pixels", "false_negative_pixels", "iou", "pixel_f1", "flagged_area_fraction"):
            if not math.isclose(measured["metrics"]["raw_threshold"][key], raw_prior[key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Saved prediction fails frozen raw metric reproduction: {item['id']} / {key}")
        if "regions" in old and old["regions"] != min(measured["qualifying_component_count"], rule["localiser_max_regions"]):
            raise ValueError("Reconstructed region count differs from the frozen result")
        rows.append({"id": item["id"], "family": item["family"], "label": item["label"], "width": width, "height": height,
                     "predictions": {key: _record(root, root / record["path"]) for key, record in saved.items()}, **measured})
    return _write_result(output, rows=rows, rule=rule, manifest_sha=manifest_sha, model=frozen["model"],
                         weights_revision=frozen["weights_revision"], inference_device=frozen.get("device", frozen.get("inference_device")),
                         operation="remeasure_saved_predictions_without_inference", source_results_sha256=sha(frozen_bytes),
                         source_results_path=path.name, frozen_raw_metrics_reproduced=True,
                         saved_prediction_integrity=("Original binary files had no recorded checksums; hashes captured now. Reproducing prior metric counts is not proof of original spatial identity."
                                                     if any("predictions" not in row for row in frozen["rows"]) else "Saved grid checksums verified"))


def evaluate(device: str, root: Path = ROOT, *, output: Path | None = None):
    from app.detectors.image import jpeg_bitstream
    from app.detectors.image import localiser as loc
    from app.detectors.image.adapters.loaders import load_mesorch
    from app.detectors.image.mesorch import predict, to_batch

    output = output or root / "mesorch-smoke-results-v2.json"
    if output.exists():
        raise FileExistsError(f"Choose a new --output; refusing to overwrite {output}")
    prediction_dir = root / (output.stem + "-grids")
    prediction_dir.mkdir(exist_ok=False)
    manifest, manifest_sha = read_manifest(root)
    rule = policy()
    loaded = load_mesorch("mesorch.full_98", device, "float32")
    rows = []
    for item in manifest["items"]:
        # Ground-truth masks describe the stored pixel space, so the benchmark
        # deliberately evaluates without applying EXIF orientation to only one side.
        data, pixels, truth = source(root, item)
        height, width = pixels.shape[:2]
        start = time.monotonic()
        native = predict(loaded.obj, to_batch(pixels), device)[0]
        grid = loc.grid_shape(width, height, loc.GRID_MAX_SIDE["screening"])
        values = cv2.resize(native, grid, interpolation=cv2.INTER_LINEAR)
        scalar = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
        values = scalar.astype(np.float32) / 255
        structure = jpeg_bitstream.read(data)
        unreliable, _ = loc.conditions_mask(pixels, structure.quality_luma if structure else None)
        reliable = loc.reliable_grid(unreliable, grid)
        selected = (values >= loc.THRESHOLD) & reliable
        measured = measure(selected, truth, rule, values=values)
        predictions = {}
        for name, grid_values in (("scalar", scalar), ("reliable", reliable.astype(np.uint8) * 255),
                                  ("threshold", selected.astype(np.uint8) * 255)):
            path = prediction_dir / f"{item['id']}.{name}.png"
            Image.fromarray(grid_values).save(path)
            predictions[name] = _record(root, path)
        row = {"id": item["id"], "family": item["family"], "label": item["label"],
               "width": width, "height": height, "seconds": time.monotonic() - start,
               "predictions": predictions, **measured}
        rows.append(row)
        print(f"{item['id']}: qualified={measured['qualifying_component_count']} raw IoU={measured['metrics']['raw_threshold']['iou']:.3f}", flush=True)
    return _write_result(output, rows=rows, rule=rule, manifest_sha=manifest_sha, model="loc.mesorch",
                         weights_revision=loaded.revision, inference_device=device, operation="inference_and_measurement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate", "remeasure"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="New results path; existing results are never overwritten")
    parser.add_argument("--source-results", type=Path, help="Frozen inference results for remeasure")
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        parser.error("--count must be between 1 and 100 per row group")
    if args.action == "prepare":
        prepare(args.count, args.root)
    elif args.action == "evaluate":
        evaluate(args.device, args.root, output=args.output)
    else:
        remeasure(args.root, source_results=args.source_results, output=args.output)
