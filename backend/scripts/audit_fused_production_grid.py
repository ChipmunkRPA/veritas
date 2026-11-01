"""CPU-only, fixed screening-grid audit of saved independent FUSED predictions.

`--declare ROOT` freezes the policy before inference. Audits never load weights,
rotate the dataset reference mask, optimize thresholds, or modify prior outputs.
The replay covers a single map's component caps, not product packet admission.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detectors.image import jpeg_bitstream  # noqa: E402
from app.detectors.image import localiser as loc  # noqa: E402
from app.evidence.image_packets import MAX_PER_MAP  # noqa: E402
from scripts.localization_benchmark import mask_metrics, sha, source  # noqa: E402

MODES = ("native_source_float", "screening_float", "recorded_threshold", "reliable_threshold",
         "area_retained_components", "localiser_region_cap", "per_map_display_cap")


def implementation_hashes() -> dict:
    backend = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), backend / "scripts/fused_research_eval.py", Path(loc.__file__),
             Path(jpeg_bitstream.__file__), backend / "app/detectors/image/adapters/localisers.py",
             backend / "app/evidence/image_packets.py", backend / "scripts/localization_benchmark.py"]
    return {str(p.relative_to(backend)): sha(p.read_bytes()) for p in paths}


def protocol() -> dict:
    return {"schema": "chipmunk.fused-production-grid-policy/1", "mask_threshold": 0.5,
            "image_gate_threshold": 0.5, "comparison": "greater_or_equal", "threshold_tuning": False,
            "native_probability_shape": [512, 512], "profile": "screening", "grid_max_side": loc.GRID_MAX_SIDE["screening"],
            "grid_shape": "loc.grid_shape(stored_width, stored_height, screening_max_side)",
            "probability_resize": "cv2.INTER_LINEAR directly from native512 to aspect-correct grid",
            "recorded_quantization": "np.rint(np.clip(float_grid, 0, 1) * 255).astype(uint8); read as float32 / 255",
            "conditions": {"clipped_high": loc.CLIP_HIGH, "clipped_low": loc.CLIP_LOW,
                           "clip_dilate_px": loc.CLIP_DILATE_PX, "flat_window": loc.FLAT_WINDOW,
                           "flat_std": loc.FLAT_STD, "compressed_quality": loc.COMPRESSED_QUALITY,
                           "compressed_range": loc.COMPRESSED_RANGE, "upscale_block": loc.UPSCALE_BLOCK,
                           "upscale_ratio": loc.UPSCALE_RATIO,
                           "jpeg_quality": "same jpeg_bitstream.read(original_bytes).quality_luma or null",
                           "grid_reliability": "loc.reliable_grid: native unreliable area fraction < 0.5"},
            "connectivity": 8, "minimum_region_fraction": loc.MIN_REGION_FRACTION,
            "localiser_max_regions": loc.MAX_REGIONS, "per_map_max_regions": MAX_PER_MAP,
            "component_order": "mean rounded 4dp desc, area share rounded 4dp desc, native bbox y then x asc",
            "components": "loc.components on exact recorded values and reliability; independently matched to label memberships",
            "source_projection": "Pillow Image.Resampling.NEAREST, identical to evidence overlay; never fill bboxes",
            "coordinate_space": "original stored pixels; no independent EXIF transpose of image or reference",
            "production_comparable_exclusions": ["EXIF orientation other than absent/1", "minimum source side below 64"],
            "modes": list(MODES), "gates": ["ungated", "image_gated"],
            "metrics_domain": "Entire stored source image, including unreliable truth pixels as possible false negatives",
            "scope": "Single-map screening grid and component caps only; no tiled inference, cross-map competition, packet identity/admission, or total packet cap",
            "interpretation": "Conditions exclusion is a measurement limitation, never proof that excluded/unmarked pixels are authentic. Classifier gate does not prove localization. Convenience holdout, not representative accuracy or product admission.",
            "differences_from_old_audit": "Old audit thresholds source-size PyTorch bilinear arrays and uses no reliability or caps. Grid source projection here uses Pillow nearest, not cv2 nearest.",
            "implementation_sha256": implementation_hashes()}


def write_new(path: Path, value: dict) -> None:
    with path.open("x") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def declare(root: Path) -> dict:
    document = {"declared_at": datetime.now(UTC).isoformat(), "inference_has_started": False, **protocol()}
    write_new(root / "production-grid-policy.json", document)
    return document


def to_source(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)) > 0


def component_masks(values: np.ndarray, reliable: np.ndarray, width: int, height: int):
    """Recover all label memberships, checking exact capped output against runtime."""
    binary = (values >= loc.THRESHOLD) & reliable
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    gh, gw = values.shape
    sx, sy = width / gw, height / gh
    minimum = max(1, math.ceil(loc.MIN_REGION_FRACTION * values.size))
    records = []
    for label in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < minimum:
            continue
        inside = values[labels == label]
        # Preserve runtime's floating-operation order at decimal rounding ties.
        x0, y0 = round(x * sx, 1), round(y * sy, 1)
        x1, y1 = round((x + w) * sx, 1), round((y + h) * sy, 1)
        records.append({"component_label": label, "cells": area,
                        "bbox": {"x": x0, "y": y0, "w": round(x1 - x0, 1), "h": round(y1 - y0, 1)},
                        "stats": {"mean": round(float(inside.mean()), 4), "max": round(float(inside.max()), 4),
                                  "area_share": round(area / values.size, 4)}})
    records.sort(key=lambda r: (-r["stats"]["mean"], -r["stats"]["area_share"], r["bbox"]["y"], r["bbox"]["x"]))
    regions, runtime_stats = loc.components(values, reliable, width, height)
    capped = records[:loc.MAX_REGIONS]
    if ([r["region"]["bbox"] for r in regions] != [r["bbox"] for r in capped]
            or runtime_stats != [r["stats"] for r in capped]
            or [r["score"] for r in regions] != [r["stats"]["mean"] for r in capped]):
        raise ValueError("Independent component membership/rank differs from exact production components")
    masks = {"reliable_threshold": binary,
             "area_retained_components": np.isin(labels, [r["component_label"] for r in records]),
             "localiser_region_cap": np.isin(labels, [r["component_label"] for r in capped]),
             "per_map_display_cap": np.isin(labels, [r["component_label"] for r in capped[:MAX_PER_MAP]])}
    return masks, {"threshold_component_count": n - 1, "qualifying_component_count": len(records),
                   "minimum_component_cells": minimum, "ranked_components": records,
                   "runtime_components_exactly_matched": True, "runtime_regions": regions}


def replay(native: np.ndarray, pixels: np.ndarray, truth: np.ndarray, *, jpeg_quality: int | None,
           source_probabilities: np.ndarray, score: float):
    height, width = truth.shape
    if native.shape != (512, 512) or native.dtype != np.float32:
        raise ValueError("Native probabilities must be the original float32 512-square map")
    if source_probabilities.shape != truth.shape or pixels.shape != (*truth.shape, 3):
        raise ValueError("Source coordinates disagree")
    if (not np.isfinite(native).all() or not np.isfinite(source_probabilities).all()
            or not np.isfinite(score) or not 0 <= score <= 1
            or native.min() < 0 or native.max() > 1
            or source_probabilities.min() < 0 or source_probabilities.max() > 1):
        raise ValueError("Invalid probabilities or classifier response")
    grid = loc.grid_shape(width, height, loc.GRID_MAX_SIDE["screening"])
    floating = cv2.resize(native, grid, interpolation=cv2.INTER_LINEAR)
    encoded = np.rint(np.clip(floating, 0, 1) * 255).astype(np.uint8)
    values = encoded.astype(np.float32) / 255
    excluded, conditions = loc.conditions_mask(pixels, jpeg_quality)
    reliable = loc.reliable_grid(excluded, grid)
    grid_masks, components = component_masks(values, reliable, width, height)
    grid_masks = {"screening_float": floating >= 0.5, "recorded_threshold": values >= 0.5, **grid_masks}
    projected = {"native_source_float": source_probabilities >= 0.5,
                 **{name: to_source(mask, width, height) for name, mask in grid_masks.items()}}
    projected_reliable = to_source(reliable, width, height)
    metrics = {"ungated": {name: mask_metrics(mask, truth) for name, mask in projected.items()},
               "image_gated": {name: mask_metrics(mask if score >= 0.5 else np.zeros_like(mask), truth)
                               for name, mask in projected.items()}}
    measurement = {"grid": list(grid), "score": score, "image_gate_positive": score >= 0.5,
                   "jpeg_quality_luma": jpeg_quality, "conditions": conditions,
                   "reliable_grid_cells": int(reliable.sum()), "grid_cells": reliable.size,
                   "reliable_grid_fraction": float(reliable.mean()), "fully_unmeasurable": not bool(reliable.any()),
                   "excluded_native_pixels": int(excluded.sum()),
                   "excluded_truth_pixels_native_conditions": int((excluded & truth).sum()),
                   "excluded_truth_pixels_projected_grid": int((~projected_reliable & truth).sum()),
                   "quantization_threshold_changed_cells": int(((floating >= .5) != (values >= .5)).sum()),
                   "uniform_response_on_reliable_cells": bool(reliable.any() and np.ptp(values[reliable]) == 0),
                   **components, "metrics": metrics}
    return measurement, encoded, reliable, grid_masks


def summary(rows: list[dict]) -> dict:
    edited = [r for r in rows if r["label"]]
    authentic = [r for r in rows if not r["label"]]
    out = {"edited_n": len(edited), "authentic_n": len(authentic),
           "fully_unmeasurable_n": sum(r["fully_unmeasurable"] for r in rows),
           "edited_image_gate_positive": sum(r["image_gate_positive"] for r in edited),
           "authentic_image_gate_false_positive": sum(r["image_gate_positive"] for r in authentic),
           "metrics": {}}
    for gate in ("ungated", "image_gated"):
        out["metrics"][gate] = {}
        for mode in MODES:
            positives = [r["metrics"][gate][mode] for r in edited]
            controls = [r["metrics"][gate][mode] for r in authentic]
            union = sum(m["intersection"] + m["false_positive_pixels"] + m["false_negative_pixels"] for m in positives)
            out["metrics"][gate][mode] = {
                "edited_mean_iou": float(np.mean([m["iou"] for m in positives])) if positives else None,
                "edited_mean_f1": float(np.mean([m["pixel_f1"] for m in positives])) if positives else None,
                "edited_micro_iou": sum(m["intersection"] for m in positives) / union if union else None,
                "edited_zero_overlap": sum(m["intersection"] == 0 for m in positives),
                "authentic_with_positive_pixels": sum(m["predicted_positive_pixels"] > 0 for m in controls),
                "authentic_mean_flagged_fraction": float(np.mean([m["flagged_area_fraction"] for m in controls])) if controls else None}
    return out


def probability(directory: Path, record: dict, expected_shape: tuple[int, int]) -> np.ndarray:
    path = directory / record["path"]
    if path.parent != directory or sha(path.read_bytes()) != record["sha256"]:
        raise ValueError("Saved probability hash/path mismatch")
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != record["shape"] or values.shape != expected_shape or values.dtype != np.float32:
        raise ValueError("Saved probability coordinate/dtype mismatch")
    return values


def png(path: Path, values: np.ndarray) -> dict:
    Image.fromarray(values).save(path)
    return {"path": path.name, "sha256": sha(path.read_bytes()), "shape": list(values.shape)}


def audit(directory: Path) -> dict:
    output = directory / "production-grid-audit.json"
    assets = directory / "production-grid-masks"
    if output.exists() or assets.exists():
        raise FileExistsError("Use new output; previous audit artifacts must not be overwritten")
    result_bytes = (directory / "results.json").read_bytes()
    result = json.loads(result_bytes)
    root = Path(result["manifest_path"]).parent
    manifest_bytes = (root / "manifest.json").read_bytes()
    if sha(manifest_bytes) != result["manifest_sha256"]:
        raise ValueError("Manifest changed")
    policy_bytes = (root / "production-grid-policy.json").read_bytes()
    declared = json.loads(policy_bytes)
    if sha(policy_bytes) != result.get("production_grid_policy_sha256"):
        raise ValueError("Predeclared production grid policy is missing or changed")
    if {k: v for k, v in declared.items() if k not in {"declared_at", "inference_has_started"}} != protocol():
        raise ValueError("Audit implementation or production policy changed after declaration")
    manifest = json.loads(manifest_bytes)
    items = {item["id"]: item for item in manifest["items"]}
    identifiers = [r["id"] for r in result["rows"]]
    if len(items) != len(manifest["items"]) or identifiers != result["selected_items"] or len(identifiers) != len(set(identifiers)):
        raise ValueError("Incomplete, duplicate or reordered inference rows")
    assets.mkdir()
    rows = []
    for row in result["rows"]:
        item = items[row["id"]]
        if (row["family"], row["label"]) != (item["family"], item["label"]):
            raise ValueError("Result label differs from source manifest")
        data, pixels, truth = source(root, item)
        with Image.open(io.BytesIO(data)) as image:
            orientation = image.getexif().get(274)
        excluded = []
        if orientation not in (None, 1):
            excluded.append("Stored-coordinate inference is not EXIF-oriented production input")
        if min(truth.shape) < 64:
            excluded.append("Source falls below the production localiser minimum side")
        structure = jpeg_bitstream.read(data)
        native = probability(directory, row["probabilities"]["native"], (512, 512))
        projected = probability(directory, row["probabilities"]["source"], truth.shape)
        measured, encoded, reliable, masks = replay(native, pixels, truth, source_probabilities=projected,
                                                  jpeg_quality=structure.quality_luma if structure else None, score=row["score"])
        original_metric = measured["metrics"]["ungated"]["native_source_float"]
        for field in ("intersection", "false_positive_pixels", "false_negative_pixels", "iou", "pixel_f1"):
            if original_metric[field] != row[field]:
                raise ValueError("Saved source probability map does not reproduce frozen inference metrics")
        artifacts = {"recorded_scalar": png(assets / (item["id"] + ".scalar.png"), encoded),
                     "reliability": png(assets / (item["id"] + ".reliable.png"), reliable.astype(np.uint8) * 255)}
        for gate in ("ungated", "image_gated"):
            for mode, mask in masks.items():
                shown = mask if gate == "ungated" or row["score"] >= .5 else np.zeros_like(mask)
                artifacts[gate + "." + mode] = png(assets / (item["id"] + "." + gate + "." + mode + ".png"), shown.astype(np.uint8) * 255)
        rows.append({"id": item["id"], "family": item["family"], "label": item["label"],
                     "source_sha256": item["image_sha256"], "reference_sha256": item.get("mask_sha256"),
                     "source_dimensions": [truth.shape[1], truth.shape[0]], "exif_orientation": orientation,
                     "production_comparable": not excluded, "production_comparable_exclusion_reasons": excluded,
                     "probabilities": row["probabilities"], "artifacts": artifacts, **measured})
    families = ["all", *sorted({r["family"] for r in rows})]
    document = {"schema": "chipmunk.fused-production-grid-audit/1", "source_result_sha256": sha(result_bytes),
                "manifest_sha256": sha(manifest_bytes), "policy_sha256": sha(policy_bytes), "protocol": declared,
                "libraries": {"numpy": np.__version__, "opencv": cv2.__version__, "pillow": Image.__version__},
                "coordinate_exclusions": dict(Counter(reason for r in rows for reason in r["production_comparable_exclusion_reasons"])),
                "exif_orientations": dict(Counter(str(r["exif_orientation"]) for r in rows)),
                "metric_conventions": "Macro averages contain edited images only; complete source domain retains excluded truth as misses. Authentic FP is images with any retained pixel, not calibrated probability. Missing/EXIF-incompatible samples are explicitly excluded only from production-comparable summary.",
                "all_stored_coordinate_summary": {f: summary([r for r in rows if f == "all" or r["family"] == f]) for f in families},
                "production_comparable_summary": {f: summary([r for r in rows if r["production_comparable"] and (f == "all" or r["family"] == f)]) for f in families},
                "rows": rows}
    write_new(output, document)
    print(json.dumps(document["production_comparable_summary"]["all"], indent=2), flush=True)
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--declare", type=Path, metavar="ROOT")
    args = parser.parse_args()
    if bool(args.directory) == bool(args.declare):
        parser.error("Choose either --declare ROOT before inference or a completed inference directory")
    if args.declare:
        declare(args.declare)
    else:
        audit(args.directory)
