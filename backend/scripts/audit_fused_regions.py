"""Offline, fixed-protocol region audit of existing FUSED research probabilities.

No model inference, test-set optimization, or product admission. Both ungated and
image-score-gated policies are reported; the audit does not choose a policy.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sha(data):
    return hashlib.sha256(data).hexdigest()


def metrics(prediction, truth):
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    return {"iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "pixel_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "flagged_fraction": float(prediction.mean()), "intersection": tp,
            "false_positive_pixels": fp, "false_negative_pixels": fn}


def audit(directory: Path):
    source = (directory / "results.json").read_bytes()
    result = json.loads(source)
    data = directory.parent
    manifest_bytes = (data / "manifest.json").read_bytes()
    if sha(manifest_bytes) != result["manifest_sha256"]:
        raise ValueError("Benchmark manifest changed")
    items = {item["id"]: item for item in json.loads(manifest_bytes)["items"]}
    if len(result["rows"]) != len(result["selected_items"]):
        raise ValueError("Wait for the complete selected inference run")
    rows = []
    for row in result["rows"]:
        item = items[row["id"]]
        probabilities = np.load(directory / (item["id"] + ".probabilities.npy"), allow_pickle=False)
        if probabilities.ndim != 2 or not np.isfinite(probabilities).all():
            raise ValueError("Invalid probability raster")
        raw = (probabilities >= 0.5).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        minimum = math.ceil(raw.size * 0.001)
        kept = [label for label in range(1, count) if int(stats[label, cv2.CC_STAT_AREA]) >= minimum]
        selected = np.isin(labels, kept)
        gated = selected if row["score"] >= 0.5 else np.zeros_like(selected)
        truth = np.zeros_like(selected)
        if item["label"]:
            mask_bytes = (data / item["mask"]).read_bytes()
            if sha(mask_bytes) != item["mask_sha256"]:
                raise ValueError("Reference hash mismatch")
            with Image.open(io.BytesIO(mask_bytes)) as mask:
                truth = np.asarray(mask.convert("L")) > 127
            if truth.shape != selected.shape:
                raise ValueError("Reference dimensions mismatch")
        rows.append({"id": item["id"], "family": item["family"], "label": item["label"],
                     "image_score": row["score"], "image_classification_positive": row["score"] >= 0.5,
                     "component_count_raw": count - 1, "minimum_component_pixels": minimum,
                     "retained_component_count": len(kept),
                     "retained_component_areas": [int(stats[label, cv2.CC_STAT_AREA]) for label in kept],
                     "component_filtered": metrics(selected, truth),
                     "component_filtered_image_gated": metrics(gated, truth)})
    summary = {}
    for family in ["all"] + sorted({row["family"] for row in rows}):
        selected = [row for row in rows if family == "all" or row["family"] == family]
        positive = [row for row in selected if row["label"]]
        authentic = [row for row in selected if not row["label"]]
        group = {"edited_n": len(positive), "authentic_n": len(authentic),
                 "edited_image_positive": sum(row["image_classification_positive"] for row in positive),
                 "authentic_image_false_positive": sum(row["image_classification_positive"] for row in authentic)}
        for policy in ("component_filtered", "component_filtered_image_gated"):
            group[policy] = {
                "edited_mean_iou": float(np.mean([row[policy]["iou"] for row in positive])) if positive else None,
                "edited_mean_f1": float(np.mean([row[policy]["pixel_f1"] for row in positive])) if positive else None,
                "authentic_with_regions": sum(row[policy]["flagged_fraction"] > 0 for row in authentic),
                "authentic_mean_flagged_fraction": float(np.mean([row[policy]["flagged_fraction"] for row in authentic])) if authentic else None,
            }
        summary[family] = group
    document = {"source_result_sha256": sha(source), "manifest_sha256": sha(manifest_bytes),
                "protocol": {"mask_threshold": 0.5, "connectivity": 8, "minimum_component_area_fraction": 0.001,
                             "image_gate_threshold": 0.5, "coordinate_space": "original stored pixels"},
                "limitation": "Both policies fixed before this audit; neither is selected or admitted by this test. "
                              "Convenience sample and interpreter parity limitations remain.",
                "summary": summary, "rows": rows}
    (directory / "region-audit.json").write_text(json.dumps(document, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    audit(parser.parse_args().directory)
