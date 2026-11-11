"""Fit and transfer a regional response on already qualified native image pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.signal import convolve2d

from scripts import image_bg_real_detail_deficit_v265 as lama
from scripts import image_flux_product_comparison_v288 as flux

GRID = 24
SCALES = (4, 6, 8)
QUALITY = ("source", "jpeg75")
KINDS = ("original", "ai", "tone", "blur", "copy_move")
FIT_GROUPS = (0, 1, 3, 4, 6, 7, 9, 10)
POINT_GROUPS = (2, 5, 8, 11)
WEIGHT_SHA = "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387"
RIDGE = 4.0
RESULT = flux.PANELS / "image-flux-product-comparison-complete-v288.json"
INPUTS = flux.TRIAL / "inputs.json"
ADAPTER = flux.ROOT / "backend/app/detectors/image/commfor.py"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identity(weight: Path) -> dict:
    assert file_digest(weight) == WEIGHT_SHA
    return {
        "schema": "chipmunk.image-relative-patch-transfer/1",
        "study_sha256": file_digest(Path(__file__)),
        "adapter_sha256": file_digest(ADAPTER),
        "weight_sha256": WEIGHT_SHA,
        "inputs": {
            "lama": file_digest(lama.PRIOR_RESULT),
            "flux": file_digest(INPUTS),
            "baseline": file_digest(RESULT),
        },
        "fit_groups": FIT_GROUPS,
        "point_groups": POINT_GROUPS,
        "formats": QUALITY,
        "region_grid": GRID,
        "region_scales": SCALES,
        "ridge": RIDGE,
        "versions": {name: version(name) for name in ("numpy", "scipy", "pillow", "torch", "timm", "safetensors")},
        "checkpoint": "OwensLab/commfor-model-384@6076002bf0d9dd37537f965ee2f06f826c333b61",
    }


def source_grid(shape: tuple[int, int]) -> np.ndarray:
    """Map native pixels to the fixed full-image output grid."""
    y, x = (np.arange(n) * GRID // n for n in shape)
    return y[:, None] * GRID + x[None, :]


def area_fraction(mask: np.ndarray) -> np.ndarray:
    cells = source_grid(mask.shape).ravel()
    totals = np.bincount(cells, minlength=GRID**2)
    changed = np.bincount(cells, weights=mask.ravel(), minlength=GRID**2)
    return (changed / totals).reshape(GRID, GRID)


def relative_change(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> np.ndarray:
    area = area_fraction(mask)
    inside = area >= 0.5
    remote = ~binary_dilation(area > 0, structure=np.ones((3, 3), dtype=bool))
    if not inside.any() or not remote.any():
        raise ValueError("The fit pair has no interior or remote reference cells")
    delta = after.astype(np.float64) - before
    return delta[inside].mean(axis=0) - delta[remote].mean(axis=0)


def strongest_region(response: np.ndarray) -> tuple[float, tuple[int, int, int]]:
    """Choose one spatially supported region from the robust within-image response."""
    center = np.median(response)
    spread = max(float(np.median(np.abs(response - center))) * 1.4826, 0.02)
    normalized = (response - center) / spread
    best = (-math.inf, (0, 0, 0))
    for side in SCALES:
        score = convolve2d(normalized, np.full((side, side), math.sqrt(side / 6) / side**2), mode="valid")
        y, x = np.unravel_index(np.argmax(score), score.shape)
        if score[y, x] > best[0]:
            best = float(score[y, x]), (int(y), int(x), side)
    return best


def overlap(region: tuple[int, int, int], truth: np.ndarray) -> tuple[float, float]:
    y, x, side = region
    rows, cols = (np.arange(n) * GRID // n for n in truth.shape)
    selected = ((y <= rows) & (rows < y + side))[:, None] & ((x <= cols) & (cols < x + side))[None, :]
    intersection = int((truth & selected).sum())
    return intersection / int((truth | selected).sum()), intersection / int(truth.sum())


def lama_inputs():
    inputs, cases, index = lama.prior.lease(lama.prior.TRIAL, lama.PRIOR_FREEZE_SHA)
    for group, entry in enumerate(inputs["sources"]):
        for encoding, kind, image, masks in lama.views(cases, index, group, entry):
            if encoding in QUALITY:
                yield group, encoding, kind, image, masks


def flux_inputs():
    cases, index = flux.source_cases()
    inputs = json.loads(INPUTS.read_bytes())["sources"]
    assert len(cases) == len(inputs) == 6
    for group, (case, entry) in enumerate(zip(cases, inputs, strict=True)):
        assert case["source_id"] == entry["source_id"]
        images, masks = flux.material(cases, index, group)
        for encoding, quality in zip(QUALITY, (0, 75), strict=True):
            for kind in KINDS:
                payload, decoded = flux.previous.encode(images[kind], quality)
                assert flux.original.view_binding(payload, decoded) == entry["inputs"][str(quality)][kind]
                yield group, encoding, kind, decoded, masks


def feature_reader(weight: Path):
    import torch

    from app.detectors.image import commfor

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = commfor.load_model(weight, "cpu")
    raw_direction = model.head.weight.detach().cpu().numpy()[0].copy()

    def features(image):
        pixels = np.asarray(image.resize((384, 384), Image.Resampling.BILINEAR))
        batch = commfor.to_batch(pixels, [(0, 0, "full")])
        with torch.inference_mode():
            tokens = model.forward_features(batch)
        assert tokens.shape == (1, GRID**2 + 1, len(raw_direction))
        result = tokens[0, 1:].numpy().reshape(GRID, GRID, -1).copy()
        assert np.isfinite(result).all()
        return result

    return features, raw_direction


def fit(rows: dict) -> np.ndarray:
    fitting = []
    for group in FIT_GROUPS:
        for encoding in QUALITY:
            variants = rows[group, encoding]
            original = variants["original"][0]
            ai, mask = variants["ai"]
            positive = relative_change(original, ai, mask)
            ordinary = [relative_change(original, *variants[kind]) for kind in KINDS[2:]]
            fitting.append(positive - np.mean(ordinary, axis=0))
    features = np.stack(fitting)
    scale = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 0.02)
    features = features / scale
    # Ridge in the small dual space, with equal weights for each source/format pair.
    direction = features.T @ np.linalg.solve(features @ features.T + RIDGE * np.eye(len(features)),
                                            np.ones(len(features)))
    return direction / np.linalg.norm(direction)


def inputs_to_features(inputs, extract, label: str) -> dict:
    rows = {}
    for group, encoding, kind, image, masks in inputs:
        group_rows = rows.setdefault((group, encoding), {})
        assert kind not in group_rows
        group_rows[kind] = extract(image), masks[kind]
        if len(group_rows) == len(KINDS):
            print(f"{label} group {group} {encoding} complete", flush=True)
    return rows


def learn_points(rows, directions):
    points = {}
    for arm, direction in directions.items():
        group_max = [max(strongest_region(rows[group, encoding][kind][0] @ direction)[0]
                         for encoding in QUALITY for kind in KINDS if kind != "ai") for group in POINT_GROUPS]
        points[arm] = {"point": sorted(group_max)[-2], "control_group_maxima": group_max}
    return points


def measure(rows, direction, point):
    result = {}
    for encoding in QUALITY:
        counts = Counter()
        ious, recall = [], []
        for (_, fmt), kinds in rows.items():
            if fmt != encoding:
                continue
            counts["groups"] += 1
            for kind, (features, mask) in kinds.items():
                score, region = strongest_region(features @ direction)
                flag = score > point
                counts[f"{kind}_tested"] += 1
                counts[f"{kind}_flagged"] += flag
                if kind == "ai":
                    iou, hit = overlap(region, mask) if flag else (0.0, 0.0)
                    counts["localized"] += iou >= 0.30 and hit >= 0.50
                    ious.append(iou)
                    recall.append(hit)
        result[encoding] = dict(counts) | {"mean_iou": float(np.mean(ious)), "mean_recall": float(np.mean(recall))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("weight", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    locked = json.loads(json.dumps(identity(args.weight)))
    if args.action == "freeze":
        args.output.mkdir(mode=0o700)
        write(args.output / "freeze.json", locked)
        return
    assert locked == json.loads((args.output / "freeze.json").read_bytes())
    start = time.monotonic()
    extract, raw = feature_reader(args.weight)
    development = inputs_to_features(lama_inputs(), extract, "development")
    assert len(development) == 24
    directions = {"classifier_projection": raw, "relative_pair_fit": fit(development)}
    points = learn_points(development, directions)
    write(args.output / "points.json", {name: row | {"direction_sha256": digest(directions[name].tobytes())}
                                       for name, row in points.items()})
    transfer = inputs_to_features(flux_inputs(), extract, "transfer")
    assert len(transfer) == 12
    results = {arm: measure(transfer, direction, points[arm]["point"]) for arm, direction in directions.items()}
    write(args.output / "result.json", {
        "schema": "chipmunk.image-relative-patch-result/1",
        "freeze_sha256": file_digest(args.output / "freeze.json"),
        "points_sha256": file_digest(args.output / "points.json"),
        "status": "complete",
        "source_groups": {"fit": len(FIT_GROUPS), "point": len(POINT_GROUPS), "transfer": 6},
        "forwards": sum(len(rows) for rows in development.values()) + sum(len(rows) for rows in transfer.values()),
        "elapsed_s": round(time.monotonic() - start, 3),
        "transfer": results,
        "serving_admitted": False,
    })


if __name__ == "__main__":
    main()
