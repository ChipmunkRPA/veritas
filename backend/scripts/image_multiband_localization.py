# SPDX-License-Identifier: Apache-2.0
"""Compare compact JPEG localizers trained on separate spatial frequency bands."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from scripts import image_region_rank_compare as rank
from scripts.image_removal_peer import POINTS, summary

SCALES = {"fine": (.45, .95, 1.9, 3.8), "coarse": (1.9, 3.8, 7.6, 15.2)}
ARMS = (*SCALES, "agreement")
SCHEMA = "chipmunk.image-multiband-localization/1"
EPOCHS = 8
OCCUPANCY = .35
SEED = 161126


def image(path, expected, mode):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("A measurement input differs from the frozen JPEG panel")
    with Image.open(io.BytesIO(data)) as source:
        return np.asarray(source.convert(mode))


def features(rgb):
    cell = rank.CELL
    height, width = (n // cell for n in rgb.shape[:2])
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    blurred = {scale: cv2.GaussianBlur(ycc, (0, 0), scale) for scale in sorted(set(sum(SCALES.values(), ())))}
    grids = {}
    for name, scales in SCALES.items():
        stack = []
        for fine, coarse in zip(scales, scales[1:]):
            delta = (blurred[fine] - blurred[coarse])[:height * cell, :width * cell]
            tiles = delta.reshape(height, cell, width, cell, 3)
            stack.append(.5 * np.log1p((tiles**2).mean(axis=(1, 3))))
        grid = np.concatenate(stack, axis=2)
        center = np.median(grid, axis=(0, 1))
        spread = np.median(np.abs(grid - center), axis=(0, 1))
        standardized = np.clip((grid - center) / np.maximum(.15, 1.4826 * spread), -6, 6)
        # Keep both absolute detail measurements and their position in the photo's distribution.
        grids[name] = torch.from_numpy(np.concatenate((grid, standardized), axis=2).transpose(2, 0, 1).copy())
    return grids


def target(truth):
    cell = rank.CELL
    h, w = (value // cell for value in truth.shape)
    return torch.from_numpy(truth[:h * cell, :w * cell].reshape(h, cell, w, cell).mean(axis=(1, 3)).astype(np.float32))


def records(root, roles):
    data = rank.read(root / "manifest.json")
    groups = {g["tag"]: g for g in data["groups"] if g["split"] in roles}
    for entry in data["images"]:
        group = groups.get(entry["group"])
        if group:
            row = entry | {"source": group["source"], "family": group["family"], "shape": group["shape"]}
            rgb = image(root / row["file"], row["sha256"], "RGB")
            truth = image(root / row["mask"], row["mask_sha256"], "L") > 0
            if list(truth.shape) != row["shape"] or rgb.shape[:2] != truth.shape or truth.sum() != row["changed_pixels"]:
                raise ValueError("The image coverage changed")
            yield row, features(rgb), truth


def model():
    return nn.Sequential(nn.Conv2d(18, 12, 3, padding=1), nn.GELU(),
                         nn.Conv2d(12, 12, 3, dilation=2, padding=2), nn.GELU(),
                         nn.Conv2d(12, 8, 3, dilation=3, padding=3), nn.GELU(), nn.Conv2d(8, 1, 1))


def readout(network, grid):
    return network(grid.unsqueeze(0))[0, 0]


def loss(logits, occupied):
    foreground = logits[occupied >= OCCUPANCY]
    background = logits[occupied == 0]
    hard = background.topk(max(1, background.numel() // 20)).values
    negative = .4 * F.softplus(background).mean() + .6 * F.softplus(hard).mean()
    return F.softplus(-foreground).mean() + negative if foreground.numel() else negative


def train(rows):
    networks = {band: model() for band in SCALES}
    optimizers = {band: torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.001)
                  for band, net in networks.items()}
    paired = defaultdict(dict)
    for row, grid, truth in rows:
        paired[row["source"], row["role"]][row["quality"]] = (grid, target(truth))
    if not all(set(values) == set(rank.QUALITY) for values in paired.values()):
        raise ValueError("Training requires matched JPEG qualities at every site")
    variants = list(paired.values())
    rng = np.random.default_rng(SEED)
    for epoch in range(EPOCHS):
        total = defaultdict(float)
        for idx in rng.permutation(len(variants)):
            rows75, rows90 = (variants[idx][quality] for quality in rank.QUALITY)
            for band, net in networks.items():
                large, mild = (readout(net, packet[0][band]) for packet in (rows75, rows90))
                value = (loss(large, rows75[1]) + loss(mild, rows90[1])) * .5
                value = value + .10 * F.smooth_l1_loss(large, mild)
                optimizers[band].zero_grad()
                value.backward()
                optimizers[band].step()
                total[band] += value.item()
        print(f"Epoch {epoch + 1}/{EPOCHS}  " + "  ".join(f"{k} {v / len(variants):.4f}" for k, v in total.items()), flush=True)
    return {band: net.eval() for band, net in networks.items()}


def best_region(values, geometry, truth):
    boxes, rings, size, ring_size = geometry
    table = rank.integral(values)
    core = rank.window(table, boxes)
    background = (rank.window(table, rings) - core) / ring_size
    scores = core / size + .25 * (core / size - background)
    index = int(np.argmax(scores))
    count = int(truth.sum())
    best = None
    if count:
        area = size * rank.CELL**2
        shared = rank.window(rank.integral(truth), boxes * rank.CELL)
        useful = (shared * 2 >= count) & (shared * 10 >= (area + count - shared) * 3)
        if useful.any():
            best = 1 + int(np.sum(scores > np.max(scores[useful])))
    left, top, right, bottom = (int(value) for value in boxes[index] * rank.CELL)
    return {"box": [left, top, right, bottom], "selected": (right-left) * (bottom-top),
            "truth": count, "intersection": int(truth[top:bottom, left:right].sum()), "score": float(scores[index]),
            "best_possible_rank": best}


def scale_and_fit(net, rows):
    with torch.inference_mode():
        levels = {band: np.concatenate([readout(net[band], grids[band]).numpy()[::3, ::3].ravel()
                                       for row, grids, _ in rows if row["role"] == "ordinary"]) for band in SCALES}
    return {band: (float(np.median(values)), max(.01, float(np.std(values)))) for band, values in levels.items()}


@torch.inference_mode()
def evaluate(networks, scales, rows, baseline):
    for row, grids, truth in rows:
        comparisons = baseline[row["file"]]
        if comparisons.get("sha256") != row["sha256"]:
            raise ValueError("The current and candidate checks received different JPEGs")
        scores = {}
        for band, net in networks.items():
            center, scale = scales[band]
            scores[band] = (readout(net, grids[band]).numpy() - center) / scale
        scores["agreement"] = (scores["fine"] + scores["coarse"]) / 2
        geometry = rank.layout(tuple(row["shape"]))
        yield row | {"current": comparisons["current"],
                     "checks": {band: best_region(scores[band], geometry, truth) for band in ARMS}}


def choose(rows):
    cutoffs, selected = {}, {}
    for quality in rank.QUALITY:
        candidates = []
        calibration = [row for row in rows if row["quality"] == quality]
        for band in ARMS:
            maxima = defaultdict(lambda: -float("inf"))
            for row in calibration:
                if row["role"] in rank.CONTROLS and not rank.current_flags(row):
                    maxima[row["source"]] = max(maxima[row["source"]], row["checks"][band]["score"])
            controls = sorted(maxima.values(), reverse=True)
            for label, allowed in POINTS.items():
                key = f"{quality}/{band}/{label}"
                point = cutoffs[key] = controls[allowed]
                result = summary(calibration, band, point)
                candidates.append(((-result["added_feathered"], -result["located_combined"],
                                    result["added_ordinary_sources"], result["misplaced"], len(candidates)), key))
        selected[str(quality)] = min(candidates)[1]
    return cutoffs, selected


def validation_baseline(earlier):
    _, measured = rank.incumbent(earlier / "fresh-reference.jsonl")
    repeat = {row["sha256"]: row["agreement"] for row in rank.read(earlier / "agreement.json.gz")["rows"]}
    return {file: {"sha256": row["sha256"], "current": {"jpeg": row["jpeg"], "regional": repeat.get(row["sha256"], row["regional"])}}
            for file, row in measured.items()}


def run(earlier, retained, held, output):

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    training = list(records(retained / "deliveries", {"fit", "transfer"}))
    validation = earlier / "confirmation/deliveries"
    fitted = train(training)
    scales = scale_and_fit(fitted, training)
    training_sources = sorted({row["source"] for row, _, _ in training})
    del training
    print(f"Fitted on {len(training_sources)} original source groups.", flush=True)
    choices = list(evaluate(fitted, scales, records(validation, {"fresh"}), validation_baseline(earlier)))
    selection_sources = sorted({row["source"] for row in choices})
    if set(training_sources).intersection(selection_sources):
        raise ValueError("Validation contains a fitted photograph")
    cutoffs, selected = choose(choices)
    frozen = {"schema": SCHEMA, "training_sources": training_sources, "selection_sources": selection_sources,
              "scale": scales, "cutoffs": cutoffs, "selected": selected,
              "fitting_inputs": rank.sha(retained / "deliveries/manifest.json"),
              "selection_inputs": rank.sha(validation / "manifest.json"),
              "weights": {band: {key: value.tolist() for key, value in net.state_dict().items()} for band, net in fitted.items()},
              "code_sha256": rank.sha(Path(__file__))}
    rank.save(output / "freeze.json.gz", frozen)
    print(json.dumps({"frozen": selected}, sort_keys=True), flush=True)
    incumbent = {row["file"]: row for row in rank.read(held / "held.json.gz")["held"]}
    final = list(evaluate(fitted, scales, records(held / "panel/deliveries", {"fresh"}), incumbent))
    if {row["source"] for row in final}.intersection(training_sources + selection_sources):
        raise ValueError("Transfer contains a trained or selected photograph")
    results = {key: summary([row for row in final if row["quality"] == int(key.split("/")[0])], key.split("/")[1], cutoff)
               for key, cutoff in cutoffs.items()}
    rank.save(output / "result.json.gz", {"freeze": frozen, "selection": choices, "held": final, "results": results})
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("earlier", "retained", "held", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    run(args.earlier, args.retained, args.held, args.output)
