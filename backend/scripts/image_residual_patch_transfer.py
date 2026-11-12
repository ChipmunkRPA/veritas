# SPDX-License-Identifier: Apache-2.0
"""Test small learned residual features with matched processing controls."""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as fn

from scripts import image_edge_spread_transfer as panel
from scripts import image_relative_patch_transfer as spatial
from scripts import image_residual_color_transfer as incumbent
from scripts import image_wavelet_region_transfer as wavelet

SCHEMA = "chipmunk.image-residual-patch-transfer/1"
ARMS = ("binary_residual", "matched_processing")
BASELINE = "format_reference"
FILES = ("plan.json", "attempts.jsonl", "result.json")
PATCH = 24
GRID = spatial.GRID
SEED = 61111
EPOCHS = 8
FIT_BATCH = 64
SCORE_BATCH = 128
PATCHES_PER_SOURCE = 16
LEARNING_RATE = .002
CONSISTENCY = .30
POINT_FLOOR = .25


class PatchHead(nn.Module):
    """Encode high-pass RGB at two fixed scales with a small learned patch head."""

    def __init__(self, classes):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(6, 12, 3, stride=2), nn.Tanh(),
                                  nn.Conv2d(12, 16, 3, stride=2), nn.Tanh())
        self.classify = nn.Linear(32, classes)

    def forward(self, rgb):
        fine = rgb - fn.avg_pool2d(rgb, 3, 1, padding=1, count_include_pad=False)
        broad = rgb - fn.avg_pool2d(rgb, 7, 1, padding=3, count_include_pad=False)
        residual = torch.cat((fine * 16, broad * 8), dim=1).clamp(-3, 3)
        features = self.conv(residual.contiguous(memory_format=torch.channels_last))
        pooled = torch.cat((features.mean(dim=(2, 3)), features.abs().mean(dim=(2, 3))), dim=1)
        return self.classify(pooled)


def response(logits):
    return logits[..., -1] - torch.logsumexp(logits[..., :-1], dim=-1)


def protocol(args):
    directories = {"fit": args.fit, "calibration": args.calibration, "earlier": args.earlier,
                   "later": args.later, "fresh": args.fresh}
    plans = [panel.read(folder / "plan.json")["cases"] for folder in directories.values()]
    sizes = [8, 8, 12, 12, 7]
    for name in ("flickr_photo_id", "flickr_owner"):
        groups = [{row[name] for row in plan} for plan in plans]
        panel.require([len(group) for group in groups] == sizes and len(set.union(*groups)) == sum(sizes),
                      "Expected distinct recorded photographers and source images")
    qualification = panel.read(args.fresh / "result.json")
    counts = qualification["qualified"]
    panel.require(set(counts) <= {"sdxl", "kandinsky"} and sum(counts.values()) >= 4
                  and min(counts.get("sdxl", 0), counts.get("kandinsky", 0)) >= 1,
                  "The fresh panel requires at least four source groups and both editing families")
    code = (*map(lambda module: Path(module.__file__), (panel, panel.paired, panel.blend, incumbent, spatial, wavelet)),
            Path(__file__))
    return {
        "schema": SCHEMA,
        "code": {path.name: panel.sha(path) for path in code},
        "intakes": {label: {name: panel.sha(folder / name) for name in FILES} for label, folder in directories.items()},
        "qualified_fresh_by_family": counts,
        "runtime": {name: version(name) for name in ("numpy", "opencv-python-headless", "pillow", "torch", "safetensors")},
        "recipe": {
            "arms": list(ARMS), "patch_pixels": PATCH, "grid": GRID, "seed": SEED, "epochs": EPOCHS,
            "fit_batch": FIT_BATCH, "learning_rate": LEARNING_RATE, "paired_delivery_consistency": CONSISTENCY,
            "target_patch_occupancy": .8, "max_training_patches_per_source": PATCHES_PER_SOURCE,
            "controls": "Eight separate source groups. Strictly exceed the second-highest control-group maximum, floor .25.",
            "localization": "Best supported native rectangle from the fixed 24 x 24 grid. IoU >= 0.30 and recall >= 0.50.",
            "advancement": "Processing-aware arm must add two feathered locations in each reused panel and one in fresh, "
                           "preserve at least the binary arm's useful feathered count and add at most four false control "
                           "flags on at most three groups versus the format incumbent per panel. "
                           "Fresh promotion requires >=6 groups and >=2 per editor.",
            "confidence_contribution": 0,
        },
    }


def configured():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def scan_patches(pixels, ys, xs):
    half = PATCH // 2
    padded = np.pad(pixels, ((half, half), (half, half), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (PATCH, PATCH), axis=(0, 1))
    return np.ascontiguousarray(windows[ys, xs])


def centers(truth, slot):
    half = PATCH // 2
    ys, xs = np.nonzero(truth)
    yy, xx = np.meshgrid(np.arange(max(half, ys.min()), min(truth.shape[0] - half, ys.max() + 1), 8),
                         np.arange(max(half, xs.min()), min(truth.shape[1] - half, xs.max() + 1), 8), indexing="ij")
    y, x = yy.ravel(), xx.ravel()
    area = cv2.integral(truth.astype(np.uint8))
    count = area[y + half, x + half] + area[y - half, x - half] - area[y + half, x - half] - area[y - half, x + half]
    valid = np.flatnonzero(count >= .8 * PATCH * PATCH)
    order = sorted(valid, key=lambda i: hashlib.sha256(f"{SEED}:{slot}:{int(y[i])}:{int(x[i])}".encode()).digest())
    panel.require(len(order) >= 4, "A development source lacks four interior training patches")
    return y[order[:PATCHES_PER_SOURCE]], x[order[:PATCHES_PER_SOURCE]]


def native_deliveries(pixels):
    return [np.asarray(panel.paired.encoded(pixels, fmt)[1]) for fmt in panel.FORMATS]


def training_patches(folder):
    _, plan, receipt = panel.scoring.checked_inputs(folder)
    batches, targets, sizes = [], [], []
    for slot, _, images, truth in panel.scoring.panels(folder, plan, receipt):
        ys, xs = centers(truth, slot)
        y, x = np.nonzero(truth)
        bounds = int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1
        original, processed = images["original"], images["processed"]
        feathered = panel.blend.local(original, images["edited"], bounds, feather=True)
        controls = wavelet.view_controls(original, processed, bounds)
        variants = (*controls, images["edited"], feathered)
        labels = (0, 1, 1, 0, 0, 0, 2, 2)
        for pixels, label in zip(variants, labels, strict=True):
            batches.append(np.stack([scan_patches(view, ys, xs) for view in native_deliveries(pixels)], axis=1))
            targets.extend([label] * len(ys))
        sizes.append(len(ys))
    panel.require(len(sizes) == 8, "Need eight development source groups")
    return torch.from_numpy(np.concatenate(batches)), torch.tensor(targets, dtype=torch.long), sizes


def train(batches, labels, arm):
    torch.manual_seed(SEED)
    order = torch.Generator().manual_seed(SEED + 1)
    controlled = arm == ARMS[1]
    net = PatchHead(3 if controlled else 2).to(memory_format=torch.channels_last)
    targets = labels if controlled else (labels == 2).long()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=.01)
    losses = []
    for _ in range(EPOCHS):
        epoch = []
        for ids in torch.randperm(len(targets), generator=order).split(FIT_BATCH):
            optimizer.zero_grad(set_to_none=True)
            inputs = batches[ids].reshape(-1, 3, PATCH, PATCH).float() / 255
            logits = net(inputs).reshape(len(ids), 2, -1)
            loss = fn.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets[ids].repeat_interleave(2))
            if controlled:
                contrasts = response(logits)
                loss += CONSISTENCY * fn.smooth_l1_loss(contrasts[:, 0], contrasts[:, 1])
            loss.backward()
            optimizer.step()
            epoch.append(float(loss.detach()))
        panel.require(np.isfinite(epoch).all(), "Nonfinite learning step")
        losses.append(float(np.mean(epoch)))
    return net.eval(), losses


@torch.inference_mode()
def propose(nets, pixels):
    yy, xx = np.meshgrid(*(np.minimum(n - 1, (2 * np.arange(GRID) + 1) * n // (2 * GRID))
                           for n in pixels.shape[:2]), indexing="ij")
    crops = torch.from_numpy(scan_patches(pixels, yy.ravel(), xx.ravel()))
    predictions = {}
    for name, net in nets.items():
        logits = torch.cat([response(net(batch.float() / 255)) for batch in crops.split(SCORE_BATCH)])
        panel.require(len(logits) == GRID ** 2 and torch.isfinite(logits).all().item(), "Invalid crop responses")
        score, region = spatial.strongest_region(logits.numpy().reshape(GRID, GRID))
        predictions[name] = {"score": score, "grid_region": list(region)}
    return predictions


def points(folder, nets):
    maxima = {fmt: {arm: [] for arm in ARMS} for fmt in panel.FORMATS}
    _, plan, receipt = panel.scoring.checked_inputs(folder)
    for _, _, images, _ in panel.scoring.panels(folder, plan, receipt):
        controls = panel.development_controls(images["original"], images["processed"])
        for fmt in panel.FORMATS:
            results = [propose(nets, np.asarray(panel.paired.encoded(view, fmt)[1])) for view in controls.values()]
            for arm in ARMS:
                maxima[fmt][arm].append(max(row[arm]["score"] for row in results))
    panel.require(all(len(arm) == 8 for row in maxima.values() for arm in row.values()), "Require eight point groups")
    return {fmt: {arm: {"point": max(POINT_FLOOR, sorted(rows)[-2]), "source_maxima": rows}
                  for arm, rows in results.items()} for fmt, results in maxima.items()}


def confirm(folder, nets, calibrated):
    for source, views, known in panel.paired.inputs(folder):
        for role, pixels in panel.native_variants(views, source["box"]).items():
            truth = np.any(pixels != views["original"], axis=2) if role in panel.POSITIVE else np.zeros_like(known)
            panel.require(not np.any(truth & ~known), "Edit escapes the qualified source rectangle")
            y, x = (np.arange(n) * GRID // n for n in truth.shape)
            for fmt in panel.FORMATS:
                body, image = panel.paired.encoded(pixels, fmt)
                rgb = np.asarray(image)
                proposals = propose(nets, rgb)
                measurements = {}
                for arm, item in proposals.items():
                    top, left, side = item["grid_region"]
                    mask = (((top <= y) & (y < top + side))[:, None] & ((left <= x) & (x < left + side))[None, :])
                    item["ungated_useful"] = panel.useful(panel.scoring.measure(mask, truth))
                    mask = mask & (item["score"] > calibrated[fmt][arm]["point"])
                    measurements[arm] = panel.scoring.measure(mask, truth)
                ref = incumbent.format_reference(fmt, body, rgb)
                mask = (panel.paired.boxed(ref["box"], truth.shape, True) if ref and ref["selected"] else np.zeros_like(truth))
                measurements[BASELINE] = panel.scoring.measure(mask, truth)
                yield {"slot": source["slot"], "generator": source["generator"], "role": role, "format": fmt,
                       "input_sha256": panel.paired.shared.digest(body), "truth_sha256": panel.paired.shared.digest(truth.tobytes()),
                       "proposals": proposals | {BASELINE: ref}, "measurements": measurements}


def summary(rows):
    results = {}
    for fmt in panel.FORMATS:
        results[fmt] = {}
        for arm in (*ARMS, BASELINE):
            counts = Counter(dict.fromkeys(("sharp", "feathered", "ordinary", "original", "added_feathered",
                                            "added_controls", "ungated_useful"), 0))
            groups = set()
            for row in rows:
                if row["format"] != fmt:
                    continue
                role, m = row["role"], row["measurements"]
                if role in panel.POSITIVE:
                    hit = panel.useful(m[arm])
                    counts["sharp" if role == "edited" else "feathered"] += hit
                    counts["added_feathered"] += role == "feathered_edit" and hit and not panel.useful(m[BASELINE])
                    counts["ungated_useful"] += arm != BASELINE and row["proposals"][arm]["ungated_useful"]
                elif m[arm]["reported"]:
                    counts["ordinary"] += 1
                    counts["original"] += role == "original"
                    if not m[BASELINE]["reported"]:
                        counts["added_controls"] += 1
                        groups.add(row["slot"])
            results[fmt][arm] = dict(counts) | {"added_control_sources": len(groups)}
    return results


def advancement(results, qualified_fresh):
    passed = {}
    for fmt in panel.FORMATS:
        checks = []
        for label, content in results.items():
            candidate, binary = (content[fmt][arm] for arm in (ARMS[1], ARMS[0]))
            checks.append(candidate["added_feathered"] >= (1 if label == "fresh" else 2)
                          and candidate["feathered"] >= binary["feathered"]
                          and candidate["added_controls"] <= 4 and candidate["added_control_sources"] <= 3)
        passed[fmt] = all(checks) and sum(qualified_fresh.values()) >= 6 and min(qualified_fresh.values()) >= 2
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "confirm"))
    for name in ("fit", "calibration", "earlier", "later", "fresh", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    frozen = protocol(args)
    freeze = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        panel.write(freeze, frozen)
        return
    panel.require(frozen == panel.read(freeze), "Frozen method or sources changed")
    configured()
    start = time.monotonic()
    points_path = args.output / "fit.json"
    if args.stage == "fit":
        batches, labels, counts = training_patches(args.fit)
        nets, histories, weights = {}, {}, {}
        for arm in ARMS:
            nets[arm], histories[arm] = train(batches, labels, arm)
            path = args.output / f"{arm}.safetensors"
            save_file({key: value.contiguous() for key, value in nets[arm].state_dict().items()}, path)
            weights[arm] = panel.sha(path)
            print(arm, "fit complete", flush=True)
        calibrated = points(args.calibration, nets)
        panel.write(points_path, {"freeze_sha256": panel.sha(freeze), "patches_per_source": counts,
                    "training_pairs": len(labels), "mean_batch_losses": histories, "weight_sha256": weights,
                    "points": calibrated, "elapsed_s": round(time.monotonic() - start, 3)})
        print("Operating points sealed", flush=True)
        return
    calibrated = panel.read(points_path)
    panel.require(calibrated["freeze_sha256"] == panel.sha(freeze), "Fitted points do not match freeze")
    nets = {}
    for arm in ARMS:
        path = args.output / f"{arm}.safetensors"
        panel.require(panel.sha(path) == calibrated["weight_sha256"][arm], "Fitted weights differ")
        net = PatchHead(3 if arm == ARMS[1] else 2).to(memory_format=torch.channels_last)
        net.load_state_dict(load_file(path), strict=True)
        nets[arm] = net.eval()
    results, bindings, sizes = {}, {}, {}
    for label in ("earlier", "later", "fresh"):
        rows = list(confirm(getattr(args, label), nets, calibrated["points"]))
        expected = sum(frozen["qualified_fresh_by_family"].values()) if label == "fresh" else 12
        panel.require(len(rows) == expected * 16, "Missing confirmation views")
        journal = args.output / f"{label}-rows.json"
        panel.write(journal, rows)
        results[label], bindings[label], sizes[label] = summary(rows), panel.sha(journal), len(rows)
        print(label, len(rows), "measured views", flush=True)
    panel.write(args.output / "result.json", {
        "freeze_sha256": panel.sha(freeze), "fit_sha256": panel.sha(points_path), "rows_sha256": bindings,
        "view_counts": sizes, "results": results, "advances": advancement(results, frozen["qualified_fresh_by_family"]),
        "elapsed_s": round(time.monotonic() - start, 3),
    })


if __name__ == "__main__":
    main()
