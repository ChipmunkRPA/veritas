# SPDX-License-Identifier: Apache-2.0
"""Compare matched, multi-depth patch directions on localized generative edits."""

from __future__ import annotations

import argparse
import time
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from scripts import image_edge_spread_transfer as panel
from scripts import image_relative_patch_transfer as spatial
from scripts import image_residual_color_transfer as incumbent
from scripts import image_semantic_context_transfer as encoder
from scripts import image_wavelet_region_transfer as wavelet

SCHEMA = "chipmunk.image-multidepth-transfer/1"
ARMS = ("last_layer", "three_depths")
LAYERS = (3, 6, 12)
WIDTH = 384
RIDGE = .5
POINT_GROUPS = 12
BASELINE = "format_reference"


def calibration_sources(folder, other_plans):
    plan = panel.read(folder / "plan.json")
    receipt = panel.read(folder / "acquisition.json")
    panel.require(panel.sha(folder / "plan.json") == receipt["roster_sha256"], "Calibration intake roster differs")
    names = {str(item["flickr_owner"]).strip().casefold() for group in other_plans for item in group}
    candidates = []
    for source, asset in zip(plan["sources"], receipt["attempts"], strict=True):
        panel.require(source["slot"] == asset["slot"], "Calibration asset slot mismatch")
        if asset["status"] == "qualified" and source["author"].strip().casefold() not in names:
            panel.require(source["licence"] in ("CC BY 3.0", "CC BY 4.0", "CC BY-SA 3.0", "CC BY-SA 4.0"),
                          "Unsupported source terms")
            candidates.append(asset)
    candidates.sort(key=lambda row: (row["sha256"], row["slot"]))
    selected = candidates[:POINT_GROUPS]
    panel.require(len(selected) == POINT_GROUPS and len({row["sha256"] for row in selected}) == POINT_GROUPS,
                  "The selection requires twelve distinct ordinary sources")
    for asset in selected:
        panel.require(panel.sha(folder / asset["name"]) == asset["sha256"], "Calibration JPEG differs")
    return selected


def profile(args):
    directories = (args.fit, args.earlier, args.later)
    plans = [panel.read(folder / "plan.json")["cases"] for folder in directories]
    for field in ("flickr_photo_id", "flickr_owner"):
        ids = [{item[field] for item in group} for group in plans]
        panel.require([len(group) for group in ids] == [8, 12, 12] and len(set.union(*ids)) == 32,
                      "Expected distinct source and credited account groups")
    weights = {file: panel.sha(args.checkpoint / file) for file in ("config.json", "model.safetensors")}
    panel.require(weights == {"config.json": encoder.CONFIG_SHA256, "model.safetensors": encoder.SEMANTIC_SHA256},
                  "Pinned Apache-2.0 backbone files differ")
    journals = {"earlier": panel.sha(args.earlier_rows), "later": panel.sha(args.later / "outcomes.json")}
    panel.require(journals == panel.REFERENCE, "Exact existing encoded confirmation journals required")
    return {
        "schema": SCHEMA, "weights": weights, "journals": journals,
        "implementation": {Path(module.__file__).name: panel.sha(Path(module.__file__))
                           for module in (panel, spatial, incumbent, wavelet)}
                          | {Path(__file__).name: panel.sha(Path(__file__))},
        "intakes": {name: {key: panel.sha(folder / key) for key in ("plan.json", "attempts.jsonl", "result.json")}
                    for name, folder in zip(("fit", "earlier", "later"), directories, strict=True)},
        "calibration_roster": {key: panel.sha(args.calibrate / key) for key in ("plan.json", "acquisition.json")},
        "calibration_sources": [{key: row[key] for key in ("slot", "name", "sha256")}
                                for row in calibration_sources(args.calibrate, plans)],
        "protocol": {"layers": list(LAYERS), "arms": list(ARMS), "grid": spatial.GRID, "ridge": RIDGE,
                     "input": "bicubic square at 336 pixels, ImageNet normalization, deterministic single-thread CPU",
                     "heads": "fit within-source contrast of both positives against five ordinary views; "
                              "weight eight source means equally; compare last-layer-only to concatenated early, middle and final patches",
                     "selection": "strictly above second-highest of twelve per-source control maxima",
                     "test": "compare added useful feathered regions and additional ordinary flags versus regional format incumbent",
                     "advance_per_format_and_panel": {"additional_feathered_min": 2, "additional_controls_max": 4,
                                                      "additional_control_sources_max": 3},
                     "confidence_weight": 0},
        "versions": {name: version(name) for name in ("torch", "transformers", "numpy", "scipy", "pillow", "safetensors")},
    }


def reader(checkpoint):
    import torch
    from torch.nn import functional as fn
    from transformers import Dinov2Model

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = Dinov2Model.from_pretrained(checkpoint, local_files_only=True, use_safetensors=True).eval()
    mean, std = (np.asarray(row, np.float32) for row in ((.485, .456, .406), (.229, .224, .225)))
    side = 14 * spatial.GRID

    @torch.inference_mode()
    def read(image):
        pixels = np.asarray(image.resize((side, side), Image.Resampling.BICUBIC), dtype=np.float32) / 255
        values = torch.from_numpy((pixels - mean) / std).permute(2, 0, 1).unsqueeze(0).contiguous()
        output = model(pixel_values=values, output_hidden_states=True)
        states = [output.hidden_states[index][0, 1:] for index in LAYERS[:-1]] + [output.last_hidden_state[0, 1:]]
        features = torch.cat([fn.normalize(fn.layer_norm(state, (WIDTH,)), dim=-1) for state in states], dim=-1)
        panel.require(tuple(features.shape) == (spatial.GRID ** 2, len(LAYERS) * WIDTH)
                      and torch.isfinite(features).all().item(), "Invalid fixed-grid embeddings")
        return features.reshape(spatial.GRID, spatial.GRID, -1).numpy()

    return read


def fit(folder, read):
    training = {fmt: [] for fmt in panel.FORMATS}
    _, plan, receipt = panel.scoring.checked_inputs(folder)
    for _, _, images, truth in panel.scoring.panels(folder, plan, receipt):
        y, x = np.nonzero(truth)
        bounds = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
        original = images["original"]
        feathered = panel.blend.local(original, images["edited"], bounds, feather=True)
        variants = (images["edited"], feathered, *wavelet.view_controls(original, images["processed"], bounds)[:5])
        for fmt in panel.FORMATS:
            features = [read(panel.paired.encoded(image, fmt)[1]) for image in variants]
            contrasts = [spatial.relative_change(control, positive, np.any(variant != original, axis=2))
                         for positive, variant in zip(features[:2], variants[:2], strict=True) for control in features[2:]]
            training[fmt].append(np.mean(contrasts, axis=0))
    panel.require(all(len(group) == 8 for group in training.values()), "Require eight fitted source groups")
    weights = {}
    for fmt, group in training.items():
        directions = np.stack(group)
        weights[fmt] = {}
        for arm, values in zip(ARMS, (directions[:, -WIDTH:], directions), strict=True):
            scaled = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
            direction = scaled.T @ np.linalg.solve(scaled @ scaled.T + RIDGE * np.eye(len(group)), np.ones(len(group)))
            weights[fmt][arm] = (direction / max(float(np.linalg.norm(direction)), 1e-8)).tolist()
    return weights


def proposals(features, weights):
    return {arm: {"score": score, "region": list(region)}
            for arm, grid in zip(ARMS, (features[..., -WIDTH:], features), strict=True)
            for score, region in (spatial.strongest_region(grid @ weights[arm]),)}


def controls(folder, roster, read, weights):
    maxima = {fmt: {arm: [] for arm in ARMS} for fmt in panel.FORMATS}
    for asset in roster:
        with Image.open(folder / asset["name"]) as image:
            original = image.convert("RGB")
            original.thumbnail((640, 640), Image.Resampling.LANCZOS)
        w, h = original.size
        source = np.asarray(original)
        bounds = (w // 3, h // 3, 7 * w // 12, 7 * h // 12)
        blur = np.asarray(original.filter(ImageFilter.GaussianBlur(1.25)))
        tone = np.asarray(ImageEnhance.Contrast(original).enhance(1.2))
        sharp = np.asarray(ImageEnhance.Sharpness(original).enhance(1.5))
        resize = np.asarray(original.resize((w * 3 // 4, h * 3 // 4), Image.Resampling.BICUBIC)
                            .resize((w, h), Image.Resampling.BICUBIC))
        ordinary = (source, resize, *(panel.blend.local(source, item, bounds, feather=True) for item in (blur, tone, sharp)))
        for fmt in panel.FORMATS:
            measured = [proposals(read(panel.paired.encoded(image, fmt)[1]), weights[fmt]) for image in ordinary]
            for arm in ARMS:
                maxima[fmt][arm].append(max(row[arm]["score"] for row in measured))
    return {fmt: {arm: {"point": sorted(values)[-2], "source_maxima": values} for arm, values in arms.items()}
            for fmt, arms in maxima.items()}


def confirm(folder, reference, read, weights, points):
    frozen = {panel._key(row): row for row in panel.read(reference)}
    panel.require(len(frozen) == 192, "Require a complete 192-view reference")
    for source, images, known in panel.paired.inputs(folder):
        for role, pixels in panel.native_variants(images, source["box"]).items():
            truth = np.any(pixels != images["original"], axis=2) if role in panel.POSITIVE else np.zeros_like(known)
            panel.require(not np.any(truth & ~known), "Edit extends outside the qualified source mask")
            y, x = (np.arange(n) * spatial.GRID // n for n in truth.shape)
            truth_sha = panel.paired.shared.digest(truth.tobytes())
            for fmt in panel.FORMATS:
                body, image = panel.paired.encoded(pixels, fmt)
                key = (source["slot"], source["generator"], role, fmt)
                prior = frozen.pop(key)
                body_sha = panel.paired.shared.digest(body)
                panel.require(prior["input_sha256"] == body_sha and prior.get("truth_sha256", truth_sha) == truth_sha
                              and next(iter(prior["measurements"].values()))["target"] == int(truth.sum()),
                              "Frozen encoding or truth differs")
                result = proposals(read(image), weights[fmt])
                measurements = {}
                for arm, item in result.items():
                    top, left, side = item["region"]
                    region = ((top <= y) & (y < top + side))[:, None] & ((left <= x) & (x < left + side))[None, :]
                    measurements[arm] = panel.scoring.measure(region & (item["score"] > points[fmt][arm]["point"]), truth)
                    item["ungated_useful"] = panel.useful(panel.scoring.measure(region, truth))
                baseline = incumbent.format_reference(fmt, body, np.asarray(image))
                mask = (panel.paired.boxed(baseline["box"], truth.shape, True) if baseline and baseline["selected"]
                        else np.zeros_like(truth))
                measurements[BASELINE] = panel.scoring.measure(mask, truth)
                yield dict(zip(("slot", "generator", "role", "format"), key, strict=True)) | {
                    "input_sha256": body_sha, "truth_sha256": truth_sha,
                    "proposals": result | {BASELINE: baseline}, "measurements": measurements,
                }
    panel.require(not frozen, "Incomplete confirmation roster")


def summary(rows):
    panel.require(len(rows) == 192, "Incomplete scored panel")
    results = {}
    for fmt in panel.FORMATS:
        positive = [row for row in rows if row["format"] == fmt and row["role"] in panel.POSITIVE]
        ordinary = [row for row in rows if row["format"] == fmt and row["role"] in panel.CONTROLS]
        results[fmt] = {}
        for arm in (*ARMS, BASELINE):
            hits = [row for row in positive if panel.useful(row["measurements"][arm])]
            flags = [row for row in ordinary if row["measurements"][arm]["reported"]]
            extra = [row for row in flags if not row["measurements"][BASELINE]["reported"]]
            results[fmt][arm] = {
                "sharp": sum(row["role"] == "edited" for row in hits),
                "feathered": sum(row["role"] == "feathered_edit" for row in hits),
                "ordinary": len(flags), "original": sum(row["role"] == "original" for row in flags),
                "extra_ordinary": len(extra), "extra_ordinary_sources": len({row["slot"] for row in extra}),
                "extra_feathered": sum(row["role"] == "feathered_edit" and not panel.useful(row["measurements"][BASELINE])
                                       for row in hits),
                "ungated_useful": (sum(row["proposals"][arm]["ungated_useful"] for row in positive)
                                   if arm in ARMS else None),
            }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "confirm"))
    for name in ("checkpoint", "fit", "calibrate", "earlier", "later", "earlier_rows", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    plan = profile(args)
    freeze_path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        panel.write(freeze_path, plan)
        return
    panel.require(plan == panel.read(freeze_path), "Frozen source or protocol changed")
    cv2.setNumThreads(1)
    read = reader(args.checkpoint)
    point_path = args.output / "points.json"
    if args.stage == "fit":
        fitted = fit(args.fit, read)
        weights = {fmt: {arm: np.asarray(value) for arm, value in rows.items()} for fmt, rows in fitted.items()}
        points = controls(args.calibrate, plan["calibration_sources"], read, weights)
        panel.write(point_path, {"freeze_sha256": panel.sha(freeze_path), "weights": fitted, "points": points})
        print("Eight paired fit sources and twelve independent control sources complete", flush=True)
        return
    fitted = panel.read(point_path)
    panel.require(fitted["freeze_sha256"] == panel.sha(freeze_path), "Operating points differ from freeze")
    weights = {fmt: {arm: np.asarray(value) for arm, value in rows.items()} for fmt, rows in fitted["weights"].items()}
    results, journals = {}, {}
    started = time.monotonic()
    for label, folder, prior in (("earlier", args.earlier, args.earlier_rows),
                                  ("later", args.later, args.later / "outcomes.json")):
        rows = list(confirm(folder, prior, read, weights, fitted["points"]))
        journal = args.output / f"{label}-rows.json"
        panel.write(journal, rows)
        results[label], journals[label] = summary(rows), panel.sha(journal)
        print(label, len(rows), "verified views", flush=True)
    criteria = plan["protocol"]["advance_per_format_and_panel"]
    advanced = {fmt: all((row := scores[fmt]["three_depths"])["extra_feathered"] >= criteria["additional_feathered_min"]
                         and row["extra_ordinary"] <= criteria["additional_controls_max"]
                         and row["extra_ordinary_sources"] <= criteria["additional_control_sources_max"]
                         for scores in results.values()) for fmt in panel.FORMATS}
    panel.write(args.output / "result.json", {"freeze_sha256": panel.sha(freeze_path), "points_sha256": panel.sha(point_path),
                "journals": journals, "results": results, "three_depths_advances": advanced,
                "elapsed_s": round(time.monotonic() - started, 3)})


if __name__ == "__main__":
    main()
