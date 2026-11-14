"""Compare exact horizontal and vertical repeats of small source-image edits."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import image_regional_flip_confirmation as previous

SCHEMA = "chipmunk.image-regional-two-reflections/1"
ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "docs/evaluation-panels/image-regional-flip-confirmation-2026-11-13.json"
METHOD = {
    "sources": "Replay all 108 exact PNGs, 28 photos and four editor families in the earlier horizontal-repeat study.",
    "baseline": "Replay the app's original canvas and horizontal reflection. Preserve an original component if at least half of its grid cells recur.",
    "vertical_corroboration": "Retain any original component supported at the existing half-component point in either reflection.",
    "reflected_proposal": "Keep the existing result. When neither original component survives horizontal support, look for a pair of horizontally and vertically reflected components with at least half of each component in agreement. Propose only their intersection, with the original image reliability mask.",
    "useful": "At least half the known changed pixels and at least 0.30 intersection over union on the native image.",
    "advance": "Recover three new useful locations across at least two editor families with at most two additional ordinary photos and at most two extra misplaced edited-file highlights.",
}
ARMS = ("baseline", "either_repeat", "two_reflection_proposal")


def frozen(args):
    parent = json.loads(PARENT.read_bytes())
    bound = parent["freeze"]
    previous.require(previous.file_sha(args.checkpoint) == bound["checkpoint_sha256"]
                     and [previous.file_sha(p / "result.json") for p in args.bg] == bound["bg_intake_sha256"],
                     "The earlier source banks or checkpoint differ")
    return parent, {"schema": SCHEMA, "method": METHOD, "parent_sha256": previous.file_sha(PARENT),
                    "runner_sha256": previous.file_sha(Path(__file__)), "checkpoint_sha256": bound["checkpoint_sha256"]}


def measure(args):
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from app.detectors.image import regional, regional_support

    saved = args.path / "freeze.json"
    parent, lock = frozen(args)
    previous.require(json.loads(saved.read_bytes()) == lock, "The method differs from the freeze")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = regional.load_model(args.checkpoint)
    rows = []
    for (plan, png, rgb, truth), expected in zip(previous.materialize(args), parent["rows"], strict=True):
        previous.require(plan == expected["input"], "Rebuilt source or ground truth differs")

        def read(pixels, data=png):
            pixels = np.ascontiguousarray(pixels)
            result = model.predict(Image.fromarray(pixels))
            return regional_support.readout(result.raw_mask, result.candidate_eligibility["eligible"],
                                            regional.PIXEL_THRESHOLD, data, pixels)

        def union(components, reference):
            return np.logical_or.reduce(components, axis=0) if components else np.zeros_like(reference)

        def record(mask, truth=truth):
            h, w = truth.shape
            native = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)) > 0
            return [int(native.sum()), int((native & truth).sum())]

        original = read(rgb)
        horizontal = read(rgb[:, ::-1])
        vertical = read(rgb[::-1])
        hmask = horizontal.selected_grid_mask[:, ::-1]
        vmask = vertical.selected_grid_mask[::-1]
        initial = original.component_masks
        overlap = [[int(item.sum()), int((item & hmask).sum()), int((item & vmask).sum())] for item in initial]
        retained = [item for item, (size, across, _) in zip(initial, overlap, strict=True) if 2 * across >= size]
        baseline = union(retained, hmask)
        previous.require(record(baseline) == expected["repeat"]
                         and record(original.selected_grid_mask) == expected["native"], "Incumbent masks changed")
        either = union([item for item, (size, across, down) in zip(initial, overlap, strict=True)
                        if 2 * max(across, down) >= size], hmask)
        confirmed = baseline
        proposals = []
        if not retained:
            for candidate in horizontal.component_masks:
                left = candidate[:, ::-1] & original.reliable
                for candidate_vertical in vertical.component_masks:
                    upper = candidate_vertical[::-1] & original.reliable
                    shared = left & upper
                    count = int(shared.sum())
                    if count and count * 2 >= max(int(left.sum()), int(upper.sum())):
                        proposals.append((count, shared))
            if proposals:
                confirmed = max(proposals, key=lambda item: item[0])[1]
        row = {"input": plan, "original_overlap_grid_cells": overlap,
               "native_pixels": dict(zip(ARMS, map(record, (baseline, either, confirmed)), strict=True))}
        rows.append(row)
        if len(rows) % 12 == 0:
            print(f"Measured {len(rows)} of {len(parent['rows'])} deliveries", flush=True)
    receipt = {"freeze": lock, "rows": rows}
    previous.write(args.path / "receipt.json", receipt | {"result": summarize(receipt, parent)})


def summarize(receipt, parent):
    lock = receipt["freeze"]
    previous.require(lock["schema"] == SCHEMA and lock["method"] == METHOD
                     and lock["runner_sha256"] == previous.file_sha(Path(__file__))
                     and lock["parent_sha256"] == previous.file_sha(PARENT)
                     and len(receipt["rows"]) == len(parent["rows"]) == 108, "The frozen evaluation differs")
    counts = defaultdict(Counter)
    gained = defaultdict(set)
    false_groups = defaultdict(set)
    for row, previous_row in zip(receipt["rows"], parent["rows"], strict=True):
        plan = row["input"]
        scores = row["native_pixels"]
        previous.require(plan == previous_row["input"] and scores["baseline"] == previous_row["repeat"],
                         "A retained comparison changed")
        truth = plan["edited_pixels"]
        base = previous.useful(scores["baseline"], truth)
        for arm in ARMS:
            area, overlap = scores[arm]
            previous.require(0 <= overlap <= min(area, truth) and 0 <= area <= plan["shape"][0] * plan["shape"][1],
                             "Invalid source-pixel geometry")
            count = counts[arm]
            use = previous.useful((area, overlap), truth)
            count.update(edited=bool(truth), ordinary=not truth, useful=use, wrong=bool(truth and area and not use),
                         flagged=bool(area and not truth), gained=use and not base, lost=base and not use,
                         added_wrong=bool(truth and area and not use and (not scores["baseline"][0] or base)))
            if use and not base:
                gained[arm].add(plan["family"])
            if not truth and area and not scores["baseline"][0]:
                false_groups[arm].add(plan["group"])
    return {arm: {"counts": counts[arm], "gained_families": sorted(gained[arm]), "extra_ordinary_groups": len(false_groups[arm]),
                  "passes": counts[arm]["gained"] >= 3 and len(gained[arm]) >= 2 and counts[arm]["lost"] == 0
                  and len(false_groups[arm]) <= 2 and counts[arm]["added_wrong"] <= 2} for arm in ARMS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--sources", type=Path, default=ROOT / "backend/data/localization-eval")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/weights/native-regional-v1/regional-model.safetensors")
    parser.add_argument("--bg", type=Path, nargs=2)
    args = parser.parse_args()
    if args.phase == "freeze":
        _, lock = frozen(args)
        args.path.mkdir(mode=0o700)
        previous.write(args.path / "freeze.json", lock)
    elif args.phase == "measure":
        measure(args)
    else:
        receipt = json.loads(args.path.read_bytes())
        result = summarize(receipt, json.loads(PARENT.read_bytes()))
        previous.require(result == receipt["result"], "The published counts do not replay")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
