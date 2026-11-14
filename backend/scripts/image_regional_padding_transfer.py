# SPDX-License-Identifier: Apache-2.0
"""Test whether out-of-image padding changes small-edit localization on the same submitted pixels."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import image_small_region_repeat as prior

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "docs/evaluation-panels/image-small-region-repeat-2026-11-14.json"
SCHEMA = "chipmunk.image-regional-padding-transfer/1"
ARMS = ("serving", "edge", "reflect", "both")
METHOD = {
    "fixed_source": "All decoded source pixels, source location, geometry graph, native response threshold, source area "
                    "rule and reliable-grid component rules match the existing JPEG75 comparison.",
    "arms": "The original fills the rest of its 1024 canvas with black. Edge extends its last rows and columns. Reflect "
            "mirrors its border without repeating the boundary pixel. Both accepts an edge component only if at least "
            "two thirds of each component overlap in the edge and reflect views, and preserves existing flags.",
    "useful": "At least 0.30 IoU and 0.50 exact pre-encode changed-pixel recall.",
    "transfer": "A proposal needs at least three added useful groups from two editors, with no lost useful location "
                "and at most two newly flagged ordinary groups. The source panel had prior exposure.",
}


def freeze(args):
    parent = prior.read(PARENT)
    plan = prior.freeze(args)
    prior.require(parent["freeze"] == plan, "Parent receipt and its exact source roles differ")
    model_files = ("regional.py", "regional_support.py", "regional_graph/RelayFormer.py")
    return {"schema": SCHEMA, "method": METHOD, "parent_sha256": prior.sha(PARENT.read_bytes()),
            "checkpoint_sha256": prior.CHECKPOINT, "cases": plan["cases"],
            "bindings": {name: prior.sha((ROOT / "backend/app/detectors/image" / name).read_bytes())
                         for name in model_files} | {"runner": prior.sha(Path(__file__).read_bytes())}}


def regional_padding(model, rgb, tensor, geometry, mode):
    import numpy as np
    import torch

    from app.detectors.image import regional
    from app.evidence.image_eligibility import area_gate

    height, width = rgb.shape[:2]
    prior.require(geometry["fit_size"] == [width, height] and max(height, width) == 768,
                  "This experiment requires unchanged, unscaled source pixels")
    padded = np.pad(rgb, ((0, 1024 - height), (0, 1024 - width), (0, 0)), mode=mode)
    mean = np.asarray([0.485, 0.456, 0.406], np.float32)
    deviation = np.asarray([0.229, 0.224, 0.225], np.float32)
    values = (padded.astype(np.float32) / np.float32(255) - mean) / deviation
    changed = torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)[None]))
    prior.require(torch.equal(changed[:, :, :height, :width], tensor[:, :, :height, :width]),
                  "Experimental padding changed decoded source pixels")
    grid_h, grid_w = (2 * height + 1023) // 1024, (2 * width + 1023) // 1024
    expected = torch.stack([
        changed[0, :, row * 496:row * 496 + 528, col * 496:col * 496 + 528]
        for row in range(grid_h) for col in range(grid_w)
    ])
    calls = []

    def record(_module, args):
        prior.require(torch.equal(args[0], expected), "Actual model patches differ from padding definition")
        calls.append(regional.tensor_sha(args[0]))

    hook = model.model.vit.patch_embed.register_forward_pre_hook(record)
    try:
        with torch.inference_mode():
            output = model.model(image=changed, origin_shape=torch.tensor([[height, width]]), clip_len=torch.tensor([1]))
    finally:
        hook.remove()
    prior.require(len(calls) == 1, "Patch binding did not complete")
    raw = regional.source_map(output, geometry)
    count = int(np.count_nonzero(raw > regional.PIXEL_THRESHOLD))
    return raw, area_gate(count, int(np.ceil(regional.MINIMUM_AREA_FRACTION * raw.size)), raw.size), calls[0]


def support_pair(a, b):
    size, other, shared = int(a.sum()), int(b.sum()), int((a & b).sum())
    return [size, other, shared]


def score_case(model, entry, official):
    import io

    import numpy as np
    from PIL import Image

    from app.detectors.image import regional, regional_support

    body, truth = entry[-2:]
    with Image.open(io.BytesIO(body), formats=["JPEG"]) as received:
        rgb = np.asarray(received.convert("RGB"))
    height, width = truth.shape
    image = Image.fromarray(rgb)
    original = model.predict(image)
    tensor, geometry = regional.prepare(image)
    prior.require(geometry["input_float32_sha256"] == original.geometry["input_float32_sha256"]
                  == official["native_input_sha256"], "The serving image input changed")
    native = regional_support.readout(original.raw_mask, original.candidate_eligibility["eligible"],
                                      regional.PIXEL_THRESHOLD, body, rgb)

    def pixels(mask):
        converted = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)) > 0
        return [int(converted.sum()), int((converted & truth).sum())]

    old = pixels(native.selected_grid_mask)
    prior.require(old == official["serving"], "Serving result differs from the prior receipt")
    details, new = {}, {}
    for name in ("edge", "reflect"):
        raw, gate, patch_hash = regional_padding(model, rgb, tensor, geometry, name)
        readout = regional_support.readout(raw, True, regional.PIXEL_THRESHOLD, body, rgb)
        new[name] = readout.component_masks if gate["eligible"] else ()
        difference = np.abs(raw - original.raw_mask)
        details[name] = {"patches_sha256": patch_hash, "eligible": gate["eligible"],
                         "mean_delta": round(float(difference.mean()), 7),
                         "near_strip_mean_delta": round(float(np.mean(np.concatenate(
                             (difference[-64:, :].ravel(), difference[:-64, -64:].ravel())))), 7),
                         "center_mean_delta": round(float(difference[128:-128, 128:-128].mean()), 7),
                         "ungated": pixels(readout.selected_grid_mask),
                         "selected": pixels(readout.selected_grid_mask) if gate["eligible"] else [0, 0]}
    pairs = [support_pair(a, b) + [i, j] for i, a in enumerate(new["edge"]) for j, b in enumerate(new["reflect"])]
    selected = native.selected_grid_mask
    if old[0] == 0:
        selected = next((new["edge"][i] for a, b, shared, i, _ in pairs if 3 * shared >= 2 * max(a, b)), selected)
    return {"jpeg_sha256": prior.sha(body), "serving": old, **details, "pairs": pairs, "both": pixels(selected)}


def review(doc):
    lock, rows = doc["freeze"], doc["rows"]
    prior.require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["checkpoint_sha256"] == prior.CHECKPOINT
                  and len(lock["cases"]) == len(rows) == 186, "Missing or changed review metadata")
    stats = {arm: defaultdict(Counter) for arm in ARMS}
    regained = {arm: set() for arm in ARMS}
    lost = {arm: set() for arm in ARMS}
    ordinary = {arm: set() for arm in ARMS}
    for case, row in zip(lock["cases"], rows, strict=True):
        target, family, group = (case[k] for k in ("changed", "family", "group"))
        prior.require(row["jpeg_sha256"] == case["jpeg_sha256"] and bool(target) == (case["role"] in prior.POSITIVE),
                      "Measured image differs from exact source or edit label")
        matched = any(3 * hit >= 2 * max(a, b) for a, b, hit, _, _ in row["pairs"])
        prior.require(row["both"] == row["serving"] if row["serving"][0] else bool(row["both"][0]) == matched,
                      "Combined result differs from agreement and preservation rule")
        for arm in ARMS:
            detail = row[arm]
            if isinstance(detail, dict):
                prior.require(len(bytes.fromhex(detail["patches_sha256"])) == 32
                              and all(0 <= detail[k] <= 1 for k in ("mean_delta", "near_strip_mean_delta", "center_mean_delta"))
                              and detail["selected"] == (detail["ungated"] if detail["eligible"] else [0, 0]),
                              "Experimental padding measurement is invalid")
                stats[arm][family]["ungated_useful"] += prior.useful(detail["ungated"], target)
                detail = detail["selected"]
            area, hit = detail
            prior.require(0 <= hit <= min(area, target) and 0 <= area <= case["pixels"], "Out-of-range native pixel counts")
            good, old = prior.useful(detail, target), prior.useful(row["serving"], target)
            stats[arm][family].update(edited=bool(target), ordinary=not target, useful=good,
                                      misplaced=bool(target and area and not good), ordinary_flags=bool(not target and area))
            if good and not old:
                regained[arm].add((family, group))
            if old and not good:
                lost[arm].add((family, group))
            if not target and area and not row["serving"][0]:
                ordinary[arm].add((family, group))
    return {arm: {"counts": dict(stats[arm]), "added": sorted(map(list, regained[arm])),
                  "lost": sorted(map(list, lost[arm])), "new_ordinary_groups": sorted(map(list, ordinary[arm])),
                  "passes": (len(regained[arm]) >= 3 and len({family for family, _ in regained[arm]}) >= 2 and not lost[arm]
                             and len(ordinary[arm]) <= 2)}
            for arm in ARMS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--sources", type=Path, default=ROOT / "backend/data/localization-eval")
    parser.add_argument("--bg", type=Path, nargs=2)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/weights/native-regional-v1/regional-model.safetensors")
    args = parser.parse_args()
    if args.phase == "review":
        receipt = prior.read(args.run)
        result = review(receipt)
        prior.require(receipt["outcome"] == result, "Saved outcomes differ")
    elif args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        prior.save(args.run / "freeze.json", freeze(args))
        return
    else:
        import cv2
        import torch

        from app.detectors.image import regional

        lock = prior.read(args.run / "freeze.json")
        prior.require(lock == freeze(args), "Native sources or implementation changed after freeze")
        cv2.setNumThreads(1)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        model = regional.load_model(args.checkpoint)
        rows = []
        for entry, case, official in zip(prior.deliveries(args), lock["cases"], prior.read(PARENT)["rows"], strict=True):
            prior.require(prior.identity(*entry) == case and official["jpeg_sha256"] == case["jpeg_sha256"],
                          "Correlated variant does not match the frozen source")
            rows.append(score_case(model, entry, official))
            if len(rows) % 12 == 0:
                print(f"Recomputed {len(rows)} of {len(lock['cases'])} images", flush=True)
        receipt = {"freeze": lock, "rows": rows}
        result = review(receipt)
        prior.save(args.run / "receipt.json", receipt | {"outcome": result})
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
