# SPDX-License-Identifier: Apache-2.0
"""Test exact JPEG90 components against a compression-free reflection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from PIL import __version__ as pillow_version

from app.detectors.image import regional, regional_support
from scripts import image_regional_jpeg90_carrier as carrier
from scripts import image_regional_jpeg90_quality_confirmation as confirmation_sources
from scripts import image_regional_jpeg_transfer as prior
from scripts import image_regional_tile_rescue_transfer as tile
from scripts.review_image_regional_jpeg90_reflection import (
    DESIGN,
    SCHEMA,
    cases,
    development_maxima,
    summarize,
)

ROOT = Path(__file__).resolve().parents[2]
LEDGERS = {
    "development": ROOT / "docs/evaluation-panels/image-regional-jpeg90-direct-2026-11-13.json",
    "confirmation": ROOT / "docs/evaluation-panels/image-regional-jpeg90-quality-confirmation-2026-11-13.json",
}


def freeze(args):
    paths = (Path(__file__), Path(carrier.__file__), Path(prior.__file__), Path(tile.__file__),
             Path(confirmation_sources.__file__), Path(confirmation_sources.source.__file__),
             Path(regional.__file__), Path(regional_support.__file__), Path(regional_support.loc.__file__),
             Path(__file__).with_name("review_image_regional_jpeg90_reflection.py"))
    prior.require(prior.file_sha(prior.CHECKPOINT) == regional.CHECKPOINT_SHA256, "Pinned regional weights differ")
    return {"schema": SCHEMA, "design": DESIGN, "checkpoint_sha256": regional.CHECKPOINT_SHA256,
            "prior_sha256": {role: prior.file_sha(path) for role, path in LEDGERS.items()},
            "code_sha256": {str(path.relative_to(ROOT)): prior.file_sha(path) for path in paths},
            "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "opencv": cv2.__version__, "pillow": pillow_version}}


def response_quantiles(scores, component, reliable_outside):
    active = component.astype(np.uint8)
    outside = ((cv2.dilate(active, np.ones((17, 17), np.uint8)) > 0)
               & (cv2.dilate(active, np.ones((5, 5), np.uint8)) == 0) & reliable_outside)
    ring_count = int(outside.sum())
    if ring_count < 32 or 4 * ring_count < int(component.sum()):
        return None
    inside = np.quantile(scores[component], .50, method="nearest")
    low, center, high = np.quantile(scores[outside], [.10, .50, .90], method="nearest")
    return [int(inside), int(low), int(center), int(high), ring_count]


def assess(model, index, group, body, rgb, truth, expected_mask, native_count, expected_truth):
    truth_digest = prior.sha(truth.tobytes())
    prior.require(truth_digest == expected_truth, "Native truth differs from the frozen exact edit mask")
    row = {"index": index, "group": group, "input_sha256": prior.sha(body), "truth_sha256": truth_digest,
           "native_mask_sha256": expected_mask, "reflection": None, "components": []}
    if not native_count:
        return row
    height, width = truth.shape
    source = np.asarray(rgb)
    native = prior.native_readout(model.predict(rgb), body, source)
    mask = native.source_mask(width, height)
    prior.require(mask.shape == native.selected_grid_mask.shape and prior.sha(mask.tobytes()) == expected_mask,
                  "Original JPEG90 mask failed exact replay")
    reversed_rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    prediction = model.predict(reversed_rgb)
    aligned_raw = np.ascontiguousarray(prediction.raw_mask[:, ::-1])
    hard = regional_support.readout(aligned_raw, prediction.candidate_eligibility["eligible"],
                                    regional.PIXEL_THRESHOLD, body, source)
    score = np.rint(hard.floating.astype(np.float64) * 4095).astype(np.uint16)
    outside = native.reliable & ~native.selected_grid_mask
    row["reflection"] = {"aligned_float32_sha256": prior.sha(aligned_raw.tobytes()),
                         "reflected_rgb_sha256": prediction.geometry["source_rgb_sha256"],
                         "reflected_input_sha256": prediction.geometry["input_float32_sha256"],
                         "hard_mask_sha256": prior.sha(hard.selected_grid_mask.tobytes())}
    for component in native.component_masks:
        y, x = np.where(component)
        row["components"].append({"area": int(x.size), "bounds": [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)],
                                  "mask_sha256": prior.sha(component.tobytes()), "intersection": int(np.count_nonzero(component & truth)),
                                  "hard_overlap": int(np.count_nonzero(component & hard.selected_grid_mask)),
                                  "contrast": response_quantiles(score, component, outside)})
    return row


def inputs(args, development, confirmation):
    if args.phase in {"development", "replay"}:
        aliases = {}
        for index, (delivery, saved) in enumerate(zip(carrier.deliveries(args), development["rows"], strict=True)):
            key, jpeg75, truth, pixels = delivery
            prior.require(all(saved[field] == value for field, value in tile.witness(key, jpeg75, truth).items()),
                          "Development delivery or exact truth differs")
            body, image = prior.derived_view(Image.fromarray(pixels))
            prior.require(prior.sha(body) == saved["q90_sha256"], "Development JPEG90 bytes differ")
            yield (index, aliases.setdefault(key["group"], len(aliases) + 1), body, image, truth, saved["q90_mask_sha256"],
                   saved["q90"]["predicted_positive_pixels"], saved["truth_sha256"])
    if args.phase in {"confirmation", "replay"}:
        for index, (delivery, planned, saved) in enumerate(zip(confirmation_sources.cases(args), confirmation["freeze"]["rows"],
                                                               confirmation["rows"], strict=True), start=200):
            key, _, _, body, image, truth = delivery
            prior.require(key == planned, "Confirmation delivery or exact truth differs")
            yield (index, key["group"], body, image, truth, saved["mask_sha256"]["direct_q90"],
                   saved["measures"]["direct_q90"]["predicted_positive_pixels"], key["truth_sha256"])


def calibration(rows, development):
    cutoff, maxima = development_maxima(rows, development)
    return {"cutoff": [cutoff.numerator, cutoff.denominator],
            "ordinary_group_maxima": {str(k): [v.numerator, v.denominator] for k, v in maxima.items()}}


def smoke():
    component = np.zeros((32, 32), bool)
    component[10:22, 10:22] = True
    reliable = np.ones_like(component)
    values = np.full(component.shape, 410, np.uint16)
    values[component] = 820
    values[8:24, 8:24] += 1000
    result = response_quantiles(values, component, reliable)
    prior.require(result == [1820, 410, 410, 410, 528], "Response quantile geometry differs")
    prior.require(response_quantiles(values, component, ~reliable) is None, "Masked ring produced contrast")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "confirmation", "replay"))
    for name in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint", "output-fresh", "external",
                 "tile-run", "main-roster", "external-roster", "confirmation-roster", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    args.direct = True
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    smoke()
    locked = freeze(args)
    lock = args.output / "freeze.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        prior.write(lock, locked)
        return
    prior.require(locked == prior.read(lock), "Frozen design or source commitments differ")
    development, confirmation = (prior.read(path) for path in LEDGERS.values())
    points = args.output / "calibration.json"
    development_rows = args.output / "development.json"
    if args.phase == "confirmation":
        dev_rows = prior.read(development_rows)
        cal = calibration(dev_rows, development)
        prior.require(cal == prior.read(points), "Frozen development cutoff differs")
    model = regional.load_model(prior.CHECKPOINT)
    rows = []
    for args_in in inputs(args, development, confirmation):
        rows.append(assess(model, *args_in))
        if len(rows) % 25 == 0:
            print(f"{args.phase} measured {len(rows)}", flush=True)
    if args.phase == "development":
        prior.write(development_rows, rows)
        prior.write(points, calibration(rows, development))
        return
    if args.phase == "confirmation":
        rows = dev_rows + rows
    result = {"freeze": locked, "rows": rows}
    joined = list(cases(result, development, confirmation))
    cutoff, _ = development_maxima(rows[:200], development)
    cal = calibration(rows[:200], development)
    prior.require(cal == prior.read(points), "Frozen development cutoff differs")
    result.update(calibration=cal, summary=summarize(joined, cutoff))
    receipt = args.output / "receipt.json"
    if args.phase == "confirmation":
        prior.write(receipt, result)
    else:
        prior.require(result == prior.read(receipt), "Frozen native reflection replay differs")
    print(json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
