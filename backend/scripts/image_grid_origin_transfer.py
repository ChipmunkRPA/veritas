"""Measure exact source-region support when the regional model grid moves."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.detectors.image import regional, regional_support, splice
from app.evidence.image_eligibility import area_gate
from scripts import image_active_png_transfer as active
from scripts import image_edge_spread_transfer as sources
from scripts import image_flux_product_comparison_v288 as flux
from scripts.image_regional_flip_transfer import combine
from scripts.review_image_regional_flip_transfer import restore_masks

SCHEMA = "chipmunk.image-grid-origin-transfer/1"
OFFSETS = ((0, 32), (32, 0))
COHORTS = ("earlier", "later", "flux")
ORIGINAL = ("native", "flip", "recompression")
SHIFTED = ("horizontal_offset", "vertical_offset")
ARMS = (*ORIGINAL, *SHIFTED, "offset_intersection", "original_supported_by_both", "original_supported_by_either")
POSITIVES = {"edited", "feathered_edit", "ai"}
ADVANCE = {"original_supported_by_both": "Keep every useful flip location and lower ordinary flags by at least 20% in both retained panels.",
           "offset_intersection": "Gain useful locations on at least two photos over flip with no more ordinary flags in each retained panel."}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def shifted_source(rgb, offset):
    """Place an unchanged source rectangle in a reflected, offset input."""
    dy, dx = offset
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 64:
        raise ValueError("Expected a native RGB image")
    if min(dy, dx) < 0 or max(rgb.shape[0] + dy, rgb.shape[1] + dx) > 1024:
        raise ValueError("Offset image would change scale or source geometry")
    return np.pad(rgb, ((dy, 0), (dx, 0), (0, 0)), mode="reflect")


def masks_at_source(model, data, pixels, offset):
    height, width = pixels.shape[:2]
    shifted = shifted_source(pixels, offset)
    prediction = model.predict(Image.fromarray(shifted))
    dy, dx = offset
    if prediction.raw_mask.shape != shifted.shape[:2]:
        raise ValueError("The predicted map does not match the shifted input")
    raw = prediction.raw_mask[dy:dy + height, dx:dx + width]
    active = int(np.count_nonzero(raw > regional.PIXEL_THRESHOLD))
    gate = area_gate(active, int(np.ceil(regional.MINIMUM_AREA_FRACTION * raw.size)), raw.size)
    readout = regional_support.readout(raw, gate["eligible"], regional.PIXEL_THRESHOLD, data, pixels)
    return readout.selected_grid_mask, prediction.geometry["input_float32_sha256"], sha(shifted.tobytes())


def combine_arms(base, shifted):
    horizontal, vertical = (shifted[arm] for arm in SHIFTED)
    common = horizontal & vertical
    return {**base, **shifted, "offset_intersection": common,
            "original_supported_by_both": combine(base["native"], common)["supported_components"],
            "original_supported_by_either": combine(base["native"], horizontal | vertical)["supported_components"]}


def profile(args):
    prior = sources.read(args.flip_run / "result.json")
    if sources.sha(args.flip_run / "rows.json") != prior["rows_sha256"] or prior["views"] != 192:
        raise ValueError("The exact previous image journal is required")
    model, model_hash = active.weights(regional.MODEL_ID)
    old, old_hash = active.weights("tidal.splice_v30_1")
    code = (Path(__file__), Path(regional.__file__), Path(regional_support.__file__), Path(splice.__file__),
            Path(sources.__file__), Path(sources.paired.__file__), Path(flux.__file__), Path(combine.__code__.co_filename))
    root = Path(regional.__file__).resolve().parent
    code += (root / "localiser.py", root / "jpeg_bitstream.py", *(root / "regional_graph").glob("*.py"))
    cases, _ = flux.source_cases()
    protocol = {
        "schema": SCHEMA, "source_views": [96, 96, len(cases) * len(flux.KINDS)], "cohorts": list(COHORTS),
        "intakes": {name: {file: sources.sha(folder / file) for file in ("plan.json", "result.json", "attempts.jsonl")}
                    for name, folder in zip(COHORTS[:2], (args.earlier, args.later), strict=True)},
        "flux_source_receipt_sha256": sources.sha(flux.V285_RESULT),
        "previous_flip_result_sha256": sources.sha(args.flip_run / "result.json"),
        "previous_flip_journal_sha256": prior["rows_sha256"],
        "code_sha256": {str(path.relative_to(sources.ROOT)): sources.sha(path) for path in code},
        "weights_sha256": {"regional": model_hash, "recompression": old_hash},
        "method": {"offsets_yx": [list(offset) for offset in OFFSETS], "padding": "reflect above or left", "arms": list(ARMS),
                   "raw_point": regional.PIXEL_THRESHOLD, "gate_fraction": regional.MINIMUM_AREA_FRACTION,
                   "native_component_overlap": 0.5, "useful_iou": 0.3, "useful_recall": 0.5, "advance": ADVANCE,
                   "confidence_contribution": 0},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__, "threads": 2, "device": "cpu"},
    }
    return protocol, model, old


def inputs(args):
    prior_rows = {(row["panel"], row["slot"], row["role"]): row for row in sources.read(args.flip_run / "rows.json")}
    for cohort, path in zip(COHORTS[:2], (args.earlier, args.later), strict=True):
        for source, views, native_truth in sources.paired.inputs(path):
            slot = source["slot"]
            for role, pixels in sources.native_variants(views, source["box"]).items():
                row = prior_rows.pop((cohort, slot, role))
                truth = np.any(pixels != views["original"], axis=2) if role in POSITIVES else np.zeros_like(native_truth)
                data, image = sources.paired.encoded(pixels, "png")
                if row["truth_sha256"] != sha(truth.tobytes()) or row["input_sha256"] != sha(data):
                    raise ValueError("The source geometry or encoded input differs from the previous run")
                proof = args.flip_run / "masks" / f"{cohort}-{slot:02d}-{role}.bin"
                old_masks = restore_masks(proof, row)
                yield cohort, slot, role, data, image, truth, {
                    "native": old_masks["native"], "flip": old_masks["supported_components"], "recompression": old_masks["recompression"]}
    if prior_rows:
        raise ValueError("Previous PNG views are incomplete")
    cases, index = flux.source_cases()
    for slot in range(len(cases)):
        views, native_truth = flux.material(cases, index, slot)
        for role in flux.KINDS:
            data, image = flux.previous.encode(views[role], 0)
            truth = native_truth[role] if role in POSITIVES else np.zeros_like(native_truth[role])
            yield "flux", slot, role, data, image, truth, None


def evaluate(args, model, older):
    for cohort, slot, role, data, image, truth, base in inputs(args):
        pixels = np.asarray(image)
        if base is None:
            native, _, _ = masks_at_source(model, data, pixels, (0, 0))
            reflected = np.ascontiguousarray(pixels[:, ::-1])
            mirrored, _, _ = masks_at_source(model, data, reflected, (0, 0))
            old_raw = splice.predict(older, image)
            old_gate = np.count_nonzero(old_raw > splice.PIXEL_THRESHOLD) >= splice.MINIMUM_ACTIVE_CELLS
            old_mask = regional_support.readout(old_raw, old_gate, splice.PIXEL_THRESHOLD, data, pixels).selected_grid_mask
            base = {"native": native, "flip": combine(native, mirrored[:, ::-1])["supported_components"],
                    "recompression": old_mask}
        shifted = {}
        commitments = []
        for arm, offset in zip(SHIFTED, OFFSETS, strict=True):
            shifted[arm], prepared, padded = masks_at_source(model, data, pixels, offset)
            commitments.append({"offset_yx": list(offset), "padded_rgb_sha256": padded, "prepared_float32_sha256": prepared})
        masks = combine_arms(base, shifted)
        packed = b"".join(np.packbits(masks[arm], axis=None).tobytes() for arm in (*ORIGINAL, *SHIFTED))
        path = args.output / "masks" / f"{cohort}-{slot:02d}-{role}.bin"
        with path.open("xb") as out:
            out.write(packed)
        yield {"cohort": cohort, "slot": slot, "role": role, "input_sha256": sha(data), "truth_sha256": sha(truth.tobytes()),
               "shape": list(truth.shape), "offset_commitments": commitments, "mask_sha256": sha(packed),
               "measurements": {arm: sources.scoring.measure(mask, truth) for arm, mask in masks.items()}}


def summarize(rows):
    if len(rows) != 222 or len({(r["cohort"], r["slot"], r["role"]) for r in rows}) != len(rows):
        raise ValueError("Need all 222 distinct PNG views")
    totals = {}
    for cohort in COHORTS:
        positive = [r for r in rows if r["cohort"] == cohort and r["role"] in POSITIVES]
        control = [r for r in rows if r["cohort"] == cohort and r["role"] not in POSITIVES]
        results = {}
        for arm in ARMS:
            detected = [r for r in positive if sources.useful(r["measurements"][arm])]
            false = [r for r in control if r["measurements"][arm]["reported"] > 0]
            results[arm] = {"useful": len(detected), "missed": len(positive) - len(detected), "controls_flagged": len(false),
                            "controls_flagged_by_type": {role: sum(r["role"] == role for r in false)
                                                         for role in sorted({r["role"] for r in control})},
                            "useful_lost_vs_flip": sum(r not in detected and sources.useful(r["measurements"]["flip"]) for r in positive),
                            "additional_useful_groups_vs_flip": len({r["slot"] for r in detected if not sources.useful(r["measurements"]["flip"])})}
        totals[cohort] = {"positive": len(positive), "controls": len(control), "arms": results}
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for key in ("earlier", "later", "flip_run", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    protocol, native, older = profile(args)
    lock = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        sources.write(lock, protocol)
        return
    if protocol != sources.read(lock):
        raise ValueError("Inputs, implementation or weights differ from the freeze")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    (args.output / "masks").mkdir()
    start = time.monotonic()
    rows = []
    journal = args.output / "rows.jsonl"
    model, splice_model = regional.load_model(native), splice.load_model(older)
    with journal.open("x") as out:
        for row in evaluate(args, model, splice_model):
            rows.append(row)
            out.write(json.dumps(row, allow_nan=False) + "\n")
            out.flush()
            if len(rows) % 12 == 0:
                print(f"{len(rows)}/222 source views measured", flush=True)
    sources.write(args.output / "result.json", {"schema": SCHEMA, "freeze_sha256": sources.sha(lock),
                  "journal_sha256": sources.sha(journal), "elapsed_s": round(time.monotonic() - start, 3),
                  "inference_calls": len(rows) * 2 + 3 * 30, "counts": summarize(rows)})


if __name__ == "__main__":
    main()
