# SPDX-License-Identifier: Apache-2.0
"""Evaluate whether learned JPEG regions survive a second encoding of one upload."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
import torch
from PIL import Image
from PIL import __version__ as pillow_version

from app.detectors.image import regional, regional_support
from scripts import image_directional_structure_transfer as panel
from scripts import image_jpeg_diagnostic_v189 as earlier
from scripts.localization_benchmark import mask_metrics

SCHEMA = "chipmunk.image-regional-jpeg-transfer/1"
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "models/weights/native-regional-v1" / regional.CHECKPOINT_FILE
COUNTS = {"confirmation": 96, "external": 64}
POSITIVE = ("sharp", "feathered")
ARMS = ("current_luma", "native", "corroborated")
DESIGN = {
    "delivery": "One JPEG75 4:2:0 upload. Use decoded upright RGB without registration or a source pair.",
    "native": {"pixel_cutoff": regional.PIXEL_THRESHOLD, "area_fraction": regional.MINIMUM_AREA_FRACTION,
               "component_cap": 2},
    "derived": {"quality": 90, "subsampling": "4:2:0", "optimize": False, "progressive": False},
    "primary": "Retain an unchanged native component when at least half of its grid pixels also belong to a selected component in the derived view.",
    "quality_support": "Apply the original upload conditions mask to both views.",
    "useful": {"intersection_over_union": .30, "changed_pixel_recall": .50},
    "comparison": "Existing aligned JPEG luma region, sensitive native learned regions alone, and corroborated native learned regions.",
    "confidence_weight": 0,
}
read, write, require = panel.read, panel.write, panel.require


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def native_readout(prediction, source_bytes, rgb):
    return regional_support.readout(
        prediction.raw_mask, prediction.candidate_eligibility["eligible"],
        regional.PIXEL_THRESHOLD, source_bytes, rgb,
    )


def retained_components(components, second):
    retained = np.zeros_like(second)
    overlap = []
    for component in components:
        area = int(component.sum())
        agreement = int(np.count_nonzero(component & second))
        overlap.append({"component_cells": area, "agreeing_cells": agreement})
        if agreement * 2 >= area:
            retained |= component
    return retained, overlap


def derived_view(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, subsampling="4:2:0", optimize=False, progressive=False)
    body = buffer.getvalue()
    with Image.open(io.BytesIO(body)) as decoded:
        return body, decoded.convert("RGB")


def project(mask, image):
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)) > 0


def evaluate(model, body, truth, incumbent):
    with Image.open(io.BytesIO(body)) as image_file:
        require(image_file.format == "JPEG", "This study requires a JPEG source")
        image = image_file.convert("RGB")
    rgb = np.asarray(image)
    native = model.predict(image)
    original = native_readout(native, body, rgb)
    traces = {"source_rgb_sha256": native.geometry["source_rgb_sha256"],
              "model_input_sha256": native.geometry["input_float32_sha256"],
              "native_raw_sha256": sha(native.raw_mask.tobytes()), "derived": None, "overlap": []}
    corroborated = np.zeros_like(original.selected_grid_mask)
    if original.component_masks:
        recompressed_body, recompressed_image = derived_view(image)
        second = model.predict(recompressed_image)
        derived = native_readout(second, body, rgb)
        corroborated, traces["overlap"] = retained_components(original.component_masks, derived.selected_grid_mask)
        traces["derived"] = {"input_sha256": sha(recompressed_body),
                             "model_input_sha256": second.geometry["input_float32_sha256"],
                             "raw_sha256": sha(second.raw_mask.tobytes())}
    if incumbent is None:
        reference = panel.previous.prior.base.comparison.format_reference("jpeg75", body, rgb)
        incumbent = reference["box"] if reference and reference["selected"] else None
    else:
        incumbent = incumbent["box"]
    current = np.zeros_like(truth)
    if incumbent:
        x0, y0, x1, y1 = incumbent
        current[y0:y1, x0:x1] = True
    selected = (current, original.source_mask(*image.size), project(corroborated, image))
    return traces | {
        "metrics": {arm: mask_metrics(mask, truth) | {"selected_sha256": sha(mask.tobytes())}
                    for arm, mask in zip(ARMS, selected, strict=True)}
    }


def is_useful(metrics):
    return metrics["iou"] >= .30 and (metrics["recall"] or 0.) >= .50


def summarize(rows):
    strata, paired = {}, {}
    for arm in ARMS:
        ordinary_groups = defaultdict(set)
        strata[arm] = {}
        for role in sorted({row["role"] for row in rows}):
            subset = [row for row in rows if row["role"] == role]
            hits = [row for row in subset if row["metrics"][arm]["predicted_positive_pixels"]]
            groups = {row["source_group"] for row in hits}
            if role not in POSITIVE:
                ordinary_groups[role] = groups
            strata[arm][role] = {"views": len(subset), "flags": len(hits), "flagged_groups": len(groups),
                                 "useful": sum(is_useful(row["metrics"][arm]) for row in hits)}
        strata[arm]["ordinary_groups_flagged"] = len(set().union(*ordinary_groups.values()))
        positive = [row for row in rows if row["role"] in POSITIVE]
        strata[arm]["wrong_regions"] = sum(bool(row["metrics"][arm]["predicted_positive_pixels"])
                                          and not is_useful(row["metrics"][arm]) for row in positive)
    for candidate, baseline in (("native", "current_luma"), ("corroborated", "current_luma"), ("corroborated", "native")):
        deltas = {}
        for role in POSITIVE:
            new, lost = [], []
            for row in rows:
                if row["role"] != role:
                    continue
                extra, prior = (is_useful(row["metrics"][arm]) for arm in (candidate, baseline))
                if extra and not prior:
                    new.append(row)
                elif prior and not extra:
                    lost.append(row)
            deltas[role] = {"gains": len(new), "losses": len(lost),
                            "gain_groups": len({row["source_group"] for row in new}),
                            "gain_families": sorted({row["family"] for row in new}),
                            "loss_groups": len({row["source_group"] for row in lost})}
        paired[f"{candidate}/{baseline}"] = deltas
    return {"sources": len({row["source_group"] for row in rows}), "views": len(rows), "strata": strata,
            "paired": paired, "derived_forwards": sum(row["derived"] is not None for row in rows)}


def smoke():
    one, two, shifted = np.zeros((8, 8), bool), np.zeros((8, 8), bool), np.zeros((8, 8), bool)
    one[1:3, 1:3] = two[4:6, 4:6] = shifted[1:2, 1:3] = shifted[4, 4] = True
    supported, agreements = retained_components((one, two), shifted)
    require(np.array_equal(supported, one) and agreements == [
        {"component_cells": 4, "agreeing_cells": 2}, {"component_cells": 4, "agreeing_cells": 1},
    ], "A same-coordinate half-overlap must preserve exactly one original component")


def source_audit(main_plan, main_rows, external_plan, external_intake):
    held_ids = {row["source"] for row in read(main_rows)}
    main = [row for row in read(main_plan)["cases"] if row["recipient"] in held_ids]
    second = [row for row, intake in zip(read(external_plan)["cases"], read(external_intake)["rows"], strict=True)
              if intake["status"] == "qualified"]

    def identities(rows):
        return ({str(row["flickr_photo_id"]) for row in rows},
                {row["flickr_owner"].casefold() for row in rows},
                {row["original"]["sha256"] for row in rows})

    banks = [identities(main), identities(second)]
    require([tuple(map(len, bank)) for bank in banks] == [(12, 12, 12), (7, 7, 7)]
            and not any(left & right for left, right in zip(*banks, strict=True)),
            "Held banks must separate source photo IDs, authors and byte commitments")
    require(file_sha(earlier.PLAN) == earlier.PLAN_SHA, "Historical regional exposure plan differs")
    old = read(earlier.PLAN)["rows"]
    for role in {row["role"] for row in old}:
        rows = [row for row in old if row["role"] == role]
        known = ({str(row["flickr_photo_id"]) for row in rows},
                 {urlsplit(row["author_profile"]).path.rstrip("/").rsplit("/", 1)[-1].casefold() for row in rows},
                 {row["original_sha256"] for row in rows})
        require(not any(left & right for bank in banks for left, right in zip(bank, known, strict=True)),
                "Held bank overlaps recorded local regional sources")
    return {"main_photos": 12, "external_photos": 7, "checked_fields": ["photo_id", "author", "source_sha256"],
            "regional_plan_sha256": earlier.PLAN_SHA, "recorded_regional_roles": sorted({row["role"] for row in old})}


def study_profile(args):
    paths = [Path(__file__), Path(panel.__file__), Path(panel.previous.__file__),
             Path(panel.previous.fresh.__file__), Path(panel.external.__file__),
             Path(regional.__file__), Path(regional_support.__file__)]
    held_paths = (args.output_fresh / "plan.json", args.output_fresh / "measurements.json",
                  args.external / "plan.json", args.external / "intake.json")
    require(file_sha(CHECKPOINT) == regional.CHECKPOINT_SHA256, "Reviewed regional state required")
    return {
        "schema": SCHEMA, "design": DESIGN, "model_sha256": regional.CHECKPOINT_SHA256,
        "code_sha256": {str(path.relative_to(ROOT)): file_sha(path) for path in paths},
        "input_ledgers_sha256": {f"{path.parent.name}/{path.name}": file_sha(path) for path in held_paths},
        "source_disjointness": source_audit(*held_paths),
        "runtime": {"torch": str(torch.__version__), "opencv": cv2.__version__, "numpy": np.__version__,
                    "pillow": pillow_version, "device": "cpu", "cpu_threads": 2, "dtype": "float32"},
    }


def assess(args, lock, model):
    reports = {}
    destination = args.output
    for phase, views in panel.bindings(args, lock):
        if phase not in COUNTS:
            continue
        path = destination / f"{phase}.jsonl"
        saved = None if args.stage == "run" else [json.loads(line) for line in path.read_text().splitlines()]
        rows = []
        if saved is None:
            handle = path.open("x")
        try:
            for key, body, truth, incumbent in views:
                require(sha(body) == key["input_sha256"] and sha(truth.tobytes()) == key["truth_sha256"],
                        "JPEG delivery or changed-pixel truth differs")
                # Both non-LaMa external families use the same photographed scene.
                group = "paired_external_photo" if phase == "external" and key["family"] != "lama" else key["source"]
                result = key | {"source_group": group} | evaluate(model, body, truth, incumbent)
                if saved is None:
                    handle.write(json.dumps(result, allow_nan=False) + "\n")
                    handle.flush()
                else:
                    require(result == saved[len(rows)], "Exact forward or localization replay differs")
                rows.append(result)
                if len(rows) % 8 == 0:
                    print(phase, len(rows), flush=True)
        finally:
            if saved is None:
                handle.close()
        require(len(rows) == COUNTS[phase] and (saved is None or len(saved) == len(rows)),
                "Complete source roster required")
        reports[phase] = summarize(rows)
    require(set(reports) == set(COUNTS) and [reports[phase]["sources"] for phase in COUNTS] == [12, 7],
            "Two complete held banks with grouped photo pairs required")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    for name in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint",
                 "output-fresh", "external", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    args.exclude = []
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    smoke()
    freeze = study_profile(args)
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(args.output / "freeze.json", freeze)
        return
    require(freeze == read(args.output / "freeze.json"), "Frozen experiment differs")
    require(args.stage != "review" or args.publication is not None, "Review publication path required")
    lock, _, _ = panel.previous.fresh.previous(args)
    start = time.monotonic()
    result = freeze | {"results": assess(args, lock, regional.load_model(CHECKPOINT))}
    if args.stage == "run":
        write(args.output / "result.json", result)
    else:
        require(result == read(args.output / "result.json"), "Aggregate result differs on replay")
        write(args.publication, result | {"exact_source_replays": sum(COUNTS.values()),
                                          "replay_seconds": round(time.monotonic() - start, 2)})


if __name__ == "__main__":
    main()
