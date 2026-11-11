"""Transfer a frozen crop cue and test a local RGB adjacency cue on small edits."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import image_background_patch_transfer as panel
from scripts import image_bg_real_insertion_score as scoring
from scripts import image_crop_remote_context as crop
from scripts import image_flux_product_comparison_v288 as product
from scripts import image_relative_patch_transfer as shared

FORMATS = ("png", "jpeg75")
POINT_FILE_SHA256 = "7de6ed718c1819b4d027640053244e834b09dc1a62ea74e44e3704335985ca85"
TAU = 0.004
SCALES = (6, 8)
STEP = 2


def energy_field(pixels):
    """Average bounded 3x3 RGB differences at distances of one and two pixels."""
    linear = pixels.astype(np.float32) / 255.0
    height, width = pixels.shape[:2]
    padded = np.pad(linear, ((2, 2), (2, 2), (0, 0)), mode="reflect")
    result = np.zeros((height, width), dtype=np.float32)
    for radius in (1, 2):
        for dy in (-radius, 0, radius):
            for dx in (-radius, 0, radius):
                if dy or dx:
                    neighbor = padded[2+dy:2+dy+height, 2+dx:2+dx+width]
                    diff = np.square(neighbor - linear).sum(axis=2)
                    result += np.tanh(diff / (TAU * TAU))
    return result / 16


def measure_region(pixels):
    """Rank a region by lower local energy than its immediate surroundings."""
    summed = np.pad(energy_field(pixels).astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    height, width = pixels.shape[:2]
    rows, cols = ([(n * i + shared.GRID - 1) // shared.GRID for i in range(shared.GRID + 1)]
                  for n in (height, width))

    def total(x0, y0, x1, y1):
        return summed[y1, x1] + summed[y0, x0] - summed[y1, x0] - summed[y0, x1]

    best = (-float("inf"), ())
    for side in SCALES:
        for y in range(0, shared.GRID - side + 1, STEP):
            top, bottom = rows[y], rows[y + side]
            outer_top, outer_bottom = rows[max(y - 1, 0)], rows[min(y + side + 1, shared.GRID)]
            for x in range(0, shared.GRID - side + 1, STEP):
                left, right = cols[x], cols[x + side]
                outer_left, outer_right = cols[max(x - 1, 0)], cols[min(x + side + 1, shared.GRID)]
                inner_area = (bottom - top) * (right - left)
                outer_area = (outer_bottom - outer_top) * (outer_right - outer_left) - inner_area
                inner = total(left, top, right, bottom)
                outer = total(outer_left, outer_top, outer_right, outer_bottom) - inner
                gap = outer / outer_area - inner / inner_area
                if gap > best[0]:
                    best = float(gap), (left, top, right, bottom)
    return best


def encoded(pixels, fmt):
    return shared.flux.previous.encode(Image.fromarray(pixels), 0 if fmt == "png" else 75)


def inputs(folder):
    intake = panel.roster.read(folder / "result.json")
    plan = panel.roster.read(folder / "plan.json")
    journal = folder / "attempts.jsonl"
    panel.roster.require(panel.roster.sha(folder / "plan.json") == intake["plan_sha256"]
                         and panel.roster.sha(journal) == intake["attempts_sha256"], "Intake commitments changed")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    for case, row in zip(plan["cases"], rows, strict=True):
        assert case["slot"] == row["slot"] and case["generator"] == row["generator"]
        if row["status"] != "qualified":
            continue
        native = {}
        for role, receipt in row["assets"].items():
            path = folder / "assets" / f"{row['slot']:02d}-{role}.png"
            body = path.read_bytes()
            assert shared.digest(body) == receipt["sha256"]
            native[role] = panel.source.png_pixels(body, case["size"])
        assert tuple(row["box"]) == panel.region(native["semantic"][:, :, 0] > 0, case["recipient"])
        views, truth = panel.compose(native, row["box"])
        assert shared.digest(truth.tobytes()) == row["truth_sha256"]
        assert {key: shared.digest(value.tobytes()) for key, value in views.items()} == row["case_sha256"]
        yield row, views, truth


def energy_points(dev):
    report, plan, attempts = scoring.checked_inputs(dev)
    group = {fmt: [] for fmt in FORMATS}
    for _, _, native, _ in scoring.panels(dev, plan, attempts):
        for fmt in FORMATS:
            group[fmt].append(max(measure_region(np.asarray(encoded(native[role], fmt)[1]))[0]
                                  for role in ("original", "processed")))
    assert len(group["png"]) == len(group["jpeg75"]) == report["qualified"] >= 2
    return {fmt: {"point": sorted(values)[-2], "fit_group_maxima": values} for fmt, values in group.items()}


def boxed(box, shape, show):
    mask = np.zeros(shape, dtype=bool)
    if show:
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = True
    return mask


def summarise(rows):
    groups = defaultdict(list)
    for row in rows:
        for name, measurement in row["measurements"].items():
            groups[row["generator"], row["format"], name, row["role"]].append(measurement)
    output = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for (family, fmt, method, role), values in groups.items():
        measures = {"views": len(values), "flags": sum(v["reported"] > 0 for v in values)}
        if role == "edited":
            measures["useful"] = sum(v["iou"] >= .30 and v["recall"] >= .50 for v in values)
            measures["mean_iou"] = float(np.mean([v["iou"] for v in values]))
        output[family][fmt][method][role] = measures
    return output


def run(args):
    import torch

    from app.detectors.image import regional_support, splice
    from scripts.image_native_comparison_readout_v45 import source_mask

    torch.set_num_threads(2)
    current = shared.flux.ROOT / product.SPLICE
    identity = {
        "schema": "chipmunk.background-patch-score/1",
        "script_sha256": shared.file_digest(Path(__file__)),
        "composition_sha256": shared.file_digest(Path(panel.__file__)),
        "intake_sha256": shared.file_digest(args.intake / "result.json"),
        "development_sha256": shared.file_digest(args.development / "result.json"),
        "source_crop_code_sha256": shared.file_digest(Path(crop.__file__)),
        "weights_sha256": shared.file_digest(args.weights),
        "current_png_sha256": shared.file_digest(current),
        "previous_remote_point_sha256": shared.file_digest(args.points),
        "region_temperature": TAU,
        "adjacency_offsets": [1, 2],
        "region_side_cells": list(SCALES),
        "step_cells": STEP,
        "formats": list(FORMATS),
    }
    assert identity["weights_sha256"] == shared.WEIGHT_SHA
    assert identity["current_png_sha256"] == product.SPLICE_SHA
    assert identity["previous_remote_point_sha256"] == POINT_FILE_SHA256
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        shared.write(args.output / "freeze.json", identity)
        return
    assert identity == json.loads((args.output / "freeze.json").read_bytes())
    before = time.monotonic()
    remote = json.loads(args.points.read_bytes())
    energy = energy_points(args.development)
    shared.write(args.output / "energy_points.json", energy)
    current_model = splice.load_model(current)
    crop_model = crop.inference(args.weights)
    observations = []
    with (args.output / "rows.jsonl").open("x") as log:
        for row, views, truth in inputs(args.intake):
            for role, pixels in views.items():
                expected = truth if role == "edited" else np.zeros_like(truth)
                for fmt in FORMATS:
                    body, image = encoded(pixels, fmt)
                    pixels_for_check = np.asarray(image)
                    box, relative, remote_median = crop_model(image)
                    passing_crop = relative >= remote["relative"]
                    passing_remote = passing_crop and remote_median < remote["remote_upper_strict"]
                    score, energy_box = measure_region(pixels_for_check)
                    regions = {"relative_crop": boxed(box, truth.shape, passing_crop),
                               "remote_crop": boxed(box, truth.shape, passing_remote),
                               "energy_deficit": boxed(energy_box, truth.shape, score > energy[fmt]["point"])}
                    if fmt == "png":
                        raw = splice.predict(current_model, image)
                        gate = np.count_nonzero(raw > splice.PIXEL_THRESHOLD) >= splice.MINIMUM_ACTIVE_CELLS
                        mask = regional_support.readout(raw, bool(gate), splice.PIXEL_THRESHOLD, body,
                                                       pixels_for_check).source_mask(image.width, image.height)
                        assert np.array_equal(mask, source_mask(raw, bool(gate), splice.PIXEL_THRESHOLD,
                                                                body, pixels_for_check, expected))
                        regions["current_png"] = mask
                    outcome = {"slot": row["slot"], "generator": row["generator"], "role": role, "format": fmt,
                               "input_sha256": shared.digest(body), "measurements": {
                                   name: scoring.measure(mask, expected) for name, mask in regions.items()}}
                    observations.append(outcome)
                    log.write(json.dumps(outcome, allow_nan=False) + "\n")
                    log.flush()
            print(f"Group {row['slot']} complete", flush=True)
    counts = Counter(row["generator"] for row in observations)
    assert len(observations) == sum(panel.roster.read(args.intake / "result.json")["qualified"].values()) * 10
    shared.write(args.output / "result.json", {
        "schema": identity["schema"], "views_by_generator": counts, "views": len(observations),
        "freeze_sha256": shared.file_digest(args.output / "freeze.json"),
        "results_sha256": shared.file_digest(args.output / "rows.jsonl"),
        "energy_points_sha256": shared.file_digest(args.output / "energy_points.json"),
        "current_png_forwards": len(observations) // 2, "crop_inputs": 25 * len(observations),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_s": round(time.monotonic() - before, 3), "results": summarise(observations),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for key in ("intake", "development", "weights", "points", "output"):
        parser.add_argument(key, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
