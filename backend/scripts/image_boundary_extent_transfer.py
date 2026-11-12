# SPDX-License-Identifier: Apache-2.0
"""Evaluate flexible native rectangle extent and measured support along its sides."""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from app.detectors.image import sharp_boundary
from scripts import image_edge_spread_transfer as panel
from scripts import image_reference_orientation_transfer as commons

SCHEMA = "chipmunk.image-boundary-extent-transfer/1"
ARMS = ("fixed", "flexible", "distributed")
COHORTS = ("earlier", "later", "newest")
CONTROLS = ("original", "local_processing", "global_processing", "blur", "feathered_processing", "local_contrast",
            "small_processing", "small_blur")
POSITIVES = ("edited", "feathered_edit", "small_sharp", "small_feathered")
STEP = 4
WIDTH_STEP = 16
MIN_SIDE = 36
MARGIN = 15
REFINE = 9
MAX_PROPOSALS = 36


def prefix(values, axis):
    borders = ((1, 0), (0, 0)) if axis == 0 else ((0, 0), (1, 0))
    return np.pad(values.cumsum(axis=axis), borders)


def coarse(vertical, horizontal, shape):
    height, width = shape
    for bw in range(MIN_SIDE, round(width * .35) + 1, WIDTH_STEP):
        left = np.arange(MARGIN, width - MARGIN - bw, STEP)
        right = left + bw
        for bh in range(MIN_SIDE, round(height * .35) + 1, WIDTH_STEP):
            top = np.arange(MARGIN, height - MARGIN - bh, STEP)
            bottom = top + bh
            scores = (vertical[np.ix_(bottom, left)] - vertical[np.ix_(top, left)]) / bh
            scores *= (vertical[np.ix_(bottom, right)] - vertical[np.ix_(top, right)]) / bh
            scores *= (horizontal[np.ix_(top, right)] - horizontal[np.ix_(top, left)]) / bw
            scores *= (horizontal[np.ix_(bottom, right)] - horizontal[np.ix_(bottom, left)]) / bw
            for _ in range(2):
                y, x = np.unravel_index(int(scores.argmax()), scores.shape)
                yield float(scores[y, x]), (int(left[x]), int(top[y]), int(right[x]), int(bottom[y]))
                scores[max(0, y-6):y+7, max(0, x-6):x+7] = -1


def close(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    total = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return 7 * w * h >= 3 * total


def refine(box, vertical, horizontal):
    left, top, right, bottom = box
    x0, y0, x1, y1 = box
    for _ in range(2):
        line = vertical[bottom] - vertical[top]
        left, right = (p - REFINE + int(line[p-REFINE:p+REFINE+1].argmax()) for p in (x0, x1))
        line = horizontal[:, right] - horizontal[:, left]
        top, bottom = (p - REFINE + int(line[p-REFINE:p+REFINE+1].argmax()) for p in (y0, y1))
    return left, top, right, bottom


def sides(box, vertical, horizontal):
    left, top, right, bottom = box
    yr = np.linspace(top, bottom, 5, dtype=int)
    xr = np.linspace(left, right, 5, dtype=int)
    quarters = np.stack([np.diff(vertical[yr, left]) / np.diff(yr), np.diff(horizontal[top, xr]) / np.diff(xr),
                         np.diff(vertical[yr, right]) / np.diff(yr), np.diff(horizontal[bottom, xr]) / np.diff(xr)])
    means = quarters @ np.diff(yr) / (bottom - top)
    means[[1, 3]] = quarters[[1, 3]] @ np.diff(xr) / (right - left)
    return [float(value) for value in means], quarters.tolist()


def propose(pixels):
    panel.require(sharp_boundary.supported_size(pixels.shape[1], pixels.shape[0]), "Unsupported native dimensions")
    rgb = pixels.astype(np.float32)
    v = sharp_boundary._step_response(rgb)
    h = sharp_boundary._step_response(rgb.transpose(1, 0, 2)).T
    vertical, horizontal = prefix(v, 0), prefix(h, 1)
    broad_v = prefix(cv2.dilate(v, np.ones((1, 13), dtype=np.uint8)), 0)
    broad_h = prefix(cv2.dilate(h, np.ones((13, 1), dtype=np.uint8)), 1)
    bases, outputs = [], dict.fromkeys(ARMS[1:])
    for _, seed in sorted(coarse(broad_v, broad_h, pixels.shape[:2]), reverse=True):
        if any(close(seed, previous) for previous in bases):
            continue
        bases.append(seed)
        box = refine(seed, vertical, horizontal)
        if min(box[2] - box[0], box[3] - box[1]) >= 24:
            averages, quarters = sides(box, vertical, horizontal)
            for arm, evidence in ((ARMS[1], averages), (ARMS[2], [sorted(side)[1] for side in quarters])):
                score = math.prod(max(0., x) for x in evidence) ** .25
                if outputs[arm] is None or score > outputs[arm]["score"]:
                    outputs[arm] = {"box": list(box), "score": score, "means": averages, "quarters": quarters}
        if len(bases) == MAX_PROPOSALS:
            break
    panel.require(all(outputs.values()), "No assessable boundary proposals")
    return outputs


def protocol(args):
    study_paths = (args.calibration, args.earlier, args.later, args.newest)
    plans = [panel.read(path / "plan.json")["cases"] for path in study_paths]
    for field in ("flickr_photo_id", "flickr_owner"):
        groups = [{row[field] for row in plan} for plan in plans]
        panel.require([len(group) for group in groups] == [8, 12, 12, 7] and len(set.union(*groups)) == 39,
                      "Expected disjoint recorded source and author groups")
    ordinary = commons.binding(args.commons)["sources"]
    code = (Path(__file__), *(Path(mod.__file__) for mod in (sharp_boundary, panel, panel.paired, panel.blend, commons)))
    return {"schema": SCHEMA, "code": {path.name: panel.sha(path) for path in code},
            "intakes": {label: {name: panel.sha(folder / name) for name in ("plan.json", "result.json", "attempts.jsonl")}
                        for label, folder in zip(("calibration", *COHORTS), study_paths, strict=True)},
            "commons": {"acquisition": panel.sha(args.commons / "acquisition.json"), "controls": ordinary[::3]},
            "rules": {"point": "Greater than the second-largest control maximum from eight separate source groups.",
                      "advance": "At least five new small sharp locations in at least three groups. Preserve at least 85 percent of incumbent sharp locations and add no more than four net ordinary flags and no more than one on the untouched Commons views.",
                      "selected_arms": list(ARMS), "held_formats": ["png"], "seed_grid": STEP, "dimension_step": WIDTH_STEP,
                      "refinement": REFINE, "proposal_limit": MAX_PROPOSALS}}


def calibrate(folder):
    _, plan, checked = panel.scoring.checked_inputs(folder)
    maxima = defaultdict(list)
    for _, _, native, _ in panel.scoring.panels(folder, plan, checked):
        scores = [propose(rgb) for rgb in panel.development_controls(native["original"], native["processed"]).values()]
        for arm in ARMS[1:]:
            maxima[arm].append(max(value[arm]["score"] for value in scores))
    panel.require(all(len(maxima[arm]) == 8 for arm in ARMS[1:]), "Incomplete calibration")
    return {arm: {"point": sorted(values)[-2], "source_maxima": values} for arm, values in maxima.items()}


def small_rectangle(box, slot):
    x, y, right, bottom = box
    w, h = right - x, bottom - y
    bw, bh = (round(w * .47), round(h * .65)) if slot % 2 else (round(w * .68), round(h * .46))
    left = x + (w - bw) * (1 + slot % 3) // 4
    top = y + (h - bh) * (1 + (slot // 3) % 3) // 4
    return left, top, left + bw, top + bh


def view_inputs(folder):
    for case, native, truth in panel.paired.inputs(folder):
        original, box = native["original"], case["box"]
        small = small_rectangle(box, case["slot"])
        blur = np.asarray(Image.fromarray(original).filter(ImageFilter.GaussianBlur(1.25)))
        views = panel.native_variants(native, box) | {
            "small_sharp": panel.blend.local(original, native["edited"], small),
            "small_feathered": panel.blend.local(original, native["edited"], small, feather=True),
            "small_processing": panel.blend.local(original, native["local_processing"], small),
            "small_blur": panel.blend.local(original, blur, small),
        }
        for role, rgb in views.items():
            known = np.any(original != rgb, axis=2) if role in POSITIVES else np.zeros_like(truth)
            panel.require(not (known & ~truth).any() and (role not in POSITIVES or known.any()), "Unexpected truth geometry")
            yield case["slot"], case["generator"], role, rgb, known


def observe(cohort, slot, editor, role, rgb, truth, points):
    data, png = panel.paired.encoded(rgb, "png")
    pixels = np.asarray(png)
    baseline = sharp_boundary.measure(pixels)
    proposals = propose(pixels) | {ARMS[0]: {"box": list(baseline.box), "score": baseline.score}}
    thresholds = {arm: value["point"] for arm, value in points.items()} | {ARMS[0]: sharp_boundary.REVIEW_POINT}
    counts = {}
    for arm, item in proposals.items():
        active = item["score"] > thresholds[arm]
        x, y, r, b = item["box"]
        counts[arm] = [int((r - x) * (b - y)) if active else 0, int(truth.sum()), int(truth[y:b, x:r].sum()) if active else 0]
    return {"cohort": cohort, "slot": slot, "editor": editor, "role": role, "input_sha256": commons.sha(data),
            "truth_sha256": commons.sha(truth.tobytes()), "proposals": proposals, "counts": counts}


def useful(counts):
    area, target, overlap = counts
    return target > 0 and 10 * overlap >= 3 * (area + target - overlap) and 2 * overlap >= target


def summarize(rows):
    counts = {arm: defaultdict(Counter) for arm in ARMS}
    held = {arm: Counter() for arm in ARMS[1:]}
    source_gains = {arm: set() for arm in ARMS[1:]}
    for row in rows:
        role = row["role"]
        baseline = row["counts"][ARMS[0]]
        for arm, geometry in row["counts"].items():
            counts[arm][role].update({"views": 1, "flags": bool(geometry[0]), "useful": useful(geometry)})
            if arm == ARMS[0]:
                continue
            if role == "small_sharp" and useful(geometry) and not useful(baseline):
                held[arm]["gains"] += 1
                source_gains[arm].add((row["cohort"], row["slot"]))
            if role == "edited" and useful(baseline):
                held[arm]["incumbent_sharp"] += 1
                held[arm]["preserved_sharp"] += useful(geometry)
            if role in CONTROLS:
                held[arm]["net_ordinary"] += bool(geometry[0]) - bool(baseline[0])
            if role == "commons_original":
                held[arm]["added_commons"] += bool(geometry[0] and not baseline[0])
    decisions = {arm: dict(tally) | {"distinct_gains": len(source_gains[arm]),
                    "advances": tally["gains"] >= 5 and len(source_gains[arm]) >= 3
                    and 100 * tally["preserved_sharp"] >= 85 * tally["incumbent_sharp"]
                    and tally["net_ordinary"] <= 4 and tally["added_commons"] <= 1} for arm, tally in held.items()}
    return {"counts": counts, "decision": decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "holdout", "verify"))
    for name in ("calibration", *COHORTS, "commons", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    frozen, points_path = (args.output / filename for filename in ("freeze.json", "points.json"))
    lock = protocol(args)
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        panel.write(frozen, lock)
        return
    panel.require(panel.read(frozen) == lock, "Frozen study differs")
    if args.phase == "fit":
        panel.write(points_path, calibrate(args.calibration))
        return
    start = time.monotonic()
    points = panel.read(points_path)
    rows = []
    for cohort in COHORTS:
        rows.extend(observe(cohort, *item, points) for item in view_inputs(getattr(args, cohort)))
        print(cohort, len(rows), flush=True)
    for item in lock["commons"]["controls"]:
        with Image.open(args.commons / item["file"]) as raw, ImageOps.exif_transpose(raw) as oriented:
            rgb = np.asarray(ImageOps.fit(oriented.convert("RGB"), (704, 544), Image.Resampling.LANCZOS))
        rows.append(observe("commons", item["slot"], None, "commons_original", rgb, np.zeros(rgb.shape[:2], dtype=bool), points))
    result = summarize(rows) | {"freeze_sha256": panel.sha(frozen), "points_sha256": panel.sha(points_path), "views": len(rows)}
    journal, aggregate = (args.output / filename for filename in ("rows.json", "result.json"))
    if args.phase == "holdout":
        panel.write(journal, rows)
        panel.write(aggregate, result)
    else:
        panel.require(calibrate(args.calibration) == points and panel.read(journal) == rows and panel.read(aggregate) == result,
                      "Exact fit, held or aggregate replay differs")
        panel.write(args.output / "audit.json", {"views": len(rows), "held_measurements": 3 * len(rows),
                    "result_sha256": panel.sha(aggregate), "elapsed_s": round(time.monotonic() - start, 3)})
    print(result["decision"], "elapsed", round(time.monotonic() - start, 3), flush=True)


if __name__ == "__main__":
    main()
