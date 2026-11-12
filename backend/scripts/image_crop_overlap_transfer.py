"""Test location estimates inferred from the overlap pattern of existing crop responses."""

from __future__ import annotations

import argparse
import json
import resource
import time
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np

from app.detectors.image import commfor
from scripts import image_background_patch_score as background
from scripts import image_bg_real_insertion_score as insertion
from scripts import image_crop_context_transfer as crops
from scripts import image_edge_spread_transfer as panels
from scripts import image_relative_patch_transfer as source

SCHEMA = "chipmunk.image-crop-overlap-transfer/1"
FORMATS = ("png", "jpeg75")
ARMS = ("strongest_relative_crop", "overlap_centered", "overlap_detrended")
POSITIVES = {"edited", "feathered_edit"}
GRID = 24
SIZES = (4, 6, 8)
STEP = 2
RULE = {
    "fit": "Strictly above the second-highest of eight control-group maxima per format and arm.",
    "useful": "Native changed-pixel IoU at least .30 and recall at least .50.",
    "advance": "Detrended arm adds at least two useful feathered locations on distinct photos in each panel, loses at most one useful location per panel and adds at most two control flags and one untouched flag per panel.",
    "response": "One nonnegative occupancy coefficient after removing a constant or an affine crop-position trend. Rank by residual dot product with the unit-norm projected template.",
}


def read(path):
    return json.loads(path.read_bytes())


def digest_grid(values):
    return source.digest(np.ascontiguousarray(values).tobytes())


def bound(length, value):
    return (length * value + GRID - 1) // GRID


@lru_cache(maxsize=32)
def templates(width, height):
    boxes = np.array([(crops.boundaries(width, x)[0], crops.boundaries(height, y)[0],
                       crops.boundaries(width, x)[1], crops.boundaries(height, y)[1])
                      for y, x in product(crops.STARTS, repeat=2)], dtype=np.int64)
    candidates = np.array([(bound(width, x), bound(height, y), bound(width, x + w), bound(height, y + h))
                           for w, h in product(SIZES, repeat=2)
                           for y in range(0, GRID - h + 1, STEP)
                           for x in range(0, GRID - w + 1, STEP)], dtype=np.int64)
    low = np.maximum(candidates[:, None, :2], boxes[None, :, :2])
    high = np.minimum(candidates[:, None, 2:], boxes[None, :, 2:])
    intersections = np.maximum(high - low, 0).prod(axis=2)
    area = (candidates[:, 2:] - candidates[:, :2]).prod(axis=1)
    occupancy = intersections / area[:, None]
    suitable = ((occupancy > 0).sum(axis=1) >= 3) & ((occupancy == 0).sum(axis=1) >= 4) & (occupancy.max(axis=1) >= .5)
    candidates, occupancy = candidates[suitable], occupancy[suitable]
    centers = (boxes[:, :2] + boxes[:, 2:]) / np.array([2 * width, 2 * height]) - .5
    regressors = np.column_stack([np.ones(len(boxes)), centers])
    basis, _ = np.linalg.qr(regressors)
    normalized = []
    for rank in (1, 3):
        projection = basis[:, :rank]
        residual = occupancy - (occupancy @ projection) @ projection.T
        norm = np.linalg.norm(residual, axis=1, keepdims=True)
        normalized.append(np.divide(residual, norm, out=np.zeros_like(residual), where=norm > 1e-12))
    return boxes, candidates, normalized


def proposals(logits, width, height):
    values = np.asarray(logits, dtype=np.float64)
    original, candidates, normalized = templates(width, height)
    if values.shape != (len(original),) or not np.isfinite(values).all():
        raise ValueError("Expected 25 finite crop responses")
    peak, scores = crops.scores(values)
    result = {ARMS[0]: {"score": scores["relative_crop"], "box": original[peak].tolist()}}
    for arm, signatures in zip(ARMS[1:], normalized, strict=True):
        matches = signatures @ values
        best = int(np.argmax(matches))
        result[arm] = {"score": float(max(0., matches[best])), "box": candidates[best].tolist()}
    return result


def input_plan(development, earlier, later):
    dev, dev_plan, dev_rows = insertion.checked_inputs(development)
    if dev["qualified"] != 8:
        raise ValueError("Expected eight qualified fitting source groups")
    plans = [dev_plan, *(read(folder / "plan.json") for folder in (earlier, later))]
    owners = [{case["flickr_owner"] for case in plan["cases"]} for plan in plans]
    if [len(owner) for owner in owners] != [8, 12, 12] or any(a & b for a, b in ((owners[0], owners[1]), (owners[0], owners[2]), (owners[1], owners[2]))):
        raise ValueError("Study groups or credited photographers overlap")
    return dev_plan, dev_rows


def identity(args):
    weight_hash = source.file_digest(args.weight)
    if weight_hash != source.WEIGHT_SHA:
        raise ValueError("Unexpected crop checkpoint digest")
    input_plan(args.development, args.earlier, args.later)
    inputs = {
        label: {name: source.file_digest(folder / name) for name in ("plan.json", "result.json", "attempts.jsonl")}
        for label, folder in (("development", args.development), ("earlier", args.earlier), ("later", args.later))
    }
    code = [Path(__file__), Path(crops.__file__), Path(commfor.__file__), Path(background.__file__),
            Path(insertion.__file__), Path(panels.__file__)]
    return {"schema": SCHEMA, "inputs": inputs, "weight_sha256": weight_hash,
            "code_sha256": {path.name: source.file_digest(path) for path in code},
            "arms": list(ARMS), "formats": list(FORMATS), "grid": GRID, "sizes": list(SIZES), "step": STEP,
            "rule": RULE, "confidence_contribution": 0,
            "groups": {"development": 8, "earlier": 12, "later": 12}}


def source_views(args, *, development):
    if development:
        plan, rows = input_plan(args.development, args.earlier, args.later)
        for slot, _, images, _ in insertion.panels(args.development, plan, rows):
            for role in ("original", "processed"):
                yield "development", slot, "SD1.5", role, images[role], np.zeros(images[role].shape[:2], dtype=bool)
    else:
        for cohort, folder in (("earlier", args.earlier), ("later", args.later)):
            for row, views, truth in background.inputs(folder):
                for role, pixels in panels.native_variants(views, row["box"]).items():
                    target = np.any(pixels != views["original"], axis=2) if role in POSITIVES else np.zeros_like(truth)
                    yield cohort, row["slot"], row["generator"], role, pixels, target


def extractor(weight):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = commfor.load_model(weight, "cpu")
    mean = torch.tensor(commfor.MEAN)[None, :, None, None]
    std = torch.tensor(commfor.STD)[None, :, None, None]
    cache, counters = {}, {"model_batches": 0, "cached_batches": 0}

    def apply(image):
        prepared = [pixels for _, pixels in crops.crops(image)]
        bindings = [digest_grid(pixels) for pixels in prepared]
        logits = []
        for start in range(0, len(prepared), 5):
            key = tuple(bindings[start:start + 5])
            if key not in cache:
                batch = torch.from_numpy(np.stack(prepared[start:start + 5])).permute(0, 3, 1, 2).float() / 255
                cache[key] = [float(value) for value in commfor.logits(model, (batch - mean) / std, "cpu")]
                counters["model_batches"] += 1
            else:
                counters["cached_batches"] += 1
            logits.extend(cache[key])
        return logits, bindings

    return apply, counters


def measurements(result, truth, points):
    measured = {}
    for arm, item in result.items():
        x0, y0, x1, y1 = item["box"]
        chosen = item["score"] > points[arm]
        measured[arm] = {"reported": (x1 - x0) * (y1 - y0) if chosen else 0,
                         "overlap": int(np.count_nonzero(truth[y0:y1, x0:x1])) if chosen else 0,
                         "target": int(np.count_nonzero(truth))}
    return measured


def record(args, apply, *, development, points=None):
    for cohort, slot, editor, role, pixels, truth in source_views(args, development=development):
        for fmt in FORMATS:
            encoded, image = background.encoded(pixels, fmt)
            logits, bindings = apply(image)
            result = proposals(logits, image.width, image.height)
            yield {"cohort": cohort, "slot": slot, "editor": editor, "role": role, "format": fmt,
                   "shape": list(truth.shape), "input_sha256": source.digest(encoded), "truth_sha256": digest_grid(truth),
                   "crop_sha256": bindings, "logits": logits, "proposals": result,
                   **({"measurements": measurements(result, truth, points[fmt])} if points else {})}


def fit(rows):
    result = {}
    for fmt in FORMATS:
        group = {}
        for row in (item for item in rows if item["format"] == fmt):
            best = group.setdefault(row["slot"], {arm: -np.inf for arm in ARMS})
            for arm in ARMS:
                best[arm] = max(best[arm], row["proposals"][arm]["score"])
        if len(group) != 8:
            raise ValueError("Expected all eight independent fitting groups")
        result[fmt] = {arm: sorted(row[arm] for row in group.values())[-2] for arm in ARMS}
    return result


def useful(row, arm):
    report = row["measurements"][arm]
    common, target, proposed = (report[key] for key in ("overlap", "target", "reported"))
    return target > 0 and 10 * common >= 3 * (target + proposed - common) and 2 * common >= target


def summarize(rows):
    if len(rows) != 384 or len({(row["cohort"], row["slot"], row["role"], row["format"]) for row in rows}) != 384:
        raise ValueError("The complete comparison requires 384 distinct held views")
    report = {}
    for cohort, fmt in product(("earlier", "later"), FORMATS):
        cells = [row for row in rows if (row["cohort"], row["format"]) == (cohort, fmt)]
        controls = [row for row in cells if row["role"] not in POSITIVES]
        values = {}
        for arm in ARMS:
            hits = [row for row in cells if useful(row, arm)]
            false = [row for row in controls if row["measurements"][arm]["reported"]]
            values[arm] = {"sharp": sum(row["role"] == "edited" for row in hits),
                           "feathered": sum(row["role"] == "feathered_edit" for row in hits),
                           "ordinary_flags": len(false), "untouched_flags": sum(row["role"] == "original" for row in false),
                           "added_feathered_groups": len({row["slot"] for row in hits if row["role"] == "feathered_edit"
                                                           and not useful(row, ARMS[0])}),
                           "lost_baseline_locations": sum(useful(row, ARMS[0]) and not useful(row, arm) for row in cells)}
        report[f"{cohort}/{fmt}"] = values
    return report


def run(args):
    locked = args.output / "freeze.json"
    if identity(args) != read(locked):
        raise ValueError("The frozen image crop protocol differs")
    start = time.monotonic()
    apply, counters = extractor(args.weight)
    development = list(record(args, apply, development=True))
    points = fit(development)
    source.write(args.output / "development.json", development)
    source.write(args.output / "points.json", points)
    held = []
    journal = args.output / "held.jsonl"
    with journal.open("x") as stream:
        for row in record(args, apply, development=False, points=points):
            held.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if len(held) % 16 == 0:
                print(f"{len(held)}/384 held views", flush=True)
    source.write(args.output / "result.json", {
        "schema": SCHEMA, "freeze_sha256": source.file_digest(locked), "held_sha256": source.file_digest(journal),
        "points_sha256": source.file_digest(args.output / "points.json"), "counts": summarize(held),
        "development_views": len(development), "held_views": len(held), **counters,
        "elapsed_s": round(time.monotonic() - start, 3), "peak_rss_bytes_mac": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for key in ("development", "earlier", "later", "weight", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        profile = identity(args)
        args.output.mkdir(mode=0o700)
        source.write(args.output / "freeze.json", profile)
    else:
        run(args)


if __name__ == "__main__":
    main()
