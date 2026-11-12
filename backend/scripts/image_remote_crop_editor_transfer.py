# SPDX-License-Identifier: Apache-2.0
"""Replay the already fixed remote-crop rule on additional source/edit groups."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import image_crop_context_transfer as crops
from scripts import image_crop_overlap_transfer as inference
from scripts import image_crop_remote_context as method
from scripts import image_edge_spread_transfer as panel
from scripts import image_relative_patch_transfer as source
from scripts.review_image_residual_sign_transfer import hits, selected_mask

SCHEMA = "chipmunk.image-remote-crop-editor-transfer/1"
ARMS = ("relative_only", "remote_control")
POINTS_SHA = "7de6ed718c1819b4d027640053244e834b09dc1a62ea74e44e3704335985ca85"


def freeze(args):
    old = panel.read(args.prior / "result.json")
    journal = panel.sha(args.prior / "held.jsonl")
    panel.require(journal == old["held_sha256"] and old["held_views"] == 384, "Earlier exact crop receipts differ")
    panel.require(panel.sha(args.points) == POINTS_SHA and panel.sha(args.checkpoint) == source.WEIGHT_SHA,
                  "The historical point or pinned MIT checkpoint differs")
    points = panel.read(args.points)
    panel.require(points["relative"] == method.RELATIVE_POINT, "Relative point changed")
    folders = {name: getattr(args, name) for name in ("earlier", "later", "fresh")}
    return {
        "schema": SCHEMA, "prior_rows": journal, "points_sha256": POINTS_SHA,
        "weight_sha256": source.WEIGHT_SHA, "points": points,
        "intakes": {label: {name: panel.sha(folder / name) for name in ("plan.json", "result.json", "attempts.jsonl")}
                    for label, folder in folders.items()},
        "code": {Path(module.__file__).name: panel.sha(Path(module.__file__)) for module in (crops, inference, method, panel)}
                | {Path(__file__).name: panel.sha(Path(__file__))},
        "rules": "Preserve the pre-existing native SD1.5 points unchanged. Treat JPEG as a transport stress test. "
                 "Compare added and lost regions against the relative-only arm on matched pictures.",
    }


def inspect(logits, boxes, points):
    box, relative, median = method.readout(boxes, logits)
    relative_flag = relative >= points["relative"]
    return {"box": list(box), "relative": relative, "remote_median": median,
            "relative_only": relative_flag, "remote_control": relative_flag and median < points["remote_upper_strict"]}


def measurement(shape, truth, box, active):
    mask = selected_mask(shape, {"box": box}, active)
    return {"reported": int(mask.sum()), "target": int(truth.sum()), "overlap": int(np.count_nonzero(mask & truth))}


def observe(label, folder, lock, *, cached=None, model=None):
    for case, native, known in panel.paired.inputs(folder):
        for role, pixels in panel.native_variants(native, case["box"]).items():
            truth = np.any(pixels != native["original"], axis=2) if role in panel.POSITIVE else np.zeros_like(known)
            truth_sha = source.digest(truth.tobytes())
            for fmt in panel.FORMATS:
                delivered, image = panel.paired.encoded(pixels, fmt)
                key = (label, case["slot"], role, fmt)
                digest = source.digest(delivered)
                boxes, inputs = zip(*crops.crops(image), strict=True)
                bindings = [inference.digest_grid(patch) for patch in inputs]
                if cached is not None:
                    row = cached.pop(key)
                    panel.require((digest, truth_sha, bindings) == (row["input_sha256"], row["truth_sha256"], row["crop_sha256"]),
                                  "A re-used score has a different input")
                    logits = row["logits"]
                else:
                    logits, actual = model(image)
                    panel.require(bindings == actual, "Prepared model inputs differ")
                panel.require(len(logits) == 25 and all(math.isfinite(x) for x in logits), "Missing or invalid crop logits")
                response = inspect(logits, boxes, lock["points"])
                yield {"key": list(key), "editor": case["generator"], "input_sha256": digest, "truth_sha256": truth_sha,
                       "crop_sha256": bindings, "logits": logits, "response": response,
                       "measurements": {arm: measurement(truth.shape, truth, response["box"], response[arm]) for arm in ARMS}}


def summary(rows):
    results = {}
    for fmt in panel.FORMATS:
        results[fmt] = {}
        for arm in ARMS:
            counts = Counter(dict.fromkeys(("sharp", "feathered", "ordinary", "original", "processed", "views"), 0))
            for row in rows:
                _, _, role, delivery = row["key"]
                if delivery != fmt:
                    continue
                counts["views"] += 1
                measured = row["measurements"][arm]
                if role in panel.POSITIVE:
                    counts["sharp" if role == "edited" else "feathered"] += hits(measured)
                elif measured["reported"]:
                    counts["ordinary"] += 1
                    counts["original"] += role == "original"
                    counts["processed"] += role in ("local_processing", "global_processing", "feathered_processing")
            results[fmt][arm] = dict(counts)
    return results


def reverify(args, lock, summary_result):
    total = 0
    for label in ("earlier", "later", "fresh"):
        path = args.output / f"{label}-rows.json"
        rows = panel.read(path)
        panel.require(panel.sha(path) == summary_result["journals"][label], "Saved crop journal changed")
        replay = {}
        for row in rows:
            key = tuple(row["key"])
            panel.require(key not in replay, "Duplicate observed view")
            replay[key] = row
        for case, images, known in panel.paired.inputs(getattr(args, label)):
            for role, pixels in panel.native_variants(images, case["box"]).items():
                truth = np.any(pixels != images["original"], axis=2) if role in panel.POSITIVE else np.zeros_like(known)
                for fmt in panel.FORMATS:
                    row = replay.pop((label, case["slot"], role, fmt))
                    encoded, image = panel.paired.encoded(pixels, fmt)
                    boxes, inputs = zip(*crops.crops(image), strict=True)
                    panel.require(row["input_sha256"] == source.digest(encoded) and row["truth_sha256"] == source.digest(truth.tobytes())
                                  and row["crop_sha256"] == [inference.digest_grid(patch) for patch in inputs],
                                  "Independently rebuilt delivery or crop bytes differ")
                    a = row["logits"]
                    i = max(range(len(a)), key=a.__getitem__)
                    relative = float(a[i] - np.median(a))
                    x, y, right, bottom = boxes[i]
                    others = [value for (x1, y1, right1, bottom1), value in zip(boxes, a, strict=True)
                              if min(right, right1) <= max(x, x1) or min(bottom, bottom1) <= max(y, y1)]
                    remote = float(np.median(others))
                    flags = (relative >= lock["points"]["relative"],
                             relative >= lock["points"]["relative"] and remote < lock["points"]["remote_upper_strict"])
                    panel.require(row["response"] == {"box": [x, y, right, bottom], "relative": relative, "remote_median": remote,
                                                      **dict(zip(ARMS, flags, strict=True))},
                                  "Independent crop rule differs")
                    for arm, flag in zip(ARMS, flags, strict=True):
                        overlap = int(truth[y:bottom, x:right].sum()) if flag else 0
                        reported = (right - x) * (bottom - y) if flag else 0
                        panel.require(row["measurements"][arm] == {"reported": reported, "target": int(truth.sum()),
                                                                  "overlap": overlap}, "Source region overlap differs")
                    total += 1
        panel.require(not replay and summary(rows) == summary_result["results"][label], "Incomplete held or aggregate replay")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "verify"))
    for field in ("checkpoint", "points", "prior", "earlier", "later", "fresh", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    lock = freeze(args)
    path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        panel.write(path, lock)
        return
    panel.require(panel.read(path) == lock, "Frozen transfer definition differs")
    result_path = args.output / "result.json"
    if args.stage == "verify":
        result = panel.read(result_path)
        panel.require(result["freeze_sha256"] == panel.sha(path), "Frozen method mismatch")
        total = reverify(args, lock, result)
        panel.write(args.output / "audit.json", {"result_sha256": panel.sha(result_path), "views": total,
                    "crop_bindings": total * 25, "region_counts": total * 2})
        print(total, "views verified", flush=True)
        return
    start = time.monotonic()
    with (args.prior / "held.jsonl").open() as journal:
        old = [json.loads(line) for line in journal]
    previous = {(row["cohort"], row["slot"], row["role"], row["format"]): row for row in old}
    panel.require(len(previous) == len(old) == 384, "Incomplete earlier result binding")
    counts, digests = {}, {}
    for label in ("earlier", "later", "fresh"):
        if label == "fresh":
            model, calls = inference.extractor(args.checkpoint)
            rows = list(observe(label, args.fresh, lock, model=model))
        else:
            rows = list(observe(label, getattr(args, label), lock, cached=previous))
        panel.require(len(rows) == (112 if label == "fresh" else 192), "Missing held views")
        journal = args.output / f"{label}-rows.json"
        panel.write(journal, rows)
        digests[label], counts[label] = panel.sha(journal), summary(rows)
        print(label, len(rows), "views", flush=True)
    panel.require(not previous, "An earlier view did not replay")
    panel.write(result_path, {"freeze_sha256": panel.sha(path), "journals": digests, "results": counts,
                "prior_cached_views": 384, "fresh_views": 112, "fresh_batches": calls,
                "elapsed_s": round(time.monotonic() - start, 3)})


if __name__ == "__main__":
    main()
