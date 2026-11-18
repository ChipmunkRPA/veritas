# SPDX-License-Identifier: Apache-2.0
"""Compare a frozen public forgery localizer with current native JPEG checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.bench import image_jpeg_boundary as metrics
from app.bench import image_pscc_localization as detector
from scripts.audio_native_alignment_panel import digest, read, save
from scripts.image_jpeg_texture_prediction_panel import _body, _inputs, _reference

SCHEMA = "chipmunk.jpeg-pscc-localization/1"
BUDGETS = (0, 1, 2)


def method():
    return {"publisher_revision": detector.REVISION, "source_files": detector.SOURCE_FILES,
            "inference_sha256": digest(Path(detector.__file__)), "script_sha256": digest(Path(__file__))}


def rules(rows):
    points, preferred = {}, {}
    for quality in metrics.QUALITIES:
        bank = [row for row in rows if row["quality"] == quality]
        options = points[str(quality)] = {"fixed_mask": {"published_map": {"point": .5}}}
        candidates = []
        for arm in detector.ARMS[1:]:
            grouped = defaultdict(list)
            for row in bank:
                score = row["checks"][arm]["score"]
                if score is not None and row["role"] not in metrics.POSITIVE and not metrics.incumbent_flag(row):
                    grouped[row["source_group"]].append(score)
            maxima = sorted((max(scores) for scores in grouped.values()), reverse=True)
            if len(maxima) <= max(BUDGETS):
                raise ValueError("Insufficient separately sourced ordinary control proposals")
            options[arm] = {}
            for budget in BUDGETS:
                point = maxima[budget]
                options[arm][str(budget)] = {"point": point}
                counts = metrics.measure(bank, arm, point)
                utility = (2 * counts["added_correct"] + counts["added_feather"]
                           - counts["added_control_groups"] - counts["candidate_wrong_location"])
                candidates.append(((utility, -budget, -counts["added_control"], arm),
                                   {"arm": arm, "allowance": budget, "point": point}))
        preferred[str(quality)] = max(candidates, key=lambda entry: entry[0])[1]
    return points, preferred


def report(rows, points, preferred):
    result = metrics.summarize(rows, points, preferred)
    for quality in metrics.QUALITIES:
        positives = [row for row in rows if row["quality"] == quality and row["role"] in metrics.POSITIVE]
        result[f"{quality}/all"]["best_fixed_mask_overlap"] = {
            "cases": len(positives),
            "any_cutoff_useful": sum(any(metrics.useful(region) for region in row["mask_probe"]) for row in positives),
        }
    return result


def _area(mask, indexes):
    import numpy as np

    y, x = np.nonzero(mask)
    selected = int(len(x))
    return {"box": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if selected else None,
            "selected": selected, "intersection": int(np.count_nonzero(mask[indexes])), "truth": len(indexes[0])}


def evaluate(inputs, assets, output, device, *, frozen=None):
    import cv2
    import numpy as np
    import torch
    from PIL import __version__ as pillow

    from scripts.export_image_jpeg_band_review import portable_mask
    from scripts.image_cross_quality_fit import load_mask

    runtime = {"device": device, "torch": torch.__version__, "numpy": np.__version__,
               "opencv": cv2.__version__, "pillow": pillow}
    receipt = read(frozen) if frozen else None
    if receipt and (receipt["method"] != method() or receipt["runtime"] != runtime):
        raise ValueError("Held inputs require the frozen implementation and runtime")
    examples = [example for bank, _ in inputs for example in _inputs(bank)]
    sources = sorted({group["rgb_sha256"] for _, _, group in examples})
    if not sources or receipt and set(sources).intersection(receipt["sources"]):
        raise ValueError("The comparison requires sources disjoint from calibration")
    current = {}
    for _, reference in inputs:
        current.update(_reference(reference))
    if any((item["file"], item["sha256"]) not in current for _, item, _ in examples):
        raise ValueError("Missing an exact current-check comparison")
    output.mkdir(mode=0o700)
    masks = output / "masks"
    masks.mkdir(mode=0o700)
    model = detector.Localizer(assets, device=device)
    remaining = Counter(item["sha256"] for _, item, _ in examples)
    cache, rows, forwards = {}, [], 0
    for directory, item, group in examples:
        file, sha = item["file"], item["sha256"]
        body = _body(directory, item)
        truth = load_mask(directory, item)
        indexes = truth.nonzero()
        incumbent = current[file, sha]
        if list(truth.shape) != group["shape"] or any(
            entry and entry["truth"] != len(indexes[0]) for entry in incumbent.values()
        ):
            raise ValueError("The comparator and candidate need the same exact changed pixels")
        if sha not in cache:
            maps = model.maps(body)
            forwards += 1
            if maps[0].shape != truth.shape:
                raise ValueError("Decoded image and mask use different coordinates")
            geometry = {}
            for arm, (score, area) in detector.selected_regions(maps).items():
                geometry[arm] = (score, area, portable_mask(area, masks / f"{sha}-{arm}.pbm") if area.any() else None)
            cache[sha] = (geometry, maps[0])
        geometry, fine = cache[sha]
        row = {key: item[key] for key in ("file", "sha256", "quality", "role", "mask_sha256")}
        rows.append(row | {
            "family": group["family"], "source_group": group["rgb_sha256"], "shape": group["shape"], "current": incumbent,
            "checks": {arm: {"score": score, "mask": pin, **_area(area, indexes)}
                       for arm, (score, area, pin) in geometry.items()},
            "mask_probe": [_area(fine >= step / 10, indexes) for step in range(1, 10)] if len(indexes[0]) else [],
        })
        remaining[sha] -= 1
        if remaining[sha] == 0:
            del cache[sha]
        if len(rows) % 72 == 0 or len(rows) == len(examples):
            print(f"Measured {len(rows)}/{len(examples)} deliveries, {forwards} unique JPEGs", flush=True)
    points, preferred = (receipt["points"], receipt["preferred"]) if receipt else rules(rows)
    outcomes = report(rows, points, preferred)
    result = {"schema": SCHEMA, "method": method(), "sources": sources, "runtime": runtime,
              "inputs": [{"manifest_sha256": digest(bank / "manifest.json"), "reference_sha256": digest(reference)}
                         for bank, reference in inputs],
              "freeze": digest(frozen) if receipt else None, "rows": rows, "unique_forwards": forwards,
              "points": points, "preferred": preferred, "outcomes": outcomes}
    save(output / "result.json.gz", result)
    print(json.dumps({q: outcomes[f"{q}/all"]["selected"] for q in metrics.QUALITIES}, sort_keys=True), flush=True)


def review(path, frozen=None):
    data = read(path)
    rows = data["rows"]
    if (data["schema"] != SCHEMA or not rows or data["sources"] != sorted({row["source_group"] for row in rows})
            or data["unique_forwards"] != len({row["sha256"] for row in rows})):
        raise ValueError("Expected a complete score ledger")
    for row in rows:
        total = row["shape"][0] * row["shape"][1]
        if row["checks"].keys() != set(detector.ARMS) or row["quality"] not in metrics.QUALITIES:
            raise ValueError("A row lacks part of its predeclared comparison")
        for item in (*row["checks"].values(), *row["mask_probe"]):
            selected, truth = item["selected"], item["truth"]
            if not 0 <= item["intersection"] <= min(selected, truth) <= max(selected, truth) <= total:
                raise ValueError("A mask count lies outside its source extent")
        if len(row["mask_probe"]) != (9 if row["role"] in metrics.POSITIVE else 0):
            raise ValueError("Missing fixed-threshold diagnostic")
    if frozen:
        review(frozen)
        selection = read(frozen)
        if (data["freeze"] != digest(frozen)
                or any(data[key] != selection[key] for key in ("method", "runtime", "points", "preferred"))
                or set(data["sources"]).intersection(selection["sources"])):
            raise ValueError("A held study differs from its freeze or includes calibration photos")
    elif data["freeze"] is not None or rules(rows) != (data["points"], data["preferred"]):
        raise ValueError("The fixed operating points do not reproduce")
    counts = report(rows, data["points"], data["preferred"])
    if counts != data["outcomes"]:
        raise ValueError("Reported outcome counts do not reproduce")
    return counts


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    for name in ("freeze", "hold"):
        command = commands.add_parser(name)
        command.add_argument("assets", type=Path)
        command.add_argument("output", type=Path)
        command.add_argument("--input", action="append", nargs=2, required=True, type=Path, dest="inputs")
        command.add_argument("--device", choices=("cpu", "mps"), default="cpu")
        if name == "hold":
            command.add_argument("--frozen", type=Path, required=True)
    command = commands.add_parser("review")
    command.add_argument("path", type=Path)
    command.add_argument("--frozen", type=Path)
    args = vars(cli.parse_args())
    if args.pop("action") == "review":
        print(json.dumps(review(**args), sort_keys=True))
    else:
        evaluate(**args)


if __name__ == "__main__":
    main()
