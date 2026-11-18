# SPDX-License-Identifier: Apache-2.0
"""Fit and compare contextual reconstruction readouts on exact JPEG regions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from app.bench import image_jpeg_boundary as metrics
from app.bench import image_masked_reconstruction as detector
from scripts.audio_native_alignment_panel import digest, read, save
from scripts.image_jpeg_texture_prediction_panel import _body, _inputs, _reference
from scripts.image_pscc_localization_panel import _area

SCHEMA = "chipmunk.jpeg-masked-reconstruction/1"
HEADS = ("texture_only", "paired_context")
BUDGETS = (0, 1, 2)
FIT_POINTS_PER_EDIT = 96


def method():
    from app.detectors.image import jpeg_boundary_localizer

    return {"publisher_revision": detector.REVISION, "source_files": detector.ASSETS,
            "inference_sha256": digest(Path(detector.__file__)), "script_sha256": digest(Path(__file__)),
            "region_code_sha256": digest(Path(jpeg_boundary_localizer.__file__))}


def split(sources):
    ranked = sorted(sources, key=lambda value: hashlib.sha256(f"masked-jpeg/split/1/{value}".encode()).digest())
    return sorted(ranked[:16]), sorted(ranked[16:])


def fit_samples(model, examples):
    import cv2
    import numpy as np

    from scripts.image_cross_quality_fit import load_mask

    outputs = {name: [] for name in HEADS}
    cache = {}

    def lookup(directory, row, group):
        sha = row["sha256"]
        if sha not in cache:
            maps = model.observe(_body(directory, row))
            expected = tuple((side + detector.CELL - 1) // detector.CELL for side in group["shape"])
            if any(values.shape[-2:] != expected for values in maps):
                raise ValueError("Reconstruction measurements do not align with the native cell grid")
            cache[sha] = tuple(value.reshape(value.shape[0], -1).T for value in maps[:2])
        return cache[sha]

    for directory, changed, group in examples:
        if changed["role"] not in metrics.POSITIVE:
            continue
        truth = load_mask(directory, changed)
        if list(truth.shape) != group["shape"] or int(truth.sum()) != changed["changed_pixels"]:
            raise ValueError("The pixel mask differs from its frozen JPEG delivery")
        n = detector.CELL
        height, width = truth.shape
        margin = ((0, -height % n), (0, -width % n))
        cell_shape = ((height + n - 1) // n, n, (width + n - 1) // n, n)
        counts = np.pad(truth, margin).reshape(cell_shape).sum(axis=(1, 3))
        area = np.pad(np.ones_like(truth), margin).reshape(cell_shape).sum(axis=(1, 3))
        inside = np.flatnonzero(2 * counts >= area)
        outside = np.flatnonzero(cv2.dilate((counts > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
        if not len(inside) or not len(outside):
            raise ValueError("A fitting edit needs measured interior and unchanged neighboring cells")
        seed = int(changed["sha256"][:16], 16)
        rng = np.random.default_rng(seed)
        pos = rng.choice(inside, FIT_POINTS_PER_EDIT, replace=len(inside) < FIT_POINTS_PER_EDIT)
        neg = rng.choice(outside, FIT_POINTS_PER_EDIT, replace=len(outside) < FIT_POINTS_PER_EDIT)
        matched = {row["sha256"]: lookup(bank, row, metadata) for bank, row, metadata in examples
                   if row["quality"] == changed["quality"] and row["role"] not in metrics.POSITIVE}
        if len(matched) < 4:
            raise ValueError("Each edit needs ordinary views of the same source and delivery quality")
        chosen = rng.integers(len(matched), size=FIT_POINTS_PER_EDIT)
        for index, name in enumerate(HEADS):
            examined = lookup(directory, changed, group)[index]
            ordinary = np.stack([sample[index][pos] for sample in matched.values()])[chosen, np.arange(len(pos))]
            outputs[name].append((examined[pos], ordinary, examined[neg]))
    return outputs, len(cache)


def fit_head(examples):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    positive, matched, background = [np.concatenate(items).astype(np.float64) for items in zip(*examples, strict=True)]
    samples = np.concatenate((positive, matched, background))
    center, scale = samples.mean(axis=0), samples.std(axis=0).clip(.01)
    samples = (samples - center) / scale
    count = len(positive)
    weights = np.repeat([.5 / count, .25 / count, .25 / count], count)
    labels = np.r_[np.ones(count), np.zeros(2 * count)]
    deltas = samples[:count] - samples[count:2 * count]

    def loss(params):
        w, bias = params[:-1], params[-1]
        logits = samples @ w + bias
        rank_logits = deltas @ w
        error = weights * (expit(logits) - labels)
        rank_error = .5 * (expit(rank_logits) - 1) / count
        value = weights @ (np.logaddexp(0., logits) - labels * logits)
        value += .5 * np.logaddexp(0., -rank_logits).mean() + .015 * (w @ w)
        grad = np.r_[error @ samples + rank_error @ deltas + .03 * w, error.sum()]
        return value, grad

    fitted = minimize(loss, np.zeros(samples.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 100})
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError("The paired readout failed to converge")
    return {"center": center.tolist(), "scale": scale.tolist(), "weights": fitted.x[:-1].tolist(),
            "bias": float(fitted.x[-1]), "pairs": count, "loss": float(fitted.fun)}


def calibrate(rows):
    points, preferred = {}, {}
    for quality in metrics.QUALITIES:
        bank = [row for row in rows if row["quality"] == quality]
        points[str(quality)] = {}
        choices = []
        for arm in detector.ARMS:
            ordinary = defaultdict(list)
            for row in bank:
                response = row["checks"][arm]["score"]
                if response is not None and row["role"] not in metrics.POSITIVE and not metrics.incumbent_flag(row):
                    ordinary[row["source_group"]].append(response)
            maxima = sorted((max(scores) for scores in ordinary.values()), reverse=True)
            if len(maxima) <= max(BUDGETS):
                raise ValueError("The calibration needs more eligible ordinary photo groups")
            points[str(quality)][arm] = {str(budget): {"point": maxima[budget]} for budget in BUDGETS}
            for budget in BUDGETS:
                point = maxima[budget]
                counts = metrics.measure(bank, arm, point)
                benefit = (2 * counts["added_correct"] + counts["added_feather"]
                           - counts["added_control_groups"] - counts["candidate_wrong_location"])
                choices.append(((benefit, -budget, -counts["added_control"], -detector.ARMS.index(arm)),
                                {"arm": arm, "allowance": budget, "point": point}))
        preferred[str(quality)] = max(choices, key=lambda entry: entry[0])[1]
    return points, preferred


def evaluate(inputs, assets, output, device, *, frozen=None):
    import cv2
    import numpy as np
    import scipy
    import torch
    import transformers
    from PIL import __version__ as pillow

    from scripts.export_image_jpeg_band_review import portable_mask
    from scripts.image_cross_quality_fit import load_mask

    runtime = {"device": device, "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
               "pillow": pillow, "transformers": transformers.__version__, "scipy": scipy.__version__}
    implementation = method()
    receipt = read(frozen) if frozen else None
    if receipt and (receipt["method"] != implementation or receipt["runtime"] != runtime):
        raise ValueError("Held inputs must use the frozen implementation and runtime")
    groups = defaultdict(list)
    reference = {}
    for bank, baseline in inputs:
        reference.update(_reference(baseline))
        for example in _inputs(bank):
            groups[example[2]["rgb_sha256"]].append(example)
    if not groups or receipt and set(groups).intersection(receipt["sources"]):
        raise ValueError("The held comparison requires sources outside fitting and calibration")
    output.mkdir(mode=0o700)
    masks = output / "masks"
    masks.mkdir(mode=0o700)
    model = detector.Reconstruction(assets, device)
    if receipt:
        heads, fit, measured, fit_forwards = receipt["heads"], [], sorted(groups), 0
    else:
        if len(groups) != 24:
            raise ValueError("Development needs sixteen fitting photos and eight separate calibration photos")
        fit, measured = split(groups)
        samples, fit_forwards = defaultdict(list), 0
        for i, source in enumerate(fit, 1):
            found, forwards = fit_samples(model, groups[source])
            fit_forwards += forwards
            for name, rows in found.items():
                samples[name].extend(rows)
            print(f"Fit measurements from {i}/{len(fit)} photos, {fit_forwards} unique forwards", flush=True)
        heads = {name: fit_head(samples[name]) for name in HEADS}

    rows, forwards = [], 0
    for i, source in enumerate(measured, 1):
        cache = {}
        for directory, item, group in groups[source]:
            file, sha = item["file"], item["sha256"]
            truth = load_mask(directory, item)
            indexes = truth.nonzero()
            incumbent = reference[file, sha]
            if (list(truth.shape) != group["shape"] or len(indexes[0]) != item["changed_pixels"]
                    or any(value and value["truth"] != len(indexes[0]) for value in incumbent.values())):
                raise ValueError("The two checks need the same native pixels and changed region")
            if sha not in cache:
                found = detector.proposals(model.observe(_body(directory, item)), heads, truth.shape)
                cache[sha] = {name: (score, mask, portable_mask(mask, masks / f"{sha}-{name}.pbm") if mask.any() else None)
                              for name, (score, mask) in found.items()}
                forwards += 1
            base = {k: item[k] for k in ("file", "sha256", "quality", "role", "mask_sha256")}
            rows.append(base | {"family": group["family"], "source_group": source, "shape": group["shape"], "current": incumbent,
                                "checks": {name: {"score": score, "mask": pin, **_area(mask, indexes)}
                                           for name, (score, mask, pin) in cache[sha].items()}})
        print(f"Measured {i}/{len(measured)} photos, {len(rows)} deliveries, {forwards} unique forwards", flush=True)
    points, preferred = (receipt["points"], receipt["preferred"]) if receipt else calibrate(rows)
    counts = metrics.summarize(rows, points, preferred)
    result = {"schema": SCHEMA, "method": implementation, "runtime": runtime, "heads": heads,
              "sources": sorted(groups), "fit_sources": fit, "measured_sources": measured,
              "inputs": [{"manifest_sha256": digest(bank / "manifest.json"), "reference_sha256": digest(baseline)}
                         for bank, baseline in inputs], "freeze": digest(frozen) if receipt else None,
              "rows": rows, "unique_forwards": forwards, "fit_forwards": fit_forwards,
              "points": points, "preferred": preferred, "outcomes": counts}
    save(output / "result.json.gz", result)
    print(json.dumps({quality: counts[f"{quality}/all"]["selected"] for quality in metrics.QUALITIES}, sort_keys=True), flush=True)


def review(path, frozen=None):
    data = read(path)
    rows = data["rows"]
    seen = sorted({row["source_group"] for row in rows})
    if (data["schema"] != SCHEMA or not rows or data["measured_sources"] != seen
            or sorted(data["fit_sources"] + seen) != data["sources"]
            or data["unique_forwards"] != len({(row["source_group"], row["sha256"]) for row in rows})):
        raise ValueError("Expected complete, source-separated JPEG results")
    for row in rows:
        total = row["shape"][0] * row["shape"][1]
        if set(row["checks"]) != set(detector.ARMS) or row["quality"] not in metrics.QUALITIES:
            raise ValueError("A row is missing part of the comparison")
        for region in row["checks"].values():
            selected, truth = region["selected"], region["truth"]
            if not 0 <= region["intersection"] <= min(selected, truth) <= max(selected, truth) <= total:
                raise ValueError("A mask count falls outside the source extent")
    if frozen:
        review(frozen)
        locked = read(frozen)
        fields = ("method", "runtime", "heads", "points", "preferred")
        if (data["freeze"] != digest(frozen) or any(data[field] != locked[field] for field in fields)
                or set(data["sources"]).intersection(locked["sources"])):
            raise ValueError("The held reading differs from the frozen comparison")
    elif (data["freeze"] is not None or split(data["sources"]) != (data["fit_sources"], seen)
          or calibrate(rows) != (data["points"], data["preferred"])):
        raise ValueError("The development split or operating points do not reproduce")
    result = metrics.summarize(rows, data["points"], data["preferred"])
    if result != data["outcomes"]:
        raise ValueError("The reported outcome counts differ")
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("freeze", "hold"):
        build = commands.add_parser(name)
        build.add_argument("assets", type=Path)
        build.add_argument("output", type=Path)
        build.add_argument("--device", choices=("cpu", "mps"), required=True)
        build.add_argument("--input", action="append", dest="inputs", nargs=2, type=Path, required=True)
        if name == "hold":
            build.add_argument("--frozen", type=Path, required=True)
    compare = commands.add_parser("review")
    compare.add_argument("path", type=Path)
    compare.add_argument("--frozen", type=Path)
    args = vars(cli.parse_args())
    if args.pop("command") == "review":
        print(json.dumps(review(**args), sort_keys=True))
    else:
        evaluate(**args)


if __name__ == "__main__":
    main()
