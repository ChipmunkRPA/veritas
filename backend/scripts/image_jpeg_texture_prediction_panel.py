# SPDX-License-Identifier: Apache-2.0
"""Fit an intact-photo JPEG texture predictor and assess exact edited regions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from app.bench import image_jpeg_boundary as metrics
from scripts.audio_native_alignment_panel import digest, read, save

SCHEMA = "chipmunk.jpeg-texture-prediction-panel/1"
ARMS = ("texture_deficit", "texture_surprise", "neighbor_deficit", "joint_deficit")
QUALITIES = (75, 90)
BUDGETS = (0, 1, 2)
SEED = 20261117
PATCH = 30
EPOCHS = 16
BATCH = 12


def _inputs(directory):
    source = read(directory / "manifest.json")
    groups = {row["tag"]: row for row in source["groups"]}
    for row in source["images"]:
        group = groups[row["group"]]
        yield directory, row, group


def _body(directory, row):
    body = (directory / row["file"]).read_bytes()
    if hashlib.sha256(body).hexdigest() != row["sha256"]:
        raise ValueError("The JPEG differs from its frozen delivery")
    return body


def _runtimes():
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    return torch


def _train(rows, output):
    import numpy as np
    from safetensors.torch import save_file

    from app.bench import image_jpeg_texture_prediction as detector

    torch = _runtimes()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    examples = [detector.measure(_body(directory, item)) for directory, item, _ in rows]
    network = detector.model()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.0025, weight_decay=.006)
    losses = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(examples) * 5)
        total, count = 0., 0
        for start in range(0, len(order), BATCH):
            fields = []
            for index in order[start:start + BATCH]:
                sample = examples[index % len(examples)]
                height, width = sample[0].shape[1:]
                y = int(rng.integers(0, height - PATCH + 1))
                x = int(rng.integers(0, width - PATCH + 1))
                fields.append([values[:, y:y + PATCH, x:x + PATCH] for values in sample])
            coarse, ring, target = (torch.from_numpy(np.stack(values)) for values in zip(*fields, strict=True))
            target = target[..., 3:-3, 3:-3]
            error = []
            for branch in network(coarse, ring):
                estimate, log_scale = branch[..., 3:-3, 3:-3].chunk(2, dim=1)
                scale = log_scale.clamp(-2.8, 1.4)
                displacement = (estimate - target) * torch.exp(-scale)
                error.append((scale + 2 * torch.log1p(displacement.square() / 3)).mean())
            loss = torch.stack(error).mean()
            if not torch.isfinite(loss):
                raise ValueError("Intact-photo prediction became nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            count += 1
        losses.append(round(total / count, 7))
        if epoch % 4 == 3:
            print(f"Texture fit epoch {epoch + 1}/{EPOCHS}, loss {losses[-1]:.6f}", flush=True)
    save_file(network.state_dict(), output / "texture.safetensors")
    return network.eval(), losses


def _reference(path):
    if path.suffix == ".jsonl":
        with path.open() as stream:
            next(stream)
            rows = [json.loads(line) for line in stream]
        return {(row["file"], row["sha256"]): {k: row[k] for k in ("jpeg", "regional")} for row in rows}
    holdout = read(path)
    return {(row["file"], row["sha256"]): row["current"] for row in holdout["rows"]}


def _geometry(mask, truth):
    import numpy as np

    from app.detectors.image.jpeg_band_localizer import CELL

    area = np.repeat(np.repeat(mask, CELL, 0), CELL, 1)
    height, width = area.shape
    yy, xx = np.nonzero(area)
    return {"box": [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)] if len(xx) else None,
            "selected": int(area.sum()), "intersection": int(np.count_nonzero(area & truth[:height, :width])),
            "truth": int(np.count_nonzero(truth))}


def _measure(rows, network, comparator):
    from app.bench import image_jpeg_texture_prediction as detector
    from scripts.image_cross_quality_fit import load_mask

    torch = _runtimes()
    cache = {}
    results = []
    masks = {}
    with torch.inference_mode():
        for directory, item, group in rows:
            file, sha = item["file"], item["sha256"]
            baseline = comparator[file, sha]
            if sha not in cache:
                low, context, target = detector.measure(_body(directory, item))
                output = network(torch.from_numpy(low[None]), torch.from_numpy(context[None]))
                maps = detector.errors(*(values.numpy()[0] for values in output), target)
                cache[sha] = {arm: detector.locate(hot) for arm, hot in maps.items()}
            mask_key = (directory, item["mask"], item["mask_sha256"])
            if mask_key not in masks:
                masks[mask_key] = load_mask(directory, item)
            truth = masks[mask_key]
            if tuple(group["shape"]) != truth.shape or int(truth.sum()) != item["changed_pixels"]:
                raise ValueError("The observation and exact edit mask differ in source geometry")
            if any(item and item["truth"] != int(truth.sum()) for item in baseline.values()):
                raise ValueError("The baseline and new reading use different exact changed pixels")
            scores = {arm: {"score": candidate[0], **_geometry(candidate[1], truth)}
                      for arm, candidate in cache[sha].items()}
            results.append({k: item[k] for k in ("file", "sha256", "quality", "role", "mask_sha256")}
                           | {"family": group["family"], "source_group": group["rgb_sha256"],
                              "shape": group["shape"], "current": baseline, "checks": scores})
            if len(results) % 72 == 0:
                print(f"Measured {len(results)}/{len(rows)} deliveries", flush=True)
    return results


def _selection(rows):
    choices = {}
    for quality in QUALITIES:
        same = [row for row in rows if row["quality"] == quality]
        options = {}
        for arm in ARMS:
            groups = defaultdict(list)
            for row in same:
                result = row["checks"][arm]["score"]
                if row["role"] not in metrics.POSITIVE and not metrics.incumbent_flag(row) and result is not None:
                    groups[row["source_group"]].append(result)
            ordered = sorted((max(scores) for scores in groups.values()), reverse=True)
            if len(ordered) <= max(BUDGETS):
                raise ValueError("Too few separate original groups for an operating point")
            for budget in BUDGETS:
                point = ordered[budget]
                measured = metrics.measure(same, arm, point)
                utility = (2 * measured["added_correct"] + measured["added_feather"]
                           - measured["added_control_groups"] - measured["candidate_wrong_location"])
                index = f"{arm}/{budget}"
                options[index] = {"arm": arm, "allowance": budget, "point": point, "counts": measured, "utility": utility}
        preferred = max(options, key=lambda k: (options[k]["utility"], -options[k]["allowance"], k))
        choices[str(quality)] = {"preferred": preferred, "options": options}
    return choices


def _method():
    from app.bench import image_jpeg_texture_prediction as detector

    return {"measurements": digest(Path(detector.__file__)), "script": digest(Path(__file__))}


def fit(bank, ordinary_fills, benchmark, baseline_fills, output):
    all_rows = list(_inputs(bank))
    sources = {g["rgb_sha256"] for _, _, g in all_rows}
    if len(sources) < 22:
        raise ValueError("Need at least 22 ready source photos")
    ordered = sorted(sources, key=lambda key: hashlib.sha256(f"{SCHEMA}/{key}".encode()).hexdigest())
    selection = set(ordered[:8])
    fitting = set(ordered[8:])
    train_rows = [(d, item, g) for d, item, g in all_rows
                  if g["rgb_sha256"] in fitting and item["role"] == "ordinary" and g["family"] == "lama"]
    if len(train_rows) != 2 * len(fitting):
        raise ValueError("Intact fitting needs one JPEG per quality and source")
    calibration = [(d, item, g) for d, item, g in (*all_rows, *_inputs(ordinary_fills)) if g["rgb_sha256"] in selection]
    output.mkdir(mode=0o700)
    network, losses = _train(train_rows, output)
    baseline = _reference(benchmark) | _reference(baseline_fills)
    outcomes = _measure(calibration, network, baseline)
    chosen = _selection(outcomes)
    receipt = {"schema": SCHEMA, "code": _method(), "weight_sha256": digest(output / "texture.safetensors"),
               "fit_sources": sorted(fitting), "selection_sources": sorted(selection), "losses": losses,
               "training_files": sorted(row["sha256"] for _, row, _ in train_rows),
               "input_manifests": [digest(p / "manifest.json") for p in (bank, ordinary_fills)],
               "choices": chosen, "rows": outcomes}
    save(output / "freeze.json.gz", receipt)
    print(json.dumps({q: data["preferred"] for q, data in chosen.items()}, sort_keys=True), flush=True)


def _outcomes(rows, choices):
    summary = {}
    for quality in QUALITIES:
        relevant = [row for row in rows if row["quality"] == quality]
        if not relevant:
            raise ValueError("A confirmation JPEG quality is missing")
        summary[str(quality)] = {key: metrics.measure(relevant, spec["arm"], spec["point"])
                                 for key, spec in choices[str(quality)]["options"].items()}
    return summary


def hold(bank, ordinary_fills, baseline, baseline_fills, frozen, output):
    from safetensors.torch import load_file

    from app.bench import image_jpeg_texture_prediction as detector

    freeze = read(frozen / "freeze.json.gz")
    weights = frozen / "texture.safetensors"
    if freeze["code"] != _method() or freeze["weight_sha256"] != digest(weights):
        raise ValueError("Confirmation must use the frozen model and choices")
    examples = [*_inputs(bank), *_inputs(ordinary_fills)]
    groups = {group["rgb_sha256"] for _, _, group in examples}
    if groups.intersection(freeze["fit_sources"] + freeze["selection_sources"]):
        raise ValueError("Held photo matches a fit or selection group")
    references = _reference(baseline)
    if baseline_fills:
        references.update(_reference(baseline_fills))
    network = detector.model()
    network.load_state_dict(load_file(weights), strict=True)
    rows = _measure(examples, network.eval(), references)
    counts = _outcomes(rows, freeze["choices"])
    save(output, {"schema": SCHEMA, "freeze_sha256": digest(frozen / "freeze.json.gz"), "held_sources": sorted(groups),
                  "input_manifests": [digest(p / "manifest.json") for p in (bank, ordinary_fills)],
                  "baseline_sha256": [digest(p) for p in (baseline, baseline_fills) if p],
                  "rows": rows, "counts": counts})
    print(json.dumps({q: out[freeze["choices"][q]["preferred"]] for q, out in counts.items()}, sort_keys=True), flush=True)


def _validate_rows(rows, expected):
    for row in rows:
        height, width = row["shape"]
        if row["source_group"] not in expected or row["quality"] not in QUALITIES or set(row["checks"]) != set(ARMS):
            raise ValueError("A recorded candidate lies outside the frozen photo or codec scope")
        for reading in row["checks"].values():
            selected, actual, intersect = (reading[key] for key in ("selected", "truth", "intersection"))
            if not 0 <= intersect <= min(selected, actual) <= max(selected, actual) <= height * width:
                raise ValueError("A recorded intersection has impossible area")
            bounds = reading["box"]
            if bounds:
                x, y, right, bottom = bounds
                if not (0 <= x < right <= width and 0 <= y < bottom <= height and 0 < selected <= (right - x) * (bottom - y)):
                    raise ValueError("A recorded JPEG region has invalid native bounds")
            if bool(bounds) != bool(selected) or (reading["score"] is not None) != bool(selected):
                raise ValueError("Empty regions cannot carry local texture scores")


def review(freeze, held):
    fitted, confirmation = read(freeze), read(held)
    groups = set(confirmation["held_sources"])
    if (fitted["schema"] != SCHEMA or confirmation["schema"] != SCHEMA or confirmation["freeze_sha256"] != digest(freeze)
            or groups.intersection(fitted["fit_sources"] + fitted["selection_sources"])):
        raise ValueError("The published point and held roster are not source-disjoint")
    _validate_rows(fitted["rows"], set(fitted["selection_sources"]))
    _validate_rows(confirmation["rows"], groups)
    if fitted["choices"] != _selection(fitted["rows"]) or confirmation["counts"] != _outcomes(confirmation["rows"], fitted["choices"]):
        raise ValueError("The published point or held counts could not be reproduced")
    print(json.dumps(confirmation["counts"], sort_keys=True))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    for name, positionals in (
        ("fit", ("bank", "ordinary_fills", "benchmark", "baseline_fills", "output")),
        ("hold", ("bank", "ordinary_fills", "baseline", "frozen", "output")),
        ("review", ("freeze", "held")),
    ):
        sub = commands.add_parser(name)
        for field in positionals:
            sub.add_argument(field, type=Path)
        if name == "hold":
            sub.add_argument("--baseline-fills", type=Path)
    args = vars(cli.parse_args())
    globals()[args.pop("action")](**args)
