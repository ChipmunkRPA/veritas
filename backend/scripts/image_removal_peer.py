# SPDX-License-Identifier: Apache-2.0
"""Test whether similar patches in a JPEG help locate a small removal."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import image_cross_quality_fit as inputs
from scripts import image_region_rank_compare as rank
from scripts import image_region_rank_features as measurements
from scripts.image_region_rank_reference import intersections

SCHEMA = "chipmunk.jpeg-removal-peers/1"
METHODS = ("residual", "with_peers")
FAMILIES = {"big-lama", "lama"}
PEERS = 24
EXCLUSION = 4
POINTS = {"strict": 0, "wider": 2}
DESIGN = {
    "comparison": "Fit texture readings alone and with signed departures from similar, spatially separate patches.",
    "peers": "Match coarse luminance and chroma detail, color and gradient direction. Exclude tiles within four grid cells.",
    "fit": "Train on older removal pairs. Match sharp and feathered edits to the same rectangle in four ordinary "
           "versions and to background rectangles of the same size. Weight source photos equally.",
    "selection": "Use a separate 12-photo archive to select from cutoffs that add up to zero or two ordinary photo groups.",
    "held": "Group results by the input JPEG quality and require thirty percent IoU and half of the changed pixels.",
}


def peer_departure(grid):
    import numpy as np
    from scipy.spatial import cKDTree

    height, width = grid.shape[:2]
    coarse_detail = grid[..., 9:12]
    color_and_direction = grid[..., -5:]
    context = np.concatenate((coarse_detail, color_and_direction), axis=2).reshape(height * width, -1)
    low, high = np.quantile(context, (.25, .75), axis=0)
    # Prevent near-uniform channels from dominating the appearance match.
    scaled = context / np.maximum(high - low, .10)
    count = PEERS + (2 * EXCLUSION + 1) ** 2
    _, indices = cKDTree(scaled).query(scaled, k=count, workers=1)
    positions = np.arange(height * width)
    dy = np.abs(positions[:, None] // width - indices // width)
    dx = np.abs(positions[:, None] % width - indices % width)
    closest = np.argsort(np.maximum(dx, dy) <= EXCLUSION, axis=1, kind="stable")[:, :PEERS]
    peers = np.take_along_axis(indices, closest, axis=1)
    # Compare just the fine and medium residual bands on which we did not match.
    texture = grid[..., :9].reshape(height * width, -1)
    nearby_readings = texture[peers]
    median = np.median(nearby_readings, axis=1)
    spread = 1.4826 * np.median(np.abs(nearby_readings - median[:, None]), axis=1)
    departure = np.clip((texture - median) / np.maximum(spread, .15), -8., 8.)
    return departure.reshape(height, width, -1).astype(np.float32)


def texture_views(data, size):
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as source:
        if source.format != "JPEG" or source.size[::-1] != tuple(size):
            raise ValueError("Delivered JPEG and native truth geometry differ")
        rgb = np.asarray(source.convert("RGB"))
    grid = measurements.residual_grid(rgb)
    departure = peer_departure(grid)
    return {
        "residual": grid,
        "with_peers": np.concatenate((grid, departure, np.abs(departure)), axis=2),
    }


def deliveries(directory, split):
    manifest = rank.read(directory / "manifest.json")
    return [(group, rows) for group, rows in inputs.images(manifest, (split,)) if group["family"] in FAMILIES]


def original(directory, row):
    body = (directory / row["file"]).read_bytes()
    if hashlib.sha256(body).hexdigest() != row["sha256"]:
        raise ValueError("Delivered JPEG bytes changed")
    return body


def train(directory, groups):
    import numpy as np

    pairs = defaultdict(list)
    for group, cases in groups:
        geometry = rank.layout(tuple(group["shape"]))
        boxes = geometry[0]
        dimensions = boxes[:, 2:] - boxes[:, :2]
        rng = np.random.default_rng(int(group["source"][:12], 16))
        for quality in rank.QUALITY:
            grids = {role: texture_views(original(directory, cases[role, quality]), group["shape"])
                     for role in (*rank.POSITIVE, *rank.CONTROLS)}
            for role in rank.POSITIVE:
                truth = inputs.load_mask(directory, cases[role, quality])
                locations, overlap = rank.best_boxes(truth, geometry)
                matched, displaced = [], []
                for index in locations:
                    wrong = np.flatnonzero((overlap <= .10) & np.all(dimensions == dimensions[index], axis=1))
                    if wrong.size:
                        matched.append(index)
                        displaced.append(int(rng.choice(wrong)))
                if not matched:
                    continue
                for method in METHODS:
                    correct = rank.regions(grids[role][method], geometry, matched)
                    other = rank.regions(grids[role][method], geometry, displaced)
                    controls = [rank.regions(grids[control][method], geometry, matched) for control in rank.CONTROLS]
                    for negative in (other, *controls):
                        pairs[quality, method].extend((group["source"], yes, no) for yes, no in zip(correct, negative, strict=True))
    return {str(q): {method: head(pairs[q, method]) for method in METHODS} for q in rank.QUALITY}


def head(pairs):
    import numpy as np

    identities, edited, decoys = zip(*pairs, strict=True)
    groups = Counter(identities)
    weights = np.array([1 / (len(groups) * groups[group]) for group in identities])
    edited, decoys = np.asarray(edited), np.asarray(decoys)
    center = weights @ ((edited + decoys) / 2)
    scale = np.maximum(np.sqrt(weights @ ((edited - center)**2 + (decoys - center)**2) / 2), .001)
    differences = (edited - decoys) / scale
    scatter = (differences.T * weights) @ differences + .30 * np.eye(scale.size)
    coefficients = np.linalg.solve(scatter, weights @ differences)
    return {"center": center.tolist(), "scale": scale.tolist(), "weights": coefficients.tolist(),
            "fitted_sources": len(groups), "pairs": len(pairs)}


def measure(directory, groups, heads, current):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    rows = []
    for group, cases in groups:
        geometry = rank.layout(tuple(group["shape"]))
        for (role, quality), row in cases.items():
            body = original(directory, row)
            truth = inputs.load_mask(directory, row)
            prior = current(body, row, truth)
            grids = texture_views(body, truth.shape)
            checks = {}
            true_positions = None
            if role in rank.POSITIVE:
                boxes, _, cells, _ = geometry
                overlap = rank.window(rank.integral(truth), boxes * rank.CELL)
                true_positions = ((2 * overlap >= int(truth.sum()))
                                  & (10 * overlap >= 3 * (cells * rank.CELL**2 + int(truth.sum()) - overlap)))
            for name, grid in grids.items():
                scores = rank.all_scores(grid, geometry, heads[str(quality)][name])
                best = int(np.argmax(scores))
                x, y, right, bottom = map(int, geometry[0][best] * rank.CELL)
                oracle_rank = None
                if true_positions is not None and true_positions.any():
                    strongest_true = float(scores[true_positions].max())
                    oracle_rank = 1 + int((scores > strongest_true).sum())
                checks[name] = {"score": float(scores[best]), "box": [x, y, right, bottom],
                                "selected": (right - x) * (bottom - y), "intersection": int(truth[y:bottom, x:right].sum()),
                                "truth": int(truth.sum()), "best_possible_rank": oracle_rank}
            rows.append({**row, "source": group["source"], "shape": group["shape"], "checks": checks, "current": prior})
    return rows


def incumbent(journal, *, agreement=None):
    """Read the identical delivered bytes used by each of the current JPEG checks."""
    _, measured = rank.incumbent(journal)
    by_digest = {row["sha256"]: row["agreement"] for row in rank.read(agreement)["rows"]} if agreement else {}
    predict = None

    def current(body, row, truth):
        nonlocal predict
        result = measured[row["file"]]
        if result["sha256"] != row["sha256"]:
            raise ValueError("Current JPEG comparison belongs to a different upload")
        regional = result["regional"]
        if row["quality"] == 90:
            if row["sha256"] in by_digest:
                regional = by_digest[row["sha256"]]
            elif agreement is None:
                if predict is None:
                    predict = quality90_repeat()
                regional = predict(body, truth)
            else:
                raise ValueError("Missing the original JPEG90 comparison")
        return {"jpeg": result["jpeg"], "regional": regional}

    return current


def quality90_repeat():
    import numpy as np
    import torch
    from PIL import Image

    from app.detectors.image import regional
    from app.detectors.image.adapters.regional import evaluated_jpeg
    from scripts import image_regional_jpeg90_quality_agreement as repeat
    from scripts import image_regional_jpeg_transfer as native

    torch.set_num_threads(2)
    model = regional.load_regional(regional.MODEL_ID, "cpu", "float32").obj

    def predict(body, truth):
        if not evaluated_jpeg(body, quality=90):
            return None
        with Image.open(io.BytesIO(body)) as source:
            image = source.convert("RGB")
        rgb = np.asarray(image)
        original_reading = native.native_readout(model.predict(image), body, rgb)
        if not original_reading.regions:
            return intersections(np.zeros_like(truth), truth)
        _, reencoded = repeat.delivered_view(image)
        view_reading = native.native_readout(model.predict(reencoded), body, rgb)
        selected, _ = repeat.agreement(original_reading, view_reading)
        return intersections(native.project(selected, image), truth)

    return predict


def summary(rows, method, cutoff):
    counts = Counter()
    added_originals = set()
    found_sources = set()
    for row in rows:
        candidate = row["checks"][method]
        current_ok = any(map(rank.useful, row["current"].values()))
        current_flag = rank.current_flags(row)
        proposed = candidate["score"] > cutoff
        if row["role"] in rank.POSITIVE:
            correct = rank.useful(candidate)
            new = proposed and correct and not current_ok
            counts.update(edited=1, located_current=current_ok, located_combined=(current_ok or new),
                          added_feathered=new and row["role"] == "feather",
                          misplaced=proposed and not correct, best_region_matches=correct,
                          possible_regions=candidate["best_possible_rank"] is not None)
            if new:
                found_sources.add(row["source"])
        else:
            counts.update(ordinary=1, ordinary_current=current_flag, ordinary_combined=current_flag or proposed)
            if proposed and not current_flag:
                added_originals.add(row["source"])
    counts.update(added_ordinary_sources=len(added_originals), added_edited_sources=len(found_sources))
    return dict(counts)


def operating_points(selection):
    cutoffs, selected = {}, {}
    for quality in rank.QUALITY:
        rows = [row for row in selection if row["quality"] == quality]
        choices = []
        for method in METHODS:
            maximum = {}
            for row in rows:
                if row["role"] in rank.CONTROLS and not rank.current_flags(row):
                    key, score = row["source"], row["checks"][method]["score"]
                    maximum[key] = max(maximum.get(key, score), score)
            ordinary = sorted(maximum.values(), reverse=True)
            for label, allowed in POINTS.items():
                key = f"{quality}/{method}/{label}"
                cutoffs[key] = ordinary[allowed]
                result = summary(rows, method, cutoffs[key])
                choices.append(((-result["added_feathered"], -result["located_combined"],
                                 result["added_ordinary_sources"], result["misplaced"], len(choices)), key))
        selected[str(quality)] = min(choices)[1]
    return cutoffs, selected


def implementation():
    return {Path(path).name: rank.sha(Path(path)) for path in (__file__, rank.__file__, measurements.__file__)}


def fit(retained, earlier, output):
    import cv2

    cv2.setNumThreads(1)
    learning, selection = retained / "deliveries", earlier / "confirmation/deliveries"
    fit_groups = deliveries(learning, "transfer")
    select_groups = deliveries(selection, "fresh")
    fitting_source = [group["source"] for group, _ in fit_groups]
    selection_source = [group["source"] for group, _ in select_groups]
    if len(set(fitting_source + selection_source)) != len(fitting_source) + len(selection_source):
        raise ValueError("A source overlaps fitting and calibration")
    heads = train(learning, fit_groups)
    current = incumbent(earlier / "fresh-reference.jsonl", agreement=earlier / "agreement.json.gz")
    calibrated = measure(selection, select_groups, heads, current)
    cutoffs, selected = operating_points(calibrated)
    frozen = {"schema": SCHEMA, "design": DESIGN, "code": implementation(), "heads": heads, "cutoffs": cutoffs,
              "selected": selected, "fit_sources": fitting_source, "selection_sources": selection_source,
              "training_manifest_sha256": rank.sha(learning / "manifest.json"),
              "selection_manifest_sha256": rank.sha(selection / "manifest.json"), "selection": calibrated}
    rank.save(output, frozen)
    print(json.dumps({key: summary([row for row in calibrated if row["quality"] == int(quality)], key.split("/")[1],
                                   cutoffs[key]) for quality, key in selected.items()}, sort_keys=True), flush=True)


def evaluate(frozen_path, media, journal, output):
    frozen = rank.read(frozen_path)
    if frozen["schema"] != SCHEMA or frozen["design"] != DESIGN or frozen["code"] != implementation():
        raise ValueError("Running method differs from the fixed comparison")
    held = deliveries(media, "fresh")
    seen = frozen["fit_sources"] + frozen["selection_sources"]
    if any(group["source"] in seen for group, _ in held):
        raise ValueError("The held panel contains a fitted source")
    rows = measure(media, held, frozen["heads"], incumbent(journal))
    result = {"frozen": frozen, "held_manifest_sha256": rank.sha(media / "manifest.json"), "held": rows,
              "summary": recount(rows, frozen)}
    verify(result)
    rank.save(output, result)
    print(json.dumps({key: result["summary"][key] for key in frozen["selected"].values()}, sort_keys=True), flush=True)


def recount(held, frozen):
    totals = {}
    for key, value in frozen["cutoffs"].items():
        quality, method, _ = key.split("/")
        rows = [row for row in held if row["quality"] == int(quality)]
        totals[key] = summary(rows, method, value)
        totals[f"{key}/512+"] = summary([row for row in rows if min(row["shape"]) >= 512], method, value)
    return totals


def verify(result):
    frozen = result["frozen"]
    held = result["held"]
    assert frozen["schema"] == SCHEMA and frozen["design"] == DESIGN
    assert (frozen["cutoffs"], frozen["selected"]) == operating_points(frozen["selection"])
    assert not {row["source"] for row in held} & set(frozen["fit_sources"] + frozen["selection_sources"])
    assert len({row["file"] for row in held}) == len(held)
    for row in (*held, *frozen["selection"]):
        assert set(row["checks"]) == set(METHODS)
        for outcome in row["checks"].values():
            x, y, right, bottom = outcome["box"]
            assert 0 <= x < right <= row["shape"][1] and 0 <= y < bottom <= row["shape"][0]
            assert outcome["selected"] == (right - x) * (bottom - y) and outcome["truth"] == row["changed_pixels"]
            assert 0 <= outcome["intersection"] <= min(outcome["selected"], outcome["truth"])
    assert recount(held, frozen) == result["summary"]
    return {"held": len(held), "sources": len({row["source"] for row in held}), "selected": frozen["selected"]}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="stage", required=True)
    for mode, arguments in (
        ("fit", ("retained", "earlier", "output")),
        ("evaluate", ("freeze", "deliveries", "journal", "output")),
        ("verify", ("receipt",)),
    ):
        parser = commands.add_parser(mode)
        for argument in arguments:
            parser.add_argument(argument, type=Path)
    args = cli.parse_args()
    if args.stage == "fit":
        fit(args.retained, args.earlier, args.output)
    elif args.stage == "evaluate":
        evaluate(args.freeze, args.deliveries, args.journal, args.output)
    else:
        print(json.dumps(verify(rank.read(args.receipt)), sort_keys=True))
