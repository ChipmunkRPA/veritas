# SPDX-License-Identifier: Apache-2.0
"""Fit and compare dense JPEG edit maps with and without matched-encode consistency."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import image_region_rank_compare as comparison
from scripts import image_region_rank_inputs as panel

SCHEMA = "chipmunk.image-cross-quality-comparison/1"
FIT = ("fit", "held")
ARMS = ("separate", "pooled", "matched_encode")
RIDGE = .035
CONSISTENCY = .5
POINTS = {"strict": 0, "wider": 2}
METHOD = {
    "measure": "Read dequantized luma magnitude, occupancy and spread at 20 AC frequencies plus decoded YCrCb residual bands. "
               "Use 16-pixel cells, the adjacent neighborhood and its difference from a wider ring.",
    "fit": "Supervise exact changed-pixel cells, hard negative surroundings and the same cell positions in four ordinary controls. "
           "Both JPEG deliveries of each case contribute equally. Within each image assign equal weight to both sampled classes.",
    "arms": "Fit independent quality slopes, a shared slope, and the same shared fit with a penalty on response differences "
            "between exactly corresponding cells of the two encodes. Use proportional sharing across original source groups.",
    "selection": "Keep the strongest connected component of the native response field. Set each arm and carrier's two points "
                 "on ordinary source-group maxima where the current checks did not already select a region. Choose on added "
                 "useful regions in the retained calibration partition, with feathered edits first.",
    "match": "Require 0.30 intersection over union and 0.50 recall of exact before-encoding changed pixels.",
}


def images(manifest, roles):
    groups = {row["tag"]: row for row in manifest["groups"] if row["split"] in roles}
    by_group = defaultdict(dict)
    for delivery in manifest["images"]:
        if delivery["group"] in groups:
            key = delivery["role"], delivery["quality"]
            by_group[delivery["group"]][key] = delivery
    for key, compiled in by_group.items():
        assert set(compiled) == {(role, q) for role in (*panel.POSITIVE, *panel.CONTROLS) for q in panel.QUALITY}
        yield groups[key], compiled


def load_mask(media, row):
    import numpy as np
    from PIL import Image

    path = media / row["mask"]
    if panel.sha(path) != row["mask_sha256"]:
        raise ValueError("Changed native mask commitment")
    with Image.open(path) as source:
        mask = np.asarray(source.convert("L")) > 0
    if int(mask.sum()) != row["changed_pixels"]:
        raise ValueError("Native mask extent differs")
    return mask


def read_features(media, row):
    from app.detectors.image.jpeg_cross_quality import features

    body = (media / row["file"]).read_bytes()
    if hashlib.sha256(body).hexdigest() != row["sha256"]:
        raise ValueError("Changed delivered JPEG")
    return features(body)


def downsample(mask, shape):
    from app.detectors.image.jpeg_cross_quality import CELL

    rows, cols = shape
    return mask[:rows*CELL, :cols*CELL].reshape(rows, CELL, cols, CELL).mean(axis=(1, 3), dtype="float32")


def sample(fraction, support, *, edit, random):
    import cv2
    import numpy as np

    near = cv2.dilate((support > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    near &= fraction == 0
    positive = np.flatnonzero(fraction > 0)
    focused = np.flatnonzero(near if edit else support > 0)
    background = np.flatnonzero((fraction == 0) & (support == 0) & ~near)

    def choose(positions, limit):
        return random.choice(positions, min(len(positions), limit), replace=False) if len(positions) else positions

    selected = [choose(positive, 256), choose(focused, 256), choose(background, 256)]
    mass = (.50, .35, .15) if edit else (0., .65, .35)
    if not len(selected[2]):
        background = np.flatnonzero((fraction == 0) & (support == 0))
        selected[2] = choose(background, 256)
    if edit and not len(selected[0]):
        raise ValueError("An annotated fitting edit has no native supporting cell")
    locations = np.concatenate(selected)
    labels = np.r_[.4 + .6 * fraction.ravel()[selected[0]], np.zeros(sum(map(len, selected[1:])))]
    weights = np.concatenate([np.full(len(part), portion / max(len(part), 1)) for part, portion in zip(selected, mass, strict=True)])
    weights /= weights.sum()
    return locations, labels, weights


def examples(media, manifest):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    a, b, labels, weights, checked = [], [], [], [], []
    for group, cases in images(manifest, FIT):
        identity = int(group["rgb_sha256"][:16], 16)
        random = np.random.default_rng(identity)
        support_mask = load_mask(media, cases["sharp", 75])
        for role in (*panel.POSITIVE, *panel.CONTROLS):
            first, second = (read_features(media, cases[role, q]) for q in panel.QUALITY)
            shape = first.shape
            assert second.shape == shape and cases[role, 75]["mask_sha256"] == cases[role, 90]["mask_sha256"]
            if role in panel.POSITIVE:
                mask = load_mask(media, cases[role, 75])
            else:
                mask = np.zeros_like(support_mask)
            take, values, importance = sample(downsample(mask, shape[:2]), downsample(support_mask, shape[:2]),
                                               edit=role in panel.POSITIVE, random=random)
            a.append(first.reshape(-1, shape[-1])[take])
            b.append(second.reshape(-1, shape[-1])[take])
            labels.append(values)
            # Give each known edit as much mass as two matched controls.
            weights.append(importance * (2. if role in panel.POSITIVE else 1.))
        checked.append(group["rgb_sha256"])
        if len(checked) % 12 == 0:
            print(f"Measured the fitting cell pairs from {len(checked)} photos", flush=True)
    return np.concatenate(a), np.concatenate(b), np.concatenate(labels), np.concatenate(weights), checked


def heads(a, b, labels, weights):
    import numpy as np

    weights /= weights.sum()
    target = labels - weights @ labels

    def fit(first, second=None, *, paired=False):
        # Fit the same number of parameters and use the same objectives for
        # both shared arms, except for the paired consistency penalty.
        views = (first,) if second is None else (first, second)
        centers = [weights @ view for view in views]
        center = np.mean(centers, axis=0)
        scale = np.maximum(.025, np.sqrt(np.mean([weights @ (view - center)**2 for view in views], axis=0)))
        importance = np.sqrt(weights[:, None])
        observed = [((view - center) / scale) * importance for view in views]
        covariance = sum(view.T @ view for view in observed) / len(views)
        covariance += RIDGE * np.eye(len(center))
        if paired:
            diff = observed[0] - observed[1]
            covariance += CONSISTENCY * (diff.T @ diff)
        response = sum(view.T @ (target * importance[:, 0]) for view in observed) / len(views)
        reduced = np.linalg.solve(covariance, response)
        return {"center": center.tolist(), "inverse_scale": (1 / scale).tolist(), "coefficients": reduced.tolist(),
                "intercept": float(weights @ labels)}

    return {"separate": {"75": fit(a), "90": fit(b)}, "pooled": fit(a, b), "matched_encode": fit(a, b, paired=True)}


def location(grid_mask, truth):
    import numpy as np

    from app.detectors.image.jpeg_cross_quality import CELL

    measured = np.repeat(np.repeat(grid_mask, CELL, axis=0), CELL, axis=1)
    bounds = measured.shape
    yy, xx = np.where(measured)
    box = [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1] if yy.size else None
    return {"box": box, "pixel_sha256": hashlib.sha256(measured.tobytes()).hexdigest(), "selected": int(measured.sum()),
            "intersection": int((measured & truth[:bounds[0], :bounds[1]]).sum()), "truth": int(truth.sum())}


def score(media, manifest, roles, models, baseline):
    import cv2

    from app.detectors.image.jpeg_cross_quality import region

    cv2.setNumThreads(1)
    output = []
    for group, cases in images(manifest, roles):
        for row in cases.values():
            q = row["quality"]
            supplied = baseline[row["file"]]
            assert supplied["sha256"] == row["sha256"]
            grid = read_features(media, row)
            truth = load_mask(media, row)
            readings = {}
            for arm in ARMS:
                head = models[arm][str(q)] if arm == "separate" else models[arm]
                value, mask = region(grid, head)
                readings[arm] = {"score": value} | location(mask, truth)
            output.append({key: row[key] for key in ("file", "sha256", "quality", "role", "group", "mask_sha256")} |
                          {"split": group["split"], "family": group["family"], "shape": group["shape"],
                           "source_group": group["rgb_sha256"], "current": {k: supplied[k] for k in ("jpeg", "regional")},
                           "checks": readings})
        if len(output) % 144 == 0:
            print(f"Compared {len(output)} input deliveries in {', '.join(roles)}", flush=True)
    return output


def allowed(row, arm, point):
    return row["checks"][arm]["selected"] > 0 and row["checks"][arm]["score"] > point


def choose(calibration):
    levels = {}
    for q in panel.QUALITY:
        rows = [row for row in calibration if row["quality"] == q]
        for arm in ARMS:
            groups = defaultdict(float)
            for row in rows:
                if row["role"] in panel.CONTROLS and not comparison.current_flags(row):
                    key = row["source_group"]
                    groups[key] = max(groups[key], row["checks"][arm]["score"])
            assert len(groups) >= 16
            ranked = sorted(groups.values(), reverse=True)
            levels[f"{q}/{arm}"] = {name: ranked[count] for name, count in POINTS.items()}

    def order(arm):
        selected = [row for row in calibration if row["role"] in panel.POSITIVE and allowed(
            row, arm, levels[f"{row['quality']}/{arm}"]["wider"]) and not comparison.current_flags(row)]
        useful = [row for row in selected if comparison.useful(row["checks"][arm])]
        return -sum(row["role"] == "feather" for row in useful), -len(useful), len(selected) - len(useful), ARMS.index(arm)

    return min(ARMS, key=order), levels


def summarize(rows, points):
    totals, groups = defaultdict(Counter), defaultdict(set)
    for row in rows:
        current = comparison.current_flags(row)
        useful = any(map(comparison.useful, row["current"].values()))
        positive = row["role"] in panel.POSITIVE
        for arm in ARMS:
            result = row["checks"][arm]
            localized = comparison.useful(result)
            for kind, point in points[f"{row['quality']}/{arm}"].items():
                key = f"{row['split']}/{row['quality']}/{arm}/{kind}"
                winner = not current and allowed(row, arm, point)
                totals[key].update(edited=positive, ordinary=not positive, current_useful=positive and useful,
                                   candidate_useful=positive and localized and allowed(row, arm, point),
                                   candidate_ordinary=not positive and allowed(row, arm, point),
                                   added_useful=positive and winner and localized, top_useful=positive and localized,
                                   added_misplaced=positive and winner and not localized, current_ordinary=not positive and current,
                                   added_ordinary=not positive and winner)
                if winner and not positive:
                    groups[key].add(row["source_group"])
    return {key: dict(tally) | {"added_ordinary_groups": len(groups[key])} for key, tally in sorted(totals.items())}


def implementation():
    model = Path(__file__).resolve().parents[1] / "app/detectors/image/jpeg_cross_quality.py"
    return {"features": panel.sha(model), "comparison": panel.sha(Path(__file__))}


def fit(retained, output):
    media = retained / "deliveries"
    manifest_path = media / "manifest.json"
    manifest = panel.read(manifest_path)
    a, b, targets, weights, used = examples(media, manifest)
    models = heads(a, b, targets, weights)
    del a, b, targets, weights
    _, reference = comparison.incumbent(retained / "reference.jsonl")
    calibrated = score(media, manifest, ("calibration",), models, reference)
    chosen, levels = choose(calibrated)
    roles = {"fit": used, "calibration": sorted({row["source_group"] for row in calibrated})}
    assert len(set(used)) == len(used) and not set(used).intersection(roles["calibration"])
    plan = {"schema": SCHEMA, "method": METHOD, "code": implementation(), "roles": roles,
            "manifest_sha256": panel.sha(manifest_path), "heads": models, "levels": levels, "chosen": chosen,
            "calibration": calibrated, "held_observations_at_selection": 0}
    panel.save(output, plan)
    print(json.dumps({"fit_groups": len(used), "calibration_groups": len(roles["calibration"]), "selected": chosen}, sort_keys=True))


def assess(lock, retained, fresh, fresh_reference, output):
    frozen = panel.read(lock)
    assert frozen["code"] == implementation() and frozen["method"] == METHOD
    original = retained / "deliveries"
    manifest = original / "manifest.json"
    assert frozen["manifest_sha256"] == panel.sha(manifest)
    _, incumbent = comparison.incumbent(retained / "reference.jsonl")
    _, fresh_current = comparison.incumbent(fresh_reference)
    transferred = score(original, panel.read(manifest), ("transfer",), frozen["heads"], incumbent)
    fresh_manifest = fresh / "manifest.json"
    independent = score(fresh, panel.read(fresh_manifest), ("fresh",), frozen["heads"], fresh_current)
    results = {"schema": SCHEMA, "freeze": frozen, "freeze_sha256": panel.sha(lock), "fresh_sha256": panel.sha(fresh_manifest),
               "rows": transferred + independent, "summary": summarize(transferred + independent, frozen["levels"])}
    verify(results)
    panel.save(output, results)


def verify(result):
    frozen = result["freeze"]
    assert result["schema"] == frozen["schema"] == SCHEMA and frozen["method"] == METHOD
    assert choose(frozen["calibration"]) == (frozen["chosen"], frozen["levels"])
    assert frozen["held_observations_at_selection"] == 0
    all_groups = {role: set(rows) for role, rows in frozen["roles"].items()}
    for role in ("fresh", "transfer"):
        all_groups[role] = {row["source_group"] for row in result["rows"] if row["split"] == role}
    assert len(set.union(*all_groups.values())) == sum(map(len, all_groups.values()))
    for row in result["rows"]:
        expected = {part["truth"] for part in row["current"].values() if part is not None}
        for value in row["checks"].values():
            width, height = row["shape"][::-1]
            match = value["intersection"]
            assert 0 <= match <= min(value["selected"], value["truth"]) <= width * height
            assert not expected or expected == {value["truth"]}
            assert bool(value["truth"]) == (row["role"] in panel.POSITIVE)
            bounds = value["box"]
            assert bool(bounds) == bool(value["selected"])
            if bounds:
                x, y, x1, y1 = bounds
                assert 0 <= x < x1 <= width and 0 <= y < y1 <= height and value["selected"] <= (x1-x)*(y1-y)
    assert summarize(result["rows"], frozen["levels"]) == result["summary"]
    return {"selected": frozen["chosen"], "groups": {role: len(groups) for role, groups in all_groups.items()},
            "holdout_deliveries": len(result["rows"])}


def open_receipt(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_subparsers(dest="phase", required=True)
    for name, args in (("fit", ("retained", "output")), ("assess", ("lock", "retained", "fresh", "fresh_reference", "output")),
                       ("verify", ("receipt",))):
        subparser = phase.add_parser(name)
        for argument in args:
            subparser.add_argument(argument, type=Path)
    settings = vars(parser.parse_args())
    step = settings.pop("phase")
    if step == "verify":
        print(json.dumps(verify(open_receipt(settings["receipt"])), sort_keys=True))
    else:
        {"fit": fit, "assess": assess}[step](**settings)
