"""Compare local color-residual correlation on fixed paired image edits."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts import image_independent_segmenter_transfer as paired_inputs
from scripts import image_wavelet_region_transfer as panel

SCHEMA = "chipmunk.image-color-residual-transfer/1"
CHANNEL_PAIRS = ((0, 1), (0, 2), (1, 2))
ARMS = ("local_deviation", "fitted_color")
RIDGE = 2.0
VARIANCE_FLOOR = 0.25
DIFF_SCALE_FLOOR = 0.08


def grid_view(values):
    side = panel.CELL
    h, w = (size // side for size in values.shape[:2])
    return values[: h * side, : w * side].reshape(h, side, w, side, 3).transpose(0, 2, 1, 3, 4)


def correlations(left, right, pairs):
    left = left - left.mean(axis=(2, 3), keepdims=True)
    right = right - right.mean(axis=(2, 3), keepdims=True)
    first, second = zip(*pairs, strict=True)
    a, b = left[..., first], right[..., second]
    covariance = np.mean(a * b, axis=(2, 3))
    scale = np.sqrt(np.maximum(a.var(axis=(2, 3)), VARIANCE_FLOOR)
                    * np.maximum(b.var(axis=(2, 3)), VARIANCE_FLOOR))
    return np.arctanh(np.clip(covariance / scale, -0.95, 0.95))


def features(image):
    rgb = np.asarray(image, dtype=np.float32)
    lab = cv2.cvtColor(rgb / 255, cv2.COLOR_RGB2LAB) * 2.55
    output = []
    for values in (rgb, lab):
        smooth = cv2.GaussianBlur(values, (5, 5), 1.0)
        coarse = cv2.GaussianBlur(values, (9, 9), 2.0)
        fine = grid_view(values - smooth)
        middle = grid_view(smooth - coarse)
        output.extend((correlations(fine, fine, CHANNEL_PAIRS), correlations(middle, middle, CHANNEL_PAIRS)))
        # Compare horizontal and vertical lag without crossing block boundaries.
        diagonal = ((1, 1), (2, 2))
        output.append(correlations(fine[:, :, :, :-1], fine[:, :, :, 1:], diagonal)
                      - correlations(fine[:, :, :-1], fine[:, :, 1:], diagonal))
    return np.concatenate(output, axis=2).astype(np.float64)


def ring_contrast(grid, boxes):
    integral = np.pad(grid.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)))
    rim = np.array([-1, -1, 1, 1])
    inside, separation, outside = boxes - rim, boxes + rim, boxes + 3 * rim
    w, h = (boxes[:, 2:] - boxes[:, :2]).T
    core = panel.sums(integral, inside) / ((w - 2) * (h - 2))[:, None]
    ring = (panel.sums(integral, outside) - panel.sums(integral, separation)) / (
        (w + 6) * (h + 6) - (w + 2) * (h + 2))[:, None]
    return core - ring


def rank(image, boxes, weights, scale):
    measurements = ring_contrast(features(image), boxes)
    normalized = measurements / scale
    responses = (np.sqrt(np.mean(normalized ** 2, axis=1)), measurements @ weights)
    proposals = {}
    for arm, scores in zip(ARMS, responses, strict=True):
        index = int(scores.argmax())
        proposals[arm] = float(scores[index]), (boxes[index] * panel.CELL).tolist()
    return proposals


def development_views(folder):
    _, plan, attempts = panel.scoring.checked_inputs(folder)
    return panel.scoring.panels(folder, plan, attempts)


def fit(folder, fmt):
    paired, control = [], []
    for _, _, views, truth in development_views(folder):
        source = views["original"]
        candidates = panel.regions(tuple(size // panel.CELL for size in truth.shape))
        box = panel.check_case(truth, candidates)
        native_box = tuple(int(value) for value in box[0] * panel.CELL)
        controls = panel.view_controls(source, views["processed"], native_box)
        positives = (views["edited"], panel.blend.local(source, views["edited"], native_box, feather=True))
        measured = [ring_contrast(features(panel.paired.encoded(pixels, fmt)[1]), box)[0]
                    for pixels in (*positives, *controls)]
        control.extend(measured[2:])
        paired.append((np.array(measured[:2])[:, None] - np.array(measured[2:])).reshape(-1, len(measured[0])))
    assert len(paired) == 8
    contrasts = np.concatenate(paired)
    scale = np.maximum(np.sqrt(np.mean(np.square(control), axis=0)), DIFF_SCALE_FLOOR)
    standardized = contrasts / scale
    covariance = standardized.T @ standardized / len(standardized)
    weights = np.linalg.solve(covariance + RIDGE * np.eye(len(scale)), standardized.mean(axis=0)) / scale
    return weights, scale


def control_points(folder, fmt, weights, scale):
    maxima = {arm: [] for arm in ARMS}
    for _, _, views, _ in development_views(folder):
        pixels = views["original"]
        h, w = pixels.shape[:2]
        controls = panel.view_controls(pixels, views["processed"], (w // 3, h // 3, w * 7 // 12, h * 7 // 12))
        candidates = panel.regions(tuple(size // panel.CELL for size in pixels.shape[:2]))
        measurements = [rank(panel.paired.encoded(control, fmt)[1], candidates, weights, scale) for control in controls]
        for arm in ARMS:
            maxima[arm].append(max(row[arm][0] for row in measurements))
    assert all(len(values) == 8 for values in maxima.values())
    return {arm: {"point": sorted(values)[-2], "maxima": values} for arm, values in maxima.items()}


def confirmation(directory, locked, fitted, baseline):
    for source, image, truth, truth_sha in paired_inputs._inputs(directory, locked):
        fmt = source["format"]
        params = fitted[fmt]
        boxes = panel.regions(tuple(size // panel.CELL for size in truth.shape))
        ranked = rank(image, boxes, np.array(params["weights"]), np.array(params["scale"]))
        measurements = {arm: panel.scoring.measure(panel.paired.boxed(box, truth.shape,
                        value > params["controls"][arm]["point"]), truth) for arm, (value, box) in ranked.items()}
        if baseline is not None:
            digest, measured = baseline.pop(paired_inputs._key(source))
            assert digest == source["input_sha256"] and measured["target"] == int(truth.sum())
            measurements["current"] = measured
        yield {**{key: source[key] for key in ("slot", "role", "format", "generator", "input_sha256")},
               "truth_sha256": truth_sha, "measurements": measurements, "proposals": ranked}
    assert not baseline


def counts(rows):
    totals = Counter()
    for row in rows:
        for arm, m in row["measurements"].items():
            group = (row["format"], arm, row["role"])
            totals[group + ("views",)] += 1
            totals[group + ("flags",)] += m["reported"] > 0
            if m["target"]:
                union = m["reported"] + m["target"] - m["overlap"]
                totals[group + ("useful",)] += 10 * m["overlap"] >= 3 * union and 2 * m["overlap"] >= m["target"]
    return {"/".join(group): total for group, total in sorted(totals.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "confirm"))
    for name in ("training", "controls", "earlier", "earlier_boundary", "later", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    folders = (args.training, args.controls, args.earlier, args.later)
    groups = [{case["flickr_photo_id"] for case in json.loads((path / "plan.json").read_bytes())["cases"]} for path in folders]
    assert [len(group) for group in groups] == [8, 8, 12, 12] and len(set.union(*groups)) == 40
    receipts = [path / name for path in folders for name in ("plan.json", "result.json")]
    receipts += [args.earlier_boundary / "panel/result.json", args.later / "score-result.json"]
    dependencies = (Path(__file__), Path(panel.__file__), Path(panel.paired.__file__), Path(panel.blend.__file__),
                    Path(panel.scoring.__file__), Path(paired_inputs.__file__))
    seal = {"schema": SCHEMA, "sources_per_role": [8, 8, 12, 12], "source_receipt_sha256": [panel.sha(p) for p in receipts],
            "code_sha256": {p.name: panel.sha(p) for p in dependencies}, "arms": list(ARMS), "ridge": RIDGE,
            "variance_floor": VARIANCE_FLOOR, "scale_floor": DIFF_SCALE_FLOOR,
            "review_point": "strictly above second-highest development source maximum",
            "minimum_iou": 0.30, "minimum_recall": 0.50}
    freeze_path, points_path = args.output / "freeze.json", args.output / "points.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        panel.write(freeze_path, seal)
        return
    assert seal == json.loads(freeze_path.read_bytes())
    cv2.setNumThreads(1)
    if args.stage == "fit":
        fitted = {}
        for fmt in panel.paired.FORMATS:
            weights, scale = fit(args.training, fmt)
            controls = control_points(args.controls, fmt, weights, scale)
            fitted[fmt] = {"weights": weights.tolist(), "scale": scale.tolist(), "controls": controls}
        panel.write(points_path, fitted)
        return

    started = time.monotonic()
    fitted = json.loads(points_path.read_bytes())
    panels = (
        ("earlier", args.earlier, paired_inputs._receipt(args.earlier_boundary / "panel", "rows.json", "result.json", "journal_sha256"),
         paired_inputs._baseline(args.earlier, args.earlier_boundary)),
        ("later", args.later, paired_inputs._receipt(args.later, "outcomes.json", "score-result.json", "outcomes_sha256"), None),
    )
    result = {"schema": SCHEMA, "freeze_sha256": panel.sha(freeze_path), "points_sha256": panel.sha(points_path), "panels": {}}
    for name, directory, locked, baseline in panels:
        rows = list(confirmation(directory, locked, fitted, baseline))
        assert len(rows) == 192
        journal = args.output / f"{name}-rows.json"
        panel.write(journal, rows)
        result["panels"][name] = {"views": len(rows), "counts": counts(rows), "journal_sha256": panel.sha(journal)}
        print(f"Checked {len(rows)} exact {name} inputs", flush=True)
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    panel.write(args.output / "result.json", result)


if __name__ == "__main__":
    main()
