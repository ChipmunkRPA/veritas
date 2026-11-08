"""Separate recorded-grid/source/component audit; never loads neural weights."""

from __future__ import annotations

import argparse
import hashlib
import io
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def require(value, reason):
    if not value:
        raise ValueError(reason)


def resize(values, height, width):
    y = (np.arange(height, dtype=np.float64) + 0.5) * values.shape[0] / height - 0.5
    x = (np.arange(width, dtype=np.float64) + 0.5) * values.shape[1] / width - 0.5
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return ndimage.map_coordinates(
        values.astype(np.float64), [yy, xx], order=1, mode="nearest", prefilter=False
    )


def reconstruct(grids, width, height):
    base = resize(grids[0] / 255.0, height, width)
    tw, th = math.ceil(width * 0.6), math.ceil(height * 0.6)
    records = [(x, y) for y in (0, height - th) for x in (0, width - tw)]
    parts, weights = [], []
    for grid, (x, y) in zip(grids[1:], records, strict=True):
        values = resize(grid / 255.0, th, tw)
        left, top = math.ceil(tw * 0.06) if x else 0, math.ceil(th * 0.06) if y else 0
        right = tw - (math.ceil(tw * 0.06) if x + tw < width else 0)
        bottom = th - (math.ceil(th * 0.06) if y + th < height else 0)
        placed, mass = np.zeros((height, width)), np.zeros((height, width))
        placed[y + top : y + bottom, x + left : x + right] = values[top:bottom, left:right]
        mass[y + top : y + bottom, x + left : x + right] = 1
        parts.append(placed)
        weights.append(mass)
    denominator = np.sum(weights, axis=0)
    require((denominator > 0).all(), "Independent source coverage has a gap")
    return {"whole": base, "zoom": (base + np.sum(parts, axis=0) / denominator) / 2}


def display_metric(raw, pixels, data, truth, threshold=0.5, eligible=True):
    from app.detectors.image import jpeg_bitstream
    from app.detectors.image import localiser as loc
    from app.evidence.image_packets import MAX_PER_MAP

    height, width = truth.shape
    values = (
        np.rint(
            np.clip(cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float64), 0, 1)
            * 255
        )
        / 255.0
    )
    structure = jpeg_bitstream.read(data)
    excluded, _ = loc.conditions_mask(pixels, structure.quality_luma if structure else None)
    # Sources are at most1024 pixels: the existing screening grid equals source.
    binary = (values >= threshold) & ~excluded
    labels, number = ndimage.label(binary, structure=np.ones((3, 3)))
    counts = np.bincount(labels.ravel())
    minimum = max(1, math.ceil(loc.MIN_REGION_FRACTION * values.size))
    chosen = []
    for identity, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None or counts[identity] < minimum:
            continue
        mask = labels[box] == identity
        inside = values[box][mask]
        y, x = box[0].start, box[1].start
        chosen.append(
            (
                (-round(float(inside.mean()), 4), -round(int(counts[identity]) / values.size, 4), y, x),
                identity,
            )
        )
    selected = [identity for _, identity in sorted(chosen)[: min(loc.MAX_REGIONS, MAX_PER_MAP)]]
    predicted = np.isin(labels, selected) if eligible else np.zeros(truth.shape, bool)
    tp, fp, fn = (
        int(np.count_nonzero(predicted & truth)),
        int(np.count_nonzero(predicted & ~truth)),
        int(np.count_nonzero(~predicted & truth)),
    )
    return {
        "intersection": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "pixels": int(truth.size),
        "reference_positive_pixels": tp + fn,
        "predicted_positive_pixels": tp + fp,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "pixel_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "flagged_area_fraction": (tp + fp) / truth.size,
    }


def stats(rows, arm):
    edited = [r for r in rows if r["kind"] == "ai"]
    return {
        "files": len(rows),
        "edited": len(edited),
        "assessed": len(rows),
        "any_overlap": sum(r["metrics"][arm]["intersection"] > 0 for r in edited),
        "useful_iou_at_least_03": sum(r["metrics"][arm]["iou"] >= 0.3 for r in edited),
        "mean_edited_iou": sum(r["metrics"][arm]["iou"] for r in edited) / len(edited),
        "mean_edited_pixel_f1": sum(r["metrics"][arm]["pixel_f1"] for r in edited) / len(edited),
        "original_highlights": sum(
            r["metrics"][arm]["predicted_positive_pixels"] > 0 for r in rows if r["kind"] == "original"
        ),
        "ordinary_highlights": sum(
            r["metrics"][arm]["predicted_positive_pixels"] > 0 for r in rows if r["kind"] == "ordinary"
        ),
    }


def aggregate(outcomes):
    require(
        len(outcomes) == len({r["id"] for r in outcomes}) == 288, "All288 exact diagnostic outcomes required"
    )
    result = {"formats": {}}
    for fmt in ("native", "jpeg75"):
        rows = [r for r in outcomes if r["format"] == fmt]
        require(
            len(rows) == 144
            and all(sum(r["kind"] == kind for r in rows) == 48 for kind in ("original", "ordinary", "ai")),
            "Complete48 source triplets per format required",
        )
        arms = {arm: stats(rows, arm) for arm in ("current", "whole", "zoom")}
        q = arms["zoom"]
        gates = {}
        for reference in ("current", "whole"):
            r = arms[reference]
            gates[reference + "_more_useful"] = q["useful_iou_at_least_03"] > r["useful_iou_at_least_03"]
            gates[reference + "_higher_iou"] = q["mean_edited_iou"] > r["mean_edited_iou"]
        gates["preserve_current_useful"] = all(
            r["metrics"]["zoom"]["iou"] >= 0.3
            for r in rows
            if r["kind"] == "ai" and r["metrics"]["current"]["iou"] >= 0.3
        )
        for kind in ("original", "ordinary"):
            key = kind + "_highlights"
            gates["current_" + kind + "_budget"] = q[key] <= arms["current"][key] + 1
            gates["whole_" + kind + "_budget"] = q[key] <= arms["whole"][key]
        gates["complete_coverage"] = all(v["assessed"] == 144 for v in arms.values())
        intervals = {}
        ai = [r for r in rows if r["kind"] == "ai"]
        for reference in ("current", "whole"):
            delta = np.asarray([r["metrics"]["zoom"]["iou"] - r["metrics"][reference]["iou"] for r in ai])
            rng = np.random.default_rng(20261108)
            intervals[reference] = {
                "mean_iou_difference": float(delta.mean()),
                "source_bootstrap_95_percentile": np.quantile(
                    rng.choice(delta, (5000, 48), replace=True).mean(1), [0.025, 0.975]
                ).tolist(),
            }
        families = {
            family: {arm: stats([r for r in rows if r["family"] == family], arm) for arm in arms}
            for family in ("firefly", "sd2", "flux-fill")
        }
        result["formats"][fmt] = {
            "arms": arms,
            "gates": gates,
            "qualifies_fresh_confirmation": all(gates.values()),
            "paired_source_intervals": intervals,
            "families": families,
        }
    return result


def review(trial, output):
    from scripts import run_image_iml_zoom_v16 as run

    plan = run.check(trial)
    run.check_points(trial)
    points, measured = run.read(trial / "points.json"), {}
    for phase in [
        "point-native",
        "point-jpeg75",
        *[f"diag-{f}-{k}" for f in run.FAMILIES for k in run.FORMATS],
    ]:
        part = run.lines(trial / (phase + ".jsonl"))
        complete = run.read(trial / (phase + "-complete.json"))
        require(
            len(part) == complete["complete"] == 48
            and complete["journal_sha256"] == run.sha(trial / (phase + ".jsonl")),
            "Original full source phase required",
        )
        measured.update({r["id"]: r for r in part})
    outcomes = {r["id"]: r for r in run.read(trial / "private-outcomes.json")}
    max_difference, windows = 0.0, 0
    point_values = {fmt: {arm: {} for arm in ("whole", "zoom")} for fmt in run.FORMATS}
    for case in plan:
        row = measured[case["id"]]
        # Reconstruct the ordinary operation separately, sharing only Pillow's
        # declared blur/codec primitives and the immutable publisher source.
        data, rgb, truth = run.source(run.ROOT / case["root"], case["item"])
        if case["kind"] == "ordinary":
            h, w = rgb.shape[:2]
            image = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(2)), np.float64) * 1.08
            x, y = np.arange(w)[None, :], np.arange(h)[:, None]
            alpha = np.maximum(
                0,
                np.minimum(
                    1,
                    np.minimum(
                        np.minimum(x - w // 4 + 0.5, 3 * w // 4 - x - 0.5),
                        np.minimum(y - h // 4 + 0.5, 3 * h // 4 - y - 0.5),
                    )
                    / 16,
                ),
            )
            rgb = np.rint(
                rgb.astype(np.float64) * (1 - alpha[..., None]) + np.clip(image, 0, 255) * alpha[..., None]
            ).astype(np.uint8)
            encoded = io.BytesIO()
            Image.fromarray(rgb).save(encoded, format="PNG")
            data = encoded.getvalue()
            truth = np.zeros(rgb.shape[:2], bool)
        data, rgb = run.input_bytes(rgb, data, case["format"])
        require(
            hashlib.sha256(data).hexdigest() == row["input_sha256"] and run.digest(rgb) == row["rgb_sha256"],
            "Independent ordinary/JPEG/source reconstruction differs",
        )
        path = trial / "private" / row["grids_file"]
        require(
            path.parent == trial / "private" and run.sha(path) == row["grids_sha256"],
            "Recorded grid packet changed",
        )
        with np.load(path, allow_pickle=False) as saved:
            grids = saved["grids"].copy()
        require(
            grids.dtype == np.uint8 and grids.shape == (5, 256, 256), "Every lossless response grid required"
        )
        maps = reconstruct(grids, row["width"], row["height"])
        primary = run.method.reconstruct(grids, row["width"], row["height"])
        for arm in ("whole", "zoom"):
            difference = float(np.max(np.abs(maps[arm] - primary[arm])))
            require(difference <= 1e-12, "Independent whole/tile interpolation differs")
            max_difference = max(max_difference, difference)
            metric = display_metric(maps[arm], rgb, data, truth)
            require(
                metric == row["responses"][arm]["ungated"],
                "Independent component/rank/source-mask metrics differ",
            )
            frac = int(np.count_nonzero(np.rint(maps[arm] * 255) >= 128)) / truth.size
            require(
                frac == row["responses"][arm]["active_fraction"],
                "Independent recorded source activity differs",
            )
            if case["role"] == "point":
                point_values[case["format"]][arm].setdefault(case["source_id"], []).append(
                    frac if metric["predicted_positive_pixels"] else 0.0
                )
            else:
                eligible = frac >= points[case["format"]][arm]["minimum_active_fraction"]
                require(
                    display_metric(maps[arm], rgb, data, truth, eligible=eligible)
                    == outcomes[case["id"]]["metrics"][arm],
                    "Independent point-gated diagnostic metric differs",
                )
        if case["role"] == "diagnostic":
            if case["kind"] == "ordinary":
                references = run.lines(trial / ("reference-" + case["format"] + ".jsonl"))
                reference = next(r for r in references if r["id"] == case["id"])
                path = trial / "private" / reference["raw_file"]
                require(
                    run.sha(path) == reference["raw_file_sha256"], "Actual ordinary reference array changed"
                )
                with np.load(path, allow_pickle=False) as saved:
                    raw = saved["raw"].copy()
            else:
                raw, reference = run.cached(case, current=True)
            require(reference["input_sha256"] == row["input_sha256"], "Current/reference exact input differs")
            metric = display_metric(
                raw,
                rgb,
                data,
                truth,
                threshold=0.7 if case["format"] == "native" else 0.5,
                eligible=reference["gate"],
            )
            require(
                metric == outcomes[case["id"]]["metrics"]["current"],
                "Independent actual current-reference metrics differ",
            )
        require(
            row["reduction_max_difference"] <= 1e-12 and len(row["tile_proofs"]) == 4,
            "Complete inherited inline native checks required",
        )
        windows += 1
    for fmt in run.FORMATS:
        for arm, groups in point_values[fmt].items():
            require(
                len(groups) == 24 and all(len(v) == 2 for v in groups.values()),
                "Every genuine source pair required",
            )
            cut = max(
                0.001, math.nextafter(sorted((max(v) for v in groups.values()), reverse=True)[1], math.inf)
            )
            require(
                cut == points[fmt][arm]["minimum_active_fraction"],
                "Independent genuine-only source point differs",
            )
    original = run.read(trial / "result.json")
    require(
        aggregate(list(outcomes.values()))["formats"] == original["formats"],
        "Complete aggregate reconstruction differs",
    )
    require(windows == 384, "All point and diagnostic source cases required")
    run.write(
        output,
        {
            "schema": "chipmunk.image-iml-zoom-complete-review/16",
            "source_file_cases": windows,
            "recorded_uint8_grids": windows * 5,
            "diagnostic_arm_metrics": 288 * 3,
            "maximum_stitch_difference": max_difference,
            "all_native_source_mask_metrics_exact": True,
            "all_genuine_points_exact": True,
            "all_aggregates_exact": True,
            "new_neural_forwards": 0,
            "scope": "Source/grid/component arithmetic; inline reduction/input checks inherited, not every neural output re-executed.",
            "result_sha256": run.sha(trial / "result.json"),
            "reviewer_sha256": run.sha(Path(__file__)),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    review(args.trial, args.output)
