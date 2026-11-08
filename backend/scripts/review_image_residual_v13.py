"""Separate affine, median, connected-component and source-coordinate replay.

Does not fit heads, choose points, load neural weights or change study outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import run_image_residual_v13 as frozen


def scores(values, head):
    # Combine the projection and coefficients first, rather than applying the
    # training implementation's two matrix products to every feature cell.
    coefficients = np.asarray(head["projection"]) @ np.asarray(head["weight"])
    z = (values.astype(np.float64) - head["center"]) / head["scale"] @ coefficients + head["bias"]
    raw = 1 / (1 + np.exp(-z))
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(raw, 1, mode="symmetric"), (3, 3))
    return np.median(windows, axis=(-2, -1))


def source_mask(values, threshold, width, height):
    binary = values > threshold
    seen, kept = np.zeros((64, 64), bool), np.zeros((64, 64), bool)
    for y, x in np.argwhere(binary):
        if seen[y, x]:
            continue
        queue, cells = [(int(y), int(x))], []
        seen[y, x] = True
        while queue:
            a, b = queue.pop()
            cells.append((a, b))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    c, d = a + dy, b + dx
                    if 0 <= c < 64 and 0 <= d < 64 and binary[c, d] and not seen[c, d]:
                        seen[c, d] = True
                        queue.append((c, d))
        if len(cells) >= 16:
            for a, b in cells:
                kept[a, b] = True
    result = np.zeros((height, width), bool)
    for y, x in np.argwhere(kept):
        result[
            (y * height + 63) // 64 : ((y + 1) * height + 63) // 64,
            (x * width + 63) // 64 : ((x + 1) * width + 63) // 64,
        ] = True
    return result


def arithmetic(prediction, truth):
    p, t = int(np.count_nonzero(prediction)), int(np.count_nonzero(truth))
    overlap = int(np.count_nonzero(np.logical_and(prediction, truth)))
    union = p + t - overlap
    return {
        "intersection": overlap,
        "false_positive_pixels": p - overlap,
        "false_negative_pixels": t - overlap,
        "reference_positive_pixels": t,
        "predicted_positive_pixels": p,
        "pixels": truth.size,
        "iou": overlap / union if union else None,
        "pixel_f1": 2 * overlap / (p + t) if p + t else None,
        "flag": p > 0,
        "selected_area_fraction": p / truth.size,
    }


def summarize(rows):
    ai = [r["metrics"] for r in rows if r["label"] == 1 and r["variant"] == "original"]
    original = [r["metrics"] for r in rows if r["label"] == 0 and r["variant"] == "original"]
    ordinary = [r["metrics"] for r in rows if r["variant"] == "ordinary_patch"]
    return {
        "edited_inputs": len(ai),
        "edited_flagged": sum(r["flag"] for r in ai),
        "edited_iou_at_least_0_3": sum(r["iou"] >= 0.3 for r in ai),
        "edited_mean_iou": math.fsum(r["iou"] for r in ai) / len(ai),
        "genuine_inputs": len(original),
        "genuine_flags": sum(r["flag"] for r in original),
        "ordinary_inputs": len(ordinary),
        "ordinary_flags": sum(r["flag"] for r in ordinary),
    }


def audit(trial, output):
    plan = frozen.check(trial)
    profile = frozen.read(trial / "profile/profile.json")
    result = frozen.read(trial / "result.json")
    frozen.method.require(
        profile["child_exit_code"] == 0 and profile["guard"] is None, "Successful guarded run required"
    )
    journal = [json.loads(line) for line in (trial / "features.jsonl").read_text().splitlines()]
    expected = {
        (e["key"], variant)
        for e in plan["entries"]
        for variant in (
            (["original", "lighting", "jpeg55"] if e["role"] == "fit" else ["original"])
            + (["ordinary_patch"] if e["source"]["label"] == 0 else [])
        )
    }
    actual = [(r["key"], r["variant"]) for r in journal]
    frozen.method.require(
        len(actual) == len(set(actual)) == plan["feature_cases_planned"] and set(actual) == expected,
        "Complete unique planned cases required",
    )
    entries = {e["key"]: e for e in plan["entries"]}
    for row in journal:
        entry = entries[row["key"]]
        frozen.method.require(
            all(row[key] == entry[key] for key in ("role", "cohort", "generator"))
            and row["group"] == entry["source"]["group"]
            and row["label"] == entry["source"]["label"]
            and row["source_input_sha256"] == entry["source"]["input_sha256"],
            "Source role/label/binding changed",
        )
    heads = {name: frozen.read(trial / (name + "-head-private.json")) for name in frozen.POLICY["arms"]}
    cases = {(r["key"], r["variant"]): r for r in journal if r["role"] != "fit"}
    outcomes = frozen.read(trial / "outcomes-private.json")
    frozen.method.require(
        len(outcomes) == 2 * len(cases)
        and len({(r["key"], r["variant"], r["arm"]) for r in outcomes}) == len(outcomes),
        "Both complete diagnostic arms required",
    )
    truth_by_key = {}
    for key in cases:
        if key[0] not in truth_by_key:
            _, truth_by_key[key[0]] = frozen.load(plan, entries[key[0]])
    maximum = 0.0
    point_maxima = {name: defaultdict(list) for name in heads}
    for row in outcomes:
        case = cases[(row["key"], row["variant"])]
        head = heads[row["arm"]]
        truth = truth_by_key[row["key"]]
        frozen.method.require(
            frozen.digest(case["features_path"]) == case["features_file_sha256"]
            and frozen.digest(row["scores_path"]) == row["scores_file_sha256"]
            and frozen.method.sha_array(truth) == case["truth_sha256"] == row["truth_sha256"],
            "Source/cache/score binding changed",
        )
        with np.load(case["features_path"], allow_pickle=False) as data:
            values = data["features"]
            frozen.method.require(
                frozen.method.sha_array(values) == case["features_sha256"], "Feature array changed"
            )
            measured = scores(values, head)
        with np.load(row["scores_path"], allow_pickle=False) as data:
            recorded = data["scores"]
        frozen.method.require(
            recorded.shape == measured.shape == (64, 64)
            and np.isfinite(recorded).all()
            and np.isfinite(measured).all(),
            "Finite matching score grids required",
        )
        maximum = max(maximum, float(np.abs(measured - recorded).max()))
        if row["role"] == "point" and row["label"] == 0:
            point_maxima[row["arm"]][row["group"]].append(float(recorded.max()))
        prediction = source_mask(recorded, head["point"]["value"], case["width"], case["height"])
        metrics = arithmetic(prediction, truth)
        frozen.method.require(
            frozen.method.sha_array(prediction) == row["mask_sha256"] and metrics == row["metrics"],
            "Independent component/coordinate/count replay differs",
        )
    frozen.method.require(maximum <= 2e-12, "Independent affine/median replay differs")
    for name, head in heads.items():
        values = sorted((max(v) for v in point_maxima[name].values()), reverse=True)
        frozen.method.require(
            len(values) == 60
            and values[3] == head["point"]["value"]
            and sum(v > head["point"]["value"] for v in values) <= 3,
            "Genuine-only fixed comparison point differs",
        )
    partitions = {}
    for name in heads:
        selected = [r for r in outcomes if r["role"] == "diagnostic" and r["arm"] == name]
        partitions[name] = {
            "all": summarize(selected),
            "by_generator": {
                family: summarize([r for r in selected if r["generator"] == family])
                for family in sorted({r["generator"] for r in selected})
            },
        }
        summary = result["diagnostic"]["all"]["arms"][name]
        frozen.method.require(
            all(
                abs(summary[k] - v) <= 1e-12 if isinstance(v, float) else summary[k] == v
                for k, v in partitions[name]["all"].items()
            ),
            "Aggregate mismatch",
        )
    frozen.write(
        output,
        {
            "schema": "chipmunk.image-residual-source-arithmetic-audit/13",
            "freeze_sha256": frozen.digest(trial / "freeze.json"),
            "result_sha256": frozen.digest(trial / "result.json"),
            "audit_code_sha256": frozen.digest(Path(__file__)),
            "feature_cases_verified": len(actual),
            "role_case_counts": dict(Counter(r["role"] for r in journal)),
            "independent_score_grids": len(outcomes),
            "independent_source_masks": len(outcomes),
            "independent_genuine_comparison_points": len(heads),
            "maximum_response_difference": maximum,
            "partitions": partitions,
            "new_neural_forwards": 0,
            "outside_human_reviews": 0,
            "serving_admitted": False,
        },
    )
    print(
        json.dumps(
            {
                "audited_feature_cases": len(actual),
                "independent_source_masks": len(outcomes),
                "maximum_response_difference": maximum,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.trial.resolve(), args.output.resolve())
