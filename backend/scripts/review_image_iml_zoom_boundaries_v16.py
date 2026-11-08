"""Post-result numerical sensitivity audit; never clears the failed strict audit."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from scripts import review_image_iml_zoom_counts_v16 as counts
from scripts import review_image_iml_zoom_v16 as independent
from scripts import run_image_iml_zoom_v16 as run

require = counts.require


def source(case):
    data, rgb, truth = run.source(run.ROOT / case["root"], case["item"])
    if case["kind"] == "ordinary":
        h, w = rgb.shape[:2]
        target = np.clip(
            np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(2)), np.float64) * 1.08, 0, 255
        )
        x, y = np.arange(w)[None, :], np.arange(h)[:, None]
        distance = np.minimum(
            np.minimum(x - w // 4 + 0.5, 3 * w // 4 - x - 0.5),
            np.minimum(y - h // 4 + 0.5, 3 * h // 4 - y - 0.5),
        )
        alpha = np.clip(distance / 16, 0, 1)[..., None]
        rgb = np.rint(rgb.astype(np.float64) * (1 - alpha) + target * alpha).astype(np.uint8)
        encoded = io.BytesIO()
        Image.fromarray(rgb).save(encoded, format="PNG")
        data, truth = encoded.getvalue(), np.zeros(rgb.shape[:2], bool)
    if case["format"] == "jpeg75":
        encoded = io.BytesIO()
        Image.fromarray(rgb).save(encoded, format="JPEG", quality=75)
        data = encoded.getvalue()
        with Image.open(io.BytesIO(data)) as im:
            rgb = np.asarray(im.convert("RGB")).copy()
    return data, rgb, truth


def quantized(values):
    return np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)


def boundary_difference(a, b):
    error = float(np.max(np.abs(a - b)))
    require(error <= 1e-12, "Independent continuous maps exceed the original tolerance")
    qa, qb = quantized(a), quantized(b)
    changed = qa != qb
    if changed.any():
        require(
            np.max(np.abs(qa[changed].astype(int) - qb[changed].astype(int))) == 1,
            "Discrete difference exceeds a neighboring quantization level",
        )
        v = a[changed] * 255
        distance = np.abs(v - (np.floor(v) + 0.5))
        require((distance <= error * 255 + 1e-12).all(), "Difference is not at a rounding discontinuity")
    return error, int(changed.sum())


def zero_metric(truth):
    fn = int(truth.sum())
    return {
        "intersection": 0,
        "false_positive_pixels": 0,
        "false_negative_pixels": fn,
        "pixels": int(truth.size),
        "reference_positive_pixels": fn,
        "predicted_positive_pixels": 0,
        "iou": 0.0 if fn else 1.0,
        "pixel_f1": 0.0 if fn else 1.0,
        "precision": None,
        "recall": 0.0 if fn else None,
        "flagged_area_fraction": 0.0,
    }


def review(trial, output):
    lock_path = run.ROOT / "docs/evaluation-panels/image-iml-zoom-v16-boundary-audit-freeze.json"
    lock = run.read(lock_path)
    require(lock["post_result_diagnostic"] is True, "Diagnostic scope must remain explicit")
    for name, value in lock["code_sha256"].items():
        require(run.sha(run.ROOT / name) == value, "Published supplementary audit changed")
    for name, value in lock["base_artifacts_sha256"].items():
        require(run.sha(trial / name) == value, "Original measured artifact changed")
    plan = run.check(trial)
    run.check_points(trial)
    original = run.read(trial / "result.json")
    strict = run.read(trial / "review-profile/profile.json")
    strict_log = trial / "review-profile/worker.log"
    require(
        strict["child_exit_code"] == 1
        and strict["guard"] is None
        and "Independent component/rank/source-mask metrics differ" in strict_log.read_text(),
        "Original failed strict audit must remain present",
    )
    outcome = {r["id"]: r for r in run.read(trial / "private-outcomes.json")}
    points = run.read(trial / "points.json")
    measured = {}
    for phase in ["point-" + f for f in run.FORMATS] + [
        f"diag-{g}-{f}" for g in run.FAMILIES for f in run.FORMATS
    ]:
        rows = run.lines(trial / (phase + ".jsonl"))
        complete = run.read(trial / (phase + "-complete.json"))
        require(
            len(rows) == complete["planned"] == complete["complete"] == 48
            and complete["failed"] == 0
            and complete["journal_sha256"] == run.sha(trial / (phase + ".jsonl")),
            "Every exact completed native case required",
        )
        measured.update({r["id"]: r for r in rows})
    require(len(measured) == len(plan) == 384, "Complete source plan required")
    max_error, cells, metric_cases = 0.0, 0, 0
    sections, control_values, diagnostic = {}, {}, []
    for index, case in enumerate(plan):
        row = measured[case["id"]]
        data, rgb, truth = source(case)
        require(
            hashlib.sha256(data).hexdigest() == row["input_sha256"] and run.digest(rgb) == row["rgb_sha256"],
            "Independent original/ordinary/JPEG input reconstruction differs",
        )
        path = trial / "private" / row["grids_file"]
        require(run.sha(path) == row["grids_sha256"], "Recorded grid packet changed")
        with np.load(path, allow_pickle=False) as saved:
            grids = saved["grids"].copy()
        require(grids.dtype == np.uint8 and grids.shape == (5, 256, 256), "All five recorded grids required")
        a = run.method.reconstruct(grids, row["width"], row["height"])
        b = independent.reconstruct(grids, row["width"], row["height"])
        item = {k: case[k] for k in ("id", "source_id", "family", "kind", "format", "label")}
        item.update(metrics={}, responses={}, zero=zero_metric(truth))
        for arm in ("whole", "zoom"):
            error, changed = boundary_difference(a[arm], b[arm])
            max_error, cells = max(max_error, error), cells + changed
            # Independent components on the exact canonical map must still
            # reproduce the frozen stored metric without any numeric tolerance.
            canonical = independent.display_metric(a[arm], rgb, data, truth)
            require(canonical == row["responses"][arm]["ungated"], "Canonical component replay differs")
            require(
                run.method.fraction(a[arm]) == row["responses"][arm]["active_fraction"],
                "Canonical source activity differs",
            )
            other = independent.display_metric(b[arm], rgb, data, truth)
            frac = int(np.count_nonzero(quantized(b[arm]) >= 128)) / truth.size
            different = other != canonical
            require(not different or changed > 0, "Discrete metric mismatch without a rounding boundary")
            metric_cases += int(different)
            key = "/".join((case["role"], case["format"], case["kind"], arm))
            s = sections.setdefault(
                key,
                {
                    "cases": 0,
                    "changed_quantized_cells": 0,
                    "metric_mismatch_cases": 0,
                    "maximum_positive_pixel_difference": 0,
                },
            )
            s["cases"] += 1
            s["changed_quantized_cells"] += changed
            s["metric_mismatch_cases"] += int(different)
            s["maximum_positive_pixel_difference"] = max(
                s["maximum_positive_pixel_difference"],
                abs(other["predicted_positive_pixels"] - canonical["predicted_positive_pixels"]),
            )
            if case["role"] == "point":
                control_values.setdefault((case["format"], arm, case["source_id"]), []).append(
                    frac if other["predicted_positive_pixels"] else 0.0
                )
            else:
                item["responses"][arm] = {"fraction": frac, "ungated": other}
                item["metrics"][arm] = (
                    canonical
                    if run.method.fraction(a[arm]) >= points[case["format"]][arm]["minimum_active_fraction"]
                    else item["zero"]
                )
                require(
                    item["metrics"][arm] == outcome[case["id"]]["metrics"][arm],
                    "Canonical gated outcome differs",
                )
        if case["role"] == "diagnostic":
            if case["kind"] == "ordinary":
                reference = next(
                    r
                    for r in run.lines(trial / ("reference-" + case["format"] + ".jsonl"))
                    if r["id"] == case["id"]
                )
                path = trial / "private" / reference["raw_file"]
                require(
                    run.sha(path) == reference["raw_file_sha256"], "Actual ordinary-reference array changed"
                )
                with np.load(path, allow_pickle=False) as saved:
                    raw = saved["raw"].copy()
            else:
                raw, reference = run.cached(case, current=True)
            require(reference["input_sha256"] == row["input_sha256"], "Exact current-reference input differs")
            current = independent.display_metric(
                raw,
                rgb,
                data,
                truth,
                threshold=0.7 if case["format"] == "native" else 0.5,
                eligible=reference["gate"],
            )
            require(
                current == outcome[case["id"]]["metrics"]["current"],
                "Current-reference source-mask replay differs",
            )
            item["metrics"]["current"] = current
            diagnostic.append(item)
        if (index + 1) % 48 == 0:
            print(
                json.dumps(
                    {
                        "audited_file_cases": index + 1,
                        "metric_mismatch_cases": metric_cases,
                        "quantized_cell_differences": cells,
                    }
                ),
                flush=True,
            )

    require(
        len(diagnostic) == 288
        and len(control_values) == 96
        and all(len(v) == 2 for v in control_values.values()),
        "All source roles required",
    )
    alternate_points = {fmt: {} for fmt in run.FORMATS}
    for fmt in run.FORMATS:
        for arm in ("whole", "zoom"):
            values = [max(v) for (f, a, _), v in control_values.items() if f == fmt and a == arm]
            alternate_points[fmt][arm] = max(0.001, math.nextafter(sorted(values, reverse=True)[1], math.inf))
    canonical_rows = [{k: v for k, v in r.items() if k not in ("responses", "zero")} for r in diagnostic]
    require(
        independent.aggregate(canonical_rows)["formats"] == original["formats"],
        "Canonical full aggregate differs",
    )
    counts.review(canonical_rows, original)
    sensitivities = {}
    for mode in ("original_sealed_points", "independent_equivalent_points"):
        rows = []
        for r in diagnostic:
            metrics = {"current": r["metrics"]["current"]}
            for arm in ("whole", "zoom"):
                cut = (
                    points[r["format"]][arm]["minimum_active_fraction"]
                    if mode == "original_sealed_points"
                    else alternate_points[r["format"]][arm]
                )
                response = r["responses"][arm]
                metrics[arm] = response["ungated"] if response["fraction"] >= cut else r["zero"]
            rows.append(
                {
                    **{k: r[k] for k in ("id", "source_id", "family", "kind", "format", "label")},
                    "metrics": metrics,
                }
            )
        result = independent.aggregate(rows)
        counts.review(rows, result)
        sensitivities[mode] = result["formats"]
    receipt = {
        "schema": "chipmunk.image-iml-zoom-boundary-sensitivity/16",
        "post_result_diagnostic": True,
        "original_strict_reconstruction": "failed; unchanged",
        "original_strict_profile_sha256": run.sha(trial / "review-profile/profile.json"),
        "original_strict_log_sha256": run.sha(strict_log),
        "result_sha256": run.sha(trial / "result.json"),
        "reviewer_sha256": run.sha(Path(__file__)),
        "diagnostic_freeze_sha256": run.sha(lock_path),
        "source_file_cases": 384,
        "recorded_uint8_grids": 1920,
        "canonical_diagnostic_arm_metrics_exact": 864,
        "all_canonical_components_and_metrics_exact": True,
        "maximum_independent_continuous_map_difference": max_error,
        "changed_quantized_cells": cells,
        "ungated_metric_mismatch_cases": metric_cases,
        "breakdown": sections,
        "independent_genuine_points": alternate_points,
        "all_genuine_points_bitwise_exact": all(
            alternate_points[f][a] == points[f][a]["minimum_active_fraction"]
            for f in run.FORMATS
            for a in ("whole", "zoom")
        ),
        "sensitivity_formats": sensitivities,
        "all_sensitivity_replacement_decisions_rejected": all(
            not f["qualifies_fresh_confirmation"] for v in sensitivities.values() for f in v.values()
        ),
        "new_neural_forwards": 0,
        "outside_human_reviewers": 0,
        "serving_admitted": False,
        "scope": "Complete source/grid/component replay and floating-order sensitivity; does not pass the original exact independent-mask audit or rerun native model outputs.",
    }
    run.write(output, receipt)
    print(
        json.dumps({k: v for k, v in receipt.items() if k not in ("breakdown", "sensitivity_formats")}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    review(args.trial, args.output)
