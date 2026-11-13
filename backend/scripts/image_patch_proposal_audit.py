# SPDX-License-Identifier: Apache-2.0
"""Audit blind proposal limits and native earlier-copy differences on prior JPEG studies."""

from __future__ import annotations

import argparse
import io
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import reference_alignment
from app.evidence import image_reference
from scripts import image_residual_color_transfer as color
from scripts import image_three_class_patch_fresh as fresh
from scripts import image_three_class_patch_probe as heads

SCHEMA = "chipmunk.image-patch-proposal-audit/1"
POINT = 0.15636499383156965
DELIVERIES = ("earlier_png", "earlier_jpeg90")
DESIGN = {
    "unpaired": "Top existing two-band RGB residual rectangle, with and without its unchanged JPEG75 operating point.",
    "paired": "Same-framing production earlier-copy route using exact earlier RGB in PNG or its single JPEG90 delivery.",
    "truth": "Original native pre-final-encode changed pixels. At least 30% IoU and 50% recall.",
    "paired_measurement": "Union of actual masks emitted by production, including the exact returned source coordinates.",
    "oracle": "Exhaustive allowed 14-pixel grid boxes and unchanged head geometry, with true mask supplied only to the audit.",
    "proposal_union": "Set union of saved three-class top box and existing independent color top box, without threshold; zero model forwards.",
    "scope": "Development using two evaluated panels; no model scoring, calibration, serving or confidence changes.",
}
read, write, require, sha = heads.read, heads.write, heads.require, heads.sha


def can_cover(exact):
    target = int(exact.sum())
    if not target:
        return False
    rows, cols = (size // heads.PATCH for size in exact.shape)
    table = np.pad(exact.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    for h in heads.DESIGN["geometry"]["patch_sides"]:
        for w in heads.DESIGN["geometry"]["patch_sides"]:
            if max(h, w) > 2*min(h, w) or h*w > .10*rows*cols or h + 2 > rows or w + 2 > cols:
                continue
            x, y = np.meshgrid(np.arange(1, cols - w), np.arange(1, rows - h))
            x, y = x * heads.PATCH, y * heads.PATCH
            covered = heads.base.sums(table, x, y, x + w * heads.PATCH, y + h * heads.PATCH)
            if np.any((2 * covered >= target) & (10 * covered >= 3 * (target + w*h*heads.PATCH**2 - covered))):
                return True
    return False


def paired(reference, submitted, submitted_path, out, exact):
    report = image_reference.inspect(reference, submitted_path, evidence_dir=out, allow_proportional_resize=True,
                                     allow_oriented_crop=True, allow_oriented_resize=True, allow_embedded_photo=True)
    require(report["reference"]["sha256"] == fresh.roster.sha(reference) and report["submitted"]["sha256"] == submitted,
            "Earlier-copy source digest changed")
    mask = np.zeros(exact.shape, dtype=bool)
    record = report.get("mask")
    if record is not None:
        path = out / record["file"]
        require(record["sha256"] == fresh.roster.sha(path), "Earlier-copy mask digest changed")
        with Image.open(path) as image:
            values = np.asarray(image)
        require(values.shape == exact.shape and np.isin(values, (0, 255)).all(), "Invalid native mask")
        mask = values > 0
        require(int(mask.sum()) == sum(region["selected_pixels"] for region in report["regions"]), "Native region support differs")
        for region in report["regions"]:
            x, y, w, h = (region[key] for key in ("x", "y", "w", "h"))
            require(0 <= x < x + w <= exact.shape[1] and 0 <= y < y + h <= exact.shape[0], "Native region outside source")
    else:
        require(not report["regions"], "Unexpected region without a native mask")
    return {"status": report["status"], "selected": int(mask.sum()), "target": int(exact.sum()),
            "intersection": int((mask & exact).sum()), "mask_sha256": sha(mask.tobytes())}


def choose_points():
    point_file = Path("docs/evaluation-panels/image-residual-color-transfer-points-2026-11-11.json")
    require(read(point_file)["points"]["jpeg75"]["mid_agreement"]["point"] == POINT, "Frozen color point differs")
    return fresh.roster.sha(point_file)


def panel_data(args, lock):
    args.later = args.later_f5
    prior_rows = read(args.prior_probe / "held.json")
    require(len(prior_rows) == 160, "Expected exact prior held rows")
    prior_original = {item["recipient"]: values["original"]
                      for item, values, _, _ in heads.bound_cases(args, lock["profile"], "held")}
    prior_views = heads.views(args, lock["profile"], "held")
    yield "earlier", ((saved, key, body, truth, prior_original[key["source"]])
                      for saved, (key, body, truth) in zip(prior_rows, prior_views, strict=True))
    new_plan, known = fresh.planned(args)
    require(new_plan == read(args.output_fresh / "plan.json"), "Fresh roster freeze differs")
    new_rows = read(args.output_fresh / "measurements.json")
    require(len(new_rows) == 96, "Expected exact transfer rows")
    new_views = ((key, body, truth, native["original"]) for item, native, target in fresh.groups(args.output_fresh, new_plan, known)
                 for key, body, truth, _ in fresh.views(item, native, target))
    yield "newer", ((saved, key, body, truth, original)
                    for saved, (key, body, truth, original) in zip(new_rows, new_views, strict=True))


def run(args, lock):
    cv2.setNumThreads(1)
    findings = []
    references = {}
    for panel, rows in panel_data(args, lock):
        for saved, identity, body, truth, original in rows:
            require(all(saved[k] == v for k, v in identity.items()), "JPEG or exact change mask differs")
            rgb = heads.base.decode(body)
            candidate = color.response(rgb)["mid_agreement"] if min(rgb.shape[:2]) >= 256 else None
            box = candidate["box"] if candidate else None
            source = saved["source"]
            if source not in references:
                with io.BytesIO() as stream:
                    Image.fromarray(original).save(stream, format="PNG")
                    references[source] = (stream.getvalue(), heads.base.encode(original, 90))
            with tempfile.TemporaryDirectory() as workspace:
                root = Path(workspace)
                submitted = root / "submitted.jpg"
                submitted.write_bytes(body)
                comparison = {}
                for label, earlier in zip(DELIVERIES, references[source], strict=True):
                    reference = root / f"{label}.image"
                    reference.write_bytes(earlier)
                    comparison[label] = paired(reference, submitted=identity["input_sha256"],
                                               submitted_path=submitted, out=root / label, exact=truth)
            color_overlap = heads.overlap(truth, box)
            findings.append(identity | {"panel": panel, "paired": comparison, "oracle": can_cover(truth),
                "color_box": box, "color_overlap": color_overlap, "color_flag": bool(candidate and candidate["score"] > POINT),
                "color_useful": heads.base.geometry.useful(color_overlap), "prior_three_class": saved["arms"]["three_class"],
                "prior_luma": saved["arms"]["current_luma"]})
        print("completed", panel, flush=True)
    require(len(findings) == 256, "Expected 256 matched view deliveries")
    return findings


def summary(rows):
    results, flagged_groups = defaultdict(Counter), defaultdict(set)
    for row in rows:
        role = row["role"]
        key = f"{row['panel']}/{row['family']}/{role}"
        positive = role in heads.HELD_POSITIVE
        prior = row["prior_three_class"]
        color_flag = row["color_flag"]
        results[key].update(views=1, geometry_useful=row["oracle"], blind_head_useful_before=prior["useful"],
                            color_useful_before=row["color_useful"], blind_union_useful_before=prior["useful"] or row["color_useful"],
                            color_flags=color_flag, color_useful=color_flag and row["color_useful"])
        if color_flag and not positive:
            flagged_groups[f"{row['panel']}/color"].add(row["source"])
        for label, check in row["paired"].items():
            area, target, match = (check[name] for name in ("selected", "target", "intersection"))
            useful = target > 0 and 10*match >= 3*(area + target - match) and 2*match >= target
            results[key].update({f"{label}_comparable": check["status"] == "comparable", f"{label}_flags": area > 0,
                                 f"{label}_useful": useful, f"{label}_misplaced": positive and area > 0 and not useful})
            if area and not positive:
                flagged_groups[f"{row['panel']}/{label}"].add(row["source"])
    return {"strata": results, "ordinary_flagged_groups": {key: len(values) for key, values in flagged_groups.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    for arg in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint", "output-fresh", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    args.exclude = []
    lock, _, _ = fresh.previous(args)
    commitments = {
        "prior_held": fresh.roster.sha(args.prior_probe / "held.json"),
        "new_held": fresh.roster.sha(args.output_fresh / "measurements.json"),
        "code": {Path(path).name: fresh.roster.sha(Path(path))
                 for path in (__file__, color.__file__, reference_alignment.__file__, image_reference.__file__)},
        "color_points": choose_points(),
    }
    require(commitments["prior_held"] == read(args.prior_probe / "result.json")["commitments"]["held.json"]
            and commitments["new_held"] == read(args.output_fresh / "results.json")["measurements_sha256"],
            "Earlier measured rows changed")
    freeze = args.output / "freeze.json"
    design = {"schema": SCHEMA, "design": DESIGN, "commitments": commitments}
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(freeze, design)
        return
    require(read(freeze) == design, "Proposal audit freeze differs")
    start = time.monotonic()
    rows = run(args, lock)
    result = design | summary(rows) | {"deliveries": len(rows), "model_calls": 0, "paired_calls": 2*len(rows)}
    if args.stage == "run":
        write(args.output / "rows.json", rows)
        write(args.output / "result.json", result)
        print("seconds", round(time.monotonic() - start, 2), flush=True)
        return
    require(args.publication is not None, "Aggregate destination required")
    require(rows == read(args.output / "rows.json") and result == read(args.output / "result.json"),
            "Actual replay differs")
    write(args.publication, result | {"verified_exact_replays": len(rows),
                                    "private_rows_sha256": fresh.roster.sha(args.output / "rows.json")})
    print("replay seconds", round(time.monotonic() - start, 2), flush=True)


if __name__ == "__main__":
    main()
