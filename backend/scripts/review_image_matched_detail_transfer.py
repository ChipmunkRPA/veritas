# SPDX-License-Identifier: Apache-2.0
"""Independently replay content-matched image detail decisions and native overlaps."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.image-matched-detail-transfer/1"
NEW = ("spatial", "matched")
OLD = ("sharp_boundary", "research_baseline", "previous_tile_agreement")
HELD_ROLES = {"held_lama": (12, ("original", "edited", "feathered_edit", "tone", "local_blur", "resample")),
              "held_generated": (6, ("original", "edited", "feathered_edit", "global_processing", "feathered_processing")),
              "screen": (12, ("template", "photo", "edit"))}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def exact_overlap(row, entry):
    pixels = entry["pixels"]
    if entry["box"] is None:
        area = 0
        require(entry["score"] is None and entry["flag"] is False, "Unassessed proposal claims an outcome")
    else:
        x0, y0, x1, y1 = entry["box"]
        height, width = row["shape"]
        require(all(type(n) is int for n in entry["box"]) and 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height,
                "Candidate rectangle is outside the source")
        area = (x1 - x0) * (y1 - y0)
        require(math.isfinite(entry["score"]), "Proposed region lacks a finite response")
    intersection, known = pixels["overlap"], row["target"]
    require(pixels["reported"] == area and pixels["target"] == known and 0 <= intersection <= min(area, known),
            "Native pixel counts do not describe the proposed region")
    return bool(known and Fraction(intersection, known) >= Fraction(1, 2)
                and Fraction(intersection, area + known - intersection) >= Fraction(3, 10))


def check_proposal(entry, design):
    if entry["box"] is None:
        return
    x0, y0, x1, y1 = entry["box"]
    cell = design["cell"]
    shape = ((x1 - x0) // cell, (y1 - y0) // cell)
    require(not any(n % cell for n in entry["box"]) and set(shape) <= set(design["rectangle_side_cells"])
            and max(shape) <= 2 * min(shape), "Region is outside the fixed native grid")
    feature, direction = entry["feature"].split(":")
    require(feature in design["features"] and direction in {"higher", "lower"}, "Unsupported residual feature or direction")
    width, height = shape
    cells, valid, agreeing = width * height, entry["eligible_cells"], entry["agreeing_cells"]
    require(cells >= valid >= Fraction(4, 5) * cells and valid >= agreeing >= Fraction(3, 5) * valid,
            "Region lacks the minimum count of agreeing cells")
    score = (entry["core"] - max(0., entry["ring"]) / 2) * (cells / 9.) ** design["area_power"]
    require(math.isclose(score, entry["score"], rel_tol=1e-12, abs_tol=1e-12), "Response does not reproduce from core and ring")


def matched_burden(rows):
    results = {}
    for arm in NEW:
        by_score = defaultdict(list)
        for row in rows:
            score = row["responses"][arm]["score"]
            if score is not None:
                by_score[score].append(row)
        false, useful_count = set(), 0
        at_budget = {str(n): 0 for n in (0, 1, 2, 4, 8)}
        for score in sorted(by_score, reverse=True):
            for row in by_score[score]:
                if row["target"]:
                    useful_count += int(exact_overlap(row, row["responses"][arm]))
                else:
                    false.add(tuple(row["key"][:2]))
            for limit in at_budget:
                if len(false) <= int(limit):
                    at_budget[limit] = useful_count
        results[arm] = at_budget
    return results


def verify(data):
    require(data["schema"] == SCHEMA and set(data["calibration_points"]) == set(NEW), "Unknown benchmark or arm")
    design = data["design"]
    rows = data["held"]
    cal = data["calibration"]
    expected = {("cal", i, role) for i in range(8) for role in ("original", "tone", "blur", "resample")}
    require(len(cal) == len(expected) and {tuple(row["key"]) for row in cal} == expected, "Calibration rows are missing")
    expected = {(part, i, role) for part, (n, roles) in HELD_ROLES.items() for i in range(n) for role in roles}
    require(len(rows) == len(expected) and {tuple(row["key"]) for row in rows} == expected, "Held rows are missing")
    points = data["calibration_points"]
    for arm in NEW:
        ordered = sorted(max((r["responses"][arm]["score"] or 0.) for r in cal if r["key"][1] == i) for i in range(8))
        require(points[arm] == ordered[-2], "Operating point differs from eight independent calibration groups")
    table = defaultdict(lambda: defaultdict(Counter))
    controls = defaultdict(set)
    changes = {"added_primary_feathered": [], "added_other_feathered": [], "lost_useful": []}
    for row in rows:
        role = row["key"][-1]
        stage = row["key"][0]
        positive = role in {"edited", "feathered_edit", "edit"}
        require(positive == bool(row["target"]) and set(row["responses"]) == set(NEW + OLD), "Held labels or arms differ")
        outcomes = {}
        for arm, entry in row["responses"].items():
            overlaps = exact_overlap(row, entry)
            if arm in NEW:
                check_proposal(entry, design)
                require(entry["flag"] == (entry["score"] is not None and entry["score"] > points[arm]),
                        "Flag does not use the frozen review point")
            outcomes[arm] = bool(entry["flag"] and overlaps)
            table[f"{stage}/{role}"][arm].update({"cases": 1, "assessed": entry["score"] is not None,
                                                  "flagged": entry["flag"], "useful": outcomes[arm],
                                                  "useful_before_point": overlaps})
            if not positive and entry["flag"]:
                controls[arm].add(tuple(row["key"][:2]))
        if outcomes["research_baseline"] and not outcomes["matched"]:
            changes["lost_useful"].append(row["key"])
        if outcomes["matched"] and not outcomes["research_baseline"] and role in {"feathered_edit", "edit"}:
            changes["added_primary_feathered" if stage == "held_lama" else "added_other_feathered"].append(row["key"])
    burden = controls["matched"] - controls["research_baseline"]
    ui = sum(key[0] == "screen" for key in burden)
    photo = len(burden) - ui
    gate = design["gate"]
    passed = (len(changes["added_primary_feathered"]) >= gate["minimum_added_primary_feathered"]
              and len(changes["added_other_feathered"]) >= gate["minimum_added_other_feathered"]
              and len(changes["lost_useful"]) <= gate["maximum_lost_useful"]
              and photo <= gate["maximum_extra_photo_groups"] and ui <= gate["maximum_extra_screen_groups"])
    computed = {"by_panel_role": {key: dict(counts) for key, counts in table.items()},
                "ordinary_groups": {arm: dict(Counter(part for part, _ in controls[arm])) for arm in NEW + OLD},
                "candidate_oracle_useful": sum(row["geometry"]["useful_proposals"] > 0 for row in rows if row["target"]),
                "changes_against_research_baseline": changes, "extra_photo_groups": photo,
                "extra_screen_groups": ui, "passed": passed}
    require(computed == data["result"], "Held counts do not reproduce")
    require(matched_burden(rows) == data["retrospective_at_equal_group_budgets"], "Equal-burden ranking does not reproduce")
    return len(rows) * len(NEW + OLD), len(cal)


def private_report(run):
    seal, calibration, result = (read(run / name) for name in ("freeze.json", "calibration.json", "result.json"))
    require(calibration["freeze_sha256"] == result["freeze_sha256"] == digest(run / "freeze.json")
            and result["calibration_sha256"] == digest(run / "calibration.json")
            and result["held_sha256"] == digest(run / "held.json"), "Original run commitments differ")
    held = read(run / "held.json")
    return {"schema": SCHEMA, "design": seal["design"], "commitments": {
                name: digest(run / name) for name in ("freeze.json", "calibration.json", "held.json", "result.json")},
            "source_benchmark_sha256": seal["source_commitment"], "calibration_points": calibration["points"],
            "calibration": [{"key": r["key"], "responses": {a: {"score": v["score"]} for a, v in r["responses"].items()}}
                            for r in calibration["rows"]],
            "held": held, "result": result["result"], "retrospective_at_equal_group_budgets": matched_burden(held)}


def native_readback(args, output):
    from types import SimpleNamespace

    import cv2
    import numpy as np

    from scripts import image_matched_detail_transfer as trial

    cv2.setNumThreads(1)
    local, default_tile = trial.defaults()
    config = SimpleNamespace(commons=local / "image-affine-native-scale-2026-11-11",
        primary=local / "image-bg-real-acquisition-v225", backup=local / "image-bg-real-backup-intake-v262",
        legacy=local / "image-residual-patch-held-2026-11-11", previous=local / "image-crop-layout-2026-11-12",
        alternate=args.alternate, tile_run=args.tile_run or default_tile)
    freeze = read(args.native / "freeze.json")
    require(trial.freeze(config) == freeze, "Native replay changed the source or measurement contract")
    cal_rows = read(args.native / "calibration.json")["rows"]
    masks, ranges, maps = 0, 0, 0
    phases = ((trial.CALIBRATION_PANELS, cal_rows), (trial.HELD_PANELS, output["held"]))
    for names, recorded in phases:
        for ((key, _, _, rgb, truth), witness), row in zip(trial.joined_cases(config, names, freeze), recorded, strict=True):
            require(tuple(row["key"]) == key and int(np.count_nonzero(truth)) == witness["target"], "Native mask identity differs")
            masks += 1
            for entry in row["responses"].values():
                if entry["box"] is None:
                    continue
                x0, y0, x1, y1 = entry["box"]
                actual = sum(np.count_nonzero(line) for line in truth[y0:y1, x0:x1])
                require(actual == entry["pixels"]["overlap"], "Source-mask overlap differs at the recorded crop")
                ranges += 1
            for arm, response in zip(NEW, trial.references(*trial.measurements(rgb)), strict=True):
                entry = row["responses"][arm]
                pin = hashlib.sha256(np.asarray(response, dtype="<f8").tobytes()).hexdigest()
                require(pin == entry["score_map_sha256"], "Native residual map differs")
                maps += 1
                if entry["box"] is None:
                    continue
                x0, y0, x1, y1 = (p // freeze["design"]["cell"] for p in entry["box"])
                feature, direction = entry["feature"].split(":")
                raw = response[..., freeze["design"]["features"].index(feature)]
                raw = raw if direction == "higher" else -raw
                center = raw[y0:y1, x0:x1]
                sides = np.r_[raw[y0 - 1, x0 - 1:x1 + 1], raw[y1, x0 - 1:x1 + 1], raw[y0:y1, x0 - 1], raw[y0:y1, x1]]
                require(math.isclose(float(np.nanmean(center)), entry["core"], abs_tol=1e-12)
                        and math.isclose(float(np.nanmean(sides)), entry["ring"], abs_tol=1e-12)
                        and int(np.isfinite(center).sum()) == entry["eligible_cells"]
                        and int((center > .75).sum()) == entry["agreeing_cells"], "Exact native region response differs")
    return {"native_mask_views": masks, "native_region_overlaps": ranges, "native_response_maps": maps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--alternate", type=Path)
    parser.add_argument("--tile-run", type=Path)
    args = parser.parse_args()
    data = private_report(args.native) if args.native else read(args.report)
    decisions, calibration = verify(data)
    audit = {"held_decisions": decisions, "calibration_views": calibration}
    if args.native:
        require(args.alternate is not None, "Native replay needs the qualified alternate-generator archive")
        audit |= native_readback(args, data)
        with args.report.open("x") as file:
            json.dump(data | {"audit": audit}, file, sort_keys=True, indent=2, allow_nan=False)
            file.write("\n")
    print(audit)


if __name__ == "__main__":
    main()
