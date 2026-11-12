# SPDX-License-Identifier: Apache-2.0
"""Recompute crop, layout and intervention decisions without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path

SCHEMA = "chipmunk.image-crop-layout-transfer/1"
ARMS = ("remote", "layout", "intervention", "combined")
POSITIVE = {"edited", "feathered_edit", "edit"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), "Invalid byte commitment")


def area(box):
    return max(0, box[2]-box[0]) * max(0, box[3]-box[1])


def intersection(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def candidate_boxes(size):
    width, height = size

    def edges(length, start):
        return (length*start+23)//24, (length*(start+8)+23)//24

    return [[left, top, right, bottom] for top, bottom in (edges(height, i) for i in range(0, 17, 4))
            for left, right in (edges(width, j) for j in range(0, 17, 4))]


def measure(counts):
    target, selected, shared = counts
    return target > 0 and shared/(target+selected-shared) >= .30 and shared/target >= .50


def check_overlap(box, counts, size, positive):
    target, selected, shared = counts
    require(box[0] >= 0 and box[1] >= 0 and box[2] <= size[0] and box[3] <= size[1]
            and selected == area(box) > 0 and (target > 0) == positive and all(type(v) is int for v in counts)
            and 0 <= shared <= min(selected, target) <= max(selected, target) <= math.prod(size),
            "Native region overlap is inconsistent")


def geometry(row, lock):
    response, logits = row["response"], row["logits"]
    points = lock["baseline_points"]
    boxes = candidate_boxes(row["size"])
    require(len(logits) == len(boxes) == 25 and all(math.isfinite(x) for x in logits), "Incomplete crop scores")
    index = max(range(len(logits)), key=logits.__getitem__)
    peak = boxes[index]
    relative = logits[index] - statistics.median(logits)
    remote = statistics.median([score for box, score in zip(boxes, logits, strict=True) if intersection(box, peak) == 0])
    selected = relative >= points["relative"]
    require(response == {"box": peak, "relative": relative, "remote_median": remote, "relative_only": selected,
                         "remote_control": selected and remote < points["remote_upper_strict"]}, "Frozen remote crop rule differs")
    check_overlap(peak, row["counts"], row["size"], row["role"] in POSITIVE)
    if boundary := row["sharp_boundary"]:
        require(math.isfinite(boundary["score"]) and 0 <= boundary["score"] <= 3.00001, "Invalid native boundary response")
        check_overlap(boundary["box"], boundary["counts"], row["size"], row["role"] in POSITIVE)


def layout_veto(row, rule):
    layout = row["layout"]
    return (layout["aligned_rows"] >= rule["minimum_aligned_rows"] and layout["word_coverage"] >= rule["minimum_layout_coverage"]
            and layout["baseline_spread"] >= rule["minimum_baseline_spread"])


def check_rows(result):
    lock = result["freeze"]
    require(lock["schema"] == result["schema"] == SCHEMA, "Unexpected study schema")
    require(sha((json.dumps(lock, sort_keys=True, indent=2)+"\n").encode()) == result["freeze_sha256"], "Frozen plan changed")
    fit, held = result["fit"]["rows"], result["held"]
    require(len(fit) == 24 and len(held) == 164, "Incomplete experimental roles")
    counts = Counter()
    unique = set()
    screen = {(row["slot"], row["role"]): row for row in lock["screens"]}
    for row in (*fit, *held):
        key = (row["domain"], row["slot"], row["role"])
        require(key not in unique, "Repeated source role")
        unique.add(key)
        counts[row["domain"]] += 1
        for field in ("input_sha256", "truth_sha256", "crop_sha256"):
            digest(row[field])
        if row["domain"] == "screen":
            expected = screen.pop(key[1:])
            require((row["input_sha256"], row["truth_sha256"]) == (expected["input"], expected["truth"]), "Chromium fixture differs")
        geometry(row, lock)
        perturbation = row["intervention"]
        expected = row["domain"] == "fit" or row["response"]["remote_control"]
        require(bool(perturbation) == bool(row["layout"]) == expected, "Unexpected supplemental input selection")
        if perturbation:
            require(perturbation["outer"]-perturbation["inner"] == perturbation["effect"]
                    and math.isfinite(perturbation["effect"]) and len(perturbation["inputs"]) == 2,
                    "Intervention response arithmetic differs")
            for value in perturbation["inputs"]:
                digest(value)
    require(counts == {"fit": 24, "patch": 56, "lama": 72, "screen": 36} and not screen, "Source group roles differ")
    maxima = {str(slot): max(row["intervention"]["effect"] for row in fit if row["slot"] == slot) for slot in range(8)}
    threshold = max(0., sorted(maxima.values())[-2])
    require(result["fit"]["points"] == {"strict_point": threshold, "maxima": maxima}, "Pre-held fit point changed")
    return threshold


def independent_summary(rows, threshold, lock):
    groups, counts = defaultdict(set), defaultdict(Counter)
    for row in rows:
        domain, slot, role = (row[key] for key in ("domain", "slot", "role"))
        flag = row["response"]["remote_control"]
        excluded = flag and layout_veto(row, lock["runtime"])
        intervention = flag and row["intervention"]["effect"] > threshold
        arms = {"remote": flag, "layout": flag and not excluded, "intervention": intervention,
                "combined": intervention and not excluded}
        for arm in ARMS:
            selected = arms[arm]
            useful = selected and measure(row["counts"])
            counts[f"{domain}/{role}/{arm}"].update({"views": 1, "flags": selected, "useful": useful, "layout_confounds": excluded})
            if selected and role not in POSITIVE:
                if domain == "screen":
                    groups[f"screen_controls/{arm}"].add(slot)
                else:
                    counts["photo_control_flags"][arm] += 1
            if role == "feathered_edit" and useful:
                counts["feathered_useful"][arm] += 1
                groups[f"feathered_families/{arm}"].add(domain)
        sharp = row["sharp_boundary"]
        selected = bool(sharp and sharp["score"] > .8292734622955322)
        counts[f"{domain}/{role}/sharp_boundary"].update({"views": 1, "assessed": bool(sharp), "flags": selected,
                                                          "useful": selected and measure(sharp["counts"])})
        if selected and domain != "screen" and role not in POSITIVE:
            counts["photo_control_flags"]["sharp_boundary"] += 1
    criteria = lock["criteria"]
    advances = (counts["feathered_useful"]["combined"] >= criteria["minimum_useful_feathered"]
                and len(groups["feathered_families/combined"]) >= criteria["minimum_feathered_source_families"]
                and counts["photo_control_flags"]["combined"] - counts["photo_control_flags"]["sharp_boundary"] <= criteria["maximum_extra_ordinary_photo_views"]
                and len(groups["screen_controls/combined"]) <= criteria["maximum_control_screenshot_groups"]
                and len(groups["screen_controls/remote"]) >= criteria["minimum_ordinary_screenshot_challenge_groups"]
                and counts["screen/edit/combined"]["useful"] >= criteria["minimum_useful_screen_edits"]
                and counts["screen/edit/remote"]["useful"] - counts["screen/edit/combined"]["useful"] <= criteria["maximum_lost_useful_screen_edits"])
    return {"counts": {key: dict(value) for key, value in counts.items()}, "groups": {key: sorted(value) for key, value in groups.items()},
            "advances": advances}


def audit_native(result, args):
    import cv2

    from app.detectors.image import sharp_boundary
    from scripts import image_crop_layout_transfer as study

    cv2.setNumThreads(1)
    lock = result["freeze"]
    require(study.freeze(args) == lock, "Native source freeze differs")
    measurements = {(row["domain"], row["slot"], row["role"]): row for row in (*result["fit"]["rows"], *result["held"])}
    photo_oracle = Counter()
    layout_diagnostic = defaultdict(Counter)
    perturbations = 0
    for inputs in chain(study.fit_views(args, lock), study.held_views(args)):
        domain, slot, role, data, rgb, truth, saved = inputs
        row = measurements.pop((domain, slot, role))
        require((sha(data), sha(truth.tobytes()), study.overlap(row["response"]["box"], truth))
                == (row["input_sha256"], row["truth_sha256"], row["counts"]), "Exact edited-pixel commitments differ")
        if saved:
            require(row["logits"] == saved["logits"], "Frozen original crop responses changed")
        boxes = candidate_boxes(row["size"])
        with study.Image.fromarray(rgb).crop(tuple(row["response"]["box"])) as native:
            crop = study.np.asarray(native.resize((384, 384), study.Image.Resampling.BILINEAR))
        require(sha(crop.tobytes()) == row["crop_sha256"], "The exact native source crop changed")
        if role in POSITIVE:
            key = f"{domain}/{role}"
            photo_oracle[key] += any(measure(study.overlap(box, truth)) for box in boxes)
        boundary = row["sharp_boundary"]
        boundary_flag = bool(boundary and boundary["score"] > sharp_boundary.REVIEW_POINT)
        inspect_layout = bool(row["intervention"] or (domain == "screen" and (boundary_flag or row["response"]["relative_only"])))
        words = study.ocr_boxes(args, data) if inspect_layout else []
        if row["intervention"]:
            require(row["intervention"]["inputs"] == [sha(item.tobytes()) for item in study.stencils(crop)], "Perturbed crop pixels differ")
            require(row["layout"] == study.layout(words, row["response"]["box"]), "Native OCR layout changed")
            perturbations += 2
        if domain == "screen":
            for arm, active, proposal in (("relative_only", row["response"]["relative_only"], row["response"]),
                                          ("sharp_boundary", boundary_flag, boundary)):
                layout_diagnostic[f"{role}/{arm}"].update({"flags": active, "layout_matches": bool(
                    active and study.layout_confounded(study.layout(words, proposal["box"])))})
        if boundary:
            fixed = sharp_boundary.measure(rgb)
            require(boundary == {"score": fixed.score, "box": list(fixed.box), "counts": study.overlap(fixed.box, truth)}, "Incumbent source crop differs")
    require(not measurements, "Native study inputs incomplete")
    return {"native_input_views": 188, "intervention_crop_inputs": perturbations, "geometry_oracle_useful": dict(photo_oracle),
            "diagnostic_screen_layout": {key: dict(value) for key, value in layout_diagnostic.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--native", nargs=8, type=Path)
    options = parser.parse_args()
    record = json.loads(options.receipt.read_bytes())
    point = check_rows(record)
    counts = independent_summary(record["held"], point, record["freeze"])
    require(record["counts"] == counts, "Held counts differ")
    value = {"views": 188, "frozen_point": point, "counts": counts}
    if options.native:
        names = ("commons", "lama", "generated", "baseline", "weight", "tesseract", "tessdata", "run")
        args = argparse.Namespace(**dict(zip(names, options.native, strict=True)))
        value["native_replay"] = audit_native(record, args)
    print(json.dumps(value, indent=2, sort_keys=True))
