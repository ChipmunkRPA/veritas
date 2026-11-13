# SPDX-License-Identifier: Apache-2.0
"""Measure face, head and lower-picture change around existing face cue boundaries."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from scripts import video_face_hd_retry_scene_challenge as source
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
SPEC = {
    "schema": "chipmunk.video-face-subject-context/1",
    "input": "The previously frozen 61 exact 720p30 H.264 deliveries and completed Full face results",
    "development_groups": {"grid": [1, 3, 5, 7], "rtfs": [1, 3, 5]},
    "confirmation_groups": {"grid": [2, 4, 6, 8], "rtfs": [2, 4, 6, 7]},
    "report_interval": ["short_range", "boundary_refinement", "coarse"],
    "boundary_radius_frames": 8,
    "geometry_offsets_frames": [-9, -6, -3, 3, 6, 9],
    "geometry_rule": "Median of at least four single-face flank observations with boxes overlapping the median by half",
    "inner_ellipse": [0.43, 0.47],
    "head_annulus": [0.82, 0.88],
    "lower_box_relative_to_face_center": [-0.66, 0.86, 0.66, 1.55],
    "minimum_region_fraction_inside_picture": 0.8,
    "output_resolution": [96, 128],
    "motion_reference": "Median absolute luma change of adjacent pairs in the 12 frames outside each boundary-search interval",
    "selection": "Largest positive central face excess in each nonoverlapping boundary window, before considering other areas",
    "minimum_face_luma_excess": 4.0,
    "minimum_head_luma_excess": 3.0,
    "minimum_lower_luma_excess": 3.0,
    "minimum_lower_to_face_excess_ratio": 0.25,
    "rule": "Both boundaries must show sufficient central, head annulus and lower-region excess. "
            "A no-broad-change result gives no positive claim. Authored times are used only by the separate timing diagnostic.",
    "gate": {"minimum_cutaway_sources_qualified": 3, "minimum_reported_swap_fraction_assessed": 0.75,
             "maximum_swap_deliveries_with_broad_annotation": 1},
}


def plan(parent):
    from app.detectors.video import decode

    sparse_lock = read(parent/"runtime.json")
    source_ledger = read(parent/"media.json")
    require(source_ledger["freeze_sha256"] == sha(parent/"freeze.json")
            and sparse_lock["held_sha256"] == sha(parent/"held.jsonl"), "Input comparison changed")
    require(len(source_ledger["rows"]) == 61, "Complete 61-delivery bank required")
    return {"protocol": SPEC, "source_manifest_sha256": sha(parent/"media.json"), "upstream_run": sparse_lock["held_sha256"],
            "runner_sha256": sha(__file__), "decode_sha256": sha(decode.__file__), "opencv": cv2.__version__,
            "face_asset_sha256": sha(ROOT/"models/weights/face_detection_yunet_2023mar.onnx")}


def regions(shape, face):
    height, width = shape[:2]
    x, y, w, h = face
    cx, cy = x + w/2, y + h/2
    lx, ty, rx, by = SPEC["lower_box_relative_to_face_center"]
    a, b = SPEC["head_annulus"]
    left, top = int(np.floor(cx - a*w)), int(np.floor(cy - b*h))
    right, bottom = int(np.ceil(cx + a*w)), int(np.ceil(cy + by*h))
    if right <= left or bottom <= top or w < 48 or h < 48:
        return None
    nw, nh = SPEC["output_resolution"]
    xs = (left+(np.arange(nw)+.5)*(right-left)/nw-cx)/w
    ys = (top+(np.arange(nh)+.5)*(bottom-top)/nh-cy)/h
    xx, yy = np.meshgrid(xs, ys)
    ai, bi = SPEC["inner_ellipse"]
    inner = (xx/ai)**2 + (yy/bi)**2 <= 1
    outer = ((xx/a)**2 + (yy/b)**2 <= 1) & ~inner
    lower = (xx >= lx) & (xx <= rx) & (yy >= ty) & (yy <= by) & ~outer
    mapx, mapy = (xx*w+cx-.5).astype("float32"), (yy*h+cy-.5).astype("float32")
    visible = (mapx >= 0) & (mapx <= width-1) & (mapy >= 0) & (mapy <= height-1)
    masks = [mask & visible for mask in (inner, outer, lower)]
    if any(not mask.any() or np.count_nonzero(seen) < SPEC["minimum_region_fraction_inside_picture"]*np.count_nonzero(mask)
           for mask, seen in zip((inner, outer, lower), masks, strict=True)):
        return None
    return mapx, mapy, masks


def transitions(near, count):
    radius = SPEC["boundary_radius_frames"]
    possible = list(range(max(1, near-radius), min(count, near+radius+1)))
    baseline = [i for i in (*range(near-radius-12, near-radius), *range(near+radius+1, near+radius+13)) if 0 < i < count]
    return possible, baseline


def pair_metrics(pictures, intervals, geometry):
    mx, my, masks = geometry
    grayscale = {}
    for frame_index in sorted({i for edge in intervals for i in (edge-1, edge)}):
        reduced = cv2.remap(pictures[frame_index], mx, my, cv2.INTER_LINEAR)
        grayscale[frame_index] = cv2.GaussianBlur(cv2.cvtColor(reduced, cv2.COLOR_RGB2GRAY), (3, 3), 0).astype("float32")
    result = {}
    for i in intervals:
        difference = abs(grayscale[i]-grayscale[i-1])
        result[i] = [float(np.mean(difference[mask])) for mask in masks]
    return result


def face_rectangle(pictures, anchors, detector):
    from app.detectors.image.faces import Face, detect_faces, iou

    boxes = []
    for i in anchors:
        faces = detect_faces(cv2.cvtColor(pictures[i], cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
        if len(faces) == 1:
            boxes.append(faces[0])
    if len(boxes) < 4:
        return None
    rectangle = [float(value) for value in np.median([f.box() for f in boxes], axis=0)]
    if sum(iou(f, Face(*rectangle, 1.0)) >= .5 for f in boxes) < 4:
        return None
    return rectangle


def evaluate(path, table, media, observed, detector, target):
    from app.detectors.video.decode import decode_frames

    if target is None:
        return {"status": "no_reported_range"}
    start, end = target
    count = len(table)
    anchors = sorted({near+step for near, steps in ((start, (-9, -6, -3)), (end, (3, 6, 9))) for step in steps
                      if 0 <= near+step < count})
    windows = [transitions(near, count) for near in target]
    if not all(window and len(ref) >= 6 for window, ref in windows):
        return {"status": "insufficient_timing"}
    edges = set(i for each in windows for group in each for i in group)
    needed = sorted(set(anchors) | {i for edge in edges for i in (edge-1, edge)})
    pictures = {}
    for frame in decode_frames(path, table, needed, timeout_s=30):
        require(frame.pts_verified and frame.pts == table.pts[frame.index]
                and source.review.digest(frame.rgb) == media["rgb_sha256"][frame.index], "Native input picture differs")
        pictures[frame.index] = frame.rgb
    require(set(pictures) == set(needed), "Native picture coverage differs")
    rect = face_rectangle(pictures, anchors, detector)
    if rect is None:
        return {"status": "insufficient_geometry", "frames_verified": len(needed)}
    geometry = regions(pictures[needed[0]].shape, rect)
    if geometry is None:
        return {"status": "lower_region_outside_picture", "frames_verified": len(needed)}
    values = pair_metrics(pictures, edges, geometry)
    center = Fraction(start+end, 2)
    best = []
    for near, (window, reference), is_entry in zip(target, windows, (True, False), strict=True):
        support = [i for i in window if (i < center if is_entry else i >= center)]
        if not support:
            return {"status": "insufficient_timing", "frames_verified": len(needed)}
        baseline = np.median([values[i] for i in reference], axis=0)
        index = max(support, key=lambda i: (values[i][0]-baseline[0], -abs(near-i), -i))
        residual = np.maximum(0., np.asarray(values[index])-baseline)
        inside, annulus, lower = (float(x) for x in residual)
        enough_face = inside >= SPEC["minimum_face_luma_excess"]
        broad = (enough_face and annulus >= SPEC["minimum_head_luma_excess"]
                 and lower >= SPEC["minimum_lower_luma_excess"]
                 and lower >= SPEC["minimum_lower_to_face_excess_ratio"]*inside)
        best.append({"frame_index": index, "before_pts": table.pts[index-1], "after_pts": table.pts[index],
                     "input_hashes": [media["rgb_sha256"][index-1], media["rgb_sha256"][index]],
                     "residual_luma": [inside, annulus, lower], "motion_reference_luma": baseline.tolist(),
                     "adequate_face_transition": enough_face, "broad": broad})
    status = "assessed" if all(r["adequate_face_transition"] for r in best) else "insufficient_face_transition"
    return {"status": status, "broad": status == "assessed" and all(row["broad"] for row in best),
            "target": target, "observed": observed, "face_box": rect, "geometry_frames": anchors,
            "boundaries": best, "frames_verified": len(needed)}


def study(parent, output):
    from app.detectors.image.faces import load_yunet
    from app.detectors.video.decode import FrameTable

    cv2.setNumThreads(1)
    detector = load_yunet(ROOT/"models/weights/face_detection_yunet_2023mar.onnx")
    media = read(parent/"media.json")
    began = time.monotonic()
    with (parent/"held.jsonl").open() as stream, (output/"observations.jsonl").open("x") as out:
        for entry, held in zip(media["rows"], map(json.loads, stream), strict=True):
            require(all(entry[k] == held[k] for k in ("file", "group", "condition", "sha256", "truth")), "Full result binding differs")
            cohort = "grid" if held["condition"] in source.GRID_ROLES else "rtfs"
            split = "development" if entry["group"] in SPEC["development_groups"][cohort] else "confirmation"
            path = parent/"media"/entry["file"]
            require(sha(path) == entry["sha256"], "Source media differs")
            table = FrameTable.from_json(entry["clock"])
            require(source.hd.relative(table) == [Fraction(i, 30) for i in range(120)], "Unexpected source clock")
            observed = next((key for key in SPEC["report_interval"] if held["before"][key]), None)
            native = None if observed is None else source.spans.interval(held["before"][observed], table.pts)
            assay = evaluate(path, table, entry, observed, detector, native)
            truth = entry["authored_span"]
            diagnostic = None if truth is None or truth == native else evaluate(path, table, entry, "authored_boundary", detector, truth)
            if truth is not None and diagnostic is None:
                diagnostic = {"reuse_observed": True}
            print(split, entry["condition"], assay["status"], assay.get("broad", False),
                  "diagnostic", None if diagnostic is None else diagnostic.get("broad", False), flush=True)
            out.write(json.dumps({"condition": entry["condition"], "group": entry["group"], "sha256": entry["sha256"],
                                  "cohort": cohort, "split": split, "truth": truth, "observed": assay,
                                  "diagnostic": diagnostic}, allow_nan=False)+"\n")
    write(output/"runtime.json", {"observations_sha256": sha(output/"observations.jsonl"),
                                 "freeze_sha256": sha(output/"freeze.json"), "elapsed_s": time.monotonic()-began})


def report(output):
    runtime = read(output/"runtime.json")
    require(runtime["observations_sha256"] == sha(output/"observations.jsonl")
            and runtime["freeze_sha256"] == sha(output/"freeze.json"), "Scoring commitment differs")
    totals = {split: {} for split in ("development", "confirmation")}
    qualified_cutaway_sources = set()
    native_checks = 0
    with (output/"observations.jsonl").open() as stream:
        rows = list(map(json.loads, stream))
    for row in rows:
        role, split = row["condition"], row["split"]
        totals[split].setdefault(role, Counter()).update(files=1)
        for kind in ("observed", "diagnostic"):
            case = row[kind]
            if case is None:
                continue
            if case.get("reuse_observed"):
                case = row["observed"]
            else:
                native_checks += case.get("frames_verified", 0)
            data = totals[split][role]
            data[f"{kind}_{case['status']}"] += 1
            if case.get("broad"):
                data[f"{kind}_broad"] += 1
                if kind == "observed" and split == "confirmation" and "person_switch" in role:
                    qualified_cutaway_sources.add(row["group"])
            if row["truth"] and case.get("boundaries"):
                data[f"{kind}_within_two_frames"] += all(abs(b["frame_index"]-truth) <= 2
                    for b, truth in zip(case["boundaries"], row["truth"], strict=True))
    require(len(rows) == 61, "Incomplete scoring result")
    held = totals["confirmation"]
    swaps = [held[role] for role in ("publisher_swap", "tandem_publisher_swap")]
    denominator = sum(s["files"]-s["observed_no_reported_range"] for s in swaps)
    assessed = sum(s["observed_assessed"] for s in swaps)
    broad = sum(s["observed_broad"] for s in swaps)
    gate = {"cutaway_sources": len(qualified_cutaway_sources) >= SPEC["gate"]["minimum_cutaway_sources_qualified"],
            "swap_coverage": bool(denominator and assessed/denominator >= SPEC["gate"]["minimum_reported_swap_fraction_assessed"]),
            "swap_annotations": broad <= SPEC["gate"]["maximum_swap_deliveries_with_broad_annotation"]}
    write(output/"aggregate.json", {"protocol": SPEC, "runtime": runtime, "scores": totals, "native_frame_reads": native_checks,
                                    "held_qualified_cutaway_sources": len(qualified_cutaway_sources),
                                    "held_swap_reported": denominator, "held_swap_assessed": assessed, "held_swap_broad": broad,
                                    "gate": gate, "advances": all(gate.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "report"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700, parents=True)
        write(args.output/"freeze.json", plan(args.source))
        return
    lock = read(args.output/"freeze.json")
    require(lock == plan(args.source), "Frozen scope, input or implementation changed")
    if args.phase == "score":
        study(args.source, args.output)
    else:
        report(args.output)


if __name__ == "__main__":
    main()
