# SPDX-License-Identifier: Apache-2.0
"""Test local motion-compensated mouth detail against matched ordinary edits."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.shots import ShotDetector
from scripts import review_video_mouth_jaw_transfer as review
from scripts import video_mouth_jaw_transfer as panel
from scripts import video_mouth_native_texture_transfer as prior
from scripts.video_common_sources_v89 import read, require, sha, write

ARMS = ("fine", "scale_adjusted")
PRIMARY = "scale_adjusted"
SPEC = {
    "schema": "chipmunk.video-mouth-temporal-detail/1",
    "hypothesis": "A local neural redraw may lose trackable fine surface detail while retaining broad face motion.",
    "cue": "Align the eye positions. Track both ways between adjacent measured pictures. Compare the signed correlation "
           "of motion-aligned narrow and broad luminance bands in the perioral skin and nearby cheek reference. "
           "Exclude the inner mouth and require the tracking map to preserve the same region in both views. "
           "The primary arm subtracts the broad-band gap from the fine-band gap to control general motion.",
    "time": "Single submitted video, same native sample grid and source-only shot checks as the prior comparator. "
            "The soundtrack is checked for evidence and excluded from scoring.",
    "flow": {"parameters": [.5, 3, 13, 3, 5, 1.2, 0], "cycle_error_max_px": 1.25, "displacement_max_px": 12.},
    "ellipse_radii": [[15, 9], [22, 13], [28, 18]],
    "blur_sigmas": [.65, 1.55, 3.1],
    "pixel_support": {"per_band_min_std": .35, "per_region_min": 48},
    "points": "For each arm, strict exceedance of its third-largest measured ordinary development maximum or zero.",
    "coverage_floor": .75,
    "gate": {"grid_added": 4, "grid_groups": 2, "compatible_added": 2, "native_added": 2, "native_groups": 2,
             "grid_extra": 3, "native_extra": 3, "reencode_extra": 1, "hard_cut_extra": 0},
}
Y, X = np.mgrid[:panel.prior.SIZE, :panel.prior.SIZE].astype(np.float32)
XY = np.stack((X, Y), axis=-1)
ROLES = ("before", "during", "after")


def lock(parent):
    return {
        "protocol": SPEC, "matched_bank": prior.seal(parent),
        "implementation_sha256": sha(Path(__file__)),
        "reviewer_sha256": sha(Path(__file__).with_name("review_video_mouth_temporal_detail.py")),
        "baseline_runtime_sha256": sha(parent / "run.json"),
    }


def detail_pictures(gray):
    blurred = [cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma) for sigma in SPEC["blur_sigmas"]]
    return blurred[0] - blurred[1], blurred[1] - blurred[2]


def region_masks(mouth, xx=X, yy=Y):
    dx, dy = xx - mouth[0], yy - mouth[1]
    inner, middle, outer = [(dx / w)**2 + (dy / h)**2 <= 1 for w, h in SPEC["ellipse_radii"]]
    return middle & ~inner, outer & ~middle


def correlate(left, right, valid):
    a, b = left[valid], right[valid]
    if len(a) < SPEC["pixel_support"]["per_region_min"]:
        return None
    a, b = a - a.mean(), b - b.mean()
    norms = np.linalg.norm(a), np.linalg.norm(b)
    if min(norms) < SPEC["pixel_support"]["per_band_min_std"] * np.sqrt(len(a)):
        return None
    return float(np.clip(np.dot(a, b) / (norms[0] * norms[1]), -1., 1.))


def flow(left, right):
    return cv2.calcOpticalFlowFarneback(left, right, None, *SPEC["flow"]["parameters"])


def measure_direction(source, target, forward, backward):
    first, first_mouth, first_bands = source
    _, target_mouth, target_bands = target
    to = XY + forward
    inverse = cv2.remap(backward, to, None, cv2.INTER_LINEAR)
    reliable = ((to[..., 0] >= 1) & (to[..., 0] < first.shape[1] - 2)
                & (to[..., 1] >= 1) & (to[..., 1] < first.shape[0] - 2)
                & (np.linalg.norm(forward, axis=2) <= SPEC["flow"]["displacement_max_px"])
                & (np.linalg.norm(forward + inverse, axis=2) <= SPEC["flow"]["cycle_error_max_px"]))
    same_regions = [a & b & reliable for a, b in zip(region_masks(first_mouth), region_masks(target_mouth, to[..., 0], to[..., 1]), strict=True)]
    counts = [int(mask.sum()) for mask in same_regions]
    warped = [cv2.remap(band, to, None, cv2.INTER_LINEAR) for band in target_bands]
    correlations = [[correlate(a, b, mask) for mask in same_regions] for a, b in zip(first_bands, warped, strict=True)]
    if any(value is None for pair in correlations for value in pair):
        return None
    return {"pixels": counts, "fine": correlations[0], "broad": correlations[1]}


def measure_edge(first, last):
    forward, reverse = flow(first[0], last[0]), flow(last[0], first[0])
    first_view = measure_direction(first, last, forward, reverse)
    last_view = measure_direction(last, first, reverse, forward)
    return None if first_view is None or last_view is None else (first_view, last_view)


def edge_response(edge, arm):
    fine, broad = (sum(view[band][1] - view[band][0] for view in edge) / 2 for band in ("fine", "broad"))
    return fine if arm == "fine" else fine - broad


def edge_observations(path, source, detector, pictures, cache):
    require(sha(path) == source["file_sha256"], "Video differs from the qualified native bytes")
    table, frames = panel.inspect_frames(path, source["expected"])
    times = panel.source_clock(table)
    shot = ShotDetector(table)
    shots, digests, anchors = [], [], []
    for frame, digest in panel.exact_rgb(path, table, frames, source["expected"]):
        if digest not in pictures:
            geometry = panel.prior.aligned_face(frame.rgb, detector)
            if geometry is None:
                pictures[digest] = None
            else:
                aligned, _, mouth, _ = geometry
                box = panel.native_bounds(geometry)
                x0, y0, x1, y1 = box
                pictures[digest] = (aligned, mouth, detail_pictures(aligned)), box, panel.raw_digest(frame.rgb[y0:y1, x0:x1])
        shot.update(frame.index, frame.rgb)
        shots.append(len(shot.boundaries))
        digests.append(digest)
        entry = pictures[digest]
        anchors.append(None if entry is None else {
            "frame": frame.index, "pts": frame.pts, "rgb_sha256": digest, "native_crop_xyxy": entry[1], "crop_rgb_sha256": entry[2],
        })
    edges = []
    for i, (a, b) in enumerate(zip(digests, digests[1:])):
        pair = (a, b)
        elapsed = times[frames[i+1]] - times[frames[i]]
        if shots[i] != shots[i+1] or any(pictures[key] is None for key in pair) or not Fraction(1,25) <= elapsed <= Fraction(3,25):
            edges.append(None)
            continue
        if pair not in cache:
            cache[pair] = measure_edge(pictures[a][0], pictures[b][0])
        edges.append(cache[pair])
    return table, frames, shots, edges, anchors


def core_series(edges, arm):
    values = [None if edge is None else edge_response(edge, arm) for edge in edges]
    return [None, *(None if a is None or b is None else min(a, b) for a, b in zip(values, values[1:])), None]


def select(table, frames, shots, edges, anchors):
    times = panel.source_clock(table)
    results = dict.fromkeys(ARMS)
    for arm in ARMS:
        series = core_series(edges, arm)
        for width in panel.SPEC["core_samples"]:
            for begin in range(4, len(frames) - width - 3):
                end = begin + width
                if shots[begin-3] != shots[end+2]:
                    continue
                groups = [[i for i in part if series[i] is not None] for part in
                          (range(begin-3, begin), range(begin, end), range(end, end+3))]
                before, core, after = groups
                if min(len(before), len(after)) < 2 or len(core) < (4 * width + 4) // 5:
                    continue
                medians = [float(np.median([series[i] for i in group])) for group in groups]
                score = medians[1] - max(medians[0], medians[2])
                if results[arm] is not None and score <= results[arm]["score"]:
                    continue
                center = min(core, key=lambda i: (abs(series[i] - medians[1]), i))
                chosen = (before[-1], center, after[0])
                results[arm] = {
                    "score": score, "frames": [frames[begin], frames[end]], "time_base": table.time_base,
                    "source_clock_s": [str(times[frames[begin]]), str(times[frames[end]])],
                    "medians": medians, "sample_support": [len(group) for group in groups],
                    "anchors": [{**anchors[i], "role": role} for i, role in zip(chosen, ROLES, strict=True)],
                }
    return results


def development(folder, parent, detector):
    pins = read(parent / "freeze.json")
    ordinary = []
    for index, component, rows in panel.grid_cases(pins, "development"):
        pictures, edges = {}, {}
        for path, row, _ in rows:
            if row["condition"] in panel.prior.NEURAL:
                continue
            chosen = select(*edge_observations(path, row, detector, pictures, edges))
            ordinary.append({"component": component, "index": index, "condition": row["condition"],
                             "sha256": row["file_sha256"],
                             "scores": {arm: None if rec is None else rec["score"] for arm, rec in chosen.items()}})
        print("development", index, flush=True)
    require(len(ordinary) == 56, "Complete ordinary development roster required")
    points = {}
    for arm in ARMS:
        measured = sorted(row["scores"][arm] for row in ordinary if row["scores"][arm] is not None)
        require(len(measured) >= 42, "Insufficient ordinary development coverage")
        points[arm] = max(0., measured[-3])
    write(folder / "calibration.json", {"freeze_sha256": sha(folder / "freeze.json"), "ordinary": ordinary, "points": points})


def held(folder, parent, detector):
    point = read(folder / "calibration.json")
    require(point["freeze_sha256"] == sha(folder / "freeze.json"), "Seal ordinary development calibration before held scoring")
    baseline = prior.baselines(parent)
    roster = prior.roster(parent, read(parent / "freeze.json"))
    started, visited = time.monotonic(), set()
    with (folder / "held.jsonl").open("x") as journal:
        for (bank, index), (component, rows) in roster.items():
            pictures, edges = {}, {}
            for condition, (path, source) in rows.items():
                key = (bank, index, condition)
                old = baseline[key]
                require(source["file_sha256"] == old["sha256"], "Matched direct comparator source differs")
                table, frames, shots, series, anchors = edge_observations(path, source, detector, pictures, edges)
                journal.write(json.dumps({"bank": bank, "index": index, "component": component, "condition": condition,
                    "source_sha256": source["file_sha256"], "truth": old["truth"], "compatible": old["source_time_compatible"],
                    "arms": {"direct": old["arms"]["direct"], **select(table, frames, shots, series, anchors)},
                    "frames": frames, "shots": shots, "edges": series, "anchors": anchors}, allow_nan=False) + "\n")
                visited.add(key)
            journal.flush()
            print("held", bank, index, len(rows), flush=True)
    require(visited == set(baseline) and len(visited) == 160, "Exact 160 held inputs required")
    write(folder / "runtime.json", {"elapsed_s": time.monotonic() - started,
          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "held_sha256": sha(folder / "held.jsonl"), "calibration_sha256": sha(folder / "calibration.json")})


def decision(record, point, truth):
    if record is None:
        return "unassessed"
    if record["score"] <= point:
        return "measured_no_range"
    if truth is None:
        return "false_range"
    return "correct_range" if review.overlap(record["source_clock_s"], truth) >= Fraction(1, 2) else "misplaced_range"


def tally(rows, points):
    counts = defaultdict(lambda: defaultdict(Counter))
    compatible, comparison, coverage, groups = defaultdict(Counter), Counter(), defaultdict(Counter), defaultdict(set)
    for row in rows:
        bank = row["condition"] if row["bank"] == "delivery" else row["bank"]
        found = {arm: decision(row["arms"][arm], point, row["truth"]) for arm, point in points.items()}
        for arm, state in found.items():
            counts[bank][arm][state] += 1
            if row["compatible"]:
                compatible[arm][state] += 1
        label = "positive" if row["truth"] else "control"
        coverage[bank][label] += 1
        candidate, direct = found[PRIMARY], found["direct"]
        coverage[bank][label + "_assessed"] += candidate != "unassessed"
        if candidate == "correct_range" and direct != "correct_range":
            comparison[bank+"_added"] += 1
            comparison["compatible_added"] += row["compatible"]
            groups[bank].add(row["component"])
        if candidate == "false_range" and (direct != "false_range" or review.overlap(
                row["arms"]["direct"]["source_clock_s"], row["arms"][PRIMARY]["source_clock_s"]) < Fraction(1, 2)):
            comparison[bank+"_extra"] += 1
    comparison.update({bank+"_groups": len(groups[bank]) for bank in ("grid", "native")})
    criteria = {key: comparison[key] <= bound if key.endswith("_extra") else comparison[key] >= bound
                for key, bound in SPEC["gate"].items()}
    criteria["coverage"] = all(coverage[bank][label+"_assessed"] >= SPEC["coverage_floor"]*coverage[bank][label]
                               for bank in ("grid", "native") for label in ("positive", "control"))
    return {"outcomes": counts, "compatible": compatible, "comparison": comparison, "coverage": coverage,
            "criteria": criteria, "advances": all(criteria.values())}


def report(folder, parent):
    calibration, runtime = (read(folder / name) for name in ("calibration.json", "runtime.json"))
    require(runtime["calibration_sha256"] == sha(folder / "calibration.json")
            and runtime["held_sha256"] == sha(folder / "held.jsonl"), "Held journal changed")
    points = {"direct": read(parent / "calibration.json")["points"]["direct"], **calibration["points"]}
    rows = panel.prior.jsonl(folder / "held.jsonl")
    require(len(rows) == 160, "Need the exact full held roster")
    write(folder / "aggregate.json", {**tally(rows, points), "runtime": runtime, "points": points, "protocol": SPEC,
                                     "freeze_sha256": sha(folder / "freeze.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held", "report"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path, default=prior.PRIOR)
    args = parser.parse_args()
    locked = lock(args.prior)
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700, parents=True, exist_ok=False)
        write(args.folder / "freeze.json", locked)
        return
    require(read(args.folder / "freeze.json") == locked, "Frozen inputs or method changed")
    if args.phase == "report":
        report(args.folder, args.prior)
        return
    cv2.setNumThreads(1)
    action = development if args.phase == "development" else held
    action(args.folder, args.prior, load_yunet(panel.prior.WEIGHT))


if __name__ == "__main__":
    main()
