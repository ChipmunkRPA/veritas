# SPDX-License-Identifier: Apache-2.0
"""Test mouth-edge recurrence against similarly posed frames in one video."""

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
from scripts import video_mouth_native_texture_transfer as previous
from scripts.video_common_sources_v89 import read, require, sha, write

ARMS = ("pose_matched", "nearest_time")
SPEC = {
    "schema": "chipmunk.video-mouth-pose-recurrence/1",
    "aligned_mouth_crop_wh": [33, 19], "aligned_context_crop_wh": [33, 13], "context_above_mouth_px": 16,
    "pose_opening_cells_wh": [3, 2], "gradient_cells_wh": [3, 2], "orientation_bins": 6,
    "pose_distance_max": .43, "flank_samples": 12, "minimum_separation_s": "4/25", "coverage": .75,
    "points": "Per arm, strict exceedance of the third-largest ordinary development maximum or zero, whichever is higher.",
    "gate": {"grid_added": 4, "grid_components": 2, "compatible_added": 2, "grid_extra": 4,
             "native_added": 2, "native_components": 2, "native_extra": 3, "cut_extra": 0, "reencode_extra": 1},
    "method": "Select mouth-opening and face-pose matches on both sides of a short interval in the same shot. "
              "Compare spatial edge orientations inside the mouth relative to the surrounding face. "
              "Both within-clip comparisons must exceed the disagreement between the two reference pictures.",
}


def freeze(prior):
    locked = previous.seal(prior)
    return {"protocol": SPEC, "matched_bank": locked, "implementation_sha256": sha(Path(__file__))}


def gradient_layout(gray):
    bins = SPEC["orientation_bins"]
    cx, cy = SPEC["gradient_cells_wh"]
    x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(x, y)
    angle = (np.arctan2(y, x) % np.pi) * bins / np.pi
    lo = np.floor(angle).astype(int)
    fraction = angle - lo
    rows, cols = gray.shape
    cell = (np.arange(rows) * cy // rows)[:, None] * cx + (np.arange(cols) * cx // cols)
    lower = cell * bins + lo % bins
    upper = cell * bins + (lo + 1) % bins
    h = np.bincount(lower.ravel(), (magnitude * (1-fraction)).ravel(), minlength=cx*cy*bins)
    h += np.bincount(upper.ravel(), (magnitude * fraction).ravel(), minlength=cx*cy*bins)
    h = h.reshape(cx*cy, bins)
    return np.sqrt(h / np.maximum(h.sum(axis=1, keepdims=True), 120.)).ravel()


def descriptor(rgb, geometry):
    aligned, _, mouth, _ = geometry
    width, height = SPEC["aligned_mouth_crop_wh"]
    x, y = map(float, mouth)
    face = cv2.GaussianBlur(aligned.astype(np.float32), (3, 3), .65)
    mouth_patch = cv2.getRectSubPix(face, (width, height), (x, y))
    context = cv2.getRectSubPix(face, tuple(SPEC["aligned_context_crop_wh"]), (x, y-SPEC["context_above_mouth_px"]))
    flank = np.r_[context[:, :7].ravel(), context[:, -7:].ravel()]
    reference = np.median(flank)
    contrast = max(25., np.percentile(face, 85) - np.percentile(face, 15))
    opening = np.clip((reference - mouth_patch) / contrast, 0, 1)
    pooled = cv2.resize(opening, tuple(SPEC["pose_opening_cells_wh"]), interpolation=cv2.INTER_AREA).ravel()
    shape = np.r_[pooled, (x-48.)/15, (y-57.)/20]
    x0, y0, x1, y1 = panel.native_bounds(geometry)
    return shape, gradient_layout(mouth_patch), gradient_layout(context), [x0, y0, x1, y1], panel.raw_digest(rgb[y0:y1, x0:x1])


def measure(path, source, detector, cache):
    require(sha(path) == source["file_sha256"], "Verified video file differs")
    table, indices = panel.inspect_frames(path, source["expected"])
    scene = ShotDetector(table)
    shots, observations, anchors = [], [], []
    for frame, digest in panel.exact_rgb(path, table, indices, source["expected"]):
        if digest not in cache:
            geometry = panel.prior.aligned_face(frame.rgb, detector)
            cache[digest] = None if geometry is None else descriptor(frame.rgb, geometry)
        scene.update(frame.index, frame.rgb)
        shots.append(len(scene.boundaries))
        found = cache[digest]
        observations.append(None if found is None else found[:3])
        anchors.append(None if found is None else {"frame": frame.index, "pts": frame.pts, "rgb_sha256": digest,
                       "native_crop_xyxy": found[3], "crop_rgb_sha256": found[4]})
    return table, indices, shots, observations, anchors


def distances(observed):
    valid = np.asarray([entry is not None for entry in observed])
    shape, mouth, outer = (np.asarray([entry[i] if entry is not None else np.zeros(width) for entry in observed])
                           for i, width in enumerate((8, 36, 36)))
    pose = np.sqrt(np.mean((shape[:, None] - shape[None, :])**2, axis=2))
    edge = np.sqrt(np.mean((mouth[:, None] - mouth[None, :])**2, axis=2))
    context = np.sqrt(np.mean((outer[:, None] - outer[None, :])**2, axis=2))
    distance = np.maximum(0., edge - .5*context)
    return valid, pose, distance


def select(observed):
    table, frames, shots, features, anchors = observed
    times = [table.pts[i]*table.tb for i in frames]
    timeline = np.asarray(times, dtype=float)
    separated = np.abs(timeline[:, None] - timeline[None, :]) >= float(Fraction(SPEC["minimum_separation_s"])) - 1e-9
    valid, pose, change = distances(features)
    flank = SPEC["flank_samples"]
    shot = np.asarray(shots)
    results = dict.fromkeys(ARMS)
    for width in panel.SPEC["core_samples"]:
        for start in range(4, len(frames) - width - 3):
            end = start + width
            if shot[start-3] != shot[end+2]:
                continue
            targets = np.flatnonzero(valid[start:end]) + start
            left = np.flatnonzero(valid[max(0, start-flank):start] & (shot[max(0, start-flank):start] == shot[start])) + max(0, start-flank)
            right = np.flatnonzero(valid[end:end+flank] & (shot[end:end+flank] == shot[start])) + end
            if len(left) < 3 or len(right) < 3:
                continue
            for arm in ARMS:
                responses = []
                for k in targets:
                    before, after = left[separated[k, left]], right[separated[k, right]]
                    if not len(before) or not len(after):
                        continue
                    if arm == "pose_matched":
                        first, last = before[pose[k, before].argmin()], after[pose[k, after].argmin()]
                        if max(pose[k, first], pose[k, last], pose[first, last]) > SPEC["pose_distance_max"]:
                            continue
                    else:
                        first, last = before[-1], after[0]
                    pair = [float(change[k, first]), float(change[k, last]), float(change[first, last])]
                    response = min(pair[:2]) - pair[2]
                    responses.append((response, first, k, last, pair))
                if len(responses) < max(4, (3*width + 4)//5):
                    continue
                score = float(np.median([row[0] for row in responses]))
                if results[arm] is not None and results[arm]["score"] >= score:
                    continue
                _, a, b, c, pair = min(responses, key=lambda row: (abs(row[0]-score), row[2]))
                results[arm] = {"score": score, "frames": [frames[start], frames[end]], "time_base": table.time_base,
                    "source_clock_s": [str(times[start]), str(times[end])], "pose_distances": [float(pose[i, j]) for i, j in ((a,b), (b,c), (a,c))],
                    "edge_distances": pair, "matched_triplets": len(responses), "responses": [item[0] for item in responses],
                    "anchors": [{**anchors[i], "role": role} for i, role in zip((a,b,c), ("before", "during", "after"), strict=True)]}
    return results


def development(folder, doc, detector):
    ordinary = []
    for index, _, rows in panel.grid_cases(doc, "development"):
        cache = {}
        for path, entry, _ in rows:
            if entry["condition"] in panel.prior.NEURAL:
                continue
            scores = select(measure(path, entry, detector, cache))
            ordinary.append({"index": index, "condition": entry["condition"],
                             "arms": {arm: None if result is None else result["score"] for arm, result in scores.items()}})
        print("development", index, flush=True)
    require(len(ordinary) == 56, "All ordinary development views required")
    points = {}
    for arm in ARMS:
        measured = sorted(row["arms"][arm] for row in ordinary if row["arms"][arm] is not None)
        require(len(measured) >= 42, "Insufficient development coverage for a point")
        points[arm] = max(0., measured[-3])
    write(folder / "calibration.json", {"freeze_sha256": sha(folder / "freeze.json"), "points": points, "ordinary": ordinary})


def held(folder, prior, doc, detector):
    calibration = read(folder / "calibration.json")
    require(calibration["freeze_sha256"] == sha(folder / "freeze.json"), "Seal the point before held scoring")
    baseline = previous.baselines(prior)
    start = time.monotonic()
    visited = set()
    with (folder / "held.jsonl").open("x") as stream:
        for (bank, index), (component, cases) in previous.roster(prior, doc).items():
            cache = {}
            for condition, (path, source) in cases.items():
                key = bank, index, condition
                old = baseline[key]
                require(source["file_sha256"] == old["sha256"], "Direct comparator media differs")
                observed = measure(path, source, detector, cache)
                arms = {"direct": old["arms"]["direct"], **select(observed)}
                stream.write(json.dumps({"bank": bank, "index": index, "component": component, "condition": condition,
                    "source_sha256": source["file_sha256"], "truth": old["truth"], "compatible": old["source_time_compatible"],
                    "arms": arms, "sampled_frames": len(observed[1])}, allow_nan=False) + "\n")
                visited.add(key)
            stream.flush()
            print("held", bank, index, len(cases), flush=True)
    require(visited == set(baseline) and len(visited) == 160, "Held roster is incomplete")
    write(folder / "runtime.json", {"elapsed_s": time.monotonic() - start,
          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "held_sha256": sha(folder / "held.jsonl"), "calibration_sha256": sha(folder / "calibration.json")})


def classify(record, point, truth):
    if record is None:
        return "unassessed"
    if record["score"] <= point:
        return "measured_no_range"
    if truth is None:
        return "false_range"
    return "correct_range" if review.overlap(record["source_clock_s"], truth) >= Fraction(1, 2) else "misplaced_range"


def tally(rows, points):
    counts = defaultdict(lambda: defaultdict(Counter))
    compatible = defaultdict(Counter)
    scope = defaultdict(Counter)
    comparison = Counter()
    groups = defaultdict(set)
    for row in rows:
        bank = row["condition"] if row["bank"] == "delivery" else row["bank"]
        outcomes = {arm: classify(row["arms"][arm], point, row["truth"]) for arm, point in points.items()}
        for arm, state in outcomes.items():
            counts[bank][arm][state] += 1
            if row["compatible"]:
                compatible[arm][state] += 1
        direct, candidate = outcomes["direct"], outcomes["pose_matched"]
        label = "positive" if row["truth"] else "control"
        scope[bank][label] += 1
        scope[bank][label + "_assessed"] += candidate != "unassessed"
        if candidate == "correct_range" and direct != "correct_range":
            comparison[f"{bank}_added"] += 1
            comparison["compatible_added"] += row["compatible"]
            groups[bank].add(row["component"])
        if candidate == "false_range" and (direct != "false_range" or review.overlap(
                row["arms"]["direct"]["source_clock_s"], row["arms"]["pose_matched"]["source_clock_s"]) < Fraction(1, 2)):
            comparison[f"{bank}_extra"] += 1
    comparison["cut_extra"] = comparison.pop("hard_cut_extra", 0)
    comparison.update({f"{bank}_components": len(groups[bank]) for bank in ("grid", "native")})
    gate = {key: comparison[key] <= target if key.endswith("extra") else comparison[key] >= target
            for key, target in SPEC["gate"].items()}
    gate["coverage"] = all(scope[bank][label + "_assessed"] >= SPEC["coverage"] * scope[bank][label]
                           for bank in ("grid", "native") for label in ("positive", "control"))
    return {"outcomes": counts, "compatible": compatible, "coverage": scope, "comparison": comparison,
            "criteria": gate, "advance": all(gate.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "development", "held", "report"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path, default=previous.PRIOR)
    args = parser.parse_args()
    folder, prior = args.folder, args.prior
    locked = freeze(prior)
    if args.stage == "freeze":
        folder.mkdir(mode=0o700, parents=True)
        write(folder / "freeze.json", locked)
        return
    require(read(folder / "freeze.json") == locked, "Frozen method, source or runtime binding differs")
    if args.stage == "report":
        runtime, calibration = (read(folder / name) for name in ("runtime.json", "calibration.json"))
        require(runtime["held_sha256"] == sha(folder / "held.jsonl")
                and runtime["calibration_sha256"] == sha(folder / "calibration.json"), "Frozen run bytes differ")
        points = {"direct": read(prior / "calibration.json")["points"]["direct"], **calibration["points"]}
        rows = panel.prior.jsonl(folder / "held.jsonl")
        require(len(rows) == 160, "Complete held journal required")
        write(folder / "aggregate.json", {**tally(rows, points), "runtime": runtime, "points": points, "spec": SPEC})
        return
    cv2.setNumThreads(1)
    detector = load_yunet(panel.prior.WEIGHT)
    doc = read(prior / "freeze.json")
    if args.stage == "development":
        development(folder, doc, detector)
    else:
        held(folder, prior, doc, detector)


if __name__ == "__main__":
    main()
