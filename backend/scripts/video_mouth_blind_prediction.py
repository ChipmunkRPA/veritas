# SPDX-License-Identifier: Apache-2.0
"""Test a mouth-frame prediction that uses neighboring pictures only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.shots import ShotDetector
from scripts import video_mouth_luma_condition as windows
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.video-mouth-blind-prediction/1"
ARMS = ("unwarped_neighbors", "motion_predicted_neighbors")
HOPS = (1, 2)
SIZE = 96
YY, XX = np.mgrid[:SIZE, :SIZE].astype(np.float32)
XY = np.stack((XX, YY), axis=2)
SCOPE = {
    "fit": "Use the 56 existing GRID development control files. Select the third-highest ordinary maximum with a floor of 0.04 per arm.",
    "held": "Score all 92 previously held H.264 files, including 17 authored synchronized redrawing cases and 75 ordinary controls.",
    "prediction": "Compute symmetric interpolation between the outer eye-aligned frames at one and two native steps from center. The candidate uses forward and reverse dense flow from only those neighbors and compares their warped midpoint with the observed middle picture. Both hops must support a positive mouth-minus-surround change.",
    "qualification": "Use the exact local picture clock, same observed shot, complete neighboring face geometry, and the existing 8/12/16/20-frame interval selector with five-picture flanks. Require genuine point separation before adopting.",
    "release": "Require at least four strictly aligned LivePortrait locations across three groups and the strictly aligned MuseTalk location, at most two ordinary flags and none on untouched or hard-cut controls. Report support and misses.",
}


def midpoint(a, b, flow_a=None, flow_b=None):
    if flow_a is None:
        return (a.astype(np.float32) + b) * .5
    before = cv2.remap(a, XY - .5 * flow_a, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    after = cv2.remap(b, XY - .5 * flow_b, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return (before.astype(np.float32) + after) * .5


def residual(center, predicted, target, surround):
    # Take upper-middle residual, excluding single-pixel outliers.
    error = np.abs(center - predicted)
    top = float(np.percentile(error[target], 75))
    control = float(np.percentile(error[surround], 75))
    return np.log((top + 2.0) / (control + 2.0))


def measure(a, center, b, mouth):
    dx, dy = XX - mouth[0], YY - mouth[1]
    target = ((dx / 17) ** 2 + (dy / 9) ** 2) <= 1
    surround = ((dx / 31) ** 2 + (dy / 18) ** 2 <= 1) & ((dx / 22) ** 2 + (dy / 12) ** 2 >= 1)
    flow_a = cv2.calcOpticalFlowFarneback(a, b, None, .5, 2, 13, 3, 5, 1.2, 0)
    flow_b = cv2.calcOpticalFlowFarneback(b, a, None, .5, 2, 13, 3, 5, 1.2, 0)
    return [residual(center, prediction, target, surround)
            for prediction in (midpoint(a, b), midpoint(a, b, flow_a, flow_b))]


def inspect(path, detector, geometry):
    table = probe_frames(path, timeout_s=15)
    require(not (table.truncated or table.interpolated or table.rotation) and 40 <= len(table) <= 100, "Native scope differs")
    track = ShotDetector(table)
    shots, inputs, anchors = [], [], []
    for frame in decode_frames(path, table, None, timeout_s=30):
        i = frame.index
        require(i == len(shots) and frame.pts_verified and frame.pts == table.pts[i], "Picture clock differs")
        track.update(i, frame.rgb)
        shots.append(len(track.boundaries))
        located = geometry.prior.aligned_face(frame.rgb, detector)
        if located is None:
            inputs.append(None)
            anchors.append(None)
            continue
        gray, _, mouth, _ = located
        x0, y0, x1, y1 = geometry.native_bounds(located)
        inputs.append((gray, mouth))
        anchors.append({"frame": i, "pts": frame.pts, "crop_xyxy": [x0, y0, x1, y1],
                        "rgb_sha256": geometry.raw_digest(frame.rgb),
                        "crop_sha256": geometry.raw_digest(frame.rgb[y0:y1, x0:x1])})
    require(len(shots) == len(table), "Incomplete native input")
    samples = np.full((len(table), len(ARMS)), np.nan)
    for i in range(max(HOPS), len(table) - max(HOPS)):
        if shots[i - 2] != shots[i + 2] or any(row is None for row in inputs[i - 2:i + 3]):
            continue
        if (table.pts[i] - table.pts[i - 2]) != (table.pts[i + 2] - table.pts[i]):
            continue
        center, mouth = inputs[i]
        samples[i] = np.min([measure(inputs[i - hop][0], center, inputs[i + hop][0], mouth) for hop in HOPS], axis=0)
    result = windows.ranges(table, np.asarray(shots), samples, anchors)
    return table, dict(zip(ARMS, result.values(), strict=True)), dict(zip(ARMS, map(int, np.isfinite(samples).sum(axis=0)), strict=True))


def threshold(rows):
    return {arm: max(.04, sorted((row["arms"][arm]["score"] or 0 for row in rows), reverse=True)[2]) for arm in ARMS}


def freeze(prior):
    roster = prior / "carriers.json"
    context = read(roster)
    require(context["freeze_sha256"] == sha(prior / "freeze.json"), "Carrier is not the reviewed H.264 panel")
    require(Counter(row["split"] for row in context["rows"]) == {"development": 56, "held": 92}, "Unexpected panel roles")
    require(sha(roster) == "f0cd3e5e07834e74bbdf2d70e9ccf273cc77ed2814c1ac3b8ba2129285937bf7", "Unexpected input roster")
    return {"schema": SCHEMA, "worker_sha256": sha(Path(__file__)), "prior_sha256": sha(roster),
            "weight_sha256": sha(windows.geometry.prior.WEIGHT), "opencv": cv2.__version__, "protocol": SCOPE}


def score(prior, work, phase):
    lock = read(work / "freeze.json")
    require(lock == freeze(prior), "Frozen method or input roster differs")
    rows = []
    detector = load_yunet(windows.geometry.prior.WEIGHT)
    geometry = windows.geometry.prior.panel
    if phase == "held":
        points = read(work / "points.json")
        require(points["development_sha256"] == sha(work / "development.json"), "Operating point lacks a committed development run")
    for row in read(prior / "carriers.json")["rows"]:
        if row["split"] != phase:
            continue
        path = Path(row["carrier"])
        require(sha(path) == row["carrier_sha256"], "Frozen carrier differs")
        table, arms, support = inspect(path, detector, geometry)
        truth = [str(table.pts[i] * table.tb) for i in row["edit"]] if row.get("edit") else row.get("native_truth")
        rows.append({"index": len(rows), "family": row["family"], "role": row["role"], "aligned": row.get("aligned", False),
                     "group": row["group"], "source_sha256": row["carrier_sha256"], "truth": truth,
                     "time_base": table.time_base, "frames": len(table), "arms": arms, "frame_support": support,
                     "pts_sha256": hashlib.sha256(json.dumps(table.pts).encode()).hexdigest()})
        if len(rows) % 8 == 0:
            print(phase, len(rows), flush=True)
    require(len(rows) == (56 if phase == "development" else 92), "Incomplete role")
    output = work / f"{phase}.json"
    write(output, {"freeze_sha256": sha(work / "freeze.json"), "rows": rows})
    if phase == "development":
        write(work / "points.json", {"development_sha256": sha(output), "points": threshold(rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held"))
    parser.add_argument("prior", type=Path)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase == "freeze":
        args.work.mkdir(mode=0o700)
        write(args.work / "freeze.json", freeze(args.prior))
    else:
        score(args.prior, args.work, args.phase)


if __name__ == "__main__":
    main()
