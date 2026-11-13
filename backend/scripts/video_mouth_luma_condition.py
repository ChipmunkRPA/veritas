# SPDX-License-Identifier: Apache-2.0
"""Evaluate brightness-conditioned detail in one short submitted video."""

from __future__ import annotations

import argparse
import math
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.shots import ShotDetector
from scripts import video_mouth_track_cycle as geometry
from scripts.video_common_sources_v89 import read, require, sha, write
from scripts.review_video_mouth_luma_condition import ARMS, FLOOR, SCHEMA, SPLITS, points_for

PREVIOUS = {
    "freeze.json": "af7feee3a877e09414ff92892b8af923c2f2441a4007596eacf4910bd6e89ff3",
    "codec_stress_roster.json": "0ab4c837497e7d5f2c4ba890f7a4c1830f448d7b27f57e2e5a11aa28f4a35cc8",
}
SPANS = (8, 12, 16, 20)
FLANK = 5
BLOCK = 4
MIN_BLOCKS = 8
SPEC = {
    "input": "Use one submitted picture stream, native RGB and pinned single-face geometry. No second file, supplied range, renderer metadata, audio or reference camera enters scoring.",
    "block": "Use 4 by 4 native luminance blocks anchored at global grid multiples of four. Measure mean squared diagonal Haar coefficient on each of four disjoint 2 by 2 cells. Coarse gradient is the Euclidean norm of horizontal and vertical differences between block halves. Exclude mean luminance outside 32 to 224 and coarse gradient above 9.",
    "regions": "One mouth ellipse and three surrounding comparison ellipses on the eye-normalized plane. Only block-center coordinates are transformed. Pixel values always remain native.",
    "plain": "Compare the logarithm of the ratio of surrounding median diagonal detail to mouth median diagonal detail with additive 0.5 before dividing. Require eight blocks in each region pool.",
    "conditioned": "Match each mouth block to surrounding blocks within 16 luma units and 3 coarse-gradient units. Use the three nearest under normalized squared distance. Take the median of their median log detail minus mouth log detail with additive 0.5. Require eight matches and at least half the eligible mouth blocks matched.",
    "range": "Use one verified shot, 8, 12, 16 or 20 native pictures, and five flank pictures on each side. Require 80 percent observation coverage in core and flanks. Use the smaller positive median drop against both flanks and retain one strongest interval. Before, middle and first-after pictures must have native face anchors.",
    "points": "The third-largest ordinary GRID development maximum per arm with a fixed floor of 0.04. Require strict exceedance. Correct localization requires native half-open temporal IoU of at least 0.50.",
    "gate": "At least four strict LivePortrait locations across three groups and the strict MuseTalk location. Require 70 percent measured coverage for each strict neural family and ordinary controls, at most two ordinary flags and none on original, recompressed or hard-cut controls. Compare matched H.264 4:2:0 delivery separately.",
}


def freeze(prior):
    for name, digest in PREVIOUS.items():
        require(sha(prior / name) == digest, "Previously held video roster differs")
    archived = read(prior / "freeze.json")
    stress = read(prior / "codec_stress_roster.json")
    require(stress["freeze_sha256"] == PREVIOUS["freeze.json"], "Stress roster does not bind the original roster")
    inputs = geometry.prior.panel.prior
    pinned = archived["inputs_sha256"]
    require(sha(geometry.prior.WEIGHT) == pinned["detector_weights"]
            and sha(Path(inputs.__file__)) == pinned["geometry"], "Pinned face geometry differs")
    rows = [*archived["rows"], *stress["rows"]]
    require(Counter(row["split"] for row in rows) == SPLITS, "Held file scope differs")
    return {"schema": SCHEMA, "protocol": SPEC, "rows": rows, "prior_rosters": PREVIOUS,
            "code": {"worker": sha(Path(__file__)), "geometry": pinned["geometry"],
                     "reviewer": sha(Path(__file__).with_name("review_video_mouth_luma_condition.py"))},
            "face_weights": pinned["detector_weights"], "opencv": cv2.__version__, "numpy": np.__version__,
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]}


def block_detail(rgb, located):
    _, matrix, mouth, _ = located
    inverse = cv2.invertAffineTransform(matrix)
    limits = np.asarray(((-38, -34), (38, -34), (38, 15), (-38, 15))) + mouth
    bounds = limits @ inverse[:, :2].T + inverse[:, 2]
    lower = np.maximum(0, np.floor(bounds.min(axis=0) / BLOCK).astype(int) * BLOCK)
    upper = np.minimum(rgb.shape[1::-1], np.ceil(bounds.max(axis=0) / BLOCK).astype(int) * BLOCK)
    x0, y0 = lower
    nx, ny = (upper - lower) // BLOCK
    if min(nx, ny) < 4:
        return np.full(2, np.nan)
    crop = rgb[y0:y0 + BLOCK * ny, x0:x0 + BLOCK * nx].astype(np.float32)
    luma = crop @ np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
    parts = luma.reshape(ny, BLOCK, nx, BLOCK).transpose(0, 2, 1, 3)
    mean = parts.mean(axis=(2, 3))
    dx = parts[..., :, 2:].mean(axis=(2, 3)) - parts[..., :, :2].mean(axis=(2, 3))
    dy = parts[..., 2:, :].mean(axis=(2, 3)) - parts[..., :2, :].mean(axis=(2, 3))
    gradient = np.hypot(dx, dy)
    diagonal = (parts[..., ::2, ::2] - parts[..., 1::2, ::2]
                - parts[..., ::2, 1::2] + parts[..., 1::2, 1::2]) / 2
    detail = np.mean(diagonal ** 2, axis=(2, 3))
    y, x = np.mgrid[:ny, :nx] * BLOCK + (BLOCK - 1) / 2
    relative = np.stack((x + x0, y + y0), axis=-1) @ matrix[:, :2].T + matrix[:, 2] - mouth
    xx, yy = relative.transpose(2, 0, 1)
    valid = (mean >= 32) & (mean <= 224) & (gradient <= 9)
    target = valid & ((xx / 19) ** 2 + (yy / 12) ** 2 <= 1)
    surround = np.zeros_like(valid)
    for cx, cy, rx, ry in ((-26, -12, 11, 11), (26, -12, 11, 11), (0, -24, 15, 8)):
        surround |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    surround &= valid & ~target
    if min(np.count_nonzero(target), np.count_nonzero(surround)) < MIN_BLOCKS:
        return np.full(2, np.nan)
    raw_target, raw_surround = detail[target], detail[surround]
    plain = math.log((float(np.median(raw_surround)) + 0.5) / (float(np.median(raw_target)) + 0.5))
    intensity_delta = (mean[target, None] - mean[surround][None, :]) / 16
    gradient_delta = (gradient[target, None] - gradient[surround][None, :]) / 3
    distances = np.where((np.abs(intensity_delta) <= 1) & (np.abs(gradient_delta) <= 1),
                         intensity_delta ** 2 + gradient_delta ** 2, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :3]
    matched = np.isfinite(np.take_along_axis(distances, neighbors, axis=1)).all(axis=1)
    if np.count_nonzero(matched) < max(MIN_BLOCKS, (len(raw_target) + 1) // 2):
        return np.asarray((plain, np.nan))
    reference = np.median(raw_surround[neighbors[matched]], axis=1)
    score = np.median(np.log(reference + 0.5) - np.log(raw_target[matched] + 0.5))
    return np.asarray((plain, score))


def inspect(path, detector, cache):
    table = probe_frames(path, timeout_s=15)
    require(not (table.truncated or table.interpolated or table.rotation) and 40 <= len(table) <= 100,
            "Expected short native picture table")
    observations = np.full((len(table), len(ARMS)), np.nan)
    shots, anchors = [], []
    track = ShotDetector(table)
    panel = geometry.prior.panel
    for frame in decode_frames(path, table, None, timeout_s=28):
        require(frame.index == len(shots) and frame.pts_verified and frame.pts == table.pts[frame.index],
                "Picture clock differs")
        track.update(frame.index, frame.rgb)
        shots.append(len(track.boundaries))
        digest = panel.raw_digest(frame.rgb)
        if digest not in cache:
            located = panel.prior.aligned_face(frame.rgb, detector)
            if located is None:
                cache[digest] = None, None
            else:
                x0, y0, x1, y1 = panel.native_bounds(located)
                cache[digest] = block_detail(frame.rgb, located), {
                    "rgb_sha256": digest, "crop_xyxy": [x0, y0, x1, y1],
                    "crop_sha256": panel.raw_digest(frame.rgb[y0:y1, x0:x1]),
                }
        value, proof = cache[digest]
        if value is not None:
            observations[frame.index] = value
        anchors.append({**proof, "frame": frame.index, "pts": frame.pts} if proof else None)
    require(len(shots) == len(table), "Incomplete native source pictures")
    return table, np.asarray(shots), observations, anchors


def ranges(table, shots, data, anchors):
    proposals = dict.fromkeys(ARMS)
    assessed = dict.fromkeys(ARMS, False)
    windows = {}
    for width in (FLANK, *SPANS):
        parts = np.lib.stride_tricks.sliding_window_view(data, width, axis=0)
        good = np.isfinite(parts).sum(axis=-1) * 5 >= width * 4
        median = np.ma.median(np.ma.masked_invalid(parts), axis=-1).filled(np.nan)
        windows[width] = np.where(good, median, np.nan)
    for width in SPANS:
        for start in range(FLANK, len(table) - width - FLANK + 1):
            end = start + width
            picked = [anchors[i] for i in (start - 1, (start + end) // 2, end)]
            if shots[start - FLANK] != shots[end + FLANK - 1] or not all(picked):
                continue
            scores = windows[width][start] - np.maximum(windows[FLANK][start - FLANK], windows[FLANK][end])
            for index, arm in enumerate(ARMS):
                score = float(scores[index])
                if not math.isfinite(score):
                    continue
                assessed[arm] = True
                if score <= 0 or proposals[arm] and score <= proposals[arm]["score"]:
                    continue
                proposals[arm] = {"score": score, "frames": [start, end],
                                  "seconds": [str(table.pts[i] * table.tb) for i in (start, end)],
                                  "time_base": table.time_base, "anchors": picked}
    return {arm: {"measured": assessed[arm], "proposal": proposals[arm],
                  "score": proposals[arm]["score"] if proposals[arm] else None} for arm in ARMS}


def score(lock, work, phase):
    detector = load_yunet(geometry.prior.WEIGHT)
    cases = {}
    for row in lock["rows"]:
        if row["split"] == phase:
            cases.setdefault((row["family"], row["id"]), []).append(row)
    output = []
    for entries in cases.values():
        cache = {}
        for row in entries:
            path = Path(row["file"])
            require(sha(path) == row["sha"], "Frozen video bytes differ")
            table, shots, values, anchors = inspect(path, detector, cache)
            result = ranges(table, shots, values, anchors)
            truth = [str(table.pts[i] * table.tb) for i in row["edit"]] if row.get("edit") else row.get("native_truth")
            if truth:
                require(all(Fraction(value) in {p * table.tb for p in table.pts} for value in truth), "Edit clock differs")
            output.append({**row, "arms": result, "native_frames": len(table), "truth": truth,
                           "frame_support": dict(zip(ARMS, map(int, np.isfinite(values).sum(axis=0))))})
        if len(output) % 10 < len(entries):
            print(phase, len(output), flush=True)
    require(len(output) == SPLITS[phase], "Incomplete video comparison")
    path = work / f"{phase}.json"
    write(path, {"freeze_sha256": sha(work / "freeze.json"), "rows": output})
    if phase == "development":
        write(work / "points.json", {"development_sha256": sha(path), "points": points_for(output)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", *SPLITS))
    parser.add_argument("work", type=Path)
    parser.add_argument("prior", type=Path)
    args = parser.parse_args()
    lock = freeze(args.prior)
    path = args.work / "freeze.json"
    if args.phase == "freeze":
        args.work.mkdir(mode=0o700)
        write(path, lock)
        return
    require(read(path) == lock, "Frozen sources, runtime or method differ")
    if args.phase != "development":
        dev = args.work / "development.json"
        require(read(args.work / "points.json") == {"development_sha256": sha(dev),
                                                      "points": points_for(read(dev)["rows"])}, "Frozen points differ")
    cv2.setNumThreads(1)
    score(lock, args.work, args.phase)


if __name__ == "__main__":
    main()
