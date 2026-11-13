# SPDX-License-Identifier: Apache-2.0
"""Test short-range chroma variation around redrawn mouths on a single native picture clock."""

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.shots import ShotDetector
from scripts import video_mouth_blind_prediction as cohort
from scripts import video_mouth_reconstruction_transfer as geometry
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.video-mouth-temporal-chroma/1"
ARMS = ("relative_flicker", "bilateral_disagreement")
FLOOR = (.10, .10)
YY, XX = np.ogrid[:96, :96]
PATCHES = ((-18, 4), (18, 4), (-24, -12), (24, -12), (0, -23))
DESIGN = {
    "signal": "Compare short temporal chroma changes in bilateral lower-face skin with bilateral cheek skin. Subtract a common upper-face trace and use one- and two-frame second differences. The comparison makes no physiological inference.",
    "arms": "The first arm measures a rise in bilateral local chroma relative to luminance and cheeks. The second measures a rise in lower-face left-right chroma disagreement, relative to cheek disagreement. Both horizons and temporal flanks must support the increase.",
    "window": "8, 12, 16 or 20 pictures with six-picture flanks and the required one-shot support. Use the third-highest ordinary GRID development response per arm, with a preset floor of 0.10.",
    "held": "92 previously exposed H.264 deliveries, including 17 authored redraws from two workflows and 75 ordinary controls. Timing labels are independent of this check.",
    "gate": "Locate at least four timing-qualified first-family edits across three sources and the timing-qualified second-family edit, with time IoU at least 0.5. Measure at least 70% of each ordinary and neural set. Allow at most two ordinary alerts and none on unaltered or hard-cut controls.",
}


def freeze(parent):
    return {"schema": SCHEMA, "design": DESIGN, "prior_sha256": cohort.freeze(parent)["prior_sha256"],
            "geometry_sha256": sha(geometry.__file__), "weight_sha256": sha(geometry.WEIGHT),
            "worker_sha256": sha(__file__), "opencv": cv2.__version__, "numpy": np.__version__}


def observe(rgb, face):
    _, transform, mouth, shape = face
    aligned = cv2.warpAffine(rgb, transform, (96, 96), borderMode=cv2.BORDER_REFLECT_101)
    values = cv2.cvtColor(aligned, cv2.COLOR_RGB2YCrCb)
    luma = values[..., 0].astype(np.float32)
    dx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, scale=.125)
    dy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, scale=.125)
    skin = (np.hypot(dx, dy) < 18) & (luma > 16) & (luma < 239)
    samples = []
    for x, y in PATCHES:
        pixels = skin & (((XX - mouth[0] - x) / 7)**2 + ((YY - mouth[1] - y) / 6)**2 <= 1)
        if pixels.sum() < 40:
            return None, None
        samples.append(np.median(values[pixels], axis=0))
    inverse = cv2.invertAffineTransform(transform)
    x, y = mouth
    corners = np.asarray(((x-31, y-29, 1), (x+31, y-29, 1), (x-31, y+10, 1), (x+31, y+10, 1))) @ inverse.T
    left, top = np.maximum(0, np.floor(corners.min(axis=0))).astype(int)
    right, bottom = np.minimum(shape[::-1], np.ceil(corners.max(axis=0))).astype(int)
    bounds = [int(v) for v in (left, top, right, bottom)]
    return np.asarray(samples), {"mouth_crop_xyxy": bounds,
                                "crop_sha256": hashlib.sha256(rgb[top:bottom, left:right].tobytes()).hexdigest()}


def responses(values, shots):
    result = np.full((len(values), 2, 4, 3), np.nan)
    local = values[:, :4] - values[:, 4:5]
    for slot, hop in enumerate((1, 2)):
        for center in range(hop, len(values) - hop):
            if shots[center - hop] == shots[center + hop]:
                result[center, slot] = local[center] - (local[center-hop] + local[center+hop]) * .5
    return result


def interval_features(steps, length):
    windows = np.lib.stride_tricks.sliding_window_view(steps, length, axis=0)
    valid = (np.isfinite(windows[:, :, 0, 0]).sum(axis=-1) * 5 >= length * 4).all(axis=1)
    output = np.full((len(windows), 2, 2), np.nan)
    if not valid.any():
        return output
    windows = windows[valid]
    rms = np.sqrt(np.nanmean(windows**2, axis=-1))
    chroma = np.linalg.norm(rms[..., 1:], axis=-1)
    normalized = np.log((chroma + .35) / (rms[..., 0] + .75))
    flicker = np.min(normalized[..., :2], axis=-1) - np.mean(normalized[..., 2:], axis=-1)
    pair_differences = windows[:, :, [0, 2], 1:] - windows[:, :, [1, 3], 1:]
    mismatch = np.sqrt(np.nansum(np.nanmean(pair_differences**2, axis=-1), axis=-1))
    relative = np.log((mismatch + .35) / (chroma[..., [0, 2]] + chroma[..., [1, 3]] + .70))
    output[valid] = np.stack((flicker, relative[..., 0] - relative[..., 1]), axis=-1)
    return output


def strongest(values, shots, anchors, table):
    steps = responses(values, shots)
    features = {length: interval_features(steps, length) for length in (6, 8, 12, 16, 20)}
    result = {arm: {"measured": False, "proposal": None} for arm in ARMS}
    for length in (8, 12, 16, 20):
        for start in range(6, len(values) - length - 5):
            end = start + length
            indices = (start - 1, (start + end) // 2, end)
            if shots[start - 6] != shots[end + 5] or not all(anchors[i] for i in indices):
                continue
            core = features[length][start]
            difference = core - np.maximum(features[6][start - 6], features[6][end])
            for arm, raw in zip(ARMS, difference.min(axis=0), strict=True):
                if not np.isfinite(raw):
                    continue
                entry = result[arm]
                entry["measured"] = True
                previous = entry["proposal"]
                if raw > (previous["score"] if previous else 0.):
                    entry["proposal"] = {"score": float(raw), "frames": [start, end],
                                         "seconds": [str(table.pts[i] * table.tb) for i in (start, end)],
                                         "anchors": [anchors[i] for i in indices]}
    return result


def inspect(path, detector, cache):
    table = probe_frames(path, cap=110, timeout_s=15)
    require(40 <= len(table) <= 100 and not (table.truncated or table.rotation or table.interpolated), "Incomplete native profile")
    tracker = ShotDetector(table)
    values = np.full((len(table), 5, 3), np.nan)
    shots, anchors = [], []
    for frame in decode_frames(path, table, None, timeout_s=30):
        index = frame.index
        require(index == len(shots) and frame.pts_verified and frame.pts == table.pts[index], "Native clock differs")
        tracker.update(index, frame.rgb)
        shots.append(len(tracker.boundaries))
        digest = hashlib.sha256(frame.rgb).hexdigest()
        if digest not in cache:
            face = geometry.aligned_face(frame.rgb, detector)
            cache[digest] = observe(frame.rgb, face) if face is not None else (None, None)
        value, proof = cache[digest]
        if value is not None:
            values[index] = value
        anchors.append({"frame": index, "pts": frame.pts, "rgb_sha256": digest, **proof} if proof is not None else None)
    require(len(shots) == len(table), "Incomplete native decode")
    return table, strongest(values, shots, anchors, table), int(np.isfinite(values[:, 0, 0]).sum())


def points(rows):
    return {arm: max(floor, sorted((row["arms"][arm]["proposal"] or {}).get("score", 0.) for row in rows)[-3])
            for arm, floor in zip(ARMS, FLOOR, strict=True)}


def run(parent, work, phase):
    rows = []
    detector = load_yunet(geometry.WEIGHT)
    cache = {}
    for row in read(parent / "carriers.json")["rows"]:
        if row["split"] != phase:
            continue
        path = Path(row["carrier"])
        require(sha(path) == row["carrier_sha256"], "Frozen carrier differs")
        if len(cache) > 1800:
            cache.clear()
        table, arms, support = inspect(path, detector, cache)
        truth = [str(table.pts[i] * table.tb) for i in row["edit"]] if row.get("edit") else row.get("native_truth")
        rows.append({"index": len(rows), "family": row["family"], "role": row["role"], "aligned": row.get("aligned", False),
                     "group": row["group"], "sha256": row["carrier_sha256"], "truth": truth, "frames": len(table),
                     "time_base": table.time_base, "arms": arms, "frame_support": support})
        if len(rows) % 8 == 0:
            print(phase, len(rows), flush=True)
    require(len(rows) == {"development": 56, "held": 92}[phase], "Incomplete frozen panel")
    write(work / f"{phase}.json", rows)
    if phase == "development":
        write(work / "points.json", {"development_sha256": sha(work / "development.json"), "thresholds": points(rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held"))
    parser.add_argument("parent", type=Path)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    lock = freeze(args.parent)
    frozen = args.work / "freeze.json"
    if args.phase == "freeze":
        args.work.mkdir(mode=0o700)
        write(frozen, lock)
        return
    require(read(frozen) == lock, "Method or input commitment differs")
    if args.phase == "held":
        fit = args.work / "development.json"
        require(read(args.work / "points.json") == {"development_sha256": sha(fit), "thresholds": points(read(fit))},
                "Unverified development points")
    run(args.parent, args.work, args.phase)


if __name__ == "__main__":
    main()
