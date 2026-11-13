# SPDX-License-Identifier: Apache-2.0
"""Test single-upload evidence from local facial illumination and texture changes."""

from __future__ import annotations

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

ARMS = ("skin_balance", "balance_with_texture")
SCHEMA = "chipmunk.video-mouth-photometric/1"
FLOOR = (.035, .035)
YY, XX = np.ogrid[:96, :96]
DESIGN = {
    "signal": "Measure changes in lower-face skin that disagree with nearby cheeks and upper face. Use exact native picture clocks and a 96 px eye registration. Exclude the mobile central lips and high-gradient or clipped skin pixels.",
    "comparison": "Measure luma/chroma patch contrasts on both sides of the lower face. Compare its temporal change against both neighboring windows and change among the corresponding three nearby control patches. The second arm additionally requires a similarly localized loss of relative fine-detail contrast. Neither detector sees another copy or audio.",
    "windows": "8, 12, 16 or 20 native pictures bracketed by five measured pictures on each side inside a detected shot. Keep the strongest single interval. Fit a separate point on the third largest ordinary GRID response with minimum .035.",
    "assessment": "All 92 retained H.264 held deliveries from the frozen earlier roster, including both neural synthesis families. These clips have prior project exposure. No thresholds fit using held responses.",
    "gate": "At least four time-qualified LivePortrait edits across three groups and the timing-qualified MuseTalk edit, with time IoU at least 0.5. At least 70% neural and ordinary coverage. At most two ordinary alerts and none on hard cuts or unedited originals. No confidence change.",
}


def freeze(parent):
    old = cohort.freeze(parent)
    return {"schema": SCHEMA, "protocol": DESIGN, "prior_sha256": old["prior_sha256"],
            "geometry_sha256": sha(geometry.__file__), "weight_sha256": sha(geometry.WEIGHT),
            "worker_sha256": sha(__file__), "opencv": cv2.__version__, "numpy": np.__version__}


def observe(rgb, face):
    _, transform, mouth, _ = face
    aligned = cv2.warpAffine(rgb, transform, (96, 96), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    lab = cv2.cvtColor(aligned, cv2.COLOR_RGB2LAB).astype(np.float32)
    luma = lab[:, :, 0]
    dx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    dy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
    fine = np.abs(luma - cv2.GaussianBlur(luma, (5, 5), .8))
    moderate = (np.hypot(dx, dy) <= 18.) & (luma > 18) & (luma < 237)
    x, y = XX - mouth[0], YY - mouth[1]
    patches = ((-19, 5), (19, 5), (-24, -13), (24, -13), (0, -23))
    output = []
    for cx, cy in patches:
        pixels = (((x - cx) / 7) ** 2 + ((y - cy) / 6) ** 2 <= 1) & moderate
        if pixels.sum() < 40:
            return None
        values = lab[pixels]
        output.append([*np.median(values, axis=0), np.log1p(np.mean(fine[pixels]))])
    measures = np.asarray(output, dtype=np.float64)
    skin = np.stack((measures[0] - measures[2], measures[1] - measures[3]))
    cheeks = np.stack((measures[2] - measures[4], measures[3] - measures[4]))
    return np.stack((skin, cheeks))


def score_window(core, before, after):
    deviations = [core - flank for flank in (before, after)]
    change = []
    for delta in deviations:
        # The luma/chroma scale is fixed before scanning. Nearby skin must disagree less.
        color = np.linalg.norm(delta[..., :3] / (26, 11, 11), axis=-1)
        balance = float(np.min(color[0]) - np.max(color[1]))
        texture = float(np.min(-delta[0, :, 3]) - np.max(np.abs(delta[1, :, 3])))
        change.append((balance, min(balance, texture)))
    return np.min(change, axis=0)


def strongest(values, shots, points):
    proposals = dict.fromkeys(ARMS)
    measured = dict.fromkeys(ARMS, False)
    size = len(points)
    medians = {}
    for length in (5, 8, 12, 16, 20):
        windows = np.lib.stride_tricks.sliding_window_view(values, length, axis=0)
        valid = np.isfinite(windows[:, 0, 0, 0]).sum(axis=1) * 5 >= length * 4
        medians[length] = np.where(valid[:, None, None, None],
                                   np.ma.median(np.ma.masked_invalid(windows), axis=-1).filled(np.nan), np.nan)
    for length in (8, 12, 16, 20):
        for start in range(5, size - length - 4):
            stop = start + length
            indices = (start - 1, (start + stop) // 2, stop)
            if shots[start - 5] != shots[stop + 4] or not all(points[i] for i in indices):
                continue
            scores = score_window(medians[length][start], medians[5][start - 5], medians[5][stop])
            for arm, raw in zip(ARMS, scores, strict=True):
                if not np.isfinite(raw):
                    continue
                measured[arm] = True
                previous = proposals[arm]
                if raw > 0 and (previous is None or raw > previous["score"]):
                    proposals[arm] = {"score": float(raw), "frames": [start, stop],
                                      "anchors": [points[i] for i in indices]}
    return {arm: {"measured": measured[arm], "proposal": proposals[arm]} for arm in ARMS}


def inspect(path, detector, cache):
    table = probe_frames(path, cap=110, timeout_s=15)
    require(not (table.truncated or table.rotation or table.interpolated) and 40 <= len(table) <= 100,
            "Expected original-length verified video")
    track = ShotDetector(table)
    values = np.full((len(table), 2, 2, 4), np.nan)
    shots, anchors = [], []
    for frame in decode_frames(path, table, None, timeout_s=30):
        i = frame.index
        require(i == len(shots) and frame.pts_verified and table.pts[i] == frame.pts, "Unverified source frame")
        track.update(i, frame.rgb)
        shots.append(len(track.boundaries))
        digest = hashlib.sha256(frame.rgb).hexdigest()
        if digest not in cache:
            face = geometry.aligned_face(frame.rgb, detector)
            if face is None:
                cache[digest] = None, None
            else:
                x0, y0, x1, y1 = geometry.native_bounds(face)
                crop = frame.rgb[y0:y1, x0:x1]
                measurement = observe(frame.rgb, face)
                cache[digest] = (measurement, {"rgb_sha256": digest, "mouth_crop_xyxy": [x0, y0, x1, y1],
                                                "crop_sha256": hashlib.sha256(crop.tobytes()).hexdigest()})
        value, proof = cache[digest]
        if value is not None:
            values[i] = value
        anchors.append({"frame": i, "pts": frame.pts, **proof} if proof is not None else None)
    require(len(shots) == len(table), "Incomplete native video")
    result = strongest(values, shots, anchors)
    for entry in result.values():
        if proposal := entry["proposal"]:
            proposal["seconds"] = [str(table.pts[i] * table.tb) for i in proposal["frames"]]
    return table, result, int(np.isfinite(values[:, 0, 0, 0]).sum())


def cutoff(rows):
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
                     "group": row["group"], "sha256": sha(path), "truth": truth, "frames": len(table),
                     "time_base": table.time_base, "arms": arms, "frame_support": support})
        if len(rows) % 8 == 0:
            print(phase, len(rows), flush=True)
    require(len(rows) == {"development": 56, "held": 92}[phase], "Incomplete frozen panel")
    write(work / f"{phase}.json", rows)
    if phase == "development":
        write(work / "points.json", {"development_sha256": sha(work / "development.json"), "thresholds": cutoff(rows)})


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
        saved = read(args.work / "points.json")
        require(saved["development_sha256"] == sha(fit) and saved["thresholds"] == cutoff(read(fit)), "Unverified operating points")
    run(args.parent, args.work, args.phase)


if __name__ == "__main__":
    main()
