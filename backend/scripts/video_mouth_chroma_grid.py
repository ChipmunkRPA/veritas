# SPDX-License-Identifier: Apache-2.0
"""Evaluate native chroma sampling phase across short mouth edits."""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.shots import ShotDetector
from scripts import video_mouth_track_cycle as sources
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.video-mouth-chroma-grid/1"
ARMS = ("dominant_axis", "both_axes")
FLOOR = 0.03
SPANS = (8, 12, 16, 20)
FLANK = 5
MIN_PHASE = 0.10
PAIR_MINIMUM = 32
SPEC = {
    "input": "Single decoded upright clip with verified native pictures and single-face geometry. No earlier copy, audio, supplied markers, or authoring metadata enters measurement.",
    "phase": "Measure horizontal and vertical adjacent-pixel color differences by native parity in the mouth ellipse and the two cheek and upper-face comparison patches. Exclude near-clipping and large luminance transitions, and cap chroma differences. Learn dominant axis and phase sign from the median surrounding patch response in each observed shot. A mouth score subtracts the signed mouth parity from the corresponding surrounding response.",
    "measurement": "At least 32 eligible pairs per parity per region, mean chroma step at least 0.6, and two supported reference patches. A shot needs at least 16 measured reference frames per used axis, median asymmetry magnitude at least 0.10, and 75 percent agreement in sign. Measure either the strongest admissible axis or both axes using their weaker loss.",
    "range": "Scan native 8, 12, 16, or 20 picture windows with five-picture flanks in one observed shot. Require at least 80 percent supported observations in the core and both flanks, and a positive median response relative to both flanks. Keep one strongest interval.",
    "point": "Freeze the third-largest positive maximum among 56 ordinary GRID development clips, subject to a floor of 0.03, then require strict exceedance. Neither held bank is used for point selection.",
    "codec_stress": "Independently deliver original and source-time redraw of each paired LivePortrait and MuseTalk input as H.264 CRF 23 yuv420p. Preserve every rational source timestamp. Compare the same clips and exact edit ranges before and after final 4:2:0 delivery.",
    "gate": "At least four correctly localized strict LivePortrait redraws across three source groups, the strict MuseTalk edit, at least 70 percent measured coverage in each class, at most two ordinary flags and none on untouched, hard-cut or recompression controls. At least one half temporal IoU. Report the added codec stress separately.",
}


def frozen(lombard, fresh, muse, earlier_controls):
    old = sources.freeze(lombard, fresh, muse)
    earlier = read(earlier_controls / "freeze.json")
    controls = read(earlier_controls / "controls.json")
    require(
        controls["freeze_sha256"] == sha(earlier_controls / "freeze.json")
        and earlier["control_sources"] == old["control_sources"]
        and len(controls["rows"]) == 12,
        "Expected the committed ordinary edit controls from the earlier study",
    )
    rows = [*old["rows"], *controls["rows"]]
    stress = [row for row in rows if (row["family"], row["role"]) in {
        ("lombard", "source_time_neural"), ("lombard", "original"),
        ("musetalk", "same_time_neural"), ("musetalk", "original"),
    }]
    require(len(stress) == 34, "The codec stress must include all 17 matched pairs")
    return {
        "schema": SCHEMA, "protocol": SPEC, "rows": rows, "stress_sources": stress,
        "inputs_sha256": {"ordinary_controls": sha(earlier_controls / "controls.json"),
                          "detector_weights": old["weights_sha256"], "worker": sha(Path(__file__)),
                          "source_planner": sha(Path(sources.__file__)),
                          "geometry": old["geometry_code_sha256"]},
        "opencv": cv2.__version__, "numpy": np.__version__,
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
    }


def local_grid(rgb, geometry):
    """Classify pair midpoints with geometry while keeping raw pixel parity."""
    _, matrix, mouth, _ = geometry
    # The surrounding patches are above and to both sides of the mouth.
    patches = ((0, 0, 18, 10), (-25, -12, 9, 10), (25, -12, 9, 10), (0, -24, 14, 7))
    inverse = cv2.invertAffineTransform(matrix)
    bounds = np.asarray(((mouth[0] - 37, mouth[1] - 34, 1), (mouth[0] + 37, mouth[1] - 34, 1),
                         (mouth[0] - 37, mouth[1] + 13, 1), (mouth[0] + 37, mouth[1] + 13, 1))) @ inverse.T
    x0, y0 = np.maximum(0, np.floor(bounds.min(axis=0)).astype(int))
    x1, y1 = np.minimum(rgb.shape[1::-1], np.ceil(bounds.max(axis=0)).astype(int))
    measured = np.full((2, len(patches)), np.nan)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return measured
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    y, x = np.mgrid[y0:y1, x0:x1]
    relative = np.stack((x, y), axis=-1) @ matrix[:, :2].T + matrix[:, 2] - mouth
    red, green, blue = crop.transpose(2, 0, 1)
    luma = red * 0.299 + green * 0.587 + blue * 0.114
    chroma = np.stack(((red - luma) * 0.713, (blue - luma) * 0.564), axis=-1)
    valid = ((crop > 10) & (crop < 245)).all(axis=-1)
    for axis in (0, 1):
        parity = (y if axis == 0 else x) % 2
        first, second = ((np.s_[:-1], np.s_[1:]) if axis == 0 else (np.s_[:, :-1], np.s_[:, 1:]))
        mid = (relative[first] + relative[second]) * 0.5
        step = np.linalg.norm(chroma[first] - chroma[second], axis=-1)
        step = np.minimum(step, 8)
        usable = valid[first] & valid[second] & (np.abs(luma[first] - luma[second]) <= 14)
        for j, (dx, dy, rx, ry) in enumerate(patches):
            area = ((mid[..., 0] - dx) / rx) ** 2 + ((mid[..., 1] - dy) / ry) ** 2 <= 1
            mask = area & usable
            even = mask & (parity[first] == 0)
            odd = mask & (parity[first] == 1)
            if min(np.count_nonzero(even), np.count_nonzero(odd)) < PAIR_MINIMUM:
                continue
            e, o = float(np.mean(step[even])), float(np.mean(step[odd]))
            if (e + o) / 2 >= 0.6:
                measured[axis, j] = (e - o) / (e + o)
    return measured


def measure(rgb, detector, cache):
    digest = sources.prior.panel.raw_digest(rgb)
    if digest not in cache:
        geometry = sources.prior.panel.prior.aligned_face(rgb, detector)
        if geometry is None:
            cache[digest] = None, None
        else:
            x0, y0, x1, y1 = sources.prior.panel.native_bounds(geometry)
            cache[digest] = local_grid(rgb, geometry), {
                "rgb_sha256": digest, "crop_xyxy": [x0, y0, x1, y1],
                "crop_sha256": sources.prior.panel.raw_digest(rgb[y0:y1, x0:x1]),
            }
    return cache[digest]


def inspect(path, detector, cache):
    table = probe_frames(path, timeout_s=15)
    require(not (table.truncated or table.interpolated or table.rotation) and 40 <= len(table) <= 100,
            "Expected short native picture table")
    shots, anchors = [], []
    values = np.full((len(table), 2, 4), np.nan)
    track = ShotDetector(table)
    for frame in decode_frames(path, table, None, timeout_s=28):
        require(frame.index == len(shots) and frame.pts_verified and frame.pts == table.pts[frame.index],
                "Picture clock differs")
        track.update(frame.index, frame.rgb)
        shots.append(len(track.boundaries))
        observed, proof = measure(frame.rgb, detector, cache)
        if observed is not None:
            values[frame.index] = observed
        anchors.append({**proof, "frame": frame.index, "pts": frame.pts} if proof else None)
    require(len(shots) == len(table), "Incomplete native source pictures")
    return table, np.asarray(shots), values, anchors


def phase_response(values, shots):
    surrounding = values[:, :, 1:]
    supports = np.isfinite(surrounding).sum(axis=2) >= 2
    central = np.ma.median(np.ma.masked_invalid(surrounding), axis=2).filled(np.nan)
    response = np.full((len(values), len(ARMS)), np.nan)
    phase_shots = []
    for shot in np.unique(shots):
        group = shots == shot
        strengths, signs = np.zeros(2), np.zeros(2)
        for axis in (0, 1):
            indices = group & supports[:, axis]
            reference = central[indices, axis]
            if len(reference) >= 16:
                median = float(np.median(reference))
                if abs(median) >= MIN_PHASE and np.mean(reference * np.sign(median) > 0) >= 0.75:
                    strengths[axis], signs[axis] = abs(median), np.sign(median)
        phase_shots.append({"shot": int(shot), "strength": strengths.tolist(), "sign": signs.tolist()})
        axis = int(np.argmax(strengths))
        if signs[axis] == 0:
            continue
        difference = (np.where(supports[group], central[group], np.nan) - values[group, :, 0]) * signs
        response[group, 0] = difference[:, axis]
        if signs.all():
            response[group, 1] = np.min(difference, axis=1)
    return response, phase_shots


def select(table, shots, values, anchors):
    data, phase = phase_response(values, shots)
    candidates = dict.fromkeys(ARMS)
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
            score = windows[width][start] - np.maximum(windows[FLANK][start - FLANK], windows[FLANK][end])
            for i, arm in enumerate(ARMS):
                if not np.isfinite(score[i]):
                    continue
                assessed[arm] = True
                strength = float(score[i])
                if strength <= 0 or candidates[arm] and strength <= candidates[arm]["score"]:
                    continue
                candidates[arm] = {"score": strength, "frames": [start, end],
                                   "seconds": [str(table.pts[j] * table.tb) for j in (start, end)],
                                   "time_base": table.time_base, "anchors": picked}
    return {arm: {"measured": assessed[arm], "proposal": candidates[arm]} for arm in ARMS}, phase


def prepare(lock, work):
    folder = work / "transcodes"
    folder.mkdir(mode=0o700)
    cases = []
    for row in lock["stress_sources"]:
        source = Path(row["file"])
        require(sha(source) == row["sha"], "Stress source differs")
        original = probe_frames(source, timeout_s=15)
        target = folder / f"{row['family']}-{row['id']}-{row['role']}.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-copyts", "-i", str(source), "-map", "0:v:0", "-an",
            "-fps_mode", "passthrough", "-enc_time_base", str(original.tb), "-frames:v", str(len(original)),
            "-video_track_timescale", str(original.tb.denominator), "-use_editlist", "0", "-c:v", "libx264",
            "-bf", "0", "-pix_fmt", "yuv420p", "-crf", "23", "-threads:v", "1", str(target),
        ], check=True, timeout=25)
        delivered = probe_frames(target, timeout_s=15)
        require([p * original.tb for p in original.pts] == [p * delivered.tb for p in delivered.pts],
                "Stress transcode changes the exact picture clock")
        cases.append({**row, "split": "codec_stress", "file": str(target), "sha": sha(target),
                      "source_sha": row["sha"], "verified_frames": len(delivered)})
    write(work / "codec_stress_roster.json", {"freeze_sha256": sha(work / "freeze.json"), "rows": cases})


def points_for(rows):
    result = {}
    for arm in ARMS:
        values = sorted((r["arms"][arm]["proposal"]["score"] for r in rows if r["arms"][arm]["proposal"]), reverse=True)
        result[arm] = max(FLOOR, values[2] if len(values) >= 3 else 0)
    return result


def score(lock, work, phase):
    if phase == "codec_stress":
        stress = read(work / "codec_stress_roster.json")
        require(stress["freeze_sha256"] == sha(work / "freeze.json") and len(stress["rows"]) == 34,
                "Codec stress roster differs")
        roster = stress["rows"]
    else:
        roster = lock["rows"]
    detector = load_yunet(sources.prior.WEIGHT)
    groups = {}
    for row in roster:
        if row["split"] == phase:
            groups.setdefault((row["family"], row["id"]), []).append(row)
    output = []
    for entries in groups.values():
        cache = {}
        for row in entries:
            path = Path(row["file"])
            require(sha(path) == row["sha"], "Frozen video bytes differ")
            table, shots, values, anchors = inspect(path, detector, cache)
            reading, phase_shots = select(table, shots, values, anchors)
            truth = [str(table.pts[i] * table.tb) for i in row["edit"]] if row.get("edit") else row.get("native_truth")
            if truth:
                require(all(Fraction(value) in {p * table.tb for p in table.pts} for value in truth), "Edit clock differs")
            output.append({**row, "arms": reading, "phase_shots": phase_shots,
                           "native_frames": len(table), "geometry_frames": sum(item is not None for item in anchors),
                           "truth": truth})
        if len(output) % 8 < len(entries):
            print(phase, len(output), flush=True)
    require(Counter(r["split"] for r in output) == {phase: {"development": 56, "held": 146, "codec_stress": 34}[phase]},
            "Incomplete run")
    path = work / (phase + ".json")
    write(path, {"freeze_sha256": sha(work / "freeze.json"), "rows": output})
    if phase == "development":
        write(work / "points.json", {"development_sha256": sha(path), "points": points_for(output)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "development", "held", "codec_stress"))
    for name in ("work", "lombard", "fresh", "muse", "earlier_controls"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    lock = frozen(a.lombard, a.fresh, a.muse, a.earlier_controls)
    path = a.work / "freeze.json"
    if a.phase == "freeze":
        a.work.mkdir(mode=0o700)
        write(path, lock)
        return
    require(read(path) == lock, "Frozen sources, runtime or method differ")
    cv2.setNumThreads(1)
    if a.phase == "prepare":
        prepare(lock, a.work)
        return
    if a.phase != "development":
        dev = a.work / "development.json"
        require(read(a.work / "points.json") == {"development_sha256": sha(dev),
                                                 "points": points_for(read(dev)["rows"])}, "Held points were not frozen")
    score(lock, a.work, a.phase)


if __name__ == "__main__":
    main()
