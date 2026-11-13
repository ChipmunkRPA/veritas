# SPDX-License-Identifier: Apache-2.0
"""Test earlier-copy localization on matching native clocks and encoded mouth edits."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import Counter
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import video_reference as existing
from app.evidence import video_reference_local as local
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.video-paired-native-mouth-transfer/1"
INPUT_SHA = "f0cd3e5e07834e74bbdf2d70e9ccf273cc77ed2814c1ac3b8ba2129285937bf7"
ARMS = ("luminance", "luminance_and_texture")
PLAN = {
    "scope": "Use two upright, source-clock-matched H.264 4:2:0 clips. Test 360×288 GRID at 25 fps for development and 720×480 Lombard and MuseTalk around 24 fps for transfer.",
    "inputs": "Select 8 unique GRID references and 17 retained native neural source pairs with exact source-clock evidence. Each earlier file is H.264 CRF23. Both edit and matched ordinary local controls are retained carrier files.",
    "controls": "Per source, compare same clip, independent CRF32 encoding from the retained earlier file, full picture level adjustment at CRF32 and a wrong-source clip for which the reference clock is imposed. Include corresponding ordinary local resampling and pixel timing edits in the held set.",
    "comparisons": "Test the original blurred-luma difference and the weaker of that score and locally normalized high-pass luminance disagreement. Both use the same shared-picture gates and time/space selector. Search 8, 12, 16, 20 native frames and require five complete flanking frames on both sides.",
    "point": "Choose the maximum of 0.8 and largest gated ordinary GRID score plus 0.15 before observing the target sets. Admit only when the 65th-percentile background luma error is at most 3.0 and the 25th percentile at most 2.5.",
    "acceptance": "At least six correct ranges with both temporal IoU ≥0.5 and spatial IoU ≥0.1 at the selected middle frame, including four independently timing-qualified LivePortrait clips over at least three source groups and the independently timing-qualified MuseTalk clip. At most two false ranges on unchanged, compression, or level controls and no wrong-source admissions. Treat ordinary local changes as differences, not false AI flags.",
}


def freeze(prior):
    require(sha(prior) == INPUT_SHA, "Retained video roster differs")
    roster = read(prior)["rows"]
    arms = []
    for row in roster:
        family, ident, role = row["family"], row["id"], row["role"]
        if family == "grid" and role != "original" or family not in {"grid", "lombard", "musetalk"}:
            continue
        arms.append({key: value for key, value in row.items() if key not in {"file", "original"}} |
                    {"native": row["original"], "source_index": ident})
    require(Counter((r["split"], r["role"]) for r in arms) ==
            {("development", "original"): 8, ("held", "original"): 17, ("held", "source_time_neural"): 11,
             ("held", "same_time_neural"): 6, ("held", "resample_ordinary"): 11,
             ("held", "delayed_pixels_ordinary"): 11, ("held", "same_time_resample"): 6,
             ("held", "delayed_pixels"): 6}, "Unexpected source assignments")
    return {"schema": SCHEMA, "protocol": PLAN, "prior_sha256": INPUT_SHA, "runner_sha256": sha(Path(__file__)),
            "reference_code_sha256": sha(Path(local.__file__)), "roster": arms}


def authored_view(source, target, clock, condition):
    step = clock.pts[1] - clock.pts[0]
    filters = f"settb={clock.time_base},setpts=N*{step}+{clock.pts[0]}"
    if condition == "tone":
        filters += ",eq=brightness=0.055:contrast=0.91"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-map", "0:v:0", "-an",
                    "-vf", filters, "-fps_mode", "passthrough", "-frames:v", str(len(clock)), "-use_editlist", "0",
                    "-enc_time_base", clock.time_base, "-video_track_timescale", str(clock.tb.denominator),
                    "-c:v", "libx264", "-bf", "0", "-crf", "32", "-pix_fmt", "yuv420p", "-threads", "1", str(target)],
                   check=True, timeout=25)


def checked(path, reference=None):
    clock = probe_frames(path, cap=100, timeout_s=10)
    require(not (clock.truncated or clock.rotation or clock.interpolated) and 40 <= len(clock) <= 100,
            "Invalid video source profile")
    if reference is not None:
        require((clock.width, clock.height) == (reference.width, reference.height)
                and [v * clock.tb for v in clock.pts] == [v * reference.tb for v in reference.pts],
                "Picture clock changed during encoding")
    return clock


def tile_mean(values, rects):
    x, y, w, h = rects
    total = cv2.integral(values, sdepth=cv2.CV_64F)
    return (total[y+h, x+w] - total[y, x+w] - total[y+h, x] + total[y, x]) / (w*h)


def measurements(first, second, rects):
    luma, background = local.measure(first, second, rects)
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) for frame in (first, second)]
    high = [value - cv2.GaussianBlur(value, (5, 5), 1) for value in gray]
    energy = [tile_mean(np.abs(value), rects) for value in high]
    texture = tile_mean(np.abs(high[0] - high[1]), rects) / (2 + np.maximum(*energy))
    texture = np.maximum(0, texture - np.percentile(texture, 65)) * 12
    return luma, np.minimum(luma, texture), background


def top(surfaces, rects):
    values = np.asarray(surfaces)
    best = None
    x, y, w, h = rects
    for width in (8, 12, 16, 20):
        for start in range(5, len(values) - width - 4):
            end = start + width
            views = values[start-5:end+5]
            if not np.isfinite(views).all():
                continue
            response = np.median(views[5:5+width], axis=0) - np.maximum(
                np.median(views[:5], axis=0), np.median(views[-5:], axis=0))
            index = int(np.argmax(response))
            if best is None or float(response[index]) > best["score"]:
                best = {"score": float(response[index]), "frames": [start, end],
                        "tile": [int(x[index]), int(y[index]), int(x[index]+w), int(y[index]+h)]}
    return best


def inspect(paths, tables):
    rects = local.rectangles(tables[0].height, tables[0].width)
    observations, bg, hashes = [], [], []
    for a, b in existing._frame_pairs(paths, tables):
        luma, texture, background = measurements(a.rgb, b.rgb, rects)
        observations.append((luma, texture))
        bg.append(background)
        hashes.append([hashlib.sha256(item.rgb).hexdigest() for item in (a, b)])
    require(len(hashes) == len(tables[0]), "Incomplete picture comparison")
    valid = [value for value in bg if value is not None]
    if len(valid) < .90 * len(bg):
        return "insufficient_luma", {}, hashes
    if np.percentile(valid, 65) > 3 or np.percentile(valid, 25) > 2.5:
        return "no_shared_content", {}, hashes
    return "measured", {arm: top([v[i] for v in observations], rects) for i, arm in enumerate(ARMS)}, hashes


def cover(clock, native_pair, box, index):
    x, y, right, bottom = box
    pictures = []
    for path in native_pair:
        source_clock = checked(path)
        require((source_clock.width, source_clock.height) == (clock.width, clock.height)
                and existing._elapsed(source_clock) == existing._elapsed(clock), "Native edit and carrier clocks differ")
        with closing(decode_frames(path, source_clock, [index], timeout_s=10)) as frames:
            frame = next(frames)
            require(frame.pts_verified and frame.index == index, "Unverified native truth picture")
            pictures.append(frame.rgb)
    a, b = pictures
    truth = np.any(a != b, axis=2)
    covered = int(truth[y:bottom, x:right].sum())
    total = int(truth.sum())
    return {"covered": covered, "changed": total, "tile_pixels": (right-x)*(bottom-y), "index": index}


def score(lock, run, phase, point=None):
    rows = [r for r in lock["roster"] if r["split"] == phase]
    for row in rows:
        require(sha(row["carrier"]) == row["carrier_sha256"] and sha(row["native"]) == row["original_sha256"],
                "Archived source hash differs")
    originals = [r for r in rows if r["role"] == "original"]
    results = []
    for index, ref in enumerate(originals):
        old_path = Path(ref["carrier"])
        clock = checked(old_path)
        wrong = next((r for r in originals if r["family"] == ref["family"] and r["group"] != ref["group"]
                      and r["verified_frames"] >= len(clock)), None)
        cases = [(r["role"], Path(r["carrier"]), r) for r in rows
                 if (r["family"], r["source_index"]) == (ref["family"], ref["source_index"])]
        for role in ("compression", "tone", "wrong_source"):
            if role == "wrong_source" and wrong is None:
                continue
            target = run / f"{phase}-{index:02d}-{role}.mp4"
            source = wrong if role == "wrong_source" else ref
            authored_view(Path(source["native"]), target, clock, role)
            cases.append((role, target, None))
        for role, path, row in cases:
            other_clock = checked(path, clock)
            status, measures, hashes = inspect((old_path, path), (clock, other_clock))
            truth = None if row is None else (row.get("edit") or row.get("native_truth"))
            if truth and isinstance(truth[0], int):
                truth = [str(clock.pts[i] * clock.tb) for i in truth]
            evidence = {}
            for arm, candidate in measures.items():
                if point is None or candidate is None or candidate["score"] <= point[arm]:
                    continue
                start, end = candidate["frames"]
                anchors = [start - 1, (start + end) // 2, end]
                evidence[arm] = {"interval": [str(clock.pts[i] * clock.tb) for i in (start, end)],
                                 "anchors": [{"index": i, "pts": clock.pts[i], "rgb": hashes[i]} for i in anchors]}
                if truth:
                    evidence[arm]["pixels"] = cover(clock, (Path(ref["native"]), Path(row["native"])),
                                                    candidate["tile"], anchors[1])
            results.append({"family": ref["family"], "group": ref["group"], "case": index, "role": role,
                            "status": status, "earlier_sha256": ref["carrier_sha256"], "submitted_sha256": sha(path),
                            "time_base": clock.time_base, "pts_sha256": existing.canonical_hash(list(clock.pts)),
                            "pictures": len(clock), "truth": truth, "aligned": bool(row and row.get("aligned")),
                            "measurements": measures, "evidence": evidence})
        print(phase, index + 1, len(originals), flush=True)
    return results


def fit_points(rows):
    return {arm: max(.8, max([0., *[r["measurements"][arm]["score"] for r in rows
                                  if r["measurements"].get(arm) is not None]]) + .15) for arm in ARMS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held"))
    parser.add_argument("carriers", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    current = freeze(args.carriers)
    frozen = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(frozen, current)
        return
    require(read(frozen) == current, "Freeze differs")
    points = None
    if args.phase == "held":
        fit, saved = args.run / "development.json", read(args.run / "points.json")
        require(saved["development_sha256"] == sha(fit) and saved["points"] == fit_points(read(fit)["rows"]),
                "Operating point not bound to the ordinary fit")
        points = saved["points"]
    output = args.run / f"{args.phase}.json"
    rows = score(current, args.run, args.phase, points)
    write(output, {"freeze_sha256": sha(frozen), "rows": rows})
    if points is None:
        write(args.run / "points.json", {"development_sha256": sha(output), "points": fit_points(rows)})


if __name__ == "__main__":
    main()
