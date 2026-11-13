# SPDX-License-Identifier: Apache-2.0
"""Test local color differences in matched video copies without new model weights."""

import argparse
import hashlib
import json
import subprocess
from contextlib import closing
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import video_reference as reference
from app.evidence import video_reference_local as local
from app.evidence import video_reference_photometric as luma
from scripts import video_reference_extent_controls as controls
from scripts import video_reference_robust_transfer as sources
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.video-reference-chroma-transfer/1"
ARMS = ("luma", "chroma")
WINDOWS = (8, 12, 16, 20)
FLOOR = .8
MARGIN = .15
PRIOR = "6f5bac4581103053f8f57031143cab0e1dfb99f218b9ea4728aca217994698ff"
DESIGN = {
    "fit": "Eight GRID development originals, ordinary recompression and global color changes. Use the largest gated ordinary chroma response plus .15, with a floor of .8. Keep the serving luminance point.",
    "held": "17 retained, previously exposed neural redraws and 34 ordinary local mouth operations. Eight source-disjoint CREMA-D actors contribute independently authored local color edits and ordinary delivery controls.",
    "measurement": "Blur decoded Cb and Cr; remove robust corresponding-pixel global scale and shift per channel. Fit uses no face, timing label, mouth crop, or audio. Require the same broad-picture luma compatibility as serving, then select an 8/12/16/20-picture interval and expand its selected tile to a connected area.",
    "gate": "Union must add at least two useful timing-preserving redraw ranges in distinct source groups, including a second-family case, with >= .50 temporal and .10 middle-picture pixel IoU. Also recover six of eight color interventions, allow one extra neutral flag, and admit no wrong-source pairs. Ordinary local edits can produce the same evidence.",
}


def freeze(previous, public):
    require(sha(previous / "held.json") == PRIOR, "Earlier mouth comparison differs")
    roster = read(previous / "freeze.json")["roster"]
    require(sum(r["role"] in {"source_time_neural", "same_time_neural"} for r in roster) == 17, "Neural roster differs")
    sources.manifest(public)
    return {"schema": SCHEMA, "design": DESIGN, "mouth_manifest_sha256": sha(previous / "freeze.json"),
            "mouth_held_sha256": PRIOR, "public_roster_sha256": sha(public), "runner_sha256": sha(__file__),
            "luma_sha256": sha(luma.__file__), "opencv": cv2.__version__, "numpy": np.__version__}


def fit_axis(x, y):
    gain, shift = 1., float(np.median(y - x))
    for _ in range(3):
        residual = np.abs(y - (gain * x + shift))
        keep = residual <= np.quantile(residual, .72)
        actual, target = x[keep], y[keep]
        center = actual - actual.mean()
        if float(np.mean(center**2)) >= 9:
            gain = float(np.clip(np.dot(center, target - target.mean()) / np.dot(center, center), .60, 1.65))
        shift = float(np.clip(np.median(target - gain * actual), -42, 42))
    return gain, shift


def chroma_residual(a, b):
    first, second = [cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[..., 1:].astype(np.float32), (7, 7), 1.2)
                     for rgb in (a, b)]
    low_first, low_second = (cv2.resize(item, (32, 32), interpolation=cv2.INTER_AREA).reshape(-1, 2)
                             for item in (first, second))
    parts = []
    for i in range(2):
        gain, shift = fit_axis(low_second[:, i], low_first[:, i])
        parts.append(first[..., i] - np.clip(gain * second[..., i] + shift, 0, 255))
    return np.hypot(*parts)


def feature(residual, rects):
    x, y, w, h = rects
    total = cv2.integral(residual, sdepth=cv2.CV_64F)
    tiles = (total[y+h, x+w] - total[y+h, x] - total[y, x+w] + total[y, x]) / (w*h)
    grid = cv2.resize(residual, (residual.shape[1] // 8, residual.shape[0] // 8), interpolation=cv2.INTER_AREA)
    return np.maximum(0, tiles - np.percentile(tiles, 65)), np.maximum(0, grid - np.percentile(grid, 65))


def strongest(values, rects):
    values = np.asarray(values)
    best = None
    x, y, w, h = rects
    for length in WINDOWS:
        for first in range(5, len(values) - length - 4):
            stop = first + length
            core = np.median(values[first:stop], axis=0)
            neighbors = np.maximum(np.median(values[first-5:first], axis=0), np.median(values[stop:stop+5], axis=0))
            difference = core - neighbors
            i = int(np.argmax(difference))
            if best is None or difference[i] > best["score"]:
                best = {"score": float(difference[i]), "frames": [first, stop],
                        "tile": {"x": int(x[i]), "y": int(y[i]), "w": w, "h": h},
                        "samples": {"before": first - 1, "during": (first + stop) // 2, "after": stop}}
    return best


def evidence(paths, tables, choice, hashes, mask):
    x, y, w, h = (choice["tile"][key] for key in "xywh")
    selected = list(choice["samples"].values())
    anchors = {i: {"frame": i, "pts": [t.pts[i] for t in tables], "rgb_sha256": hashes[i], "crop_sha256": []}
               for i in selected}
    for side, (path, table) in enumerate(zip(paths, tables, strict=True)):
        for frame in decode_frames(path, table, selected, timeout_s=12):
            rgb = hashlib.sha256(frame.rgb).hexdigest()
            require(frame.pts_verified and rgb == hashes[frame.index][side], "Native evidence picture differs")
            anchors[frame.index]["crop_sha256"].append(hashlib.sha256(frame.rgb[y:y+h, x:x+w].tobytes()).hexdigest())
    choice["anchors"] = list(anchors.values())
    require(all(len(a["crop_sha256"]) == 2 for a in choice["anchors"]), "Native crops incomplete")
    if mask:
        truth = mask(choice["samples"]["during"])
        choice["pixel_support"] = {"changed": int(truth.sum()), "covered": int(truth[y:y+h, x:x+w].sum())}


def analyze(paths, *, points=None, mask=None):
    tables = tuple(probe_frames(path, cap=105, timeout_s=12) for path in paths)
    clocks = [reference._elapsed(table) for table in tables]
    require(clocks[0] == clocks[1] and all(len(t) >= 40 and not (t.rotation or t.interpolated or t.truncated) for t in tables),
            "Native matched clock differs")
    rects = local.rectangles(tables[0].height, tables[0].width)
    lumas, chromas, luma_grid, chroma_grid, bg, hashes = [], [], [], [], [], []
    for a, b in reference._frame_pairs(paths, tables):
        value, background, grid = luma.measurements(a.rgb, b.rgb, rects)
        lumas.append(value)
        luma_grid.append(grid)
        bg.append(background)
        color, cells = feature(chroma_residual(a.rgb, b.rgb), rects)
        chromas.append(color)
        chroma_grid.append(cells)
        hashes.append([hashlib.sha256(item.rgb).hexdigest() for item in (a, b)])
    require(len(hashes) == len(tables[0]), "Incomplete matched decode")
    status, best_luma = luma.assess(lumas, bg, rects)
    shared = status not in {"insufficient_shared_picture_content", "unverified_photometric_support"}
    result = {"status": status, "shared": shared, "pictures": len(hashes), "time_bases": [t.time_base for t in tables],
              "first_pts": [t.pts[0] for t in tables], "steps": [t.pts[1] - t.pts[0] for t in tables],
              "files_sha256": [sha(path) for path in paths], "arms": dict.fromkeys(ARMS)}
    if not shared:
        return result
    for arm, choice, grid in (("luma", best_luma, luma_grid), ("chroma", strongest(chromas, rects), chroma_grid)):
        if choice is None:
            continue
        result["arms"][arm] = {"score": choice["score"]}
        if points is None or choice["score"] <= points[arm]:
            continue
        choice["tile"] = luma.expanded_region(np.asarray(grid), choice, constrained=False)
        evidence(paths, tables, choice, hashes, mask)
        result["arms"][arm] = choice
    return result


def native_truth(paths, before):
    def mask(index):
        pictures = []
        for path, row in zip(paths, before, strict=True):
            require(sha(path) == row["original_sha256"], "Native truth bytes differ")
            table = probe_frames(path, cap=105, timeout_s=12)
            with closing(decode_frames(path, table, [index], timeout_s=10)) as frames:
                picture = next(frames)
                require(picture.pts_verified and picture.index == index, "Invalid truth picture time")
                pictures.append(picture.rgb)
        return np.any(pictures[0] != pictures[1], axis=2)
    return mask


def grade(frame, *, local_edit=False):
    value = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    region = value[130:222, 244:404, :] if local_edit else value
    region[..., 1] = np.clip(region[..., 1].astype(np.int16) + (33 if local_edit else 7), 0, 255)
    region[..., 2] = np.clip(region[..., 2].astype(np.int16) - (24 if local_edit else 5), 0, 255)
    adjusted = cv2.cvtColor(value, cv2.COLOR_YCrCb2RGB)
    if not local_edit:
        return adjusted
    result = frame.copy()
    result[130:222, 244:404] = adjusted[130:222, 244:404]
    return result


def write_video(path, frames, *, crf=30):
    controls.encode(path, frames, crf)


def ordinary(rows, previous, run):
    from scripts.video_paired_native_mouth_transfer import authored_view, checked

    fit = []
    for i, row in enumerate(rows):
        path = Path(row["carrier"])
        require(sha(path) == row["carrier_sha256"], "GRID carrier differs")
        table = checked(path)
        sources_ = {"original": path, "compression": previous / f"development-{i:02d}-compression.mp4"}
        target = run / f"dev-{i:02d}-global.mp4"
        authored_view(Path(row["native"]), target, table, "tone")
        # ffmpeg color balance uses chroma shifts and keeps the existing clock.
        dest = run / f"dev-{i:02d}-color.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-n", "-v", "error", "-threads", "1", "-i", str(target), "-map", "0:v:0",
                        "-an", "-vf", "eq=saturation=1.17:gamma_b=1.06", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "32",
                        "-threads", "1", "-bf", "0", "-fps_mode", "passthrough", "-video_track_timescale", str(table.tb.denominator),
                        str(dest)], check=True, timeout=25)
        sources_["global_chroma"] = dest
        for role, other in sources_.items():
            fit.append({"bank": "grid", "group": i, "role": role, "truth": None} | analyze((path, other)))
        print("development", i + 1, flush=True)
    return fit


def held_neural(rows, previous, points):
    prior = read(previous / "held.json")["rows"]
    by_hash = {r["carrier_sha256"]: r for r in rows}
    for item in prior:
        if item["role"] not in {"source_time_neural", "same_time_neural", "resample_ordinary", "delayed_pixels_ordinary",
                                 "same_time_resample", "delayed_pixels"}:
            continue
        first, second = (by_hash[item[k]] for k in ("earlier_sha256", "submitted_sha256"))
        paths = tuple(Path(r["carrier"]) for r in (first, second))
        require(all(sha(path) == row["carrier_sha256"] for path, row in zip(paths, (first, second), strict=True)), "Frozen carrier differs")
        truth = second.get("edit")
        if truth is None and second.get("native_truth"):
            table = probe_frames(paths[1], cap=105)
            ticks = [t * table.tb for t in table.pts]
            truth = [ticks.index(Fraction(t)) for t in second["native_truth"]]
        mask = native_truth([Path(r["native"]) for r in (first, second)], (first, second)) if truth else None
        print("held", item["family"], item["case"], item["role"], flush=True)
        yield {"bank": item["family"], "group": item["group"], "case": item["case"], "role": item["role"],
               "aligned": item["aligned"], "truth": truth} | analyze(paths, points=points, mask=mask)


def held_public(public, archive, run, points):
    rows = sources.manifest(public)
    cache = run / "sources"
    cache.mkdir(mode=0o700)
    assets = [sources.copy_source(cache, i, row, archive) for i, row in enumerate(rows)]
    for i, path in enumerate(assets):
        frames = sources.input_pictures(path)
        earlier = run / f"crema-{i:02d}-earlier.mp4"
        write_video(earlier, frames, crf=23)
        for role in ("original", "compression", "global_chroma", "local_chroma", "wrong"):
            later = run / f"crema-{i:02d}-{role}.mp4"
            truth = None
            mask = None
            altered = frames
            if role == "original":
                later = earlier
            elif role == "global_chroma":
                altered = np.stack([grade(frame) for frame in frames])
            elif role == "local_chroma":
                altered = frames.copy()
                truth = [10 + i % 3, 26 + i % 3]
                for j in range(*truth):
                    altered[j] = grade(frames[j], local_edit=True)

                def mask(index, frames=frames, altered=altered):
                    return np.any(frames[index] != altered[index], axis=2)

            elif role == "wrong":
                altered = sources.input_pictures(assets[(i+1) % len(assets)])
            if later != earlier:
                write_video(later, altered)
            yield {"bank": "crema", "group": i, "role": role, "truth": truth, "source_sha256": rows[i]["sha256"]} | analyze(
                (earlier, later), points=points, mask=mask)
        print("CREMA-D", i + 1, flush=True)


def point(rows):
    return max(FLOOR, max((r["arms"]["chroma"] or {}).get("score", 0.) for r in rows) + MARGIN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held"))
    parser.add_argument("previous", type=Path)
    parser.add_argument("public", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    lock = freeze(args.previous, args.public)
    frozen = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(frozen, lock)
        return
    require(read(frozen) == lock, "Frozen method or roster differs")
    rows = read(args.previous / "freeze.json")["roster"]
    if args.phase == "development":
        fit = ordinary([r for r in rows if r["split"] == "development"], args.previous, args.run)
        write(args.run / "development.json", fit)
        write(args.run / "points.json", {"luma": luma.POINT, "chroma": point(fit)})
        return
    points = read(args.run / "points.json")
    require(points == {"luma": luma.POINT, "chroma": point(read(args.run / "development.json"))}, "Unverified development point")
    with (args.run / "held.jsonl").open("x") as journal:
        for cohort in (held_neural(rows, args.previous, points), held_public(args.public, args.archive, args.run, points)):
            for row in cohort:
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()


if __name__ == "__main__":
    main()
