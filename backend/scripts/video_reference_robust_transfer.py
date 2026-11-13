# SPDX-License-Identifier: Apache-2.0
"""Rebuild a licensed paired-video trial with separate photometric and region arms."""

import argparse
import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import video_reference as reference
from app.evidence import video_reference_local as local
from app.evidence import video_reference_photometric as candidate
from scripts import video_reference_extent_controls as controls
from scripts import video_reference_extent_method as baseline
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.paired-video-robust-transfer/1"
RELEASE = "1658cd342dff90010aa843eaeebd53610a08b1dc"
ROLES = ("large", "small", "local_texture", "compression", "global", "wrong")
ARMS = ("baseline", "bounded_region", "robust_tone", "combined")
PLAN = {
    "basis": "Matched upright 720 x 480 H.264, 52 native pictures at 24 fps, fixed earlier-copy luminance operating point.",
    "cohorts": "Two consumed CREMA-D pairs of interest and eight prior-extent-disjoint CREMA-D sources with frozen public bytes. The eight had exposure in other project studies. No source chosen using a response from this method.",
    "operations": "Three local edits per held source. Large and small local grade with known boxes and half-open interval. Texture edit with same-clock horizontal pixel copy inside a separate box and tapered sides. Ordinary controls are strong recompression, a full-clip affine grade, and wrong source. No generator used.",
    "arms": "Baseline quantile normalization and component; bounded component alone; corresponding-pixel robust affine fit alone; and both. No change to clocks, time selector, or public confidence.",
    "advancement": "At least 7/8 large edits, 7/8 small, and 5/8 local texture by 0.5 time and 0.1 pixel IoU. At least two net locations or 0.10 improved median spatial IoU on one edit type, at most one lost location overall. At most one added false neutral flag and zero admitted wrong sources. Show source-pair counts and distinguish time from crop.",
}


def manifest(path):
    roster = read(path)
    require(roster["schema"] == "chipmunk.paired-video-photometric-sources/1" and roster["release"] == RELEASE
            and (roster["dataset_license"], roster["recordings_license"]) == ("ODbL-1.0", "DbCL-1.0"),
            "Unknown licensed source manifest")
    rows = roster["sources"]
    require(len(rows) == len({r["group"] for r in rows}) == len({r["sha256"] for r in rows}) == 8,
            "Expected eight disjoint recordings")
    for row in rows:
        require(re.fullmatch(r"VideoFlash/10\d{2}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}\.flv", row["source_path"])
                and 0 < row["bytes"] <= 1_000_000 and re.fullmatch(r"[a-f0-9]{64}", row["sha256"])
                and row["url"] == ("https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/"
                                  f"{RELEASE}/{row['source_path']}"), "Source outside the pinned publisher release")
    return rows


def copy_source(cache, index, row, archive):
    path = cache / f"source-{index:02d}.flv"
    if not path.exists():
        if archive:
            matches = [p for p in archive.glob("asset-*.flv") if p.stat().st_size == row["bytes"] and sha(p) == row["sha256"]]
            require(len(matches) == 1, "Expected one archived source by exact size and hash")
            shutil.copyfile(matches[0], path)
        else:
            request = Request(row["url"], headers={"User-Agent": "ChipmunkVerify/paired-video-review"})
            with urlopen(request, timeout=25) as response:
                data = response.read(row["bytes"] + 1)
            require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], "Publisher bytes differ")
            path.write_bytes(data)
    require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Cached source differs")
    return path


def input_pictures(path):
    clock = probe_frames(path, cap=110, timeout_s=15)
    require(len(clock) >= 52 and not (clock.rotation or clock.interpolated), "Need 52 verified upright source pictures")
    images = []
    for picture in decode_frames(path, clock, range(52), timeout_s=25):
        require(picture.pts_verified and picture.index == len(images), "Native picture clock is incomplete")
        images.append(cv2.resize(picture.rgb, (720, 480), interpolation=cv2.INTER_AREA))
    require(len(images) == 52, "Source decoder stopped early")
    return np.array(images)


def analyze(paths, mask=None):
    tables = tuple(probe_frames(path, cap=100, timeout_s=15) for path in paths)
    clocks = [reference._elapsed(table) for table in tables]
    require(clocks[0] == clocks[1] and baseline.eligible(paths, tables, clocks[0]), "The paired delivery clocks differ")
    hashes = []

    def pictures():
        for first, second in reference._frame_pairs(paths, tables):
            hashes.append([hashlib.sha256(frame.rgb).hexdigest() for frame in (first, second)])
            yield first.rgb, second.rgb

    rects = local.rectangles(tables[0].height, tables[0].width)
    raw, robust, raw_cells, robust_cells = [], [], [], []
    for first, second in pictures():
        raw.append(local.measure(first, second, rects))
        raw_cells.append(baseline._residual(first, second) if raw[-1][1] is not None else
                         np.full((first.shape[0] // baseline.CELL, first.shape[1] // baseline.CELL), np.nan))
        signal, background, cells = candidate.measurements(first, second, rects)
        robust.append((signal, background))
        robust_cells.append(cells)
    results = {}
    for name, measured, grid, constrained in (("baseline", raw, raw_cells, False),
                                               ("bounded_region", raw, raw_cells, True),
                                               ("robust_tone", robust, robust_cells, False),
                                               ("combined", robust, robust_cells, True)):
        values, backgrounds = zip(*measured, strict=True)
        status, choice = candidate.assess(values, backgrounds, rects)
        result = {"status": status, "background65": float(np.percentile(valid, 65)) if (
            valid := [v for v in backgrounds if v is not None]
        ) else None}
        if choice is not None:
            result |= {"score": choice["score"], "frames": choice["frames"], "samples": choice["samples"],
                       "tile": choice["tile"], "extent": candidate.expanded_region(np.array(grid), choice,
                                                                                  constrained=constrained)}
        results[name] = result
    require(len(hashes) == len(tables[0]), "Incomplete paired delivery")
    for result in results.values():
        if "samples" not in result:
            continue
        result["anchors"] = {role: {"frame": index, "time": str(clocks[0][index]), "rgb_sha256": hashes[index]}
                             for role, index in result.pop("samples").items()}
        if mask is not None:
            truth = mask[result["anchors"]["during"]["frame"]]
            result["changed_pixels"] = int(truth.sum())
            for arm in ("tile", "extent"):
                x, y, w, h = [result[arm][k] for k in ("x", "y", "w", "h")]
                result[f"{arm}_overlap"] = int(truth[y:y + h, x:x + w].sum())
    return {"arms": results, "clock_sha256": reference.canonical_hash([str(t) for t in clocks[0]]),
            "frame_count": len(hashes), "files_sha256": [sha(path) for path in paths]}


def edited(frames, index, role):
    span = (11 + index % 4, 28 + index % 5)
    size = {"large": (180, 112), "small": (72, 48), "local_texture": (136, 96)}
    w, h = size[role]
    x, y = 74 + (87 * index + (91 if role == "local_texture" else 0)) % 390, 75 + 54 * (index % 4)
    start, stop = span
    altered = frames.copy()
    original = frames[start:stop, y:y + h, x:x + w]
    if role == "local_texture":
        alpha = np.minimum(1, np.minimum(np.arange(w), np.arange(w)[::-1]) / 6).astype(np.float32)
        alternate = frames[start:stop, y:y + h, x + 20:x + w + 20]
        change = (1 - alpha[None, None, :, None]) * original + alpha[None, None, :, None] * alternate
    else:
        gain, offset = ((.63, 88) if index % 2 else (1.18, -63)) if role == "large" else (.68, 70)
        change = original.astype(np.float32) * gain + offset
    altered[start:stop, y:y + h, x:x + w] = np.clip(change, 0, 255).astype(np.uint8)
    return altered, span


def held(out, paths):
    rows = []
    for i, source in enumerate(paths):
        frames = input_pictures(source)
        earlier = out / f"held-{i:02d}-earlier.mp4"
        controls.encode(earlier, frames, 23)
        for role in ROLES:
            submitted = out / f"held-{i:02d}-{role}.mp4"
            span = mask = None
            altered = frames
            if role in {"large", "small", "local_texture"}:
                altered, span = edited(frames, i, role)
                mask = np.any(altered != frames, axis=3)
                require(np.flatnonzero(mask.any(axis=(1, 2))).tolist() == list(range(*span)), "Authored interval differs")
            elif role == "global":
                altered = np.clip(frames.astype(np.float32) * (.91 if i % 2 else 1.15) + (11 if i % 2 else -18),
                                  0, 255).astype(np.uint8)
            elif role == "wrong":
                altered = input_pictures(paths[(i + 1) % len(paths)])
            controls.encode(submitted, altered, 28 if span else 32)
            rows.append({"source": i, "role": role, "truth": span} | analyze((earlier, submitted), mask))
        print("held", i + 1, len(paths), flush=True)
    return rows


def diagnostic(path):
    cases = ((4, "large", (13, 30)), (5, "large", (14, 32)), (5, "tone", None))
    return [{"source": i, "role": role, "truth": truth} |
            analyze((path / f"{i}-original.mp4", path / f"{i}-{role}.mp4")) for i, role, truth in cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "diagnostic", "acquire", "held"))
    parser.add_argument("roster", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--prior-controls", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    rows = manifest(args.roster)
    bindings = {Path(name).name: sha(name) for name in (__file__, candidate.__file__, baseline.__file__, controls.__file__)}
    freeze = {"schema": SCHEMA, "protocol": PLAN, "roster_sha256": sha(args.roster), "bindings": bindings,
              "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=5).splitlines()[0]}
    if args.phase == "freeze":
        args.out.mkdir(mode=0o700)
        write(args.out / "freeze.json", freeze)
        return
    require(read(args.out / "freeze.json") == freeze, "Frozen protocol changed")
    if args.phase == "diagnostic":
        require(args.prior_controls is not None, "Prior control carriers are needed")
        write(args.out / "diagnostic.json", diagnostic(args.prior_controls))
        return
    cache = args.out / "cache"
    if args.phase == "acquire":
        cache.mkdir(exist_ok=True)
        for index, row in enumerate(rows):
            copy_source(cache, index, row, args.archive)
        return
    sources = [cache / f"source-{i:02d}.flv" for i in range(len(rows))]
    require(all(path.is_file() and sha(path) == row["sha256"] for path, row in zip(sources, rows, strict=True)),
            "Acquire the checksum-verified sources before scoring")
    write(args.out / "held.json", freeze | {"rows": held(args.out, sources)})


if __name__ == "__main__":
    main()
