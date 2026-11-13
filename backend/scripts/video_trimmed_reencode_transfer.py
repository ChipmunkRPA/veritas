# SPDX-License-Identifier: Apache-2.0
"""Compare earlier-copy localization after H.264 trimming and re-encoding."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import probe_frames
from app.evidence import video_reference as reference
from app.evidence import video_reference_local as local
from app.evidence import video_reference_photometric as photo
from app.evidence import video_reference_trimmed as trimmed
from scripts import video_reference_extent_controls as render
from scripts import video_reference_robust_transfer as source
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
FIRST = PANELS / "video-paired-local-area-sources-2026-11-13.json"
SECOND = PANELS / "video-scene-return-public-sources-2026-11-13.json"
SCHEMA = "chipmunk.video-trimmed-reencode-transfer/1"
ROLES = ("large", "small", "local_texture", "reencoded", "global_grade", "wrong_source", "whole_frame_shift")
PLAN = {
    "development": "First four previously consumed paired-local-area recordings. Determine motion admission on these only.",
    "held": "Four remaining paired-local-area recordings and seven nonoverlapping public scene-return recordings. Previously exposed sources, unseen trim deliveries. One planned pairing per role per source.",
    "render": "720 by 480 H.264 at 24 fps. Source is 52 pictures. Shorten leading and trailing margins by 3 to 6 pictures each. Encode submitted picture separately. Three edits remain between the flanks.",
    "alignment": "Mean signed two- and four-picture motion on each flank. The same offset must have adequate support on both flanks and a lead over every other offset. Points were chosen using only the first four sources. Reuse frozen robust photometric localization without tuning.",
    "gate": "Recover at least seven large, seven small and five texture edits by time IoU >= 0.50 and pixel IoU >= 0.10. Correct offset on every admitted related case. No unrelated or shifted-camera admission with a local flag and no unrelated clip admitted. At most two flags on 22 matched ordinary re-encode and grade controls. At least eight re-encode controls must be assessed.",
}


def roster():
    first = source.manifest(FIRST)
    second = [row for row in source.manifest(SECOND) if row["group"] not in {item["group"] for item in first}]
    require(len(second) == 7, "Unexpected public roster intersection")
    return first[:4], first[4:] + second


def configure():
    dev, held = roster()
    return {"schema": SCHEMA, "plan": PLAN, "sources": {"development": dev, "held": held},
            "source_rosters": [sha(FIRST), sha(SECOND)], "opencv": cv2.__version__,
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=5).splitlines()[0],
            "bindings": {Path(path).name: sha(path) for path in (
                __file__, trimmed.__file__, photo.__file__, local.__file__, render.__file__, source.__file__,
            )}}


def acquire(run, split, rows, caches):
    folder = run / "inputs" / split
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, row in enumerate(rows):
        local_file = next((p for cache in caches for p in sorted(cache.glob("source-*.flv"))
                           if p.stat().st_size == row["bytes"] and sha(p) == row["sha256"]), None)
        if local_file is not None:
            paths.append(local_file)
        else:
            paths.append(source.copy_source(folder, index, row, None))
    return paths


def edits(pictures, index, role, wrong):
    if role == "wrong_source":
        return wrong, None, None
    if role == "global_grade":
        return np.clip(pictures.astype(np.float32) * .84 + 19, 0, 255).astype(np.uint8), None, None
    if role == "whole_frame_shift":
        return np.roll(pictures, 24, axis=2), None, None
    if role == "reencoded":
        return pictures, None, None
    start, stop = 15 + index % 3, 31 + index % 4
    x, y = 90 + index * 67 % 360, 88 + index % 4 * 51
    w, h = {"large": (168, 112), "small": (70, 46), "local_texture": (120, 96)}[role]
    edited = pictures.copy()
    block = pictures[start:stop, y:y + h, x:x + w]
    if role == "local_texture":
        changed = pictures[start:stop, y:y + h, x + 19:x + w + 19]
    else:
        changed = np.clip(block.astype(np.float32) * .66 + 77, 0, 255).astype(np.uint8)
    edited[start:stop, y:y + h, x:x + w] = changed
    mask = np.any(edited != pictures, axis=3)
    require(np.flatnonzero(mask.any(axis=(1, 2))).tolist() == list(range(start, stop)), "Authored span differs")
    return edited, [start, stop], mask


def score(earlier, later, mask, lead):
    paths = (earlier, later)
    tables = tuple(probe_frames(path, cap=100, timeout_s=15) for path in paths)
    require(trimmed.eligible(paths, tables), "Encoded carrier is not eligible")
    votes = trimmed.profile(*(trimmed.read_pictures(path, table) for path, table in zip(paths, tables, strict=True)))
    alignment = trimmed.decide(votes)
    result = {"files_sha256": [sha(path) for path in paths], "source_frames": [len(table) for table in tables],
              "legacy_offset": reference._trim_offset(paths, tables), "motion_votes": votes,
              "alignment": alignment}
    if not alignment:
        return result
    offset = alignment["earlier_start"]
    rects = local.rectangles(tables[0].height, tables[0].width)
    measurements, hashes = [], []
    for first, second in reference._trimmed_pairs(paths, tables, offset):
        measurements.append(photo.measurements(first.rgb, second.rgb, rects))
        hashes.append([hashlib.sha256(frame.rgb).hexdigest() for frame in (first, second)])
    signals, backgrounds, grids = zip(*measurements, strict=True)
    status, selection = photo.assess(signals, backgrounds, rects)
    result["status"] = status
    if selection is not None:
        crop = photo.expanded_region(np.array(grids), selection, constrained=False)
        result["selection"] = {"frames": selection["frames"], "crop": crop, "score": selection["score"],
                               "anchors": {role: {"frames": [index + offset, index],
                                                  "pts": [tables[0].pts[index + offset], tables[1].pts[index]],
                                                  "rgb_sha256": hashes[index]}
                                           for role, index in selection["samples"].items()}}
        if mask is not None:
            center = mask[selection["samples"]["during"] + lead]
            x, y, w, h = (crop[key] for key in ("x", "y", "w", "h"))
            result["pixels"] = {"truth": int(center.sum()), "intersection": int(center[y:y + h, x:x + w].sum())}
    result["time_bases"] = [table.time_base for table in tables]
    return result


def evaluate(run, split, rows, caches):
    inputs = acquire(run, split, rows, caches)
    out = run / split
    out.mkdir(mode=0o700)
    groups = [source.input_pictures(path) for path in inputs]
    results = []
    for index, (row, pictures) in enumerate(zip(rows, groups, strict=True)):
        lead, tail = 3 + index % 4, 3 + (index + 1) % 4
        earlier = out / f"{index:02d}-earlier.mp4"
        render.encode(earlier, pictures, 22 + index % 3)
        for role in ROLES:
            changed, truth, mask = edits(pictures, index, role, groups[(index + 1) % len(groups)])
            later = out / f"{index:02d}-{role}.mp4"
            render.encode(later, changed[lead:-tail], (32 + index % 2) if role == "reencoded" else 27 + index % 3)
            results.append({"source": row["group"], "index": index, "role": role, "trim": [lead, tail],
                            "truth": [n - lead for n in truth] if truth else None} | score(earlier, later, mask, lead))
        print(split, index + 1, len(rows), flush=True)
    return results


def counts(results):
    output = {role: Counter() for role in ROLES}
    for row in results:
        cell = output[row["role"]]
        cell["files"] += 1
        cell["legacy"] += row["legacy_offset"] is not None
        align = row["alignment"]
        if not align:
            continue
        cell["aligned"] += 1
        cell["correct_offset"] += align["earlier_start"] == row["trim"][0]
        cell["assessed"] += row["status"] in {"no_local_range", "local_difference"}
        if "selection" not in row:
            continue
        cell["flags"] += 1
        if row["truth"]:
            a, b = row["truth"]
            x, y = row["selection"]["frames"]
            spatial = row["pixels"]
            crop = row["selection"]["crop"]
            cell["localized"] += (2 * (min(b, y) - max(a, x)) >= max(b, y) - min(a, x)
                                  and 10 * spatial["intersection"] >= spatial["truth"] + crop["w"] * crop["h"] - spatial["intersection"])
    return {role: dict(cell) for role, cell in output.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--cache", type=Path, action="append", default=[])
    args = parser.parse_args()
    cv2.setNumThreads(1)
    frozen = configure()
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(args.run / "freeze.json", frozen)
        return
    require(read(args.run / "freeze.json") == frozen, "Frozen code or inputs differ")
    dev = args.run / "development.json"
    if args.phase == "held":
        require(dev.is_file(), "Complete the locked development check first")
    rows = evaluate(args.run, args.phase, frozen["sources"][args.phase], args.cache)
    receipt = {"schema": SCHEMA, "split": args.phase, "freeze_sha256": sha(args.run / "freeze.json"),
               "development_sha256": sha(dev) if args.phase == "held" else None,
               "counts": counts(rows), "rows": rows}
    write(args.run / (args.phase + ".json"), receipt)
    print(receipt["counts"])


if __name__ == "__main__":
    main()
