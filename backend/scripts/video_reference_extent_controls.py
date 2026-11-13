# SPDX-License-Identifier: Apache-2.0
"""Confirm exact larger and smaller picture extents on disjoint camera sources."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import video_reference_local as local
from scripts import video_reference_extent_method as extent
from scripts import video_reference_extent_transfer as compare
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.video-reference-extent-controls/1"
POLICY = {
    "scope": "Same 24 fps 720x480 comparison, conventional edits only. First six sorted CREMA-D selection sources with ready originals, disjoint from earlier confirmation. No score-based inclusion.",
    "edits": "A 184x104 transient tone change and a 64x40 one, alternate locations and durations. Test identical delivery, strong recompression, full-clip global tone and mismatched sources. Exact masks are computed before H.264.",
    "decision": "The extent must improve median spatial IoU by at least 0.15 on large changes, preserve 5/6 small changes at the frozen time/spatial rule, and lose at most one small location relative to the tile. At least 5/6 large locations, at most one neutral flag and zero wrong-source admissions. This confirms an observable change comparison only.",
}
ROLES = ("original", "large", "small", "compression", "tone", "wrong")
DIR = Path("backend/data/independent-evaluation/native-mouth-clips-v120")


def freeze(previous):
    prior = read(previous)
    require(prior["schema"] == "chipmunk.video-reference-extent/1" and len(prior["transfer"]) == 69,
            "Need the completed first extent comparison")
    sources = []
    confirmation = set()
    for path in sorted(DIR.glob("extract-*/output/cases.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["condition"] != "original":
                continue
            if row["study_role"] == "confirmation":
                confirmation.add(row["source_group"])
            if row["study_role"] == "selection" and row["status"] == "ready":
                sources.append({key: row[key] for key in ("asset_index", "source_group", "file", "file_sha256", "expected")})
    chosen = sorted(sources, key=lambda r: r["asset_index"])[:6]
    require(len({r["source_group"] for r in chosen} - confirmation) == 6, "Camera groups overlap")
    code = [__file__, extent.__file__, local.__file__, compare.__file__]
    return {"schema": SCHEMA, "policy": POLICY, "prior_sha256": sha(previous),
            "code": {Path(p).name: sha(p) for p in code}, "sources": chosen}


def source(row):
    path = Path(row["file"])
    require(sha(path) == row["file_sha256"], "Native source bytes differ")
    table = probe_frames(path, cap=100, timeout_s=10)
    require(len(table) >= 50 and not table.interpolated, "Source does not have enough verified pictures")
    frames = []
    for f in decode_frames(path, table, range(50), timeout_s=20):
        require(f.pts_verified and hashlib.sha256(f.rgb).hexdigest() == row["expected"]["rgb_sha256"][f.index],
                "Original decoded picture differs")
        frames.append(cv2.resize(f.rgb, (720, 480), interpolation=cv2.INTER_NEAREST))
    require(len(frames) == 50, "Missing source frames")
    return np.array(frames)


def encode(path, frames, quality):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
           "-video_size", "720x480", "-framerate", "24", "-i", "pipe:0", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-bf", "0", "-crf", str(quality), "-threads", "1", str(path)]
    subprocess.run(cmd, check=True, input=frames.tobytes(), timeout=35)


def run(folder, lock):
    sources = lock["sources"]
    result = []
    for i, row in enumerate(sources):
        frames = source(row)
        earlier = folder / f"{i}-original.mp4"
        encode(earlier, frames, 23)
        for role in ROLES:
            truth = None
            delivered = frames
            if role in ("large", "small"):
                start, end = 12 + i % 3, 28 + 2 * (i % 3)
                w, h = (184, 104) if role == "large" else (64, 40)
                x, y = 93 + 77 * i, 95 + 24 * (i % 4)
                delivered = frames.copy()
                changed = delivered[start:end, y:y + h, x:x + w]
                changed[:] = (changed.astype(np.float32) * .63 + 88).astype(np.uint8)
                truth = [start, end]
            if role == "tone":
                delivered = np.clip(frames.astype(np.float32) * .91 + 11, 0, 255).astype(np.uint8)
            if role == "wrong":
                delivered = source(sources[(i + 1) % len(sources)])
            later = earlier
            if role != "original":
                later = folder / f"{i}-{role}.mp4"
                encode(later, delivered, 32 if role in {"compression", "tone", "wrong"} else 28)
            found = compare.compare((earlier, later)) | {"source": i, "role": role, "truth": truth}
            if truth and found["status"] == "local_difference":
                idx = found["anchors"]["during"]["index"]
                mask = np.any(delivered[idx] != frames[idx], axis=2)
                found["overlap"] = {"changed": int(mask.sum())}
                for arm in ("tile", "extent"):
                    x, y, w, h = (found[arm][key] for key in ("x", "y", "w", "h"))
                    found["overlap"][arm] = int(mask[y:y+h, x:x+w].sum())
            result.append(found)
        print(i + 1, len(sources), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    parser.add_argument("previous", type=Path)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    lock = freeze(args.previous)
    seal = args.folder / "freeze.json"
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700)
        write(seal, lock)
    else:
        require(read(seal) == lock, "The confirmation inputs or method changed")
        write(args.folder / "result.json", {"schema": SCHEMA, "policy": POLICY, "freeze_sha256": sha(seal),
                                           "source_commitments": [r["file_sha256"] for r in lock["sources"]],
                                           "rows": run(args.folder, lock)})


if __name__ == "__main__":
    main()
