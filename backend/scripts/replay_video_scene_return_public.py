# SPDX-License-Identifier: Apache-2.0
"""Rebuild the released single-upload picture-return benchmark from public sources."""

import argparse
import hashlib
from pathlib import Path

import cv2

from app.detectors.video.decode import decode_frames, probe_frames
from scripts import review_video_scene_return_transfer as audit
from scripts import video_reference_robust_transfer as download
from scripts import video_scene_return_transfer as study
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
ANCESTOR = ROOT / "docs/evaluation-panels/video-scene-return-transfer-2026-11-13.json"
ROSTER = ROOT / "docs/evaluation-panels/video-scene-return-public-sources-2026-11-13.json"


def prepare_sources(folder, roster, archive):
    cache = folder / "publisher"
    cache.mkdir(exist_ok=True)
    result = []
    rows = download.manifest(ROSTER)
    require(len(roster["native"]) == len(rows), "Native source commitment count differs")
    for index, (row, native) in enumerate(zip(rows, roster["native"], strict=True)):
        path = download.copy_source(cache, index, row, archive)
        table = probe_frames(path, timeout_s=15)
        require(len(table) >= 55 and not (table.rotation or table.interpolated or table.truncated),
                "Publisher picture clock is outside the frozen source profile")
        pts_sha = hashlib.sha256(",".join(map(str, table.pts)).encode()).hexdigest()
        require(pts_sha == native["pts_sha256"] and table.time_base == native["time_base"],
                "Native timestamp table changed")
        hashes = []
        for frame in decode_frames(path, table, None, timeout_s=25):
            require(frame.pts_verified and frame.index == len(hashes), "Unverified source picture")
            hashes.append(hashlib.sha256(frame.rgb).hexdigest())
        combined = hashlib.sha256("".join(hashes).encode()).hexdigest()
        require(len(hashes) == len(table) and combined == native["native_rgb_sha256"],
                "Decoded source pictures changed")
        result.append({"file": str(path), "sha256": row["sha256"], "group": row["group"],
                       "frame_table": table.to_json(), "rgb_sha256": hashes})
    return result


def replay(folder, archive):
    roster, expected = read(ROSTER), read(ANCESTOR)
    require(roster["original_ledger_sha256"] == sha(ANCESTOR) and study.bindings() == expected["bindings"],
            "Released study or detector differs")
    require(cv2.__version__ == expected["opencv"], "Use the study OpenCV version")
    folder.mkdir(mode=0o700)
    sources = prepare_sources(folder, roster, archive)
    lock = {"sources": sources, "intervals": study.INTERVALS, "bindings": study.bindings()}
    write(folder / "freeze.json", lock)
    study.prepare(folder, lock)
    study.score(folder, lock)
    actual = read(folder / "results.json")
    require(actual["rows"] == expected["rows"], "Rebuilt media or measured result differs from the public ledger")
    audit.validate(actual)
    checked = audit.replay_sources(actual, folder / "media")
    return {"schema": "chipmunk.video-scene-return-public-replay/1", "source_count": len(sources),
            "source_download_bytes": sum(row["bytes"] for row in roster["sources"]),
            "roster_sha256": sha(ROSTER), "released_ledger_sha256": sha(ANCESTOR),
            "media_deliveries": len(actual["rows"]), "exact_ranges": sum(row["candidate"] == row["truth"]
                                                                      for row in actual["rows"] if row["truth"]),
            "independent_source_checks": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    receipt = replay(args.run, args.archive)
    if args.receipt:
        write(args.receipt, receipt)
    print(receipt)


if __name__ == "__main__":
    main()
