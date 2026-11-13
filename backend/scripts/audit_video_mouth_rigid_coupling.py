# SPDX-License-Identifier: Apache-2.0
"""Reopen emitted native frame evidence and verify the distinct CREMA-D clock scope."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from app.detectors.video.decode import decode_frames, probe_frames
from scripts.video_common_sources_v89 import read, require, sha, write
from scripts.video_mouth_jaw_transfer import ARCHIVE, native_groups


def payload_sha(data):
    return hashlib.sha256(data).hexdigest()


def audit(run):
    lock, held, points = (read(run / f"{k}.json") for k in ("freeze", "held", "calibration"))
    require(held["freeze_sha256"] == sha(run / "freeze.json"), "Held freeze differs")
    by_id = {tuple(r["key"]): r for r in lock["cases"]}
    n_flags = n_pictures = 0
    for row in held["rows"]:
        found = [
            p
            for arm, result in row["results"].items()
            if (p := result["proposal"]) and p["score"] > points["points"][arm]
        ]
        if not found:
            continue
        n_flags += len(found)
        case = by_id[tuple(row["key"])]
        file = Path(case["path"])
        require(sha(file) == case["file_sha256"], "Native evidence source differs")
        table = probe_frames(file, timeout_s=12)
        require(
            list(table.pts) == row["native"]["picture_pts"] and table.time_base == row["native"]["time_base"],
            "Observed presentation table differs",
        )
        selected = {a["index"]: a for record in found for a in record["anchors"]}
        actual = set()
        for frame in decode_frames(file, table, sorted(selected), timeout_s=15):
            wanted = selected[frame.index]
            x0, y0, x1, y1 = wanted["box"]
            require(
                frame.pts_verified
                and frame.pts == wanted["pts"]
                and payload_sha(frame.rgb.tobytes()) == wanted["rgb_sha256"]
                and payload_sha(frame.rgb[y0:y1, x0:x1].tobytes()) == wanted["crop_sha256"],
                "Evidence frame or exact native crop differs",
            )
            actual.add(frame.index)
        require(actual == set(selected), "Native evidence frame inventory differs")
        n_pictures += len(actual)
    scope = Counter()
    groups = native_groups(ARCHIVE)
    for variants in groups.values():
        for item in variants:
            stamps = item["expected"]["picture_pts_ms"]
            require(
                len(stamps) >= 2 and (stamps[1] - stamps[0]) < 40,
                "Additional source may enter original frame clock scope",
            )
            scope["outside_frozen_clock"] += 1
    return {
        "held_sha256": sha(run / "held.json"),
        "verified_flag_ranges": n_flags,
        "verified_native_flag_crops": n_pictures,
        "different_clock_controls": dict(scope),
        "different_clock_source_groups": len(groups),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    a = p.parse_args()
    output = audit(a.run)
    write(a.run / "source-audit.json", output)
    print(output)


if __name__ == "__main__":
    main()
