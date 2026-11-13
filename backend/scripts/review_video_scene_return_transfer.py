# SPDX-License-Identifier: Apache-2.0
"""Recompute the standalone picture return comparison from published scalars."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-scene-return-transfer/1"
ROLES = ("unchanged", "layout_cut", "speaker_and_layout_cut", "speaker_only", "camera_shift", "grade", "full_camera")
INTERVALS = ((33, 40), (45, 56), (60, 77), (70, 93))
DIGEST = re.compile("[0-9a-f]{64}")


def require(check, reason):
    if not check:
        raise ValueError(reason)


def validate(document):
    rows = document["rows"]
    require(document["schema"] == SCHEMA and len(rows) == 56
            and DIGEST.fullmatch(document["freeze_sha256"]) and DIGEST.fullmatch(document["media_sha256"])
            and all(DIGEST.fullmatch(value) for value in document["bindings"].values()), "Invalid frozen roster")
    observed = set()
    totals = {role: dict(files=0, exact=0, misplaced=0, no_range=0, baseline_exact=0, baseline_any=0) for role in ROLES}
    for row in rows:
        pair, direction, role = row["pair"], row["direction"], row["role"]
        key = pair, direction, role
        require(pair in range(4) and direction in range(2) and role in ROLES and key not in observed, "Duplicate or invalid case")
        observed.add(key)
        truth, found, tiles, anchors = (row[name] for name in ("truth", "candidate", "tiles", "anchors"))
        require(truth == (list(INTERVALS[pair]) if "cut" in role else None), "Reference differs from the predeclared interval")
        edges = row["baseline_edges"]
        require(edges == sorted(set(edges)) and all(1 <= edge < 120 for edge in edges), "Invalid baseline boundaries")
        tb = Fraction(row["time_base"])
        require(tb > 0 and (Fraction(1, 30) / tb).denominator == 1
                and DIGEST.fullmatch(row["pts_sha256"]) and DIGEST.fullmatch(row["sha256"]), "Invalid source commitments")
        if found is None:
            require(not anchors and not tiles, "Unreported case contains evidence")
        else:
            start, stop = found
            require(4 <= start < stop < 117 and 6 <= stop - start <= 24, "Invalid candidate interval")
            require(tiles == sorted(set(tiles)) and len(tiles) >= 24 and all(0 <= tile < 60 for tile in tiles), "Invalid supporting tile grid")
            counts = [0, 0, 0, 0]
            for tile in tiles:
                counts[2 * (tile // 10 >= 3) + (tile % 10 >= 5)] += 1
            require(min(counts) >= 4, "Missing supporting quadrant")
            require([frame["index"] for frame in anchors] == [start - 1, (start + stop) // 2, stop], "Invalid anchor positions")
            require(all(DIGEST.fullmatch(frame["sha256"]) for frame in anchors), "Invalid decoded frame hash")
            origins = {frame["pts"] * tb - Fraction(frame["index"], 30) for frame in anchors}
            require(len(origins) == 1, "Inconsistent exact timestamp anchors")
        entry = totals[role]
        entry["files"] += 1
        entry["exact"] += found is not None and found == truth
        entry["misplaced"] += found is not None and found != truth
        entry["no_range"] += found is None
        entry["baseline_exact"] += truth is not None and set(truth).issubset(edges)
        entry["baseline_any"] += bool(edges)
    controls = [row for row in rows if row["truth"] is None]
    new_flags = sum(row["candidate"] is not None and not row["baseline_edges"] for row in controls)
    qualifies = totals["layout_cut"]["exact"] >= 6 and new_flags <= 1
    return {"totals": totals, "control_files": len(controls), "additional_control_flags": new_flags, "qualifies": qualifies}


def replay_sources(document, folder):
    anchors = clocks = 0
    for row in document["rows"]:
        name = f"{row['pair']}-{row['direction']}-{row['role']}.mp4"
        path = folder / name
        with path.open("rb") as handle:
            require(hashlib.file_digest(handle, "sha256").hexdigest() == row["sha256"], "Delivery hash differs")
        raw = subprocess.check_output(
            ["ffmpeg", "-nostdin", "-v", "error", "-copyts", "-i", str(path), "-map", "0:v:0", "-an",
             "-fps_mode", "passthrough", "-enc_time_base", "demux", "-pix_fmt", "rgb24",
             "-f", "framehash", "-hash", "sha256", "-"], text=True, timeout=30,
        ).splitlines()
        require(f"#tb 0: {row['time_base']}" in raw, "Framehash stream clock differs")
        frames = [line.replace(" ", "").split(",") for line in raw if not line.startswith("#")]
        ticks = [int(frame[2]) for frame in frames]
        require(len(ticks) == 120 and hashlib.sha256(json.dumps(ticks).encode()).hexdigest() == row["pts_sha256"],
                "Source picture timestamps differ")
        for anchor in row["anchors"]:
            frame = frames[anchor["index"]]
            require(int(frame[2]) == anchor["pts"] and frame[5] == anchor["sha256"], "Original decoded picture differs")
            anchors += 1
        clocks += len(ticks)
    return {"native_frames_verified": anchors, "native_timestamps_verified": clocks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--media", type=Path)
    args = parser.parse_args()
    raw = args.ledger.read_bytes()
    document = json.loads(raw)
    output = {"ledger_sha256": hashlib.sha256(raw).hexdigest(), **validate(document)}
    if args.media:
        output.update(replay_sources(document, args.media))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
