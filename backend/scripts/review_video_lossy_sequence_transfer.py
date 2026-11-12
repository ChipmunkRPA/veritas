"""Replay the lossy video comparison and verify the published aggregate."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from itertools import groupby, pairwise
from pathlib import Path

SCHEMA = "chipmunk.video-lossy-sequence-transfer-review/1"
POINT = (1.6, .4, .65)
ARMS = ("exact", "picture", "motion")
CODECS = ("h264", "vp9")
CONTROLS = ("original", "cut", "hold", "reverse", "spacing")


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def agreement(pair, target):
    if pair is None or target is None:
        return False
    for offset in (0, 2):
        first, last = pair[offset:offset + 2]
        start, end = target[offset:offset + 2]
        if max(0, min(last, end) - max(first, start) + 1) * 2 < max(last, end) - min(first, start) + 1:
            return False
    return True


def counts(rows):
    result = {}
    for codec in CODECS:
        cases = [row for row in rows if row["codec"] == codec]
        copies = [row for row in cases if row["kind"] == "copy"]
        controls = [row for row in cases if row["kind"] in CONTROLS]
        hits = {arm: {row["group"] for row in copies if agreement(row[arm], row["truth"])} for arm in ARMS}
        flags = {arm: Counter(row["kind"] for row in controls if row[arm] is not None) for arm in ARMS}
        added = [row["kind"] for row in controls if row["motion"] is not None and row["exact"] is None]
        result[codec] = {
            "positives": len(copies),
            "controls": len(controls),
            "arms": {arm: {"useful": len(hits[arm]), "groups": len(hits[arm]), "controls_flagged": flags[arm].total(),
                           "flags_by_kind": {kind: flags[arm][kind] for kind in CONTROLS}} for arm in ARMS},
            "added_useful_groups": len(hits["motion"] - hits["exact"]),
            "lost_exact": len(hits["exact"] - hits["motion"]),
            "added_controls": len(added),
            "added_clock_or_hold_flags": sum(kind in ("hold", "spacing") for kind in added),
        }
    return result


def verify_public(data):
    if data["schema"] != SCHEMA or tuple(data["point"]) != POINT:
        raise ValueError("Unexpected review format or threshold commitment")
    for role, number in (("development", 6), ("held", 8)):
        panel = data[role]
        rows = panel["rows"]
        groups = {row["group"] for row in rows}
        keys = {(row["group"], row["codec"], row["kind"]) for row in rows}
        expected = {(group, codec, kind) for group in groups for codec in CODECS for kind in ("copy", *CONTROLS)}
        if len(groups) != number or len(rows) != len(expected) or keys != expected:
            raise ValueError(f"Incomplete {role} matched panel")
        for row in rows:
            if (row["truth"] is not None) != (row["kind"] == "copy"):
                raise ValueError("Unexpected authored truth")
            for key in (*ARMS, "truth"):
                pair = row[key]
                if pair is not None:
                    a, b, c, d = pair
                    if not (0 <= a < b < c < d < 180 and b - a == d - c):
                        raise ValueError("Invalid pair bounds")
        summary = counts(rows)
        gate = all(row["added_useful_groups"] >= 4 and row["lost_exact"] == 0 and row["added_controls"] <= 1
                   and row["added_clock_or_hold_flags"] == 0 for row in summary.values())
        if summary != panel["summary"] or gate != panel["passes"]:
            raise ValueError(f"{role} counts or gate differ from case rows")
    print("Arithmetic verified across 72 development and 96 held deliveries.")


def ffmpeg(path, arguments, *, text=False):
    return subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-threads", "1",
                                    "-copyts", "-i", f"file:{path}", "-map", "0:v:0", "-an", "-sn", "-dn",
                                    "-filter_threads", "1", "-vf", "format=rgb24", "-fps_mode", "passthrough",
                                    *arguments, "pipe:1"], timeout=20, text=text)


def native_picture_data(path, row):
    import numpy as np

    external = ffmpeg(path, ["-enc_time_base", row["time_base"], "-c:v", "rawvideo", "-f", "framehash", "-hash", "sha256"],
                      text=True).splitlines()
    if f"#tb 0: {row['time_base']}" not in external:
        raise ValueError("Independent muxer returned another time base")
    timeline = [line.split(",") for line in external if not line.startswith("#")]
    pts = [int(fields[2]) for fields in timeline]
    hashes = [fields[5].strip() for fields in timeline]
    if pts != row["pts"] or hashes != row["rgb_sha256"] or any(int(fields[4]) != 320 * 180 * 3 for fields in timeline):
        raise ValueError("Independent muxer returned different native pictures or clocks")
    expected = [50 * frame for frame in range(180)]
    if row["kind"] == "spacing":
        _, b, n = (17 + row["group"], 114 + 2 * row["group"], (19, 23, 29)[row["group"] % 3])
        for i in range(b, b + n):
            expected[i] += 10 * (i % 2)
    if Fraction(row["time_base"]) != Fraction(1, 1000) or expected != pts:
        raise ValueError("Held delivery native clock differs from the authored recipe")
    raw = ffmpeg(path, ["-c:v", "rawvideo", "-f", "rawvideo"])
    raster = np.frombuffer(raw, dtype="u1").reshape(180, 30, 6, 40, 8, 3)
    sums = raster.sum(axis=(2, 4)).astype(np.float64)
    luminance = (77 * sums[..., 0] + 150 * sums[..., 1] + 29 * sums[..., 2]) / (48 * 256)
    return pts, hashes, luminance.reshape(180, -1)


def linked_supports(links, support):
    groups = []
    run = []
    for i, (linked, moving) in enumerate(zip(links, support, strict=True)):
        if not linked:
            if run:
                groups.append(run)
            run = []
        elif moving:
            if run and i - run[-1] > 3:
                groups.append(run)
                run = []
            run.append(i)
    if run:
        groups.append(run)
    return groups


def ranked_pairs(pts, lag, runs):
    for run in runs:
        right = 0
        for left, start in enumerate(run):
            while right < len(run) and run[right] < start + lag - 1:
                right += 1
            if right - left >= 8:
                end = run[right - 1] + 1
                duration = pts[end] - pts[start]
                if duration >= 500 and 2 * (right - left) >= end - start:
                    yield duration, -start, -start - lag, end


def approximate(pts, frames):
    import numpy as np

    distance, floor, direction = POINT
    clocks = np.asarray([b - a for a, b in pairwise(pts)])
    delta = frames[1:] - frames[:-1]
    delta -= delta.mean(axis=1)[:, None]
    energy = np.sqrt(np.square(delta).mean(axis=1))
    winners = dict.fromkeys(ARMS[1:])
    for lag in range(9, len(pts) - 8):
        close = np.square(frames[lag:] - frames[:-lag]).mean(axis=1) <= distance * distance
        links = close[:-1] & close[1:] & (clocks[:-lag] == clocks[lag:]) & (clocks[:-lag] > 0) & (clocks[:-lag] <= 125)
        if int(links.sum()) < 8:
            continue
        low, high = np.minimum(energy[:-lag], energy[lag:]), np.maximum(energy[:-lag], energy[lag:])
        angle = (delta[:-lag] * delta[lag:]).mean(axis=1) >= direction * energy[:-lag] * energy[lag:]
        eligible = (low >= floor) & (low >= .55 * high) & angle
        for arm, support in (("picture", links), ("motion", eligible)):
            candidates = ranked_pairs(pts, lag, linked_supports(links, support))
            winner = max(candidates, default=None)
            if winner is not None and (winners[arm] is None or winner > winners[arm]):
                winners[arm] = winner
    pairs = {}
    for arm, candidate in winners.items():
        if candidate is None:
            pairs[arm] = None
        else:
            _, first, other, end = candidate
            pairs[arm] = [-first, end, -other, end + first - other]
    return pairs


def replay(run, artifact):
    from scripts.video_exact_sequence_transfer import oracle

    data = {"schema": SCHEMA, "point": list(POINT)}
    dev, held = (read(run / f"{name}.json") for name in ("development", "held"))
    if (tuple(dev["point"]) != POINT or held["point"] != dev["point"]
            or held["runner_sha256"] != dev["runner_sha256"]
            or held["runner_sha256"] != sha_file(Path(__file__).with_name("video_lossy_sequence_transfer.py"))):
        raise ValueError("Development and held freeze differ")
    data["runner_sha256"] = held["runner_sha256"]
    known = [{source["original_source_sha256"] for source in panel["sources"]} for panel in (dev, held)]
    if list(map(len, known)) != [6, 8] or known[0] & known[1]:
        raise ValueError("Held source coverage overlaps the development roster")
    frame_count = 0
    for group, rows in groupby(held["rows"], key=lambda row: row["group"]):
        for row in rows:
            path = run / "media" / f"held-{group:02d}-{row['codec']}-{row['kind']}.mkv"
            if sha_file(path) != row["encoded_sha256"]:
                raise ValueError("Held delivery bytes changed")
            pts, hashes, frames = native_picture_data(path, row)
            frame_count += len(pts)
            exact = oracle(pts, hashes, Fraction(1, 1000))
            if exact != row["exact"] or any(row[arm] != pair for arm, pair in approximate(pts, frames).items()):
                raise ValueError(f"Different independently computed outcome for {group} {row['codec']} {row['kind']}")
        print(f"Verified group {group}, {frame_count} native pictures.", flush=True)
    data["verification"] = {"held_pictures": frame_count, "held_deliveries": len(held["rows"]), "source_digest_overlap": 0,
                            "held_frame_clock_and_outcome_replay": True}
    for panel in (dev, held):
        data[panel["role"]] = {"summary": panel["summary"], "passes": panel["passes"],
                               "rows": [{key: row[key] for key in ("group", "codec", "kind", "truth", *ARMS)} for row in panel["rows"]]}
    verify_public(data)
    with artifact.open("x") as file:
        file.write(json.dumps(data, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--replay", type=Path, help="Private study directory for independent native playback")
    args = parser.parse_args()
    if args.replay is None:
        verify_public(read(args.artifact))
    else:
        replay(args.replay, args.artifact)
