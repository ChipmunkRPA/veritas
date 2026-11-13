# SPDX-License-Identifier: Apache-2.0
"""Review scalar decisions and independently reopen H.264 picture evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-mouth-codec-prediction/1"
ARMS = ("inter_coverage_loss", "vector_dispersion")
SPLITS = {"development": 56, "held": 92}
FLOOR = 0.04


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def overlap(proposal, truth):
    a, b = map(Fraction, proposal)
    c, d = map(Fraction, truth)
    return max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))


def points_for(rows):
    return {arm: max(FLOOR, (sorted((r["arms"][arm]["score"] or 0 for r in rows), reverse=True) + [0] * 3)[2])
            for arm in ARMS}


def totals(rows, arm, point, split):
    counts, roles, groups = Counter(), {}, set()
    positive, ordinary = [], []
    for row in rows:
        result = row["arms"][arm]
        score, seconds, measured = result["score"], result["seconds"], result["measured"]
        require((score is not None) == bool(seconds) and (score is None or math.isfinite(score) and score > 0 and measured),
                "Proposal validity differs")
        neural = bool(row["truth"])
        flag = score is not None and score > point
        overlap_half = bool(seconds and neural and overlap(seconds, row["truth"]) >= Fraction(1, 2))
        proof = result.get("proof")
        require(bool(proof) == (split == "held" and (flag or overlap_half)), "Selected proof coverage differs")
        if proof:
            start, end = proof["frames"]
            anchors = proof["anchors"]
            clock = Fraction(proof["time_base"])
            require([a["frame"] for a in anchors] == [start - 1, (start + end) // 2, end]
                    and anchors[0]["pts"] * clock < Fraction(seconds[0]) <= anchors[1]["pts"] * clock
                    < Fraction(seconds[1]) == anchors[-1]["pts"] * clock
                    and all(len(a["crop_sha256"]) == len(a["rgb_sha256"]) == 64 for a in anchors),
                    "Source-clock anchors differ")
        marks = {"files": 1, "measured": measured, "flagged": flag,
                 "located": flag and overlap_half, "overlap_before_point": overlap_half,
                 "pictures": row["native_frames"], "supported_pictures": row["frame_support"][arm]}
        roles.setdefault(f"{row['family']}:{row['role']}", Counter()).update(marks)
        key = "neural" if neural else "ordinary"
        counts.update({f"{key}_{k}": v for k, v in marks.items()})
        if row["aligned"]:
            counts.update({f"strict_{row['family']}_{k}": v for k, v in marks.items()})
            if row["family"] == "lombard" and flag and overlap_half:
                groups.add(row["group"])
            if measured:
                positive.append(score or 0)
        if not neural and measured:
            ordinary.append(score or 0)
        if flag and not neural and row["role"] in {"original", "reencode", "hard_cut"}:
            counts["essential_ordinary_flags"] += 1
        if flag and neural and not overlap_half:
            counts["misplaced_neural_flags"] += 1
    comparisons = len(positive) * len(ordinary)
    twice_wins = sum(2 * (p > o) + (p == o) for p in positive for o in ordinary)
    result = {"counts": dict(counts), "roles": {k: dict(v) for k, v in roles.items()},
              "strict_liveportrait_located_groups": len(groups),
              "strict_auc_among_measured": str(Fraction(twice_wins, 2 * comparisons)) if comparisons else None}
    if split == "held":
        result["gate"] = {
            "liveportrait": counts["strict_lombard_located"] >= 4 and len(groups) >= 3,
            "musetalk": counts["strict_musetalk_located"] == counts["strict_musetalk_files"] == 1,
            "ordinary": counts["ordinary_flagged"] <= 2 and not counts["essential_ordinary_flags"],
            "coverage": all(counts[f"{k}_files"] and counts[f"{k}_measured"] * 10 >= counts[f"{k}_files"] * 7
                            for k in ("strict_lombard", "strict_musetalk", "ordinary")),
        }
    return result


def probe(path):
    result = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,time_base,pix_fmt,codec_name:frame=best_effort_timestamp,pict_type", "-of", "json", str(path),
    ], timeout=20))
    require(len(result["streams"]) == 1, "Expected one selected video stream")
    stream = result["streams"][0]
    return stream, [(int(f["best_effort_timestamp"]), f["pict_type"]) for f in result["frames"]]


def frames_from(path, clock, stream, wanted):
    w, h = stream["width"], stream["height"]
    frame_size = w * h * 3
    args = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-protocol_whitelist", "file",
            "-noautorotate", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "rgb24",
            "-f", "rawvideo", "-"]
    with subprocess.Popen(args, stdout=subprocess.PIPE) as worker:
        try:
            for i, (pts, _) in enumerate(clock):
                rgb = worker.stdout.read(frame_size)
                require(len(rgb) == frame_size, "Incomplete native readback")
                if i not in wanted:
                    continue
                a = wanted[i]
                x0, y0, x1, y1 = a["crop_xyxy"]
                require(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, "Source crop lies outside its picture")
                crop = b"".join(rgb[(y * w + x0) * 3:(y * w + x1) * 3] for y in range(y0, y1))
                require(pts == a["pts"] and hashlib.sha256(rgb).hexdigest() == a["rgb_sha256"]
                        and hashlib.sha256(crop).hexdigest() == a["crop_sha256"], "Source frame, time, or crop differs")
            require(not worker.stdout.read(1) and worker.wait(timeout=20) == 0, "Unexpected additional picture data")
        finally:
            if worker.poll() is None:
                worker.kill()


def assemble(work):
    lock_file, carriers_file = work / "freeze.json", work / "carriers.json"
    lock, carrier = read(lock_file), read(carriers_file)
    require(carrier["freeze_sha256"] == sha(lock_file), "Carrier freeze differs")
    point_file = read(work / "points.json")
    require(point_file["development_sha256"] == sha(work / "development.json"), "Frozen point differs")
    receipt = {"schema": SCHEMA, "freeze_sha256": sha(lock_file), "carriers_sha256": sha(carriers_file),
               "protocol": lock["protocol"], "source_rosters": lock["source_rosters"],
               "points": point_file["points"], "splits": {}}
    groups = {}
    for split, count in SPLITS.items():
        score_path = work / f"{split}.json"
        observed = read(score_path)
        require(observed["carriers_sha256"] == receipt["carriers_sha256"] and len(observed["rows"]) == count,
                "Scoring scope differs")
        pending = {(r["family"], r["id"], r["role"]): r for r in carrier["rows"] if r["split"] == split}
        rows, audit = [], Counter()
        for row in observed["rows"]:
            frozen = pending.pop((row["family"], row["id"], row["role"]))
            require(all(row[k] == v for k, v in frozen.items()), "Frozen case differs")
            group = ("fresh" if row["family"] == "derived_ordinary" else row["family"], row["group"])
            public = {k: row[k] for k in ("family", "role", "truth", "types", "frame_support", "native_frames")}
            public.update(group=groups.setdefault(group, len(groups) + 1), aligned=row.get("aligned", False), arms={})
            path, source = Path(row["carrier"]), Path(row["original"])
            require(sha(path) == row["carrier_sha256"] and sha(source) == row["original_sha256"], "Media bytes differ")
            stream, clock = probe(path)
            original, original_clock = probe(source)
            tick, original_tick = map(Fraction, (stream["time_base"], original["time_base"]))
            require([p * tick for p, _ in clock] == [p * original_tick for p, _ in original_clock]
                    and stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
                    and len(clock) == row["native_frames"] == row["verified_frames"]
                    and dict(Counter(t for _, t in clock)) == row["types"], "Carrier clock or decoded profile differs")
            audit["picture_clocks"] += len(clock)
            wanted = {}
            for arm in ARMS:
                answer = row["arms"][arm]
                p = answer["proposal"]
                public["arms"][arm] = {"score": answer["score"], "measured": answer["measured"],
                                       "seconds": p["seconds"] if p else None}
                if split != "held" or p is None or (p["score"] <= receipt["points"][arm]
                        and not (row["truth"] and overlap(p["seconds"], row["truth"]) >= Fraction(1, 2))):
                    continue
                require([str(clock[i][0] * tick) for i in p["frames"]] == p["seconds"],
                        "Selected range differs from carrier")
                wanted.update({a["frame"]: a for a in p["anchors"]})
                public["arms"][arm]["proof"] = {k: p[k] for k in ("frames", "time_base", "anchors")}
                audit["proposals"] += 1
            if wanted:
                frames_from(path, clock, stream, wanted)
                audit.update({"files": 1, "frame_crop_pairs": len(wanted)})
            rows.append(public)
        require(not pending, "Scoring omitted source cases")
        receipt["splits"][split] = {"rows": rows, "sha256": sha(score_path), "audit": dict(audit),
                                    "totals": {arm: totals(rows, arm, receipt["points"][arm], split) for arm in ARMS}}
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--assemble", type=Path)
    a = parser.parse_args()
    if a.assemble:
        result = assemble(a.assemble)
        with a.receipt.open("x") as handle:
            json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    receipt = read(a.receipt)
    require(receipt["schema"] == SCHEMA and set(receipt["splits"]) == set(SPLITS), "Published scope differs")
    require(points_for(receipt["splits"]["development"]["rows"]) == receipt["points"], "Published points differ")
    for split, count in SPLITS.items():
        block = receipt["splits"][split]
        rows = block["rows"]
        replayed = {arm: totals(rows, arm, receipt["points"][arm], split) for arm in ARMS}
        require(len(rows) == count and block["totals"] == replayed, "Published result differs")
        found = [[r["arms"][a]["proof"]["anchors"] for a in ARMS if "proof" in r["arms"][a]] for r in rows]
        audit = block["audit"]
        require(audit["picture_clocks"] == sum(r["native_frames"] for r in rows)
                and audit.get("files", 0) == sum(bool(a) for a in found)
                and audit.get("proposals", 0) == sum(map(len, found))
                and audit.get("frame_crop_pairs", 0) == sum(len({p["frame"] for a in found_row for p in a})
                                                           for found_row in found), "Readback counts differ")
        print(split, json.dumps({arm: v["counts"] for arm, v in replayed.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
