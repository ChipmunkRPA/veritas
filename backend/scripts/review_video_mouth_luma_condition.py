# SPDX-License-Identifier: Apache-2.0
"""Recompute brightness-conditioned decisions and replay source picture evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import runpy
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-mouth-luma-condition/1"
ARMS = ("plain", "conditioned")
FLOOR = 0.04
SPLITS = {"development": 56, "held": 146, "codec_stress": 34}
SHA256 = re.compile(r"[0-9a-f]{64}")
_PRIOR = runpy.run_path(str(Path(__file__).with_name("review_video_mouth_track_cycle.py")))
read, sha, require, overlap = (_PRIOR[name] for name in ("read", "sha", "require", "overlap"))


def points_for(rows):
    points = {}
    for arm in ARMS:
        scores = sorted((r["arms"][arm]["score"] for r in rows if r["arms"][arm]["score"] is not None), reverse=True)
        points[arm] = max(FLOOR, scores[2] if len(scores) >= 3 else 0)
    return points


def totals(rows, points, split):
    output = {}
    for arm in ARMS:
        counts, by_role, groups = Counter(), {}, set()
        for row in rows:
            reading = row["arms"][arm]
            score, seconds = reading["score"], reading["seconds"]
            require((score is not None) == bool(seconds) and (score is None or math.isfinite(score) and score > 0
                    and reading["measured"]), "Proposal and score disagree")
            flag = score is not None and score > points[arm]
            useful = bool(seconds and row["truth"] and overlap(seconds, row["truth"]) >= Fraction(1, 2))
            neural = row["truth"] is not None
            proof = reading.get("native_evidence")
            require((proof is not None) == (split != "development" and (flag or useful)),
                    "Selected proposal native evidence is missing or extra")
            if proof:
                a, b = proof["frames"]
                anchors = proof["anchors"]
                require([v["frame"] for v in anchors] == [a - 1, (a + b) // 2, b]
                        and Fraction(seconds[0]) > anchors[0]["pts"] * Fraction(proof["time_base"])
                        and Fraction(seconds[0]) <= anchors[1]["pts"] * Fraction(proof["time_base"]) < Fraction(seconds[1])
                        and Fraction(seconds[1]) == anchors[2]["pts"] * Fraction(proof["time_base"])
                        and all(SHA256.fullmatch(v["rgb_sha256"]) and SHA256.fullmatch(v["crop_sha256"])
                                and 0 <= v["crop_xyxy"][0] < v["crop_xyxy"][2]
                                and 0 <= v["crop_xyxy"][1] < v["crop_xyxy"][3] for v in anchors),
                        "Before, during and after source anchors differ")
            marks = {"files": 1, "measured": reading["measured"], "flags": flag, "localized": flag and useful,
                     "useful_before_point": useful, "supported_pictures": row["frame_support"][arm],
                     "pictures": row["native_frames"]}
            role = f"{row['family']}:{row['role']}"
            by_role.setdefault(role, Counter()).update(marks)
            key = "neural" if neural else "ordinary"
            counts.update({f"{key}_{k}": v for k, v in marks.items()})
            if row["aligned"]:
                counts.update({f"strict_{row['family']}_{k}": v for k, v in marks.items()})
                if row["family"] == "lombard" and flag and useful:
                    groups.add(row["group"])
            if flag and neural and not useful:
                counts["wrong_locations"] += 1
            if flag and not neural and row["role"] in {"original", "reencode", "hard_cut"}:
                counts["essential_ordinary_flags"] += 1
        output[arm] = {"counts": dict(counts), "by_role": {k: dict(v) for k, v in by_role.items()},
                       "strict_liveportrait_localized_groups": len(groups)}
        if split == "held":
            output[arm]["gate"] = {"liveportrait": counts["strict_lombard_localized"] >= 4 and len(groups) >= 3,
                                   "musetalk": counts["strict_musetalk_localized"] == counts["strict_musetalk_files"] == 1,
                                   "ordinary": counts["ordinary_flags"] <= 2 and not counts["essential_ordinary_flags"],
                                   "coverage": all(counts[f"{key}_files"] and counts[f"{key}_measured"] * 10
                                                   >= 7 * counts[f"{key}_files"] for key in (
                                                       "strict_lombard", "strict_musetalk", "ordinary"))}
    return output


def stress_clocks(rows):
    from app.detectors.video.decode import probe_frames

    cases = {(r["family"], r["id"], r["role"]): r for r in rows if r["split"] == "held"}
    count = 0
    for row in rows:
        if row["split"] != "codec_stress":
            continue
        source = cases[(row["family"], row["id"], row["role"])]
        require(sha(Path(source["file"])) == source["sha"] == row["source_sha"]
                and sha(Path(row["file"])) == row["sha"], "Stress source bytes differ")
        a, b = (probe_frames(Path(r["file"]), timeout_s=15) for r in (source, row))
        streams = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,pix_fmt",
            "-of", "json", row["file"],
        ], timeout=10))["streams"]
        require(streams == [{"codec_name": "h264", "codec_type": "video", "pix_fmt": "yuv420p"}]
                and len(b) == row["verified_frames"] and [p * a.tb for p in a.pts] == [p * b.tb for p in b.pts],
                "Stress carrier or exact native clock differs")
        count += len(b)
    return count


def assemble(work):
    from app.detectors.video.decode import probe_frames

    audit_file = runpy.run_path(str(Path(__file__).with_name("review_video_mouth_contour_phase.py")))["audit_file"]
    lock_file = work / "freeze.json"
    lock = read(lock_file)
    points = read(work / "points.json")
    require(points["development_sha256"] == sha(work / "development.json"), "Point is unbound")
    receipt = {"schema": SCHEMA, "protocol": lock["protocol"], "prior_rosters": lock["prior_rosters"],
               "freeze_sha256": sha(lock_file), "points": points["points"], "splits": {},
               "stress_clock_reopened_pictures": stress_clocks(lock["rows"])}
    groups = {}
    for split, count in SPLITS.items():
        scored = work / f"{split}.json"
        block = read(scored)
        require(block["freeze_sha256"] == sha(lock_file) and len(block["rows"]) == count, "Observed split differs")
        pending = {(r["family"], r["id"], r["role"]): r for r in lock["rows"] if r["split"] == split}
        rows, audit = [], Counter()
        for row in block["rows"]:
            frozen = pending.pop((row["family"], row["id"], row["role"]))
            require(all(row[k] == v for k, v in frozen.items()), "Observed roster differs")
            group = ("fresh" if row["family"] == "derived_ordinary" else row["family"], row["group"])
            public = {k: row[k] for k in ("family", "role", "truth", "native_frames", "frame_support")}
            public.update(group=groups.setdefault(group, len(groups) + 1), aligned=row.get("aligned", False), arms={})
            wanted = {}
            for arm in ARMS:
                answer = row["arms"][arm]
                proposal = answer["proposal"]
                public["arms"][arm] = {"measured": answer["measured"], "score": answer["score"],
                                       "seconds": proposal["seconds"] if proposal else None}
                if split == "development" or not proposal:
                    continue
                if (proposal["score"] <= receipt["points"][arm] and
                        not (row["truth"] and overlap(proposal["seconds"], row["truth"]) >= Fraction(1, 2))):
                    continue
                table = probe_frames(Path(row["file"]), timeout_s=15)
                require(table.time_base == proposal["time_base"] and
                        [str(table.pts[i] * table.tb) for i in proposal["frames"]] == proposal["seconds"],
                        "Proposal and source range differ")
                wanted.update({a["frame"]: a for a in proposal["anchors"]})
                public["arms"][arm]["native_evidence"] = {key: proposal[key] for key in ("time_base", "frames", "anchors")}
                audit["proposals"] += 1
            if wanted:
                audit["frame_and_crop_pairs"] += audit_file(Path(row["file"]), row, wanted)
                audit["files"] += 1
            rows.append(public)
        require(not pending, "Scoring omits cases")
        receipt["splits"][split] = {"rows": rows, "sha256": sha(scored),
                                    "native_audit": dict(audit), "totals": totals(rows, receipt["points"], split)}
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--assemble", type=Path)
    args = parser.parse_args()
    if args.assemble:
        receipt = assemble(args.assemble)
        with args.receipt.open("x") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    receipt = read(args.receipt)
    require(receipt["schema"] == SCHEMA and set(receipt["splits"]) == set(SPLITS), "Unknown scope")
    require(points_for(receipt["splits"]["development"]["rows"]) == receipt["points"], "Frozen points differ")
    for split, block in receipt["splits"].items():
        observed = totals(block["rows"], receipt["points"], split)
        require(len(block["rows"]) == SPLITS[split] and observed == block["totals"], "Outcome totals differ")
        print(split, json.dumps({k: v["counts"] for k, v in observed.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
