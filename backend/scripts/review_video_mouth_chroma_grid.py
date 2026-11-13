# SPDX-License-Identifier: Apache-2.0
"""Recompute video chroma-phase decisions and verify native source exhibits."""

from __future__ import annotations

import argparse
import json
import re
import runpy
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

ARMS = ("dominant_axis", "both_axes")
SCHEMA = "chipmunk.video-mouth-chroma-grid/1"
FLOOR = 0.03
PRIOR = runpy.run_path(str(Path(__file__).with_name("review_video_mouth_track_cycle.py")))
read, sha, require, overlap = (PRIOR[name] for name in ("read", "sha", "require", "overlap"))
SPLITS = {"development": 56, "held": 146, "codec_stress": 34}
SHA256 = re.compile(r"[a-f0-9]{64}")


def points_for(rows):
    result = {}
    for arm in ARMS:
        scores = sorted((r["arms"][arm]["score"] for r in rows if r["arms"][arm]["score"] is not None), reverse=True)
        result[arm] = max(FLOOR, scores[2] if len(scores) >= 3 else 0)
    return result


def totals(rows, points):
    result = {}
    for arm in ARMS:
        stats, by_role, groups = Counter(), {}, set()
        for row in rows:
            reading = row["arms"][arm]
            require(bool(reading["seconds"]) == (reading["score"] is not None)
                    and (reading["measured"] or reading["score"] is None), "Proposal availability differs")
            flag = reading["score"] is not None and reading["score"] > points[arm]
            overlap_half = bool(reading["seconds"] and row["truth"]
                                and overlap(reading["seconds"], row["truth"]) >= Fraction(1, 2))
            label = "neural" if row["truth"] else "ordinary"
            located = flag and overlap_half
            proof = reading.get("native_evidence")
            if proof:
                a, b = proof["frames"]
                anchors = proof["anchors"]
                require((flag or overlap_half) and [x["frame"] for x in anchors] == [a - 1, (a + b) // 2, b]
                        and str(anchors[-1]["pts"] * Fraction(proof["time_base"])) == reading["seconds"][1]
                        and all(SHA256.fullmatch(x["rgb_sha256"]) and SHA256.fullmatch(x["crop_sha256"])
                                and x["crop_xyxy"][0] < x["crop_xyxy"][2] and x["crop_xyxy"][1] < x["crop_xyxy"][3]
                                for x in anchors), "Published native exhibit anchors differ")
            marks = {"files": 1, "measured": reading["measured"], "flags": flag,
                     "localized": located, "useful_before_point": overlap_half}
            by_role.setdefault(f"{row['family']}:{row['role']}", Counter()).update(marks)
            stats.update({f"{label}_{k}": v for k, v in marks.items()})
            if row["aligned"]:
                stats.update({f"strict_{row['family']}_{k}": v for k, v in marks.items()})
                if row["family"] == "lombard" and located:
                    groups.add(row["group"])
            if label == "ordinary" and flag and row["role"] in {"original", "reencode", "hard_cut"}:
                stats["essential_ordinary_flags"] += 1
            if label == "neural" and flag and not located:
                stats["wrong_locations"] += 1
        result[arm] = {"counts": dict(stats), "by_role": {k: dict(v) for k, v in by_role.items()},
                       "strict_liveportrait_localized_groups": len(groups),
                       "gate": {"liveportrait": stats["strict_lombard_localized"] >= 4 and len(groups) >= 3,
                                "musetalk": stats["strict_musetalk_localized"] == stats["strict_musetalk_files"] == 1,
                                "ordinary": stats["ordinary_flags"] <= 2 and not stats["essential_ordinary_flags"],
                                "coverage": all(stats[f"{key}_files"] and stats[f"{key}_measured"] * 10
                                                >= 7 * stats[f"{key}_files"]
                                                for key in ("strict_lombard", "strict_musetalk", "ordinary"))}}
    return result


def assemble(run):
    from app.detectors.video.decode import probe_frames

    audit_file = runpy.run_path(str(Path(__file__).with_name("review_video_mouth_contour_phase.py")))["audit_file"]
    freeze = run / "freeze.json"
    lock = read(freeze)
    points = read(run / "points.json")
    stress = run / "codec_stress_roster.json"
    transcodes = read(stress)
    require(transcodes["freeze_sha256"] == sha(freeze)
            and points["development_sha256"] == sha(run / "development.json"), "Native replay freezes differ")
    parents = {(r["family"], r["id"], r["role"]): r for r in lock["stress_sources"]}
    transcode_pictures = 0
    for row in transcodes["rows"]:
        source = parents.pop((row["family"], row["id"], row["role"]))
        require(sha(Path(source["file"])) == source["sha"] == row["source_sha"] and sha(Path(row["file"])) == row["sha"],
                "Stress source or delivered bytes differ")
        original, delivered = (probe_frames(Path(path), timeout_s=15) for path in (source["file"], row["file"]))
        streams = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,pix_fmt",
            "-of", "json", row["file"],
        ], timeout=10))["streams"]
        require(len(delivered) == row["verified_frames"]
                and streams == [{"codec_name": "h264", "codec_type": "video", "pix_fmt": "yuv420p"}]
                and [pts * original.tb for pts in original.pts] == [pts * delivered.tb for pts in delivered.pts],
                "Source and stress native clocks differ")
        transcode_pictures += len(delivered)
    require(not parents, "Missing codec stress source")
    report = {"schema": SCHEMA, "protocol": lock["protocol"], "freeze_sha256": sha(freeze),
              "codec_stress_roster_sha256": sha(stress), "stress_reopened_timestamps": transcode_pictures,
              "stress_codec": "h264", "stress_pixel_format": "yuv420p",
              "points": points["points"], "splits": {}}
    aliases = {}
    for split, count in SPLITS.items():
        path = run / (split + ".json")
        observed = read(path)
        require(observed["freeze_sha256"] == sha(freeze) and len(observed["rows"]) == count, "Run scope differs")
        roster = transcodes["rows"] if split == "codec_stress" else lock["rows"]
        pending = {(r["family"], r["id"], r["role"]): r for r in roster if r["split"] == split}
        records = []
        crops = Counter()
        for row in observed["rows"]:
            frozen = pending.pop((row["family"], row["id"], row["role"]))
            require(all(row[k] == v for k, v in frozen.items()), "Scored roster row differs")
            family = "fresh" if row["family"] == "derived_ordinary" else row["family"]
            key = family, row["group"]
            record = {"id": len(records) + 1, "family": row["family"], "role": row["role"],
                      "group": aliases.setdefault(key, len(aliases) + 1), "truth": row["truth"],
                      "aligned": row.get("aligned", False), "phase_shots": row["phase_shots"], "arms": {}}
            wanted = {}
            for arm, reading in row["arms"].items():
                p = reading["proposal"]
                record["arms"][arm] = {"measured": reading["measured"], "score": p["score"] if p else None,
                                       "seconds": p["seconds"] if p else None}
                if split == "development" or p is None:
                    continue
                if p["score"] <= points["points"][arm] and not (row["truth"] and overlap(p["seconds"], row["truth"]) >= Fraction(1, 2)):
                    continue
                table = probe_frames(Path(row["file"]), timeout_s=15)
                a, b = p["frames"]
                require(table.time_base == p["time_base"]
                        and p["seconds"] == [str(table.pts[i] * table.tb) for i in (a, b)]
                        and [v["frame"] for v in p["anchors"]] == [a - 1, (a + b) // 2, b],
                        "Proposal source times differ")
                wanted.update({v["frame"]: v for v in p["anchors"]})
                record["arms"][arm]["native_evidence"] = {key: p[key] for key in ("frames", "time_base", "anchors")}
                crops["proposals"] += 1
            if wanted:
                crops["frame_and_crop_pairs"] += audit_file(Path(row["file"]), row, wanted)
                crops["files"] += 1
            records.append(record)
        require(not pending, "Native replay omits cases")
        report["splits"][split] = {"sha256": sha(path), "rows": records, "totals": totals(records, points["points"]),
                                    "native_audit": dict(crops), "verified_picture_timestamps": sum(r["native_frames"] for r in observed["rows"])}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--assemble", type=Path)
    args = parser.parse_args()
    if args.assemble:
        with args.receipt.open("x") as handle:
            json.dump(assemble(args.assemble), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    receipt = read(args.receipt)
    require(receipt["schema"] == SCHEMA and set(receipt["splits"]) == set(SPLITS), "Unexpected replay scope")
    require(points_for(receipt["splits"]["development"]["rows"]) == receipt["points"], "Frozen points differ")
    for split, block in receipt["splits"].items():
        require(len(block["rows"]) == SPLITS[split] and totals(block["rows"], receipt["points"]) == block["totals"],
                "Decision totals differ")
        print(split, json.dumps({arm: row["counts"] for arm, row in block["totals"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
