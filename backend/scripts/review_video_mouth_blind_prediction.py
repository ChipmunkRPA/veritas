# SPDX-License-Identifier: Apache-2.0
"""Recount neighbor prediction results and optionally reopen the original carriers."""

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-mouth-blind-prediction/1"
ARMS = ("unwarped_neighbors", "motion_predicted_neighbors")
SPLITS = {"development": 56, "held": 92}
ORDINARY_ROLES = {"original", "reencode", "hard_cut"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value.read_bytes() if isinstance(value, Path) else value).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def iou(a, b):
    start, end = map(Fraction, a)
    left, right = map(Fraction, b)
    return max(0, min(end, right) - max(start, left)) / (max(end, right) - min(start, left))


def validate_anchor(proposal, row):
    start, end = proposal["frames"]
    before, during, after = proposal["anchors"]
    tb = Fraction(row["time_base"])
    first, last = map(Fraction, proposal["seconds"])
    require(5 <= start < end <= row["frames"] - 5 and end - start in {8, 12, 16, 20}
            and proposal["time_base"] == row["time_base"]
            and [a["frame"] for a in proposal["anchors"]] == [start - 1, (start + end) // 2, end]
            and before["pts"] * tb < first <= during["pts"] * tb < last == after["pts"] * tb,
            "Proposal and native frame positions disagree")
    for a in proposal["anchors"]:
        x0, y0, x1, y1 = a["crop_xyxy"]
        require(0 <= x0 < x1 and 0 <= y0 < y1 and all(
            len(a[k]) == 64 and len(bytes.fromhex(a[k])) == 32 for k in ("rgb_sha256", "crop_sha256")
        ), "Malformed crop commitment")


def calculate(blocks):
    fit = blocks["development"]
    points = {arm: max(.04, sorted((r["arms"][arm]["score"] or 0 for r in fit), reverse=True)[2]) for arm in ARMS}
    output = {}
    for split, rows in blocks.items():
        require(len(rows) == SPLITS[split] and [r["index"] for r in rows] == list(range(len(rows))),
                "Scored roster is incomplete")
        output[split] = {}
        for arm in ARMS:
            counts, groups, roles = Counter(), set(), Counter()
            for r in rows:
                value = r["arms"][arm]
                score, proposal = value["score"], value.get("proposal")
                require(type(value["measured"]) is bool and 0 <= r["frame_support"][arm] <= r["frames"]
                        and (score is None or value["measured"] and math.isfinite(score) and score > 0),
                        "Measurement differs from its recorded support")
                flag = score is not None and score > points[arm]
                useful = bool(r["truth"] and value.get("seconds") and iou(r["truth"], value["seconds"]) >= Fraction(1, 2))
                require((score is not None) == bool(value.get("seconds"))
                        and bool(proposal) == (split == "held" and (flag or useful)),
                        "Missing or unexpected proposal evidence")
                if proposal:
                    require(score == proposal["score"] and value["seconds"] == proposal["seconds"],
                            "Proposal response or interval differs")
                    validate_anchor(proposal, r)
                label = "neural" if r["truth"] else "ordinary"
                counts.update({f"{label}_files": 1, f"{label}_measured": value["measured"], f"{label}_flags": flag})
                if r["truth"]:
                    counts.update({"localized": flag and useful, "misplaced": flag and not useful})
                    if r["aligned"]:
                        counts.update({f"qualified_{r['family']}_files": 1,
                                       f"qualified_{r['family']}_localized": flag and useful})
                        if flag and useful and r["family"] == "lombard":
                            groups.add(r["group"])
                if not r["truth"] and flag:
                    roles[r["role"]] += 1
            output[split][arm] = {"counts": dict(counts), "ordinary_flag_roles": dict(roles),
                                  "qualified_liveportrait_groups": len(groups)}
            if split == "held":
                output[split][arm]["passed"] = (counts["qualified_lombard_localized"] >= 4 and len(groups) >= 3
                                                and counts["qualified_musetalk_localized"] == 1
                                                and counts["ordinary_flags"] <= 2 and not ORDINARY_ROLES.intersection(roles))
    return points, output


def replay(path, row):
    require(sha(path) == row["source_sha256"], "Carrier content differs")
    command = ["ffmpeg", "-nostdin", "-v", "error", "-copyts", "-i", str(path), "-map", "0:v:0",
               "-an", "-fps_mode", "passthrough"]
    output = subprocess.check_output(command + ["-enc_time_base", "demux", "-pix_fmt", "rgb24", "-f",
                                                 "framehash", "-hash", "sha256", "-"], text=True, timeout=20)
    lines = output.splitlines()
    tb = next(line.split(": ", 1)[1] for line in lines if line.startswith("#tb 0: "))
    width, height = map(int, next(line.split(": ", 1)[1] for line in lines if line.startswith("#dimensions 0: ")).split("x"))
    hashes = [line.replace(" ", "").split(",") for line in lines if not line.startswith("#")]
    pts = [int(parts[2]) for parts in hashes]
    require(tb == row["time_base"] and len(pts) == row["frames"] and sha(json.dumps(pts).encode()) == row["pts_sha256"],
            "Decoded carrier clock differs")
    anchors = {}
    for value in row["arms"].values():
        for anchor in value.get("proposal", {}).get("anchors", []):
            index = anchor["frame"]
            require(anchors.setdefault(index, anchor) == anchor, "Conflicting source crop")
    if anchors:
        selector = "+".join(f"eq(n,{i})" for i in sorted(anchors))
        frames = subprocess.check_output(command + ["-vf", f"select='{selector}'", "-pix_fmt", "rgb24",
                                                     "-f", "rawvideo", "-"], timeout=20)
        size, stride = width * height * 3, width * 3
        require(len(frames) == len(anchors) * size, "Incomplete RGB replay")
        for j, index in enumerate(sorted(anchors)):
            frame = frames[j * size:(j + 1) * size]
            a = anchors[index]
            x0, y0, x1, y1 = a["crop_xyxy"]
            require(x1 <= width and y1 <= height, "Crop is outside its source")
            crop = b"".join(frame[y * stride + x0 * 3:y * stride + x1 * 3] for y in range(y0, y1))
            require(a["pts"] == pts[index] and a["rgb_sha256"] == hashes[index][-1] == sha(frame)
                    and a["crop_sha256"] == sha(crop), "Frame or crop bytes differ")
    return pts, len(anchors)


def assemble(work, carrier_file):
    lock_file = work / "freeze.json"
    lock, carriers = read(lock_file), read(carrier_file)
    require(lock["schema"] == SCHEMA and sha(carrier_file) == lock["prior_sha256"], "Unexpected frozen carrier roster")
    receipt = {"schema": SCHEMA, "freeze_sha256": sha(lock_file), "method": lock, "splits": {}, "reopened": {}}
    recorded_points = read(work / "points.json")
    require(recorded_points["development_sha256"] == sha(work / "development.json"), "Unbound fit")
    points = recorded_points["points"]
    for split in SPLITS:
        doc = read(work / f"{split}.json")
        require(doc["freeze_sha256"] == receipt["freeze_sha256"], "Unbound result")
        rows = doc["rows"]
        roster = [r for r in carriers["rows"] if r["split"] == split]
        pics = anchors = 0
        for r, item in zip(rows, roster, strict=True):
            require(r["source_sha256"] == item["carrier_sha256"] and r["frames"] == item["verified_frames"]
                    and all(r[k] == item[k] for k in ("role", "family", "group"))
                    and r["aligned"] == item.get("aligned", False), "Scored source differs from the frozen roster")
            for arm in ARMS:
                value = r["arms"][arm]
                proposal = value.pop("proposal")
                value["seconds"] = proposal["seconds"] if proposal else None
                if split == "held" and proposal and (value["score"] > points[arm] or
                                                      r["truth"] and iou(r["truth"], value["seconds"]) >= Fraction(1, 2)):
                    value["proposal"] = proposal
            pts, a = replay(Path(item["carrier"]), r)
            truth = [str(pts[i] * Fraction(r["time_base"])) for i in item["edit"]] if item.get("edit") else item.get("native_truth")
            require(r["truth"] == truth, "Authored interval differs")
            for value in r["arms"].values():
                if proposal := value.get("proposal"):
                    require(proposal["seconds"] == [str(pts[i] * Fraction(r["time_base"])) for i in proposal["frames"]],
                            "Proposal interval differs from the decoded clock")
            pics += len(pts)
            anchors += a
        receipt["splits"][split] = rows
        receipt["reopened"][split] = {"timestamps": pics, "frame_and_crop_pairs": anchors}
    receipt["points"], receipt["summary"] = calculate(receipt["splits"])
    require(receipt["points"] == points, "Fit operating points differ")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--assemble-from", type=Path)
    parser.add_argument("--carriers", type=Path)
    args = parser.parse_args()
    if args.assemble_from is not None:
        require(args.carriers is not None, "Assembly requires the frozen carrier roster")
        receipt = assemble(args.assemble_from, args.carriers)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    else:
        receipt = read(args.receipt)
        require(receipt["schema"] == SCHEMA and calculate(receipt["splits"]) == (receipt["points"], receipt["summary"]),
                "Public decision totals differ")
    print(json.dumps({"reopened": receipt["reopened"], "summary": receipt["summary"]}, indent=2))


if __name__ == "__main__":
    main()
