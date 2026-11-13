# SPDX-License-Identifier: Apache-2.0
"""Independently recount localized single-file mouth observations."""

import argparse
import hashlib
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-mouth-photometric/1"
ARMS = ("skin_balance", "balance_with_texture")
POSITIVES = {("lombard", "source_time_neural"), ("musetalk", "same_time_neural")}
SHA = re.compile(r"[0-9a-f]{64}")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load(path):
    return json.loads(path.read_bytes())


def local_overlap(proposal, truth):
    start, stop = map(Fraction, proposal["seconds"])
    before, after = map(Fraction, truth)
    union = max(stop, after) - min(start, before)
    return 2 * (min(stop, after) - max(start, before)) >= union


def check_proposal(row, value):
    require(type(value["measured"]) is bool, "Invalid coverage")
    proposal = value["proposal"]
    if not proposal:
        return
    start, stop = proposal["frames"]
    require(value["measured"] and row["frame_support"] >= 14 and proposal["score"] > 0
            and stop - start in {8, 12, 16, 20} and 5 <= start < stop <= row["frames"] - 5, "Invalid time proposal")
    before, during, after = proposal["anchors"]
    require([a["frame"] for a in (before, during, after)] == [start - 1, (start + stop) // 2, stop],
            "Evidence frame order differs")
    tb = Fraction(row["time_base"])
    first, last = map(Fraction, proposal["seconds"])
    require(before["pts"] * tb < first <= during["pts"] * tb < last == after["pts"] * tb, "Evidence timestamps differ")
    for anchor in (before, during, after):
        x, y, right, bottom = anchor["mouth_crop_xyxy"]
        require(0 <= x < right and 0 <= y < bottom and all(SHA.fullmatch(anchor[k]) for k in ("rgb_sha256", "crop_sha256")),
                "Invalid source crop commitment")


def report(document):
    require(document["freeze"]["schema"] == SCHEMA, "Unknown photometric study")
    fit, held = (document[key] for key in ("development", "held"))
    fitted_bytes = (json.dumps(fit, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(hashlib.sha256(fitted_bytes).hexdigest() == document["points"]["development_sha256"],
            "Published development commitment differs")
    require(len(fit) == 56 and len(held) == 92, "Incomplete matched panel")
    for rows in (fit, held):
        require([r["index"] for r in rows] == list(range(len(rows))), "Incomplete delivery order")
        for row in rows:
            require(row["arms"].keys() == set(ARMS) and SHA.fullmatch(row["sha256"])
                    and 0 <= row["frame_support"] <= row["frames"] and Fraction(row["time_base"]) > 0, "Invalid delivery")
            for value in row["arms"].values():
                check_proposal(row, value)
    points = {arm: max(.035, sorted((row["arms"][arm]["proposal"] or {}).get("score", 0.) for row in fit)[-3]) for arm in ARMS}
    require(points == document["points"]["thresholds"], "Frozen operating points differ")
    require(sum((r["family"], r["role"]) in POSITIVES for r in held) == 17, "Invalid redraw denominator")
    results = {}
    for arm, point in points.items():
        stats = Counter()
        qualified_groups = set()
        for row in held:
            value = row["arms"][arm]
            neural = (row["family"], row["role"]) in POSITIVES
            group = row["family"] if neural else "ordinary"
            stats[f"{group}_measured"] += int(value["measured"])
            proposal = value["proposal"]
            correct = bool(neural and proposal and local_overlap(proposal, row["truth"]))
            if neural:
                stats[f"{group}_overlaps_before_point"] += int(correct)
            if proposal is None or proposal["score"] <= point:
                continue
            stats[f"{group}_flags"] += 1
            stats[f"{group}_localized"] += int(correct)
            if neural and row["aligned"] and correct:
                stats[f"{group}_timing_qualified"] += 1
                if group == "lombard":
                    qualified_groups.add(row["group"])
            if not neural and row["role"] in {"hard_cut", "original"}:
                stats["critical_ordinary_flags"] += 1
        results[arm] = dict(stats) | {"qualified_first_family_groups": len(qualified_groups), "passes": (
            stats["lombard_timing_qualified"] >= 4 and len(qualified_groups) >= 3 and stats["musetalk_timing_qualified"] >= 1
            and 10 * (stats["lombard_measured"] + stats["musetalk_measured"]) >= 7 * 17
            and 10 * stats["ordinary_measured"] >= 7 * 75 and stats["ordinary_flags"] <= 2 and not stats["critical_ordinary_flags"]
        )}
    return {"points": points, "results": results}


def replay_flagged(document, carrier_file):
    from verify_native_video_interval import decode_rgb_frames, native_crop, probe, sha_file

    require(sha_file(carrier_file) == document["freeze"]["prior_sha256"], "Archived source roster differs")
    paths = {row["carrier_sha256"]: Path(row["carrier"]) for row in load(carrier_file)["rows"]}
    files, anchors = 0, 0
    for row in document["held"]:
        selected = {}
        for arm, value in row["arms"].items():
            if (proposal := value["proposal"]) and proposal["score"] > document["points"]["thresholds"][arm]:
                for anchor in proposal["anchors"]:
                    require(selected.setdefault(anchor["frame"], anchor) == anchor, "Contradictory source crops")
        if not selected:
            continue
        path = paths[row["sha256"]]
        require(sha_file(path) == row["sha256"], "Carrier bytes differ")
        width, height, tb, pts, _ = probe(path)
        indices = sorted(selected)
        ticks = [pts[index] for index in indices]
        require(str(tb) == row["time_base"] and len(pts) == row["frames"], "Native clock differs")
        pictures = decode_rgb_frames(path, indices, ticks, tb, width * height * 3)
        for index, picture in zip(indices, pictures, strict=True):
            entry = selected[index]
            x, y, right, bottom = entry["mouth_crop_xyxy"]
            require(right <= width and bottom <= height and pts[index] == entry["pts"]
                    and hashlib.sha256(picture).hexdigest() == entry["rgb_sha256"], "Native evidence frame differs")
            crop = native_crop(picture, width, x, y, right - x, bottom - y)
            require(hashlib.sha256(crop).hexdigest() == entry["crop_sha256"], "Native evidence crop differs")
        files += 1
        anchors += len(indices)
    return {"flagged_files": files, "exact_native_crops": anchors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--assemble", type=Path)
    parser.add_argument("--carriers", type=Path)
    args = parser.parse_args()
    if args.assemble is None:
        document = load(args.receipt)
    else:
        document = {key: load(args.assemble / f"{key}.json") for key in ("freeze", "points", "development", "held")}
        fit = (args.assemble / "development.json").read_bytes()
        require(hashlib.sha256(fit).hexdigest() == document["points"]["development_sha256"], "Development commitment differs")
    result = report(document)
    if args.carriers is not None:
        result["source_replay"] = replay_flagged(document, args.carriers)
    if args.assemble is not None:
        with args.receipt.open("x", encoding="utf-8") as output:
            json.dump(document, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
