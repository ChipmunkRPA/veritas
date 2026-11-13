# SPDX-License-Identifier: Apache-2.0
"""Read and independently recount the temporal mouth chroma study."""

import argparse
import hashlib
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-mouth-temporal-chroma/1"
ARMS = ("relative_flicker", "bilateral_disagreement")
POSITIVES = {("lombard", "source_time_neural"), ("musetalk", "same_time_neural")}
DIGEST = re.compile(r"[0-9a-f]{64}")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_bytes())


def overlap(proposal, truth):
    start, stop = map(Fraction, proposal["seconds"])
    before, after = map(Fraction, truth)
    return 2 * (min(stop, after) - max(start, before)) >= max(stop, after) - min(start, before)


def validate(row, result):
    proposal = result["proposal"]
    require(type(result["measured"]) is bool, "Invalid measured state")
    if proposal is None:
        return
    start, end = proposal["frames"]
    anchors = proposal["anchors"]
    require(result["measured"] and start >= 6 and end <= row["frames"] - 6 and end - start in (8, 12, 16, 20)
            and proposal["score"] > 0 and row["frame_support"] >= 16, "Invalid native proposal")
    require([anchor["frame"] for anchor in anchors] == [start - 1, (start + end) // 2, end], "Incorrect anchor indices")
    pts = [anchor["pts"] * Fraction(row["time_base"]) for anchor in anchors]
    first, last = map(Fraction, proposal["seconds"])
    require(pts[0] < first <= pts[1] < last == pts[2], "Native evidence time differs")
    for anchor in anchors:
        x, y, right, bottom = anchor["mouth_crop_xyxy"]
        require(0 <= x < right and 0 <= y < bottom and all(DIGEST.fullmatch(anchor[key])
                for key in ("rgb_sha256", "crop_sha256")), "Invalid native crop commitment")


def report(doc):
    require(doc["freeze"]["schema"] == SCHEMA, "Unknown temporal chroma study")
    dev, held = doc["development"], doc["held"]
    committed = (json.dumps(dev, indent=2, allow_nan=False) + "\n").encode()
    require(hashlib.sha256(committed).hexdigest() == doc["points"]["development_sha256"], "Development differs")
    require((len(dev), len(held)) == (56, 92), "Incomplete study")
    for rows in (dev, held):
        for index, row in enumerate(rows):
            require(row["index"] == index and DIGEST.fullmatch(row["sha256"]) and row["arms"].keys() == set(ARMS)
                    and Fraction(row["time_base"]) > 0 and 0 <= row["frame_support"] <= row["frames"], "Invalid delivery row")
            for outcome in row["arms"].values():
                validate(row, outcome)
    points = {arm: max(.10, sorted((row["arms"][arm]["proposal"] or {}).get("score", 0.) for row in dev)[-3])
              for arm in ARMS}
    require(doc["points"]["thresholds"] == points, "Frozen operating points differ")
    groups = Counter(row["family"] for row in held if (row["family"], row["role"]) in POSITIVES)
    require(groups == {"lombard": 11, "musetalk": 6}, "Redraw families differ")
    require(sum(row["aligned"] for row in held if (row["family"], row["role"]) in POSITIVES) == 11,
            "Timing-qualified subset differs")
    results = {}
    for arm, point in points.items():
        counts = Counter()
        qualified_groups = set()
        for row in held:
            is_neural = (row["family"], row["role"]) in POSITIVES
            family = row["family"] if is_neural else "ordinary"
            outcome = row["arms"][arm]
            proposal = outcome["proposal"]
            counts[f"{family}_measured"] += outcome["measured"]
            correct = bool(is_neural and proposal and overlap(proposal, row["truth"]))
            if is_neural:
                counts[f"{family}_raw_locations"] += correct
            if not proposal or proposal["score"] <= point:
                continue
            counts[f"{family}_flags"] += 1
            counts[f"{family}_locations"] += correct
            if correct and row["aligned"]:
                counts[f"{family}_timing_qualified"] += 1
                if family == "lombard":
                    qualified_groups.add(row["group"])
            if not is_neural:
                counts[f"ordinary_{row['role']}_flags"] += 1
        results[arm] = dict(counts) | {"passes": (counts["lombard_timing_qualified"] >= 4 and len(qualified_groups) >= 3
            and counts["musetalk_timing_qualified"] >= 1 and 10 * counts["ordinary_measured"] >= 7 * 75
            and 10 * (counts["lombard_measured"] + counts["musetalk_measured"]) >= 7 * 17
            and counts["ordinary_flags"] <= 2 and not (counts["ordinary_hard_cut_flags"] or counts["ordinary_original_flags"]))}
    return {"points": points, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--assemble", type=Path)
    parser.add_argument("--carriers", type=Path)
    args = parser.parse_args()
    if args.assemble is None:
        doc = read(args.receipt)
    else:
        doc = {key: read(args.assemble / f"{key}.json") for key in ("freeze", "points", "development", "held")}
    summary = report(doc)
    if args.carriers is not None:
        from review_video_mouth_photometric_transfer import replay_flagged

        summary["source_replay"] = replay_flagged(doc, args.carriers)
        if args.assemble is not None:
            doc["source_replay"] = summary["source_replay"]
    if args.assemble is not None:
        with args.receipt.open("x", encoding="utf-8") as output:
            json.dump(doc, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
