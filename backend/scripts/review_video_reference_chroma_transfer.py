# SPDX-License-Identifier: Apache-2.0
"""Recompute paired luma and chroma localization and validate source geometry."""

import argparse
import hashlib
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-reference-chroma-transfer/1"
ARMS = ("luma", "chroma")
NEURAL = {"source_time_neural", "same_time_neural"}
NEUTRAL = {"original", "compression", "global_chroma"}
SHA = re.compile(r"[0-9a-f]{64}")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def spatial_support(choice, truth):
    start, end = choice["frames"]
    a, b = truth
    if 2 * (min(b, end) - max(a, start)) < max(b, end) - min(a, start):
        return False
    area = choice["tile"]["w"] * choice["tile"]["h"]
    p = choice["pixel_support"]
    return 10 * p["covered"] >= p["changed"] + area - p["covered"]


def validate(row, choice, point):
    if choice is None:
        return False
    if "frames" not in choice:
        require(choice["score"] <= point, "Above-point response is missing native evidence")
        return False
    s, e = choice["frames"]
    require(row["shared"] and choice["score"] > point and s >= 5 and e <= row["pictures"] - 5 and e - s in (8, 12, 16, 20),
            "Proposal range differs from frozen selection")
    require([a["frame"] for a in choice["anchors"]] == [s - 1, (s + e) // 2, e], "Evidence roles differ from source range")
    require([a["pts"] for a in choice["anchors"]] == [[o + i * n for o, n in zip(row["first_pts"], row["steps"], strict=True)]
                for i in (s - 1, (s + e) // 2, e)], "Evidence clock differs")
    require(all(SHA.fullmatch(digest) for a in choice["anchors"] for key in ("rgb_sha256", "crop_sha256") for digest in a[key]),
            "Invalid picture or crop commitment")
    x, y, w, h = (choice["tile"][key] for key in "xywh")
    require(0 <= x < x + w <= 720 and 0 <= y < y + h <= 480
            and all(len(a[key]) == 2 for a in choice["anchors"] for key in ("rgb_sha256", "crop_sha256")),
            "Crop exceeds source geometry or the pair lacks both evidence sides")
    if row["truth"]:
        support = choice["pixel_support"]
        require(0 <= support["covered"] <= min(w * h, support["changed"]) <= 720 * 480, "Invalid pixel support")
    return True


def report(document):
    require(document["freeze"]["schema"] == SCHEMA, "Unknown paired chroma experiment")
    fit, held = document["development"], document["held"]
    require(len(fit) == 24 and len(held) == 91, "Incomplete blinded panel")
    points = document["points"]
    require(points == {"luma": 1.5712549522005277,
                       "chroma": max(.8, max((row["arms"]["chroma"] or {}).get("score", 0.) for row in fit) + .15)},
            "Frozen operating point differs")
    counts, added = Counter(), set()
    for row in held:
        require(row["arms"].keys() == set(ARMS) and len(row["files_sha256"]) == 2
                and all(SHA.fullmatch(digest) for digest in row["files_sha256"]), "Invalid pair commitment")
        paces = [step * Fraction(base) for step, base in zip(row["steps"], row["time_bases"], strict=True)]
        require(len(paces) == 2 and paces[0] == paces[1] > 0 and 40 <= row["pictures"] <= 100, "Invalid matched native cadence")
        role = ("neural" if row["role"] in NEURAL else "neutral" if row["role"] in NEUTRAL else
                row["role"] if row["role"] in {"wrong", "local_chroma"} else "ordinary_local")
        counts[role] += 1
        counts[role + "_assessed"] += row["shared"]
        flags = {arm: validate(row, choice, points[arm]) for arm, choice in row["arms"].items()}
        if role == "wrong":
            counts["wrong_source_admitted"] += row["shared"]
        for arm, flagged in flags.items():
            counts[f"{role}_{arm}_flags"] += flagged
        if not row["truth"]:
            counts[role + "_union_flags"] += any(flags.values())
            continue
        correct = {arm: flagged and spatial_support(row["arms"][arm], row["truth"]) for arm, flagged in flags.items()}
        for arm, hit in correct.items():
            counts[f"{role}_{arm}_correct"] += hit
            if role == "neural":
                counts[f"{row['bank']}_{arm}_correct"] += hit
                counts[f"{arm}_timing_qualified"] += hit and row["aligned"]
        counts[role + "_union_correct"] += any(correct.values())
        if role == "neural" and correct["chroma"] and not correct["luma"]:
            added.add((row["bank"], row["group"]))
            counts["neural_additions"] += 1
            counts["qualified_neural_additions"] += row["aligned"]
            counts[f"{row['bank']}_added"] += 1
    require([counts[key] for key in ("neural", "ordinary_local", "local_chroma", "wrong", "neutral")] == [17, 34, 8, 8, 24],
            "Incorrect case mix")
    counts["added_source_groups"] = len(added)
    passed = (counts["neural_additions"] >= 2 and len(added) >= 2 and counts["musetalk_added"] >= 1
              and counts["local_chroma_union_correct"] >= 6 and counts["neutral_union_flags"] - counts["neutral_luma_flags"] <= 1
              and counts["wrong_source_admitted"] == 0)
    return dict(counts) | {"passes": passed}


def native(rows, locations):
    from verify_native_video_interval import decode_rgb_frames, native_crop, probe, sha_file

    cache = set()
    pairs = 0
    for row in rows:
        for choice in row["arms"].values():
            if not choice or "anchors" not in choice:
                continue
            x, y, w, h = (choice["tile"][key] for key in "xywh")
            indices = [a["frame"] for a in choice["anchors"]]
            for side, digest in enumerate(row["files_sha256"]):
                identity = digest, tuple(indices), x, y, w, h
                if identity in cache:
                    continue
                path = locations[digest]
                width, height, base, pts, _ = probe(path)
                require((width, height) == (720, 480) and sha_file(path) == digest and len(pts) == row["pictures"]
                        and base == Fraction(row["time_bases"][side]), "Archived native evidence carrier differs")
                ticks = [pts[i] for i in indices]
                pictures = decode_rgb_frames(path, indices, ticks, base, width * height * 3)
                for index, rgb, anchor in zip(indices, pictures, choice["anchors"], strict=True):
                    crop = native_crop(rgb, width, x, y, w, h)
                    require(pts[index] == anchor["pts"][side] and hashlib.sha256(rgb).hexdigest() == anchor["rgb_sha256"][side]
                            and hashlib.sha256(crop).hexdigest() == anchor["crop_sha256"][side], "Archived native crop differs")
                cache.add(identity)
            pairs += 1
    return {"flagged_arm_ranges": pairs, "distinct_carrier_range_crops": 3 * len(cache)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--assemble", type=Path)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if args.assemble:
        run = args.assemble
        data = {key: json.loads((run / f"{key}.json").read_bytes()) for key in ("freeze", "points", "development")}
        data["held"] = [json.loads(line) for line in (run / "held.jsonl").read_text().splitlines()]
    else:
        data = json.loads(args.ledger.read_bytes())
    summary = report(data)
    if args.assemble and args.previous:
        from verify_native_video_interval import sha_file

        roster = json.loads((args.previous / "freeze.json").read_bytes())["roster"]
        paths = {row["carrier_sha256"]: Path(row["carrier"]) for row in roster}
        paths.update({sha_file(path): path for path in args.assemble.glob("crema-*.mp4")})
        data["native_replay"] = native(data["held"], paths)
        summary["native_replay"] = data["native_replay"]
    if args.assemble:
        with args.ledger.open("x", encoding="utf-8") as output:
            json.dump(data, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
