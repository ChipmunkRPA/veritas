# SPDX-License-Identifier: Apache-2.0
"""Recompute the paired video boundary comparison and replay native crops."""

import argparse
import json
from collections import Counter, defaultdict
from fractions import Fraction
from math import isfinite
from pathlib import Path
from statistics import median, quantiles

from review_video_reference_chroma_transfer import SHA, native, require
from verify_native_video_interval import sha_file

SCHEMA = "chipmunk.video-reference-boundary-transfer/1"
CONFIRMATION = "chipmunk.video-reference-measured-interval-confirmation/1"


def refinement(series, start, stop, *, confirmed):
    lo = max(0, start - 10)
    hi = min(len(series), stop + 10)
    if any(value is None for value in series[lo:hi]):
        return None
    left = series[lo:max(3, start - 4)]
    right = series[min(len(series) - 3, stop + 4):hi]
    flank = max(median(left), median(right))
    trim = (stop - start) // 4
    margin = median(series[start + trim:stop - trim]) - flank
    if margin < 1.5:
        return None
    threshold = max(flank + .30 * margin, *(quantiles(values, n=5, method="inclusive")[-1] + .5
                                           for values in (left, right)))
    middle = (start + stop) // 2
    if series[middle] < threshold:
        return None
    a = next((i + 1 for i in reversed(range(max(1, start - 7), middle)) if series[i] < threshold), max(1, start - 7))
    b = next((i for i in range(middle, min(len(series) - 2, stop + 6)) if series[i] < threshold),
             min(len(series) - 2, stop + 6))
    if (a < 2 or b + 2 > len(series) or b - a < 4 or abs(a - start) > 6 or abs(b - stop) > 6
            or any(value >= threshold for value in series[a - 2:a] + series[b:b + 2])):
        return None
    if min(series[a] - series[a - 1], series[b - 1] - series[b]) < max(.75, (.50 if confirmed else .18) * margin):
        return None
    return [a, b]


def evidence(row, record):
    start, stop = record["frames"]
    indices = [start - 1, (start + stop) // 2, stop]
    require(record["samples"] == dict(zip(("before", "during", "after"), indices, strict=True)), "Native preview roles differ")
    require([anchor["frame"] for anchor in record["anchors"]] == indices, "Native preview indices differ")
    x, y, w, h = (record["tile"][k] for k in "xywh")
    require(0 <= x < x + w <= 720 and 0 <= y < y + h <= 480, "Native crop exceeds source")
    for i, anchor in zip(indices, record["anchors"], strict=True):
        require(anchor["pts"] == [offset + i * step for offset, step in zip(row["first_pts"], row["steps"], strict=True)],
                "Native preview clock differs")
        for key in ("rgb_sha256", "crop_sha256"):
            require(len(anchor[key]) == 2 and all(SHA.fullmatch(value) for value in anchor[key]), "Native image commitment differs")
    if row["truth"]:
        pixels = record["pixel_support"]
        require(0 <= pixels["covered"] <= min(w * h, pixels["changed"]) <= 720 * 480, "Invalid spatial overlap")


def review(receipt):
    schema = receipt["freeze"]["schema"]
    require(schema in (SCHEMA, CONFIRMATION), "Unknown boundary study")
    confirmed = schema == CONFIRMATION
    tally, groups, redraw_groups, errors = Counter(), set(), set(), defaultdict(list)
    redraw_gains = qualified_gains = 0
    for row in receipt["rows"]:
        status, truth, length = (row[key] for key in ("status", "truth", "pictures"))
        require(40 <= length <= 100 and len(row["files_sha256"]) == 2
                and all(SHA.fullmatch(digest) for digest in row["files_sha256"]), "Invalid paired source commitment")
        first, second = [Fraction(base) * step for base, step in zip(row["time_bases"], row["steps"], strict=True)]
        require(first == second and Fraction(1, 25) <= first <= Fraction(1, 23), "Invalid paired source clock")
        category = "abrupt" if row["role"] in {"grade", "texture"} else row["role"]
        tally[category] += 1
        old, new = (row["arms"][arm] for arm in ("serving", "refined"))
        tally[category + "_old"] += old is not None
        tally[category + "_refined"] += new is not None
        if old is None:
            require(new is None and status != "local_difference", "The temporal refinement introduced a new finding")
            continue
        s, e = old["frames"]
        require(status == "local_difference" and e - s in {8, 12, 16, 20} and 5 <= s < e <= length - 5
                and isfinite(old["score"]) and old["score"] > 1.5712549522005277, "Original local proposal differs")
        require(len(row["series"]) == length and all(value is None or (isfinite(value) and value >= 0) for value in row["series"]),
                "Invalid seed series")
        require(refinement(row["series"], s, e, confirmed=confirmed) == (new["frames"] if new else None),
                "Refined range differs from paired evidence")
        evidence(row, old)
        if new:
            evidence(row, new)
            require(new["tile"] == old["tile"] and new["score"] == old["score"], "Temporal refinement changed its region or score")
        if truth is None:
            continue
        result = new or old
        before, after = [tuple(abs(a - b) for a, b in zip(truth, case["frames"], strict=True)) for case in (old, result)]
        tally[category + "_old_exact"] += before == (0, 0)
        tally[category + "_final_exact"] += after == (0, 0)
        tally[category + "_worse"] += max(after[0] - before[0], after[1] - before[1]) > (1 if category == "abrupt" else 2)
        if category == "abrupt" and before != (0, 0) and after == (0, 0):
            groups.add(row["group"])
        if category == "redraw":
            errors[row["bank"]].append((sum(before), sum(after)))
            if row["aligned"]:
                errors["timing_qualified"].append((sum(before), sum(after)))
            if sum(after) < sum(before):
                redraw_gains += 1
                qualified_gains += row["aligned"]
                redraw_groups.add((row["bank"], row["group"]))
        area = old["tile"]["w"] * old["tile"]["h"]
        support = [11 * case["pixel_support"]["covered"] >= case["pixel_support"]["changed"] + area for case in (old, result)]
        tally[category + "_pixel_losses"] += support == [True, False]
    require([tally[key] for key in ("abrupt", "fade", "delivery", "global", "wrong", "redraw")] == [16, 8, 8, 8, 8, 17],
            "Boundary cohort coverage differs")
    medians = {family: [median(first), median(second)] for family, rows in errors.items() for first, second in [zip(*rows, strict=True)]}
    passed = (tally["abrupt_final_exact"] >= 12 and tally["abrupt_final_exact"] - tally["abrupt_old_exact"] >= 6
              and len(groups) >= 4 and not tally["abrupt_worse"] and tally["fade_refined"] <= 1
              and all(a >= b for a, b in medians.values()) and tally["redraw_worse"] <= 1
              and all(not tally[name + "_refined"] for name in ("delivery", "global", "wrong"))
              and tally["abrupt_pixel_losses"] == tally["redraw_pixel_losses"] == 0)
    summary = dict(tally) | {"source_groups_with_new_exact_range": len(groups), "redraw_median_total_endpoint_error": medians,
                             "passes": passed}
    if confirmed:
        summary.update(redraw_endpoint_gains=redraw_gains, timing_qualified_redraw_endpoint_gains=qualified_gains,
                       redraw_endpoint_gain_groups=len(redraw_groups),
                       passes=passed and len(redraw_groups) >= 2 and qualified_gains >= 1)
    require(summary == receipt["summary"], "Recomputed summary differs")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_bytes())
    summary = review(receipt)
    if args.run:
        require(args.previous is not None, "Native replay requires the prior archived redraw panel")
        earlier = json.loads((args.previous / "freeze.json").read_bytes())
        carriers = {r["carrier_sha256"]: Path(r["carrier"]) for r in earlier["roster"]}
        carriers.update({sha_file(path): path for path in args.run.glob("held-*.mp4")})
        summary["native_replay"] = native(receipt["rows"], carriers)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
