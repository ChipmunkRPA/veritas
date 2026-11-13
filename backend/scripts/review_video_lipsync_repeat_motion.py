# SPDX-License-Identifier: Apache-2.0
"""Recompute the repeat-motion decision ledger without media or dependencies."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-lipsync-repeat-motion/1"
CARRIERS = ("pcm", "aac")
POSITIVE = ("sound_later", "sound_earlier", "local_late", "local_early")
CONTROL = ("genuine", "gain", "lowpass", "global", "picture_copy", "matched_av",
           "matched_mouth_sound", "independent_sound", "quiet_repeated_bed")
SYNC = {"global", "matched_av", "matched_mouth_sound"}
SPLITS = ("development", "retained", "transfer")
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def overlap(a, b):
    return Fraction(max(0, min(a[1], b[1]) - max(a[0], b[0])), max(a[1], b[1]) - min(a[0], b[0]))


def useful(pairs, truth):
    return bool(truth) and any(min(overlap(a, b) for a, b in zip(pair, truth, strict=True)) >= Fraction(1, 2)
                               for pair in pairs)


def arms(case, point):
    motion = [r["samples"] for r in case["motion"] if r["distance"] > point]
    return {"nominated": case["similar_audio"], "static": case["static"], "motion": motion,
            "combined": [*case["static"], *motion]}


def measured_groups(rows):
    return [row for row in rows if "cases" in row]


def calibrate(rows, grid):
    trials = []
    for point in grid:
        positive = ordinary = 0
        for group in measured_groups(rows):
            for case in group["cases"]:
                selected = arms(case, point)["motion"]
                positive += useful(selected, case["truth"])
                ordinary += bool(selected) and not case["truth"]
        if not ordinary and positive:
            trials.append((positive, point))
    return max(trials)[1] if trials else max(grid)


def pair_valid(pair, count):
    a, b = pair
    return 0 <= a[0] < a[1] <= b[0] < b[1] <= count and a[1] - a[0] == b[1] - b[0]


def validate(case, floor):
    count = case["sample_count"]
    require(40320 <= count <= 41344 and (case["truth"] is not None) == (case["role"] in POSITIVE),
            "Invalid source or authored role")
    require(all(SHA256.fullmatch(case[key]) for key in ("sha256", "pcm_sha256", "pts_sha256")),
            "Invalid source commitment")
    require(all(pair_valid(pair, count) for pair in case["similar_audio"]) and all(
        pair in case["similar_audio"] for pair in case["static"]), "Acoustic nomination differs")
    if case["truth"]:
        require(pair_valid(case["truth"], 40320), "Invalid authored repeat range")
    indices = set()
    audited = False
    for item in case["motion"]:
        require(0 <= item["distance"] <= 2 and item["samples"] in case["similar_audio"],
                "Invalid motion nomination")
        require(all(SHA256.fullmatch(h) for h in item["pcm_sha256"]) and len(item["pcm_sha256"]) == 2,
                "Invalid source audio crop commitment")
        frame_pairs = item["frame_pairs"]
        lag = item["samples"][1][0] - item["samples"][0][0]
        require(len(frame_pairs) >= 5 and len(frame_pairs) == len({a for a, _ in frame_pairs})
                and all(0 <= a < b < 63 and b == (640 * a + 320 + lag) // 640 for a, b in frame_pairs),
                "Invalid observed native frame pairs")
        if item["distance"] > floor:
            audited = True
            indices.update(index for pair in frame_pairs for index in pair)
    require(audited == ("native_audit" in case), "Missing native audit")
    if audited:
        anchors = case["anchors"]
        require(sorted(indices) == [anchor["index"] for anchor in anchors] and
                case["native_audit"] == {"timestamps": 63, "samples": count, "crops": len(indices)},
                "Observed native audit is incomplete")
        for anchor in anchors:
            x0, y0, x1, y1 = anchor["native_mouth_xyxy"]
            require(0 <= x0 < x1 <= 480 and 0 <= y0 < y1 <= 360 and (x1 - x0, y1 - y0) == (24, 16)
                    and anchor["pts"] * Fraction(case["time_base"]) == Fraction(anchor["index"], 25)
                    and all(SHA256.fullmatch(anchor[key]) for key in (
                        "rgb_sha256", "native_gray_sha256", "native_mouth_sha256")), "Invalid native crop commitment")


def summarize(groups, carrier, point, gate):
    totals, audit = Counter(), Counter()
    outcomes = {arm: Counter() for arm in ("nominated", "static", "motion", "combined")}
    new_groups = set()
    for group in groups:
        for case in group["cases"]:
            if case["carrier"] != carrier:
                continue
            truth = case["truth"]
            totals.update(cases=1, positive=bool(truth), ordinary=not truth, measured=case["assessed"],
                          current_measured=case["current"]["measured"], current_flags=bool(case["current"]["ranges"]))
            audit.update(case.get("native_audit", {}))
            selected = arms(case, point)
            successes = {arm: useful(pairs, truth) for arm, pairs in selected.items()}
            for arm, pairs in selected.items():
                outcomes[arm].update(useful=successes[arm], misplaced=bool(pairs and truth and not successes[arm]),
                                     ordinary=bool(pairs and not truth), synchronized=bool(pairs and case["role"] in SYNC))
            combined, static = successes["combined"], successes["static"]
            totals.update(added=combined and not static, lost=static and not combined,
                          ordinary_added=not truth and bool(selected["combined"]) and not selected["static"])
            if combined and not static:
                new_groups.add(group["group"])
    checks = {"added": totals["added"] >= gate["added"], "groups": len(new_groups) >= gate["groups"],
              "lost": totals["lost"] <= gate["lost"], "ordinary_added": totals["ordinary_added"] <= gate["ordinary_added"],
              "synchronized": outcomes["combined"]["synchronized"] <= gate["synchronized"],
              "coverage": bool(totals["cases"]) and totals["measured"] / totals["cases"] >= gate["coverage"]}
    return {"totals": dict(totals), "outcomes": {arm: dict(value) for arm, value in outcomes.items()}, "audit": dict(audit),
            "groups_with_gain": len(new_groups), "gate": checks, "passes": all(checks.values())}


def report(receipt):
    require(receipt["schema"] == SCHEMA and receipt["point"] == calibrate(receipt["rows"]["development"], receipt["grid"]),
            "Development selection does not recompute")
    known_groups = set()
    summary = {}
    for split in SPLITS:
        rows = receipt["rows"][split]
        groups = measured_groups(rows)
        absent = rows[0]["outside_shared_duration"]
        geometry = [row["unqualified_geometry"] for row in rows if "unqualified_geometry" in row]
        group_ids = [row["group"] for row in groups] + absent + geometry
        require(len(group_ids) == {"development": 4, "retained": 11, "transfer": 16}[split]
                and len(set(group_ids)) == len(group_ids) and not set(group_ids) & known_groups, "Unexpected group roster")
        known_groups.update(group_ids)
        for group in groups:
            require(SHA256.fullmatch(group["source_sha256"]) and
                    {(case["carrier"], case["role"]) for case in group["cases"]} ==
                    {(carrier, role) for carrier in CARRIERS for role in (*POSITIVE, *CONTROL)}
                    and len(group["cases"]) == 2 * (len(POSITIVE) + len(CONTROL)), "Missing or repeated media roles")
            for case in group["cases"]:
                validate(case, min(receipt["grid"]))
        summary[split] = {carrier: summarize(groups, carrier, receipt["point"], receipt["gate"]) for carrier in CARRIERS}
        summary[split]["roster"] = {"planned": len(group_ids), "complete": len(groups), "duration_declines": len(absent),
                                    "geometry_declines": len(geometry)}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    a = p.parse_args()
    data = json.loads(a.receipt.read_bytes())
    require(data["summary"] == report(data), "Published results differ from the independent tally")
    for split, result in data["summary"].items():
        print(split, "point", data["point"], "roster", result["roster"])
        for carrier in CARRIERS:
            row = result[carrier]
            print(carrier, row["totals"], row["outcomes"], row["audit"], "passes", row["passes"])


if __name__ == "__main__":
    main()
