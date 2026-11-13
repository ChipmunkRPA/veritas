# SPDX-License-Identifier: Apache-2.0
"""Recount the paired soundtrack experiment without media or model libraries."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-soundtrack-reencode/1"
CARRIERS = ("pcm", "aac")
POSITIVE = ("late_short", "early_short", "late_long", "early_long")
SYNCHRONIZED = ("matched_picture_late", "matched_picture_early", "matched_mouth_late", "matched_mouth_early")
ORDINARY = ("reencode", "gain", "lowpass", "global", "picture_only", *SYNCHRONIZED,
            "independent_sound", "wrong_reference")
METHODS = ("exact", "whole_picture", "local_picture")
GATE = {"min_useful_fraction": "1/2", "max_ordinary": 1, "max_protected": 0,
        "min_control_assessed_fraction": "3/4", "min_iou": "3/5", "max_lag_error": 16}
PROTECTED = {*SYNCHRONIZED, "global", "wrong_reference"}
HOP, SR, PICTURES = 640, 16000, 63
HASH = re.compile(r"[a-f0-9]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256((json.dumps(value, indent=2, allow_nan=False) + "\n").encode()).hexdigest()


def truth(group, role):
    short = 16
    length = short if role.endswith("_short") or role in SYNCHRONIZED else 22
    start = 21 + group % 3
    span = [start * HOP, (start + length) * HOP]
    if role not in POSITIVE:
        return None
    displacement = (5 if length == short else 7) * HOP
    return [span, -displacement if role.startswith("early") else displacement]


def iou(left, right):
    overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    return Fraction(overlap, max(left[1], right[1]) - min(left[0], right[0]))


def point(development, carrier):
    peaks = [row["picture"]["local"] for group in development for row in group["cases"]
             if row["carrier"] == carrier and row["role"] == "reencode"]
    require(len(peaks) >= 3, "At least three verified development carriers required")
    return round(max(1.75, max(peaks) + .35), 6)


def actions(row, local_point):
    base = "unassessed"
    if row["sound"]["windows"] >= 8:
        base = "local_shift" if row["sound"]["match"] else "no_local_shift"
    p = row["picture"]
    if base == "local_shift" and not row["evidence"]:
        base = "unassessed"
    return {
        "exact": base if p["same_rgb"] else "unassessed",
        "whole_picture": base if p["q65"] <= 2.5 and p["q90"] <= 4.0 else "unassessed",
        "local_picture": (base if (p["q65"] <= 2.5 and p["q90"] <= 4.0 and p["local"] <= local_point)
                          else "unassessed"),
    }


def validate_case(row, group, local_point):
    require(row["truth"] == truth(group, row["role"]), "Authored span or shift differs")
    require(all(HASH.fullmatch(value) for side in row["sources"] for value in
                (side["sha256"], side["pcm_sha256"], side["rgb_sha256"], side["pts_sha256"])), "Source digest differs")
    for source in row["sources"]:
        require(Fraction(source["relative_start"]) == 0 and source["pictures"] == PICTURES
                and PICTURES * HOP <= source["samples"] <= PICTURES * HOP + 1024, "Source clock outside frozen scope")
    earlier, later = row["sources"]
    require(earlier["samples"] == later["samples"]
            and row["picture"]["same_rgb"] == (earlier["rgb_sha256"] == later["rgb_sha256"]), "Carrier mismatch")
    require(all(0 <= row["picture"][key] <= 255 for key in ("q65", "q90", "local")), "Picture response out of range")
    match = row["sound"]["match"]
    evidence = row["evidence"]
    if match:
        a, b = match["submitted_samples"]
        lag = match["shift_samples"]
        require(match["earlier_samples"] == [a - lag, b - lag] and 1920 <= abs(lag) <= 5760,
                "Shift support differs")
        if evidence:
            require(0 <= a < b <= (PICTURES - 1) * HOP and 0 <= a - lag < b - lag <= (PICTURES - 1) * HOP,
                    "Audio witness falls beyond the picture clocks")
            require([item["samples"] for item in evidence["sound"]] == [[a - lag, b - lag], [a, b]]
                    and all(HASH.fullmatch(item["sha256"]) for item in evidence["sound"]), "Sound witness differs")
            wanted = [max(0, a // HOP - 1), (a + b) // (2 * HOP), (b + HOP - 1) // HOP]
            require([item["frame"] for item in evidence["pictures"]] == wanted, "Native frame witness positions differ")
            for item in evidence["pictures"]:
                require(len(item["paired"]) == 2 and all(
                    HASH.fullmatch(view["sha256"]) and Fraction(view["time_base"]) * view["pts"] == Fraction(item["frame"], 25)
                    for view in item["paired"]), "Native frame witness differs")
        else:
            require(min(a, a - lag) < 0 or max(b, b - lag) > (PICTURES - 1) * HOP, "Eligible witness is missing")
    else:
        require(evidence is None, "Unselected file has witnesses")
    require(row["actions"] == actions(row, local_point), "Recorded action differs")


def summarize(receipt):
    lock = receipt["freeze"]
    require(lock["schema"] == SCHEMA and lock["gate"] == GATE and receipt["freeze_sha256"] == digest(lock),
            "Frozen protocol differs")
    require(receipt["point_sha256"] == digest(receipt["points"]), "Point commitment differs")
    groups = receipt["groups"]
    require([row["group"] for row in groups] == list(range(201, 217)) + list(range(301, 305)), "Planned roster differs")
    require([row["group"] for row in groups if row["bank"] == "development"] == list(range(201, 205)),
            "Development split differs")
    require([row["group"] for row in groups if row["bank"] == "conversational"] == list(range(301, 305)),
            "Conversational split differs")
    require([row["group"] for row in groups if row["bank"] == "expressive"] == list(range(205, 217)),
            "Expressive split differs")
    development = [row for row in groups if row["bank"] == "development"]
    require(receipt["points"]["development_sha256"] == digest(development)
            and receipt["points"]["points"] == {carrier: point(development, carrier) for carrier in CARRIERS},
            "Development points differ")
    require([group["source_sha256"] for group in groups] == lock["qualified_sources"], "Qualified source IDs differ")
    summary = defaultdict(dict)
    for group in groups:
        cases = group["cases"]
        if not group["source_sha256"]:
            require(not cases, "Unqualified source contains outcomes")
            continue
        require([(row["carrier"], row["role"]) for row in cases] ==
                [(c, role) for c in CARRIERS for role in (*POSITIVE, *ORDINARY)], "Delivery census differs")
        for row in cases:
            validate_case(row, group["group"], receipt["points"]["points"][row["carrier"]])
    for bank in ("expressive", "conversational"):
        bank_groups = [row for row in groups if row["bank"] == bank]
        for carrier in CARRIERS:
            cases = [row for group in bank_groups for row in group["cases"] if row["carrier"] == carrier]
            results = {}
            for method in METHODS:
                positive = [row for row in cases if row["truth"]]
                ordinary = [row for row in cases if not row["truth"]]
                correct = []
                for row in positive:
                    if row["actions"][method] == "local_shift":
                        expected, lag = row["truth"]
                        match = row["sound"]["match"]
                        if iou(expected, match["submitted_samples"]) >= Fraction(GATE["min_iou"]) and abs(
                            lag - match["shift_samples"]
                        ) <= GATE["max_lag_error"]:
                            correct.append(row)
                flags = [row for row in ordinary if row["actions"][method] == "local_shift"]
                protected = sum(row["role"] in PROTECTED for row in flags)
                # A wrong reference or actual local picture change should abstain rather than count as assessed.
                ordinary_negatives = [row for row in ordinary if row["role"] in
                                      {"reencode", "gain", "lowpass", "global", "independent_sound"}]
                ordinary_assessed = sum(row["actions"][method] != "unassessed" for row in ordinary_negatives)
                results[method] = {
                    "useful": len(correct), "positive": len(positive), "ordinary": len(ordinary),
                    "planned_positive": len(bank_groups) * len(POSITIVE), "planned_ordinary": len(bank_groups) * len(ORDINARY),
                    "positive_assessed": sum(row["actions"][method] != "unassessed" for row in positive),
                    "misplaced": sum(row["actions"][method] == "local_shift" for row in positive) - len(correct),
                    "ordinary_flags": len(flags), "protected_flags": protected,
                    "ordinary_assessed": ordinary_assessed, "ordinary_applicable": len(ordinary_negatives),
                    "flag_roles": sorted(row["role"] for row in flags),
                    "advances": len(correct) >= Fraction(GATE["min_useful_fraction"]) * len(bank_groups) * len(POSITIVE)
                    and len(flags) <= GATE["max_ordinary"] and protected <= GATE["max_protected"]
                    and ordinary_assessed >= Fraction(GATE["min_control_assessed_fraction"]) * len(ordinary_negatives),
                }
            summary[bank][carrier] = results
    return dict(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    data = json.loads(parser.parse_args().receipt.read_bytes())
    result = summarize(data)
    require(data["summary"] == result, "Stored summary differs")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
