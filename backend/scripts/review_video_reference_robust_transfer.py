# SPDX-License-Identifier: Apache-2.0
"""Recompute paired video time and extent decisions from a scalar ledger."""

import argparse
import json
import re
from fractions import Fraction
from pathlib import Path
from statistics import median

ARMS = ("baseline", "bounded_region", "robust_tone", "combined")
EDIT_ROLES = ("large", "small", "local_texture")
NEUTRAL_ROLES = ("compression", "global")
NOT_ADMITTED = ("unverified_photometric_support", "insufficient_shared_picture_content")
SHA = re.compile(r"[a-f0-9]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def measure(row, arm):
    entry = row["arms"][arm]
    frames, truth = entry.get("frames"), row["truth"]
    require(entry["status"] in (*NOT_ADMITTED, "no_local_range", "no_complete_flanks", "local_difference")
            and (frames is not None) == (entry["status"] == "local_difference"), "Invalid comparison status")
    if frames is None:
        return False, 0.
    start, stop = frames
    require(stop - start in {8, 12, 16, 20} and 5 <= start < stop <= row["frame_count"] - 5
            and entry["score"] > 1.5712549522005277 and entry["background65"] <= 3., "Invalid selected interval")
    expected = {"before": start - 1, "during": (start + stop) // 2, "after": stop}
    require(entry["anchors"].keys() == expected.keys(), "Missing before/during/after pictures")
    for role, index in expected.items():
        sample = entry["anchors"][role]
        require(sample["frame"] == index and Fraction(sample["time"]) == Fraction(index, 24)
                and len(sample["rgb_sha256"]) == 2 and all(SHA.fullmatch(v) for v in sample["rgb_sha256"]),
                "Anchor does not match the delivery clock")
    for basis in ("tile", "extent"):
        x, y, w, h = (entry[basis][key] for key in ("x", "y", "w", "h"))
        require(min(x, y) >= 0 and min(w, h) > 0 and x + w <= 720 and y + h <= 480, "Bad picture rectangle")
    if truth is None:
        return False, 0.
    left, right = truth
    time = 2 * (min(stop, right) - max(start, left)) >= max(stop, right) - min(start, left)
    area = entry["extent"]["w"] * entry["extent"]["h"]
    overlap, changed = entry["extent_overlap"], entry["changed_pixels"]
    require(0 <= overlap <= min(area, changed) and area + changed > overlap, "Impossible mask counts")
    return time, overlap / (area + changed - overlap)


def summarize(data):
    require(data["schema"] == "chipmunk.paired-video-robust-transfer/1", "Unknown ledger")
    rows = data["rows"]
    roles = (*EDIT_ROLES, *NEUTRAL_ROLES, "wrong")
    require(len(rows) == 48 and {(row["source"], row["role"]) for row in rows} ==
            {(source, role) for source in range(8) for role in roles}, "Unexpected delivery roster")
    values = {arm: {} for arm in ARMS}
    for row in rows:
        require(row["frame_count"] == 52 and row["arms"].keys() == set(ARMS)
                and len(row["files_sha256"]) == 2 and all(SHA.fullmatch(v) for v in row["files_sha256"])
                and SHA.fullmatch(row["clock_sha256"])
                and bool(row["truth"]) == (row["role"] in EDIT_ROLES), "Source coverage differs")
        for arm in ARMS:
            values[arm][row["source"], row["role"]] = measure(row, arm)
    results = {}
    old_useful = {key for key, (time, iou) in values["baseline"].items() if time and iou >= .1}
    for arm in ARMS:
        measured = values[arm]
        useful = {key for key, (time, iou) in measured.items() if time and iou >= .1}
        stats = {}
        for role in EDIT_ROLES:
            stats[role] = {"useful": sum(key[1] == role for key in useful),
                           "timely": sum(time for (_, name), (time, _) in measured.items() if name == role),
                           "median_iou": median(iou for (_, name), (_, iou) in measured.items() if name == role)}
        false_flags = {role: sum(row["arms"][arm]["status"] == "local_difference" for row in rows if row["role"] == role)
                       for role in NEUTRAL_ROLES}
        stats["neutral_flags"] = false_flags
        stats["neutral_assessed"] = {role: sum(row["arms"][arm]["status"] not in NOT_ADMITTED
                                              for row in rows if row["role"] == role) for role in NEUTRAL_ROLES}
        stats["wrong_source_admissions"] = sum(row["arms"][arm]["status"] not in NOT_ADMITTED for row in rows
                                               if row["role"] == "wrong")
        stats["added_locations"] = len(useful - old_useful)
        stats["lost_locations"] = len(old_useful - useful)
        results[arm] = stats
    old = results["baseline"]
    for arm, stats in results.items():
        stats["advances"] = (arm != "baseline" and stats["large"]["useful"] >= 7 and stats["small"]["useful"] >= 7
                             and stats["local_texture"]["useful"] >= 5 and stats["lost_locations"] <= 1
                             and stats["wrong_source_admissions"] == 0
                             and sum(stats["neutral_flags"].values()) <= sum(old["neutral_flags"].values()) + 1
                             and (stats["added_locations"] - stats["lost_locations"] >= 2 or any(
                                 stats[role]["median_iou"] - old[role]["median_iou"] >= .1 for role in EDIT_ROLES)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(json.loads(args.ledger.read_bytes())), sort_keys=True))


if __name__ == "__main__":
    main()
