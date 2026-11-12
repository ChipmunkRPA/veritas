"""Recompute the source-aligned writing study from the public scalar receipt."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

SCHEMA = "chipmunk.writing-source-aligned-transfer/1"
FAMILIES = ("granite", "olmo")
CONTROLS = ("self", "ordinary")
ROLES = (*CONTROLS, *FAMILIES)
METHODS = ("magnitude", "distributed")
ARMS = ("current", "any_difference", *METHODS, "magnitude_union", "distributed_union")
SHA256 = re.compile("[0-9a-f]{64}")
GROUP = re.compile("[0-9a-f]{16}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def measures(row):
    magnitude = row["magnitude"]
    thirds, kept, words = (row[name] for name in ("changed_thirds", "longest_kept", "destination_words"))
    conserved = row["conserved_content"]
    require(all(type(value) is int for value in (thirds, kept, words)) and 0 <= thirds <= 3
            and 0 <= kept <= words and words > 0 and 0 <= magnitude <= 1 and 0 <= conserved <= 1,
            "Invalid source-aligned measures")
    if magnitude == 0:
        require(thirds == 0 and kept == words, "An identical token sequence has changed operations")
    else:
        require(thirds > 0, "A changed token sequence lacks changed operations")
    return {"magnitude": magnitude,
            "distributed": magnitude * (thirds / 3) * (1 - kept / words) * min(1, conserved / .45)}


def useful(overlap):
    shared, union, truth = overlap
    require(all(type(value) is int for value in overlap) and 0 <= shared <= truth <= union and truth > 0,
            "Invalid location overlap")
    return 10 * shared >= 3 * union and 2 * shared >= truth


def evaluate(case, points, ordinary_magnitude):
    feature = measures(case["features"])
    a, b = case["focus"]
    require(type(a) is type(b) is int and 0 <= a < b and SHA256.fullmatch(case["fixture_sha256"]),
            "Invalid case focus or commitment")
    generated = case["role"] in FAMILIES
    hull = case["destination_hull"]
    overlap = case["destination_overlap"]
    require((hull is None) == (overlap is None) == (feature["magnitude"] == 0), "Edit geometry differs from its score")
    require((case["role"] == "self") == (hull is None), "Self comparison and revised case differ from their roles")
    if hull is not None:
        lo, hi = hull
        require(type(lo) is type(hi) is int and a <= lo <= hi <= b and overlap[0] <= hi - lo
                and overlap[1] == overlap[2] <= b - a,
                "Edit hull or overlap is outside the committed focus")
    located = hull is not None and useful(overlap)
    native = case["native"]
    require(len(native) == 3 and len({item["sha256"] for item in native}) == 3, "Incomplete native context geometry")
    target_sizes = {window["overlap"][2] for window in native}
    require(len(target_sizes) == 1 and (overlap is None or overlap[2] in target_sizes), "Mismatched focus sizes")
    for window in native:
        require(SHA256.fullmatch(window["sha256"]) and math.isfinite(window["score"]) and 0 <= window["score"] <= 1,
                "Invalid native input commitment or response")
        useful(window["overlap"])
        require(window["overlap"][2] <= b - a, "Native context target exceeds the focus")
    current = [item for item in native if item["score"] >= .5][:6]
    selected = {"current": (bool(current), any(useful(item["overlap"]) for item in current)),
                "any_difference": (hull is not None, located)}
    for arm, point in points.items():
        flag = hull is not None and feature[arm] > point
        hit = flag and located
        selected[arm] = flag, hit
        selected[f"{arm}_union"] = bool(current) or flag, selected["current"][1] or hit
    matched = abs(feature["magnitude"] - ordinary_magnitude) <= .15
    return {arm: {"flag": flag, "useful": generated and hit, "wrong_only": generated and flag and not hit,
                  "magnitude_matched_useful": generated and hit and matched}
            for arm, (flag, hit) in selected.items()}


def replay(packet):
    require(packet["schema"] == SCHEMA, "Unknown writing study")
    selection = packet["selection"]
    ordinary = selection["ordinary"]
    require(len(ordinary) == 9 and len({row["group"] for row in ordinary}) == 9
            and all(GROUP.fullmatch(row["group"]) for row in ordinary), "Invalid selection roster")
    points = {arm: sorted(measures(row["features"])[arm] for row in ordinary)[-2] for arm in METHODS}
    require(points == selection["points"], "Selection points differ from the frozen rule")
    cases = packet["cases"]
    rosters = {role: {row["group"] for row in cases if row["role"] == role} for role in ROLES}
    groups = rosters["ordinary"]
    require(len(groups) == 10 and groups == rosters["self"] == set(packet["sources"])
            and all(GROUP.fullmatch(group) for group in groups)
            and not groups.intersection(row["group"] for row in ordinary)
            and all(rosters[family] <= groups for family in FAMILIES)
            and sum(map(len, rosters.values())) == len(cases), "Incomplete or overlapping held roster")
    for source in packet["sources"].values():
        require(source["license"] == "https://creativecommons.org/licenses/by/4.0/"
                and set(source["versions"]) == {"before", "after"}
                and all(SHA256.fullmatch(version["xml_sha256"]) for version in source["versions"].values()),
                "Invalid publisher version commitment")
    measured = {row["group"]: row["features"]["magnitude"] for row in cases if row["role"] == "ordinary"}
    decisions = [(row, evaluate(row, points, measured[row["group"]])) for row in cases]
    counts, ordinary_groups = {}, {}
    for arm in ARMS:
        ordinary_groups[arm] = {row["group"] for row, result in decisions if row["role"] in CONTROLS and result[arm]["flag"]}
        counts[arm] = {"ordinary_group_flags": len(ordinary_groups[arm])}
        for role in ROLES:
            results = [result[arm] for row, result in decisions if row["role"] == role]
            counts[arm][role] = {"files": len(results), **{
                name: sum(result[name] for result in results)
                for name in ("flag", "useful", "wrong_only", "magnitude_matched_useful")
            }}
    gained = [row for row, result in decisions if result["distributed"]["useful"] and not result["current"]["useful"]]
    families = sorted({row["role"] for row in gained})
    additions = len(ordinary_groups["distributed"] - ordinary_groups["current"])
    lost = sum(result["current"]["useful"] and not result["distributed_union"]["useful"] for _, result in decisions)
    gates = {"minimum_per_family": all(len(rosters[family]) >= 6 for family in FAMILIES),
             "gained_locations": len(gained), "gained_works": len({row["group"] for row in gained}),
             "gained_families": families, "added_ordinary_groups": additions, "lost_current_locations": lost}
    gates["advances"] = (gates["minimum_per_family"] and gates["gained_works"] >= 2
                         and families == list(FAMILIES) and additions <= 1 and not lost)
    matched = [row for row in cases if row["role"] in FAMILIES
               and abs(row["features"]["magnitude"] - measured[row["group"]]) <= .15]
    matches = [{"group": row["group"], "family": row["role"],
                "ordinary_magnitude": measured[row["group"]], "generated_magnitude": row["features"]["magnitude"]}
               for row in matched]
    stress = packet["stress"]
    stress_groups = {row["group"] for row in stress["cases"]}
    require(len(stress_groups) == len(stress["cases"]) <= stress["rights_qualified_groups"]
            and not stress_groups.intersection(row["group"] for row in ordinary), "Diagnostic works are not distinct")
    for row in stress["cases"]:
        require(row["group"] not in groups and measures(row["features"])["magnitude"] >= .45,
                "Ineligible diagnostic revision")
    stress_flags = {arm: sum(measures(row["features"])[arm] > points[arm] for row in stress["cases"]) for arm in METHODS}
    occurrences = [window["sha256"] for row in cases for window in row["native"]]
    intake, generation, baseline = (packet[key] for key in ("intake", "generation", "baseline"))
    require(intake["catalog_pairs"] == sum(intake["excluded"].values()) + len(groups)
            and generation["admitted"] == {family: len(rosters[family]) for family in FAMILIES}
            and generation["attempts"] == generation["settings"]["attempts"] * len(groups) * len(FAMILIES)
            and baseline["unique_inputs"] == len(set(occurrences)), "Intake or inference denominators differ")
    native = packet["native_replay"]
    require(native["publisher_versions"] == 2 * len(groups)
            and native["source_files_pinned"] == 2 * (intake["catalog_pairs"] - intake["excluded"]["previously_scored_group"])
            and native["case_ranges"] == len(cases) and native["native_context_occurrences"] == len(occurrences)
            and native["generation_attempts"] == generation["attempts"] and native["arm_decisions"] == len(decisions) * len(ARMS)
            and packet["diagnostic_native_replay"]["exact_version_positions"] == 2 * len(stress_groups)
            and all(SHA256.fullmatch(value) for value in packet["commitments"].values()), "Native replay commitments differ")
    return {"counts": counts, "gates": gates, "magnitude_matched_generated": matches, "stress_flags": stress_flags,
            "audited_cases": len(cases), "audited_arm_decisions": len(cases) * len(ARMS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    packet = json.loads(args.receipt.read_bytes())
    result = replay(packet)
    require(result == packet["summary"], "Published writing results differ from the measured scalars")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
