"""Replay the published clipped-rank writing trial without models or source prose."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.writing-clipped-rank-transfer/1"
METHODS = ("raw_log_rank", "clipped_log_rank", "raw_lrr", "clipped_lrr", "tmr_focus")
FAMILIES = ("granite", "olmo")
ROLES = ("self", "ordinary", *FAMILIES)
COHORTS = ("selection", "held", "diagnostic")
SHA256 = re.compile("[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def useful(row):
    overlap, total, truth = row
    require(all(type(value) is int for value in row) and 0 <= overlap <= truth <= total and truth > 0,
            "Invalid native character overlap")
    return overlap * 2 >= truth and overlap * 10 >= total * 3


def marks(values, point):
    eligible = Counter(value for value in values if value is not None and value > point)
    output = []
    for value in sorted(eligible, reverse=True):
        if eligible[value] > 1 or len(output) == 2:
            break
        output.append(values.index(value))
    return output


def scores(row, caps):
    output = {name: [] for name in METHODS}
    for totals, tmr in zip(row["token_reductions"], row["tmr_scores"], strict=True):
        require(math.isfinite(tmr) and 0 <= tmr <= 1, "Invalid secondary detector response")
        output["tmr_focus"].append(tmr)
        if totals is None:
            for name in METHODS[:-1]:
                output[name].append(None)
            continue
        count, rank, bounded_rank, nll, bounded_nll = totals
        require(type(count) is int and count > 0 and all(math.isfinite(x) for x in totals[1:])
                and 0 <= bounded_rank <= min(rank, count * caps["log_rank"] + 1e-12)
                and 0 <= bounded_nll <= min(nll, count * caps["nll"] + 1e-12), "Invalid causal reductions")
        output["raw_log_rank"].append(-rank / count)
        output["clipped_log_rank"].append(-bounded_rank / count)
        output["raw_lrr"].append(nll / rank if rank else None)
        output["clipped_lrr"].append(bounded_nll / bounded_rank if bounded_rank else None)
    return output


def evaluate(row, values, points):
    native, focus = (list(map(useful, row[key])) for key in ("native_overlaps", "focus_overlaps"))
    require(len(native) == len(focus) == len(row["native_scores"]) == 3
            and all(math.isfinite(x) and 0 <= x <= 1 for x in row["native_scores"])
            and focus == [False, True, False] and len({item[2] for item in row["native_overlaps"] + row["focus_overlaps"]}) == 1,
            "Native focus coverage differs")
    incumbent = [i for i, value in enumerate(row["native_scores"]) if value >= .5][:6]
    generated = row["role"] in FAMILIES
    output = {"current": (bool(incumbent), generated and any(native[i] for i in incumbent))}
    for method in METHODS:
        chosen = marks(values[method], points[method])
        pair = bool(chosen), generated and any(focus[i] for i in chosen)
        output[method] = pair
        output[f"{method}_union"] = tuple(a or b for a, b in zip(pair, output["current"], strict=True))
    return output


def summary(rows):
    counts = {}
    for arm in rows[0]["decisions"]:
        controls = {r["group"] for r in rows if r["role"] in ROLES[:2] and r["decisions"][arm][0]}
        totals = {}
        for role in ROLES:
            subset = [r["decisions"][arm] for r in rows if r["role"] == role]
            flags, hits = (sum(pair[index] for pair in subset) for index in (0, 1))
            totals[role] = {"files": len(subset), "flags": flags, "useful": hits}
            if role in FAMILIES:
                totals[role]["wrong_only"] = flags - hits
        counts[arm] = {"roles": totals, "ordinary_work_flags": len(controls)}
    return counts


def paired(rows):
    by_case = {(r["group"], r["role"]): r for r in rows}
    output = {}
    for name in ("current", *METHODS):
        wins = Counter()
        for row in rows:
            group = row["group"]
            if row["role"] not in FAMILIES or (group, "ordinary") not in by_case:
                continue
            alternatives = (row, by_case[group, "ordinary"], by_case[group, "self"])
            ai, human, original = [r["native_scores"][1] if name == "current" else r["values"][name][1] for r in alternatives]
            if None not in (ai, human, original):
                wins["comparable"] += 1
                wins["above_natural_revision"] += ai > human
                wins["above_both_genuine_versions"] += ai > max(human, original)
        output[name] = dict(wins)
    return output


def replay(packet):
    require(packet["schema"] == SCHEMA, "Unknown writing trial")
    rows = packet["cases"]
    require(len(rows) == len({(r["group"], r["role"]) for r in rows})
            and all(r["cohort"] in COHORTS and r["role"] in ROLES and SHA256.fullmatch(r["input_sha256"]) for r in rows),
            "Invalid case roster")
    roster = {cohort: {r["group"] for r in rows if r["cohort"] == cohort} for cohort in COHORTS}
    require(sum(map(len, roster.values())) == len(set().union(*roster.values()))
            and [len(roster[cohort]) for cohort in COHORTS] == [5, 10, 2], "Held works are not separate")
    sources = packet["sources"]
    require(set(sources) == roster["selection"] | roster["held"]
            and all(item["cohort"] in COHORTS[:2] and group in roster[item["cohort"]]
                    and item["license"] == "https://creativecommons.org/licenses/by/4.0/"
                    and set(item["versions"]) == {"before", "after"}
                    and all(SHA256.fullmatch(version["xml_sha256"]) and SHA256.fullmatch(version["body_block_sha256"])
                            for version in item["versions"].values()) for group, item in sources.items()),
            "Publisher source commitments differ")
    caps = packet["clipping"]
    for name in ("log_rank", "nll"):
        ordered = sorted(caps["selection_tokens"][name])
        require(ordered and all(math.isfinite(value) and value >= 0 for value in ordered)
                and caps["caps"][name] == ordered[math.ceil(.90 * len(ordered)) - 1], "Frozen clipping cap differs")
    require(len(caps["selection_tokens"]["log_rank"]) == len(caps["selection_tokens"]["nll"])
            == caps["selection_token_occurrences"], "Missing paired token reductions")
    for row in rows:
        require(len(row["token_reductions"]) == len(row["tmr_scores"]) == 3 and row["token_reductions"][0] is None,
                "Incomplete native token scope")
    rows = [r | {"values": scores(r, caps["caps"])} for r in rows]
    selection = [r for r in rows if r["cohort"] == "selection" and r["role"] in ROLES[:2]]
    points = {name: sorted(max(value for row in selection if row["group"] == group for value in row["values"][name]
                               if value is not None) for group in roster["selection"])[-2] for name in METHODS}
    require(points == packet["points"], "Frozen selection point differs")
    rows = [r | {"decisions": evaluate(r, r["values"], points)} for r in rows]
    cohorts = {cohort: [r for r in rows if r["cohort"] == cohort] for cohort in COHORTS}
    for cohort in COHORTS[:2]:
        group_ids = {row["group"] for row in cohorts[cohort] if row["role"] == "self"}
        ordinary = {row["group"]: row for row in cohorts[cohort] if row["role"] == "ordinary"}
        require(group_ids == set(ordinary) == roster[cohort], "Genuine revision controls are incomplete")
        require(all(.28 <= row["edit_distance"] <= .85 and row["edit_distance"] == sources[group]["ordinary_edit_distance"]
                    for group, row in ordinary.items()), "Missing substantial genuine revision")
        for row in cohorts[cohort]:
            if row["role"] in FAMILIES:
                control = ordinary[row["group"]]
                require(abs(row["edit_distance"] - control["edit_distance"]) <= .15
                        and abs(row["focus_words"] - control["focus_words"]) <= 3, "Generated comparison is unmatched")
    generation = packet["generation"]
    admitted = {(row["group"], row["family"]): row["accepted"] for row in generation["admission"]}
    seen = {(row["group"], row["role"]) for row in rows if row["cohort"] != "diagnostic" and row["role"] in FAMILIES}
    attempts = generation["settings"]["attempts"]
    require(set(admitted) == {(group, family) for group in sources for family in FAMILIES}
            and len(admitted) == len(generation["admission"]) and generation["attempts"] == attempts * len(admitted)
            and all(value is None or type(value) is int and 0 <= value < attempts for value in admitted.values())
            and {key for key, value in admitted.items() if value is not None} == seen, "Generation receipts differ")
    held = cohorts["held"]
    gains = {}
    existing = {r["group"] for r in held if r["role"] in ROLES[:2] and r["decisions"]["current"][0]}
    for name in METHODS:
        added = [r for r in held if r["decisions"][name][1] and not r["decisions"]["current"][1]]
        newly_flagged = {r["group"] for r in held if r["role"] in ROLES[:2] and r["decisions"][name][0]} - existing
        gains[name] = {"added_cases": len(added), "added_works": len({r["group"] for r in added}),
                       "added_families": sorted({r["role"] for r in added}), "added_ordinary_works": len(newly_flagged)}
    admission = {family: sum(row["role"] == family for row in held) for family in FAMILIES}
    primary = gains["clipped_log_rank"]
    advance = (min(admission.values()) >= 6 and primary["added_works"] >= 3
               and primary["added_families"] == list(FAMILIES) and primary["added_ordinary_works"] <= 1)
    native = packet["native_replay"]
    require(native["case_ranges"] == len(rows) and native["context_occurrences"] == len(rows) * 3
            and native["decisions"] == len(rows) * (1 + 2 * len(METHODS))
            and native["causal_tokens"] == sum(item[0] for row in rows for item in row["token_reductions"] if item)
            and native["publisher_versions"] == 2 * sum(len(roster[c]) for c in COHORTS[:2])
            and native["generation_attempt_bindings"] == generation["attempts"]
            and all(SHA256.fullmatch(digest) for digest in packet["commitments"].values()), "Native replay totals differ")
    return {"counts": {cohort: summary(members) for cohort, members in cohorts.items()},
            "pair_order": {cohort: paired(members) for cohort, members in cohorts.items()},
            "gains": gains, "held_admission": admission, "advances": advance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    packet = json.loads(args.receipt.read_bytes())
    result = replay(packet)
    require(result == packet["summary"], "Published writing trial differs from the measured inputs")
    print(f"Replayed {len(packet['cases'])} cases and {packet['native_replay']['decisions']} arm decisions. Advances: {result['advances']}.")


if __name__ == "__main__":
    main()
