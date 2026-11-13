"""Replay the text-free closed-class and paired-draft comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.writing-draft-grammar-evaluation/1"
HEX = re.compile(r"[0-9a-f]{64}")
FAMILIES = {"journal": {"qwen", "smol2"}, "fresh": {"granite", "olmo"}, "replication": {"granite", "olmo"}}
GRAMMAR = ("lexical", "lower_density", "skeleton")


def require(value, message):
    if not value:
        raise ValueError(message)


def response(value):
    require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1, "Invalid saved response")
    return min(1 - 1e-7, max(1e-7, value))


def paired(pairs):
    if pairs is None:
        return False
    require(len(pairs) == 2 and all(len(head) == 2 for head in pairs), "Invalid paired readings")

    def odds(value):
        bounded = response(value)
        return bounded / (1 - bounded)

    return all(math.log(odds(new) / odds(old)) >= math.log(2) for old, new in pairs)


def exact_span(value):
    require(len(value) == 2 and all(type(n) is int for n in value) and 0 < value[0] < value[1],
            "Invalid half-open source span")


def baseline(row):
    selected = []
    for window in row["windows"]:
        require(HEX.fullmatch(window["sha256"]), "Invalid native input digest")
        hit, union, truth = window["overlap"]
        require(all(type(n) is int for n in (hit, union, truth)) and 0 <= hit <= truth <= union and truth > 0,
                "Invalid source overlap")
        response(window["response"])
        if window["response"] >= .5:
            selected.append(10 * hit >= 3 * union and 2 * hit >= truth)
    prior_pair = paired(row.get("pair"))
    return bool(selected) or prior_pair, any(selected[:6]) or prior_pair


def metrics(row):
    focus = row["feature"]
    require(set(focus) == set(GRAMMAR) | {"source_word_counts"}, "Invalid grammar measurements")
    a, b = focus["source_word_counts"]
    require(type(a) is type(b) is int and a > 0 and b > 0 and 0 <= focus["lexical"] <= 1
            and -1 <= focus["lower_density"] <= 1 and 0 <= focus["skeleton"] <= 1, "Invalid grammar range")
    return focus


def arms(rows, point):
    base = {i: baseline(row) for i, row in enumerate(rows)}
    ordinary = {row["group"] for i, row in enumerate(rows) if row["role"] in ("ordinary", "self") and base[i][0]}

    def compare(mark):
        located, gains, false, by_kind = Counter(), [], set(ordinary), Counter()
        for i, row in enumerate(rows):
            role = row["role"]
            if role in ("ordinary", "self"):
                if mark(row):
                    false.add(row["group"])
                if base[i][0] or mark(row):
                    by_kind[role] += 1
            elif base[i][1] or mark(row):
                located[role] += 1
                if not base[i][1]:
                    gains.append(row)
        return {"useful": sum(located.values()), "useful_by_family": dict(located),
                "added_locations": len(gains), "added_work_groups": len({row["group"] for row in gains}),
                "added_families": sorted({row["role"] for row in gains}), "ordinary_groups": len(false),
                "added_ordinary_groups": len(false - ordinary), "ordinary_by_version": dict(by_kind)}

    def grammar(row, feature, cutoff):
        return row["native_band"] and row["feature"] is not None and metrics(row)[feature] > cutoff

    outcomes = {"incumbent": compare(lambda row: False)}
    outcomes.update({name: compare(lambda row, name=name: grammar(row, name, point[name])) for name in GRAMMAR})
    for name in ("lower_density", "skeleton"):
        available = {row["feature"]["lexical"] for row in rows if row["native_band"] and row["feature"] is not None} | {-1}
        options = (compare(lambda row, p=p: grammar(row, "lexical", p)) for p in available)
        outcomes[name]["lexical_max_useful_same_false_budget"] = max(
            option["useful"] for option in options if option["ordinary_groups"] <= outcomes[name]["ordinary_groups"])
    outcomes["wide"] = compare(lambda row: paired(row["wide"]))
    return outcomes


def validate_cases(rows, scope, groups):
    role_by_group = {}
    for row in rows:
        group, role = row["group"], row["role"]
        require(group in groups and row["publisher"] == groups[group]["publisher"], "Case source metadata changed")
        require(role in {"ordinary", "self"} | FAMILIES[scope], "Unknown role for source cohort")
        roles = role_by_group.setdefault(group, set())
        require(role not in roles, "Duplicate case")
        roles.add(role)
        exact_span(row["old_focus"])
        exact_span(row["focus"])
        require(all(HEX.fullmatch(row[f"{side}_sha256"]) for side in ("earlier", "later")), "Invalid source digests")
        require((row["earlier_sha256"] == row["later_sha256"]) == (role == "self"), "Unchanged control differs")
        require(isinstance(row["windows"], list) and len(row["windows"]) >= 3, "Missing native windows")
        require(row["wide"] is None or role != "self" and len(row["wide_inputs"]) == 2
                and all(HEX.fullmatch(pin) for pin in row["wide_inputs"]), "Invalid extended paired binding")
        if row["wide"] is not None:
            require(row["shape"] in ([2, 2], [1, 2], [2, 1]), "Invalid wider edit geometry")
    require(role_by_group.keys() == groups.keys() and all({"ordinary", "self"} <= r for r in role_by_group.values()),
            "Matched genuine controls changed")


def wide_counts(rows):
    eligible = [row for row in rows if row["wide"] is not None]
    tested = {
        "eligible_by_role": dict(Counter(row["role"] for row in eligible)),
        "flags_by_role": dict(Counter(row["role"] for row in eligible if paired(row["wide"]))),
        "eligible_source_groups": len({row["group"] for row in eligible}),
        "native_same_paragraph_by_role": dict(Counter(row["role"] for row in eligible if row["publisher_native"])),
        "native_same_paragraph_flags": dict(Counter(row["role"] for row in eligible
                                                    if row["publisher_native"] and paired(row["wide"]))),
    }
    tested["generated_gains"] = sum(row["role"] not in ("ordinary", "self") and paired(row["wide"])
                                    and not baseline(row)[1] for row in eligible)
    tested["ordinary_work_groups"] = len({row["group"] for row in rows if row["role"] in ("ordinary", "self")
                                           and (baseline(row)[0] or paired(row["wide"]))})
    return tested


def exact_engine_counts(rows):
    qualified = [row for row in rows if row.get("engine_wide_eligible")]
    require(all(row["shape"] == [2, 2] and row["wide"] is not None for row in qualified),
            "Invalid exact two-sentence app eligibility")
    return wide_counts(qualified)


def replay(data):
    require(data["schema"] == SCHEMA and data["overall_confidence_weight"] == 0, "Unexpected evidence scope")
    groups = {row["group"]: row for row in data["sources"]}
    require(len(groups) == len(data["sources"]), "Source groups overlap")
    splits = {cohort: {group: row for group, row in groups.items() if row["cohort"] == cohort}
              for cohort in ("selection", "journal", "fresh", "replication")}
    require([len(v) for v in splits.values()] == [15, 21, 14, data["replication_intake"]["qualified"]],
            "Frozen source group counts changed")
    selection = data["selection"]
    require(len(selection) == 15 and {row["group"] for row in selection} == splits["selection"].keys(),
            "Selection group roster changed")
    points = {feature: sorted(metrics(row)[feature] for row in selection)[-2] for feature in GRAMMAR}
    held = {}
    for cohort, counts in (("journal", {"ordinary": 21, "self": 21, "qwen": 16, "smol2": 16}),
                           ("fresh", {"ordinary": 14, "self": 14, "granite": 5, "olmo": 3})):
        cases = data["held"][cohort]
        validate_cases(cases, cohort, splits[cohort])
        require(Counter(row["role"] for row in cases) == counts, "Held cohort denominators changed")
        held[cohort] = arms(cases, points) | {"wide_scope": wide_counts(cases), "exact_engine_scope": exact_engine_counts(cases)}
    all_cases = data["held"]["journal"] + data["held"]["fresh"]
    held["combined"] = arms(all_cases, points)
    replication = data["replication"]
    validate_cases(replication, "replication", splits["replication"])
    require(all(row["publisher_native"] and row["shape"] == [2, 2] for row in replication),
            "Replication cases must keep both exact native paragraphs")
    observed = Counter(row["role"] for row in replication)
    require(all(observed[family] == data["replication_generation"].get(f"{family}_admitted", 0)
                and data["replication_generation"][f"{family}_attempts"] == 8 * len(splits["replication"])
                for family in FAMILIES["replication"]), "Replication generation disposition changed")
    require(sum(data["replication_intake"].values()) == 72, "Replication intake denominator changed")
    require(data["replay_counts"]["verified_held_cases"] == len(all_cases)
            and data["replay_counts"]["replication_cases"] == len(replication), "Native replay count differs")
    return {"points": points, "held": held, "replication": wide_counts(replication),
            "replication_exact_engine_scope": exact_engine_counts(replication),
            "replication_by_role": dict(observed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    data = json.loads(args.receipt.read_bytes())
    payload = {key: value for key, value in data.items() if key not in ("integrity_sha256", "summary")}
    integrity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(data["integrity_sha256"] == integrity, "Receipt contents changed")
    result = replay(data)
    require(result == data["summary"], "Stored outcome differs from replay")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
