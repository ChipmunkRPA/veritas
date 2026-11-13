"""Replay a paired writing-punctuation evaluation from text-free receipts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.writing-punctuation-stability/1"
HEX = re.compile(r"[a-f0-9]{64}")
GENERATED = {"granite", "olmo", "qwen", "smol2"}
MODES = {"original", "dash-ascii", "dash-em", "quotes-ascii", "apostrophe-curly"}
ARMS = ("current", "guard", "agreement", "agreement_guard")


def require(valid, message):
    if not valid:
        raise ValueError(message)


def flag(reading):
    encoded = reading["score"]
    score = float.fromhex(encoded)
    require(0 <= score <= 1 and score.hex() == encoded and type(reading["complete"]) is bool,
            "Invalid fixed-point response")
    return reading["complete"] and score >= .5


def useful(window):
    intersection, union, truth = window["overlap"]
    require(0 <= intersection <= min(union, truth) and truth <= union, "Invalid useful-location overlap")
    return truth > 0 and intersection * 10 >= 3 * union and intersection * 2 >= truth


def touches(changes, spans):
    return any(start <= change[0] < end for change in changes for start, end in spans)


def check_row(row):
    changes = row["changed"]
    require(set(row["versions"]) == set(changes) == MODES and changes["original"] == []
            and row["versions"]["original"] == row["text_sha256"], "Frozen rendered geometry changed")
    for mode, replacements in changes.items():
        positions = [entry[0] for entry in replacements]
        require(positions == sorted(set(positions)) and all(pos >= 0 for pos in positions), "Invalid glyph positions")
        for _, before, after in replacements:
            valid = ((mode == "dash-ascii" and before in {8211, 8212} and after == 45)
                     or (mode == "dash-em" and before in {45, 8211} and after == 8212)
                     or (mode == "quotes-ascii" and (before, after) in {(8216, 39), (8217, 39), (8220, 34), (8221, 34)})
                     or (mode == "apostrophe-curly" and (before, after) == (39, 8217)))
            require(valid, "A change was outside the frozen punctuation set")
        require(HEX.fullmatch(row["versions"][mode]) and bool(replacements) == (
            row["versions"][mode] != row["text_sha256"]), "Punctuation view identity changed")
    previous = 0
    for window in row["windows"]:
        focus, spans, hashes = (window[key] for key in ("focus", "spans", "hashes"))
        require(len(spans) in {2, 3} and focus in spans and previous <= focus[0] < focus[1]
                and all(0 <= start < end for start, end in spans) and set(hashes) == MODES
                and all(HEX.fullmatch(value) for value in hashes.values()), "Native window shape changed")
        previous = focus[1]
        for mode, digest in hashes.items():
            require((digest != hashes["original"]) == touches(changes[mode], spans), "Glyph and context bindings differ")
        useful(window)


def stable(hashes, scores):
    variations = set(hashes.values()) - {hashes["original"]}
    if not all(scores[digest]["complete"] for digest in variations):
        return True
    return all(flag(scores[digest]) for digest in variations)


def counts(cases, outcome):
    partitions = ("total", *(sorted({case["cohort"] for case in cases})), "changed")
    summary = {part: {arm: Counter() for arm in ARMS} for part in partitions}
    false_groups = defaultdict(set)
    lost_useful = defaultdict(set)
    for case in cases:
        cohort, case_id = case["cohort"], case["case"]
        part = ("total", cohort, "changed") if any(case["changed"].values()) else ("total", cohort)
        generated = case["role"] in GENERATED
        selected = outcome[case_id]
        for arm, windows in selected.items():
            hit = generated and any(useful(window) for window in windows)
            for key in part:
                values = summary[key][arm]
                values["generated"] += generated
                values["ordinary"] += not generated
                values["useful"] += hit
                values["wrong_location_only"] += generated and bool(windows) and not hit
                values["ordinary_flags"] += not generated and bool(windows)
                if windows and not generated:
                    false_groups[key, arm].add((cohort, case["group"]))
            if arm in {"guard", "agreement_guard"}:
                baseline = "current" if arm == "guard" else "agreement"
                if generated and not hit and any(useful(window) for window in selected[baseline]):
                    lost_useful[arm].add(case_id)
    for part in partitions:
        for arm in ARMS:
            summary[part][arm]["ordinary_groups_flagged"] = len(false_groups[part, arm])
    return summary, false_groups, lost_useful


def replay(receipt):
    require(receipt["schema"] == SCHEMA and receipt["point"] == .5, "Fixed review point changed")
    cases = receipt["cases"]
    require(len(cases) == len({row["case"] for row in cases}) == 108
            and len({(row["cohort"], row["group"]) for row in cases}) == 31
            and len({row["text_sha256"] for row in cases}) == 108, "Frozen held roster changed")
    primary, second = (receipt["responses"][arm] for arm in ("primary", "second"))
    actual_primary, actual_second = set(), set()
    decision = {}
    affected_generated, affected_sources, affected_generators = 0, set(), set()
    transitions = Counter()
    changed_contexts = 0
    for row in cases:
        check_row(row)
        if any(row["changed"].values()) and row["role"] in GENERATED:
            affected_generated += 1
            affected_sources.add(row["cohort"])
            affected_generators.add(row["role"])
        windows = row["windows"]
        actual_primary.update(pin for window in windows for pin in window["hashes"].values())
        current = [w for w in windows if flag(primary[w["hashes"]["original"]])][:6]
        actual_second.update(pin for window in current for pin in window["hashes"].values())
        agreement = [w for w in current if flag(second[w["hashes"]["original"]])]
        decision[row["case"]] = {
            "current": current,
            "guard": [w for w in current if stable(w["hashes"], primary)],
            "agreement": agreement,
            "agreement_guard": [w for w in agreement if stable(w["hashes"], primary) and stable(w["hashes"], second)],
        }
        for window in windows:
            hashes = window["hashes"]
            original = hashes["original"]
            if len(set(hashes.values())) > 1:
                changed_contexts += 1
            for mode, digest in hashes.items():
                if digest == original:
                    continue
                before, after = flag(primary[original]), flag(primary[digest])
                if before != after:
                    transitions["primary_positive_to_negative" if before else "primary_negative_to_positive"] += 1
                if window in current:
                    before, after = flag(second[original]), flag(second[digest])
                    if before != after:
                        transitions["second_positive_to_negative" if before else "second_negative_to_positive"] += 1

    require(actual_primary == set(primary) and actual_second == set(second), "Measured input roster changed")
    complete = all(row["complete"] for scores in (primary, second) for row in scores.values())
    summary, ordinary, lost = counts(cases, decision)
    gates = {}
    for arm, baseline in (("guard", "current"), ("agreement_guard", "agreement")):
        cleared = ordinary["total", baseline] - ordinary["total", arm]
        gates[arm] = {"cleared_ordinary_groups": len(cleared), "cleared_cohorts": len({item[0] for item in cleared}),
                      "lost_generated_files": len(lost[arm]), "passes": complete and not lost[arm] and len(cleared) >= 2
                      and len({item[0] for item in cleared}) >= 2 and affected_generated >= 12
                      and len(affected_sources) >= 2 and len(affected_generators) >= 2}
    return {"summary": summary, "gates": gates, "transitions": dict(transitions), "changed_contexts": changed_contexts,
            "affected_generated": affected_generated, "affected_generated_cohorts": len(affected_sources),
            "affected_generated_families": len(affected_generators), "complete": complete,
            "primary_inputs": len(actual_primary), "second_inputs": len(actual_second)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    receipt = json.loads(parser.parse_args().receipt.read_bytes())
    result = replay(receipt)
    require(result == receipt["result"], "Punctuation robustness aggregate changed")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
