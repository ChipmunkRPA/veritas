"""Verify new-room WPE results from the anonymous scalar receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.audio-wpe-fresh-room/1"
ROOMS = ("meeting", "lecture")
LEVELS = (0, -5)
WINDOWS = ((16000, 48000), (48000, 80000))
TRUTH = ((23200, 38560), (55200, 70560))
MONO = 5.356589748054317
OLD_WPE = 7.353627562469017


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(value):
    content = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(content.encode()).hexdigest()


def pick(row, arm):
    views = row["responses"][arm]
    require(len(views) in ({1} if arm == "mono" else {0, 1}), "Invalid view count")
    candidates = []
    for view in views:
        require(len(view) == 2, "Invalid window count")
        for index, score in enumerate(view):
            if score is not None:
                require(type(score) in (float, int) and math.isfinite(score), "Invalid score")
                candidates.append((score, -index))
    return max(candidates, default=None)


def validate(row, role):
    require(set(row) == {"case", "group", "role", "room", "level", "kind", "truth", "family", "responses", "alternate_fft_max_lsb"},
            "Unexpected scalar payload")
    require(row["role"] == role and row["room"] in ROOMS and row["alternate_fft_max_lsb"] in (0, 1),
            "Invalid cohort or alternate FFT audit")
    kind = row["kind"]
    require(kind in ("untouched", "ordinary-0", "ordinary-1", "synthetic-0", "synthetic-1", "background_only"),
            "Unknown control or edit")
    require(row["level"] in LEVELS if kind != "background_only" else row["level"] is None, "Unexpected SNR")
    if kind.startswith("synthetic"):
        require(row["truth"] == [list(TRUTH[int(kind[-1])])] and bool(row["family"]), "Wrong authored edit")
    else:
        require(row["truth"] == [] and row["family"] is None, "Control contains labels")
    require(set(row["responses"]) == {"mono", "wpe"}, "Unexpected score arms")
    pick(row, "mono")
    pick(row, "wpe")


def readout(rows, arm, point):
    results = {x: set() for x in ("useful", "wrong", "ordinary", "background", "unassessed")}
    positions = {}
    for row in rows:
        case = row["case"]
        reading = pick(row, arm)
        if reading is None:
            results["unassessed"].add(case)
            continue
        value, slot = reading
        if value <= point:
            continue
        if not row["truth"]:
            results["background" if row["kind"] == "background_only" else "ordinary"].add(case)
            continue
        x, y = WINDOWS[-slot]
        for a, b in row["truth"]:
            overlap = max(0, min(y, b) - max(x, a))
            if 2 * overlap >= b-a and 10 * overlap >= 3 * (max(y,b)-min(x,a)):
                results["useful"].add(case)
                positions[case] = {"group": row["group"], "level": row["level"], "edit": row["kind"], "family": row["family"],
                                   "room": row["room"], "window": [x, y], "truth": row["truth"]}
                break
        else:
            results["wrong"].add(case)
    return results, positions


def check(receipt):
    lock = receipt["freeze"]
    require(receipt["schema"] == lock["schema"] == SCHEMA and receipt["freeze_sha256"] == digest(lock), "Freeze changed")
    require(lock["windows"] == [list(w) for w in WINDOWS] and lock["levels"] == list(LEVELS)
            and lock["edit_intervals"] == [list(w) for w in TRUTH] and lock["old_mono"] == MONO and lock["old_wpe"] == OLD_WPE,
            "Pinned method changed")
    development, held = (receipt[role]["rows"] for role in ("development", "assessment"))
    for role, rows, n in (("development", development, 24), ("assessment", held, 62)):
        require(len(rows) == n, "Missing source deliveries")
        for row in rows:
            validate(row, role)
        usable = sum(score is not None for row in rows for views in row["responses"].values() for view in views for score in view)
        require(0 < receipt[role]["forwards"] <= usable, "Invalid model count")
    require(sorted(row["case"] for row in development + held) == list(range(86)), "Case sequence differs")
    require(not {row["group"] for row in development} & {row["group"] for row in held}, "Development and assessment overlap")
    groups = Counter()
    fit = defaultdict(list)
    for row in development:
        require(row["kind"] in {"untouched", "ordinary-0"}, "Development contains non-genuine control")
        groups[row["group"]] += 1
        mono, wpe = (pick(row, arm) for arm in ("mono", "wpe"))
        if wpe is not None and (mono is None or mono[0] <= MONO):
            fit[row["group"]].append(wpe[0])
    require(len(fit) == 3 and len(groups) == 3 and set(groups.values()) == {8}, "Development groups incomplete")
    require(len({(row["group"], row["room"], row["level"], row["kind"]) for row in development+held}) == 86, "Repeated render")
    roles = defaultdict(list)
    for row in held:
        roles[row["group"]].append(row)
    require(Counter(len(rows) for rows in roles.values()) == {1: 6, 8: 2, 20: 2}, "Assessment reader count differs")
    for variants in roles.values():
        if len(variants) == 1:
            require(variants[0]["kind"] == "background_only", "Unexpected isolated recording")
            continue
        kinds = {"untouched", "ordinary-0", "ordinary-1", "synthetic-0", "synthetic-1"} if len(variants) == 20 else {"untouched", "ordinary-0"}
        require({(r["room"], r["level"], r["kind"]) for r in variants} ==
                {(room, level, kind) for room in ROOMS for level in LEVELS for kind in kinds}, "Incomplete matched render group")
        families = defaultdict(set)
        for row in variants:
            families[row["kind"]].add(row["family"])
        require(all(len(items) == 1 for items in families.values()), "Synthesis family changes across rooms")
    ranks = sorted((max(values) for values in fit.values()), reverse=True)
    points = {"old_wpe": OLD_WPE, "old_mono": MONO, **{f"budget_{index}": value for index, value in enumerate(ranks)}}
    require(receipt["point"] == {"points": points, "development": digest(receipt["development"]), "freeze": receipt["freeze_sha256"]},
            "Development point receipt differs")
    report = {}
    for cohort in ("pooled", *[f"snr_{x}" for x in LEVELS], *ROOMS, "background"):
        subset = [row for row in held if cohort in ("pooled", row["room"], f"snr_{row['level']}")
                  or cohort == "background" and row["kind"] == "background_only"]
        rows = {row["case"]: row for row in subset}
        mono, _ = readout(subset, "mono", MONO)

        def groups_for(ids, cases=rows):
            return sorted({cases[i]["group"] for i in ids})

        def edits_for(ids, cases=rows):
            return sorted({(cases[i]["group"], cases[i]["kind"]) for i in ids})

        arms = {}
        for name, point in points.items():
            wpe, positions = readout(subset, "wpe", point)
            useful = mono["useful"] | wpe["useful"]
            arms[name] = {"useful_deliveries": len(useful), "positive_groups": groups_for(useful), "unique_edits": edits_for(useful),
                          "ordinary_groups": groups_for(mono["ordinary"] | wpe["ordinary"]),
                          "background_groups": groups_for(mono["background"] | wpe["background"]),
                          "added_deliveries": [positions[i] for i in sorted(wpe["useful"] - mono["useful"])],
                          "wrong_wpe_deliveries": len(wpe["wrong"]), "unassessed_wpe": len(wpe["unassessed"])}
        report[cohort] = {"positive_deliveries": sum(bool(row["truth"]) for row in subset),
                          "positive_groups": len({row["group"] for row in subset if row["truth"]}),
                          "ordinary_groups": len({row["group"] for row in subset if not row["truth"] and row["kind"] != "background_only"}),
                          "background_groups": sum(row["kind"] == "background_only" for row in subset),
                          "mono_useful_deliveries": len(mono["useful"]), "mono_positive_groups": groups_for(mono["useful"]),
                          "mono_unique_edits": edits_for(mono["useful"]), "mono_ordinary_groups": groups_for(mono["ordinary"]),
                          "mono_background_groups": groups_for(mono["background"]), "wrong_mono_deliveries": len(mono["wrong"]),
                          "unassessed_mono": len(mono["unassessed"]), "arms": arms}
    require(json.dumps(report, sort_keys=True) == json.dumps(receipt["summary"], sort_keys=True), "Result differs from scalar replay")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(json.loads(args.receipt.read_bytes())), indent=2))


if __name__ == "__main__":
    main()
