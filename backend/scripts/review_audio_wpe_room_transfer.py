"""Verify the WPE room test from its anonymous scalar receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

SCHEMA = "chipmunk.audio-wpe-room/1"
WINDOWS = ((16000, 48000), (48000, 80000))
CLEAN = ("dry", "early_room")
STAGES = (*CLEAN, "full_room", "snr_5", "snr_0", "snr_-5", "snr_-10", "background_only")


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(value):
    body = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(body.encode()).hexdigest()


def decisions(rows, arm, point):
    flags = {key: set() for key in ("located", "ordinary", "background", "wrong", "unassessed")}
    locations = {}
    for row in rows:
        group = row["group"]
        tracks = row["responses"][arm]
        choices = [(score, -i) for track in tracks for i, score in enumerate(track) if score is not None]
        if not choices:
            flags["unassessed"].add(group)
            continue
        score, index = max(choices)
        if score <= point:
            continue
        if not row["truth"]:
            flags["ordinary"].add(group)
            if row["kind"] == "background_only":
                flags["background"].add(group)
            continue
        start, end = WINDOWS[-index]
        for a, b in row["truth"]:
            overlap = max(0, min(end, b) - max(start, a))
            if overlap * 2 >= b - a and overlap * 10 >= (max(b, end) - min(a, start)) * 3:
                flags["located"].add(group)
                locations[group] = {"family": row["family"], "room": row["room"], "window": [start, end], "truth": row["truth"]}
                break
        else:
            flags["wrong"].add(group)
    return flags, locations


def check_scalar(row, role):
    require(set(row) == {"case", "group", "room", "role", "kind", "stage", "truth", "family", "acoustic", "responses"},
            "Unexpected public response payload")
    require(row["role"] == role and row["stage"] in STAGES and row["room"] in {"stairway", "aula_carolina"}, "Invalid cohort")
    require(row["kind"] in {"genuine", "publisher_edit", "background_only"}
            and bool(row["truth"]) == bool(row["family"]) == (row["kind"] == "publisher_edit")
            and (row["kind"] == "background_only") == (row["stage"] == "background_only"), "Invalid edit status")
    require(all(type(a) is int and type(b) is int and 0 <= a < b <= 96000 for a, b in row["truth"]), "Invalid edit range")
    require(set(row["responses"]) == {"wpe", "mono"}, "Unexpected voice response")
    for arm, tracks in row["responses"].items():
        if arm == "mono" and role == "development":
            require(tracks is None, "Unneeded development mono response")
        else:
            require(isinstance(tracks, list) and len(tracks) in ({1} if arm == "mono" else {0, 1})
                    and all(len(track) == 2 and all(x is None or type(x) in (int, float) and math.isfinite(x) for x in track)
                            for track in tracks), "Invalid window responses")
    acoustic = row["acoustic"]
    if acoustic is not None:
        require(len(acoustic["edit_db"]) == len(row["truth"]) > 0
                and all(math.isfinite(x) for x in [acoustic["full_db"], *acoustic["edit_db"]]), "Invalid acoustic values")


def check(receipt, ancestor=None):
    freeze = receipt["freeze"]
    require(receipt["schema"] == freeze["schema"] == SCHEMA, "Unexpected receipt schema")
    require(receipt["freeze_sha256"] == digest(freeze), "Freeze changed")
    require(freeze["windows"] == [list(x) for x in WINDOWS] and freeze["levels_db"] == [5, 0, -5, -10]
            and freeze["prior_points"]["mono"] == 5.356589748054317, "Native window or original point changed")
    require(freeze["alternate_fft_cases"] == 151 and 0 <= freeze["alternate_fft_exact_cases"] <= 151
            and freeze["alternate_fft_max_lsb"] in (0, 1), "Invalid alternate FFT audit")
    fit, assessment = (receipt[key]["rows"] for key in ("development", "assessment"))
    for role, rows, count in (("development", fit, 40), ("assessment", assessment, 111)):
        require(len(rows) == count and len({(row["group"], row["stage"]) for row in rows}) == count, "Missing paired case")
        for row in rows:
            check_scalar(row, role)
        visible = sum(x is not None for row in rows for view in row["responses"]["wpe"] for x in view)
        require(0 < receipt[role]["forwards"] <= visible, "Invalid model calls")
    require(len({r["case"] for r in [*fit, *assessment]}) == 151, "Repeated case")
    groups = Counter(row["group"] for row in fit)
    require(len(groups) == 8 and set(groups.values()) == {5} and not groups.keys() & {row["group"] for row in assessment}
            and all(row["kind"] == "genuine" and row["stage"] in STAGES[2:-1] for row in fit), "Speaker roles overlap")
    eligible = [(row["group"], x) for row in fit for track in row["responses"]["wpe"] for x in track if x is not None]
    require({g for g, _ in eligible} == groups.keys(), "Incomplete development speech")
    point = max(x for _, x in eligible)
    require(receipt["point"] == {"point": point, "development_sha256": digest(receipt["development"]),
                                 "freeze_sha256": receipt["freeze_sha256"]}, "Development fit changed")
    cohort_groups = defaultdict(list)
    for row in assessment:
        cohort_groups[row["group"]].append(row)
    for rows in cohort_groups.values():
        reference = rows[0]
        expected = {"background_only"} if reference["kind"] == "background_only" else set(STAGES[:-1])
        require({row["stage"] for row in rows} == expected and
                all(all(row[key] == reference[key] for key in ("truth", "room", "family", "kind")) for row in rows),
                "Paired source changes room, role or truth")
    if ancestor is not None:
        require(digest(ancestor) == freeze["ancestor_public"] and ancestor["freeze_sha256"] == freeze["ancestor_freeze"]
                and ancestor["private_roster_sha256"] == freeze["ancestor_roster"], "Prior mono comparison changed")
        by_case = {row["case"]: row for row in ancestor["assessment"]["rows"]}
        require(len(by_case) == len(assessment), "Prior mono comparison is incomplete")
        for row in assessment:
            earlier = by_case[row["case"]]
            require(earlier["responses"]["mono"] == row["responses"]["mono"] and
                    all(row[key] == earlier[key] for key in ("truth", "room", "family", "kind", "group", "stage")),
                    "Prior mono window or exact source binding changed")
    result, advancement = {}, {}
    for profile, wpe_point in (("fitted", point), ("prior_mono", freeze["prior_points"]["mono"])):
        additions, false, dry_false, background = {}, set(), set(), set()
        stages = {}
        for stage in STAGES:
            cohort = [row for row in assessment if row["stage"] == stage]
            kinds = Counter(row["kind"] for row in cohort)
            require(kinds == ({"background_only": 6} if stage == "background_only" else {"publisher_edit": 7, "genuine": 8}),
                    "Stage roles differ")
            mono, mono_ranges = decisions(cohort, "mono", freeze["prior_points"]["mono"])
            wpe, wpe_ranges = decisions(cohort, "wpe", wpe_point)
            extra = wpe["ordinary"] - mono["ordinary"]
            gained = {group: wpe_ranges[group] for group in wpe["located"] - mono["located"]}
            stages[stage] = {"mono": {key: sorted(value) for key, value in mono.items()}, "mono_ranges": mono_ranges,
                             "wpe": {key: sorted(value) for key, value in wpe.items()}, "added_ordinary": sorted(extra),
                             "added_locations": gained}
            if stage in CLEAN:
                dry_false.update(extra)
            else:
                false.update(extra)
                additions.update(gained)
            background.update(wpe["background"] - mono["background"])
        families = {location["family"] for location in additions.values()}
        advancement[profile] = {"unique_edit_groups": len(additions), "generator_families": sorted(families),
                                "added_ordinary": sorted(false - background), "added_background": sorted(background),
                                "added_dry_or_early": sorted(dry_false),
                                "passes": len(additions) >= 2 and len(families) >= 2 and len(false) <= 1
                                and not background and not dry_false}
        result[profile] = stages
    acoustics = {}
    for stage in STAGES[:-1]:
        values = [row["acoustic"] for row in assessment if row["stage"] == stage and row["acoustic"] is not None]
        acoustics[stage] = {"clips": len(values),
                            "median_full_db": median(v["full_db"] for v in values) if values else None,
                            "median_edit_db": median(median(v["edit_db"]) for v in values) if values else None}
    summary = {"conditions": result, "advancement": advancement, "acoustics": acoustics}
    require(summary == receipt["summary"], "Receipt results differ from scalar decisions")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--ancestor", type=Path)
    args = parser.parse_args()
    ancestor = json.loads(args.ancestor.read_bytes()) if args.ancestor else None
    print(json.dumps(check(json.loads(args.receipt.read_bytes()), ancestor), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
