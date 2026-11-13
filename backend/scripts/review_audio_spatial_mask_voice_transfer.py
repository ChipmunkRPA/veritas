"""Recompute held directional-mask findings from anonymous scalar rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

SCHEMA = "chipmunk.audio-spatial-mask-voice/1"
ARMS = ("mono", "earlier_either", "directional")
COUNTS = {"surrey": (6, 20), "bbc": (6, 20), "balanced_target": (8, 28),
          "lateral_target": (8, 28), "second_competitor": (8, 28)}
WINDOWS = ((16000, 48000), (48000, 80000))


def require(test, message):
    if not test:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_digest(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()).hexdigest()


def candidates(row):
    if "scores" in row:
        return row["scores"]
    return [None if view is None else [entry[1] for entry in view] for view in row["reads"]]


def readouts(row):
    views = [view for view in candidates(row) if view is not None]
    direction = [max((view[i] for view in views if view[i] is not None), default=None) for i in range(len(WINDOWS))]
    arms = {**row["baselines"], "directional": direction}
    return {arm: max(((value, -i) for i, value in enumerate(arms[arm]) if value is not None), default=None) for arm in ARMS}


def points(rows):
    table = [readouts(row) for row in rows if row["location"] == "surrey" and row["truth"] is None]
    return {arm: max(row[arm][0] for row in table if row[arm] is not None) for arm in ARMS}


def location_matches(truth, span):
    intersection = min(truth[1], span[1]) - max(truth[0], span[0])
    union = (truth[1] - truth[0]) + (span[1] - span[0]) - intersection
    return 2 * intersection >= truth[1] - truth[0] and 10 * intersection >= 3 * union


def tallies(table, cutoffs):
    summary, cases, groups, background = {}, {}, {}, {}
    for arm in ARMS:
        located, false, background_only = set(), set(), set()
        assessed, genuine_flags, edited_flags = 0, 0, 0
        for row, readings in table:
            response = readings[arm]
            if response is None:
                continue
            assessed += 1
            score, index = response
            if score <= cutoffs[arm]:
                continue
            if row["truth"] is not None:
                edited_flags += 1
                if location_matches(row["truth"], WINDOWS[-index]):
                    located.add(row["case"])
            else:
                false.add(row["group"])
                genuine_flags += 1
                if row["kind"] == "background_only":
                    background_only.add(row["group"])
        summary[arm] = {"assessed": assessed, "unassessed": len(table) - assessed, "useful": len(located),
                        "wrong_location": edited_flags - len(located), "edited_flags": edited_flags,
                        "ordinary_groups": len(false), "ordinary_clips": genuine_flags,
                        "background_groups": len(background_only)}
        cases[arm], groups[arm], background[arm] = located, false, background_only
    for arm in ARMS:
        found = cases[arm] - cases["mono"]
        summary[arm].update(additional_locations=len(found), lost_locations=len(cases["mono"] - cases[arm]),
                            additive_total_locations=len(cases["mono"] | cases[arm]),
                            additional_ordinary_groups=len(groups[arm] - groups["mono"]),
                            additional_background_groups=len(background[arm] - background["mono"]),
                            additional_families=sorted({row["family"] for row, _ in table if row["case"] in found}))
    return summary, cases


def acoustic(rows):
    gain, edit_gain = [], []
    for row in rows:
        if row["truth"] is None:
            continue
        views = [pair for pair in row["candidate_snr_db"] if pair is not None and pair[0] is not None]
        if not views:
            continue
        best = max(views, key=lambda pair: pair[0])
        source, span = row["original_snr_db"]
        gain.append(best[0] - source)
        edit_gain.append(best[1] - span)
    return {"edits_with_candidate": len(gain),
            "median_oracle_full_snr_gain_db": round(statistics.median(gain), 6) if gain else None,
            "median_oracle_edit_snr_gain_db": round(statistics.median(edit_gain), 6) if edit_gain else None}


def replay(rows):
    fixed = points(rows)
    layouts = {}
    additional_groups, lost_groups = set(), set()
    new_families = set()
    for location, (edits, ordinary) in COUNTS.items():
        cohort = [row for row in rows if row["location"] == location]
        table = [(row, readouts(row)) for row in cohort]
        measures, reported = tallies(table, fixed)
        budget_ranks = {arm: {} for arm in ARMS}
        for row, readings in table:
            if row["truth"] is None:
                for arm, response in readings.items():
                    if response is not None:
                        ranks = budget_ranks[arm]
                        ranks[row["group"]] = max(ranks.get(row["group"], -math.inf), response[0])
        ranks = {arm: sorted(by_group.values(), reverse=True) for arm, by_group in budget_ranks.items()}
        equal = {}
        for budget in (0, 1, 2):
            operating = {arm: values[budget] if len(values) > budget else -math.inf for arm, values in ranks.items()}
            equal[str(budget)] = {arm: row["useful"] for arm, row in tallies(table, operating)[0].items()}
        additions = []
        for row, readings in table:
            if row["case"] in reported["directional"] - reported["mono"]:
                additions.append({"case": row["case"], "family": row["family"], "truth_samples": row["truth"],
                                  "reported_samples": list(WINDOWS[-readings["directional"][1]])})
                if location != "surrey":
                    additional_groups.add(row["group"])
                    new_families.add(row["family"])
            if row["case"] in reported["mono"] - reported["directional"] and location != "surrey":
                lost_groups.add(row["group"])
        layouts[location] = {"edits": edits, "ordinary": ordinary, "fixed": measures, "acoustic": acoustic(cohort),
                             "retrospective_equal_groups": equal, "new_ranges": additions}
    held = [value["fixed"]["directional"] for name, value in layouts.items() if name != "surrey"]
    advancement = {"two_additional_groups_in_two_families": len(additional_groups) >= 2 and len(new_families) >= 2,
                   "no_new_background_groups": all(row["additional_background_groups"] == 0 for row in held),
                   "at_most_one_new_ordinary_group_per_layout": all(row["additional_ordinary_groups"] <= 1 for row in held)}
    return {"points": fixed, "layouts": layouts, "additional_held_groups": len(additional_groups),
            "standalone_lost_held_groups": len(lost_groups), "additional_families": sorted(new_families),
            "advancement": advancement}


def validate(rows):
    require([row["case"] for row in rows] == list(range(160)), "Case sequence differs")
    for row in rows:
        require(row["location"] in COUNTS and type(row["group"]) is int and row["group"] >= 0,
                "Source alias or role differs")
        lag = row["lag_modes_samples"]
        require(len(lag) in {0, 2, 3} and lag == sorted(set(lag)) and all(type(x) is int and -22 <= x <= 22 for x in lag)
                and all(b - a >= 4 for a, b in zip(lag, lag[1:])) and len(lag) == len(row["scores"])
                == len(row["candidate_snr_db"]), "Directional geometry differs")
        require(set(row["baselines"]) == set(ARMS) - {"directional"}, "Baseline views differ")
        for view in (*row["baselines"].values(), *row["scores"], row["original_snr_db"], *row["candidate_snr_db"]):
            require(view is None or (len(view) == 2 and all(x is None or type(x) is float and math.isfinite(x) for x in view)),
                    "Invalid numeric view")
        truth = row["truth"]
        require((truth is not None) == (row["kind"] == "synthetic"), "Truth and labels differ")
        if truth is not None:
            require(len(truth) == 2 and all(type(x) is int for x in truth) and 0 <= truth[0] < truth[1] <= 96000
                    and row["family"] in {"stock", "clone", "expressive"}, "Invalid authored edit")
        else:
            require(row["family"] is None and row["kind"] in {"untouched", "ordinary_same_host", "ordinary_other_voice", "background_only"},
                    "Invalid ordinary control")
    for location, expected in COUNTS.items():
        cohort = [row for row in rows if row["location"] == location]
        edits = sum(row["truth"] is not None for row in cohort)
        require((edits, len(cohort) - edits) == expected, "Room counts differ")
    development = {row["group"] for row in rows if row["location"] == "surrey"}
    held = {row["group"] for row in rows if row["location"] != "surrey"}
    require(not development & held, "A foreground group crosses evaluation roles")


def publish(folder, receipt):
    freeze = read(folder / "freeze.json")
    fit = read(folder / "points.json")
    source_audit = read(folder / "source-audit.json")
    files = [folder / f"{role}.json" for role in ("development", "held")]
    development, held = [read(path) for path in files]
    require(fit["development_sha256"] == digest(files[0]) and fit["points"] == points(development["rows"])
            and development["freeze_sha256"] == held["freeze_sha256"] == digest(folder / "freeze.json"),
            "Frozen development or held rows changed")
    require([value["cases"] for value in source_audit.values()] == [26, 134], "Native source audit is incomplete")
    aliases, public = {}, []
    for row in development["rows"] + held["rows"]:
        public.append({key: row[key] for key in ("case", "location", "kind", "truth", "family", "lag_modes_samples",
                                                 "original_snr_db", "candidate_snr_db", "baselines")}
                      | {"group": aliases.setdefault(row["group"], len(aliases)), "scores": candidates(row)})
    validate(public)
    result = {"schema": SCHEMA, "freeze": freeze, "freeze_sha256": digest(folder / "freeze.json"),
              "native_audit": source_audit, "private_ledgers_sha256": [digest(path) for path in files],
              "forward_counts": [development["forwards"], held["forwards"]],
              "rows": public, "rows_sha256": scalar_digest(public), "results": replay(public)}
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--from-run", type=Path)
    args = parser.parse_args()
    if args.from_run is not None:
        publish(args.from_run, args.receipt)
    data = read(args.receipt)
    require(data["schema"] == SCHEMA, "Unsupported directional-mask receipt")
    rows = data["rows"]
    validate(rows)
    require(data["rows_sha256"] == scalar_digest(rows) and data["results"] == replay(rows), "Public outcome replay differs")
    print(f"Reproduced positions and ordinary-group rates from {len(rows)} cases.")


if __name__ == "__main__":
    main()
