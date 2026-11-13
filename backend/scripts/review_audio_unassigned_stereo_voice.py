"""Recount the published unassigned-stereo audio comparison using only Python."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

METHODS = ("original", "balanced", "either", "both")
ROOMS = {"surrey": (6, 20), "bbc": (6, 20), "balanced_target": (8, 28),
         "lateral_target": (8, 28), "second_competitor": (8, 28)}
SPANS = ((16000, 48000), (48000, 80000))


def scalar_commitment(rows):
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def check(condition, problem):
    if not condition:
        raise ValueError(problem)


def readout(row, common):
    views = [[None, None] if values is None else values for values in row["scores"]]
    result = {}
    for index, (original, a, b) in enumerate(zip(*views, strict=True)):
        candidates = [score for score in (a, b) if score is not None]
        if common and (original is None or not candidates):
            continue
        routed = views[1 + row["route"]][index] if row["route"] is not None else None
        scores = [original, routed, max(candidates) if candidates else None,
                  min(candidates) if len(candidates) == 2 else None]
        for method, score in zip(METHODS, scores, strict=True):
            if score is not None and (method not in result or score > result[method][0]):
                result[method] = score, SPANS[index]
    return result


def location_matches(truth, span):
    intersection = min(truth[1], span[1]) - max(truth[0], span[0])
    union = (truth[1] - truth[0]) + (span[1] - span[0]) - intersection
    return 2 * intersection >= truth[1] - truth[0] and 10 * intersection >= 3 * union


def counts(table, points):
    tallies = {}
    references = {}
    for method in METHODS:
        passed, ordinary, room_only = set(), set(), set()
        true_flags, wrong, false, assessed = 0, 0, 0, 0
        for row, arms in table:
            if method not in arms:
                continue
            assessed += 1
            score, span = arms[method]
            if score <= points[method]:
                continue
            if row["truth"] is None:
                false += 1
                ordinary.add(row["group"])
                if row["kind"] == "background_only":
                    room_only.add(row["group"])
            else:
                true_flags += 1
                if location_matches(row["truth"], span):
                    passed.add(row["case"])
                else:
                    wrong += 1
        references[method] = (passed, ordinary, room_only)
        tallies[method] = {"assessed": assessed, "unassessed": len(table) - assessed,
                           "AI flags": true_flags, "useful_locations": len(passed), "wrong_locations": wrong,
                           "false_clips": false, "false_groups": len(ordinary), "background_flags": len(room_only),
                           "inconclusive": assessed - false - true_flags}
    incumbent, false_incumbent, background_incumbent = references["original"]
    for method, (passed, ordinary, room_only) in references.items():
        gained = passed - incumbent
        tallies[method].update(new_locations=len(gained), lost_locations=len(incumbent - passed),
                               new_false_groups=len(ordinary - false_incumbent),
                               new_background_flags=len(room_only - background_incumbent),
                               new_families=sorted({row["family"] for row, _ in table if row["case"] in gained}))
    return tallies


def replay(rows):
    summaries = {}
    for common in (False, True):
        tables = {name: [(row, readout(row, common)) for row in rows if row["location"] == name] for name in ROOMS}
        selection = [readings for row, readings in tables.pop("surrey") if row["truth"] is None]
        points = {method: max(arms[method][0] for arms in selection if method in arms) for method in METHODS}
        layouts = {}
        for name, table in tables.items():
            equal = {}
            groups = {method: {} for method in METHODS}
            for row, arms in table:
                if row["truth"] is None:
                    for method, (score, _) in arms.items():
                        prior = groups[method].get(row["group"], -math.inf)
                        groups[method][row["group"]] = max(prior, score)
            ranks = {method: sorted(values.values(), reverse=True) for method, values in groups.items()}
            for budget in (0, 1, 2):
                caps = {method: values[budget] if budget < len(values) else -math.inf for method, values in ranks.items()}
                equal[str(budget)] = {method: row["useful_locations"] for method, row in counts(table, caps).items()}
            edits, ordinary = ROOMS[name]
            layouts[name] = {"edited": edits, "ordinary": ordinary,
                             "fixed": counts(table, points), "equal_false_group": equal}
        summaries["shared" if common else "all_admitted"] = {"points": points, "trials": layouts}
    return summaries


def validate(rows):
    check([row["case"] for row in rows] == list(range(sum(sum(counts) for counts in ROOMS.values()))),
          "Case aliases differ")
    for row in rows:
        check(set(row) == {"case", "location", "group", "kind", "family", "truth", "route", "scores"}
              and type(row["group"]) is int and row["group"] >= 0 and row["location"] in ROOMS
              and (row["route"] is None or type(row["route"]) is int and row["route"] in (0, 1))
              and len(row["scores"]) == 3, "Invalid case metadata")
        check(row["scores"][0] is not None, "Missing original view")
        for view in row["scores"]:
            check(view is None or (len(view) == 2 and all(x is None or (type(x) is float and math.isfinite(x))
                                                         for x in view)), "Invalid window readings")
        truth = row["truth"]
        check((truth is not None) == (row["kind"] == "synthetic"), "Invalid edit label")
        if truth is not None:
            check(len(truth) == 2 and 0 <= truth[0] < truth[1] <= 96000
                  and row["family"] in {"stock", "clone", "expressive"}, "Invalid edit coordinates")
        else:
            check(row["family"] is None and row["kind"] in {"untouched", "ordinary_same_host",
                                                           "ordinary_other_voice", "background_only"},
                  "Invalid ordinary role")
    role_sets = {name: {row["group"] for row in rows if row["location"] == name} for name in ROOMS}
    check(not role_sets["surrey"] & role_sets["bbc"]
          and all(not role_sets["surrey"] & role_sets[name] and not role_sets["bbc"] & role_sets[name]
                  for name in ("balanced_target", "lateral_target", "second_competitor")),
          "A source group crosses separated panels")
    for name, (edits, ordinary) in ROOMS.items():
        labels = [row["truth"] is not None for row in rows if row["location"] == name]
        check(sum(labels) == edits and len(labels) - sum(labels) == ordinary, "A room's case counts differ")


def publish(folder, receipt):
    result = json.loads((folder / "result.json").read_bytes())
    ledger = (folder / "private-windows.jsonl").read_bytes()
    freeze, *entries = ledger.splitlines()
    check(hashlib.sha256(ledger).hexdigest() == result["private_ledger_sha256"]
          and freeze.decode() == result["freeze_sha256"], "Private audit inputs differ")
    aliases, rows = {}, []
    for entry in entries:
        private = json.loads(entry)
        group = private["group"]
        scores = [[entry[1] for entry in view] if view is not None else None for view in private["reads"]]
        rows.append({key: private[key] for key in ("case", "location", "kind", "family", "truth", "route")}
                    | {"group": aliases.setdefault(group, len(aliases)), "scores": scores})
    validate(rows)
    check(replay(rows) == result["summary"], "Private and independent scalar decisions differ")
    result.update(scalar_replay_sha256=scalar_commitment(rows), scalar_replay_rows=rows)
    with receipt.open("x") as output:
        json.dump(result, output, allow_nan=False, indent=2, sort_keys=True)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--from-run", type=Path, help="Publish anonymized scalar rows from a completed native audit")
    args = parser.parse_args()
    if args.from_run is not None:
        publish(args.from_run, args.receipt)
    source = json.loads(args.receipt.read_bytes())
    check(source["schema"] == "chipmunk.audio-unassigned-stereo-voice/1", "Unsupported receipt")
    rows = source["scalar_replay_rows"]
    check(scalar_commitment(rows) == source["scalar_replay_sha256"], "Scalar ledger commitment differs")
    validate(rows)
    check(replay(rows) == source["summary"], "Published decisions differ")
    print(f"Reproduced fixed and equal-false-group counts from {len(rows)} cases.")


if __name__ == "__main__":
    main()
