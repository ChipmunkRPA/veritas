"""Independently replay prospective grouped audio points and held localization."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from . import review_audio_edge_window_transfer as prior
else:
    import review_audio_edge_window_transfer as prior

ROOT = prior.ROOT
RECEIPT = ROOT / "docs/evaluation-panels/audio-edge-window-grouped-2026-11-12.json"
SCHEMA = "chipmunk.audio-edge-window-grouped/1"
ARMS = ("full", "combined")
GRIDS = {arm: prior.GRIDS[arm] for arm in ARMS}
BUDGETS = (2, 1)
read, sha, check = prior.read, prior.sha, prior.check


def independently_fit(rows):
    groups = prior.groups(rows, held=False)
    points = {}
    for arm, grid in GRIDS.items():
        maximums = [max(prior.strongest(row, grid)[0] for row in group) for group in groups.values()]
        descending = sorted(maximums, reverse=True)
        points[arm] = {str(b): {"point": descending[b], "calibration_false_groups": sum(p > descending[b] for p in maximums)}
                       for b in BUDGETS}
    return points


def verify_held(rows):
    check(len(rows) == 50, "Held file denominator changed")
    groups = defaultdict(list)
    for row in rows:
        check(set(row) == {"arm", "group", "family", "coverage", "truth", "scores", "snr_db", "room_group"}
              and type(row["group"]) is int and row["room_group"] in {1, 2}
              and len(row["scores"]) == len(prior.GRID)
              and all(v is None or type(v) in {int, float} and math.isfinite(v) for v in row["scores"]),
              "Invalid held window or anonymous row")
        groups[row["group"]].append(row)
        truth = row["truth"]
        check((truth is None and row["arm"] in prior.CONTROLS | {"background_only"})
              or (row["arm"] == "synthetic" and type(truth) is list and len(truth) == 2 and all(type(v) is int for v in truth)
                  and 0 <= truth[0] < truth[1] <= 96000 and truth[1] - truth[0] in {10240, 12800, 15360}),
              "Unexpected edit range or ordinary label")
    check(set(groups) == set(range(1, 15)), "Held group denominator changed")
    for group in groups.values():
        reference = group[0]
        check(all(row[key] == reference[key] for row in group for key in ("family", "coverage", "snr_db", "room_group")),
              "Matched rows have different source conditions")
        arms = {row["arm"] for row in group}
        check((len(group) == 4 and arms == prior.CONTROLS | {"synthetic"} and reference["snr_db"] in {-5, -8}
               and reference["coverage"] in {"early", "interior", "late"})
              or (len(group) == 1 and arms == {"background_only"} and reference["snr_db"] is None
                  and reference["coverage"] == "room" and reference["family"] is None), "Missing matched genuine controls")
    edited = [row for row in rows if row["truth"]]
    check(Counter((row["family"], row["snr_db"]) for row in edited)
          == dict.fromkeys(((family, snr) for family in ("clone", "expressive", "stock") for snr in (-5, -8)), 2)
          and Counter(row["coverage"] for row in edited) == {"early": 2, "interior": 8, "late": 2}
          and Counter(row["room_group"] for row in rows) == {1: 25, 2: 25}, "Held scenario balance changed")
    return edited


def summarize(rows, points):
    edited = verify_held(rows)
    reports = {}
    for arm, grid in GRIDS.items():
        groups = {}
        for budget in BUDGETS:
            point = points[arm][str(budget)]["point"]
            result = prior.measure(rows, grid, point)
            groups[str(budget)] = {
                "total": result | {
                    "unflagged_generated_edits": len(edited) - result["generated_flags"],
                    "generated_flags_without_useful_location": result["generated_flags"] - result["useful_locations"],
                },
                "by_noise": {str(snr): prior.measure([row for row in rows if row["snr_db"] == snr], grid, point)
                             for snr in (-5, -8)},
                "by_room": {str(room): prior.measure([row for row in rows if row["room_group"] == room], grid, point)
                            for room in (1, 2)},
            }
        reports[arm] = {
            "frozen_budget_outcomes": groups,
            "held_location_geometry": {
                "reachable": sum(any(prior.useful(w, row["truth"]) for w in grid) for row in edited),
                "admitted": sum(any(w in grid and p is not None and prior.useful(w, row["truth"])
                                    for p, w in zip(row["scores"], prior.GRID, strict=True)) for row in edited),
                "strongest_before_point": sum(prior.useful(prior.strongest(row, grid)[1], row["truth"]) for row in edited),
            },
        }
    return reports


def validate(receipt):
    reference = read(prior.RECEIPT)
    prior.validate(reference)
    locks = receipt["commitments"]
    check(receipt["schema"] == SCHEMA and receipt["budgets"] == {"primary": 2, "sensitivity": 1}
          and receipt["grids"] == {arm: [list(w) for w in grid] for arm, grid in GRIDS.items()}
          and receipt["serving_admitted"] is False and receipt["customer_confidence_changed"] is False,
          "Wrong grouped calibration or serving scope")
    expected_files = {"backend/scripts/audio_edge_window_grouped_sources.py", "backend/scripts/audio_edge_window_grouped_transfer.py"}
    check(set(locks["runner_sources"]) == expected_files and all(sha(ROOT / name) == pin for name, pin in locks["runner_sources"].items())
          and locks["reviewer"] == sha(Path(__file__)) and locks["prior_ordinary"] == sha(prior.RECEIPT)
          and locks["public_held_rows"] == prior.row_digest(receipt["held"]), "Changed code or score commitments")
    check(receipt["points"] == independently_fit(reference["selection"]), "Calibration points used different ordinary groups")
    check(receipt["results"] == summarize(receipt["held"], receipt["points"]), "Held results differ from replay")
    check(receipt["new_classifier_windows"] == sum(p is not None for row in receipt["held"] for p in row["scores"]),
          "Unexpected native inference count")


def native_ancestry(inputs):
    from scripts import audio_edge_window_grouped_sources as sources
    from scripts import audio_edge_window_transfer as exploratory

    prior.source_separation(list(exploratory.calibration_cases()), inputs)
    prior.source_separation(list(exploratory.fresh.held(exploratory.HELD)), inputs)
    identities = {(row["family"], row["donor"]["canonical_speaker"]) for row, _ in inputs if row["truth"]}
    check(Counter(family for family, _ in identities) == {"clone": 4, "expressive": 4, "stock": 4},
          "Held generator profiles overlap")
    rooms = defaultdict(set)
    for row, _ in inputs:
        bg = row["background"]
        a = bg["capture_start_sample"]
        rooms[bg["room"]].add((a, a + bg["samples"]))
    check(set(rooms) == set(sources.ROOMS)
          and all(len(cuts) == 7 and all(b <= c for (_, b), (c, _) in zip(sorted(cuts), sorted(cuts)[1:])) for cuts in rooms.values()),
          "New held room cuts overlap")
    earlier = sources.earlier
    earlier.previous.participants(earlier.previous.ANNOTATIONS, (*sources.ROOMS, *earlier.ROOMS, *sources.FRESH.SESSIONS))


def build(run, held_path, out):
    from scripts import audio_edge_window_grouped_transfer as runner

    freeze_path, points_path, score_path = (run / f"{name}.json" for name in ("freeze", "points", "held"))
    freeze, points, scores = (read(path) for path in (freeze_path, points_path, score_path))
    lock, ordinary = runner.freeze(held_path)
    check(freeze == lock and points["freeze"] == scores["freeze"] == sha(freeze_path)
          and scores["points"] == sha(points_path) and points["operating_points"] == independently_fit(ordinary),
          "Held results are unbound from the pre-inference calibration")
    inputs = list(runner.sources.cases(held_path))
    native_ancestry(inputs)
    public = prior.verified_rows(((row, raw, {}) for row, raw in inputs), scores["records"])
    for row, (source, _) in zip(public, inputs, strict=True):
        row.update(snr_db=source["snr_db"], room_group=source["room_group"])
    receipt = {
        "schema": SCHEMA, "budgets": {"primary": 2, "sensitivity": 1}, "grids": freeze["grids"],
        "serving_admitted": False, "customer_confidence_changed": False,
        "commitments": {"freeze": sha(freeze_path), "points": sha(points_path), "held_native_scores": sha(score_path),
                        "prior_ordinary": sha(prior.RECEIPT), "reviewer": sha(Path(__file__)), "held_sources": freeze["held_sources"],
                        "public_held_rows": prior.row_digest(public),
                        "runner_sources": {name: pin for name, pin in freeze["implementation"].items()
                                           if name.startswith("backend/scripts/audio_edge_window_grouped_")}},
        "held": public, "points": points["operating_points"], "results": summarize(public, points["operating_points"]),
        "new_classifier_windows": sum(w["score"] is not None for row in scores["records"] for w in row["windows"]),
    }
    validate(receipt)
    if out.exists():
        check(read(out) == receipt, "Published grouped receipt differs from native replay")
    else:
        with out.open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print("Replayed frozen calibration, 50 native inputs, 350 planned window bindings, and disjoint source and room roles.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", nargs=2, type=Path, metavar=("RUN", "HELD"))
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    if args.native:
        build(*args.native, args.receipt)
    else:
        validate(read(args.receipt))
        print("Replayed both grouped calibration points and all held outcomes.")


if __name__ == "__main__":
    main()
