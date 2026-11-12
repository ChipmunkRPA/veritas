"""Replay noisy short-edit window decisions from the public scalar ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "docs/evaluation-panels/audio-edge-window-transfer-2026-11-12.json"
PRIOR = ROOT / "docs/evaluation-panels/audio-communication-serving-grid-2026-11-12.json"
RATE = 16000
GRIDS = {
    "full": ((0, 64000), (32000, 96000)),
    "edge": ((0, 32000), (32000, 64000), (64000, 96000)),
    "shifted": ((16000, 48000), (48000, 80000)),
    "combined": ((0, 32000), (16000, 48000), (32000, 64000), (48000, 80000), (64000, 96000)),
}
GRID = GRIDS["full"] + GRIDS["combined"]
CONTROLS = {"untouched", "ordinary_same_host", "ordinary_other_voice"}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def row_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def groups(rows, held):
    by_group = defaultdict(list)
    check(len(rows) == (26 if held else 40), "Incorrect calibration or held denominator")
    for row in rows:
        check(set(row) == {"group", "arm", "family", "coverage", "truth", "scores"} and type(row["group"]) is int
              and len(row["scores"]) == len(GRID)
              and all(v is None or type(v) in {int, float} and math.isfinite(v) for v in row["scores"]),
              "Bad anonymous window scores")
        by_group[row["group"]].append(row)
        truth = row["truth"]
        check((truth is None and row["arm"] in CONTROLS | {"background_only"})
              or (held and row["arm"] == "synthetic" and type(truth) is list and len(truth) == 2 and all(type(v) is int for v in truth)
                  and 0 <= truth[0] < truth[1] <= 96000 and truth[1] - truth[0] in {10240, 12800, 15360}),
              "Unexpected truth or control assignment")
    check(set(by_group) == set(range(1, 9 if held else 17)), "Group ordinals changed")
    for rows_in_group in by_group.values():
        arms = {row["arm"] for row in rows_in_group}
        family = {row["family"] for row in rows_in_group}
        coverage = {row["coverage"] for row in rows_in_group}
        expected = CONTROLS | ({"synthetic"} if held else set())
        check((arms == {"background_only"} and len(rows_in_group) == 1 and family == {None} and coverage == {"room"})
              or (arms == expected and len(rows_in_group) == len(expected) and len(family) == 1
                  and family <= {"clone", "expressive", "stock"} and len(coverage) == 1
                  and coverage <= {"early", "interior", "late"}), "Missing matched source controls")
    edits = [row for row in rows if row["truth"]]
    if held:
        check(Counter(row["family"] for row in edits) == {"clone": 2, "expressive": 2, "stock": 2}
              and Counter(row["coverage"] for row in edits) == {"early": 1, "late": 1, "interior": 4},
              "Held edit positions or family balance changed")
    return by_group


def overlap(window, truth):
    return 0 if window is None else max(0, min(window[1], truth[1]) - max(window[0], truth[0]))


def useful(window, truth):
    intersection = overlap(window, truth)
    if not intersection:
        return False
    width = truth[1] - truth[0]
    union = window[1] - window[0] + width - intersection
    return 10 * intersection >= 3 * union and 2 * intersection >= width


def strongest(row, grid):
    options = [(value, window) for value, window in zip(row["scores"], GRID, strict=True)
               if window in grid and value is not None]
    return max(options, key=lambda pair: (pair[0], -pair[1][0]), default=(-math.inf, None))


def measure(rows, grid, point):
    counts, false_groups, hits = Counter(), set(), []
    for row in rows:
        score, window = strongest(row, grid)
        counts["assessed_files"] += window is not None
        flag = score > point
        if truth := row["truth"]:
            counts["generated_assessed"] += window is not None
            counts["generated_flags"] += flag
            counts["selected_half_covered"] += flag and 2 * overlap(window, truth) >= truth[1] - truth[0]
            if flag and useful(window, truth):
                hits.append({"group": row["group"], "family": row["family"], "coverage": row["coverage"],
                             "truth": truth, "window": list(window)})
        elif flag:
            counts["genuine_false_files"] += 1
            counts["background_false_files"] += row["arm"] == "background_only"
            false_groups.add(row["group"])
    return {key: counts[key] for key in (
        "assessed_files", "generated_assessed", "generated_flags", "selected_half_covered",
        "genuine_false_files", "background_false_files",
    )} | {"false_genuine_groups": len(false_groups), "useful_locations": len(hits), "locations": hits}


def summarize(selection, held):
    groups(selection, held=False)
    by_group = groups(held, held=True)
    edits = [row for row in held if row["truth"]]
    table = {}
    for arm, grid in GRIDS.items():
        point = max(strongest(row, grid)[0] for row in selection)
        check(math.isfinite(point), "No assessed calibration control")
        false_scores = sorted((max(strongest(row, grid)[0] for row in group if not row["truth"])
                               for group in by_group.values()), reverse=True)
        table[arm] = {
            "ordinary_point": point,
            "selection_assessed_files": sum(strongest(row, grid)[1] is not None for row in selection),
            "held_geometry": {
                "any_useful_range_planned": sum(any(useful(w, row["truth"]) for w in grid) for row in edits),
                "any_useful_range_admitted": sum(any(w in grid and p is not None and useful(w, row["truth"])
                                                     for p, w in zip(row["scores"], GRID, strict=True)) for row in edits),
                "best_admitted_range_useful": sum(useful(strongest(row, grid)[1], row["truth"]) for row in edits),
                "score_exceeds_all_three_matched_controls": sum(
                    strongest(row, grid)[0] > max(strongest(other, grid)[0] for other in by_group[row["group"]] if not other["truth"])
                    for row in edits),
            },
            "fixed": measure(held, grid, point),
            "equal_held_false_groups": {str(n): measure(held, grid, false_scores[n]) for n in (0, 1, 2)},
        }
    return table


def prior_point_transfer(held):
    point = read(PRIOR)["fixed_short_window_logit"]
    return {"published_two_second_point": point, "held": {arm: measure(held, grid, point) for arm, grid in GRIDS.items()}}


def validate(receipt):
    lock = receipt["commitments"]
    check(receipt["schema"] == "chipmunk.audio-edge-window-transfer/1" and receipt["sample_rate_hz"] == RATE
          and receipt["grids"] == {key: [list(w) for w in grid] for key, grid in GRIDS.items()}
          and receipt["serving_admitted"] is False and receipt["customer_confidence_changed"] is False,
          "Wrong fixed study profile")
    check(lock["prior"] == sha(PRIOR) and lock["reviewer"] == sha(Path(__file__))
          and set(lock["runner_sources"]) == {f"backend/scripts/audio_edge_window_{name}.py" for name in ("sources", "transfer")}
          and all(pin == sha(ROOT / name) for name, pin in lock["runner_sources"].items()),
          "Committed methods or previous receipt changed")
    for key in ("selection", "held"):
        check(lock[f"{key}_public_rows"] == row_digest(receipt[key]), f"Anonymous {key} rows changed")
    check(receipt["results"] == summarize(receipt["selection"], receipt["held"]), "Reported audio comparison differs from replay")
    check(receipt["cross_grid_transfer"] == prior_point_transfer(receipt["held"]), "Published point transfer differs from replay")
    counts = {key: sum(score is not None and window in measured for row in receipt[key]
                       for score, window in zip(row["scores"], GRID, strict=True))
              for key, measured in (("selection", GRIDS["edge"]), ("held", GRID))}
    check(receipt["new_classifier_windows"] == counts, "New voice inference count differs from replay")


def verified_rows(inputs, score_rows):
    from scripts import audio_edge_window_transfer as runner

    anonymous, groups = [], {}
    for (row, raw, cached), recorded in zip(inputs, score_rows, strict=True):
        check(recorded["id"] == row["id"] and recorded["pcm_sha256"] == runner.digest(raw),
              "Model result belongs to different native audio")
        scores = []
        for window, grid in zip(recorded["windows"], GRID, strict=True):
            a, b = grid
            check((window["start"], window["end"]) == grid and window["pcm_sha256"] == runner.digest(raw[2*a:2*b])
                  and window["reused"] == (grid in cached), "Native window commitment changed")
            if grid in cached:
                check(window["score"] == cached[grid]["score"], "Reused window score changed")
            scores.append(window["score"])
        anonymous.append({key: row[key] for key in ("arm", "family", "coverage", "truth")} | {
            "group": groups.setdefault(row["group"], len(groups) + 1), "scores": scores,
        })
    return anonymous


def source_separation(calibration, held):
    def keys(inputs, field):
        return {witness[field] for row, *_ in inputs for role in ("host", "donor") if (witness := row.get(role))}

    for field in ("canonical_speaker", "source_sha256"):
        check(not keys(calibration, field) & keys(held, field), f"Source roles share {field}")
    matched = defaultdict(list)
    for row, raw in held:
        if row["edit"]:
            a, b = row["edit"]
            check(row["background"]["edit_overlap_ms"] * RATE * 2 >= (b - a) * 1000,
                  "Held edit lacks overlapping background words")
            matched[row["group"]].append((row["edit"], raw[:2*a], raw[2*b:]))
    check(all(len(records) == 4 and len({(tuple(edit), before, after) for edit, before, after in records}) == 1
              for records in matched.values()), "Native inputs differ outside their known insert")


def build(run, held_path, receipt_path):
    from scripts import audio_edge_window_transfer as runner

    paths = {key: run / f"{key}.json" for key in ("freeze", "points", "selection", "held")}
    freeze, points, selection, held = (read(paths[key]) for key in paths)
    frozen = sha(paths["freeze"])
    check(freeze == runner.frozen(held_path) and all(record["freeze"] == frozen for record in (points, selection, held))
          and points["calibration"] == sha(paths["selection"]) and held["points"] == sha(paths["points"]),
          "Held scores are unbound from the pre-registered ordinary point")
    calibration_inputs = list(runner.calibration_cases())
    held_inputs = list(runner.fresh.held(held_path))
    source_separation(calibration_inputs, held_inputs)
    point_rows = verified_rows(calibration_inputs, selection["records"])
    held_rows = verified_rows(((row, raw, {}) for row, raw in held_inputs), held["records"])
    results = summarize(point_rows, held_rows)
    check({arm: row["ordinary_point"] for arm, row in results.items()} == points["values"], "Ordinary calibration point changed")
    receipt = {
        "schema": runner.SCHEMA, "sample_rate_hz": RATE, "grids": freeze["grids"],
        "serving_admitted": False, "customer_confidence_changed": False,
        "commitments": {key: sha(path) for key, path in paths.items()} | {
            "prior": sha(PRIOR), "reviewer": sha(Path(__file__)), "held_sources": freeze["held_sources"],
            "runner_sources": {name: pin for name, pin in freeze["code"].items() if name.startswith("backend/scripts/audio_edge_")},
            "selection_public_rows": row_digest(point_rows), "held_public_rows": row_digest(held_rows),
        },
        "selection": point_rows, "held": held_rows, "results": results, "cross_grid_transfer": prior_point_transfer(held_rows),
        "new_classifier_windows": {key: sum(window["score"] is not None and not window["reused"]
                                            for record in ledger["records"] for window in record["windows"])
                                   for key, ledger in (("selection", selection), ("held", held))},
    }
    validate(receipt)
    if receipt_path.exists():
        check(read(receipt_path) == receipt, "Existing public result differs from native input replay")
    else:
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print("Verified 66 native recordings and 462 planned windows, with separate ordinary sources and matched held edit samples.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", nargs=2, type=Path, metavar=("RUN", "HELD"))
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    if args.native:
        build(*args.native, args.receipt)
    else:
        validate(read(args.receipt))
        print("Replayed separate ordinary calibration and held decisions for all four grids.")


if __name__ == "__main__":
    main()
