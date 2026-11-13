"""Test source-bound sustained writing responses on a retained full worker panel."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

from scripts import review_writing_revision_worker_transfer as worker
from scripts import writing_revision_trajectory_transfer as source

SCHEMA = "chipmunk.writing-sustained-response/1"
ARMS = ("classifier_plateau", "two_reading_plateau")
WIDTHS = (3, 4, 5, 6, 7, 8)
FLANK = 2
PANELS = {
    "writing-revision-worker-fit-2026-11-13.json": "fit",
    "writing-revision-worker-complete-2026-11-13.json": "held",
}
PROTOCOL = {
    "fitting": "Use 16 separate genuine work maxima. Require a strictly higher response than the second-highest genuine maximum, with a positive floor of 0.1.",
    "scoring": "Select at most one interior run of 3–8 consecutive same-paragraph sentence centers with two measured flanks on each side. Compare the interior median response to the stronger flanking median and require at least two-thirds of interior centers above both flank medians.",
    "comparison": "Score a robustly scaled classifier logit alone and the weaker of its response and the direction-adjusted exact sentence ratio. Each reading keeps its own scale. Allow no missing or neighbor-expanded ratio in the two-reading arm.",
    "admission": "A separate inspection cue would need at least three new strict locations on distinct source groups across both editing families, at most two genuine source groups flagged, and a separate fresh replication. Raw readings have no confidence weight.",
}


def standardize(numbers, floor):
    center = median(numbers)
    scale = max(floor, median(abs(value - center) for value in numbers))
    return [(value - center) / scale for value in numbers]


def reading(candidate, start, end):
    inside = candidate[start:end]
    before = candidate[start - FLANK:start]
    after = candidate[end:end + FLANK]
    if None in inside + before + after:
        return None
    edge = max(median(before), median(after))
    if 3 * sum(value > edge for value in inside) < 2 * len(inside):
        return 0.
    return max(0., median(inside) - edge)


def select(row):
    classifier = standardize([source.odds(worker.number(value)) for _, value in row["raw"]], .25)
    ratio = [(-worker.number(value["raw"]) if value is not None and not value["neighbor_expansion"] else None)
             for value in row["ratio"]]
    normalized = iter(standardize([value for value in ratio if value is not None], .01)) if any(
        value is not None for value in ratio) else iter(())
    ratio = [next(normalized) if value is not None else None for value in ratio]
    ranges = row["ranges"]
    choices = {arm: {"maximum": None, "eligible": 0, "proposal": None} for arm in ARMS}
    for width in WIDTHS:
        for start in range(FLANK, len(ranges) - width - FLANK + 1):
            end = start + width
            if ranges[start - FLANK]["paragraph"] != ranges[end + FLANK - 1]["paragraph"]:
                continue
            primary = reading(classifier, start, end)
            corroboration = reading(ratio, start, end)
            for arm, value in zip(ARMS, (primary, min(primary, corroboration) if corroboration is not None else None), strict=True):
                if value is None:
                    continue
                record = choices[arm]
                record["eligible"] += 1
                if record["maximum"] is None or value > record["maximum"]:
                    record["maximum"] = value
                    record["proposal"] = [start, end]
    return choices


def frozen_fit(file):
    fit = source.read(file)
    source.require(fit["schema"] == worker.PUBLIC_SCHEMA and len(fit["rows"]) == 16
                   and len({row["group"] for row in fit["rows"]}) == 16
                   and all(row["role"] == "fit" and row["family"] == "original" for row in fit["rows"]),
                   "Expected the separate frozen genuine controls")
    inputs = [select(row) for row in fit["rows"]]
    values = {}
    for arm in ARMS:
        usable = sorted(row[arm]["maximum"] for row in inputs if row[arm]["maximum"] is not None)
        source.require(len(usable) >= 12, "Too few genuine controls to define a point")
        values[arm] = {"point": max(.1, usable[-2]).hex(), "genuine_measured": len(usable)}
    return {"schema": SCHEMA, "protocol": PROTOCOL, "fit_ledger_sha256": source.digest(file),
            "worker_reader_sha256": source.digest(Path(worker.__file__)),
            "source_reader_sha256": source.digest(Path(source.__file__)),
            "scorer_sha256": source.digest(Path(__file__)), "fit": values}


def analyze(lock, ledger):
    source.require(ledger["completed_batches"] == ledger["planned_batches"] == 5, "Incomplete worker comparison")
    cases = [row for row in ledger["rows"] if row["role"] == "held"]
    originals = {row["group"]: row["text_sha256"] for row in cases if row["family"] == "original"}
    source.require(len(cases) == 140 and len(originals) == 20, "Held source roster differs")
    fit_groups = {row["group"] for row in ledger["rows"] if row["role"] == "fit"}
    source.require(len(fit_groups) == 16 and fit_groups.isdisjoint(originals), "Source groups cross fitting and held roles")
    output, totals = [], {arm: {} for arm in ("app", *ARMS)}
    new_groups, new_families = {arm: set() for arm in ARMS}, {arm: set() for arm in ARMS}
    points = {arm: worker.number(lock["fit"][arm]["point"]) for arm in ARMS}
    for index, row in enumerate(cases):
        label = "original" if row["family"] == "original" else (
            "byte_identical" if originals[row["group"]] == row["text_sha256"] else row["family"])
        choices = select(row)
        original = worker.metrics(row, [[worker.sentence_indices(row, *span) for span in packet["spans"]] for packet in row["packets"]])
        measured = {"app": original}
        for arm, chosen in choices.items():
            score = chosen["maximum"]
            if score is not None:
                source.require(math.isfinite(score), "Non-finite response")
            selected = score is not None and score > points[arm]
            proposal = chosen["proposal"]
            measured[arm] = worker.metrics(row, [[tuple(proposal)]] if selected else [])
            chosen["maximum"] = score.hex() if score is not None else None
            chosen["source_range"] = [row["ranges"][proposal[0]]["start"], row["ranges"][proposal[1] - 1]["end"]] if selected else None
            if measured[arm]["strict_location"] and not original["strict_location"] and label not in {"original", "byte_identical"}:
                new_groups[arm].add(row["group"])
                new_families[arm].add(label)
            measured[arm]["assessed"] = score is not None
        for arm, metrics in measured.items():
            totals[arm].setdefault(label, Counter()).update({"files": 1, **metrics})
        output.append({"row": index, "stratum": label, "measurements": choices})
    decisions = {arm: {"additional_strict_source_groups": len(new_groups[arm]), "additional_strict_families": sorted(new_families[arm]),
                       "genuine_flags": totals[arm]["original"]["file_highlighted"]} for arm in ARMS}
    return {"method": lock, "held": output, "summary": totals, "decisions": decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score", "review"))
    parser.add_argument("panels", type=Path)
    parser.add_argument("freeze", type=Path)
    parser.add_argument("receipt", type=Path, nargs="?")
    args = parser.parse_args()
    fit, held = (args.panels / name for name in PANELS)
    expected = frozen_fit(fit)
    if args.action == "freeze":
        with args.freeze.open("x") as output:
            json.dump(expected, output, indent=2, sort_keys=True)
        return
    lock = source.read(args.freeze)
    source.require(lock == expected, "The current method differs from its frozen protocol")
    if args.receipt is None:
        parser.error("score and review require a receipt")
    ledger = source.read(held)
    fit_rows = source.read(fit)["rows"]
    source.require(fit_rows == [row for row in ledger["rows"] if row["role"] == "fit"], "Held comparison binds different controls")
    receipt = analyze(lock, ledger) | {"held_ledger_sha256": source.digest(held)}
    if args.action == "score":
        with args.receipt.open("x") as output:
            json.dump(receipt, output, indent=2, sort_keys=True)
            output.write("\n")
    else:
        source.require(source.read(args.receipt) == receipt, "The sustained responses did not replay")
    print(json.dumps(receipt["decisions"], indent=2))


if __name__ == "__main__":
    main()
