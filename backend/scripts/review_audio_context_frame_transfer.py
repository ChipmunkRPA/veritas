"""Read frozen time-aligned voice responses on the original noisy PCM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

from scripts import review_audio_stock_voice_transfer as prior

SCHEMA = "chipmunk.audio-context-aligned-frame-readout/1"
RATE = 16000
WINDOWS = prior.WINDOWS
POINT = prior.SHORT_POINT
CELL = 6400
CELLS = 15
VIEWS = ("context_floor", "remote_contrast", "localized_return")
ARMS = ("raw_point", *VIEWS)
BGS = ("background", "isolated")
BUDGETS = (0, 1, 2)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_choice(row):
    return max(((value, -i, WINDOWS[i]) for i, value in enumerate(row["scores"]) if value is not None), default=None)


def proposals(row):
    by_cell = list(zip(*row["grid"], strict=True)) if row["grid"] else [()] * CELLS
    levels = [sorted(value for value in seq if value is not None) for seq in by_cell]
    contextual = [median(column) if column else None for column in levels]
    floor = [column[0] if len(column) >= 2 else None for column in levels]
    candidates = {name: [] for name in VIEWS}
    for i, (left, right) in enumerate(zip(floor, floor[1:], strict=False)):
        if left is None or right is None:
            continue
        response = min(left, right)
        interval = (i * CELL, (i+2) * CELL)
        candidates["context_floor"].append((response, -i, interval))
        distant = [value for j, value in enumerate(contextual) if j not in range(i-2, i+4) and value is not None]
        if len(distant) >= 4 and (contrast := response - median(distant)) > 0:
            candidates["remote_contrast"].append((response + contrast, -i, interval))
        if i < 2 or i+3 >= CELLS:
            continue
        before, after = contextual[i-2:i], contextual[i+2:i+4]
        if None not in before and None not in after and (contrast := response - max(mean(before), mean(after))) > 0:
            candidates["localized_return"].append((response + 2*contrast, -i, interval))
    return {name: max(values, default=None) for name, values in candidates.items()} | {"raw_point": raw_choice(row)}


def fit_points(rows):
    results = {}
    for arm in ARMS:
        ordinary, background = {}, []
        for row in rows:
            if row["truth"] or (raw := raw_choice(row)) and raw[0] > POINT:
                continue
            reading = proposals(row)[arm]
            if reading is None:
                continue
            if row["kind"] in BGS:
                background.append(reading[0])
                continue
            group = row["group"]
            ordinary[group] = max(ordinary.get(group, -math.inf), reading[0])
        ceilings = sorted(ordinary.values(), reverse=True)
        prior.require(ceilings, "Missing ordinary point-fitting responses")
        results[arm] = [max(ceilings[min(i, len(ceilings)-1)], max(background, default=-math.inf)) for i in BUDGETS]
    return results


def readout(rows, arm=None, point=None):
    stats = Counter(dict.fromkeys(("positive", "ordinary", "background", "unassessed", "correct", "missed", "wrong_window",
                                    "false_ordinary", "false_background", "additional_correct", "additional_wrong_window",
                                    "additional_false_ordinary", "additional_false_background"), 0))
    false_groups, added_false_groups = set(), set()
    readings = []
    for row in rows:
        truth = row["truth"]
        kind = "positive" if truth is not None else "background" if row["kind"] in BGS else "ordinary"
        stats[kind] += 1
        raw = raw_choice(row)
        chosen = raw if raw is not None and raw[0] > POINT else None
        added = False
        if chosen is None and arm is not None:
            candidate = proposals(row)[arm]
            if candidate is not None and candidate[0] > point:
                chosen, added = candidate, True
        if chosen is None:
            stats["missed"] += kind == "positive"
            stats["unassessed"] += raw is None
            continue
        if kind == "ordinary":
            false_groups.add(row["group"])
            if added:
                added_false_groups.add(row["group"])
        outcome = "correct" if truth and prior.useful(chosen[2], truth) else "wrong_window" if truth else f"false_{kind}"
        stats[outcome] += 1
        if added:
            stats[f"additional_{outcome}"] += 1
            readings.append({k: row[k] for k in ("group", "kind", "capture", "truth")} | {"interval": list(chosen[2]), "outcome": outcome})
    return dict(stats, false_ordinary_groups=len(false_groups), additional_false_ordinary_groups=len(added_false_groups),
                added_readings=readings)


def placement(rows, arm):
    positive = [row for row in rows if row["truth"] is not None]
    available = [(proposal, row["truth"]) for row in positive if (proposal := proposals(row)[arm]) is not None]
    return {"positive": len(positive), "available": len(available),
            "best_useful": sum(prior.useful(proposal[2], truth) for proposal, truth in available)}


def check_rows(rows):
    require = prior.require
    counts = 0
    for row in rows:
        require(set(row) == {"group", "kind", "capture", "truth", "pcm_sha256", "scores", "grid"}
                and prior.sha256(row["pcm_sha256"]), "Unexpected row data")
        require(len(row["scores"]) == len(WINDOWS) and all(v is None or math.isfinite(v) for v in row["scores"]),
                "Unexpected native window response")
        observed = [(i, window) for i, (score, window) in enumerate(zip(row["scores"], WINDOWS, strict=True)) if score is not None]
        require(len(row["grid"]) == len(observed), "Missing same-clock contextual responses")
        for responses, (_, (a, b)) in zip(row["grid"], observed, strict=True):
            require(len(responses) == CELLS, "Invalid source clock cell count")
            for i, value in enumerate(responses):
                if value is not None:
                    require(math.isfinite(value) and max(0, min((i+1)*CELL, b) - max(i*CELL, a)) >= 3520,
                            "Frame support exceeds its source window")
        counts += len(observed)
    return counts


def check_source(rows, source, role):
    if role == "fit":
        subjects = [row for row in source["rows"] if row["condition"] in ("-5dB", "background")]
        expected = [(row["group"], row["kind"], row["truth"], row["pcm_sha256"], row["scores"], None) for row in subjects]
    elif role == "transfer":
        subjects = [row for row in source["rows"] if row["snr"] == -5 or row["kind"] == "isolated"]
        expected = [(row["group"], row["kind"], row["edit_clock"], row["pcm_sha256"], row["original_scores"], row["capture"])
                    for row in subjects]
    else:
        subjects = [row for row in source["assessment"]["rows"] if row["level"] == -5]
        expected = []
        for row in subjects:
            available = {tuple(item["range"]): item["score"] for item in row["readings"]["short"]}
            expected.append((int(row["case"].split("-")[2]), row["kind"],
                             row["source_edit_range"] if row["kind"].startswith("synthetic_") else None,
                             row["pcm_sha256"], [available.get(window) for window in WINDOWS], 0))
    actual = [tuple(row[k] for k in ("group", "kind", "truth", "pcm_sha256", "scores", "capture")) for row in rows]
    prior.require(actual == expected, "Original responses or source labels differ from prior receipt")


def verify(receipt, directory):
    require = prior.require
    require(receipt["schema"] == SCHEMA and receipt["cell_samples"] == CELL and receipt["sample_rate"] == RATE,
            "Changed readout method or clock")
    fitting = receipt["rows"]["fit"]
    require({role: len(rows) for role, rows in receipt["rows"].items()}
            == {"fit": 44, "transfer": 43, "same_words": 14}, "Incomplete historical transfer panel")
    require(receipt["forwards"] == {role: check_rows(rows) for role, rows in receipt["rows"].items()}, "Inference count differs")
    for role, rows in receipt["rows"].items():
        name, commitment = receipt["source_receipts"][role]
        require(Path(name).name == name and digest(directory / name) == commitment, "Earlier source receipt changed")
        check_source(rows, read(directory / name), role)
    points = fit_points(fitting)
    require(points == receipt["points"], "Operating points do not reproduce")
    totals = {}
    for role, rows in receipt["rows"].items():
        totals[role] = {"existing": readout(rows),
                        "placement_before_operating_point": {arm: placement(rows, arm) for arm in ARMS},
                        **{arm: [readout(rows, arm, point) for point in points[arm]] for arm in ARMS}}
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = read(args.receipt)
    results = verify(receipt, args.receipt.parent)
    prior.require(results == receipt["results"], "Recorded outcomes differ from independent scalar replay")
    print(json.dumps(results, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
