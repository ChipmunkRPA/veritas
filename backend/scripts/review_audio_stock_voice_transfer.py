"""Verify the scalar-only noisy stock-speech transfer receipt."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.audio-stock-voice-transfer/1"
RATE = 16000
WINDOWS = tuple((start, start + 2 * RATE) for start in range(0, 5 * RATE, RATE))
TRUTHS = ((23200, 38560), (55200, 70560))
LEVELS = ("dry", "0dB", "-5dB")
CONTROLS = ("untouched", "same_reader_splice", "different_reader_splice")
SYNTHESIS = ("piper", "kitten")
GRID_SPARSE = (1, 3)
GRID_FULL = tuple(range(len(WINDOWS)))
SHORT_POINT = 6.822523597593649
ROUTES = {
    "existing_short": (GRID_FULL, SHORT_POINT),
    "earlier_sparse": (GRID_SPARSE, 5.356589748054317),
    "equal_point_sparse": (GRID_SPARSE, SHORT_POINT),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def useful(window, truth):
    x, y = window
    a, b = truth
    overlap = max(0, min(b, y) - max(a, x))
    return 2 * overlap >= b - a and 10 * overlap >= 3 * (max(b, y) - min(a, x))


def strongest(row, slots):
    return max(((row["scores"][i], -i) for i in slots if row["scores"][i] is not None), default=None)


def evaluate(rows, slots, point):
    result = Counter(dict.fromkeys(("positives", "ordinary", "background", "correct", "wrong_window", "no_flag_edit",
                                    "unassessed_edit", "unassessed_ordinary", "unassessed_background", "false_ordinary",
                                    "false_background", "strongest_aligned", "paired_checks", "paired_higher", "paired_equal"), 0))
    locations, ordinary_groups, distinct = [], set(), set()
    matched = {(row["group"], row["condition"]): row for row in rows if row["kind"] == "different_reader_splice"}
    for row in rows:
        positive = row["truth"] is not None
        background = row["kind"] == "background"
        result["positives" if positive else "background" if background else "ordinary"] += 1
        if positive:
            ordinary = matched.get((row["group"], row["condition"]))
            if ordinary is not None:
                strengths = [(row["scores"][i], ordinary["scores"][i]) for i in slots
                             if useful(WINDOWS[i], row["truth"])
                             and row["scores"][i] is not None and ordinary["scores"][i] is not None]
                if strengths:
                    edit_score = max(pair[0] for pair in strengths)
                    plain_score = max(pair[1] for pair in strengths)
                    result["paired_checks"] += 1
                    result["paired_higher"] += edit_score > plain_score
                    result["paired_equal"] += edit_score == plain_score
        item = strongest(row, slots)
        if item is None:
            result["unassessed_edit" if positive else "unassessed_background" if background else "unassessed_ordinary"] += 1
            continue
        score, index = item
        window = WINDOWS[-index]
        aligned = positive and useful(window, row["truth"])
        result["strongest_aligned"] += aligned
        if score <= point:
            result["no_flag_edit"] += positive
        elif aligned:
            result["correct"] += 1
            distinct.add((row["group"], row["kind"]))
            locations.append({"group": row["group"], "kind": row["kind"], "condition": row["condition"],
                              "window": list(window), "truth": row["truth"]})
        elif positive:
            result["wrong_window"] += 1
        elif background:
            result["false_background"] += 1
        else:
            result["false_ordinary"] += 1
            ordinary_groups.add(row["group"])
    return dict(result, correct_distinct=len(distinct), false_ordinary_groups=len(ordinary_groups), locations=locations)


def check(receipt):
    require(receipt["schema"] == SCHEMA and sha256(receipt["native_freeze_sha256"]), "Invalid native commitment")
    require(receipt["windows"] == [list(w) for w in WINDOWS] and receipt["points"] == {k: v[1] for k, v in ROUTES.items()},
            "Frozen comparison changed")
    rows = receipt["rows"]
    require(len(rows) == 124 and len({r["pcm_sha256"] for r in rows}) == 124, "Frozen delivery roster changed")
    roster = {(i, k, level) for i in range(8) for k in (*SYNTHESIS, *CONTROLS) for level in LEVELS}
    roster.update((f"background-{i}-{j}", "background", "background") for i in range(2) for j in range(2))
    require({(r["group"], r["kind"], r["condition"]) for r in rows} == roster, "Matched conditions are missing")
    for row in rows:
        require(set(row) == {"group", "kind", "condition", "truth", "pcm_sha256", "scores"} and sha256(row["pcm_sha256"]),
                "Unexpected scalar payload")
        truth = list(TRUTHS[row["group"] % 2]) if row["kind"] in SYNTHESIS else None
        require(row["truth"] == truth and len(row["scores"]) == len(WINDOWS), "Sample clock or scheduled windows changed")
        require(all(x is None or type(x) in (int, float) and math.isfinite(x) for x in row["scores"]), "Invalid admitted score")
    forwards = sum(score is not None for row in rows for score in row["scores"])
    require(receipt["model_forwards"] == forwards > 0, "Classifier forwards differ from admitted windows")
    cohorts = {"all": rows, **{c: [r for r in rows if r["condition"] == c] for c in (*LEVELS, "background")}}
    cohorts.update({kind: [r for r in rows if r["kind"] in (kind, "different_reader_splice")] for kind in SYNTHESIS})
    return {name: {level: evaluate(cohort, slots, point) for level, cohort in cohorts.items()}
            for name, (slots, point) in ROUTES.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    readout = check(receipt)
    require(readout == receipt["summary"], "Published comparison differs from scalar replay")
    print(json.dumps(readout, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
