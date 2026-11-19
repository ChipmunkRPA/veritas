"""Recount anonymous writing-edit positions without source prose or model weights."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.mmwr-minimal-edit-receipt/1"
TRIALS = {"short_current": 12, "short_floor_10": 10, "full_current": 12}
SIZES = ("minimal", "moderate")
FAMILIES = ("a", "b")
FIDELITY = ("preserved", "changed", "uncertain")
MAX_RECEIPT_BYTES = 128_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounds(pair, length):
    return (isinstance(pair, list) and len(pair) == 2 and all(type(x) is int for x in pair)
            and 0 <= pair[0] < pair[1] <= length)


def length(spans):
    return sum(b - a for a, b in spans)


def useful(proposal, target):
    a, b = proposal
    common = sum(max(0, min(b, right) - max(a, left)) for left, right in target)
    count = length(target)
    return common > 0 and 2 * common >= count and 10 * common >= 3 * (b - a + count - common)


def roles(cases):
    require(len(cases) == 19, "One or more views are missing")
    require([case["work"] for case in cases[:7]] == list(range(1, 8)), "Untouched works differ")
    require([case["role"] for case in cases] == ["untouched"] * 7 + ["earlier", "corrected"] + ["edited"] * 10,
            "Unexpected view roles")
    require(cases[7]["work"] == cases[8]["work"] == 6, "The correction no longer matches its untouched source")
    keys = set()
    for case in cases[9:]:
        work, family, size, fidelity = (case[key] for key in ("work", "family", "size", "fidelity"))
        key = (work, family, size)
        require(work in range(1, 8) and family in FAMILIES and size in SIZES and fidelity in FIDELITY and key not in keys,
                "Invalid or duplicate edited view")
        keys.add(key)
    return keys


def replay_attempts(rows, admitted):
    planned = {(work, family, size) for work in range(1, 8) for family in FAMILIES for size in SIZES}
    seen = set()
    qualified = set()
    total = 0
    for work, family, size, attempts, chosen in rows:
        key = (work, family, size)
        require(key in planned and key not in seen and len(attempts) == 3 and all(type(x) is bool for x in attempts),
                "Invalid generation slot")
        seen.add(key)
        total += len(attempts)
        first = attempts.index(True) + 1 if any(attempts) else None
        require(chosen == first, "The selection did not use the first eligible attempt")
        if chosen is not None:
            qualified.add(key)
    require(planned == seen and admitted == qualified, "Generation outcomes do not match the evaluated views")
    return {"slots": len(seen), "attempts": total, "admitted": len(qualified)}


def replay_trial(cases, result, minimum_sentences):
    rows = result["views"]
    require(len(rows) == len(cases), "One or more measurements are missing")
    require(rows[5] == rows[7], "The duplicate earlier correction view must match the untouched view")
    totals = Counter()
    controls = Counter()
    edits = {}
    for case, row in zip(cases, rows, strict=True):
        n, sentences = row["characters"], row["sentences"]
        require(type(n) is int and 0 < n <= 100_000 and type(sentences) is int and 1 <= sentences <= 128,
                "Invalid text or sentence length")
        require(row["assessed"] is (sentences >= minimum_sentences), "The sentence floor was not applied")
        selected, focal, changed = (row[key] for key in ("selected", "focal", "changed"))
        require(len(selected) <= 6 and all(bounds(span, n) for span in selected) and (row["assessed"] or not selected),
                "Invalid or unassessed selection")
        totals["views"] += 1
        totals["assessed"] += row["assessed"]
        totals["flagged"] += bool(selected)
        if case["role"] != "edited":
            require(not focal and not changed, "A control has generated edit truth")
            controls[case["role"] + "_flagged"] += bool(selected)
            continue
        require(bounds(focal, n) and changed, "Edit truth is missing")
        end = focal[0]
        for span in changed:
            require(bounds(span, n) and end <= span[0] < span[1] <= focal[1], "Changed words leave the focal sentence")
            end = span[1]
        count = edits.setdefault(f"{case['size']}/{case['family']}/{case['fidelity']}", Counter())
        hit = any(useful(selection, [focal]) for selection in selected)
        count["views"] += 1
        count["assessed"] += row["assessed"]
        count["sentence_hits"] += hit
        count["changed_word_hits"] += any(useful(selection, changed) for selection in selected)
        count["misplaced"] += len(selected) - hit
        count["sentence_can_match_changed_words"] += useful(focal, changed)
    return {"totals": totals, "controls": controls, "edits": edits,
            "distinct_classifier_inputs": result["distinct_classifier_inputs"]}


def review(data):
    require(data["schema"] == SCHEMA and set(data["trials"]) == set(TRIALS), "Unsupported receipt")
    cases = data["cases"]
    admissions = replay_attempts(data["attempts"], roles(cases))
    trials = {trial: replay_trial(cases, data["trials"][trial], minimum) for trial, minimum in TRIALS.items()}
    baseline = data["trials"]["short_current"]["views"]
    widened = data["trials"]["short_floor_10"]["views"]
    full = data["trials"]["full_current"]["views"]
    for short, probe, long in zip(baseline, widened, full, strict=True):
        require(all(short[key] == probe[key] for key in ("characters", "sentences", "focal", "changed")),
                "Short and widened views do not share exact input positions")
        require(10 <= short["sentences"] <= 11 and 16 <= long["sentences"] <= 84, "Unexpected excerpt geometry")
        if short["focal"]:
            delta = long["focal"][0] - short["focal"][0]
            require([[a + delta, b + delta] for a, b in [short["focal"], *short["changed"]]]
                    == [long["focal"], *long["changed"]], "The full-body view did not retain the same edit")
    verified = set()
    expected = {i for i, case in enumerate(cases, 1)
                if case["role"] in {"earlier", "corrected"} or case.get("fidelity") == "preserved"}
    for case_id, chars, focus, paragraph, run, changed_count in data["native_carrier_witnesses"]:
        require(case_id in expected and case_id not in verified and bounds(focus, chars) and bounds(run, chars)
                and type(paragraph) is int and paragraph > 0 and length([focus]) == length([run])
                and changed_count == len(baseline[case_id - 1]["changed"]), "Invalid native carrier position")
        verified.add(case_id)
    require(verified == expected, "A native carrier witness is missing")
    return {"generation": admissions, "trials": trials, "native_carrier_witnesses": len(verified)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    with gzip.open(args.receipt, "rb") as stream:
        raw = stream.read(MAX_RECEIPT_BYTES + 1)
    require(len(raw) <= MAX_RECEIPT_BYTES, "Receipt is too large")
    print(json.dumps(review(json.loads(raw)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
