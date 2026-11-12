"""Replay writing agreement positions and counts without application or model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

WORDS = re.compile(r"\w[\w'’-]*")
ARMS = ("current", "second_only", "agreement", "current_plus_rescue")
CONTROLS = {"original", "human_splice"}


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def previous_rows(panel_root):
    long = panel_root / "writing-grammar-seams-2026-11-11"
    original = read(long / "baseline.json")
    for row in read(long / "cases.json")["cases"]:
        windows = original["contexts"][f"{row['group']}/{row['kind']}"]
        current = [original["native"][window["sha256"]]["score"] for window in windows]
        yield "writing-grammar-seams", row["publisher"], row["group"], row["kind"], row["text"], row["text_sha256"], row["span"], row["bounds"], windows, current
    short = panel_root / "writing-option-order-transfer-2026-11-11"
    sources = read(short / "sources.json")["sources"]
    for row in read(short / "private-order-readout.json"):
        case = row["case"]
        value = (short / "fixtures" / (case["id"] + ".txt")).read_text()
        windows = row["contexts"]
        current = [window["native"]["score"] for window in windows]
        yield "writing-option-order-transfer", sources[case["group"]]["publisher"], case["group"], case["kind"], value, case["sha256"], case["replacement"], [(0, len(value))], windows, current


def bit_positions(text, spans):
    return sum(1 << i for a, b in spans for i in range(a, b) if not text[i].isspace())


def reproduce(row, scores):
    panel, publisher, group, kind, text, text_pin, truth, paragraphs, windows, current = row
    check(sha(text) == text_pin, "Fixture text differs")
    focuses = [entry["spans"][min(i, 1)] for i, entry in enumerate(windows)]
    check(all(0 <= a < b <= len(text) for a, b in focuses), "Invalid source range")
    check(all(left[1] <= right[0] for left, right in zip(focuses, focuses[1:])), "Overlapping source ranges")
    comparison = defaultdict(list)
    second = []
    seen = set()
    for i, entry in enumerate(windows):
        ranges = focuses[max(i - 1, 0):i + 2]
        encoded = " ".join(text[a:b] for a, b in ranges)
        pin = sha(encoded)
        check(ranges == entry["spans"] and pin == entry["sha256"], "Exact context input differs")
        row = scores[pin]
        tokens = row["coverage"]
        check(0 < tokens["tokens"] <= 512 and tokens["token_windows"] == 1
              and tokens.get("covered_tokens", tokens["tokens"]) == tokens["tokens"]
              and tokens.get("omitted_tokens", 0) == 0
              and math.isfinite(row["score"]) and 0 <= row["score"] <= 1, "Invalid complete response")
        seen.add(pin)
        second.append(row["score"])
        a, b = focuses[i]
        if len(ranges) == 3 and len(WORDS.findall(text[a:b])) >= 12:
            key = i if len(paragraphs) == 1 else next(j for j, (x, y) in enumerate(paragraphs) if x <= a < b <= y)
            comparison[key].append(i)
    kept = [i for i, score in enumerate(current) if score >= 0.5][:6]
    next_kept = [i for i, score in enumerate(second) if score >= 0.5][:6]
    both = [i for i in kept if second[i] >= 0.5]
    groups = sorted((min(indexes, key=lambda i: (-second[i], i)) for indexes in comparison.values()),
                    key=lambda i: (-second[i], i))
    rescued = kept.copy()
    if len(groups) >= 2:
        best, runner = groups[:2]
        a, b = second[best], second[runner]
        if best not in kept and a >= 0.5 and a > b and a * (1 - b) >= 2 * b * (1 - a):
            rescued.append(best)
    marks = dict(zip(ARMS, (kept, next_kept, both, rescued), strict=True))
    target = bit_positions(text, [truth])
    matched = {}
    for arm, indices in marks.items():
        bits = [bit_positions(text, windows[i]["spans"]) for i in indices]
        matched[arm] = kind not in CONTROLS and any(
            10 * (candidate & target).bit_count() >= 3 * (candidate | target).bit_count() for candidate in bits
        )
    return {"panel": panel, "publisher": publisher, "group": group, "kind": kind, "text_sha256": text_pin,
            "marks": marks, "useful": matched}, seen


def tally(rows, arm):
    human = [r for r in rows if r["kind"] in CONTROLS]
    flags = [(r["group"], r["marks"][arm]) for r in human if r["marks"][arm]]
    human_groups = {group for group, _ in flags}
    old_groups = {r["group"] for r in human if r["marks"]["current"]}
    gained = [r["group"] for r in rows if r["useful"][arm] and not r["useful"]["current"]]
    return {
        "human_flagged": len(flags), "human_groups_flagged": len(human_groups),
        "human_groups_added": len(human_groups - old_groups), "human_groups_cleared": len(old_groups - human_groups),
        "useful": sum(r["useful"][arm] for r in rows), "gained": len(gained), "gained_groups": len(set(gained)),
        "lost": sum(r["useful"]["current"] and not r["useful"][arm] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("panels", type=Path)
    args = parser.parse_args()
    output = read(args.run / "aggregate.json")
    for file, key in (("freeze.json", "freeze"), ("scores.json", "scores"), ("private-readout.json", "readout")):
        check(sha((args.run / file).read_bytes()) == output[f"{key}_sha256"], f"{key} commitment differs")
    for panel, files in output["panels"].items():
        for file, expected in files.items():
            check(sha((args.panels / f"{panel}-2026-11-11" / file).read_bytes()) == expected, "Prior panel commitment differs")
    scores = read(args.run / "scores.json")["responses"]
    all_rows = list(previous_rows(args.panels))
    reconstructed = [reproduce(row, scores) for row in all_rows]
    rows = [entry for entry, _ in reconstructed]
    check(rows == read(args.run / "private-readout.json"), "Private readout differs")
    check(len(rows) == output["documents"] and sum(len(row[-2]) for row in all_rows) == output["window_occurrences"],
          "Case or occurrence denominator differs")
    check(set().union(*(seen for _, seen in reconstructed)) == set(scores) and len(scores) == output["unique_inputs"],
          "Second-checkpoint input denominator differs")
    scopes = {"all", *(row["publisher"] for row in rows), *(row["kind"] for row in rows)}
    check(scopes == output["outcomes"].keys(), "Scopes differ")
    for scope in scopes:
        subset = [row for row in rows if scope in ("all", row["publisher"], row["kind"])]
        human = sum(row["kind"] in CONTROLS for row in subset)
        expected = {"human": human, "edits": len(subset) - human, "arms": {arm: tally(subset, arm) for arm in ARMS}}
        check(expected == output["outcomes"][scope], f"Counts differ for {scope}")
    agreement = tally(rows, "agreement")
    rescue = tally(rows, "current_plus_rescue")
    gained = [row for row in rows if row["useful"]["current_plus_rescue"] and not row["useful"]["current"]]
    axes = {key: sorted({row[key] for row in gained}) for key in ("publisher", "kind")}
    check(output["rescue_gained_scopes"] == axes, "Rescue gain scopes differ")
    check(output["advance_agreement"] == (agreement["lost"] == 0 and agreement["human_groups_cleared"] >= 2),
          "Agreement disposition differs")
    check(output["advance_rescue"] == (rescue["gained_groups"] >= 2 and rescue["human_groups_added"] <= 1
          and min(map(len, axes.values())) >= 2), "Rescue disposition differs")
    print(json.dumps({"documents": len(rows), "exact_window_occurrences": output["window_occurrences"],
                      "unique_inputs": len(scores), "outcome_counts": len(scopes) * (len(ARMS) * 8 + 2), "status": "verified"}))


if __name__ == "__main__":
    main()
