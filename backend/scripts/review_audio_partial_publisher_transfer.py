# SPDX-License-Identifier: Apache-2.0
"""Replay the frozen three-second speech decisions and native edit geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.audio-partial-publisher-transfer/1"
SHA = re.compile(r"[0-9a-f]{64}")
RATE, WINDOW, STEP = 16_000, 48_000, 24_000
ARMS = ("sparse", "dense", "combined")
SCENES = {"gain_control", "traffic", "cafe"}
DIRECTIONS = {"ai", "original", "inconclusive", "unassessed"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def span_schedules(length):
    require(type(length) is int and 6 * RATE <= length <= 14 * RATE, "Recording length is out of scope")
    last = length - WINDOW
    sparse = [0, last] if length < 9 * RATE else [0, last // 2, last]
    return [[a, a + WINDOW] for a in sparse], [[a, a + WINDOW] for a in range(0, last + 1, STEP)]


def truth_for(row):
    if row["source"] is None:
        return "isolated", []
    source, length = row["source"], row["samples"]
    clock = source["clock"]
    require(length == source["samples"] and clock[0][0] == 0 and clock[-1][1] == length
            and all(0 <= a < b <= length and c in {"spoof", "bonafide"} for a, b, c in clock)
            and all(a[1] == b[0] for a, b in zip(clock, clock[1:])), "Invalid publisher sample clock")
    edits = [[a, b] for a, b, c in clock if c == "spoof"]
    kind = "genuine" if not edits else "full" if edits == [[0, length]] else "partial"
    require(edits == source["truth"] and kind == source["kind"], "Publisher class or label intervals changed")
    return kind, edits


def outcomes(row, responses, raw):
    sparse, dense = span_schedules(row["samples"])
    require((sparse, dense) == (row["sparse"], row["dense"]), "The input schedules differ from the native positions")
    ordered = sorted(set(map(tuple, sparse + dense)))
    require(ordered == [tuple(response["span"]) for response in responses], "Incomplete native window decisions")
    directions = {}
    for response, (start, stop) in zip(responses, ordered, strict=True):
        require(SHA.fullmatch(response["pcm_sha256"]) and response["direction"] in DIRECTIONS, "Invalid window response")
        if raw is not None:
            require(digest(raw[2 * start:2 * stop]) == response["pcm_sha256"], "Native window bytes differ")
        directions[start, stop] = response["direction"]
    current = [span for span in sparse if directions[tuple(span)] == "ai"]
    if len(current) < 2 or any(directions[tuple(span)] == "original" for span in sparse):
        current = []
    candidate = [[right[0], left[1]] for left, right in zip(dense, dense[1:])
                 if directions[tuple(left)] == directions[tuple(right)] == "ai"]
    # Both window decisions use the same fixed calibration. Their intersection
    # only localizes review and is not an independent confidence estimate.
    return {"sparse": current, "dense": candidate, "combined": current + candidate}, directions


def correctly_located(span, edit):
    shared = max(0, min(span[1], edit[1]) - max(span[0], edit[0]))
    length = edit[1] - edit[0]
    union = span[1] - span[0] + length - shared
    return 2 * shared >= length and 10 * shared >= 3 * union


def study(freeze, roster, decisions, pcm_dir=None):
    require(freeze["schema"] == SCHEMA and len(roster) == len(decisions) == 92, "Invalid frozen comparison size")
    summary, sources, verdicts = defaultdict(Counter), defaultdict(list), []
    for index, (row, record) in enumerate(zip(roster, decisions, strict=True)):
        require(row["key"] == record["key"] == index and row["scene"] in SCENES, "Frozen row identity changed")
        raw = None if pcm_dir is None else (pcm_dir / f"{index:03}.pcm").read_bytes()
        require(SHA.fullmatch(row["pcm_sha256"]) and (raw is None or (len(raw) == 2 * row["samples"]
                and digest(raw) == row["pcm_sha256"])), "Prepared source commitment differs")
        kind, truth = truth_for(row)
        result, directions = outcomes(row, record["windows"], raw)
        scene = row["scene"]
        if row["source"]:
            sources[row["source"]["speaker"]].append((scene, row["source"]))
        cell = summary[scene, kind]
        cell["planned"] += 1
        for arm in ("sparse", "dense"):
            cell[f"{arm}_with_assessed_window"] += int(any(directions[tuple(span)] != "unassessed" for span in row[arm]))
        useful = {}
        for arm in ARMS:
            useful[arm] = (bool(result[arm]) if kind == "full" else
                           any(correctly_located(span, edit) for span in result[arm] for edit in truth))
            cell[f"{arm}_flagged"] += int(bool(result[arm]))
            cell[f"{arm}_useful"] += int(useful[arm])
        verdicts.append((row, kind, result, useful))
    kinds = Counter()
    generators = defaultdict(Counter)
    for deliveries in sources.values():
        require({scene for scene, _ in deliveries} == SCENES and len(deliveries) == 3, "Matched speaker deliveries differ")
        _, source = deliveries[0]
        require(all(s == source for _, s in deliveries), "Matched publisher source differs")
        kinds[source["kind"]] += 1
        if source["generator"]:
            generators[source["kind"]][source["generator"]] += 1
    require(dict(kinds) == {"partial": 12, "full": 6, "genuine": 10} and len(generators["partial"]) == 6
            and set(generators["partial"]) == set(generators["full"])
            and set(generators["partial"].values()) == {2} and set(generators["full"].values()) == {1},
            "Distinct publisher speakers or generator coverage changed")
    policy, gate = freeze["rule"], {}
    for arm in ("dense", "combined"):
        hits, clean_false, noisy_false, losses = set(), set(), set(), set()
        added_families, isolated_flags = set(), 0
        for row, kind, result, useful in verdicts:
            scene, source = row["scene"], row["source"]
            if kind == "isolated":
                isolated_flags += int(bool(result[arm]))
            elif kind == "partial" and scene != "gain_control" and useful[arm] and not useful["sparse"]:
                hits.add(source["speaker"])
                added_families.add(source["generator"])
            elif kind == "genuine" and result[arm] and not result["sparse"]:
                (clean_false if scene == "gain_control" else noisy_false).add(source["speaker"])
            if (scene == "gain_control" or kind == "full") and useful["sparse"] and not useful[arm]:
                losses.add(source["speaker"])
        gate[arm] = {"added_partial_groups_under_noise": len(hits), "added_generator_families": len(added_families),
                     "added_clean_genuine_groups": len(clean_false), "added_noisy_genuine_groups": len(noisy_false),
                     "isolated_flags": isolated_flags, "lost_useful_clean_or_full_groups": len(losses),
                     "passes": len(hits) >= policy["minimum_noisy_added_groups"]
                     and len(added_families) >= policy["minimum_generator_families"]
                     and len(noisy_false) <= policy["maximum_noisy_added_genuine_groups"]
                     and len(clean_false) <= policy["maximum_clean_added_genuine_groups"]
                     and isolated_flags <= policy["maximum_isolated_flags"]
                     and len(losses) <= policy["maximum_lost_useful_dry_or_full"]}
    return {"unique_speakers": len(sources), "checked_inputs": len(roster),
            "window_decisions": sum(len(row["windows"]) for row in decisions),
            "counts": {f"{scene}/{kind}": dict(cell) for (scene, kind), cell in sorted(summary.items())}, "gate": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    export = actions.add_parser("export")
    export.add_argument("run", type=Path)
    export.add_argument("output", type=Path)
    review = actions.add_parser("review")
    review.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.action == "export":
        freeze, roster, scored = [read(args.run / f"{name}.json") for name in ("freeze", "roster", "decisions")]
        require(digest((args.run / "roster.json").read_bytes()) == freeze["roster_sha256"]
                and digest((args.run / "freeze.json").read_bytes()) == scored["freeze_sha256"]
                and digest(Path(__file__).read_bytes()) == freeze["reviewer_sha256"], "Frozen artifacts differ")
        summary = study(freeze, roster, scored["rows"], args.run / "pcm")
        args.output.write_text(json.dumps({"freeze": freeze, "roster": roster, "decisions": scored["rows"],
                                          "summary": summary}, indent=2, sort_keys=True) + "\n")
    else:
        receipt = read(args.receipt)
        roster = receipt["roster"]
        require(digest((json.dumps(roster, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
                == receipt["freeze"]["roster_sha256"], "Public source roster differs from the frozen commitment")
        summary = study(receipt["freeze"], roster, receipt["decisions"])
        require(summary == receipt["summary"], "Public outcomes differ from the replay")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
