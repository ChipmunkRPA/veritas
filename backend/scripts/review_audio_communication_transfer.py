"""Reconstruct the fixed communication-study decision counts using only the standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def choice(windows):
    return max((w for w in windows if w["score"] is not None), key=lambda w: w["score"], default=None)


def locate(winner, truth):
    a, b = winner["start"], winner["end"]
    start, end = truth
    overlap = max(0, min(b, end) - max(a, start))
    return 2 * overlap >= end - start and 10 * overlap >= 3 * (b - a + end - start - overlap)


def review(run, parent):
    freeze, summary = (load(run / f"{name}.json") for name in ("freeze", "aggregate"))
    check(all(digest((run / f"{name}.json").read_bytes()) == value for name, value in summary["commitments"].items()),
          "Result commitments changed")
    check(all(digest((parent / name).read_bytes()) == value for name, value in freeze["sources"].items()), "Source commitments changed")
    rows, original = load(parent / "roster.json"), load(parent / "current.json")["records"]
    deliveries, scored = load(run / "inputs.json"), load(run / "scores.json")["rows"]
    check(len(rows) == len(original) == len(deliveries) == len(scored) == 52, "Panel size differs")
    counters = {arm: {"ai_flags": 0, "useful_ranges": 0, "unassessed": 0, "background_false_flags": 0}
                for arm in ("original", "communication", "both")}
    false = {arm: set() for arm in counters}
    useful = {arm: {} for arm in counters}
    for row, native, delivery, processed in zip(rows, original, deliveries, scored, strict=True):
        key = row["id"]
        check(key == native["id"] == delivery["id"] == processed["id"], "Score assigned to wrong source")
        before = (parent / "pcm" / f"{key}.pcm").read_bytes()
        encoded = run / "deliveries" / delivery["file"]
        after = encoded.with_suffix(".pcm").read_bytes()
        check(len(before) == len(after) == 192000 and digest(before) == row["pcm_sha256"] == delivery["source_pcm_sha256"]
              and digest(after) == delivery["delivered_pcm_sha256"] and digest(encoded.read_bytes()) == delivery["encoded_sha256"],
              "Delivered or original media binding changed")
        paired = []
        for a, b, extent in zip(native["windows"], processed["windows"], ((16000, 48000), (48000, 80000)), strict=True):
            start, end = extent
            check((a["start"], a["end"]) == (b["start"], b["end"]) == extent
                  and a["pcm_sha256"] == digest(before[2 * start:2 * end])
                  and b["pcm_sha256"] == digest(after[2 * start:2 * end]), "Window commitment differs")
            if a["score"] is not None and b["score"] is not None:
                paired.append({"start": start, "end": end, "score": min(a["score"], b["score"])})
        for arm, windows in zip(counters, (native["windows"], processed["windows"], paired), strict=True):
            winner = choice(windows)
            if winner is None:
                counters[arm]["unassessed"] += 1
            elif winner["score"] > freeze["cutoff"]:
                if row["truth"] is None:
                    false[arm].add(row["group"])
                    counters[arm]["background_false_flags"] += row["arm"] == "background_only"
                else:
                    counters[arm]["ai_flags"] += 1
                    if locate(winner, row["truth"]):
                        counters[arm]["useful_ranges"] += 1
                        useful[arm][key] = [winner["start"], winner["end"]]
    for arm, count in counters.items():
        count["false_groups"] = len(false[arm])
    check(counters == summary["tables"]["all"]["fixed"], "Fixed outcome arithmetic differs")
    retained = sum(useful["both"].get(key) == span for key, span in useful["original"].items())
    return {"file_pairs_verified": len(rows), "windows_verified": len(rows) * 4, "fixed_arms_verified": len(counters),
            "exact_useful_original_locations_retained": retained, "useful_original_locations": len(useful["original"]),
            "original_false_groups_removed": len(false["original"] - false["both"]),
            "new_false_groups": len(false["both"] - false["original"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("parent", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.run, args.parent), sort_keys=True))


if __name__ == "__main__":
    main()
