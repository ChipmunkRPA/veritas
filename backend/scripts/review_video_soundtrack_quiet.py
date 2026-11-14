# SPDX-License-Identifier: Apache-2.0
"""Recount waveform coverage and signed locations from frozen sample measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from scripts.review_video_soundtrack_cadence import CARRIERS, EDITS, POINTS, ROLES

SCHEMA = "chipmunk.video-soundtrack-quiet/1"
METHODS = {"current": [2560, 640, .003], "short": [1600, 320, .003],
           "quiet": [2560, 640, .00075], "short_quiet": [1600, 320, .00075]}
FRESH = ("earlier", "later", "untouched", "gain", "filter", "constant", "alternate_speech",
         "silent_interval", "repeat_bed", "quiet_gain")
SHA = re.compile(r"[0-9a-f]{64}")
SETTINGS = {
    "selection": "Each new arm runs only after the current check has no location. Preserve every current location.",
    "acceptance": "Across chronological transfer excerpts in each carrier, add locations on at least two separate "
                  "recordings, keep all current locations, add at most one control flag, and add none on a "
                  "constant-offset or repeated-bed control. Add no flags on the original matched-picture controls.",
    "geometry": "The published 24 fps panel retains its tested picture decisions and exact sample hashes. "
                "The transfer changes audio only, with the same picture by construction.",
    "transfer": "Start five seconds later, or five seconds earlier when the source ends before the later clip. "
                "Take 64,000 decoded samples from a 4.25-second extraction. Compare both signed 200 ms edits "
                "and eight ordinary audio controls under one fixed quiet delivery level reduction (12 dB).",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _location(row, role, measured, method):
    result, support = measured
    width, step, _ = METHODS[method]
    require(type(support) is int and 0 <= support <= (row["samples"] - width) // step + 1,
            "Invalid waveform coverage")
    if result is None:
        return False
    a, b = result["submitted_samples"]
    c, d = result["earlier_samples"]
    shift = result["shift_samples"]
    windows = result["window_start_samples"]
    require(all(type(n) is int for n in (a, b, c, d, shift)) and a - c == b - d == shift
            and 0 <= min(a, c) < max(b, d) <= row["samples"] and b - a >= width + 3 * step
            and 1920 <= abs(shift) <= 5760 and .86 <= result["minimum_correlation"] <= 1
            and 4 <= len(windows) == result["supporting_windows"] <= support
            and windows == sorted(set(windows))
            and all(type(n) is int and n % step == 0 and 0 <= n <= row["samples"] - width for n in windows)
            and set(result["same_time_support"]) == {"before", "after"}
            and all(len(value) == 2 and all(type(n) is int and n % step == 0 and 0 <= n <= row["samples"] - width for n in value)
                    for value in result["same_time_support"].values()), "Invalid source-sample location")
    left, right = row["edit_samples"]
    expected = -3200 if role == "earlier" else 3200
    return abs(shift - expected) <= 16 and 5 * (min(b, right) - max(a, left)) >= 3 * (max(b, right) - min(a, left))


def review(doc, parent):
    freeze, rows = doc["freeze"], doc["rows"]
    old = {row["group"]: row for row in parent["rows"] if row["rate"] == 24}
    plan = {source["group"]: source for source in parent["freeze"]["sources"]}
    require(freeze["schema"] == SCHEMA and freeze["settings"] == SETTINGS
            and freeze["methods"] == METHODS and len(rows) == 2 * len(plan) == 28
            and {(r["panel"], r["group"]) for r in rows}
            == {(panel, group) for panel in ("prior", "transfer") for group in plan}, "Incomplete comparison")
    counters = {(panel, carrier, name): Counter() for panel in ("prior", "transfer")
                for carrier in CARRIERS for name in METHODS}
    additions = {key: set() for key in counters}
    for row in rows:
        group, panel, samples = (row[k] for k in ("group", "panel", "samples"))
        prior = panel == "prior"
        role_set = ROLES if prior else FRESH
        require(row["source_sha256"] == plan[group]["sha256"] and 64000 <= samples <= 65024
                and row["edit_samples"] == old[group]["edit_samples"]
                and row["offset"] == (plan[group]["offset"] if prior else freeze["transfer_offsets"][str(group)])
                and len(row["cases"]) == len(role_set) * len(CARRIERS)
                and {(c["carrier"], c["role"]) for c in row["cases"]}
                == {(carrier, role) for carrier in CARRIERS for role in role_set}, "Invalid paired source cases")
        old_cases = {(case["carrier"], case["role"]): case for case in old[group]["cases"]}
        for case in row["cases"]:
            role, carrier = case["role"], case["carrier"]
            authored = role in EDITS
            if prior:
                original = old_cases[carrier, role]
                require(not original.get("unavailable")
                        and all(case[key] == original[key] for key in ("earlier_pcm_sha256", "submitted_pcm_sha256")),
                        "The recreated sound differs from the verified picture panel")
                qualified = all(value <= POINTS[name] for grid in original["picture"].values() for name, value in grid.items())
            else:
                qualified = True
            require(case["qualified"] is qualified and set(case["arms"]) == set(METHODS)
                    and all(SHA.fullmatch(case[key]) for key in ("earlier_pcm_sha256", "submitted_pcm_sha256")),
                    "Invalid picture qualification or waveform hash")
            require(type(case["samples"]) is int and samples <= case["samples"] <= samples + 1024
                    and abs(row["offset"] - plan[group]["offset"]) == (0 if prior else 5), "Invalid delivery clock")
            hits = {name: _location({**row, "samples": case["samples"]}, role, arm, name) for name, arm in case["arms"].items()}
            current = case["arms"]["current"][0]
            if prior:
                require(current == original["sound"], "The current measure differs from the original result")
            for name, arm in case["arms"].items():
                key = (panel, carrier, name)
                total = counters[key]
                total["planned_edits" if authored else "planned_controls"] += 1
                total["eligible_edits" if authored else "eligible_controls"] += qualified
                hit = qualified and (hits["current"] if current else hits[name])
                flag = qualified and (current is not None or arm[0] is not None)
                if authored:
                    total["useful"] += hit
                    total["misplaced"] += flag and not hit
                    if hit and not hits["current"]:
                        total["added"] += 1
                        additions[key].add(group)
                else:
                    total["flags"] += flag
                    total["added_flags"] += flag and current is None
                    total["prohibited_flags"] += flag and current is None and role in {
                        "constant", "repeat_bed", "synchronized_picture", "synchronized_mouth"}
    outcome = {"/".join(key): dict(value) | {"added_recordings": len(additions[key])} for key, value in counters.items()}
    outcome["admit"] = [name for name in METHODS if name != "current" and all(
        len(additions["transfer", carrier, name]) >= 2 and counters["transfer", carrier, name]["added_flags"] <= 1
        and not counters["transfer", carrier, name]["prohibited_flags"]
        and not counters["prior", carrier, name]["added_flags"] for carrier in CARRIERS)]
    return outcome


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("receipt", type=Path)
    cli.add_argument("parent", type=Path)
    args = cli.parse_args()
    saved, parent = (json.loads(path.read_bytes()) for path in (args.receipt, args.parent))
    require(hashlib.sha256(args.parent.read_bytes()).hexdigest() == saved["freeze"]["parent_sha256"], "Parent receipt differs")
    result = review(saved, parent)
    require(result == saved["outcome"], "Published result differs from replay")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
