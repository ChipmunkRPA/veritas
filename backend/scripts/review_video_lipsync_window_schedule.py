# SPDX-License-Identifier: Apache-2.0
"""Replay the short-video timing decisions from saved window outputs."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.video-lipsync-window-schedule/1"
ARMS = {"current": (12.5, 15), "denser": (5, 15), "narrower": (12.5, 8), "denser_narrower": (5, 8)}
CASES = ("original", "gain", "lowpass", "noise", "global_early", "global_late", "local_early", "local_late")
SHA = re.compile(r"[0-9a-f]{64}")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _useful(ranges, truth, signed_lag):
    if truth is None:
        return False
    left, right = truth
    for _, start, end, lag in ranges:
        # Stored segment endpoints are half-ticks. A useful result needs
        # at least 0.40 IoU and the signed lag within 80 ms.
        start, end = start * 320, end * 320
        if lag is not None and abs(lag - signed_lag) <= 2 and (
                5 * (min(end, right) - max(start, left)) >= 2 * (max(end, right) - min(start, left))):
            return True
    return False


def review(record):
    planned = record["sources"]
    observed = record["confirmations"]
    _require(record["schema"] == SCHEMA and len(planned) == len(observed) == 26
             and len({row["sha256"] for row in planned}) == 26,
             "The locked panel needs 26 binding-distinct source recordings")
    banks = {"acted": Counter(), "conversation": Counter()}
    development = {name: Counter() for name in ARMS}
    outcomes = {bank: {name: Counter() for name in ARMS} for bank in banks}
    useful = {bank: {name: set() for name in ARMS} for bank in banks}
    flagged_controls = {bank: {name: set() for name in ARMS} for bank in banks}
    for source, group in zip(planned, observed, strict=True):
        bank, identifier = source["bank"], source["group"]
        _require(bank == group["bank"] and identifier == group["group"] and SHA.fullmatch(source["sha256"]),
                 "A result differs from the source group order")
        is_held = source["split"] == "held"
        if is_held:
            banks[bank].update({"groups": 1, "planned_edits": 2, "planned_controls": 6})
        if "unavailable" in group:
            if is_held:
                banks[bank]["unavailable_groups"] += 1
            continue
        cases = group["cases"]
        ticks = group["pcm_samples"] // 640
        _require(len(cases) == len(CASES) and {case["case"] for case in cases} == set(CASES),
                 "The paired group lacks a recorded condition")
        for case in cases:
            role, truth, lag = (case[key] for key in ("case", "truth_samples", "truth_lag_ticks"))
            _require(SHA.fullmatch(case["pcm_sha256"]) and set(case["arms"]) == set(ARMS),
                     "The case lacks its audio or comparison commitment")
            if role.startswith("local_"):
                _require(type(lag) is int and abs(lag) in (5, 7) and (lag < 0) == role.endswith("early")
                         and 0 <= truth[0] < truth[1] <= group["pcm_samples"]
                         and (truth[1] - truth[0]) // 640 in (16, 22), "Invalid authored local edit")
            else:
                _require(truth is None and lag is None, "A control has unexpected local edit truth")
            for name, arm in case["arms"].items():
                starts, ranges = arm["starts"], arm["ranges"]
                hop, margin = ARMS[name]
                schedule = {int(hop * index) for index in range(ticks)}
                _require(starts == sorted(set(starts)) and all(type(i) is int and i in schedule
                         and margin <= i and i + 25 + margin <= ticks for i in starts),
                         "A measured window exceeds its symmetric search scope")
                _require(all(kind in {"offset_step", "confidence_collapse"} and type(a) is int and type(b) is int
                             and starts and 2 * starts[0] <= a < b <= 2 * (starts[-1] + 25)
                             and ((lag_ticks is None and kind == "confidence_collapse") or
                                  (kind == "offset_step" and type(lag_ticks) is int and 3 <= abs(lag_ticks) <= 2 * margin))
                             for kind, a, b, lag_ticks in ranges),
                         "A selected interval differs from its measured windows")
                stats = outcomes[bank][name] if is_held else development[name]
                stats["measured_edits" if truth else "measured_controls"] += bool(starts)
                indices = [index for index in range(ticks) if int(index * hop) in starts]
                stats["four_consecutive_edits" if truth else "four_consecutive_controls"] += any(
                    last - first == 3 for first, last in zip(indices, indices[3:], strict=False))
                if truth:
                    hit = _useful(ranges, truth, lag)
                    stats["useful"] += hit
                    stats["wrong_locations_or_lags"] += bool(ranges) and not hit
                    if hit and is_held:
                        useful[bank][name].add((identifier, role))
                else:
                    stats["control_flags"] += bool(ranges)
                    if ranges and is_held:
                        flagged_controls[bank][name].add((identifier, role))
    results = {}
    for bank in banks:
        results[bank] = {"scope": dict(banks[bank]), "arms": {}}
        for name in ARMS:
            additions = useful[bank][name] - useful[bank]["current"]
            lost = useful[bank]["current"] - useful[bank][name]
            new_flags = flagged_controls[bank][name] - flagged_controls[bank]["current"]
            results[bank]["arms"][name] = dict(outcomes[bank][name]) | {
                "added_useful": len(additions), "added_groups": len({group for group, _ in additions}),
                "lost_useful": len(lost), "added_control_flags": len(new_flags),
                "added_global_shift_flags": sum(role.startswith("global_") for _, role in new_flags),
            }
    for name in ARMS:
        arms = [results[bank]["arms"][name] for bank in banks]
        results.setdefault("advances", {})[name] = (sum(arm["added_useful"] for arm in arms) >= 4
            and sum(arm["added_groups"] for arm in arms) >= 3 and all(arm["added_groups"] > 0 for arm in arms)
            and sum(arm["added_control_flags"] for arm in arms) <= 1
            and all(arm["lost_useful"] == 0 and arm["added_global_shift_flags"] == 0 for arm in arms)
            and all(4 * results[bank]["arms"][name].get("measured_edits", 0) >= 3 * banks[bank]["planned_edits"]
                    for bank in banks))
    results["development"] = {name: dict(counts) for name, counts in development.items()}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    doc = json.loads(parser.parse_args().receipt.read_text())
    outcome = review(doc)
    _require(outcome == doc["outcome"], "The published timing outcome does not recompute")
    print(json.dumps(outcome, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
