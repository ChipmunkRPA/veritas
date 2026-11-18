# SPDX-License-Identifier: MIT
"""Recount the periodic cancellation diagnostic without audio or models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

from scripts.review_audio_exact_foreground import PARENT_NAME, ROLES, admitted, assess, flags_for
from scripts.review_audio_severe_overlap_voice_diagnostic import RATE, STEP, WINDOW, readings
from scripts.review_audio_severe_overlap_voice_diagnostic import SCHEMA as PARENT_SCHEMA
from scripts.review_audio_two_second_opus_transfer import matches, require

SCHEMA = "chipmunk.audio-periodic-cancellation/1"
VIEWS = ("mixture", "attenuated", "either")
POINT_BITS = {"strict": 4, "wider": 2}
LEVEL_SCALE = 1_000_000
METRICS = ("located", "control_clear", "ordinary_alerts", "background_alerts", "wrong_time_only")


def evaluate(proposals, labels):
    edited, ordinary, background = proposals
    hit = matches(sorted(edited), labels)
    collisions = {
        i: [
            role
            for role, control in zip(ROLES[1:], (ordinary, background), strict=True)
            if matches(sorted(control), [labels[i]])
        ]
        for i in hit
    }
    return (
        hit,
        collisions,
        {
            "located": len(hit),
            "control_clear": sum(not found for found in collisions.values()),
            "ordinary_alerts": bool(ordinary),
            "background_alerts": bool(background),
            "wrong_time_only": bool(edited) and not hit,
        },
    )


def summarize(receipt, parent):
    require(
        receipt["schema"] == SCHEMA
        and parent["schema"] == PARENT_SCHEMA
        and parent["role_order"] == list(ROLES)
        and (parent["sample_rate_hz"], parent["step_samples"], parent["window_samples"])
        == (RATE, STEP, WINDOW),
        "Unexpected periodic cancellation or parent format",
    )
    points = {tier: float.fromhex(parent["raw_calibration_hex"][tier]) for tier in POINT_BITS}
    groups = receipt["groups"]
    require(len(groups) == len(parent["matched_groups"]) == 11, "Expected eleven matched groups")
    counters = {tier: {view: Counter(dict.fromkeys(METRICS, 0)) for view in VIEWS} for tier in POINT_BITS}
    added = {tier: [] for tier in POINT_BITS}
    lost = {tier: [] for tier in POINT_BITS}
    pairs = dict.fromkeys(ROLES, 0)
    levels = []
    possible = forwards = reachable = 0
    for slot, ((attenuated, changes), (size, labels, roles)) in enumerate(
        zip(groups, parent["matched_groups"], strict=True), 1
    ):
        starts = [4 * RATE, 6 * RATE] if slot <= 5 else [4 * RATE]
        grid = [start + offset for start in starts for offset in (0, STEP)]
        require(grid[-1] + WINDOW <= size, "Frozen window exceeds the common source clock")
        require(
            len(attenuated) == len(ROLES)
            and all(
                len(v) == len(grid) and all(type(n) is int and n in (0, 1, 3, 7) for n in v)
                for v in attenuated
            ),
            "Invalid window admission or decision bits",
        )
        require(
            len(changes) == len(labels)
            and all(len(v) == 2 and all(type(n) is int for n in v) for v in changes),
            "Each known interval needs two measured level changes",
        )
        levels.extend(changes)
        mixture = [flags_for(readings(views[0], grid), points) for _, views, _ in roles]
        for role, flags in zip(ROLES, attenuated, strict=True):
            pairs[role] += admitted(flags)
            possible += len(flags)
            forwards += sum(map(bool, flags))
        reachable += sum(bool(matches([(t, t + WINDOW) for t in grid], [label])) for label in labels)
        for tier, bit in POINT_BITS.items():
            baseline = [assess(v, starts, bit) for v in mixture]
            filtered = [assess(v, starts, bit) for v in attenuated]
            merged = [old | new for old, new in zip(baseline, filtered, strict=True)]
            summary = [evaluate(p, labels) for p in (baseline, filtered, merged)]
            for view, (_, _, counts) in zip(VIEWS, summary, strict=True):
                counters[tier][view].update(counts)
            (before, _, _), (after, collisions, _), _ = summary
            added[tier].extend([slot, labels[i], collisions[i]] for i in sorted(after - before))
            lost[tier].extend([slot, labels[i]] for i in sorted(before - after))
    require(len(levels) == 75 and reachable == 8 and forwards <= possible == 96, "Unexpected trial scope")
    source, interference = zip(*levels, strict=True)
    ratios = [a - b for a, b in levels]
    return {
        "forwards": forwards,
        "budget": possible,
        "labels": len(levels),
        "reachable": reachable,
        "admitted_pairs": pairs,
        "results": counters,
        "added": added,
        "lost": lost,
        "acoustic": {
            "median_foreground_db": median(source) / LEVEL_SCALE,
            "median_interference_db": median(interference) / LEVEL_SCALE,
            "median_ratio_db": median(ratios) / LEVEL_SCALE,
            "improved": sum(v > 0 for v in ratios),
            "improved_over_one_db": sum(v > LEVEL_SCALE for v in ratios),
            "worsened_over_one_db": sum(v < -LEVEL_SCALE for v in ratios),
        },
    }


def review(receipt, parent_bytes):
    require(
        hashlib.sha256(parent_bytes).hexdigest() == receipt["commitments_sha256"]["parent"],
        "The published timing and mixture ledger differs",
    )
    result = summarize(receipt, json.loads(gzip.decompress(parent_bytes)))
    require(result == receipt["expected"], "Periodic cancellation decisions disagree with the saved result")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("parent", type=Path, nargs="?")
    args = parser.parse_args()
    try:
        result = review(
            json.loads(gzip.decompress(args.receipt.read_bytes())),
            (args.parent or args.receipt.with_name(PARENT_NAME)).read_bytes(),
        )
    except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
