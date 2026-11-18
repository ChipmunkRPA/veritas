# SPDX-License-Identifier: MIT
"""Recount the exact-source voice diagnostic without speech or models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

from scripts.review_audio_severe_overlap_voice_diagnostic import RATE, STEP, WINDOW, readings
from scripts.review_audio_two_second_opus_transfer import matches, require

SCHEMA = "chipmunk.audio-exact-foreground/1"
PARENT_NAME = "audio-severe-overlap-voice-diagnostic-2026-11-18.json.gz"
ROLES = ("edited", "ordinary", "background")
TIERS = ("strict", "wider")
LIMIT = 64


def assess(flags, starts, bit):
    proposals = set()
    for index, start in enumerate(starts):
        first, second = flags[2 * index : 2 * index + 2]
        if first & second & bit:
            proposals.update((offset, offset + WINDOW) for offset in (start, start + STEP))
    return proposals


def admitted(flags):
    return sum(bool(left and right) for left, right in zip(flags[::2], flags[1::2], strict=True))


def logit_margin(edited, ordinary, pair):
    values = (*edited[2 * pair : 2 * pair + 2], *ordinary[2 * pair : 2 * pair + 2])
    return None if None in values else min(values[:2]) - min(values[2:])


def scope_labels(labels, starts):
    for index, label in enumerate(labels):
        pair_indices = [
            i
            for i, start in enumerate(starts)
            if matches([(start, start + WINDOW), (start + STEP, start + STEP + WINDOW)], [label])
        ]
        if pair_indices:
            yield index, pair_indices


def flags_for(values, points):
    return [
        0 if value is None else 1 + 2 * (value > points["wider"]) + 4 * (value > points["strict"])
        for value in values
    ]


def summarize(receipt, parent):
    require(
        receipt["schema"] == SCHEMA
        and parent["schema"] == "chipmunk.audio-severe-overlap-voice-diagnostic/1"
        and parent["role_order"] == list(ROLES)
        and (parent["sample_rate_hz"], parent["step_samples"], parent["window_samples"])
        == (RATE, STEP, WINDOW)
        and receipt["forward_budget"] == LIMIT,
        "Unexpected diagnostic or parent format",
    )
    points = {tier: float.fromhex(parent["raw_calibration_hex"][tier]) for tier in TIERS}
    require(
        all(map(math.isfinite, points.values())) and points["strict"] >= points["wider"], "Invalid points"
    )
    groups = receipt["groups"]
    require(len(groups) == len(parent["matched_groups"]) == 11, "Expected eleven source groups")
    scope = Counter(all_labels=0, geometrically_reachable=0, one_to_one_ceiling=0, scheduled_pairs=0)
    scope.update(
        {
            f"{quality}_{view}_{role}": 0
            for quality in ("files_assessed", "pairs_admitted")
            for view in ("mixture", "exact_foreground")
            for role in ROLES[:2]
        }
    )
    metrics = (
        "located",
        "control_clear",
        "ordinary_alerts",
        "background_alerts",
        "edited_alerts",
        "wrong_time_only",
    )
    results = {
        tier: {view: Counter(dict.fromkeys(metrics, 0)) for view in ("mixture", "exact_foreground")}
        for tier in TIERS
    }
    additions = {tier: [] for tier in TIERS}
    losses = {tier: [] for tier in TIERS}
    margins = {"mixture": [], "exact_foreground": []}
    possible = measured = 0
    for slot, (current, earlier) in enumerate(zip(groups, parent["matched_groups"], strict=True), 1):
        exact, reduced = current
        size, labels, roles = earlier
        starts = [4 * RATE, 6 * RATE] if slot <= 5 else [4 * RATE]
        grid = [start + offset for start in starts for offset in (0, STEP)]
        require(grid[-1] + WINDOW <= size, "Frozen window exceeds the common clip clock")
        require(
            len(exact) == 2
            and all(
                len(row) == len(grid) and all(type(flag) is int and flag in (0, 1, 3, 7) for flag in row)
                for row in exact
            ),
            "Invalid exact-foreground admission or decision bits",
        )
        mixture_values = [tuple(readings(views[0], grid)) for _, views, _ in roles]
        mixture = [flags_for(values, points) for values in mixture_values]
        available = list(scope_labels(labels, starts))
        require(
            [row[0] for row in reduced] == [index for index, _ in available],
            "Wrong reduced measurement scope",
        )
        scope.update(
            all_labels=len(labels),
            geometrically_reachable=len(available),
            scheduled_pairs=len(starts),
            one_to_one_ceiling=len(matches([(t, t + WINDOW) for t in grid], labels)),
        )
        for view, flags in (("mixture", mixture), ("exact_foreground", exact)):
            for role, bits in zip(ROLES[:2], flags[:2], strict=True):
                count = admitted(bits)
                scope.update(
                    {f"pairs_admitted_{view}_{role}": count, f"files_assessed_{view}_{role}": bool(count)}
                )
        possible += sum(map(len, exact))
        measured += sum(bool(flag) for flags in exact for flag in flags)
        for tier, bit in (("strict", 4), ("wider", 2)):
            clear = {}
            for view, flags in (("mixture", mixture), ("exact_foreground", [*exact, mixture[2]])):
                edited, ordinary, background = (assess(bits, starts, bit) for bits in flags)
                hit = matches(sorted(edited), labels)
                controls = sorted(ordinary | background)
                clear[view] = {index for index in hit if not matches(controls, [labels[index]])}
                results[tier][view].update(
                    located=len(hit),
                    control_clear=len(clear[view]),
                    ordinary_alerts=bool(ordinary),
                    background_alerts=bool(background),
                    edited_alerts=bool(edited),
                    wrong_time_only=bool(edited) and not hit,
                )
            additions[tier].extend(
                [slot, index] for index in sorted(clear["exact_foreground"] - clear["mixture"])
            )
            losses[tier].extend(
                [slot, index] for index in sorted(clear["mixture"] - clear["exact_foreground"])
            )
        for (_, witnesses), (_, differences) in zip(available, reduced, strict=True):
            require(len(differences) == len(witnesses), "Missing measured margin")
            for view, headers in (("mixture", mixture), ("exact_foreground", exact)):
                engaged = [
                    pair
                    for pair in witnesses
                    if all(track[2 * pair] and track[2 * pair + 1] for track in headers[:2])
                ]
                if view == "mixture":
                    values = [logit_margin(*mixture_values[:2], pair) for pair in engaged]
                else:
                    require(
                        [
                            pair
                            for pair, value in zip(witnesses, differences, strict=True)
                            if value is not None
                        ]
                        == engaged,
                        "Exact-foreground margin and admission disagree",
                    )
                    values = [float.fromhex(value) for value in differences if value is not None]
                    require(all(map(math.isfinite, values)), "Invalid paired margin")
                if values:
                    margins[view].append(max(values))
    require(possible == LIMIT and measured <= LIMIT, "Invalid exact-source forward budget")
    return {
        "scope": dict(scope),
        "actual_forwards": measured,
        "results": results,
        "control_clear_additions": additions,
        "control_clear_losses": losses,
        "margins": {
            view: {
                "comparable_labels": len(values),
                "edited_higher": sum(value > 0 for value in values),
                "median_logit": round(median(values), 6) if values else None,
            }
            for view, values in margins.items()
        },
    }


def review(receipt, parent_bytes):
    require(
        hashlib.sha256(parent_bytes).hexdigest() == receipt["commitments_sha256"]["parent"],
        "The published timing and mixture ledger differs",
    )
    result = summarize(receipt, json.loads(gzip.decompress(parent_bytes)))
    require(result == receipt["expected"], "Exact-source decisions disagree with the saved result")
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
