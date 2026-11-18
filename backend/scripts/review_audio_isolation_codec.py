# SPDX-License-Identifier: MIT
"""Recount speech-isolation and repeat-encoding decisions without audio."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.review_audio_exact_foreground import (
    PARENT_NAME,
    ROLES,
    admitted,
    assess,
    flags_for,
    scope_labels,
)
from scripts.review_audio_severe_overlap_voice_diagnostic import RATE, STEP, WINDOW, readings
from scripts.review_audio_two_second_opus_transfer import matches, require

SCHEMA = "chipmunk.audio-isolation-by-codec/1"
LANES = ("mix_pcm", "mix_opus", "target_pcm", "target_opus", "repeat_opus")
STORED = ("mix_pcm", "target_pcm", "target_opus", "repeat_opus")
METRICS = ("located", "control_clear", "ordinary", "background", "wrong_time_only")


def _flags(rows, roles, size):
    return (len(rows) == roles and all(len(row) == size for row in rows)
            and all(type(value) is int and value in (0, 1, 3, 7) for row in rows for value in row))


def summarize(receipt, parent):
    require(receipt["schema"] == SCHEMA
            and parent["schema"] == "chipmunk.audio-severe-overlap-voice-diagnostic/1"
            and parent["role_order"] == list(ROLES)
            and (parent["sample_rate_hz"], parent["window_samples"], parent["step_samples"]) == (RATE, WINDOW, STEP)
            and len(receipt["groups"]) == len(parent["matched_groups"]) == 11,
            "Wrong codec comparison or parent timing receipt")
    points = {key: float.fromhex(parent["raw_calibration_hex"][key]) for key in ("strict", "wider")}
    result = {point: {lane: Counter(dict.fromkeys(METRICS, 0)) for lane in LANES} for point in points}
    reach, forwards, budgets, coverage = Counter(), Counter(), Counter(), Counter()
    by_location = {lane: {point: set() for point in points} for lane in LANES}
    for slot, (item, archived) in enumerate(zip(receipt["groups"], parent["matched_groups"], strict=True), 1):
        samples, edits, old = archived
        starts = [4 * RATE, 6 * RATE] if slot <= 5 else [4 * RATE]
        grid = [start + offset for start in starts for offset in (0, STEP)]
        require(grid[-1] + WINDOW <= samples and set(item) == set(STORED), "The plan changes a native source clock")
        require(all(_flags(item[lane], 2 if lane.startswith("target") else 3, len(grid)) for lane in STORED),
                "Wrong role count, frozen crop count, or decision bits")
        direct_opus = [flags_for(list(readings(views[0], grid)), points) for _, views, _ in old]
        reach["labels"] += len(edits)
        reach["reachable"] += sum(1 for _ in scope_labels(edits, starts))
        reach["one_to_one_ceiling"] += len(matches([(start, start + WINDOW) for start in grid], edits))
        reach["scheduled_pairs_per_role"] += len(starts)

        for saved, original in zip(item["repeat_opus"], direct_opus, strict=True):
            for index in range(0, len(grid), 2):
                allowed = bool(original[index] & original[index + 1] & 2)
                budgets["repeat_opus"] += 2 * allowed
                require(allowed or saved[index:index + 2] == [0, 0], "Repeat encode scored an unselected window")
        for name in STORED:
            forwards[name] += sum(bool(bit) for role in item[name] for bit in role)
            if name != "repeat_opus":
                budgets[name] += sum(map(len, item[name]))

        roles = {
            "mix_pcm": item["mix_pcm"],
            "mix_opus": direct_opus,
            "target_pcm": [*item["target_pcm"], item["mix_pcm"][2]],
            "target_opus": [*item["target_opus"], direct_opus[2]],
            "repeat_opus": item["repeat_opus"],
        }
        for name, flags in roles.items():
            for role, bitset in zip(ROLES, flags, strict=True):
                coverage[f"{name}_{role}"] += admitted(bitset)
        for point, bit in (("strict", 4), ("wider", 2)):
            previous = [assess(flags, starts, bit) for flags in direct_opus]
            for name, flags in roles.items():
                proposals = [assess(bitset, starts, bit) for bitset in flags]
                if name == "repeat_opus":
                    proposals = [raw & repeat for raw, repeat in zip(previous, proposals, strict=True)]
                edited, ordinary, background = proposals
                location = matches(sorted(edited), edits)
                clear = {index for index in location if not matches(sorted(ordinary | background), [edits[index]])}
                by_location[name][point].update((slot, index) for index in clear)
                result[point][name].update(located=len(location), control_clear=len(clear), ordinary=bool(ordinary),
                                           background=bool(background), wrong_time_only=bool(edited) and not location)
    require(reach == {"labels": 75, "reachable": 8, "one_to_one_ceiling": 8, "scheduled_pairs_per_role": 16}
            and sum(budgets[key] for key in STORED if key != "repeat_opus") == 224
            and all(forwards[key] <= budgets[key] for key in STORED),
            "Measured coverage differs from the predeclared comparison")
    changes = {}
    for name, control in (("target_pcm", "mix_pcm"), ("target_opus", "mix_opus"), ("repeat_opus", "mix_opus")):
        changes[name] = {
            point: {"added": [list(x) for x in sorted(by_location[name][point] - by_location[control][point])],
                    "lost": [list(x) for x in sorted(by_location[control][point] - by_location[name][point])]}
            for point in points
        }
    return {"scope": dict(reach), "points": points, "forwards": dict(forwards), "forward_limits": dict(budgets),
            "admitted_pairs": dict(coverage), "results": result, "changes": changes}


def review(receipt, parent_bytes):
    require(hashlib.sha256(parent_bytes).hexdigest() == receipt["commitments_sha256"]["parent"],
            "The published source-time and mixture ledger differs")
    result = summarize(receipt, json.loads(gzip.decompress(parent_bytes)))
    require(result == receipt["expected"], "Recounted codec comparison differs from the saved result")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("parent", type=Path, nargs="?")
    args = parser.parse_args()
    try:
        result = review(json.loads(gzip.decompress(args.receipt.read_bytes())),
                        (args.parent or args.receipt.with_name(PARENT_NAME)).read_bytes())
    except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
