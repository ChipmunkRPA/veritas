# SPDX-License-Identifier: MIT
"""Recount two-channel projection decisions from anonymous window bits."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.review_audio_exact_foreground import PARENT_NAME, assess
from scripts.review_audio_severe_overlap_voice_diagnostic import RATE, STEP, WINDOW
from scripts.review_audio_two_second_opus_transfer import matches, require

SCHEMA = "chipmunk.audio-stereo-transfer-projection/1"
STARTS = (24_000, 56_000)
PILOTS = ("exactly_two_modes", "strongest_two_modes")
LANES = ("mono", "projected", "either")
TIERS = {"strict": 4, "wider": 2}
METRICS = ("located", "control_clear", "untouched_flags", "ordinary_edit_flags", "wrong_time_only")


def review_pilot(cases):
    require(len(cases) == 4, "Expected four matched groups")
    counts = {tier: {lane: Counter(dict.fromkeys(METRICS, 0)) for lane in LANES} for tier in TIERS}
    grid = [(start + offset, start + offset + WINDOW) for start in STARTS for offset in (0, STEP)]
    reached = ceiling = calls = projected = 0
    for labels, roles in cases:
        require(
            len(labels) == 2
            and labels == sorted(labels)
            and all(type(a) is int and type(b) is int and 0 <= a < b <= 6 * RATE for a, b in labels),
            "Invalid source-time labels",
        )
        require(
            len(roles) == 3
            and all(len(tracks) in (1, 2, 3) for tracks in roles)
            and all(
                len(flags) == len(grid) and all(type(bit) is int and bit in (0, 1, 3, 7) for bit in flags)
                for tracks in roles
                for flags in tracks
            ),
            "Invalid role or window admission bits",
        )
        reached += sum(bool(matches(grid, [label])) for label in labels)
        ceiling += len(matches(grid, labels))
        calls += sum(bool(bit) for tracks in roles for flags in tracks for bit in flags)
        projected += sum(len(tracks) > 1 for tracks in roles)
        for tier, bit in TIERS.items():
            mono, projected_locations = [], []
            for first, *alternatives in roles:
                mono.append(assess(first, STARTS, bit))
                projected_locations.append(
                    set().union(*(assess(flags, STARTS, bit) for flags in alternatives))
                )
            either = [a | b for a, b in zip(mono, projected_locations, strict=True)]
            for lane, (edited, untouched, ordinary_edit) in zip(
                LANES, (mono, projected_locations, either), strict=True
            ):
                found = matches(sorted(edited), labels)
                controls = sorted(untouched | ordinary_edit)
                counts[tier][lane].update(
                    located=len(found),
                    control_clear=sum(not matches(controls, [labels[index]]) for index in found),
                    untouched_flags=bool(untouched),
                    ordinary_edit_flags=bool(ordinary_edit),
                    wrong_time_only=bool(edited) and not found,
                )
    return {
        "labels": 2 * len(cases),
        "individually_reachable": reached,
        "one_to_one_ceiling": ceiling,
        "model_calls": calls,
        "files_with_projection": projected,
        "results": counts,
    }


def review(receipt, parent_bytes):
    parent = json.loads(gzip.decompress(parent_bytes))
    require(
        receipt["schema"] == SCHEMA
        and receipt["grid"] == [6 * RATE, RATE, WINDOW, STEP, *STARTS]
        and receipt["parent_sha256"] == hashlib.sha256(parent_bytes).hexdigest()
        and receipt["operating_points_hex"] == parent["raw_calibration_hex"]
        and set(receipt["pilots"]) == set(PILOTS),
        "The frozen schedule or parent calibration differs",
    )
    recounted = {pilot: review_pilot(receipt["pilots"][pilot]["cases"]) for pilot in PILOTS}
    require(recounted == receipt["expected"], "Recounted stereo decisions differ")
    return recounted


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
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
