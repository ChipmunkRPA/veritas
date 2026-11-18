# SPDX-License-Identifier: MIT
"""Recount the severe-overlap voice diagnostic without audio or models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

from scripts.review_audio_two_second_opus_transfer import matches, require

SCHEMA = "chipmunk.audio-severe-overlap-voice-diagnostic/1"
RATE, STEP, WINDOW = 16_000, 8_000, 12_000
ROLES = ("edited", "ordinary", "background")
ARMS = ("mixture", "two_output_any", "mixture_plus_outputs")
LIMIT = 192


def readings(windows, starts):
    for speech, score_hex in windows:
        require(type(speech) is int and 0 <= speech <= WINDOW, "Invalid admitted speech length")
        require((score_hex is not None) == (2 * speech >= WINDOW), "Reading disagrees with speech admission")
        value = None if score_hex is None else float.fromhex(score_hex)
        require(value is None or math.isfinite(value), "Invalid saved reading")
        yield value
    require(len(windows) == len(starts), "Reading does not cover the frozen sample positions")


def assess(views, starts, point):
    admitted, proposals = set(), set()
    for index, start in enumerate(starts):
        first, second = views[2 * index : 2 * index + 2]
        if first is not None and second is not None:
            admitted.add(start)
            if min(first, second) > point:
                proposals.update((offset, offset + WINDOW) for offset in (start, start + STEP))
    return admitted, proposals


def group_summary(record, starts, points, parent):
    size, labels, roles = record
    require(type(size) is int and size >= starts[-1] + STEP + WINDOW, "Pair exceeds audio sample length")
    require(
        labels
        and all(type(a) is int and type(b) is int and 0 <= a < b <= size for a, b in labels)
        and all(left[1] <= right[0] for left, right in pairwise(labels)),
        "Invalid edit label",
    )
    require(len(roles) == len(ROLES), "Incomplete matched recordings")
    grid = [start + offset for start in starts for offset in (0, STEP)]
    scope = {(start, start + WINDOW) for start in grid}
    reachable = sum(bool(matches(sorted(scope), [label])) for label in labels)
    ceiling = len(matches(sorted(scope), labels))
    summary = {tier: defaultdict(Counter) for tier in points}
    forwards = elapsed = 0
    proposals = defaultdict(dict)
    for role, (clock, views, seconds) in zip(ROLES, roles, strict=True):
        pts, preskip, granule, packets, trim = clock
        require(
            pts == 3 * size == granule - preskip
            and preskip == 312
            and granule + trim == packets * 960
            and 0 <= trim < 960,
            "Incorrect native Opus packet clock",
        )
        require(len(views) == 3 and math.isfinite(seconds) and seconds > 0, "Invalid saved measurements")
        mixture, first, second = [tuple(readings(view, grid)) for view in views]
        earlier = parent[role]
        require(
            [size, *clock] == earlier["clock"] and earlier["truth"] == (labels if role == "edited" else []),
            "Clocks or edit labels differ from the earlier published inputs",
        )
        published_raw = {start: raw for start, raw, _ in earlier["windows"]}
        flags = [
            0 if value is None else 1 + 2 * (value > points["wider"]) + 4 * (value > points["strict"])
            for value in mixture
        ]
        require(
            flags == [published_raw[start] for start in grid],
            "Mixture responses differ from the earlier published comparison",
        )
        forwards += sum(value is not None for track in (first, second) for value in track)
        elapsed += seconds
        for tier, point in points.items():
            raw_admitted, raw = assess(mixture, starts, point)
            first_admitted, a = assess(first, starts, point)
            second_admitted, b = assess(second, starts, point)
            proposals[tier][role] = raw, a | b, raw | a | b
            summary[tier]["scope"].update(
                {
                    f"{role}_files": 1,
                    "all_labels": len(labels) if role == "edited" else 0,
                    "individually_geometrically_reachable": reachable if role == "edited" else 0,
                    "one_to_one_geometric_ceiling": ceiling if role == "edited" else 0,
                    "mixture_pairs_admitted": len(raw_admitted),
                    "any_separated_pair_admitted": len(first_admitted | second_admitted),
                }
            )
    for tier, by_role in proposals.items():
        baseline = matches(sorted(by_role["edited"][0]), labels)
        for index, arm in enumerate(ARMS):
            edited, ordinary, background = (by_role[role][index] for role in ROLES)
            hit = matches(sorted(edited), labels)
            added = hit - baseline
            controls = sorted(ordinary | background)
            summary[tier][arm].update(
                {
                    "edited_alerts": bool(edited),
                    "ordinary_alerts": bool(ordinary),
                    "background_alerts": bool(background),
                    "matched_edit_locations": len(hit),
                    "new_locations_same_schedule": len(added),
                    "lost_locations_same_schedule": len(baseline - hit),
                    "new_locations_without_control_collision": sum(
                        not matches(controls, [labels[i]]) for i in added
                    ),
                    "edited_files_with_alert_but_no_matching_label": bool(edited) and not hit,
                }
            )
    return summary, forwards, elapsed


def review(receipt, parent):
    require(
        receipt["schema"] == SCHEMA
        and receipt["sample_rate_hz"] == RATE
        and receipt["window_samples"] == WINDOW
        and receipt["step_samples"] == STEP
        and receipt["role_order"] == list(ROLES),
        "Unexpected review format",
    )
    groups = receipt["matched_groups"]
    raw_points = receipt["raw_calibration_hex"]
    require(
        len(groups) == 11 and set(raw_points) == {"strict", "wider"}, "Unexpected corpus or calibration scope"
    )
    points = {tier: float.fromhex(value) for tier, value in raw_points.items()}
    require(
        all(map(math.isfinite, points.values())) and points["strict"] >= points["wider"], "Invalid raw points"
    )
    require(
        parent["schema"] == "chipmunk.audio-recurrent-spectrum-opus/1"
        and all(point == parent["calibration"]["points"]["raw"][tier] for tier, point in points.items()),
        "Raw points differ from the earlier published calibration",
    )
    parent_groups = defaultdict(dict)
    for row in parent["confirmation"]:
        parent_groups[row["group"]][row["kind"]] = row
    require(
        len(parent_groups) == len(groups)
        and all(set(triple) == set(ROLES) for triple in parent_groups.values()),
        "Incomplete earlier matched recordings",
    )
    result = {tier: defaultdict(Counter) for tier in points}
    possible_forwards = forwards = elapsed = 0
    for slot, (group, earlier) in enumerate(zip(groups, parent_groups.values(), strict=True)):
        starts = (4 * RATE, 6 * RATE) if slot < 5 else (4 * RATE,)
        summary, actual, seconds = group_summary(group, starts, points, earlier)
        possible_forwards += len(starts) * len(ROLES) * 4
        forwards += actual
        elapsed += seconds
        for tier, arms in summary.items():
            for arm, counts in arms.items():
                result[tier][arm].update(counts)
    require(
        possible_forwards == receipt["forward_budget"] == LIMIT and forwards <= LIMIT,
        "Invalid forward budget",
    )
    recounted = {
        "actual_new_forwards": forwards,
        "forward_budget": possible_forwards,
        "files": len(groups) * len(ROLES),
        "elapsed_seconds": round(elapsed, 6),
        "summary": result,
    }
    require(recounted == receipt["expected"], "Saved results do not match the readings")
    return recounted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument(
        "parent", type=Path, nargs="?", help="Earlier recurrent-spectrum receipt, when stored separately"
    )
    args = parser.parse_args()
    try:
        data = json.loads(gzip.decompress(args.receipt.read_bytes()))
        parent_path = args.parent or args.receipt.with_name(
            "audio-recurrent-spectrum-opus-2026-11-18.json.gz"
        )
        previous = parent_path.read_bytes()
        require(
            hashlib.sha256(previous).hexdigest() == data["commitments_sha256"]["published_parent"],
            "Earlier published input receipt has changed",
        )
        result = review(data, json.loads(gzip.decompress(previous)))
    except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
