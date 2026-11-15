# SPDX-License-Identifier: Apache-2.0
"""Test denser coverage for a fixed earlier-copy speech comparison."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_reference_spatial_transfer as spatial

old, room = spatial.old, spatial.room
RATE = spatial.RATE
WIDTH = room.stereo.RANGE_SAMPLES
HOP = 1280
STARTS = tuple(range(0, 6 * RATE + 1, 6400))
NORMAL = tuple(range(0, 6 * RATE + 1, 19200))
SEPARATIONS = {"dense_same_gap": 19200, "dense_short_gap": 12800}
SCHEMA = "chipmunk.audio-clean-speech-schedule-transfer/1"
METHOD = {
    "input": "Use the same 264 native source-bound ICSI and AMI comparisons as the stereo projection test. "
             "Read the original downmix and both measured channels. Leave their recorded reference admission "
             "and broad frequency-balance exclusions in force.",
    "schedule": "Repeat the unchanged six-second measurements at sixteen starts separated by 400 ms. "
                "Compare pairs exactly 1.2 s apart and pairs exactly 0.8 s apart as separate rules. "
                "Both measured 720 ms ranges must start within 360 ms. Prefer the strongest weaker response, "
                "then the earliest range. No new fit or change to the six-second operating point.",
    "combined": "Keep the existing channel-corroborated result. For a proposed additional range require "
                "corroboration from the original stereo channels using the existing channel selection, "
                "unchanged channel correlation and half-range overlap rules.",
    "oracle": "For diagnosis only, check whether both scheduled measuring windows can test the same 720 ms range "
              "that reaches the original useful overlap rule. Use known labels only in this geometry check and "
              "when measuring usefulness. Keep ordinary controls and wrong earlier references.",
}


def shapes(reference, submitted, reason):
    from app.evidence import audio_clean_speech as single

    if reason in {"reference_match_not_supported", "consistent_frequency_balance_change"}:
        return {}
    output = {}
    for start in STARTS:
        stop = start + single.SAMPLES
        _, signal = single.locate(reference[start:stop], submitted[start:stop])
        if signal:
            output[start] = signal | {"samples": [value + start for value in signal["samples"]]}
    return output


def paired(hits, gap):
    choices = []
    for start, left in hits.items():
        right = hits.get(start + gap)
        if right is None or abs(left["samples"][0] - right["samples"][0]) > WIDTH // 2:
            continue
        stronger = left if left["relative_drop"] >= right["relative_drop"] else right
        weakest = min(left["relative_drop"], right["relative_drop"])
        choices.append(((weakest, -stronger["samples"][0]), stronger["samples"], [start, start + gap]))
    best = max(choices, key=lambda entry: entry[0], default=None)
    return {"range": best[1], "windows": best[2]} if best else {"range": None, "windows": None}


def geometry(truth, starts, separation):
    if not truth:
        return False
    for first in starts:
        second = first + separation
        if second not in starts:
            continue
        for left in range(second + 9 * HOP, first + 57 * HOP, HOP):
            if old.location([left, left + WIDTH], truth):
                return True
    return False


def capture(args):
    containers = defaultdict(list)
    for name, row, earlier, views in spatial.native_cases(args, spatial.input_panels()):
        series = {side: shapes(earlier, clip, row["measurement"]["readings"][side]["reason"]) for side, clip in views.items()}
        readings = {arm: {side: paired(hits, spacing) for side, hits in series.items()}
                    for arm, spacing in SEPARATIONS.items()}
        containers[name].append({"group": row["group"], "kind": row["kind"], "views": readings})
        if len(containers[name]) % 11 == 0:
            print(f"{name}: measured {len(containers[name])} stereo pairs.", flush=True)
    data = {"schema": SCHEMA, "method": METHOD,
            "prior": {name: old.sha(path.read_bytes()) for name, path in spatial.PANELS.items()}, "samples": containers}
    return data | {"results": evaluate(data)}


def arm_result(row, original, name):
    views = row["views"][name]
    spacing = SEPARATIONS[name]
    old.require(set(views) == {"downmix", "left", "right"}, "The native channels differ")
    for data in views.values():
        interval, witnesses = data["range"], data["windows"]
        old.require((interval is None and witnesses is None)
                    or (interval is not None and witnesses is not None and len(interval) == len(witnesses) == 2
                        and all(type(n) is int for n in (*interval, *witnesses))
                        and interval[1] - interval[0] == WIDTH and 0 <= interval[0] < interval[1] <= spatial.SAMPLES
                        and witnesses[0] in STARTS and witnesses[1] in STARTS and witnesses[1] - witnesses[0] == spacing
                        and any(start + 9 * HOP <= interval[0] <= start + 56 * HOP for start in witnesses)),
                    "A selected range lacks two distinct source-clock witnesses")
    support = original["measurement"]["support"]
    choices = room.channel_check.select(views, support)
    return views["downmix"]["range"] if room.channel_check.supporting_channels(views, support, choices) else None


def evaluate(data):
    old.require(data["schema"] == SCHEMA and data["method"] == METHOD
                and data["prior"] == {name: old.sha(path.read_bytes()) for name, path in spatial.PANELS.items()}
                and set(data["samples"]) == set(spatial.PANELS), "The frozen schedule comparison differs")
    output = {}
    for panel, reference in spatial.input_panels().items():
        expected = {(r["group"], r["kind"]): r for r in reference["rows"] if r["bank"] == "icsi"}
        rows = data["samples"][panel]
        old.require(len(rows) == len(expected) and {(r["group"], r["kind"]) for r in rows} == set(expected),
                    "The source pairs differ")
        counts = defaultdict(Counter)
        for row in rows:
            original = expected[row["group"], row["kind"]]
            truth = original["truth"]
            conventional = original["measurement"]["readings"]["downmix"]["range"] if room.downmix_with_channel_support(
                original["measurement"]) else None
            checked = {"current": conventional}
            can_measure = {"current": geometry(truth, NORMAL, 19200)}
            for name, spacing in SEPARATIONS.items():
                additional = arm_result(row, original, name)
                checked[name] = additional
                checked[f"{name}_fallback"] = conventional or additional
                can_measure[name] = geometry(truth, STARTS, spacing)
            category = room.category(row["kind"])
            for name, selected in checked.items():
                useful = old.location(selected, truth)
                counts[f"{name}/{category}"].update(files=1, flags=bool(selected), useful=useful,
                                                     added_useful=useful and not old.location(conventional, truth),
                                                     wrong_on_edit=bool(truth and selected and not useful))
            for name, possible in can_measure.items():
                counts[f"geometry/{name}/{category}"].update(possible=possible)
        output[panel] = counts
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("capture")
    for name in ("icsi", "prior_rooms", "held_rooms", "output"):
        run.add_argument(name, type=Path)
    replay = modes.add_parser("review")
    replay.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "capture":
        record = capture(args)
        body = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        with args.output.open("xb") as handle:
            handle.write(gzip.compress(body, mtime=0))
    else:
        previous = spatial.load(args.receipt)
        result = evaluate(previous)
        old.require(result == previous["results"], "The published schedule decisions differ")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
