# SPDX-License-Identifier: Apache-2.0
"""Compare closer stereo corroboration on matched earlier-copy speech tests."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from app.evidence.audio_clean_speech_stereo import common_half_range
from scripts import audio_clean_speech_schedule_fresh_rooms as fresh
from scripts import audio_clean_speech_schedule_transfer as schedule

old, room = schedule.old, schedule.room
FRESH = old.PANELS / "audio-clean-speech-schedule-fresh-rooms-2026-11-15.json.gz"
PREVIOUS = old.PANELS / "audio-clean-speech-schedule-transfer-2026-11-15.json.gz"
SCHEMA = "chipmunk.audio-clean-speech-close-confirmation/1"
STEPS = {"current": (0, 19200), "close_pair": (0, 12800), "close_three": (0, 6400, 12800)}
ARMS = ("current", "close_pair", "close_three", "close_both_channels")
METHOD = {
    "support": "The serving check uses the original comparison then the 1.2-second denser fallback. Keep that result "
               "when present. Try two new 0.8-second paths plus the earlier two-window sensitivity comparison.",
    "third_window": "Require local measurements in three consecutive 400 ms starts on each supporting view. "
                    "Require at least half a range shared among all three readings. Rank by the weakest reading, "
                    "then use the strongest reading's source interval and prefer earlier intervals on ties.",
    "both_channels": "Require a common half-range among the downmix and both physical channel readings for the "
                     "0.8-second pair. Keep the earlier correlation and per-view qualification.",
    "source": "Reuse the 264 native comparisons from the two retained panels and the matched 264 transfers on three "
              "other meetings. Require the same submitted and reference source hashes, decoded clocks and native tracks. "
              "Measure new gains beside ordinary controls, wrong references and wrong edit positions.",
}


def three_windows(values):
    hits = {row["start"]: row["result"] for row in values if row["result"] is not None}
    choices = []
    for start in hits:
        starts = [start + delta for delta in STEPS["close_three"]]
        observations = [hits.get(offset) for offset in starts]
        if all(observations) and common_half_range([item["samples"] for item in observations]):
            peak = max(observations, key=lambda item: item["relative_drop"])
            score = min(item["relative_drop"] for item in observations)
            choices.append((score, peak, starts))
    best = max(choices, key=lambda item: (item[0], -item[1]["samples"][0]), default=None)
    return None if best is None else best[1] | {"supporting_window_starts": best[2]}


def measure(earlier, native, comparison):
    from app.evidence.audio_clean_speech_long import corroborate_windows, denser_windows

    result = {name: {} for name in STEPS}
    for channel, pcm in native.items():
        values = denser_windows(earlier, pcm, comparison["readings"][channel]["reason"])
        variants = (corroborate_windows(values), corroborate_windows(values, 12800), three_windows(values))
        for name, found in zip(STEPS, variants, strict=True):
            result[name][channel] = ({"range": found["samples"], "windows": found["supporting_window_starts"]}
                                     if found else {"range": None, "windows": None})
    return result


def study_inputs():
    archive = schedule.spatial.input_panels()
    transfer = schedule.spatial.load(FRESH)
    old.require(fresh.review(transfer) == transfer["results"], "The background transfer did not replay")
    previous = schedule.spatial.load(PREVIOUS)
    old.require(schedule.evaluate(previous) == previous["results"], "The earlier schedules did not replay")
    return archive, transfer, previous


def combined_cases(transfer, args):
    import numpy as np

    dry = room.dry_icsi_inputs(args.icsi)
    known = {(item["group"], item["kind"]): item for item in transfer["samples"]}
    sources = transfer["plan"]["rows"]

    def load_source(row):
        key, meeting = row["group"], row["meeting"]
        raw = [fresh.fetch_range(meeting, channel, 44 + 2 * row["start"], 2 * fresh.SAMPLES)
               for channel in fresh.MICS]
        old.require(dict(zip(fresh.MICS, map(old.sha, raw), strict=True)) == transfer["rooms_pcm_sha256"][str(key)],
                    "A redownloaded microphone excerpt changed")
        return {meeting: [np.frombuffer(pcm, dtype="<i2") for pcm in raw]}

    with ThreadPoolExecutor(max_workers=5) as workers, TemporaryDirectory(prefix="chipmunk-close-confirm-") as work:
        carrier = Path(work) / "comparison.m4a"
        for row, tracks in zip(sources, workers.map(load_source, sources), strict=True):
            decodes = {}
            for kind, (earlier, submitted) in room.materialize(row | {"start": 0}, tracks, dry).items():
                original = known[row["group"], kind]
                pin = old.sha(submitted)
                old.require(pin == original["submitted"] and old.sha(earlier) == original["reference"],
                            "A transfer comparison changed its source bytes")
                if pin not in decodes:
                    decodes[pin] = room.stereo.stereo_pcm(carrier, submitted, original["clock"]["encoded_sha256"])
                native, clock = decodes[pin]
                old.require(clock == original["clock"] and all(old.sha(pcm.tobytes()) ==
                            original["measurement"]["readings"][side]["pcm_sha256"] for side, pcm in native.items()),
                            "The original channel clock or digest changed")
                yield "fresh", original, np.frombuffer(earlier, dtype="<i2"), native


def selected(record, support, family):
    view = record[family]
    step = STEPS[family]
    old.require(set(view) == {"downmix", "left", "right"}, "Source views changed")
    for item in view.values():
        span, starts = item["range"], item["windows"]
        old.require((span is None and starts is None) or (span and starts and len(span) == 2 and len(starts) == len(step)
                    and all(type(n) is int for n in (*span, *starts)) and span[1] - span[0] == schedule.WIDTH
                    and 0 <= span[0] < span[1] <= room.SAMPLES and set(starts) <= set(schedule.STARTS)
                    and [start - starts[0] for start in starts] == list(step)
                    and any(start + 9 * schedule.HOP <= span[0] <= start + 56 * schedule.HOP for start in starts)),
                    "A measured range must have the recorded window geometry")
    channels = room.channel_check.supporting_channels(view, support, room.channel_check.select(view, support))
    return view["downmix"]["range"] if channels else None


def pins():
    return {p.name: old.sha(p.read_bytes()) for p in (FRESH, PREVIOUS, *schedule.spatial.PANELS.values())}


def review(data, inputs=None):
    archive, transfer, previous = study_inputs() if inputs is None else inputs
    old.require(data["schema"] == SCHEMA and data["method"] == METHOD and data["parents"] == pins()
                and bool(data["panels"]) and set(data["panels"]) <= {"first", "held", "fresh"},
                "The earlier-copy comparison inputs differ")
    parents = {panel: parent["rows"] for panel, parent in archive.items()} | {"fresh": transfer["samples"]}
    prior_dense = previous["samples"] | {"fresh": transfer["samples"]}
    truth = {item["group"]: item["truth"] for item in transfer["plan"]["rows"]}
    totals = {}
    for panel, data_rows in data["panels"].items():
        retained = {(row["group"], row["kind"]): row for row in parents[panel]
                    if panel == "fresh" or row["bank"] == "icsi"}
        known = {(row["group"], row["kind"]): row["views"] for row in prior_dense[panel]}
        old.require(len(retained) == len(data_rows) and set(retained) == {(x["group"], x["kind"]) for x in data_rows},
                    "A reference pair is missing or duplicated")
        per_rule = defaultdict(Counter)
        for row in data_rows:
            key = row["group"], row["kind"]
            record = row["views"]
            old.require(record["current"] == known[key]["dense_same_gap"] and record["close_pair"] == known[key]["dense_short_gap"],
                        "A previously frozen comparison produced different ranges")
            original = retained[key]["measurement"]
            support = original["support"]
            same_gap = selected(record, support, "current")
            prior = original["readings"]["downmix"]["range"] if room.downmix_with_channel_support(original) else same_gap
            short = selected(record, support, "close_pair")
            three = selected(record, support, "close_three")
            spans = [item["range"] for item in record["close_pair"].values()]
            both = short if short and all(spans) and common_half_range(spans) else None
            truth_span = truth[key[0]][key[1]] if panel == "fresh" else retained[key]["truth"]
            for name, selected_span in zip(ARMS, (prior, prior or short, prior or three, prior or both), strict=True):
                useful = old.location(selected_span, truth_span)
                per_rule[f"{name}/{room.category(key[1])}"].update(
                    files=1, flags=bool(selected_span), useful=useful,
                    gained=useful and not old.location(prior, truth_span),
                    extra_flag=bool(selected_span and not prior), wrong_on_edit=bool(truth_span and selected_span and not useful),
                )
        totals[panel] = {key: dict(value) for key, value in per_rule.items()}
    return totals


def capture(args):
    archive, transfer, _ = inputs = study_inputs()
    panels = defaultdict(list)
    cases = (combined_cases(transfer, args) if args.panel == "fresh"
             else schedule.spatial.native_cases(args, {name: data for name, data in archive.items()
                                                       if args.panel in {"retained", name}}))
    for panel, row, reference, native in cases:
        panels[panel].append({"group": row["group"], "kind": row["kind"], "views": measure(reference, native, row["measurement"])})
        if len(panels[panel]) % 11 == 0:
            print(f"{panel}: {len(panels[panel])} pairs scored", flush=True)
    packet = {"schema": SCHEMA, "method": METHOD, "parents": pins(), "panels": panels}
    packet["results"] = review(packet, inputs)
    save(args.output, packet)


def save(path, data):
    with path.open("xb") as output:
        output.write(gzip.compress(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


def merge(retained, transfer, output):
    records = [schedule.spatial.load(path) for path in (retained, transfer)]
    inputs = study_inputs()
    for record, names in zip(records, ({"first", "held"}, {"fresh"}), strict=True):
        old.require(set(record["panels"]) == names and review(record, inputs) == record["results"],
                    "An incomplete or changed comparison cannot be merged")
    combined = records[0] | {"panels": records[0]["panels"] | records[1]["panels"]}
    combined["results"] = review(combined, inputs)
    save(output, combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    native = modes.add_parser("capture")
    for name in ("icsi", "prior_rooms", "held_rooms", "output"):
        native.add_argument(name, type=Path)
    native.add_argument("--panel", choices=("retained", "first", "held", "fresh"), required=True)
    assemble = modes.add_parser("merge")
    for name in ("retained", "transfer", "output"):
        assemble.add_argument(name, type=Path)
    portable = modes.add_parser("review")
    portable.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "capture":
        capture(args)
    elif args.mode == "merge":
        merge(args.retained, args.transfer, args.output)
    else:
        data = schedule.spatial.load(args.receipt)
        result = review(data)
        old.require(result == data["results"], "Final counts do not match the numeric replay")
        print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
