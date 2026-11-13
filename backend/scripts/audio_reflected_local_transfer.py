# SPDX-License-Identifier: Apache-2.0
"""Test focused speech inputs with controlled reflection to a fixed detector length."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_subband_partial_transfer as prior
from scripts import audio_transient_reduction_transfer as original

sources, io, clock, focused = original.sources, original.io, original.clock, original.focused
require = original.require
SCHEMA = "chipmunk.audio-reflected-local-transfer/1"
WIDTH = 19_200
HOP = 9_600
LENGTH = 48_000
METHOD = {
    "focus_samples": WIDTH,
    "hop_samples": HOP,
    "detector_input_samples": LENGTH,
    "scope": "Exact original focus within a dense inconclusive or unassessed window. Require half the original focus to contain speech.",
    "extension": "Symmetric reflection of 14400 samples per side. No level adjustment or new source coverage.",
    "point": "Largest assessed score on the disjoint genuine selection deliveries, strictly exceeded.",
    "policy": "Add the strongest assessed focus; preserve original focused ranges. Comparisons use true edit overlap only after scores freeze.",
}


def frozen():
    return {
        "schema": SCHEMA,
        "method": METHOD,
        "runner_sha256": io.sha(Path(__file__)),
        "sources": prior.frozen(),
    }


def schedule(row, native):
    eligible = sources.eligible(row, native)
    return [
        (a, a + WIDTH)
        for a in range(0, row["samples"] - WIDTH + 1, HOP)
        if any(left <= a and a + WIDTH <= right for left, right in eligible)
    ]


def reflected(wave):
    import numpy as np

    require(len(wave) == WIDTH, "Exact original focus length required")
    padding = (LENGTH - WIDTH) // 2
    return np.pad(wave, (padding, padding), mode="reflect")


def wave_at(folder, row):
    from scripts.audio_gtcrn_traffic import float_pcm

    return float_pcm(sources.native(folder, row))


def measure(rows, folder, voice, activity, phase):
    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech

    result = []
    for row, native in rows:
        wave = wave_at(folder, row)
        spans = activity.spans(wave)
        windows = []
        for a, b in schedule(row, native):
            focus = wave[a:b]
            admitted = window_in_speech(spans, a / clock.RATE, b / clock.RATE)
            pad = reflected(focus) if admitted else None
            score = logit(voice.predict([pad])[0]) if admitted else None
            windows.append(
                {
                    "span": [a, b],
                    "native_sha256": io.digest(focus.tobytes()),
                    "extended_sha256": io.digest(pad.tobytes()) if admitted else None,
                    "logit": score.hex() if admitted else None,
                }
            )
        result.append({"id": row.get("case", row.get("id")), "windows": windows})
        if len(result) % 10 == 0:
            print(phase, len(result), flush=True)
    return result


def readings(row, native, record, folder=None):
    require(
        record["id"] == row.get("case", row.get("id"))
        and [tuple(item["span"]) for item in record["windows"]] == schedule(row, native),
        "Original focus schedule differs",
    )
    wave = wave_at(folder, row) if folder else None
    for item in record["windows"]:
        a, b = item["span"]
        pin = item["extended_sha256"]
        value = float.fromhex(item["logit"]) if item["logit"] is not None else None
        require(
            clock.SHA.fullmatch(item["native_sha256"])
            and (value is None) == (pin is None)
            and (value is None or (math.isfinite(value) and clock.SHA.fullmatch(pin))),
            "Focus reading or input invalid",
        )
        if wave is not None:
            focus = wave[a:b]
            require(
                io.digest(focus.tobytes()) == item["native_sha256"]
                and (pin is None or io.digest(reflected(focus).tobytes()) == pin),
                "Exact focus or extension differs",
            )
        yield value, (a, b)


def point(rows, records, folder=None):
    values = [
        value
        for (row, native), item in zip(rows, records, strict=True)
        for value, _ in readings(row, native, item, folder)
        if value is not None
    ]
    require(values, "No admitted genuine focus")
    return {"cutoff": max(values).hex(), "admitted": len(values), "sha256": io.commitment(records)}


def outcomes(rows, records, cutoff, folder=None):
    cells, false = defaultdict(Counter), defaultdict(set)
    gained = set()
    for (row, native), record in zip(rows, records, strict=True):
        meta, scene = row["source"], row["scene"]
        label, truth = (meta["label"], meta["edits"]) if meta else ("isolated", [])
        group = meta["group"] if meta else row["case"]
        baseline = [
            tuple(span)
            for span in focused.ranges(
                row["samples"], {tuple(item["span"]): item["direction"] for item in native}
            )["focused"]
        ]
        observed = list(readings(row, native, record, folder))
        usable = [(value, span) for value, span in observed if value is not None]
        chosen = max(usable, key=lambda item: item[0], default=None)
        added = [chosen[1]] if chosen and chosen[0] > cutoff else []
        cell = cells[f"{scene}/{label}"]
        cell["files"] += 1
        cell["focus_inputs"] += len(observed)
        cell["assessed_focus_inputs"] += len(usable)
        cell["files_with_assessed_focus"] += bool(usable)
        cell["useful_best_before_point"] += bool(chosen and clock.useful([chosen[1]], truth))
        cell["focus_flagged"] += bool(added)
        cell["focus_mislocated"] += bool(truth and added and not clock.useful(added, truth))
        if added and not truth and not baseline:
            false[f"{scene}/{label}"].add(group)
        if clock.useful(added, truth) and not clock.useful(baseline, truth):
            gained.add((scene, meta["family"], group))
        for arm, ranges in (("focused", baseline), ("combined", baseline + added)):
            cell[f"{arm}_useful"] += clock.useful(ranges, truth)
            cell[f"{arm}_flagged"] += bool(ranges)
            cell[f"{arm}_incorrect_samples"] += sum(
                b - a - sum(clock.overlap((a, b), e) for e in truth) for a, b in clock.intervals_union(ranges)
            )
    return {
        "cells": dict(cells),
        "gained": [list(group) for group in sorted(gained)],
        "false_groups": {k: len(v) for k, v in false.items()},
    }


def run(folder, selection, held):
    import torch

    from scripts.audio_reduced_background_localization import assets_for_scores

    folder.mkdir(mode=0o700)
    lock = frozen()
    io.save(folder / "freeze.json", lock)
    rows, assets = sources.inputs()
    torch.set_num_threads(2)
    voice, activity = assets_for_scores({"voice_assets": assets})
    selected = measure(rows["selection"], selection, voice, activity, "selection")
    cutoff = point(rows["selection"], selected)
    io.save(folder / "point.json", cutoff)
    print("Frozen point", cutoff, flush=True)
    checked = measure(rows["held"], held, voice, activity, "held")
    io.save(folder / "readings.json", {"selection": selected, "held": checked})
    result = outcomes(rows["held"], checked, float.fromhex(cutoff["cutoff"]))
    io.save(
        folder / "receipt.json",
        {"freeze": lock, "point": cutoff, "selection": selected, "held": checked, "result": result},
    )


def review(path, selection, held):
    doc = io.read(path)
    require(doc["freeze"] == frozen(), "Frozen method changed")
    rows, _ = sources.inputs()
    cutoff = point(rows["selection"], doc["selection"], selection)
    require(cutoff == doc["point"], "Independent selection point differs")
    result = outcomes(rows["held"], doc["held"], float.fromhex(cutoff["cutoff"]), held)
    require(result == doc["result"], "Reported outcomes changed")
    print(io.encoded(result).decode(), end="")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("run", "review"))
    cli.add_argument("path", type=Path)
    cli.add_argument("--selection", type=Path)
    cli.add_argument("--held", type=Path)
    args = cli.parse_args()
    if args.phase == "run" and not all((args.selection, args.held)):
        cli.error("run requires --selection and --held")
    (run if args.phase == "run" else review)(args.path, args.selection, args.held)
