# SPDX-License-Identifier: Apache-2.0
"""Test two fixed spectral views for local speech edits under strong noise."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_focused_window_transfer as focused
from scripts import audio_vocal_event_mixtures as io
from scripts import review_audio_reduced_background_localization as clock

SCHEMA = "chipmunk.audio-subband-partial-transfer/1"
PANELS = ("audio-noise-window-transfer-2026-11-12.json", "audio-focused-window-transfer-2026-11-12.json")
BANDS = {"low_stop": [80, 450], "high_stop": [2200, 4000]}
METHOD = {
    "bandstop_hz": BANDS, "filter_order": 4,
    "baseline": "The unchanged focused-window rule and its original measured directions.",
    "routing": "Only inconclusive or unassessed three-second windows on the fixed dense grid. Admit filtered speech with the usual activity rule.",
    "point": "Strictly above the largest admitted ordinary or background-only selection logit, separately for each band.",
    "agreement": "Require two adjacent AI responses in a single view. Native AI can support either filtered view. Preserve all baseline ranges.",
    "gate": {"minimum_new_noisy_groups": 4, "minimum_families": 3, "minimum_scenes": 2,
             "maximum_new_genuine_groups": 2, "maximum_new_clean_or_isolated": 0, "maximum_added_genuine_time": .03},
}
require = focused.require


def inputs():
    older, newer = [io.read(io.PUBLIC / name) for name in PANELS]
    held_groups = {row["source"]["group"] for row in newer["roster"] if row["source"]}
    old_reads = {row["id"]: row["windows"] for row in older["selection"]["rows"]}
    selection = []
    for row in older["roster"]:
        source = row["source"]
        if row["role"] == "selection" and (source is None or (row["kind"] == "genuine" and source["group"] not in held_groups)):
            selection.append((row, old_reads[row["id"]]))
    held = list(zip(newer["roster"], (row["windows"] for row in newer["responses"]), strict=True))
    require(len(selection) == 25 and len(held) == 98, "Expected the two frozen roles")
    return {"selection": selection, "held": held}, newer["freeze"]["voice_assets"]


def dense(samples):
    return clock.schedules(samples)[1]


def eligible(row, windows):
    directions = {tuple(item["span"]): item["direction"] for item in windows}
    return [span for span in dense(row["samples"]) if directions[tuple(span)] in {"inconclusive", "unassessed"}]


def frozen():
    return {"schema": SCHEMA, "method": METHOD, "runner": io.sha(Path(__file__)),
            "prior": {name: io.sha(io.PUBLIC / name) for name in PANELS}}


def native(folder, row):
    identifier = row.get("case", row.get("id"))
    raw = (folder / "pcm" / f"{identifier:03}.pcm").read_bytes()
    require(len(raw) == 2 * row["samples"] and io.digest(raw) == row["pcm_sha256"], "Delivery PCM changed")
    return raw


def view(wave, band):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    return sosfiltfilt(butter(METHOD["filter_order"], BANDS[band], btype="bandstop", fs=clock.RATE, output="sos"),
                       wave.astype(np.float64)).astype("<f4")


def measure(rows, folder, voice, activity, phase):
    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from scripts.audio_gtcrn_traffic import float_pcm

    responses = []
    for row, original in rows:
        wave = float_pcm(native(folder, row))
        windows = []
        for band in BANDS:
            filtered = view(wave, band)
            scores = []
            for start, end in eligible(row, original):
                clip = filtered[start:end]
                score = logit(voice.predict([clip])[0]) if window_in_speech(activity.spans(clip), 0., 3.) else None
                scores.append({"span": [start, end], "logit": None if score is None else score.hex(),
                               "filtered_sha256": io.digest(clip.tobytes())})
            windows.append({"band": band, "windows": scores})
        responses.append({"id": row.get("case", row.get("id")), "bands": windows})
        if len(responses) % 10 == 0:
            print(phase, len(responses), "deliveries", flush=True)
    return responses


def readings(row, original, observed, source=None):
    require(observed["id"] == row.get("case", row.get("id")) and [arm["band"] for arm in observed["bands"]] == list(BANDS),
            "Unexpected delivery or band roster")
    samples = None
    if source is not None:
        from scripts.audio_gtcrn_traffic import float_pcm

        samples = float_pcm(native(source, row))
    planned = eligible(row, original)
    for arm in observed["bands"]:
        require([item["span"] for item in arm["windows"]] == planned, "Filtered window eligibility differs")
        filtered = view(samples, arm["band"]) if samples is not None else None
        for item in arm["windows"]:
            a, b = item["span"]
            require(clock.SHA.fullmatch(item["filtered_sha256"]) and (filtered is None or io.digest(filtered[a:b].tobytes()) == item["filtered_sha256"]),
                    "Filtered waveform commitment differs")
            value = float.fromhex(item["logit"]) if item["logit"] is not None else None
            require(value is None or math.isfinite(value), "Invalid filtered response")
            yield arm["band"], (a, b), value


def points(rows, responses, source=None):
    values = defaultdict(list)
    for (row, original), observed in zip(rows, responses, strict=True):
        for band, _, value in readings(row, original, observed, source):
            if value is not None:
                values[band].append(value)
    require(all(values[band] for band in BANDS), "No eligible genuine selection response")
    return {band: {"point": max(values[band]).hex(), "observations": len(values[band])} for band in BANDS}


def evaluate(rows, responses, cutoffs, source=None):
    cells = defaultdict(Counter)
    groups, families, scenes, genuine_groups, clean_or_isolated = set(), set(), set(), set(), set()
    for (row, original), observed in zip(rows, responses, strict=True):
        directions = {tuple(item["span"]): item["direction"] for item in original}
        native_positive = {span for span, direction in directions.items() if direction == "ai"}
        by_band = {band: native_positive.copy() for band in BANDS}
        for band, span, value in readings(row, original, observed, source):
            if value is not None and value > float.fromhex(cutoffs[band]["point"]):
                by_band[band].add(span)
        old = focused.ranges(row["samples"], directions)["focused"]
        spans = {band: clock.intersections(dense(row["samples"]), by_band[band]) for band in BANDS}
        arms = {"focused": old, **{band: old + proposal for band, proposal in spans.items()}, "combined": old + sum(spans.values(), [])}
        meta = row["source"]
        truth, label = (meta["edits"], meta["label"]) if meta else ([], "isolated")
        scene = row["scene"]
        cell = cells[f"{scene}/{label}"]
        cell["files"] += 1
        cell["genuine_samples"] += row["samples"] - sum(b - a for a, b in truth)
        prior = clock.useful(old, truth)
        for name, ranges in arms.items():
            union = clock.intervals_union(ranges)
            useful = clock.useful(ranges, truth)
            cell[f"{name}_useful"] += useful
            cell[f"{name}_flags"] += bool(ranges)
            cell[f"{name}_added"] += useful and not prior
            cell[f"{name}_genuine_samples"] += sum(b - a - sum(clock.overlap((a, b), edit) for edit in truth) for a, b in union)
        if scene != "gain_control" and clock.useful(arms["combined"], truth) and not prior:
            groups.add(meta["group"])
            families.add(meta["family"])
            scenes.add(scene)
        if not truth and arms["combined"] and not old:
            if label == "genuine" and scene != "gain_control":
                genuine_groups.add(meta["group"])
            else:
                clean_or_isolated.add((label, meta["group"] if meta else row["case"]))
    noisy = [v for key, v in cells.items() if not key.startswith("gain_control/") and not key.endswith("/isolated")]
    added_time = sum(v["combined_genuine_samples"] - v["focused_genuine_samples"] for v in noisy) / sum(v["genuine_samples"] for v in noisy)
    limit = METHOD["gate"]
    gate = {"groups": len(groups) >= limit["minimum_new_noisy_groups"], "families": len(families) >= limit["minimum_families"],
            "scenes": len(scenes) >= limit["minimum_scenes"], "genuine": len(genuine_groups) <= limit["maximum_new_genuine_groups"],
            "clean_or_isolated": len(clean_or_isolated) <= limit["maximum_new_clean_or_isolated"],
            "genuine_time": added_time <= limit["maximum_added_genuine_time"]}
    return {"cells": dict(cells), "groups": len(groups), "families": sorted(families), "scenes": sorted(scenes),
            "new_genuine_groups": len(genuine_groups), "new_clean_or_isolated": len(clean_or_isolated),
            "added_noisy_genuine_time_fraction": added_time, "gate": gate, "passed": all(gate.values())}


def run(output, selection_path, held_path):
    from scripts.audio_reduced_background_localization import assets_for_scores

    roles, assets = inputs()
    output.mkdir(mode=0o700)
    lock = frozen()
    io.save(output / "freeze.json", lock)
    print("Frozen", io.sha(output / "freeze.json"), flush=True)
    voice, activity = assets_for_scores({"voice_assets": assets})
    selection = measure(roles["selection"], selection_path, voice, activity, "selection")
    operating = points(roles["selection"], selection)
    io.save(output / "points.json", {"selection_sha256": io.digest(io.encoded(selection)), "points": operating})
    held = measure(roles["held"], held_path, voice, activity, "held")
    io.save(output / "receipt.json", {"freeze": lock, "selection": selection, "points": io.read(output / "points.json"),
                                      "held": held, "results": evaluate(roles["held"], held, operating)})


def review(path, selection_path=None, held_path=None):
    document = io.read(path)
    roles, _ = inputs()
    require(document["freeze"] == frozen() and document["points"]["selection_sha256"] == io.digest(io.encoded(document["selection"])),
            "Sub-band comparison inputs or selection commitments differ")
    operating = points(roles["selection"], document["selection"], selection_path)
    require(operating == document["points"]["points"], "Separate band operating points changed")
    result = evaluate(roles["held"], document["held"], operating, held_path)
    require(result == document["results"], "Reported band results differ")
    print(io.encoded(result).decode(), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--held", type=Path)
    args = parser.parse_args()
    if args.phase == "run":
        if not args.selection or not args.held:
            parser.error("run requires --selection and --held")
        run(args.path, args.selection, args.held)
    else:
        review(args.path, args.selection, args.held)
