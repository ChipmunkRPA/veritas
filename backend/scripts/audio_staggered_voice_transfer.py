# SPDX-License-Identifier: Apache-2.0
"""Separate speech admission from corroboration at staggered source positions."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_continuous_activity_transfer as baseline

PARENT = baseline.PARENT
RATE = 16000
WIDTH = 3 * RATE
SHORTER = 5 * RATE // 2
STEP = 3 * RATE // 2
PHASE = STEP // 2
SCHEMA = "chipmunk.audio-staggered-voice-transfer/1"
ARMS = ("current", "staggered", "shorter")
DESIGN = {
    "staggered": "Add three-second windows offset by 750 ms. Require two AI-leaning three-second windows starting 1.5 s apart on either phase.",
    "shorter": "Use 2.5-second windows on the same 750 ms phase. Require an existing AI-leaning three-second window whose midpoint "
               "is within 750 ms of the shorter window's midpoint. Report only the overlap.",
    "comparison": "Use the same reset speech-activity and the same incumbent calibration tested in both window widths. "
                  "A staggered or shorter finding must add a useful location on top of current evidence, with no removals.",
    "rule": "At least three added loud-noise useful locations across two generator families and at most one additional genuine "
            "source group, zero additional isolated backgrounds and zero new gain-only genuine flags. This panel can support "
            "a separate confirmation, since its foregrounds and backgrounds were used for earlier comparisons.",
}


def schedule(samples):
    return sorted((start, start + width) for width in (WIDTH, SHORTER) for start in range(PHASE, samples - width + 1, STEP))


def proposals(previous, extra, direction):
    if direction == "staggered":
        slots = {a: state for (a, b), state in extra.items() if b - a == WIDTH}
        return [[a + STEP, a + WIDTH] for a, state in slots.items() if state == slots.get(a + STEP) == "ai"]
    return sorted({(max(a, c), min(b, d)) for (a, b), state in extra.items() if b - a == SHORTER and state == "ai"
                   for (c, d), support in previous.items() if support == "ai" and abs(a + b - c - d) <= 2 * PHASE})


def review(parent, doc):
    lock = {"schema": SCHEMA, "method": DESIGN, "parent_sha256": baseline.io.sha(PARENT)}
    if doc["freeze"] != lock:
        raise ValueError("Frozen comparison differs")
    totals, groups, families = defaultdict(Counter), defaultdict(set), defaultdict(set)
    for source, prior, candidate in zip(parent["roster"], parent["responses"], doc["recordings"], strict=True):
        width = source["samples"]
        if source["case"] != candidate["case"] or [tuple(w["span"]) for w in candidate["windows"]] != schedule(width):
            raise ValueError("Recorded windows differ from the source schedule")
        info = source["source"]
        role, truth = (info["label"], info["edits"]) if info else ("isolated", [])
        cell = totals[f"{source['scene']}/{role}"]
        cell["files"] += 1
        old = {tuple(row["span"]): row["direction"] for row in prior["windows"]}
        extra = {}
        for row in candidate["windows"]:
            a, b = row["span"]
            spoken, reading = row["speech_samples"], row["direction"]
            if (type(spoken) is not int or not 0 <= spoken <= b - a or reading not in baseline.binding.DIRECTIONS
                    or (reading == "unassessed") != (2 * spoken < b - a)):
                raise ValueError("Window disposition differs from detected speech")
            extra[a, b] = reading
            key = "three" if b - a == WIDTH else "shorter"
            cell.update({f"{key}_planned": 1, f"{key}_scored": reading != "unassessed", f"{key}_ai": reading == "ai"})
        incumbent = baseline.regions(width, old)
        cell["current_flags"] += bool(incumbent)
        found = baseline.clock.useful(incumbent, truth)
        cell["current_useful"] += found
        for arm in ARMS[1:]:
            new = proposals(old, extra, arm)
            gain = not found and baseline.clock.useful(new, truth)
            planned = proposals(dict.fromkeys(old, "ai"), dict.fromkeys(extra, "ai"), arm)
            admitted = proposals({span: "ai" if state != "unassessed" else state for span, state in old.items()},
                                 {span: "ai" if state != "unassessed" else state for span, state in extra.items()}, arm)
            cell.update({f"{arm}_useful": found or gain, f"{arm}_added": gain, f"{arm}_flags": bool(incumbent or new),
                         f"{arm}_new_flags": bool(new) and not incumbent,
                         f"{arm}_geometry_reachable": baseline.clock.useful(planned, truth),
                         f"{arm}_admission_reachable": baseline.clock.useful(admitted, truth),
                         f"{arm}_misplaced": bool(new) and bool(truth) and not baseline.clock.useful(new, truth)})
            if gain and source["scene"] != "gain_control":
                groups[arm, "edits"].add(info["group"])
                families[arm].add(info["family"])
            if not truth and new and not incumbent:
                groups[arm, role].add(info["group"] if info else source["case"])
    decisions = {arm: {"noisy_edit_groups": len(groups[arm, "edits"]), "noisy_edit_families": sorted(families[arm]),
                       "new_genuine_groups": len(groups[arm, "genuine"]), "new_isolated": len(groups[arm, "isolated"]),
                       "passes": len(groups[arm, "edits"]) >= 3 and len(families[arm]) >= 2
                       and len(groups[arm, "genuine"]) <= 1 and not groups[arm, "isolated"]
                       and not totals["gain_control/genuine"][f"{arm}_new_flags"]} for arm in ARMS[1:]}
    return {"cells": {key: dict(cell) for key, cell in totals.items()}, "decision": decisions}


def run(native, output, parent):
    from app.detectors.audio import calibration
    from scripts.audio_gtcrn_traffic import float_pcm
    from scripts.audio_reduced_background_localization import assets_for_scores
    from scripts.audio_strong_traffic_view import network_free

    baseline.native_folder(native, parent)
    if parent["freeze"]["method"]["calibration_sha256"] != calibration.SHA256:
        raise ValueError("Existing voice calibration differs")
    output.mkdir(mode=0o700)
    lock = {"schema": SCHEMA, "method": DESIGN, "parent_sha256": baseline.io.sha(PARENT)}
    baseline.io.save(output / "freeze.json", lock)
    network_free()
    voice, activity = assets_for_scores(parent["freeze"])
    recordings = []
    for row in parent["roster"]:
        pcm = float_pcm(baseline.focused.native_wave(native, row))
        windows = []
        for start, stop in schedule(len(pcm)):
            clip = pcm[start:stop]
            speech = baseline.samples_of_speech(baseline.sample_spans(activity, clip), 0, len(clip))
            direction = baseline.predict(voice, clip) if 2 * speech >= len(clip) else "unassessed"
            windows.append({"span": [start, stop], "direction": direction, "speech_samples": speech})
        readings = {"case": row["case"], "windows": windows}
        recordings.append(readings)
        with (output / "progress.jsonl").open("a") as progress:
            progress.write(json.dumps(readings, sort_keys=True) + "\n")
        if len(recordings) % 7 == 0:
            print(f"Measured {len(recordings)}/{len(parent['roster'])} deliveries", flush=True)
    result = {"freeze": lock, "recordings": recordings}
    result["results"] = review(parent, result)
    baseline.io.save(output / "receipt.json", result)
    print(json.dumps(result["results"]["decision"], sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--native", type=Path)
    args = parser.parse_args()
    parent = baseline.io.read(PARENT)
    if args.action == "run":
        if args.native is None:
            parser.error("Specify the exact parent PCM bank")
        run(args.native, args.path, parent)
    else:
        doc = baseline.io.read(args.path)
        result = review(parent, doc)
        if result != doc["results"]:
            raise ValueError("Recorded results differ")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
