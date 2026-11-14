# SPDX-License-Identifier: Apache-2.0
"""Separate removing low rumble for speech admission from scoring filtered speech."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_continuous_activity_transfer as parent

RATE = parent.binding.RATE
SCHEMA = "chipmunk.low-frequency-speech-admission/1"
ARMS = ("original_samples", "filtered_samples")
METHOD = {"cutoff_hz": 240, "order": 3,
          "scope": "For original three-second windows below the speech minimum, rerun speech activity after fixed zero-phase "
                   "high-pass filtering of the recording. Score each newly admitted window on its original and filtered samples.",
          "decision": "Retain all current results. Require at least three added useful noisy groups across at least two generators, "
                      "no gain-only or isolated flags, and at most one additional genuine source group. "
                      "Filtered scores reuse the existing AI-leaning decision region for screening this experiment only."}


def contract():
    return {"schema": SCHEMA, "method": METHOD, "parent_sha256": parent.io.sha(parent.PARENT)}


def review(source, receipt):
    if receipt["freeze"] != contract():
        raise ValueError("Comparison design differs")
    cells, gains, families, alarms = defaultdict(Counter), defaultdict(set), defaultdict(set), defaultdict(set)
    for metadata, previous, row in zip(source["roster"], source["responses"], receipt["rows"], strict=True):
        known = {tuple(item["span"]): item["direction"] for item in previous["windows"]}
        expected = [span for span, direction in known.items() if direction == "unassessed"]
        if row["case"] != metadata["case"] or [tuple(item["span"]) for item in row["windows"]] != expected:
            raise ValueError("A previously skipped window is missing or moved")
        info = metadata["source"]
        scene = metadata["scene"]
        role, truth = (info["label"], info["edits"]) if info else ("isolated", [])
        group = info["group"] if info else metadata["case"]
        cell = cells[f"{scene}/{role}"]
        cell.update(files=1, skipped_windows=len(expected))
        base = parent.regions(metadata["samples"], known)
        existing = parent.clock.useful(base, truth)
        cell.update(current_flags=bool(base), current_useful=existing)
        for item, span in zip(row["windows"], expected, strict=True):
            speech = item["speech_samples"]
            if type(speech) is not int or not 0 <= speech <= 3 * RATE:
                raise ValueError("Filtered speech count exceeds the window")
            eligible = 2 * speech >= 3 * RATE
            cell["added_windows"] += eligible
            for arm in ARMS:
                if item[arm] not in ({"ai", "inconclusive"} if eligible else {None}):
                    raise ValueError("Response does not match the speech admission")
                cell[f"{arm}_new_ai_windows"] += item[arm] == "ai"
        for arm in ARMS:
            directions = known | {tuple(item["span"]): "ai" for item in row["windows"] if item[arm] == "ai"}
            new = parent.regions(metadata["samples"], directions)
            useful = parent.clock.useful(new, truth)
            gain = useful and not existing
            cell.update({f"{arm}_flags": bool(base or new), f"{arm}_useful": existing or useful,
                         f"{arm}_added": gain, f"{arm}_new_flags": bool(new) and not base})
            if gain and scene != "gain_control":
                gains[arm].add(group)
                families[arm].add(info["family"])
            if not truth and new and not base:
                alarms[arm, role].add(group)
    summary = {arm: {"new_noisy_groups": len(gains[arm]), "new_noisy_families": sorted(families[arm]),
                     "added_genuine_groups": len(alarms[arm, "genuine"]), "added_isolated": len(alarms[arm, "isolated"]),
                     "passes": len(gains[arm]) >= 3 and len(families[arm]) >= 2 and len(alarms[arm, "genuine"]) <= 1
                     and not alarms[arm, "isolated"] and not cells["gain_control/genuine"][f"{arm}_new_flags"]} for arm in ARMS}
    return {"cells": {key: dict(value) for key, value in cells.items()}, "summary": summary}


def run(source, native, output):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from app.detectors.audio import calibration
    from scripts.audio_gtcrn_traffic import float_pcm
    from scripts.audio_reduced_background_localization import assets_for_scores
    from scripts.audio_strong_traffic_view import network_free

    parent.native_folder(native, source)
    if source["freeze"]["method"]["calibration_sha256"] != calibration.SHA256:
        raise ValueError("Existing voice calibration differs")
    output.mkdir(mode=0o700)
    lock = contract()
    parent.io.save(output / "freeze.json", lock)
    network_free()
    voice, activity = assets_for_scores(source["freeze"])
    fitted = calibration.load()
    filter_sos = butter(METHOD["order"], METHOD["cutoff_hz"], btype="highpass", fs=RATE, output="sos")
    rows = []
    for metadata, old in zip(source["roster"], source["responses"], strict=True):
        native_pcm = float_pcm(parent.focused.native_wave(native, metadata))
        filtered = sosfiltfilt(filter_sos, native_pcm).astype(np.float32)
        windows = []
        for window in old["windows"]:
            if window["direction"] != "unassessed":
                continue
            a, b = window["span"]
            clip = filtered[a:b]
            speech = parent.samples_of_speech(parent.sample_spans(activity, clip), 0, len(clip))
            directions = dict.fromkeys(ARMS)
            if speech * 2 >= len(clip):
                for arm, score in zip(ARMS, voice.predict([native_pcm[a:b], clip]), strict=True):
                    reading = calibration.initial_decision(calibration.probability(round(score, 6), fitted), fitted)
                    directions[arm] = "ai" if reading == "ai" else "inconclusive"
            windows.append({"span": [a, b], "speech_samples": speech, **directions})
        rows.append({"case": metadata["case"], "windows": windows})
        if len(rows) % 10 == 0:
            print(f"Measured {len(rows)}/{len(source['roster'])}", flush=True)
    receipt = {"freeze": lock, "rows": rows}
    receipt["results"] = review(source, receipt)
    parent.io.save(output / "receipt.json", receipt)
    print(receipt["results"]["summary"], flush=True)


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--native", type=Path)
    args = parser.parse_args()
    source = parent.io.read(parent.PARENT)
    if args.phase == "run":
        if args.native is None:
            parser.error("A run requires the original, pinned source bank")
        run(source, args.native, args.path)
    else:
        receipt = parent.io.read(args.path)
        result = review(source, receipt)
        if result != receipt["results"]:
            raise ValueError("Saved result does not recount")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
