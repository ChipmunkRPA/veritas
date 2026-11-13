# SPDX-License-Identifier: Apache-2.0
"""Test whether transient reduction improves short edit evidence on exact PCM."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_focused_window_transfer as focused
from scripts import audio_subband_partial_transfer as sources
from scripts import audio_vocal_event_mixtures as io
from scripts import review_audio_reduced_background_localization as clock

SCHEMA = "chipmunk.audio-transient-reduction-transfer/1"
FFT, HOP, MEDIAN = 512, 128, 17
METHOD = {
    "fft_samples": FFT, "hop_samples": HOP, "median_frames_or_bins": MEDIAN, "minimum_amplitude": .5,
    "view": "Median magnitude across time versus frequency. Power-ratio harmonic gain with a 0.5 amplitude floor; original phase and length retained.",
    "routing": "Existing dense native inconclusive and unassessed windows. Reapply speech activity to each filtered view.",
    "point": "Strictly above the largest admitted genuine or isolated selection response.",
    "agreement": "Two adjacent dense windows must lean AI, with at least one newly positive filtered view. Retain the focused result.",
}
require = sources.require


def freeze():
    return {"schema": SCHEMA, "method": METHOD, "runner_sha256": io.sha(Path(__file__)), "input_plan": sources.frozen()}


def project(wave):
    import numpy as np
    from scipy.ndimage import median_filter
    from scipy.signal import istft, stft

    _, _, spectrum = stft(wave.astype(np.float64), fs=clock.RATE, window="hann", nperseg=FFT, noverlap=FFT-HOP)
    magnitude = np.abs(spectrum)
    horizontal = median_filter(magnitude, size=(1, MEDIAN), mode="nearest") ** 2
    vertical = median_filter(magnitude, size=(MEDIAN, 1), mode="nearest") ** 2
    gain = np.clip(horizontal / np.maximum(horizontal + vertical, 1e-15), .5, 1.)
    _, restored = istft(spectrum * gain, fs=clock.RATE, window="hann", nperseg=FFT, noverlap=FFT-HOP)
    require(len(restored) >= len(wave) and np.isfinite(restored).all(), "Original finite sample clock required")
    return restored[:len(wave)].astype("<f4")


def transformed(folder, row):
    from scripts.audio_gtcrn_traffic import float_pcm

    return project(float_pcm(sources.native(folder, row)))


def measure(rows, folder, voice, activity, label):
    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech

    output = []
    for row, native in rows:
        filtered = transformed(folder, row)
        checked = []
        for start, end in sources.eligible(row, native):
            clip = filtered[start:end]
            score = logit(voice.predict([clip])[0]) if window_in_speech(activity.spans(clip), 0., 3.) else None
            checked.append({"span": [start, end], "sha256": io.digest(clip.tobytes()),
                            "logit": score.hex() if score is not None else None})
        output.append({"id": row.get("case", row.get("id")), "windows": checked})
        if len(output) % 10 == 0:
            print(label, len(output), flush=True)
    return output


def readings(row, native, record, folder=None):
    require(record["id"] == row.get("case", row.get("id")), "Delivery identity changed")
    spans = sources.eligible(row, native)
    require([item["span"] for item in record["windows"]] == spans, "Native window schedule changed")
    wave = transformed(folder, row) if folder is not None else None
    for item in record["windows"]:
        start, end = item["span"]
        digest = item["sha256"]
        value = float.fromhex(item["logit"]) if item["logit"] is not None else None
        require(clock.SHA.fullmatch(digest) and (wave is None or digest == io.digest(wave[start:end].tobytes()))
                and (value is None or math.isfinite(value)), "Projected response or samples changed")
        yield (start, end), value


def point(rows, records, folder=None):
    admitted = [value for (row, native), record in zip(rows, records, strict=True)
                for _, value in readings(row, native, record, folder) if value is not None]
    require(admitted, "No admitted selection responses")
    return {"threshold": max(admitted).hex(), "admitted": len(admitted), "responses_sha256": io.commitment(records)}


def evaluate(rows, records, cutoff, folder=None):
    cells, false = defaultdict(Counter), defaultdict(set)
    additions = set()
    for (row, native), record in zip(rows, records, strict=True):
        directions = {tuple(item["span"]): item["direction"] for item in native}
        old = focused.ranges(row["samples"], directions)["focused"]
        positive = {span for span, value in directions.items() if value == "ai"}
        rescues = set()
        readings_seen = list(readings(row, native, record, folder))
        for span, value in readings_seen:
            if value is not None and value > cutoff:
                rescues.add(span)
        proposed = []
        dense = sources.dense(row["samples"])
        for (a, b), (c, d) in zip(dense, dense[1:]):
            left, right = (a, b), (c, d)
            if left in positive | rescues and right in positive | rescues and (left in rescues or right in rescues):
                proposed.append([c, min(b, d)])
        meta, scene = row["source"], row["scene"]
        label, truth = (meta["label"], meta["edits"]) if meta else ("isolated", [])
        cell = cells[f"{scene}/{label}"]
        cell["files"] += 1
        cell["filtered_windows"] += len(readings_seen)
        cell["assessed_filtered_windows"] += sum(value is not None for _, value in readings_seen)
        cell["positive_filtered_windows"] += len(rescues)
        cell["useful_additions"] += clock.useful(proposed, truth) and not clock.useful(old, truth)
        if truth and clock.useful(proposed, truth) and not clock.useful(old, truth):
            additions.add((scene, meta["family"], meta["group"]))
        if not truth and proposed and not old:
            false[f"{scene}/{label}"].add(meta["group"] if meta else row["case"])
        for label, ranges in (("focused", old), ("combined", old + proposed)):
            cell[f"{label}_flagged"] += bool(ranges)
            cell[f"{label}_useful"] += clock.useful(ranges, truth)
            cell[f"{label}_incorrect_samples"] += sum(b-a-sum(clock.overlap((a, b), edit) for edit in truth)
                                                         for a, b in clock.intervals_union(ranges))
    return {"cells": dict(cells), "new_false_groups": {key: len(value) for key, value in false.items()},
            "additional_useful_source_groups": len({group for _, _, group in additions}),
            "additional_useful_scenes": sorted({scene for scene, _, _ in additions}),
            "additional_useful_families": sorted({family for _, family, _ in additions})}


def run(directory, selection_dir, held_dir):
    import torch

    from scripts.audio_reduced_background_localization import assets_for_scores

    roles, assets = sources.inputs()
    directory.mkdir(mode=0o700)
    locked = freeze()
    io.save(directory / "freeze.json", locked)
    torch.set_num_threads(2)
    voice, activity = assets_for_scores({"voice_assets": assets})
    selection = measure(roles["selection"], selection_dir, voice, activity, "selection")
    cutoff = point(roles["selection"], selection)
    io.save(directory / "point.json", cutoff)
    print("Frozen point", cutoff, flush=True)
    held = measure(roles["held"], held_dir, voice, activity, "held")
    io.save(directory / "receipt.json", {"freeze": locked, "selection": selection, "point": cutoff, "held": held,
                                       "results": evaluate(roles["held"], held, float.fromhex(cutoff["threshold"]))})


def review(path, selection_dir=None, held_dir=None):
    roles, _ = sources.inputs()
    receipt = io.read(path)
    require(receipt["freeze"] == freeze(), "Frozen method changed")
    cutoff = point(roles["selection"], receipt["selection"], selection_dir)
    require(receipt["point"] == cutoff, "Frozen point changed")
    result = evaluate(roles["held"], receipt["held"], float.fromhex(cutoff["threshold"]), held_dir)
    require(receipt["results"] == result, "Reported decisions changed")
    print(io.encoded(result).decode(), end="")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("run", "review"))
    cli.add_argument("path", type=Path)
    cli.add_argument("--selection", type=Path)
    cli.add_argument("--held", type=Path)
    args = cli.parse_args()
    if args.phase == "run":
        if not args.selection or not args.held:
            cli.error("run requires --selection and --held")
        run(args.path, args.selection, args.held)
    else:
        review(args.path, args.selection, args.held)
