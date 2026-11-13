# SPDX-License-Identifier: Apache-2.0
"""Measure a fixed speech-frame direction change cue against exact edit times."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_focused_window_transfer as focused
from scripts import audio_subband_partial_transfer as prior
from scripts import audio_vocal_event_mixtures as io
from scripts import review_audio_reduced_background_localization as clock

SCHEMA = "chipmunk.audio-adjacent-frame-transfer/1"
LAYER, FRAME_HOP, RECEPTIVE_FIELD = 10, 320, 400
WINDOW, HOP = 12_800, 3_200
VAD_ASSET = "silero_vad.jit"
METHOD = {
    "layer": LAYER, "embedding_hop_samples": FRAME_HOP, "embedding_receptive_samples": RECEPTIVE_FIELD,
    "range_samples": WINDOW, "range_hop_samples": HOP, "speech_fraction": .5,
    "feature": "RMS of adjacent unit-normalized embedding displacements with transition centers in range.",
    "choice": "Earliest maximum admitted native range, with no additional routing or fitted head.",
    "point": "Strictly above the largest selection maximum; retain all focused ranges.",
}
require = prior.require


def freeze(assets):
    scripts = (Path(__file__), Path(prior.__file__), Path(focused.__file__), Path(clock.__file__),
               Path(__file__).with_name("audio_wavlm_features_v13.py"))
    return {"schema": SCHEMA, "method": METHOD, "scripts": {path.name: io.sha(path) for path in scripts},
            "panels": {name: io.sha(io.PUBLIC / name) for name in prior.PANELS}, "vad_sha256": assets[VAD_ASSET]}


def ranges(samples, speech):
    for start in range(0, samples - WINDOW + 1, HOP):
        end = start + WINDOW
        if sum(clock.overlap((start, end), span) for span in speech) >= WINDOW // 2:
            yield start, end


def measure(row, folder, encoder, activity):
    import torch

    from scripts.audio_gtcrn_traffic import float_pcm

    wave = float_pcm(prior.native(folder, row))
    speech = [[round(a * clock.RATE), round(b * clock.RATE)] for a, b in activity.spans(wave)]
    scores = []
    admitted = list(ranges(len(wave), speech))
    if admitted:
        with torch.inference_mode():
            inputs = torch.from_numpy(wave)[None]
            states = encoder(inputs, attention_mask=torch.ones_like(inputs, dtype=torch.long), output_hidden_states=True)
            state = states.hidden_states[LAYER][0]
            expected = (len(wave) - RECEPTIVE_FIELD) // FRAME_HOP + 1
            require(state.shape == (expected, 768) and torch.isfinite(state).all().item(), "Encoder frame clock changed")
            unit = torch.nn.functional.normalize(state, dim=-1)
            jumps = (unit[1:] - unit[:-1]).square().sum(-1).tolist()
        first_center = (RECEPTIVE_FIELD + FRAME_HOP) // 2
        for start, end in admitted:
            first = max(0, (start - first_center + FRAME_HOP - 1) // FRAME_HOP)
            stop = min(len(jumps), (end - first_center + FRAME_HOP - 1) // FRAME_HOP)
            scores.append(math.sqrt(math.fsum(jumps[first:stop]) / (stop - first)).hex())
    return {"id": row.get("case", row.get("id")), "speech": speech, "scores": scores}


def best(row, record, folder=None):
    require(record["id"] == row.get("case", row.get("id")), "Delivery ID changed")
    spans = record["speech"]
    for index, (a, b) in enumerate(spans):
        require(type(a) is type(b) is int and 0 <= a < b <= row["samples"]
                and (index == 0 or spans[index - 1][1] < a), "Invalid speech admission range")
    if folder is not None:
        prior.native(folder, row)
    readings = [(float.fromhex(value), span) for span, value in
                zip(ranges(row["samples"], spans), record["scores"], strict=True)]
    require(all(math.isfinite(score) and 0 <= score <= 2.000001 for score, _ in readings), "Invalid direction response")
    return max(readings, key=lambda reading: reading[0], default=(None, None))


def operating(rows, records, folder=None):
    scores = [best(row, record, folder)[0] for (row, _), record in zip(rows, records, strict=True)]
    admitted = [score for score in scores if score is not None]
    require(admitted, "No speech-admitted selection delivery")
    return {"point": max(admitted).hex(), "admitted_deliveries": len(admitted), "records_sha256": io.commitment(records)}


def evaluate(rows, records, point, folder=None):
    cells = defaultdict(Counter)
    false_groups = defaultdict(set)
    additions = set()
    for (row, original), record in zip(rows, records, strict=True):
        directions = {tuple(item["span"]): item["direction"] for item in original}
        baseline = focused.ranges(row["samples"], directions)["focused"]
        value, span = best(row, record, folder)
        accepted = [span] if value is not None and value > point else []
        meta = row["source"]
        scene = row["scene"]
        label, truth = (meta["label"], meta["edits"]) if meta else ("isolated", [])
        group = meta["group"] if meta else row["case"]
        cell = cells[f"{scene}/{label}"]
        cell["files"] += 1
        cell["cue_assessed"] += value is not None
        cell["cue_useful_before_point"] += bool(span and clock.useful([span], truth))
        if scene != "gain_control" and clock.useful(accepted, truth) and not clock.useful(baseline, truth):
            additions.add((group, meta["family"], scene))
        for name, marked in (("focused", baseline), ("cue", accepted), ("combined", baseline + accepted)):
            useful = clock.useful(marked, truth)
            cell[f"{name}_useful"] += useful
            cell[f"{name}_flags"] += bool(marked)
            cell[f"{name}_misplaced_edits"] += bool(truth and marked and not useful)
            cell[f"{name}_incorrect_samples"] += sum(
                b - a - sum(clock.overlap((a, b), edit) for edit in truth) for a, b in clock.intervals_union(marked)
            )
            if marked and not truth:
                false_groups[name, scene, label].add(group)
                false_groups[name, "noisy" if scene != "gain_control" else "clean", label].add(group)
    return {"cells": dict(cells), "false_groups": {"/".join(key): len(groups) for key, groups in false_groups.items()},
            "added_noisy_groups": len({group for group, _, _ in additions}),
            "added_noisy_families": sorted({family for _, family, _ in additions}),
            "added_noisy_scenes": sorted({scene for _, _, scene in additions})}


def run(directory, selection_dir, held_dir, weights):
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts import audio_wavlm_features_v13 as wavlm

    rows, assets = prior.inputs()
    directory.mkdir(mode=0o700)
    locked = freeze(assets)
    io.save(directory / "freeze.json", locked)
    print("Frozen", io.sha(directory / "freeze.json"), flush=True)
    vad_path = io.WEIGHTS / VAD_ASSET
    require(io.sha(vad_path) == assets[VAD_ASSET], "Pinned speech gate changed")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    encoder, activity = wavlm.load(weights), SpeechActivity(vad_path)

    def observations(role, source):
        records = []
        for row, _ in rows[role]:
            records.append(measure(row, source, encoder, activity))
            if len(records) % 10 == 0:
                print(role, len(records), flush=True)
        return records

    selection = observations("selection", selection_dir)
    cutoff = operating(rows["selection"], selection)
    io.save(directory / "point.json", cutoff)
    print("Frozen point", cutoff, flush=True)
    held = observations("held", held_dir)
    result = evaluate(rows["held"], held, float.fromhex(cutoff["point"]))
    io.save(directory / "receipt.json", {"freeze": locked, "selection": selection, "point": cutoff,
                                        "held": held, "results": result})


def review(path, selection_dir=None, held_dir=None):
    receipt = io.read(path)
    rows, assets = prior.inputs()
    require(receipt["freeze"] == freeze(assets), "Frozen method or input changed")
    point = operating(rows["selection"], receipt["selection"], selection_dir)
    require(point == receipt["point"], "Operating point changed")
    result = evaluate(rows["held"], receipt["held"], float.fromhex(point["point"]), held_dir)
    require(result == receipt["results"], "Held results changed")
    print(io.encoded(result).decode(), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--held", type=Path)
    parser.add_argument("--weights", type=Path, default=io.ROOT / "models/research/wavlm-v13")
    args = parser.parse_args()
    if args.phase == "run":
        if not args.selection or not args.held:
            parser.error("run requires --selection and --held")
        run(args.path, args.selection, args.held, args.weights)
    else:
        review(args.path, args.selection, args.held)
