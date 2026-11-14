# SPDX-License-Identifier: Apache-2.0
"""Test the same staggered voice measurements on matched publisher edits from SINE."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_staggered_voice_transfer as previous

io = previous.baseline.io
SOURCE = io.PUBLIC / "audio-narrow-flag-transfer-2026-11-14.json"
SCHEMA = "chipmunk.audio-staggered-sine/1"
RATE = previous.RATE
SAMPLES = 6 * RATE
SCENES = ("clean", "rain", "train", "competing_speech")
ARMS = ("current", "staggered", "shorter", "earlier_copy")


def contract():
    parent = io.read(SOURCE)
    return {"schema": SCHEMA, "source_sha256": io.sha(SOURCE), "cases": [
        {"id": row["id"], "scene": scene, "side": side, "truth": row["truth"] if side == "variant" else None}
        for row in parent["freeze"]["pairs"] for scene in SCENES for side in ("original", "variant")
    ] + [{"id": slot, "scene": scene, "side": "isolated", "truth": None}
         for scene, count in (("rain", 4), ("train", 4), ("competing_speech", len(parent["freeze"]["backgrounds"])))
         for slot in range(count)],
        "decision": "Keep earlier evidence. For either single-file addition require at least three added useful locations "
                    "across two different noisy scenes and two readers, at most one added genuine source group and zero "
                    "isolated findings. This tests editor transfer and uses already retained readers and licensed backgrounds."}


def intervals():
    return sorted(set(previous.schedule(SAMPLES)) | set(previous.baseline.binding.intervals(SAMPLES)))


def current(windows):
    sparse = previous.baseline.binding.intervals(SAMPLES)
    return list(sparse) if all(windows[span] == "ai" for span in sparse) else []


def review(doc, frozen):
    if doc["freeze"] != frozen:
        raise ValueError("Source roster differs from the pinned SINE receipt")
    cells, gains, losses = defaultdict(Counter), defaultdict(set), defaultdict(set)
    for case, row in zip(frozen["cases"], doc["rows"], strict=True):
        if any(row[key] != item for key, item in case.items()) or not previous.baseline.clock.SHA.fullmatch(row["pcm_sha256"]):
            raise ValueError("Missing or altered source identity")
        lookup = {}
        for item in row["windows"]:
            span = tuple(item["samples"])
            count, direction = item["speech_samples"], item["direction"]
            if (len(span) != 2 or type(count) is not int or not 0 <= count <= span[1] - span[0]
                    or direction not in previous.baseline.binding.DIRECTIONS
                    or (direction == "unassessed") != (2 * count < span[1] - span[0])):
                raise ValueError("Invalid speech admission or direction")
            lookup[span] = direction
        if list(lookup) != intervals() or len(row["windows"]) != len(lookup):
            raise ValueError("Missing or repeated timing window")
        base = current(lookup)
        incumbent = {span: lookup[span] for span in previous.baseline.binding.intervals(SAMPLES)}
        extras = {span: lookup[span] for span in previous.schedule(SAMPLES)}
        reference = row["earlier_copy"]
        span = reference["interval"]
        state = reference["state"]
        if (state not in {"reference_match_not_supported", "local_support_not_available", "no_local_change", "local_change",
                          "consistent_frequency_balance_change"} or bool(span) != (state == "local_change")
                or span and (len(span) != 2 or not 0 <= span[0] < span[1] <= SAMPLES or span[1] - span[0] != 15360)):
            raise ValueError("Invalid earlier-copy result")
        arms = {"current": base, **{name: base + previous.proposals(incumbent, extras, name) for name in ARMS[1:3]},
                "earlier_copy": [span] if span else []}
        scene, role, truth = case["scene"], case["side"], case["truth"]
        cell = cells[f"{scene}/{role}"]
        cell["files"] += 1
        cell["window_scored"] += sum(value != "unassessed" for value in lookup.values())
        cell["windows"] += len(lookup)
        cell["reference_admitted"] += state != "reference_match_not_supported"
        if truth:
            planned = dict.fromkeys(lookup, "ai")
            admitted = {key: "ai" if value != "unassessed" else value for key, value in lookup.items()}
            for name in ARMS[1:3]:
                for stage, available in (("planned", planned), ("admitted", admitted)):
                    candidates = previous.proposals({span: available[span] for span in incumbent},
                                                    {span: available[span] for span in extras}, name)
                    cell[f"{name}_{stage}_reachable"] += previous.baseline.clock.useful(candidates, [truth])
        for name, findings in arms.items():
            located = previous.baseline.clock.useful(findings, [truth] if truth else [])
            cell.update({f"{name}_flagged": bool(findings), f"{name}_useful": located,
                         f"{name}_misplaced": bool(truth and findings and not located)})
            if located and scene != "clean" and not previous.baseline.clock.useful(base, [truth]):
                gains[name].add((row["id"], scene))
            if findings and not truth and not base:
                losses[name].add((role, row["id"], scene if role == "isolated" else None))
    result = {name: {"noisy_additions": len(gains[name]), "noisy_scenes": len({scene for _, scene in gains[name]}),
                     "readers": len({reader for reader, _ in gains[name]}),
                     "added_genuine_groups": sum(role == "original" for role, *_ in losses[name]),
                     "added_isolated": sum(role == "isolated" for role, *_ in losses[name])} for name in ARMS[1:]}
    for name in ARMS[1:3]:
        row = result[name]
        row["passes"] = (row["noisy_additions"] >= 3 and row["noisy_scenes"] >= 2 and row["readers"] >= 2
                         and row["added_genuine_groups"] <= 1 and row["added_isolated"] == 0)
    return {"cells": {key: dict(value) for key, value in cells.items()}, "summary": result}


def score(output, bank, sounds):
    import numpy as np

    from app.evidence.audio_clean_speech_dense import locate_with_frequency_balance
    from scripts.audio_reduced_background_localization import assets_for_scores
    from scripts.audio_reference_variable_window import read_noise
    from scripts.audio_sine_resynthesis_transfer import mix, native_rows
    from scripts.audio_strong_traffic_view import network_free

    parent = io.read(SOURCE)
    frozen = contract()
    if io.read(bank / "freeze.json") != parent["freeze"]:
        raise ValueError("Retained SINE source roster differs")
    background = read_noise(sounds)
    panel = list(native_rows(bank, parent["freeze"]))
    clean = {(r["id"], r["side"]): s for r, s in panel if r["condition"] == "clean"}
    competing = {(r["id"], r["side"]): s for r, s in panel if r["condition"] == "competing_speech"}
    rooms = [row["room"] for row in parent["freeze"]["backgrounds"]]
    mixtures = {(i, scene): mix(clean[i, "original"], clean[i, "variant"], tracks[i % len(tracks)])
                for i, _ in enumerate(parent["freeze"]["pairs"]) for scene, tracks in background.items()}
    placements = {}
    for case in frozen["cases"]:
        i, scene, side = (case[key] for key in ("id", "scene", "side"))
        earlier = clean[i, "original"]
        if side == "isolated":
            values = competing[i, rooms[i]] if scene == "competing_speech" else background[scene][i]
        elif scene in {"clean", "competing_speech"}:
            values = (clean if scene == "clean" else competing)[i, side]
        else:
            values = mixtures[i, scene][side == "variant"]
        placements[i, scene, side] = earlier, values
    output.mkdir(mode=0o700)
    io.save(output / "freeze.json", frozen)
    network_free()
    voice, activity = assets_for_scores(io.read(previous.PARENT)["freeze"])
    responses = {}
    rows = []
    for case in frozen["cases"]:
        earlier, values = placements[case["id"], case["scene"], case["side"]]
        raw = values.astype("<i2").tobytes()
        signal = values.astype(np.float32) / 32768
        windows = []
        for a, b in intervals():
            clip = signal[a:b]
            key = io.digest(raw[2*a:2*b])
            if key not in responses:
                speech = previous.baseline.samples_of_speech(previous.baseline.sample_spans(activity, clip), 0, len(clip))
                direction = previous.baseline.predict(voice, clip) if speech * 2 >= len(clip) else "unassessed"
                responses[key] = speech, direction
            speech, direction = responses[key]
            windows.append({"samples": [a, b], "speech_samples": speech, "direction": direction})
        state, proposal, _ = locate_with_frequency_balance(earlier, values)
        rows.append(case | {"pcm_sha256": io.digest(raw), "windows": windows,
                    "earlier_copy": {"state": state, "interval": proposal["samples"] if proposal else None}})
        with (output / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(rows[-1], sort_keys=True) + "\n")
        if len(rows) % 8 == 0:
            print(f"Measured {len(rows)}/{len(frozen['cases'])}", flush=True)
    doc = {"freeze": frozen, "rows": rows}
    doc["results"] = review(doc, frozen)
    io.save(output / "receipt.json", doc)
    print(json.dumps(doc["results"]["summary"], sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--sounds", type=Path)
    args = parser.parse_args()
    if args.phase == "run":
        if args.bank is None or args.sounds is None:
            parser.error("Specify the exact SINE and background source folders")
        score(args.path, args.bank, args.sounds)
    else:
        doc = io.read(args.path)
        observed = review(doc, contract())
        if observed != doc["results"]:
            raise ValueError("Saved counts differ from the measurement ledger")
        print(json.dumps(observed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
