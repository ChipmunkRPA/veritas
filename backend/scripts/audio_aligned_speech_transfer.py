"""Replay the clean earlier-copy alignment and current app on held SINE readers."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from app.evidence import audio_aligned_speech as alignment
from scripts import audio_aligned_speech_difference as original
from scripts import audio_reference_variable_window as window
from scripts import audio_sine_resynthesis_transfer as sine
from scripts import audio_vocal_event_mixtures as io

PARENT = io.PUBLIC / "audio-aligned-speech-difference-2026-11-13.json"
PANELS = ("audio-sine-resynthesis-transfer", "audio-narrow-flag-transfer")
SCENE_NAMES = ("rain", "train", "competing_speech")
SCHEMA = "chipmunk.earlier-copy-aligned-speech-transfer/1"
SHA256 = re.compile(r"[a-f0-9]{64}")
POINT = .443620
RESIDUAL = .10
METHOD = {
    "selection": "Frozen alignment point and whole-recording processing guard from the earlier alignment study.",
    "scope": "Two reader-disjoint SINE panels, each separate from the preview reader. Compare each retained six-second "
             "cut-and-paste and resynthesis delivery with its clean earlier copy. Use the current earlier-copy route "
             "with dense same-clock speech, frequency balance and timing enabled.",
    "controls": "Unchanged, gain, equalized, uniformly resampled and unrelated. Rain, train and competing speech are "
                "mixed into the submitted recording only at minus 5 dB relative to the original.",
    "useful": "Require at least half of the marked edit and 0.30 temporal IoU. Compare app intervals up to 2.4 s because "
              "whole-recording sample differences do not localize a short speech edit.",
    "local_trial": "Retain the current findings. Require four additional readers across the two clean panels with at most "
                   "two newly flagged ordinary reader groups. Report noisy scope separately.",
}


def sources():
    return [io.PUBLIC / f"{name}-2026-11-14.json" for name in PANELS]


def cases(receipt, folder, noise):
    if (frozen := io.read(folder / "freeze.json")) != receipt["freeze"]:
        raise ValueError("Source preparation differs from the earlier receipt")
    pins = {(row.get("id", row.get("group")), row["condition"], row["side"]): row["pcm_sha256"]
            for row in receipt["records"]}
    groups = {}
    for row, pcm in sine.native_rows(folder, frozen):
        if pins[row["id"], row["condition"], row["side"]] != io.digest(pcm.tobytes()):
            raise ValueError("Source PCM differs from the publisher-derived receipt")
        groups.setdefault(row["id"], {})[row["condition"], row["side"]] = pcm
    originals = [groups[row["id"]]["clean", "original"] for row in frozen["pairs"]]
    for index, item in enumerate(frozen["pairs"]):
        if item["role"] not in {"cut_paste", "resyn"}:
            continue
        earlier = originals[index]
        views = groups[item["id"]]
        truth = [item["truth"]] if item["truth"] else []
        wrong = originals[(index + 1) % len(originals)]
        for kind, submitted in original.matched_controls(earlier, wrong, rate_ratio=(83, 84)).items():
            yield item, "clean", kind, [], earlier, submitted
        yield item, "clean", item["role"], truth, earlier, views["clean", "variant"]
        for scene, tracks in noise.items():
            untreated, edited = sine.mix(earlier, views["clean", "variant"], tracks[index % len(tracks)])
            yield item, scene, "unchanged", [], earlier, untreated
            yield item, scene, item["role"], truth, earlier, edited
        yield item, "competing_speech", "unchanged", [], earlier, views["competing_speech", "original"]
        yield item, "competing_speech", item["role"], truth, earlier, views["competing_speech", "variant"]


def current(folder, first, second):
    from app.evidence.audio_reference import compare

    earlier, submitted = folder / "earlier.wav", folder / "submitted.wav"
    original.wav(earlier, first)
    original.wav(submitted, second)
    record, _ = compare(earlier, submitted, include_local_timing=True, include_channel_timing=True,
                        include_clean_speech=True, use_dense_clean_speech=True, use_clean_speech_eq_guard=True)
    intervals = set()
    for key in ("audio", "level_adjusted"):
        for span in record.get(key, {}).get("ranges", []):
            measured = span.get("file_ranges", {}).get("submitted", span)
            intervals.add((measured["start_sample"], measured["end_sample"]))
    for key, field in (("clean_speech", "samples"), ("local_timing", "submitted_samples")):
        if span := record[key].get("range"):
            intervals.add(tuple(span[field]))
    return {"status": record["status"],
            "spans": [list(span) for span in sorted(intervals) if 0 < span[1] - span[0] <= 2.4 * alignment.RATE]}


def summarize(rows):
    counts = {}
    gains = set()
    ordinary = set()
    gain_panels = set()
    for row in rows:
        panel, scene, role, reader, truth = (row[key] for key in ("panel", "scene", "role", "reader", "truth"))
        selected = row["alignment"]["selected"]
        residual = row["alignment"]["processing_residual"]
        flags = bool(selected and selected["score"] > POINT and selected["earlier_span"][0] < selected["earlier_span"][1]
                     and (residual is None or residual > RESIDUAL))
        old = row["current"]["spans"]
        current_useful = any(original.common.useful(span, target) for target in truth for span in old)
        useful = flags and any(original.common.useful(selected["span"], target) for target in truth)
        new_ordinary = flags and not old and role in original.CONTROLS
        cell = counts.setdefault((panel, scene, role), Counter())
        cell.update(pairs=1, assessed=bool(selected), current_flagged=bool(old), current_useful=current_useful,
                    alignment_flagged=flags, alignment_useful=useful, added=useful and not current_useful,
                    combined_useful=useful or current_useful, new_ordinary=new_ordinary)
        if scene == "clean" and useful and not current_useful:
            gains.add(reader)
            gain_panels.add(panel)
        if scene == "clean" and new_ordinary:
            ordinary.add(reader)
    return {"cells": {"/".join(key): dict(values) for key, values in counts.items()}, "clean_added_readers": len(gains),
            "clean_added_control_readers": len(ordinary),
            "supports_clean_trial": len(gains) >= 4 and gain_panels == set(PANELS) and len(ordinary) <= 2}


def contract():
    if (alignment.SELECTION_POINT, alignment.PROCESSING_LIMIT) != (POINT, RESIDUAL) or io.read(PARENT)["point"] != POINT:
        raise ValueError("The frozen operating points differ")
    return {"schema": SCHEMA, "method": METHOD, "sources": [io.sha(path) for path in (PARENT, *sources())]}


def review(receipt):
    if any(receipt[key] != value for key, value in contract().items()):
        raise ValueError("Unexpected comparison contract")
    roster = {}
    original_readers = {entry["real"]["filename"].split("-")[1] for entry in original.bank("sine")["cases"].values()}
    for name, path in zip(PANELS, sources(), strict=True):
        pairs = [pair for pair in io.read(path)["freeze"]["pairs"] if pair["role"] in {"cut_paste", "resyn"}]
        readers = {pair["reader"] for pair in pairs if pair["role"] == "cut_paste"}
        if original_readers & readers:
            raise ValueError("The source reader groups overlap")
        original_readers.update(readers)
        for pair in pairs:
            for scene in ("clean", *SCENE_NAMES):
                controls = original.CONTROLS if scene == "clean" else ("unchanged",)
                for kind in (*controls, pair["role"]):
                    roster[name, pair["id"], scene, kind] = (
                        pair["reader"], [pair["truth"]] if pair["truth"] and kind == pair["role"] else [])
    for row in receipt["rows"]:
        key = tuple(row[field] for field in ("panel", "id", "scene", "role"))
        if roster.pop(key, None) != (row["reader"], row["truth"]):
            raise ValueError("A comparison is missing, repeated or inconsistent with the publisher label")
        a, b = row["samples"]
        if not (alignment.MIN_SAMPLES <= min(a, b) <= max(a, b) <= alignment.MAX_SAMPLES
                and len(row["sha256"]) == 2 and all(SHA256.fullmatch(value) for value in row["sha256"])
                and all(0 <= start < end <= b for start, end in row["current"]["spans"])):
            raise ValueError("Invalid source clock, input hash or current measurement")
        measurement = row["alignment"]
        median = measurement["median_path_cost"]
        residual = measurement["processing_residual"]
        if not math.isfinite(median) or median < 0 or residual is not None and (not math.isfinite(residual) or residual < 0):
            raise ValueError("Invalid alignment or processing residual")
        if span := measurement["selected"]:
            start, end = span["span"]
            old_start, old_end = span["earlier_span"]
            left, middle, right = span["means"]
            if not (0 <= old_start <= old_end <= a and min(left, middle, right) >= 0
                    and max(left, right, median) <= .7
                    and abs(span["score"] - middle + max(left, right)) <= 2e-6
                    and 0 <= start < end <= b and start % alignment.HOP == end % alignment.HOP == 0
                    and (end - start) // alignment.HOP in alignment.SIZES):
                raise ValueError("An alignment range or its supporting distances differ")
        elif residual is not None:
            raise ValueError("A processing guard reading requires a measured range")
    summary = summarize(receipt["rows"])
    if roster or summary != receipt["results"]:
        raise ValueError("The comparison roster or aggregate results differ")
    return summary


def run(folders, sounds):
    rows = []
    noise = window.read_noise(sounds)
    for name, receipt, folder in zip(PANELS, sources(), folders, strict=True):
        with TemporaryDirectory(prefix="speech-alignment-current-") as tmp:
            for item, scene, role, truth, first, second in cases(io.read(receipt), folder, noise):
                measurement = alignment.measure(first, second)
                selected = measurement["selected"]
                measurement["processing_residual"] = (alignment.processing_residual(first, second)
                                                      if selected and selected["score"] > POINT else None)
                rows.append({"panel": name, "id": item["id"], "reader": item["reader"], "scene": scene, "role": role,
                             "truth": truth, "samples": [len(first), len(second)],
                             "sha256": [io.digest(pcm.tobytes()) for pcm in (first, second)], "alignment": measurement,
                             "current": current(Path(tmp), first, second)})
        print(f"Completed {name}", flush=True)
    return contract() | {"rows": rows, "results": summarize(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "review"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--full", type=Path)
    parser.add_argument("--short", type=Path)
    parser.add_argument("--sounds", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        if args.receipt.exists() or not (args.full and args.short and args.sounds):
            parser.error("Native replay requires both source banks, backgrounds and a new receipt path.")
        receipt = run((args.full, args.short), args.sounds)
        review(receipt)
        io.save(args.receipt, receipt)
    else:
        receipt = io.read(args.receipt)
    print(json.dumps(review(receipt), sort_keys=True))


if __name__ == "__main__":
    main()
