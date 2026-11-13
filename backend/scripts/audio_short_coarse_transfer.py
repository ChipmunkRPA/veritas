# SPDX-License-Identifier: Apache-2.0
"""Compare short source-clock speech windows with and without wider context."""

from __future__ import annotations

import argparse
import hashlib
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from scripts import review_audio_reduced_background_localization as clock

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "docs/evaluation-panels/audio-noise-window-transfer-2026-11-12.json"
PARENT_PIN = "b2cbbccff57fee617b6d956be6d88bedbcb50309eb43e7d3922b7a422431fc7a"
RATE = clock.RATE
WIDTH, STEP = 2 * RATE, 3 * RATE // 4
SCHEMA = "chipmunk.audio-short-coarse-transfer/1"
ARMS = ("sparse", "dense", "short", "short_with_context", "dense_plus_short_with_context")
METHOD = {
    "width": WIDTH, "step": STEP, "end_aligned": True,
    "candidate_limit": 3, "nonmaximum_suppression": "Suppress intervals overlapping at least half of a selected interval.",
    "context": "Largest original three-second logit that fully contains the short interval, or no support.",
    "group_point": "Per-width point based on the second-highest ordinary selection speaker maximum.",
    "context_score": "Minimum strict empirical rank of short response and containing coarse response.",
    "background_point": "Each cutoff also exceeds the largest value seen on the four isolated selection backgrounds.",
    "evaluation": "A useful interval covers half a known edit with temporal IoU at least .30. Genuine time is deduplicated.",
    "required_for_followup": {"added_noisy_groups": 3, "added_generator_families": 2,
                              "maximum_added_noisy_genuine_groups": 1, "maximum_added_noisy_genuine_fraction": .02},
}


def pin(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save(path, document):
    with path.open("xb") as output:
        output.write(clock.encode(document))


def parent():
    clock.require(pin(PARENT) == PARENT_PIN, "Original speech-window receipt changed")
    document = clock.read(PARENT)
    return document, {
        stage: {row["id"]: row for row in document[stage]["rows"]} for stage in ("selection", "held")
    }


def short_windows(samples):
    last = samples - WIDTH
    return [(start, start + WIDTH) for start in sorted(set(range(0, last + 1, STEP)) | {last})]


def wave_bytes(source, row):
    raw = (source / "pcm" / f"{row['id']:03}.pcm").read_bytes()
    clock.require(len(raw) == 2 * row["samples"] and clock.digest(raw) == row["pcm_sha256"],
                  f"Native speech input changed at row {row['id']}")
    return raw


def context_score(span, responses):
    return max((float.fromhex(row["original"]) for row in responses
                if row["original"] is not None and row["span"][0] <= span[0] and span[1] <= row["span"][1]),
               default=None)


def stage_rows(document, stage):
    return [row for row in document["roster"] if row["role"] == stage]


def validate_rows(document, rows, stage, source=None):
    expected = stage_rows(document, stage)
    clock.require(len(expected) == len(rows), f"Incomplete {stage} observations")
    for row, observed in zip(expected, rows, strict=True):
        spans = short_windows(row["samples"])
        clock.require(set(observed) == {"id", "observations"} and row["id"] == observed["id"]
                      and [tuple(item["span"]) for item in observed["observations"]] == spans,
                      f"Short-window schedule differs at row {row['id']}")
        raw = wave_bytes(source, row) if source is not None else None
        for item in observed["observations"]:
            a, b = item["span"]
            score = item["score"]
            clock.require(set(item) == {"span", "pcm_sha256", "score"} and clock.SHA.fullmatch(item["pcm_sha256"])
                          and (score is None or (math.isfinite(float.fromhex(score)) and float.fromhex(score).hex() == score))
                          and (raw is None or clock.digest(raw[2*a:2*b]) == item["pcm_sha256"]),
                          f"Invalid short-window observation at row {row['id']}")


def points(document, prior, observations):
    short_groups, wide_groups = defaultdict(list), defaultdict(list)
    for row, observed in zip(stage_rows(document, "selection"), observations, strict=True):
        group = row["source"]["group"] if row["source"] is not None else None
        short_groups[group].extend(float.fromhex(item["score"]) for item in observed["observations"]
                                   if item["score"] is not None)
        wide_groups[group].extend(float.fromhex(item["original"]) for item in prior["selection"][row["id"]]["windows"]
                                  if item["original"] is not None)
    short_rank = sorted(max(values) for group, values in short_groups.items() if group is not None and values)
    wide_rank = sorted(max(values) for group, values in wide_groups.items() if group is not None and values)
    clock.require(len(short_rank) >= 10 and len(wide_rank) >= 10, "Too few genuine groups admitted for fitting")
    ordinary = [row for row in stage_rows(document, "selection") if row["source"] is None]
    by_id = {row["id"]: row for row in observations}
    background_context = max((
        min(bisect_left(short_rank, float.fromhex(item["score"])) / len(short_rank),
            bisect_left(wide_rank, wide) / len(wide_rank))
        for row in ordinary for item in by_id[row["id"]]["observations"]
        if item["score"] is not None
        and (wide := context_score(item["span"], prior["selection"][row["id"]]["windows"])) is not None
    ), default=0.0)
    return {"short_genuine_maxima": [value.hex() for value in short_rank],
            "wide_genuine_maxima": [value.hex() for value in wide_rank],
            "short_point": max(short_rank[-2], max(short_groups[None], default=-math.inf)).hex(),
            "context_rank_minimum": max((len(short_rank)-1) / len(short_rank), (len(wide_rank)-1) / len(wide_rank)),
            "isolated_context_max": background_context}


def select(observations, coarse, fit):
    raw_point = float.fromhex(fit["short_point"])
    short_rank, wide_rank = ([float.fromhex(value) for value in fit[name + "_genuine_maxima"]]
                             for name in ("short", "wide"))
    short, supported = [], []
    for item in observations:
        if item["score"] is None:
            continue
        span, score = item["span"], float.fromhex(item["score"])
        if score > raw_point:
            short.append((score, span))
        wide = context_score(span, coarse)
        if wide is None:
            continue
        rank = min(bisect_left(short_rank, score) / len(short_rank), bisect_left(wide_rank, wide) / len(wide_rank))
        if rank >= fit["context_rank_minimum"] and rank > fit["isolated_context_max"]:
            supported.append((rank, span))

    def highest(items):
        chosen = []
        for _, span in sorted(items, key=lambda item: (-item[0], item[1][0])):
            if all(2 * clock.overlap(span, previous) < WIDTH for previous in chosen):
                chosen.append(span)
                if len(chosen) == METHOD["candidate_limit"]:
                    break
        return sorted(chosen)

    return highest(short), highest(supported)


def result(document, prior, observed, fit):
    cells = defaultdict(Counter)
    gains, families, false_groups = set(), set(), set()
    for row, record in zip(stage_rows(document, "held"), observed, strict=True):
        length, scene, kind, source = (row[key] for key in ("samples", "scene", "kind", "source"))
        coarse = prior["held"][row["id"]]["windows"]
        direction = {tuple(item["span"]): item["direction"] for item in coarse}
        positive = {span for span, value in direction.items() if value == "ai"}
        sparse, dense = clock.schedules(length)
        sparse_positive = [span for span in sparse if tuple(span) in positive]
        if len(sparse_positive) < 2 or any(direction[tuple(span)] == "original" for span in sparse):
            sparse_positive = []
        dense_positive = clock.intersections(dense, positive)
        short, supported = select(record["observations"], coarse, fit)
        variants = (sparse_positive, dense_positive, short, supported, dense_positive + supported)
        truth = source["edits"] if source else []
        useful = {}
        cell = cells[f"{scene}/{kind}"]
        cell["files"] += 1
        cell["assessed_short"] += any(item["score"] is not None for item in record["observations"])
        cell["genuine_samples"] += length - sum(b-a for a,b in truth)
        for name, intervals in zip(ARMS, variants, strict=True):
            useful[name] = clock.useful(intervals, truth)
            union = clock.intervals_union(intervals)
            cell[f"{name}_flagged"] += bool(union)
            cell[f"{name}_useful"] += useful[name]
            cell[f"{name}_genuine_samples_flagged"] += sum(b-a for a,b in union) - sum(
                clock.overlap(interval, edit) for interval in union for edit in truth)
        if scene != "gain_control" and kind == "partial" and useful[ARMS[-1]] and not useful["dense"]:
            gains.add(source["group"])
            families.add(source["family"])
        if scene != "gain_control" and kind == "genuine" and supported and not dense_positive:
            false_groups.add(source["group"])
    noisy = [row for key, row in cells.items() if not key.startswith("gain_control/")]
    excess = sum(row[f"{ARMS[-1]}_genuine_samples_flagged"] - row["dense_genuine_samples_flagged"] for row in noisy)
    fraction = excess / sum(row["genuine_samples"] for row in noisy)
    target = METHOD["required_for_followup"]
    gates = {
        "added_noisy_groups": len(gains) >= target["added_noisy_groups"],
        "added_families": len(families) >= target["added_generator_families"],
        "added_genuine_groups": len(false_groups) <= target["maximum_added_noisy_genuine_groups"],
        "added_genuine_time": fraction <= target["maximum_added_noisy_genuine_fraction"],
        "clean_and_background_controls": all(cell[f"{ARMS[-1]}_flagged"] == cell["dense_flagged"]
                                             for key, cell in cells.items()
                                             if key == "gain_control/genuine" or key.endswith("/isolated")),
    }
    return {"cells": dict(cells), "new_noisy_partial_groups": len(gains), "new_families": sorted(families),
            "new_noisy_genuine_groups": len(false_groups), "added_noisy_genuine_fraction": fraction,
            "gates": gates, "passes_followup_gate": all(gates.values())}


def score(document, source, run, stage, voice, vad):
    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from scripts.audio_gtcrn_traffic import float_pcm

    rows = []
    for row in stage_rows(document, stage):
        raw = wave_bytes(source, row)
        pcm = float_pcm(raw)
        speech = vad.spans(pcm)
        observations = []
        for a, b in short_windows(row["samples"]):
            response = logit(voice.predict([pcm[a:b]])[0]).hex() if window_in_speech(speech, a / RATE, b / RATE) else None
            observations.append({"span": [a, b], "pcm_sha256": clock.digest(raw[2*a:2*b]), "score": response})
        rows.append({"id": row["id"], "observations": observations})
        if len(rows) % 8 == 0:
            print(stage, len(rows), "deliveries", flush=True)
    save(run / f"{stage}.json", rows)
    print(stage, len(rows), "complete", flush=True)
    return rows


def measure(source, run, freeze):
    from scripts.audio_reduced_background_localization import assets_for_scores

    document, prior = parent()
    clock.require(pin(source / "receipt.json") == PARENT_PIN, "Local parent receipt differs")
    run.mkdir(mode=0o700)
    save(run / "freeze.json", freeze)
    voice, vad = assets_for_scores(document["freeze"])
    selection = score(document, source, run, "selection", voice, vad)
    fit = points(document, prior, selection)
    save(run / "point.json", fit)
    print("Short-window and context points fixed.", flush=True)
    held = score(document, source, run, "held", voice, vad)
    save(run / "receipt.json", {"freeze": freeze, "selection": selection, "points": fit, "held": held,
                                "result": result(document, prior, held, fit)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("measure", "review"))
    parser.add_argument("output", type=Path, help="New run directory or the receipt to review")
    parser.add_argument("--source", type=Path, help="Prior local audio-noise-window-transfer run with original PCM")
    args = parser.parse_args()
    freeze = {"schema": SCHEMA, "method": METHOD, "parent_sha256": PARENT_PIN, "runner_sha256": pin(Path(__file__))}
    if args.phase == "measure":
        if args.source is None:
            parser.error("Measurement requires the frozen native source run")
        measure(args.source, args.output, freeze)
    else:
        document, prior = parent()
        record = clock.read(args.output)
        clock.require(record["freeze"] == freeze, "Short-window protocol differs")
        for stage in ("selection", "held"):
            validate_rows(document, record[stage], stage, args.source)
        fit = points(document, prior, record["selection"])
        clock.require(record["points"] == fit, "Selection point differs")
        summary = result(document, prior, record["held"], fit)
        clock.require(record["result"] == summary, "Held results differ")
        print(clock.encode(summary).decode(), end="")


if __name__ == "__main__":
    main()
