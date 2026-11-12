"""Recompute room-reference waveforms and decisions without model inference."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from scripts import audio_room_reference_transfer as study

STARTS = (16_000, 48_000)
LENGTH, FFT, HOP = 32_000, 512, 128
VIEWS = ("stationary_reference", "aligned_reference", "residual_20db")
ROLES = ("selection", "confirmation")
ARMS = ("observed", *VIEWS, "dry_reference")
TAPER = .5 - .5 * np.cos(2 * np.pi * np.arange(FFT) / FFT)


def transform(data, pad):
    padded = np.pad(data, FFT // 2) if pad else data
    return np.fft.rfft(np.lib.stride_tricks.sliding_window_view(padded, FFT)[::HOP] * TAPER) / TAPER.sum()


def reconstruct(spectrum, power):
    output, norm = (np.zeros(LENGTH + FFT) for _ in range(2))
    inverse = np.fft.irfft(spectrum * np.sqrt(np.clip(1. - power / np.maximum(np.abs(spectrum)**2, 1e-12), .04, 1)))
    for index, frame in enumerate(inverse * (TAPER.sum() * TAPER)):
        start = HOP * index
        output[start:start+FFT] += frame
        norm[start:start+FFT] += TAPER**2
    start = FFT // 2
    return output[start:start+LENGTH] / norm[start:start+LENGTH]


def replay(pair, start):
    stop = start + LENGTH
    dry, observed = (np.frombuffer(pair[role][1], "<i2").astype(float) for role in ("clean", "room"))
    background = observed - dry
    segment = observed[start:stop]
    mixture = transform(segment, pad=True)
    power_context = [np.abs(transform(values, pad=False))**2 for values in (background[:start], background[stop:])]
    estimated = np.concatenate(power_context).mean(axis=0)
    aligned = np.abs(transform(background[start:stop], pad=True))**2
    return {"stationary_reference": reconstruct(mixture, estimated), "aligned_reference": reconstruct(mixture, aligned),
            "residual_20db": segment - .9 * background[start:stop], "observed": segment, "dry_reference": dry[start:stop]}


def selected(rows):
    best = max(range(2), key=lambda i: rows[i]["score"])
    return rows[best]["score"], STARTS[best]


def hits(start, truth):
    a, b = truth
    shared = len(range(max(a, start), min(b, start + LENGTH)))
    return shared / (b - a) >= .50 and shared / (LENGTH + b - a - shared) >= .30


def independent_tally(observations, cutoffs):
    combined = {arm: Counter() for arm in ARMS}
    rooms, pairs = {}, {}
    accuracy = {arm: [] for arm in ARMS if arm != "dry_reference"}
    for row, score_set in observations:
        identifier, kind = row["host_group"], row["kind"]
        by_room = rooms.setdefault(str(row["room_index"]), {arm: Counter() for arm in ARMS})
        for arm, records in score_set.items():
            score, start = selected(records)
            pairs.setdefault((identifier, arm), {})[kind] = score
            found = score > cutoffs[arm]
            state = "flag" if found else "inconclusive"
            for table in (combined[arm], by_room[arm]):
                table[f"{kind}_{state}"] += 1
                if row["truth"] and hits(start, row["truth"]):
                    table["useful_before_point"] += 1
                    if found:
                        table["useful_flag"] += 1
                        table["useful_" + row["family"]] += 1
            if arm in accuracy:
                accuracy[arm] += [value["reconstruction_snr_db"] for value in records]
    for (_, arm), values in pairs.items():
        comparison = values["partial_ai"] - values["ordinary"]
        combined[arm]["matched_ai_above" if comparison > 0 else "matched_ai_below" if comparison < 0 else "matched_tie"] += 1
    return {"counts": combined, "rooms": rooms, "median_reconstruction_snr_db": {arm: median(values) for arm, values in accuracy.items()}}


def verify(args):
    folder = args.output
    read, digest, require = study.read, study.digest, study.require
    freeze, point, result = (read(folder / f"{key}.json") for key in ("freeze", "points", "result"))
    require(freeze == study.provenance(args) and result["points_sha256"] == digest(folder / "points.json")
            and point["freeze_sha256"] == digest(folder / "freeze.json"), "Study or point commitment differs")
    parents = study.parent_inputs(args.parent)
    data, forwards, snrs = {}, 0, 0
    for role in ROLES:
        file = folder / f"{role}.json"
        measured = read(file)
        require(digest(file) == result["ledgers_sha256"][role] and measured["freeze_sha256"] == point["freeze_sha256"],
                "Classifier receipt differs")
        require(role != "selection" or point["selection_sha256"] == digest(file), "Fitted point used a different selection ledger")
        data[role] = []
        roster = [pair for pair in parents if pair["room"][0]["panel"] == role]
        require(measured["forwards"] == 2 * len(roster) * len(VIEWS), "Unexpected classifier forward count")
        forwards += measured["forwards"]
        for pair, row in zip(roster, measured["rows"], strict=True):
            source = pair["room"][0]
            require(source["id"] == row["id"] and set(row["arms"]) == set(ARMS), "Scoring source or arm differs")
            for i, start in enumerate(STARTS):
                processed = replay(pair, start)
                reference = processed["dry_reference"]
                for arm, samples in processed.items():
                    saved = row["arms"][arm][i]
                    quantized = np.rint(samples)
                    clipped = int(np.count_nonzero((quantized < -32768) | (quantized > 32767)))
                    integers = np.minimum(np.maximum(quantized, -32768), 32767).astype("<i2")
                    require(study.sha(integers.tobytes()) == saved["sha256"] and saved["clipped"] == clipped
                            and math.isfinite(saved["score"]), "Independent native score window or clipping record differs")
                    residual = np.square(reference - integers).sum()
                    snr = 10 * math.log10(np.square(reference).sum() / residual) if residual else None
                    require(saved["reconstruction_snr_db"] is None if snr is None
                            else abs(snr - saved["reconstruction_snr_db"]) < 1e-9, "Waveform error measure differs")
                    if arm in ("observed", "dry_reference"):
                        label = "room" if arm == "observed" else "clean"
                        require(saved["score"] == pair[label][2][i], "Parent classifier response differs")
                    snrs += 1
            data[role].append((source, row["arms"]))
    controls = [arms for row, arms in data["selection"] if row["truth"] is None]
    points = {arm: max(record["score"] for control in controls for view in
                       (("observed", "dry_reference") if arm in ("observed", "dry_reference") else (arm, "dry_reference"))
                       for record in control[view]) for arm in ARMS}
    require(points == point["points"], "Independent calibration maxima differ")
    panels = {role: independent_tally(rows, points) for role, rows in data.items()}
    require(panels == result["panels"], "Independent outcomes differ")
    held = panels["confirmation"]["counts"]
    next_trial = {arm: held[arm]["useful_flag"] >= 2 and sum(held[arm][f"useful_{family}"] > 0 for family in
                  ("clone", "expressive", "stock")) >= 2 and held[arm]["ordinary_flag"] <= 1 and held[arm]["matched_ai_above"] >= 4
                  for arm in VIEWS[:2]}
    require(next_trial == result["followup"] and forwards == 144 and snrs == 240, "Incomplete independent outcome replay")
    audit = {"new_forwards": forwards, "score_windows": snrs, "pairs": len(parents), "decision_cells": len(ROLES) * len(ARMS),
             "waveform_reconstruction": "Independent NumPy FFT and overlap-add replay; no model inference.",
             "result_sha256": digest(folder / "result.json"), "reviewer_sha256": digest(Path(__file__))}
    study.write(folder / "audit.json", audit)
    if args.publish:
        summary = {**result, "method": freeze["method"], "assets": freeze["assets"],
                   "parent_public_sha256": freeze["parent_public_sha256"], "audit": audit,
                   "commitments": {key: digest(folder / f"{key}.json") for key in ("freeze", "points", "result", "audit")},
                   "runtime": {role: {key: read(folder / f"{role}.json")[key] for key in ("seconds", "rss_bytes")}
                               for role in ROLES}}
        study.write(args.publish, summary)
    print(audit, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "parent", "weights"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--publish", type=Path)
    verify(parser.parse_args())
