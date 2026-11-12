"""Apply sealed partial-voice heads to the previous six-second meeting-audio panel."""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import audio_partial_residual_transfer as fit
from scripts.audio_room_reference_transfer import parent_inputs

STARTS = (0, 32_000)
SCOPE = "No-refit diagnostic on 48 earlier exposed six-second inputs; use the maximum across exact windows [0,4) and [2,6) seconds."


def sources(parent):
    return sorted(((row, raw) for pair in parent_inputs(parent) for row, raw, _ in pair.values()),
                  key=lambda value: value[0]["id"])


def freeze(trial, parent, rows):
    if fit.read(trial / "result.json")["fit_sha256"] != fit.sha(trial / "fit.json"):
        raise ValueError("Expected the sealed pilot fit")
    return {"scope": SCOPE, "fit_sha256": fit.sha(trial / "fit.json"), "script_sha256": fit.sha(Path(__file__)),
            "parent_roster_sha256": fit.sha(parent / "roster.json"), "starts": list(STARTS),
            "inputs": [[row["id"], row["source_pcm_sha256"],
                        [hashlib.sha256(raw[2*a:2*(a+fit.SAMPLES)]).hexdigest() for a in STARTS]] for row, raw in rows]}


def summarize(rows):
    cells = defaultdict(Counter)
    for row in rows:
        for arm, observation in row["checks"].items():
            positive = row["kind"] == "partial_ai"
            flag, located = observation["flag"], observation["location"]
            cell = cells[row["panel"], row["condition"], arm]
            cell.update(positive=int(positive), genuine=int(not positive), true_flag=int(positive and flag),
                        false_flag=int(not positive and flag), useful=int(flag and located), threshold_free_location=int(located))
    return [{"role": role, "condition": cond, "arm": arm, **counts} for (role, cond, arm), counts in sorted(cells.items())]


def run(trial, rows, weights, lock):
    import torch

    from app.detectors.audio.upgraded import prepare_voice_window
    from scripts.run_audio_specialists_v15 import load_voice

    seal = fit.read(trial / "fit.json")
    model = load_voice({"weights": str(weights), "assets": fit.ASSETS})
    capture = fit.TemporalCapture(model)
    ledger = []
    started = time.monotonic()
    try:
        for row, raw in rows:
            winners = {}
            model_inputs = []
            for start in STARTS:
                pcm = np.frombuffer(raw[2*start:2*(start+fit.SAMPLES)], "<i2").astype(np.float32) / 32_768
                value = prepare_voice_window(pcm)
                model_inputs.append(hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest())
                capture.record = None
                with torch.inference_mode():
                    clip = model(torch.from_numpy(value)[None]).item()
                vectors, local = capture.record
                representation = fit.representation(vectors[None].astype(np.float64))[0]
                for arm, values in fit.cell_responses(representation, local.astype(np.float64), clip, seal["heads"]).items():
                    score, a, b = fit.select(values)
                    if arm not in winners or winners[arm][0] < score:
                        winners[arm] = score, start+a, start+b
            checks = {arm: {"range": [a, b], "flag": score > seal["points"][arm]["point"],
                            "location": fit.useful_range(a, b, row["truth"])} for arm, (score, a, b) in winners.items()}
            ledger.append({**{key: row[key] for key in ("id", "panel", "condition", "kind", "truth")},
                           "model_inputs": model_inputs, "checks": checks})
            if len(ledger) % 12 == 0:
                print(f"Six-second replay: {len(ledger)} files", flush=True)
    finally:
        capture.close()
    fit.write(trial / "meeting-replay.json", {"freeze_sha256": fit.sha(lock), "forwards": 2 * len(rows),
              "seconds": round(time.monotonic() - started, 3), "cells": summarize(ledger), "decisions": ledger})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    parser.add_argument("trial", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    rows = sources(args.parent)
    lock = args.trial / "meeting-replay-freeze.json"
    expected = freeze(args.trial, args.parent, rows)
    if args.phase == "freeze":
        fit.write(lock, expected)
        print(f"Frozen {len(rows)} six-second inputs without fitting", flush=True)
    else:
        if expected != fit.read(lock) or (args.trial / "meeting-replay.json").exists():
            raise ValueError("Diagnostic inputs changed or are already scored")
        run(args.trial, rows, args.weights, lock)


if __name__ == "__main__":
    main()
