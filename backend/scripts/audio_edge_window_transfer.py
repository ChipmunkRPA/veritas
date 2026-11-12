"""Compare edge-inclusive and shifted short voice windows on new noisy edits."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from app.detectors.audio import calibration
from app.detectors.audio.content import window_in_speech
from app.detectors.video.audio import voice_windows
from app.engine.detector import Profile
from app.engine.profiles import SAMPLING
from scripts import audio_communication_serving_grid as native
from scripts import audio_edge_window_sources as fresh

BASE, ROOT, RATE, SAMPLES = native.BASE, native.ROOT, native.RATE, native.SAMPLES
read, write, sha, digest, require = native.read, native.write, native.sha, native.digest, native.require
SCHEMA = "chipmunk.audio-edge-window-transfer/1"
RUN = BASE / "audio-edge-window-transfer-2026-11-12"
HELD = BASE / "audio-edge-window-held-2026-11-12"
NATIVE_RECEIPT = ROOT / "docs/evaluation-panels/audio-communication-serving-grid-2026-11-12.json"
GRIDS = {
    "full": ((0, 64000), (32000, 96000)),
    "edge": ((0, 32000), (32000, 64000), (64000, 96000)),
    "shifted": ((16000, 48000), (48000, 80000)),
    "combined": tuple((start, start + 32000) for start in range(0, 64001, 16000)),
}
GRID = GRIDS["full"] + GRIDS["combined"]
METHOD = {
    "point": "Each arm strictly exceeds the strongest admitted file response among the 40 separate ordinary controls.",
    "comparison": "Best raw logit among speech-admitted windows; break ties by earlier start. Combined uses all five short windows.",
    "admission": "Use one learned speech-activity pass on each continuous exact 16 kHz mono recording. Admit at least half speech.",
    "unit": "Count a genuine group once if any ordinary splice, untouched host, or its isolated-room control flags.",
    "location": "Useful edit range requires at least 0.30 temporal IoU and at least half the edit covered.",
    "decision": "No customer confidence fit. Group-matched and equal-held-false-group outcomes are descriptive.",
}


def calibration_cases():
    committed = read(NATIVE_RECEIPT)["commitments"]
    ledger = native.RUN / "discovery-scores.json"
    require(sha(ledger) == committed["discovery_scores"] and sha(native.RUN / "freeze.json") == committed["freeze"],
            "Published Full score or input freeze changed")
    require(read(native.RUN / "freeze.json")["model_assets"] == native.transfer.OLD.voice.ASSETS,
            "Reused Full scores have different checkpoint bytes")
    discovery = BASE / native.PANELS["discovery"]
    verified = read(native.public_receipt("discovery"))
    require(all(sha(discovery / f"{key}.json") == pin for key, pin in verified["commitments"].items()),
            "Published short-window source commitments changed")
    records = read(ledger)["records"]
    for (row, raw, shifted, _, _), record in zip(native.transfer.panel(discovery, native.PARENT), records, strict=True):
        full = record["views"]["original"]
        require(row["id"] == record["id"] and row["pcm_sha256"] == full["pcm_sha256"] == digest(raw),
                "Reused score belongs to a different recording")
        if row["truth"] is None:
            windows = full["windows"] + shifted
            yield row, raw, {(window["start"], window["end"]): window for window in windows}


def frozen(held):
    require(tuple((round(start * RATE), round(start * RATE) + len(chunk))
                  for start, chunk in voice_windows(np.zeros(SAMPLES, np.float32), SAMPLING[Profile.FULL].voice_hop_s))
            == GRIDS["full"], "Production Full window plan changed")
    require(sum(1 for _ in calibration_cases()) == 40 and sum(1 for _ in fresh.held(held)) == 26,
            "Frozen calibration or held denominator changed")
    return {
        "schema": SCHEMA, "method": METHOD, "grids": {key: [list(window) for window in grid] for key, grid in GRIDS.items()},
        "selection": {"files": 40, "groups": 16}, "held": {"files": 26, "groups": 8, "synthetic": 6},
        "prior_full_receipt": sha(NATIVE_RECEIPT), "held_sources": sha(held / "sources.json"),
        "assets": native.transfer.OLD.voice.ASSETS,
        "code": {name: sha(ROOT / name) for name in (
            "backend/scripts/audio_edge_window_transfer.py",
            "backend/scripts/audio_edge_window_sources.py",
            "backend/scripts/run_audio_specialists_v15.py",
            "backend/app/detectors/audio/upgraded.py",
            "backend/app/detectors/audio/content.py",
            "backend/app/detectors/audio/calibration.py",
            "backend/app/detectors/video/adapters/audio.py",
            "backend/app/detectors/video/audio.py",
        )},
    }


def measure(raw, model, activity, cached):
    values = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    spans = activity.spans(values)
    windows = []
    for start, end in GRID:
        window = (start, end)
        passed = window_in_speech(spans, start / RATE, end / RATE)
        binding = digest(raw[2*start:2*end])
        score = None
        if window in cached:
            known = cached[window]
            score = known["score"]
            require(known["pcm_sha256"] == binding and passed == (score is not None),
                    "Reused window PCM or voice admission changed")
        elif passed:
            probability, = model.predict([values[start:end]])
            score = calibration.logit(probability)
        windows.append({"start": start, "end": end, "score": score, "pcm_sha256": binding, "reused": window in cached})
    return windows


def score_cases(inputs, model, activity, label):
    result = []
    for index, (row, raw, cached) in enumerate(inputs, 1):
        result.append({
            "id": row["id"], "pcm_sha256": digest(raw),
            "windows": measure(raw, model, activity, cached),
        })
        if index % 10 == 0:
            print(f"{label}: measured {index} exact native files", flush=True)
    return result


def strongest(record, grid):
    values = [w["score"] for w in record["windows"] if (w["start"], w["end"]) in grid and w["score"] is not None]
    return max(values, default=-math.inf)


def score(run, held, weights):
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    lock = read(run / "freeze.json")
    require(lock == frozen(held), "Pre-inference definitions, native assets or sealed inputs changed")
    freeze_sha = sha(run / "freeze.json")
    native.transfer.OLD.spectra.incumbent.network_free()
    require(all(sha(weights / name) == pin for name, pin in lock["assets"].items()), "Pinned model or speech gate changed")
    model = load_voice({"weights": str(weights), "assets": lock["assets"]})
    activity = SpeechActivity(weights / "silero_vad.jit")
    selection = score_cases(calibration_cases(), model, activity, "ordinary calibration")
    write(run / "selection.json", {"freeze": freeze_sha, "records": selection})
    points = {arm: max(strongest(record, grid) for record in selection) for arm, grid in GRIDS.items()}
    require(all(math.isfinite(point) for point in points.values()), "Missing finite ordinary calibration point")
    write(run / "points.json", {"freeze": freeze_sha, "calibration": sha(run / "selection.json"), "values": points})
    print("Froze all four ordinary-only operating points before starting held inference.", flush=True)
    records = score_cases(((row, raw, {}) for row, raw in fresh.held(held)), model, activity, "held")
    write(run / "held.json", {"freeze": freeze_sha, "points": sha(run / "points.json"), "records": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("sources", "prepare", "score"))
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--held", type=Path, default=HELD)
    parser.add_argument("--weights", type=Path, default=native.WEIGHTS)
    args = parser.parse_args()
    if args.phase == "sources":
        fresh.prepare(args.held)
    elif args.phase == "prepare":
        lock = frozen(args.held)
        args.run.mkdir(mode=0o700)
        write(args.run / "freeze.json", lock)
        print(f"Frozen ordinary and held method: {sha(args.run / 'freeze.json')}", flush=True)
    else:
        score(args.run, args.held, args.weights)


if __name__ == "__main__":
    main()
