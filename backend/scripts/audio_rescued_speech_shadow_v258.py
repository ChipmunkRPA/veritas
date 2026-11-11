"""Score exactly the previously skipped speech admitted by the v256 shadow gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

from scripts import audio_noise_speech_gate_v256 as gate
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = gate.ROOT, gate.BASE, gate.PANEL
MODULE = "scripts.audio_rescued_speech_shadow_v258"
TRIAL = BASE / "audio-rescued-speech-shadow-v258"
STAGE, OUTPUT = TRIAL / "authored", TRIAL / "authored/output"
PUBLIC = PANEL / "audio-rescued-speech-shadow-freeze-v258.json"
PRIOR_RESULT = PANEL / "audio-noise-speech-gate-complete-v256.json"
PRIOR_SHA = "7150fa192232b946caa541e4c6896d4160810a3c10b0d54868fb96b5ea4413b3"
OBSERVATIONS = gate.OUTPUT / "private-observations.json"
PLANNED = {"kitchen_0db": 2, "kitchen_5db": 1, "traffic_0db": 3}
DIRECTIONS = ("ai", "original", "inconclusive")
CALIBRATION = ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json"
CALIBRATION_SHA = "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf"
CODE = (
    "backend/scripts/audio_rescued_speech_shadow_v258.py",
    "backend/tests/test_audio_rescued_speech_shadow_v258.py",
    "docs/audio-rescued-speech-shadow-v258-method.md",
    "backend/scripts/audio_noise_speech_gate_v256.py",
    "backend/scripts/audio_alania_transfer_v179.py",
    "backend/scripts/run_audio_specialists_v15.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/app/detectors/audio/calibration.py",
    "backend/app/detectors/audio/decode.py",
    "backend/app/detectors/audio/upgraded.py",
)
read, write, sha, require = gate.read, gate.write, gate.sha, gate.require


def select(cases, recorded, observed):
    selected = []
    for case, evidence in zip(cases, observed, strict=True):
        require(case["case"] == evidence["case"], "Exact matched observation order required")
        if evidence["spans"] is None:
            continue
        eligibility = gate.coverage(evidence["spans"])
        if eligibility["rescue"] and not eligibility["baseline"]:
            require(
                case["arm"] == "genuine"
                and case["status"] == "prepared"
                and recorded[case["case"]] == "not_applicable",
                "Every rescue must be a previously skipped prepared genuine case",
            )
            selected.append(case)
    require(
        Counter(case["condition"] for case in selected) == PLANNED,
        "Exact six-case v256 rescue roster required",
    )
    return selected


def inputs():
    require(sha(PRIOR_RESULT) == PRIOR_SHA, "Exact published v256 result required")
    verified = gate.verify()
    _, cases, recorded, _, _ = verified
    reconstructed = scoped(gate.readout, {"verify": lambda: verified})()
    require(
        reconstructed == read(PRIOR_RESULT), "Published v256 result differs from verified private receipt"
    )
    return select(cases, recorded, read(OBSERVATIONS)["matched"])


def plan(cases):
    return [{"case": case["case"], "pcm_sha256": case[gate.prior.NATIVE_HASH]} for case in cases]


def identity(cases):
    require(sha(CALIBRATION) == CALIBRATION_SHA, "Exact serving calibration required")
    return {
        "schema": "chipmunk.audio-rescued-speech-shadow-freeze/258",
        "gate_result_sha256": PRIOR_SHA,
        "serving_comparison_freeze_sha256": sha(gate.prior.PUBLIC),
        "serving_calibration_sha256": CALIBRATION_SHA,
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "planned_variants": len(cases),
        "planned_genuine_groups": len({case["group"] for case in cases}),
        "planned_by_condition": PLANNED,
        "sample_rate": gate.RATE,
        "samples_per_window": gate.SAMPLES,
        "maximum_classifier_forwards": 2 * len(cases),
        "vad_forwards_permitted": False,
        "holdout_scored": False,
        "serving_change_permitted": False,
    }


def preflight():
    return {**identity(inputs()), "status": "ready_for_freeze", "model_calls": 0}


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve all earlier attempts")
    cases = inputs()
    locked = identity(cases)
    TRIAL.mkdir(mode=0o700)
    path = TRIAL / "private-plan.json"
    write(path, plan(cases))
    locked["private_plan_sha256"] = sha(path)
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return {"status": "frozen", "freeze_sha256": sha(PUBLIC), "model_calls": 0}


def verify():
    cases = inputs()
    path = TRIAL / "private-plan.json"
    locked = {**identity(cases), "private_plan_sha256": sha(path)}
    require(
        locked == read(PUBLIC) == read(TRIAL / "freeze.json") and plan(cases) == read(path),
        "Exact prospective rescue-scoring freeze required",
    )
    return cases


def score(pcm, model, calibration, deadline):
    require(len(pcm) == gate.SAMPLES, "One exact whole-window waveform required")
    # The saved v256 eligibility replaces only the admission step of score_voice.
    (raw,) = model.predict([pcm])
    require(calibration.valid(raw), "Invalid raw voice response")
    screening = calibration.evaluate(float(raw), pcm, [(0, pcm)], model, deadline)
    require(screening["scope_supported"], "Original serving calibration must cover the waveform extent")
    return {
        "raw": float(raw),
        "decision": screening["decision"],
        "reference_estimate_ai": screening["probability_ai"],
        "stability_probe": screening["probe_status"],
        "stability_log_odds_change": screening["stability_log_odds_change"],
    }


def summarize(cases, rows):
    require(len(rows) == len(cases), "Every frozen rescue requires one completed score")
    for case, row in zip(cases, rows, strict=True):
        probe = row["stability_probe"]
        delta = row["stability_log_odds_change"]
        require(
            row["case"] == case["case"]
            and row["decision"] in DIRECTIONS
            and probe in {"computed", "unavailable", "not_required_for_decision"}
            and row["classifier_forwards"] in {1, 2}
            and (probe != "computed" or row["classifier_forwards"] == 2)
            and (probe != "not_required_for_decision" or row["classifier_forwards"] == 1)
            and ((delta is not None) == (probe == "computed"))
            and (delta is None or (math.isfinite(delta) and delta >= 0))
            and all(math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in ("raw", "reference_estimate_ai")),
            "Invalid or incomplete classifier receipt",
        )

    def counts(group):
        return {direction: sum(row["decision"] == direction for row in group) for direction in DIRECTIONS}

    return {
        "assessed_variants": len(rows),
        "genuine_groups": len({case["group"] for case in cases}),
        "directions": counts(rows),
        "directions_by_condition": {
            condition: counts(
                [row for case, row in zip(cases, rows, strict=True) if case["condition"] == condition]
            )
            for condition in PLANNED
        },
        "groups_with_false_ai": len(
            {case["group"] for case, row in zip(cases, rows, strict=True) if row["decision"] == "ai"}
        ),
        "stability_probe_counts": dict(Counter(row["stability_probe"] for row in rows)),
        "new_neural_calls": sum(row["classifier_forwards"] for row in rows),
        "vad_forwards": 0,
        "second_opinion_scored": False,
        "holdout_scored": False,
        "serving_changed": False,
    }


def result(cases, journal):
    return {
        "schema": "chipmunk.audio-rescued-speech-shadow-result/258",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "gate_result_sha256": PRIOR_SHA,
        "private_observation_journal_sha256": sha(journal),
        **summarize(cases, read(journal)),
    }


def worker():
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and not (STAGE / "profile.json").exists()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact guarded offline worker required",
    )
    cases = verify()

    def offline(event, _args):
        if event == "socket.connect":
            raise RuntimeError("Network access forbidden in the scoring worker")

    sys.addaudithook(offline)
    import torch

    from app.detectors.audio import calibration
    from app.detectors.audio.decode import read_wav
    from app.engine.boundary import Deadline
    from scripts.run_audio_specialists_v15 import load_voice

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pilot = gate.prior.baseline.serving.prior.pilot
    model = load_voice({"weights": str(pilot.WEIGHTS), "assets": pilot.ASSETS})
    calls = 0

    def count(_model, _args):
        nonlocal calls
        calls += 1

    hook = model.register_forward_pre_hook(count)
    rows = []
    try:
        for case in cases:
            pcm = read_wav(gate.prior.baseline_waveform(case))
            before = calls
            outcome = score(pcm, model, calibration, Deadline(120))
            rows.append({"case": case["case"], **outcome, "classifier_forwards": calls - before})
    finally:
        hook.remove()
    OUTPUT.mkdir(mode=0o700)
    journal = OUTPUT / "private-observations.json"
    write(journal, rows)
    write(OUTPUT / "result.json", result(cases, journal))


run = scoped(
    gate.run, {"TRIAL": TRIAL, "OUTPUT": OUTPUT, "verify": verify, "MODULE": MODULE, "__file__": __file__}
)


def readout():
    cases = verify()
    profile_path, result_path = STAGE / "profile.json", OUTPUT / "result.json"
    profile, saved = read(profile_path), read(result_path)
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "authored"
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["resource_journal_sha256"] == sha(STAGE / "resources.jsonl")
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["new_neural_calls"] == saved["new_neural_calls"]
        and saved == result(cases, OUTPUT / "private-observations.json"),
        "Exact completed scoring receipt required",
    )
    return {
        **saved,
        "resource_profile_sha256": sha(profile_path),
        "worker_duration_seconds": profile["duration_s"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"preflight": preflight, "freeze": freeze, "run": run, "worker": worker, "readout": readout}
    parser.add_argument("command", choices=commands)
    output = commands[parser.parse_args().command]()
    if output is not None:
        print(json.dumps(output, indent=2, allow_nan=False))
