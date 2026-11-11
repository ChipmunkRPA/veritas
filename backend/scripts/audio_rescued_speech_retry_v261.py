"""Retry the exact v258 shadow scoring input with one CPU thread initializer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from scripts import audio_rescued_speech_shadow_v258 as prior
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL, gate = prior.ROOT, prior.BASE, prior.PANEL, prior.gate
MODULE = "scripts.audio_rescued_speech_retry_v261"
TRIAL = BASE / "audio-rescued-speech-retry-v261"
STAGE, OUTPUT = TRIAL / "authored", TRIAL / "authored/output"
PUBLIC = PANEL / "audio-rescued-speech-retry-freeze-v261.json"
FAILED_FREEZE_SHA = "121b0de54f6eae4f37810a5e7d9bf2b880ae01831fe1e85355b3fe943e2a8d2c"
FAILED_PROFILE_SHA = "0e06d0f28f6cc99b862e85604b4cbcbc9c0c917827b48f60a5015c0848cdc293"
FAILED_LOG_SHA = "617f0477570be8883434fbc9959ed0977e97556d1b184fa04c6e7f366d2e8dd0"
CODE = (
    "backend/scripts/audio_rescued_speech_retry_v261.py",
    "backend/tests/test_audio_rescued_speech_retry_v261.py",
    "docs/audio-rescued-speech-retry-v261-method.md",
)
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require


def inputs():
    profile_path = prior.STAGE / "profile.json"
    log_path = prior.STAGE / "worker.log"
    require(
        sha(prior.PUBLIC) == FAILED_FREEZE_SHA
        and sha(profile_path) == FAILED_PROFILE_SHA
        and sha(log_path) == FAILED_LOG_SHA,
        "Exact frozen v258 failure required",
    )
    profile = read(profile_path)
    require(
        profile["child_started"] is True
        and profile["child_exit_code"] == 1
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["worker_receipt_sha256"] is None
        and profile["new_neural_calls"] is None
        and profile["runtime_sha256"] == sha(Path(prior.__file__))
        and profile["resource_journal_sha256"] == sha(prior.STAGE / "resources.jsonl")
        and "cannot set number of interop threads" in log_path.read_text()
        and not prior.OUTPUT.exists(),
        "The historical attempt must remain a thread-initialization failure without scores",
    )
    return prior.verify()


def identity():
    cases = inputs()
    frozen = read(prior.PUBLIC)
    return {
        **frozen,
        "schema": "chipmunk.audio-rescued-speech-retry-freeze/261",
        "code_sha256": {**frozen["code_sha256"], **{name: sha(ROOT / name) for name in CODE}},
        "failed_freeze_sha256": FAILED_FREEZE_SHA,
        "failed_profile_sha256": FAILED_PROFILE_SHA,
        "failed_worker_log_sha256": FAILED_LOG_SHA,
        "thread_configuration": "existing_voice_loader_only",
    }, cases


def preflight():
    locked, _ = identity()
    return {**locked, "status": "ready_for_freeze", "model_calls": 0}


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve all earlier attempts")
    locked, cases = identity()
    TRIAL.mkdir(mode=0o700)
    path = TRIAL / "private-plan.json"
    write(path, prior.plan(cases))
    require(sha(path) == locked["private_plan_sha256"], "Exact historical input plan bytes required")
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return {"status": "frozen", "freeze_sha256": sha(PUBLIC), "model_calls": 0}


def verify():
    locked, cases = identity()
    path = TRIAL / "private-plan.json"
    require(
        locked == read(PUBLIC) == read(TRIAL / "freeze.json")
        and sha(path) == locked["private_plan_sha256"]
        and read(path) == prior.plan(cases),
        "Exact prospective retry freeze required",
    )
    return cases


def classify(cases, load_voice, decode, calibration, deadline):
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
            pcm = decode(gate.prior.baseline_waveform(case))
            before = calls
            outcome = prior.score(pcm, model, calibration, deadline(120))
            rows.append({"case": case["case"], **outcome, "classifier_forwards": calls - before})
    finally:
        hook.remove()
    return rows


prior_result = scoped(prior.result, {"PUBLIC": PUBLIC})


def result(cases, journal):
    return {
        **prior_result(cases, journal),
        "schema": "chipmunk.audio-rescued-speech-retry-result/261",
        "failed_profile_sha256": FAILED_PROFILE_SHA,
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
    from app.detectors.audio import calibration
    from app.detectors.audio.decode import read_wav
    from app.engine.boundary import Deadline
    from scripts.run_audio_specialists_v15 import load_voice

    rows = classify(cases, load_voice, read_wav, calibration, Deadline)
    OUTPUT.mkdir(mode=0o700)
    journal = OUTPUT / "private-observations.json"
    write(journal, rows)
    write(OUTPUT / "result.json", result(cases, journal))


run = scoped(
    gate.run, {"TRIAL": TRIAL, "OUTPUT": OUTPUT, "verify": verify, "MODULE": MODULE, "__file__": __file__}
)
readout = scoped(
    prior.readout,
    {"verify": verify, "STAGE": STAGE, "OUTPUT": OUTPUT, "result": result, "__file__": __file__},
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"preflight": preflight, "freeze": freeze, "run": run, "worker": worker, "readout": readout}
    parser.add_argument("command", choices=commands)
    output = commands[parser.parse_args().command]()
    if output is not None:
        print(json.dumps(output, indent=2, allow_nan=False))
