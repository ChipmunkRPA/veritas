"""Challenge the fixed speech admission fallback on stronger matched noise."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from types import SimpleNamespace

from scripts import audio_noise_speech_gate_transfer_v267 as prior
from scripts import audio_short_edit_controls as controls
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = prior.ROOT, prior.BASE, prior.PANEL
MODULE = "scripts.audio_noise_speech_gate_strong_noise_v269"
TRIAL = BASE / "audio-noise-speech-gate-strong-noise-v269"
STAGE, OUTPUT = TRIAL / "authored", TRIAL / "authored/output"
PUBLIC = PANEL / "audio-noise-speech-gate-strong-noise-freeze-v269.json"
TRANSFER = PANEL / "audio-noise-speech-gate-transfer-complete-v267.json"
TRANSFER_SHA = "180578ce4f12181c42550d2fe612cbb583d0815094569967ca9500efd240afdd"
TRANSFER_FREEZE_SHA = "b59306f1c86729306d2f122eab60dd4a8d540211a95c3d31d49825d985d14d17"
CODE = (
    "backend/scripts/audio_noise_speech_gate_strong_noise_v269.py",
    "backend/tests/test_audio_noise_speech_gate_strong_noise_v269.py",
    "docs/audio-noise-speech-gate-strong-noise-v269-method.md",
)
LEVELS = (-5, -10)
CONDITIONS = ("matched_clean", *(f"{name}_{level}db" for name in prior.TAPES for level in LEVELS))
read, sha, require = prior.read, prior.sha, prior.require
shared = prior.sources.shared
mix_paired_noise = scoped(controls.mix_paired_noise, {"SNR_LEVELS_DB": LEVELS})
mix_shared_noise = scoped(
    shared.mix_shared_noise, {"SNR_LEVELS_DB": LEVELS, "mix_paired_noise": mix_paired_noise}
)
stronger_cases = scoped(
    prior.sources.cases,
    {
        "offsets": prior.offsets,
        "CONDITIONS": CONDITIONS,
        "shared": SimpleNamespace(
            ARMS=shared.ARMS,
            TAPES=shared.TAPES,
            SAMPLES=shared.SAMPLES,
            UnsupportedControl=shared.UnsupportedControl,
            mix_shared_noise=mix_shared_noise,
        ),
    },
)


def bind_cases(previous, prepared):
    old = {(row["triplet"], row["arm"], row["condition"]): row for row in previous}
    expected = [
        (row["triplet"], row["arm"], condition)
        for row in previous
        if row["condition"] == "matched_clean"
        for condition in CONDITIONS
    ]
    cases = []
    for (triplet, arm, condition), (row, payload) in zip(expected, prepared, strict=True):
        require(
            (row["triplet"], row["arm"], row["condition"]) == (triplet, arm, condition)
            and row["status"] in {"prepared", "unavailable"}
            and ((payload is None) == (row["status"] == "unavailable")),
            "The strong-noise plan must preserve the exact v267 roster and order",
        )
        former_condition = condition if condition == "matched_clean" else condition.rsplit("_", 1)[0] + "_0db"
        former = old[triplet, arm, former_condition]
        if former["status"] == "unavailable":
            require(
                payload is None and row["reason"] == former["reason"],
                "Former unavailability must be preserved",
            )
        elif payload is None:
            require(
                row["reason"].startswith("mix_unavailable:"), "Only new mixing failures may alter coverage"
            )
        else:
            require(
                all(
                    row[key] == former[key]
                    for key in (
                        "source_wav_sha256",
                        "source_prefix_sha256",
                        "noise_sample_offset",
                        "noise_excerpt_sha256",
                    )
                ),
                "Source bytes or tape excerpt changed from the verified transfer",
            )
        cases.append({**row, "payload": payload})
    return cases


def inputs():
    require(
        sha(TRANSFER) == TRANSFER_SHA and sha(prior.PUBLIC) == TRANSFER_FREEZE_SHA,
        "Exact published transfer development history required",
    )
    old_inputs = prior.verify()
    old, previous, groups, historical_controls, pcms = old_inputs
    previous_result = scoped(prior.readout, {"verify": lambda: old_inputs})()
    require(
        previous_result == read(TRANSFER)
        and previous_result["status"] == "complete"
        and len(historical_controls) == 102
        and previous_result["vad_frame_forwards"] == 27918,
        "Completed v267 transfer and separate background observations required",
    )
    source_rosters = prior.sources.verify()
    cases = bind_cases(previous, stronger_cases(source_rosters, pcms, prior.sources.source_readers()))
    require(
        len(cases) == old["matched_planned"], "All planned v267 inputs require a strong-noise disposition"
    )
    prior.sources.model_free()
    return old, cases, groups, [], pcms


def identity(cases, groups, background, old):
    require(not background, "Previously scored backgrounds are retained as history only")
    return {
        "schema": "chipmunk.audio-noise-speech-gate-strong-noise-freeze/269",
        "development_result_sha256": TRANSFER_SHA,
        "development_freeze_sha256": TRANSFER_FREEZE_SHA,
        "inherited_code_sha256": {**old["inherited_v256_code_sha256"], **old["code_sha256"]},
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "vad_sha256": old["vad_sha256"],
        "runtime_versions": old["runtime_versions"],
        "foreground_planned_by_arm": old["foreground_planned_by_arm"],
        "prepared_foreground_groups_by_arm": {
            arm: len(
                {
                    groups[arm, row["triplet"]]
                    for row in cases
                    if row["arm"] == arm
                    and row["condition"] == "matched_clean"
                    and row["payload"] is not None
                }
            )
            for arm in prior.ARMS
        },
        "matched_planned": len(cases),
        "matched_prepared": sum(row["payload"] is not None for row in cases),
        "unavailable_reasons": dict(Counter(row["reason"] for row in cases if row["payload"] is None)),
        "nominal_snr_db": list(LEVELS),
        "conditions": list(CONDITIONS),
        "mixture_offset_rule": old["mixture_offset_rule"],
        "development_background_controls_by_environment": old["background_controls_by_environment"],
        "background_controls_rerun": False,
        "speech_free_background_verified": False,
        "rescue_single_span_minimum_samples": old["rescue_single_span_minimum_samples"],
        "authenticity_detector_calls_permitted": False,
        "serving_change_permitted": False,
    }


comparison = scoped(prior.summarize, {"CONDITIONS": CONDITIONS})


def summarize(cases, groups, background, observed):
    require(not background, "Previously scored backgrounds must not enter the new run")
    return {
        **comparison(cases, groups, background, observed),
        "unavailable_reasons": dict(Counter(row["reason"] for row in cases if row["payload"] is None)),
        "background_controls_rerun": False,
    }


def result(totals, journal):
    return {
        "schema": "chipmunk.audio-noise-speech-gate-strong-noise-result/269",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "development_result_sha256": TRANSFER_SHA,
        "private_observation_journal_sha256": sha(journal),
        "new_neural_calls": totals["vad_frame_forwards"],
        **totals,
    }


FREEZE_BINDINGS = {"TRIAL": TRIAL, "PUBLIC": PUBLIC, "inputs": inputs, "identity": identity}
preflight = scoped(prior.preflight, FREEZE_BINDINGS)
freeze = scoped(prior.freeze, FREEZE_BINDINGS)
verify = scoped(prior.verify, FREEZE_BINDINGS)
WORKER_BINDINGS = {
    "verify": verify,
    "STAGE": STAGE,
    "OUTPUT": OUTPUT,
    "summarize": summarize,
    "result": result,
}
worker = scoped(prior.worker, WORKER_BINDINGS)
run = scoped(
    prior.run, {"TRIAL": TRIAL, "OUTPUT": OUTPUT, "verify": verify, "MODULE": MODULE, "__file__": __file__}
)
readout = scoped(prior.readout, {**WORKER_BINDINGS, "__file__": __file__})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"preflight": preflight, "freeze": freeze, "run": run, "worker": worker, "readout": readout}
    parser.add_argument("command", choices=commands)
    output = commands[parser.parse_args().command]()
    if output is not None:
        print(json.dumps(output, indent=2, allow_nan=False))
