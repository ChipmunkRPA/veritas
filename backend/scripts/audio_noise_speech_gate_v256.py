"""Measure a fixed partial-speech rescue on verified local audio inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from scripts import audio_matched_background_score_v250 as prior
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = prior.ROOT, prior.BASE, prior.PANEL
MODULE = "scripts.audio_noise_speech_gate_v256"
TRIAL = BASE / "audio-noise-speech-gate-v256"
STAGE = TRIAL / "authored"
OUTPUT = STAGE / "output"
PUBLIC = PANEL / "audio-noise-speech-gate-freeze-v256.json"
PRIOR_RESULT = PANEL / "audio-matched-background-score-complete-v250.json"
PRIOR_SHA = "738fb00f42781ef2afe45af72b8ab61747e7de7431000dfc6f72b78becba58e8"
SCREEN_FREEZE_SHA = "0a36ae61d0bd62e9612261f3177a85ffad8c2190ed1066ff90d78cabc07f28c5"
VAD = ROOT / "models/weights/audio-upgrade/silero_vad.jit"
VAD_SHA = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
RATE, SAMPLES, FRAME = 16_000, 48_000, 512
CODE = (
    "backend/scripts/audio_noise_speech_gate_v256.py",
    "backend/tests/test_audio_noise_speech_gate_v256.py",
    "docs/audio-noise-speech-gate-v256-method.md",
    "backend/app/detectors/audio/content.py",
    "backend/app/detectors/audio/upgraded.py",
    "backend/app/detectors/video/adapters/audio.py",
    "backend/app/detectors/video/audio.py",
    "backend/app/engine/profiles.py",
    "backend/scripts/audio_matched_background_score_v250.py",
    "backend/scripts/audio_commons_slice_screen_v247.py",
    "backend/scripts/video_appearance_runtime_v108.py",
)
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require


def matched_inputs():
    require(sha(PRIOR_RESULT) == PRIOR_SHA, "Exact published matched comparison required")
    result = read(PRIOR_RESULT)
    frozen, cases, _, _ = prior.verify()
    batches = [prior.completed_baseline(i, cases) for i in range(len(prior.CONDITIONS))]
    saved = {row["case"]: row for rows, _, _ in batches for row in rows}
    require(
        result["status"] == "complete"
        and result["freeze_sha256"] == sha(prior.PUBLIC)
        and result["baseline_profile_sha256"] == [digest for _, digest, _ in batches]
        and result["baseline_classifier_calls"] == sum(count for _, _, count in batches)
        and len(saved) == len(cases) == frozen["planned_cases"],
        "Exact completed serving receipts required",
    )
    return cases, {key: row["route_status"] for key, row in saved.items()}


def excerpt(body, start):
    window = body[2 * start : 2 * (start + SAMPLES)]
    require(len(window) == 2 * SAMPLES, "Exact three-second background PCM required")
    return window


def complement(screened, pcms):
    """Take every complete same-grid window excluded from the earlier screen by position."""
    covered = {(row["slot"], row["start_sample"]) for row in screened}
    screened_digests = {row["pcm_sha256"] for row in screened}
    environments = {row["slot"]: row["environment"] for row in screened}
    require(len(covered) == len(screened) and set(environments) == set(pcms), "Incomplete background grid")
    rows = []
    for slot, body in sorted(pcms.items()):
        for start in range(0, len(body) // 2 - SAMPLES + 1, SAMPLES):
            if (slot, start) in covered:
                continue
            digest = hashlib.sha256(excerpt(body, start)).hexdigest()
            if digest not in screened_digests:
                rows.append(
                    {"slot": slot, "start_sample": start, "environment": environments[slot], "sha256": digest}
                )
    require(rows, "No unscreened background controls remain")
    return rows


def inputs():
    cases, recorded = matched_inputs()
    # The background verifier imports app settings through its decoder. Keep
    # that import after the model- and settings-free v250 receipt checks.
    from scripts import audio_commons_slice_screen_v247 as screen

    require(sha(screen.PUBLIC) == SCREEN_FREEZE_SHA, "Exact previous background grid required")
    screened, pcms = screen.verify()
    controls = complement(screened, pcms)
    require(len(pcms) == 7, "Exact qualified background recording roster required")
    return cases, recorded, controls, pcms


def private_plan(cases, recorded, controls):
    return {
        "matched": [
            {
                "case": row["case"],
                "pcm_sha256": row[prior.NATIVE_HASH],
                "recorded_route": recorded[row["case"]],
            }
            for row in cases
        ],
        "unscreened_background": controls,
    }


def identity(cases, controls):
    require(
        sha(VAD) == VAD_SHA == prior.baseline.serving.prior.pilot.ASSETS[VAD.name],
        "Exact production speech activity checkpoint required",
    )
    return {
        "schema": "chipmunk.audio-noise-speech-gate-freeze/256",
        "matched_result_sha256": PRIOR_SHA,
        "prior_background_grid_freeze_sha256": SCREEN_FREEZE_SHA,
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "vad_sha256": VAD_SHA,
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")},
        "matched_planned": len(cases),
        "matched_prepared": sum(row["status"] == "prepared" for row in cases),
        "unscreened_controls_by_environment": dict(Counter(row["environment"] for row in controls)),
        "unscreened_recordings_by_environment": {
            name: len({row["slot"] for row in controls if row["environment"] == name})
            for name in ("kitchen", "traffic")
        },
        "sample_rate": RATE,
        "samples_per_window": SAMPLES,
        "vad_frame_samples": FRAME,
        "rescue_single_span_minimum_samples": RATE,
        "authenticity_detector_calls_permitted": False,
        "serving_change_permitted": False,
    }


def preflight():
    cases, _, controls, _ = inputs()
    return {**identity(cases, controls), "status": "ready_for_freeze", "model_calls": 0}


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every previous attempt")
    cases, recorded, controls, _ = inputs()
    frozen = identity(cases, controls)
    TRIAL.mkdir(mode=0o700)
    path = TRIAL / "private-plan.json"
    write(path, private_plan(cases, recorded, controls))
    frozen["private_plan_sha256"] = sha(path)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    return {"status": "frozen", "freeze_sha256": sha(PUBLIC), "model_calls": 0}


def verify():
    cases, recorded, controls, pcms = inputs()
    path = TRIAL / "private-plan.json"
    frozen = {**identity(cases, controls), "private_plan_sha256": sha(path)}
    require(
        frozen == read(PUBLIC) == read(TRIAL / "freeze.json")
        and read(path) == private_plan(cases, recorded, controls),
        "Exact prospective diagnostic freeze required",
    )
    return frozen, cases, recorded, controls, pcms


def coverage(spans):
    end = total = longest = 0
    for start, stop in spans:
        require(
            type(start) is int and type(stop) is int and end <= start < stop <= SAMPLES, "Invalid VAD span"
        )
        end = stop
        length = stop - start
        total += length
        longest = max(longest, length)
    baseline = total * 2 >= SAMPLES
    return {
        "speech_samples": total,
        "longest_span_samples": longest,
        "baseline": baseline,
        "rescue": baseline or longest >= RATE,
    }


def counts(rows):
    return {
        "planned": len(rows),
        "prepared": sum(row is not None for row in rows),
        "baseline_admitted": sum(row["baseline"] for row in rows if row is not None),
        "candidate_admitted": sum(row["rescue"] for row in rows if row is not None),
        "additional_admissions": sum(
            row["rescue"] and not row["baseline"] for row in rows if row is not None
        ),
    }


def summarize(cases, recorded, controls, observed):
    require(set(observed) == {"matched", "unscreened_background"}, "Unexpected observation fields")
    require(
        len(observed["matched"]) == len(cases) and len(observed["unscreened_background"]) == len(controls),
        "Incomplete observation roster",
    )
    grouped, genuine, backgrounds = defaultdict(list), defaultdict(dict), defaultdict(list)
    for row, saved in zip(cases, observed["matched"], strict=True):
        require(saved["case"] == row["case"], "Observation order differs from exact matched roster")
        spans = saved["spans"]
        require((spans is None) == (row["status"] == "unavailable"), "Unexpected waveform availability")
        outcome = coverage(spans) if spans is not None else None
        status = (
            "preparation_unavailable"
            if outcome is None
            else "ok"
            if outcome["baseline"]
            else "not_applicable"
        )
        require(
            status == recorded[row["case"]], "Replayed baseline speech admission differs from the saved run"
        )
        grouped[row["arm"], row["condition"]].append(outcome)
        if row["arm"] == "genuine":
            genuine[row["group"]][row["condition"]] = outcome
    for row, saved in zip(controls, observed["unscreened_background"], strict=True):
        require(saved["control"] == row, "Observation order differs from exact unscreened roster")
        backgrounds[row["environment"]].append((row["slot"], coverage(saved["spans"])))
    genuine_clean = [
        row
        for row in genuine.values()
        if row["matched_clean"] is not None and row["matched_clean"]["baseline"]
    ]
    transitions = {}
    for policy in ("baseline", "rescue"):
        lost = [
            sum(not outcome[policy] for condition, outcome in row.items() if condition != "matched_clean")
            for row in genuine_clean
        ]
        transitions[policy] = {
            "clean_baseline_admitted_groups": len(genuine_clean),
            "groups_with_any_noisy_exclusion": sum(value > 0 for value in lost),
            "groups_with_all_four_noisy_excluded": lost.count(4),
            "noisy_variants_excluded": sum(lost),
        }
    return {
        "by_arm_and_condition": {
            arm: {condition: counts(grouped[arm, condition]) for condition in prior.CONDITIONS}
            for arm in prior.ARMS
        },
        "clean_admitted_genuine_to_noise": transitions,
        "unscreened_background_by_environment": {
            name: {
                **counts([value for _, value in rows]),
                "recordings": len({slot for slot, _ in rows}),
                "recordings_with_additional_admissions": len(
                    {slot for slot, value in rows if value["rescue"] and not value["baseline"]}
                ),
            }
            for name, rows in sorted(backgrounds.items())
        },
        "vad_frame_forwards": (sum(row["status"] == "prepared" for row in cases) + len(controls))
        * math.ceil(SAMPLES / FRAME),
        "authenticity_detector_calls": 0,
        "human_background_listening_reviewed": False,
        "serving_changed": False,
    }


class CountedVAD:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.model(*args)

    def reset_states(self):
        return self.model.reset_states()


def worker():
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and not (STAGE / "profile.json").exists()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline guarded worker required",
    )
    _, cases, recorded, controls, pcms = verify()

    def offline(event, _args):
        if event == "socket.connect":
            raise RuntimeError("Network access forbidden in the diagnostic worker")

    sys.addaudithook(offline)
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    activity = SpeechActivity(VAD)
    counter = CountedVAD(activity.model)
    activity.model = counter

    def measure(pcm):
        before = counter.calls
        seconds = activity.spans(pcm)
        spans = [[round(a * RATE), round(b * RATE)] for a, b in seconds]
        require(
            counter.calls - before == math.ceil(SAMPLES / FRAME), "Unexpected speech-activity forward count"
        )

        def leased(model_id):
            require(model_id == "silero.speech_activity", "Only the measured speech activity may be leased")
            return SimpleNamespace(obj=SimpleNamespace(spans=lambda _: seconds))

        context = SimpleNamespace(
            profile="full",
            lease=SimpleNamespace(get=leased),
            outputs=SimpleNamespace(audio=AudioTrack("analysed", pcm, len(pcm))),
        )
        windows = PellaVoiceDetector()._windows(context)
        require(
            bool(windows) == coverage(spans)["baseline"]
            and (not windows or (len(windows) == 1 and windows[0][0] == 0 and len(windows[0][1]) == SAMPLES)),
            "Diagnostic coverage must match the exact serving window builder",
        )
        return spans

    observed = {"matched": [], "unscreened_background": []}
    for row in cases:
        pcm = prior.candidate_waveform(row) if row["status"] == "prepared" else None
        observed["matched"].append({"case": row["case"], "spans": None if pcm is None else measure(pcm)})
    for row in controls:
        body = excerpt(pcms[row["slot"]], row["start_sample"])
        observed["unscreened_background"].append({"control": row, "spans": measure(prior.float_input(body))})

    totals = summarize(cases, recorded, controls, observed)
    require(counter.calls == totals["vad_frame_forwards"], "Measured speech-activity call count differs")
    OUTPUT.mkdir(mode=0o700)
    journal = OUTPUT / "private-observations.json"
    write(journal, observed)
    write(OUTPUT / "result.json", result(totals, journal))


def result(totals, journal):
    return {
        "schema": "chipmunk.audio-noise-speech-gate-result/256",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "matched_result_sha256": PRIOR_SHA,
        "private_observation_journal_sha256": sha(journal),
        "new_neural_calls": totals["vad_frame_forwards"],
        **totals,
    }


def run():
    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(OUTPUT)],
            "Only the exact guarded speech diagnostic may start",
        )
        return subprocess.Popen([*argv[:3], MODULE, "worker"], **kwargs)

    proxy = SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda *_: True,
            "subprocess": proxy,
            "__file__": __file__,
        },
    )("authored")


def readout():
    _, cases, recorded, controls, _ = verify()
    profile_path, result_path = STAGE / "profile.json", OUTPUT / "result.json"
    profile, saved = read(profile_path), read(result_path)
    journal = OUTPUT / "private-observations.json"
    reconstructed = result(summarize(cases, recorded, controls, read(journal)), journal)
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
        and saved == reconstructed,
        "Exact successful diagnostic receipt required",
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
