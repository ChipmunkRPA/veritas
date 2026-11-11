"""Prepare provisional background slices for the independent audio comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import audio_commons_background_v234 as background
from scripts import audio_drifting_turkish_v215 as guard
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = guard.ROOT, guard.BASE, guard.PUBLIC.parent
TRIAL = BASE / "audio-commons-slice-screen-v247"
STAGE = TRIAL / "authored"
OUTPUT = STAGE / "output"
PUBLIC = PANEL / "audio-commons-slice-screen-freeze-v247.json"
INPUT = PANEL / "audio-commons-background-complete-v234.json"
INPUT_SHA256 = "3d5a702131268cf65217a163404ddc6095eefb6b199d1ed5ea97487df0a98149"
MODULE = "scripts.audio_commons_slice_screen_v247"
WEIGHTS = ROOT / "models/weights/audio-upgrade"
ASSETS = {
    "silero_vad.jit": "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720",
    "panns-Cnn10.pth": "5240bdc47444e331bbeb54fd741d2b3f933a4526e84b0c17bf3593df94b13962",
    "class_labels_indices.csv": "cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429",
}
CODE = (
    "backend/scripts/audio_commons_slice_screen_v247.py",
    "backend/tests/test_audio_commons_slice_screen_v247.py",
    "backend/scripts/audio_commons_background_v234.py",
    "backend/scripts/audio_drifting_turkish_v215.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/app/detectors/audio/upgraded.py",
    "backend/app/third_party/panns/cnn10.py",
    "docs/audio-commons-slice-screen-v247-method.md",
)
SAMPLES = 48_000
MAX_PER_RECORDING = 12
TAG_POINT = 0.30
VAD_FRAME = 512
SPEECH_TAGS = (
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Babbling",
    "Speech synthesizer",
    "Whispering",
)
MUSIC_TAGS = ("Music", "Musical instrument", "Singing", "Choir", "Rapping", "Humming")
read, write, sha, require = guard.read, guard.write, guard.sha, guard.require


def starts(sample_count):
    windows = sample_count // SAMPLES
    if windows <= MAX_PER_RECORDING:
        return [n * SAMPLES for n in range(windows)]
    return [SAMPLES * (n * (windows - 1) // (MAX_PER_RECORDING - 1)) for n in range(MAX_PER_RECORDING)]


def read_pcm(path, sample_count):
    with wave.open(str(path), "rb") as reader:
        require(
            reader.getnchannels() == 1
            and reader.getsampwidth() == 2
            and reader.getframerate() == 16_000
            and reader.getnframes() == sample_count,
            "Exact prepared background PCM format required",
        )
        pcm = reader.readframes(sample_count)
    require(len(pcm) == sample_count * 2, "Truncated background PCM")
    return pcm


def excerpt(pcm, start):
    payload = pcm[2 * start : 2 * (start + SAMPLES)]
    require(len(payload) == 2 * SAMPLES, "Exact three-second window required")
    return payload


def inputs():
    require(sha(INPUT) == INPUT_SHA256 and read(INPUT) == background.readout(), "Exact v234 result required")
    _, sources = background.verify()
    rows, pcms = [], {}
    for slot, source in enumerate(sources, 1):
        receipt_path = background.TRIAL / f"qualification-{slot:02d}.json"
        receipt = read(receipt_path)
        require(receipt["status"] == "verified_pcm", "Source PCM unavailable")
        _, path = background.asset_paths(slot, source)
        pcm = read_pcm(path, receipt["samples"])
        require(hashlib.sha256(pcm).hexdigest() == receipt["pcm_sha256"], "PCM bytes differ from v234")
        pcms[slot] = pcm
        for start in starts(receipt["samples"]):
            rows.append(
                {
                    "environment": source.environment,
                    "slot": slot,
                    "start_sample": start,
                    "samples": SAMPLES,
                    "pcm_sha256": hashlib.sha256(excerpt(pcm, start)).hexdigest(),
                    "qualification_receipt_sha256": sha(receipt_path),
                }
            )
    require(len(pcms) == 7 and rows, "Exact seven recording roster required")
    return rows, pcms


def identity(rows):
    require(
        all(sha(WEIGHTS / name) == digest for name, digest in ASSETS.items()), "Pinned screen assets differ"
    )
    return {
        "schema": "chipmunk.audio-commons-slice-screen-freeze/247",
        "background_result_sha256": INPUT_SHA256,
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "asset_sha256": ASSETS,
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")},
        "planned_by_environment": dict(Counter(row["environment"] for row in rows)),
        "maximum_windows_per_recording": MAX_PER_RECORDING,
        "samples_per_window": SAMPLES,
        "sound_tag_screen_point": TAG_POINT,
        "speech_tags": list(SPEECH_TAGS),
        "music_tags": list(MUSIC_TAGS),
        "human_listening_reviewed": False,
        "authenticity_detector_calls_permitted": False,
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve previous attempts")
    rows, _ = inputs()
    lock = identity(rows)
    TRIAL.mkdir(mode=0o700)
    path = TRIAL / "private-plan.json"
    write(path, rows)
    lock["private_plan_sha256"] = sha(path)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    return lock


def verify():
    rows, pcms = inputs()
    path = TRIAL / "private-plan.json"
    lock = {**identity(rows), "private_plan_sha256": sha(path)}
    require(
        read(path) == rows and read(TRIAL / "freeze.json") == read(PUBLIC) == lock,
        "Exact slice freeze required",
    )
    return rows, pcms


def screen_status(voice_spans, speech_tag, music_tag, constant):
    require(
        type(voice_spans) is int
        and voice_spans >= 0
        and type(constant) is bool
        and all(math.isfinite(score) and 0 <= score <= 1 for score in (speech_tag, music_tag)),
        "Invalid machine screen evidence",
    )
    if constant:
        return "constant_audio"
    if voice_spans or speech_tag >= TAG_POINT:
        return "voice_signal_detected"
    if music_tag >= TAG_POINT:
        return "music_signal_detected"
    return "machine_screened_provisional"


def slice_packet(rows, observations):
    require(len(rows) == len(observations), "Every fixed slice requires evidence")
    admitted = []
    for row, observed in zip(rows, observations, strict=True):
        require(
            all(observed[key] == value for key, value in row.items()), "Observation input identity differs"
        )
        status = screen_status(
            observed["voice_spans"], observed["speech_tag"], observed["music_tag"], observed["constant"]
        )
        require(status == observed["status"], "Saved screening direction differs")
        if status == "machine_screened_provisional":
            admitted.append(row)
    return {
        "schema": "chipmunk.audio-commons-slice-evidence/247",
        "background_result_sha256": INPUT_SHA256,
        "evidence_type": "machine_screened_provisional",
        "human_listening_reviewed": False,
        "excerpts": admitted,
    }


def summarize(rows, observed):
    packet = slice_packet(rows, observed)
    counts, groups = {}, {}
    for environment in ("kitchen", "traffic"):
        counts[environment] = dict(
            Counter(row["status"] for row in observed if row["environment"] == environment)
        )
        groups[environment] = len(
            {row["slot"] for row in packet["excerpts"] if row["environment"] == environment}
        )
    return {
        "window_outcomes_by_environment": counts,
        "provisional_recordings_by_environment": groups,
        "candidate_windows": len(rows),
        "provisional_windows": len(packet["excerpts"]),
        "sound_tag_forwards": len(rows),
        "voice_frame_forwards": len(rows) * math.ceil(SAMPLES / VAD_FRAME),
        "new_neural_calls": len(rows) * (1 + math.ceil(SAMPLES / VAD_FRAME)),
        "authenticity_detector_calls": 0,
    }


def worker():
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline local guard required",
    )
    rows, pcms = verify()

    def offline(event, _args):
        if event == "socket.connect":
            raise RuntimeError("Network access forbidden in the screening worker")

    sys.addaudithook(offline)
    import torch
    from scipy.signal import resample_poly

    from app.detectors.audio.upgraded import SoundEventsModel, SpeechActivity

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    activity = SpeechActivity(WEIGHTS / "silero_vad.jit")
    tags = SoundEventsModel(WEIGHTS / "panns-Cnn10.pth", WEIGHTS / "class_labels_indices.csv", "cpu")
    speech = [tags.labels.index(label) for label in SPEECH_TAGS]
    music = [tags.labels.index(label) for label in MUSIC_TAGS]
    observed = []
    with torch.inference_mode():
        for row in rows:
            pcm = np.frombuffer(excerpt(pcms[row["slot"]], row["start_sample"]), dtype="<i2")
            audio = pcm.astype(np.float32) / 32768.0
            voice = len(activity.spans(audio))
            values = resample_poly(audio, 2, 1).astype(np.float32)
            scores = tags.model(torch.from_numpy(values)[None])["clipwise_output"][0].numpy()
            require(scores.shape == (527,) and np.isfinite(scores).all(), "Invalid sound-tag scores")
            signal, song = float(scores[speech].max()), float(scores[music].max())
            constant = bool(np.all(pcm == pcm[0]))
            observed.append(
                {
                    **row,
                    "voice_spans": voice,
                    "speech_tag": signal,
                    "music_tag": song,
                    "constant": constant,
                    "status": screen_status(voice, signal, song, constant),
                }
            )
    OUTPUT.mkdir(mode=0o700)
    readings, packet = OUTPUT / "private-screen.json", OUTPUT / "private-admitted-slices.json"
    write(readings, observed)
    write(packet, slice_packet(rows, observed))
    write(
        OUTPUT / "result.json",
        {
            "schema": "chipmunk.audio-commons-slice-screen-result/247",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "background_result_sha256": INPUT_SHA256,
            "human_listening_reviewed": False,
            "private_screen_sha256": sha(readings),
            "private_admitted_slices_sha256": sha(packet),
            **summarize(rows, observed),
        },
    )


run = scoped(
    guard.run, {"TRIAL": TRIAL, "OUTPUT": OUTPUT, "verify": verify, "MODULE": MODULE, "__file__": __file__}
)


def readout():
    rows, _ = verify()
    profile_path, result_path = STAGE / "profile.json", OUTPUT / "result.json"
    profile, result = read(profile_path), read(result_path)
    readings, packet = OUTPUT / "private-screen.json", OUTPUT / "private-admitted-slices.json"
    observed = read(readings)
    require(
        profile["child_started"]
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["resource_journal_sha256"] == sha(STAGE / "resources.jsonl")
        and profile["worker_receipt_sha256"] == sha(result_path)
        and result["schema"] == "chipmunk.audio-commons-slice-screen-result/247"
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["background_result_sha256"] == INPUT_SHA256
        and result["human_listening_reviewed"] is False
        and result["private_screen_sha256"] == sha(readings)
        and result["private_admitted_slices_sha256"] == sha(packet)
        and read(packet) == slice_packet(rows, observed)
        and all(result[key] == value for key, value in summarize(rows, observed).items())
        and result["new_neural_calls"] == profile["new_neural_calls"],
        "Exact completed machine screen required",
    )
    return {
        **result,
        "resource_profile_sha256": sha(profile_path),
        "worker_duration_seconds": profile["duration_s"],
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"freeze": freeze, "run": run, "worker": worker, "readout": readout}
    parser.add_argument("command", choices=commands)
    result = commands[parser.parse_args().command]()
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))
