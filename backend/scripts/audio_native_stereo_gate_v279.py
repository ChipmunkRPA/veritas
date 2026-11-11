"""Compare production mono and a fixed acoustic channel choice on development stereo speech."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.detectors.audio.decode import decode_audio
from scripts import audio_native_stereo_pairs_v277 as pairs
from scripts import audio_noise_speech_gate_v256 as gate
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = gate.ROOT, gate.BASE, gate.PANEL
MODULE = "scripts.audio_native_stereo_gate_v279"
TRIAL = BASE / "audio-native-stereo-gate-v279"
STAGE, OUTPUT = TRIAL / "authored", TRIAL / "authored/output"
PUBLIC = PANEL / "audio-native-stereo-gate-freeze-v279.json"
DEV_SHA = "a247832a5e874b9c9eb90e066016db40bd8463bddc7598696cad1e8ed41bd270"
RATE, SAMPLES, FRAME = gate.RATE, gate.SAMPLES, gate.FRAME
VIEWS = ("mono", "ch0", "ch1")
MIN_BLOCKS, MIN_GAP_DB = 8, 3.0
FORWARDS_PER_VIEW = math.ceil(SAMPLES / FRAME)
TIMELINE = [{"start_sample": 0, "end_sample": SAMPLES, "source_pts": 0}]
CODE = (
    "backend/scripts/audio_native_stereo_gate_v279.py",
    "backend/tests/test_audio_native_stereo_gate_v279.py",
    "docs/audio-native-stereo-gate-v279-method.md",
    "backend/scripts/audio_native_stereo_pairs_v277.py",
    "backend/scripts/audio_noise_speech_gate_v256.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/app/detectors/audio/decode.py",
    "backend/app/detectors/audio/upgraded.py",
    "backend/app/detectors/video/adapters/audio.py",
    "backend/app/detectors/video/audio.py",
    "backend/app/engine/profiles.py",
)
read, write, sha, require = gate.read, gate.write, gate.sha, gate.require


def bind_cases(plan, conditions, rows):
    expected = [(index, offset, source, condition) for index, source in enumerate(plan)
                for offset, condition in enumerate(conditions)]
    require(len(expected) == len(rows), "Every v277 development case is required")
    for (index, offset, entry, condition), row in zip(expected, rows, strict=True):
        source = entry["source"]
        require(row["status"] == "prepared" and row["condition"] == condition
                and row["case"] == f"case-{index:03d}-{offset:02d}"
                and (row["source_slot"], row["source_group"], row["lane"]) == (source["slot"], source["group_id"], source["lane"])
                and all(row[key] == entry[key] for key in ("tape_start_sample", "forced_quieter_channel"))
                and row["source_timeline"] == TIMELINE, "The frozen development case binding differs")
    return rows


def case_path(row):
    parent = pairs.MEDIA / "development"
    path = parent / f"{row['case']}.wav"
    require(path.is_file() and path.resolve(strict=True) == path and sha(path) == row["wav_sha256"],
            "Exact v277 development stereo WAV required")
    return path


def inputs():
    pairs.stereo.holdout.model_free()
    plan, pair_freeze, _ = pairs.verify("development")
    pairs.check_development_seal()
    result, profile, journal = pairs.guarded_result("development")
    dev = {**result, "resource_profile_sha256": sha(profile), "raw_case_replay_count": 185}
    encoded = (json.dumps(dev, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    require(pairs.stereo.digest(encoded) == DEV_SHA and result["case_outcomes"] == {"prepared": 185},
            "Exact completed v277 development readout required")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    bind_cases(plan, pair_freeze["conditions"]["development"], rows)
    for row in rows:
        case_path(row)
    return dev, rows


def identity(dev, cases):
    require(sha(gate.VAD) == gate.VAD_SHA, "Exact serving speech activity weights required")
    return {
        "schema": "chipmunk.audio-native-stereo-gate-freeze/279",
        "development_readout_sha256": DEV_SHA,
        "development_freeze_sha256": dev["freeze_sha256"],
        "development_private_cases_sha256": dev["private_cases_sha256"],
        "development_admission_sha256": sha(pairs.DEVELOPMENT_SEAL),
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")},
        "vad_sha256": gate.VAD_SHA,
        "source_role": "development",
        "planned_cases": len(cases),
        "max_vad_frame_forwards": len(cases) * len(VIEWS) * FORWARDS_PER_VIEW,
        "contrast_block_samples": FRAME,
        "minimum_active_and_quiet_blocks_each": MIN_BLOCKS,
        "minimum_contrast_gap_db": MIN_GAP_DB,
        "authenticity_detector_calls_permitted": False,
    }


def preflight():
    dev, cases = inputs()
    return {**identity(dev, cases), "status": "ready_for_freeze", "model_calls": 0}


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every prior stage attempt")
    dev, cases = inputs()
    lock = identity(dev, cases)
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    return {"status": "frozen", "freeze_sha256": sha(PUBLIC), "model_calls": 0}


def verify():
    dev, cases = inputs()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == identity(dev, cases), "Exact prospective v279 freeze required")
    return cases


def read_native(row):
    path = case_path(row)
    with wave.open(str(path), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes(), wav.getcomptype())
                == (2, 2, RATE, SAMPLES, "NONE"), "Exact native 16 kHz stereo WAV header required")
        body = wav.readframes(SAMPLES + 1)
    require(len(body) == 4 * SAMPLES and pairs.stereo.digest(body) == row["stereo_pcm_sha256"], "Native stereo PCM differs")
    values = np.frombuffer(body, dtype="<i2").reshape(-1, 2)
    require([pairs.stereo.digest(values[:, i].tobytes()) for i in (0, 1)] == row["channel_pcm_sha256"],
            "Observed channel PCM differs from v277")
    return path, values


def mono_pcm(row, path):
    decoded = decode_audio(path, source_sha256=row["wav_sha256"], max_seconds=3)
    require(decoded.mapping["timeline"] == TIMELINE and decoded.mapping["source_channels"] == 2
            and decoded.mapping["source_sample_rate"] == RATE and len(decoded.pcm) == SAMPLES
            and pairs.stereo.digest(pairs.stereo.development.read_wave(decoded.wav)) == row["mixed_pcm_sha256"],
            "Production mono PCM or clock differs from v277")
    return decoded.pcm


def choose_channel(pcm, channel_spans):
    """Acoustic selection has no access to a mono decision, source labels, or an authenticity score."""
    require(pcm.shape == (SAMPLES, 2) and pcm.dtype == np.dtype("<i2") and len(channel_spans) == 2,
            "Two exact observed signed-16-bit channels are required")
    activity = np.zeros(SAMPLES, dtype=bool)
    for spans in channel_spans:
        gate.coverage(spans)
        for start, stop in spans:
            activity[start:stop] = True
    count = SAMPLES // FRAME
    masks = activity[: count * FRAME].reshape(count, FRAME)
    active, quiet = masks.all(axis=1), ~masks.any(axis=1)
    result = {"selected": None, "active_blocks": int(active.sum()), "quiet_blocks": int(quiet.sum()),
              "contrast_db": None, "reason": "insufficient_active_or_quiet_support"}
    if np.any((pcm == -32768) | (pcm == 32767)):
        return {**result, "reason": "rail_clipping"}
    if min(result["active_blocks"], result["quiet_blocks"]) < MIN_BLOCKS:
        return result
    blocks = pcm[: count * FRAME].reshape(count, FRAME, 2)
    power = np.mean(np.square(blocks, dtype=np.float64), axis=1)
    contrasts = 10 * np.log10(np.maximum(np.median(power[active], axis=0), 1) / np.maximum(np.median(power[quiet], axis=0), 1))
    result["contrast_db"] = contrasts.tolist()
    if abs(contrasts[0] - contrasts[1]) < MIN_GAP_DB:
        return {**result, "reason": "contrast_gap_below_minimum"}
    return {**result, "selected": int(np.argmax(contrasts)), "reason": "selected"}


def observation(pcm, spans):
    require(set(spans) == set(VIEWS), "All three observed activity views are required")
    channel_spans = [spans[name] for name in VIEWS[1:]]
    selected = choose_channel(pcm, channel_spans)
    swapped = choose_channel(pcm[:, ::-1], channel_spans[::-1])
    index, contrast = selected["selected"], selected["contrast_db"]
    require(swapped == {**selected, "selected": None if index is None else 1 - index,
                        "contrast_db": None if contrast is None else contrast[::-1]}, "Channel swap changed the physical choice")
    admitted = {name: gate.coverage(spans[name])["baseline"] for name in VIEWS}
    return {**selected, "admitted": admitted, "swap_invariant": True}


def summarize(cases, observations):
    counts = defaultdict(Counter)
    for case, measured in zip(cases, observations, strict=True):
        require(measured["case"] == case["case"] and measured["swap_invariant"], "Exact ordered, swap-invariant observations required")
        admitted, selected = measured["admitted"], measured["selected"]
        mono = admitted["mono"]
        chosen = selected is not None and admitted[f"ch{selected}"]
        forced = case["condition"].endswith("_forced")
        for key in (("all", "all"), ("all", case["condition"]), (case["lane"], case["condition"])):
            row = counts[key]
            row.update(planned=1, mono_admitted=int(mono), selection_available=int(selected is not None),
                       selected_admitted=int(chosen), additional_admission=int(chosen and not mono),
                       mono_or_selected_admitted=int(mono or chosen), mono_only_admitted=int(mono and not chosen),
                       forced_planned=int(forced), forced_selection_available=int(forced and selected is not None),
                       forced_assignment_matches=int(forced and selected == case["forced_quieter_channel"]),
                       swap_invariant=1)
    return {
        "overall": dict(counts["all", "all"]),
        "by_condition": {condition: dict(count) for (lane, condition), count in counts.items() if lane == "all" and condition != "all"},
        "by_family_and_condition": {lane: {condition: dict(count) for (arm, condition), count in counts.items() if arm == lane}
                                    for lane in sorted({case["lane"] for case in cases})},
        "selection_reasons": dict(Counter(row["reason"] for row in observations)),
        "vad_frame_forwards": len(cases) * len(VIEWS) * FORWARDS_PER_VIEW,
        "authenticity_detector_calls": 0,
        "holdout_cases_opened": 0,
    }


def result(cases, observations, journal):
    totals = summarize(cases, observations)
    return {"schema": "chipmunk.audio-native-stereo-gate-result/279", "status": "complete", "freeze_sha256": sha(PUBLIC),
            "development_readout_sha256": DEV_SHA, "private_observations_sha256": sha(journal),
            "new_neural_calls": totals["vad_frame_forwards"], **totals}


def worker():
    require((STAGE / "resources.jsonl").is_file() and (STAGE / "worker.log").is_file()
            and not (STAGE / "profile.json").exists() and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Exact guarded offline VAD worker required")
    cases = verify()

    def offline(event, _args):
        if event == "socket.connect":
            raise RuntimeError("Network access forbidden in the speech activity worker")

    sys.addaudithook(offline)
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    activity = SpeechActivity(gate.VAD)
    activity.model = counter = gate.CountedVAD(activity.model)
    window_builder = PellaVoiceDetector()

    def measure(pcm):
        before = counter.calls
        seconds = activity.spans(pcm)
        spans = [[round(start * RATE), round(stop * RATE)] for start, stop in seconds]
        require(counter.calls - before == FORWARDS_PER_VIEW, "Unexpected frame count for speech activity")

        def leased(model_id):
            require(model_id == "silero.speech_activity", "Only speech activity may be leased")
            return SimpleNamespace(obj=SimpleNamespace(spans=lambda _: seconds))

        context = SimpleNamespace(profile="full", lease=SimpleNamespace(get=leased),
                                  outputs=SimpleNamespace(audio=AudioTrack("analysed", pcm, len(pcm))))
        windows = window_builder._windows(context)
        require(bool(windows) == gate.coverage(spans)["baseline"]
                and (not windows or (len(windows) == 1 and windows[0][0] == 0 and len(windows[0][1]) == SAMPLES)),
                "Activity gate must match the exact serving window builder")
        return spans

    OUTPUT.mkdir(mode=0o700)
    journal, observations = OUTPUT / "private-observations.jsonl", []
    with journal.open("x") as stream:
        for case in cases:
            path, pcm = read_native(case)
            spans = {"mono": measure(mono_pcm(case, path)),
                     **{f"ch{i}": measure(pcm[:, i].astype(np.float32) / 32768) for i in (0, 1)}}
            row = {"case": case["case"], "spans": spans, **observation(pcm, spans)}
            observations.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    aggregate = result(cases, observations, journal)
    require(counter.calls == aggregate["vad_frame_forwards"], "Actual VAD forward count differs")
    write(OUTPUT / "result.json", aggregate)


run = scoped(gate.run, {"TRIAL": TRIAL, "OUTPUT": OUTPUT, "verify": verify, "MODULE": MODULE, "__file__": __file__})


def readout():
    cases = verify()
    profile, result_path, journal = STAGE / "profile.json", OUTPUT / "result.json", OUTPUT / "private-observations.jsonl"
    resource, saved = read(profile), read(result_path)
    require(resource["schema"] == "chipmunk.video-appearance-resource-profile/108" and resource["command"] == "authored"
            and resource["child_started"] is True and resource["child_exit_code"] == 0 and resource["guard"] is None
            and resource["sample_errors"] == 0 and resource["runtime_sha256"] == sha(Path(__file__))
            and resource["resource_journal_sha256"] == sha(STAGE / "resources.jsonl")
            and resource["worker_receipt_sha256"] == sha(result_path) and resource["new_neural_calls"] == saved["new_neural_calls"],
            "Exact successful guarded speech activity receipt required")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    for case, row in zip(cases, rows, strict=True):
        _, pcm = read_native(case)
        require(row == {"case": case["case"], "spans": row["spans"], **observation(pcm, row["spans"])},
                "Replayed acoustic channel choice differs")
    require(saved == result(cases, rows, journal), "Replayed speech admission accounting differs")
    return {**saved, "resource_profile_sha256": sha(profile), "worker_duration_seconds": resource["duration_s"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"preflight": preflight, "freeze": freeze, "run": run, "worker": worker, "readout": readout}
    parser.add_argument("action", choices=commands)
    output = commands[parser.parse_args().action]()
    if output is not None:
        print(json.dumps(output, allow_nan=False, indent=2, sort_keys=True))
