"""Prepare a fixed private Turkish Drifting TTS panel using pinned CPU ONNX graphs.

The browser frontend logic is adapted from Drifting TTS by Kadir Nar.
See licenses/Drifting-TTS-MIT.txt for the upstream notice.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio, read_wav, source_ranges
from scripts import audio_voxforge_tr_genuine_v199 as clock
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped
from scripts.video_appearance_native_v108 import ARCHIVE, BASE, ROOT, read, require, sha, write

NAME = "audio-drifting-turkish-v215"
MODULE = "scripts.audio_drifting_turkish_v215"
SOURCE = ARCHIVE / "audio-drifting-sources-v215"
OPTIONAL_RUNTIME = SOURCE / "optional-runtime"
TRIAL = BASE / NAME
STAGE = TRIAL / "authored"
OUTPUT = STAGE / "output"
PUBLIC = ROOT / "docs/evaluation-panels/audio-drifting-turkish-freeze-v215.json"
MODEL_REVISION = "0906c46992410a2032c6f5777b887f8f1780a6fd"
UPSTREAM_COMMIT = "b8cbe55d1614e2c79f3e7f997985c3144f1fbb4d"
ORT_VERSION = "1.23.2"
NATIVE_RATE, HOP, MELS, CHANNELS = 24000, 256, 100, 292
MIN_NATIVE, MAX_NATIVE = 3 * NATIVE_RATE, 4 * NATIVE_RATE
RATE, MIN_SAMPLES, MAX_SAMPLES = clock.RATE, clock.MIN_SAMPLES, clock.MAX_SAMPLES
VOICES = {"studio": 722, "male": 389, "female": 323}
GUIDANCE, TEMPERATURE, SEED_BASE = 2.0, 0.3, 215_000
NOISE_COORDS, NOISE_CLASSES = 32, 64
MASK32 = (1 << 32) - 1
LETTERS = "abcçdefgğhıijklmnoöpqrsştuüvwxyz"
SYMBOL_IDS = {symbol: index for index, symbol in enumerate(LETTERS + " .,!?", start=2)}

PROMPTS_NAME = "private-prompts-v215.json"
PROMPTS_SHA = "c153a2d70bc5839e29759e77dea1354b27f0b7b8a8253d6820f39c8084cb75e3"
PLANNED = 16

GRAPHS = {
    "text_encoder": (15_442_631, "5dd17e02a910d1431dd0ebfbd954c2d6fedf20b70e8db192569d277f54dcc1f5"),
    "generator": (120_568_346, "1aeff1c87e677b254f4b3ee047069341766ef7871ebe486ed4f1265148a71e77"),
    "vocoder": (233_310_687, "b7f8277cc4562bdbb6dbf9872b5840b48f095565555a446f79b7b1f442148721"),
}
METADATA = {
    "LICENSE": "86f750defa962604d2d78ae55afb1863d8f7ab58cec688a9ddee3df627fdb900",
    "README.md": "ac2f207731e3c03014f02eac81cb8e30d98410de13c55e32336e63405e7cf48f",
    "voices.json": "6d328bfa656f8587b315e2444b242dd497b3768287921a0f4970118634cd6010",
    "onnx/config.json": "ef16cfe370f495ecc25443ba741cd9268ccc140aceac95a02d66e41b698606f1",
    "model.json": "9c0d9ae36ff8299018fc0f086e5422f4639c3aa05464b1869a8ec9aa4d6ea3dd",
    "code-commit.json": "97d0fc745dca3f0da986e65704e1df35f426c2d7acbd2a3ef0de400a44e0e1cd",
    "upstream-tts.js": "95caf5a336842496b4b6ece8bd3957da239e82c2ede0c1502f4a01cf79077749",
    "upstream-text.js": "63926a2de334d7e82bafe71d4937a14a20d5222bbb8af63e3b1e6125340fb6a1",
    "upstream-text.py": "a8a3779bc295d93d6dde1e13dddc3263f368b3ad390a112dfc8d6a38b2f7241f",
    "upstream-export_onnx.py": "a524f0a9b17af8a817db80a419575b9a0ee568be1ee7cc5decb23e195de5c778",
}
CONTRACTS = {
    "text_encoder": ({"text": "int64", "speaker": "int64"}, {"cond": "float", "logw": "float"}),
    "generator": (
        {"z": "float", "cond": "float", "speaker": "int64", "cfg_scale": "float", "noise_labels": "int64"},
        {"mel": "float"},
    ),
    "vocoder": ({"mel": "float"}, {"audio": "float"}),
}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def token_ids(phrase):
    # The fixed prompts use only the publisher's plain Turkish character alphabet.
    text = phrase.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    require(
        re.fullmatch(rf"[{LETTERS}]+(?: [{LETTERS}]+)*[.!?]", text) and 25 <= len(text) <= 70,
        "One complete plain Turkish sentence required",
    )
    encoded = np.ones((1, len(text) * 2 + 1), dtype=np.int64)
    encoded[0, 1::2] = [SYMBOL_IDS[char] for char in text]
    return encoded


def roster(phrases):
    require(len(phrases) == len(set(phrases)) == PLANNED, "Exact 16 original phrases required")
    voices = tuple(VOICES)
    rows = []
    for index, phrase in enumerate(phrases):
        ids = token_ids(phrase)
        voice = voices[index % len(voices)]
        rows.append(
            {
                "slot": f"synthetic-{index:02d}",
                "phrase": phrase,
                "voice": voice,
                "voice_id": VOICES[voice],
                "seed": SEED_BASE + index,
                "tokens_sha256": digest(ids.astype("<i8").tobytes()),
            }
        )
    return rows


def private_plan():
    path = SOURCE / PROMPTS_NAME
    require(
        path.resolve(strict=True) == path and sha(path) == PROMPTS_SHA,
        "Exact private authored phrase list required",
    )
    return roster(read(path))


def verified_assets():
    for name, expected in METADATA.items():
        path = SOURCE / name
        require(
            path.resolve(strict=True) == path and sha(path) == expected,
            f"Exact publisher metadata required: {name}",
        )
    model, code = read(SOURCE / "model.json"), read(SOURCE / "code-commit.json")
    config, voices = read(SOURCE / "onnx/config.json"), read(SOURCE / "voices.json")
    require(
        model["id"] == "Vyvo/drifting-tts-tr"
        and model["sha"] == MODEL_REVISION
        and model["cardData"]["license"] == "cc-by-4.0"
        and code["sha"] == UPSTREAM_COMMIT
        and "MIT License" in (SOURCE / "LICENSE").read_text()
        and "The weights are released under CC-BY-4.0 and the code under MIT"
        in (SOURCE / "README.md").read_text()
        and {row["name"]: row["id"] for row in voices} == config["voices"] == VOICES
        and next(row for row in voices if row["name"] == "studio")["checkpoint"] == "drifting_tts_v3.1.pt"
        and (config["sample_rate"], config["hop_length"], config["n_mels"], config["cond_channels"])
        == (NATIVE_RATE, HOP, MELS, CHANNELS)
        and (config["noise_coords"], config["noise_classes"], config["temperature"])
        == (NOISE_COORDS, NOISE_CLASSES, TEMPERATURE),
        "Pinned graph, stock voice and license metadata differs",
    )
    siblings = {row["rfilename"]: row for row in model["siblings"]}
    for graph, (size, checksum) in GRAPHS.items():
        name = f"{graph}_fp16.onnx"
        relative = f"onnx/{name}"
        path, metadata = SOURCE / relative, siblings[relative]
        require(
            path.resolve(strict=True) == path
            and path.stat().st_size
            == size
            == metadata["size"]
            == metadata["lfs"]["size"]
            == config["files"][name]
            and sha(path) == checksum == metadata["lfs"]["sha256"],
            f"Exact pinned ONNX graph required: {graph}",
        )
    return config


def optional_runtime_identity():
    metadata = OPTIONAL_RUNTIME / f"onnxruntime-{ORT_VERSION}.dist-info"
    require(metadata.is_dir(), "Pinned optional ONNX Runtime 1.23.2 is required")
    dist = importlib.metadata.Distribution.at(metadata)
    require(
        dist.metadata["Name"].casefold() == "onnxruntime" and dist.version == ORT_VERSION,
        "Exact isolated ONNX Runtime package metadata required",
    )
    assets = {}
    for member in dist.files or ():
        if member.parts[0] != "onnxruntime" or member.suffix not in {".py", ".so", ".dylib"}:
            continue
        path = OPTIONAL_RUNTIME / member
        require(
            path.resolve(strict=True) == path and path.is_file(),
            "Exact isolated runtime package files required",
        )
        assets[member.as_posix()] = sha(path)
    require(
        "onnxruntime/__init__.py" in assets and any(name.endswith(".so") for name in assets),
        "Native optional CPU runtime required",
    )
    return {"version": dist.version, "package_sha256": digest(json.dumps(assets, sort_keys=True).encode())}


def identity():
    config = verified_assets()
    code = (
        Path(__file__),
        ROOT / "backend/scripts/licenses/Drifting-TTS-MIT.txt",
        ROOT / "backend/tests/test_audio_drifting_turkish_v215.py",
        ROOT / "docs/audio-drifting-turkish-v215-method.md",
        Path(runtime.__file__),
        ROOT / "backend/app/detectors/audio/decode.py",
        ROOT / "backend/app/detectors/video/decode.py",
        Path(clock.__file__),
    )
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    require(all(tools.values()), "Production decoder tools required")
    return {
        "schema": "chipmunk.audio-drifting-turkish-freeze/215",
        "publisher_model_revision": MODEL_REVISION,
        "upstream_code_commit": UPSTREAM_COMMIT,
        "model_license": "CC-BY-4.0",
        "code_and_upstream_vocoder_license": "MIT",
        "graph_bytes_and_sha256": {
            name: {"bytes": size, "sha256": checksum} for name, (size, checksum) in GRAPHS.items()
        },
        "publisher_metadata_sha256": METADATA,
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in code},
        "ffmpeg_tool_sha256": {name: sha(Path(path).resolve()) for name, path in tools.items()},
        "optional_cpu_runtime": optional_runtime_identity(),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "provider": "CPUExecutionProvider",
        "temperature": TEMPERATURE,
        "guidance_scale": GUIDANCE,
        "length_scale": 1.0,
        "native_sample_rate": NATIVE_RATE,
        "native_hop_samples": HOP,
        "production_sample_rate": RATE,
        "native_sample_bounds": [MIN_NATIVE, MAX_NATIVE],
        "production_sample_bounds": [MIN_SAMPLES, MAX_SAMPLES],
        "private_authored_phrases_sha256": PROMPTS_SHA,
        "planned_slots": PLANNED,
        "planned_slots_by_stock_voice": dict(Counter(row["voice"] for row in private_plan())),
        "stock_voice_duration_scales": config["duration_scales"],
        "role": "independent_synthetic_source_preparation",
        "detector_calls_permitted": False,
        "holdout_media_opened": False,
        "holdout_scored": False,
    }, config


def preflight():
    return identity()[0]


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve any earlier stage")
    lock, _ = identity()
    TRIAL.mkdir(mode=0o700)
    plan = TRIAL / "private-plan.json"
    write(plan, private_plan())
    lock["private_plan_sha256"] = sha(plan)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    return lock


def verify():
    lock, config = identity()
    plan = TRIAL / "private-plan.json"
    lock["private_plan_sha256"] = sha(plan)
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == lock and read(plan) == private_plan(),
        "Exact local freeze and source plan required",
    )
    return read(plan), config


class UnsupportedDuration(ValueError):
    """A fixed phrase produced an extent outside the study range."""


def random_stream(seed):
    """Reproduce the upstream browser seed stream without external JS or torch."""
    state = seed & MASK32

    def uniform():
        nonlocal state
        state = (state + 0x6D2B79F5) & MASK32
        value = ((state ^ (state >> 15)) * (state | 1)) & MASK32
        mixed = ((value ^ (value >> 7)) * (value | 61)) & MASK32
        value ^= (value + mixed) & MASK32
        return (value ^ (value >> 14)) / (MASK32 + 1)

    return uniform


def model_noise(seed, frames):
    uniform = random_stream(seed)
    noise = np.empty(MELS * frames, dtype=np.float32)
    # The upstream Float32Array rounds each normal before multiplying by temperature.
    for index in range(0, noise.size, 2):
        radius = math.sqrt(-2 * math.log(1 - uniform()))
        angle = 2 * math.pi * uniform()
        noise[index] = radius * math.cos(angle)
        noise[index + 1] = radius * math.sin(angle)
    noise = (noise.astype(np.float64) * TEMPERATURE).astype(np.float32)
    labels = np.array([[int(uniform() * NOISE_CLASSES) for _ in range(NOISE_COORDS)]], dtype=np.int64)
    return noise.reshape(1, MELS, frames), labels


def expand_condition(condition, log_durations, token_count, scale):
    require(
        condition.dtype == log_durations.dtype == np.float32
        and condition.shape == (1, CHANNELS, token_count)
        and log_durations.size == token_count
        and np.isfinite(condition).all()
        and np.isfinite(log_durations).all(),
        "Publisher text-encoder output contract differs",
    )
    # Browser Math.exp then fround, followed by a second fround after multiplying.
    with np.errstate(over="ignore", invalid="ignore"):
        durations = np.ceil(np.exp(log_durations.astype(np.float64)).astype(np.float32) * np.float32(scale))
    if not np.isfinite(durations).all():
        raise UnsupportedDuration("nonfinite_predicted_duration")
    frames = int(durations.sum(dtype=np.float64))
    if not MIN_NATIVE <= frames * HOP <= MAX_NATIVE:
        raise UnsupportedDuration("predicted_native_duration_outside_3_to_4_seconds")
    return np.repeat(condition, durations.reshape(-1).astype(np.int64), axis=2), frames


def render(row, config, run_graph):
    tokens = token_ids(row["phrase"])
    require(digest(tokens.astype("<i8").tobytes()) == row["tokens_sha256"], "Fixed sentence tokens changed")
    speaker = np.array([row["voice_id"]], dtype=np.int64)
    condition, durations = run_graph("text_encoder", {"text": tokens, "speaker": speaker})
    aligned, frames = expand_condition(
        condition, durations, tokens.size, config["duration_scales"][str(speaker[0])]
    )
    noise, labels = model_noise(row["seed"], frames)
    (mel,) = run_graph(
        "generator",
        {
            "z": noise,
            "cond": aligned,
            "speaker": speaker,
            "cfg_scale": np.array([GUIDANCE], dtype=np.float32),
            "noise_labels": labels,
        },
    )
    require(
        mel.dtype == np.float32 and mel.shape == (1, MELS, frames) and np.isfinite(mel).all(),
        "Exact finite log-mel output required",
    )
    (audio,) = run_graph("vocoder", {"mel": mel})
    require(
        audio.dtype == np.float32
        and audio.shape == (1, frames * HOP)
        and np.isfinite(audio).all()
        and np.max(np.abs(audio)) <= 1,
        "Exact finite publisher vocoder waveform required",
    )
    return audio[0], frames


def load_graphs():
    sys.path.insert(0, str(OPTIONAL_RUNTIME))
    import onnxruntime as ort

    require(
        ort.__version__ == ORT_VERSION and Path(ort.__file__).resolve().is_relative_to(OPTIONAL_RUNTIME),
        "Only the committed isolated ONNX Runtime may load",
    )
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.use_deterministic_compute = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort.disable_telemetry_events()
    sessions = {}
    for name, (inputs, outputs) in CONTRACTS.items():
        session = ort.InferenceSession(
            str(SOURCE / f"onnx/{name}_fp16.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        require(
            session.get_providers() == ["CPUExecutionProvider"]
            and {arg.name: arg.type for arg in session.get_inputs()}
            == {key: f"tensor({value})" for key, value in inputs.items()}
            and {arg.name: arg.type for arg in session.get_outputs()}
            == {key: f"tensor({value})" for key, value in outputs.items()},
            f"Exact CPU graph interface required: {name}",
        )
        sessions[name] = session
    return sessions


def native_wave(audio):
    require(
        audio.dtype == np.float32
        and audio.ndim == 1
        and MIN_NATIVE <= audio.size <= MAX_NATIVE
        and np.isfinite(audio).all(),
        "Exact bounded mono native waveform required",
    )
    pcm = audio.astype("<f4").tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        len(pcm) + 36,
        b"WAVE",
        b"fmt ",
        16,
        3,
        1,
        NATIVE_RATE,
        NATIVE_RATE * 4,
        4,
        32,
        b"data",
        len(pcm),
    )
    return header + pcm


def prepare_waveform(audio, source_path):
    native = native_wave(audio)
    source_sha = digest(native)
    with source_path.open("xb") as stream:
        stream.write(native)
    result = decode_audio(source_path, source_sha256=source_sha, max_seconds=5)
    pcm, mapping = result.pcm, result.mapping
    samples = len(pcm)
    if not MIN_SAMPLES <= samples <= MAX_SAMPLES:
        raise UnsupportedDuration("production_duration_outside_3_to_4_seconds")
    whole = clock.sample_range(0, samples)
    require(
        mapping["source_sha256"] == source_sha
        and mapping["source_codec"] == "pcm_f32le"
        and mapping["source_sample_rate"] == NATIVE_RATE
        and mapping["source_channels"] == 1
        and mapping["sample_rate"] == RATE
        and mapping["sample_format"] == "s16le"
        and mapping["samples"] == samples
        and abs(samples - audio.size * RATE / NATIVE_RATE) <= 1
        and source_ranges(mapping, 0, samples) == [whole]
        and mapping["pcm_sha256"] == clock.pcm_sha(pcm)
        and np.array_equal(read_wav(result.wav), pcm),
        "Full native-to-production source clock and signed PCM must round-trip",
    )
    require(np.any(pcm), "Production waveform is silent")
    return result.wav, {
        "native_wav_sha256": source_sha,
        "native_sample_sha256": digest(native[44:]),
        "native_samples": audio.size,
        "native_sample_rate": NATIVE_RATE,
        "samples": samples,
        "sample_rate": RATE,
        "source_range": whole,
        "wav_range": whole,
        "pcm_sha256": mapping["pcm_sha256"],
        "wav_sha256": digest(result.wav),
    }


def read_prepared_pcm(row):
    slot = row["slot"]
    require(
        row["status"] == "qualified"
        and row["sample_rate"] == RATE
        and re.fullmatch(r"synthetic-[0-9]{2}", slot)
        and MIN_SAMPLES <= row["samples"] <= MAX_SAMPLES
        and row["source_range"] == row["wav_range"] == clock.sample_range(0, row["samples"]),
        "Exact private synthetic PCM and source-range receipt required",
    )
    prepared, native = (OUTPUT / name / f"{slot}.wav" for name in ("prepared", "native"))
    require(
        all(path.resolve(strict=True) == path and path.is_file() for path in (prepared, native)),
        "Direct saved source and prepared WAVs required",
    )
    body = prepared.read_bytes()
    source = native.read_bytes()
    pcm = read_wav(body)
    require(
        len(pcm) == row["samples"]
        and clock.pcm_sha(pcm) == row["pcm_sha256"]
        and digest(body) == row["wav_sha256"]
        and digest(source) == row["native_wav_sha256"]
        and row["native_sample_rate"] == NATIVE_RATE
        and MIN_NATIVE <= row["native_samples"] <= MAX_NATIVE
        and len(source) == 44 + row["native_samples"] * 4
        and digest(source[44:]) == row["native_sample_sha256"],
        "Saved native and production artifacts differ",
    )
    return pcm


def summarize(rows, plan):
    require(
        len(rows) == len(plan) == PLANNED
        and [row["slot"] for row in rows] == [row["slot"] for row in plan]
        and all(
            row["status"] in {"qualified", "unavailable"}
            and row["label"] == 1
            and row["language"] == "tr"
            and row["voice"] == item["voice"]
            and row["graph_calls"] in (1, 3)
            and (row["status"] != "qualified" or row["graph_calls"] == 3)
            for row, item in zip(rows, plan, strict=True)
        ),
        "Every planned source and graph call must be accounted for",
    )
    qualified = [row for row in rows if row["status"] == "qualified"]
    require(len({row["pcm_sha256"] for row in qualified}) == len(qualified), "Duplicate production waveform")
    return {
        "planned_slots": PLANNED,
        "case_outcomes": dict(Counter(row["status"] for row in rows)),
        "unavailable_reasons": dict(Counter(row["reason"] for row in rows if row["status"] == "unavailable")),
        "outcomes_by_stock_voice": {
            voice: dict(Counter(row["status"] for row in rows if row["voice"] == voice)) for voice in VOICES
        },
        "new_neural_calls": sum(row["graph_calls"] for row in rows),
        "acoustic_generator_calls": sum(row["graph_calls"] == 3 for row in rows),
    }


def worker():
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline neural guard required",
    )
    plan, config = verify()
    network_attempts = []

    def offline(event, _args):
        if event == "socket.connect":
            network_attempts.append(event)
            raise RuntimeError("Network access forbidden in the source worker")

    sys.addaudithook(offline)
    sessions = load_graphs()
    OUTPUT.mkdir(mode=0o700)
    for name in ("native", "prepared"):
        (OUTPUT / name).mkdir(mode=0o700)
    rows, calls = [], []

    def run_graph(name, feeds):
        calls.append(name)
        return sessions[name].run(list(CONTRACTS[name][1]), feeds)

    journal = OUTPUT / "private-cases.jsonl"
    with journal.open("x") as stream:
        for row in plan:
            calls.clear()
            slot = row["slot"]
            receipt = {"slot": slot, "voice": row["voice"], "language": "tr", "label": 1}
            native_path = OUTPUT / "native" / f"{slot}.wav"
            try:
                audio, frames = render(row, config, run_graph)
                wav, evidence = prepare_waveform(audio, native_path)
            except UnsupportedDuration as error:
                receipt |= {"status": "unavailable", "reason": str(error)}
            else:
                with (OUTPUT / "prepared" / f"{slot}.wav").open("xb") as destination:
                    destination.write(wav)
                receipt |= {"status": "qualified", "native_mel_frames": frames, **evidence}
                read_prepared_pcm(receipt)
            require(calls == list(CONTRACTS)[: len(calls)], "Published graph call order changed")
            receipt["graph_calls"] = len(calls)
            rows.append(receipt)
            stream.write(json.dumps(receipt, allow_nan=False) + "\n")
            stream.flush()
    require(not network_attempts, "Unexpected network access during source generation")
    write(
        OUTPUT / "result.json",
        {
            "schema": "chipmunk.audio-drifting-turkish-result/215",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_case_journal_sha256": sha(journal),
            **summarize(rows, plan),
            "network_attempts": 0,
            "detector_calls": 0,
            "holdout_media_opened": False,
            "holdout_scored": False,
        },
    )


def run():
    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(OUTPUT)],
            "Only the exact guarded Turkish source worker may start",
        )
        return subprocess.Popen([*argv[:3], MODULE, "worker"], **kwargs)

    proxy = types.SimpleNamespace(
        Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
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
    plan, _ = verify()
    result_path, profile_path = OUTPUT / "result.json", STAGE / "profile.json"
    journal = OUTPUT / "private-cases.jsonl"
    result, profile = read(result_path), read(profile_path)
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "authored"
        and profile["batch"] is None
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["resource_journal_sha256"] == sha(STAGE / "resources.jsonl")
        and result["schema"] == "chipmunk.audio-drifting-turkish-result/215"
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_case_journal_sha256"] == sha(journal)
        and all(result[key] == value for key, value in summarize(rows, plan).items())
        and result["new_neural_calls"] == profile["new_neural_calls"]
        and result["network_attempts"] == result["detector_calls"] == 0
        and result["holdout_media_opened"] is result["holdout_scored"] is False,
        "Exact monitored generation and private source receipt required",
    )
    for row in rows:
        if row["status"] == "qualified":
            read_prepared_pcm(row)
    return {
        **result,
        "resource_profile_sha256": sha(profile_path),
        "worker_duration_seconds": profile["duration_s"],
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"preflight": preflight, "freeze": freeze, "run": run, "readout": readout, "worker": worker}
    parser.add_argument("command", choices=commands)
    outcome = commands[parser.parse_args().command]()
    if outcome is not None:
        print(json.dumps(outcome, indent=2, allow_nan=False))
