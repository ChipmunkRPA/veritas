"""Qualify a fixed Okur stock-voice panel with three pinned ONNX graphs."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import requests

from scripts import audio_drifting_turkish_v215 as prior
from scripts.video_appearance_corrected_v112 import scoped

ROOT, ARCHIVE = prior.ROOT, prior.ARCHIVE
NAME = "audio-okur-turkish-v240"
MODULE = "scripts.audio_okur_turkish_v240"
TRIAL, SOURCE = prior.BASE / NAME, ARCHIVE / NAME
STAGE = TRIAL / "authored"
OUTPUT = STAGE / "output"
PUBLIC = ROOT / "docs/evaluation-panels/audio-okur-turkish-freeze-v240.json"
MODEL_REV = "0c1ae1a9c5e7bcaa46bdff732bc7182c9ae7bd70"
CODE_REV = "0889eda23a3eaeda33f8c0af3af729261907bcab"
MODEL = f"https://huggingface.co/ahakan/okur-tts/resolve/{MODEL_REV}/"
METADATA = f"https://huggingface.co/ahakan/okur-tts/raw/{MODEL_REV}/"
UPSTREAM = f"https://raw.githubusercontent.com/ahmethakanbesel/okur-tts/{CODE_REV}/"
NATIVE_RATE, HOP, LATENTS, HIDDEN = 48_000, 1_920, 64, 224
MIN_FRAMES, MAX_FRAMES = 75, 100
STEPS = (0.0, 0.25, 0.5, 0.75)
LETTERS = "abcdefghijklmnopqrstuvwxyzçğıöşüâîû"
SYMBOLS = ("<pad>", "<unk>", *" !\"%&'(),-./:;?", *LETTERS)
STOI = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHRASES = (
    "sabah erken saatte parkın yanındaki küçük dükkân açıldı.",
    "ince yağmurdan sonra bahçedeki taşlar güneşte yeniden parladı.",
    "eski radyonun yanında duran mavi defteri rafa koydum.",
    "akşamüstü sahilde yürüyen çocuklar uzaktaki vapura el salladı.",
    "yokuşun başındaki fırından taze ekmek kokusu geliyordu.",
    "köprünün altından geçen dere kıyıdaki çakılları yavaşça sürükledi.",
    "annem sofraya yeşil zeytin ve sıcak çorba getirdi.",
    "köy yolundaki kırmızı otobüs çeşmenin önünde kısa süre durdu.",
)
GRAPHS = {
    "text": (
        "text.6a21984ac693.onnx",
        5_114_579,
        "6a21984ac693d07ec0096fcfbc0bffebe23255891523090eaf24f232782a94c2",
    ),
    "sample": (
        "sample.3c2a91ee5faa.onnx",
        10_987_695,
        "3c2a91ee5faa5601ae53193c203fa558a28f7e031b82103af28ee3cf366a6c63",
    ),
    "decoder": (
        "decoder.8f42fa31155e.onnx",
        7_102_508,
        "8f42fa31155e459e62f3cb2830d53c7c014e5c614b583b17440120d7fa7120c4",
    ),
}
SOURCE_FILES = {
    "README.md": (
        METADATA + "README.md",
        5496,
        "475e7cd53bcffa70c06863563681e721d9a5dd3c296bf03243ce38fea1e6ee2c",
    ),
    "NOTICE": (METADATA + "NOTICE", 1882, "98892e78692055ffff295ffdf7f32079ee25fdac9cce5c87e6f87817901a5531"),
    "acoustic.json": (
        METADATA + "acoustic.json",
        426,
        "3614ad54aa6070f7bd4c2239b6858422a6a5a941d19edc09ed552dd722ba927b",
    ),
    "onnx/config.json": (
        METADATA + "onnx/config.json",
        742,
        "7b62f4383f4d0ed0fd5f0087bdb078e9cfb9516e36a4e3e61530bf037b663860",
    ),
    "upstream/LICENSE": (
        UPSTREAM + "LICENSE",
        10173,
        "a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9",
    ),
    "upstream/export_web.py": (
        UPSTREAM + "src/okur/export_web.py",
        12306,
        "c607509c828d35b8a3518b737882b90d4695bab3741f70314b7764a204df5bfc",
    ),
    "upstream/timeline.ts": (
        UPSTREAM + "web/src/core/timeline.ts",
        2166,
        "b9cd7f0926340f70f35a09f6cc1fd9279a456e4fd1fddbdfd9c68675b0da4b43",
    ),
    **{f"onnx/{name}": (MODEL + f"onnx/{name}", size, digest) for name, size, digest in GRAPHS.values()},
}
CONTRACTS = {
    "text": ({"ids": "int64"}, {"h": "float", "dur": "float"}),
    "sample": (
        {
            "h": "float",
            "dur": "float",
            "cw": "int64",
            "wstart": "int64",
            "fw": "int64",
            "fp": "float",
            "noise": "float",
        },
        {"z": "float"},
    ),
    "decoder": ({"z": "float"}, {"audio": "float"}),
}
CODE = (
    "backend/scripts/audio_okur_turkish_v240.py",
    "backend/tests/test_audio_okur_turkish_v240.py",
    "backend/scripts/audio_drifting_turkish_v215.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/app/detectors/audio/decode.py",
    "backend/app/detectors/video/decode.py",
    "docs/audio-okur-turkish-v240-method.md",
)
read, write, sha, require, digest = prior.read, prior.write, prior.sha, prior.require, prior.digest


def token_ids(phrase):
    require(
        re.fullmatch(rf"[{LETTERS}]+(?: [{LETTERS}]+)*\.", phrase) and 28 <= len(phrase) <= 75,
        "Expected one already-normalized Turkish sentence",
    )
    return np.array([[STOI[char] for char in phrase]], dtype=np.int64)


def plan():
    require(len(set(PHRASES)) == len(PHRASES) == 8, "Exact distinct authored sentences required")
    return [
        {
            "slot": f"synthetic-{i:02d}",
            "phrase": phrase,
            "seed": 240_000 + i,
            "token_sha256": digest(token_ids(phrase).astype("<i8").tobytes()),
        }
        for i, phrase in enumerate(PHRASES)
    ]


def identity():
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    require(all(tools.values()), "Production FFmpeg and FFprobe required")
    return {
        "schema": "chipmunk.audio-okur-turkish-freeze/240",
        "publisher_model_revision": MODEL_REV,
        "publisher_code_revision": CODE_REV,
        "model_license": "CC-BY-4.0",
        "code_license": "Apache-2.0",
        "source_assets": {
            name: {"url": url, "bytes": size, "sha256": checksum}
            for name, (url, size, checksum) in SOURCE_FILES.items()
        },
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "decoder_sha256": {name: sha(Path(path).resolve()) for name, path in tools.items()},
        "optional_cpu_runtime": prior.optional_runtime_identity(),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "native_sample_rate": NATIVE_RATE,
        "native_hop_samples": HOP,
        "native_frame_bounds": [MIN_FRAMES, MAX_FRAMES],
        "fixed_sampler_times": list(STEPS),
        "planned_slots": len(PHRASES),
        "stock_voice_groups": 1,
        "detector_calls_permitted": False,
    }


def freeze():
    require(not any(path.exists() for path in (TRIAL, SOURCE, PUBLIC)), "Preserve earlier study attempts")
    locked, rows = identity(), plan()
    TRIAL.mkdir(mode=0o700)
    private = TRIAL / "private-plan.json"
    write(private, rows)
    locked["private_plan_sha256"] = sha(private)
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return locked


def verify():
    private = TRIAL / "private-plan.json"
    locked = {**identity(), "private_plan_sha256": sha(private)}
    rows = plan()
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == locked and read(private) == rows,
        "Exact prospective proof and authored plan required",
    )
    return rows


class AssetStop(ValueError):
    """One bounded publisher acquisition did not complete."""


def signed_redirect(location, headers, size, checksum):
    target = urlsplit(location)
    if not (
        target.scheme == "https"
        and target.netloc == "us.aws.cdn.hf.co"
        and re.fullmatch(r"/xet-bridge-us/[0-9a-f]{24}/[0-9a-f]{64}", target.path)
        and target.query
        and not target.fragment
        and headers.get("X-Repo-Commit") == MODEL_REV
        and headers.get("X-Linked-Etag", "").strip('"') == checksum
        and headers.get("X-Linked-Size") == str(size)
    ):
        raise AssetStop("unexpected_publisher_redirect")
    return location


def fetch_file(session, asset, target, deadline):
    source_url, size, checksum = asset
    url = source_url
    for hop in range(2):
        if time.monotonic() >= deadline:
            raise AssetStop("acquisition_time_limit")
        with session.get(url, timeout=(8, 20), allow_redirects=False, stream=True) as response:
            if response.status_code in (301, 302, 303, 307, 308) and not hop and source_url.startswith(MODEL):
                url = signed_redirect(response.headers.get("Location", ""), response.headers, size, checksum)
                continue
            if response.status_code != 200:
                raise AssetStop(f"publisher_http_{response.status_code}")
            if response.headers.get("Content-Encoding", "identity") != "identity" or response.headers.get(
                "Content-Length", str(size)
            ) != str(size):
                raise AssetStop("unexpected_asset_extent")
            partial = target.with_suffix(target.suffix + ".partial")
            count = 0
            with partial.open("xb") as stream:
                for chunk in response.iter_content(chunk_size=65_536):
                    count += len(chunk)
                    if count > size or time.monotonic() >= deadline:
                        raise AssetStop("unexpected_asset_extent_or_deadline")
                    stream.write(chunk)
            if count != size or sha(partial) != checksum:
                raise AssetStop("asset_hash_or_extent_mismatch")
            partial.rename(target)
            return {"bytes": count, "sha256": checksum}
    raise AssetStop("publisher_redirect_limit")


def acquire():
    verify()
    require(
        ARCHIVE.parent.parent.is_mount()
        and ARCHIVE.resolve(strict=True) == ARCHIVE
        and not SOURCE.exists()
        and not SOURCE.is_symlink()
        and min(shutil.disk_usage(path).free for path in (ARCHIVE, TRIAL)) >= 3 * 1024**3,
        "Fresh owned archive path and existing disk reserves required",
    )
    SOURCE.mkdir(mode=0o700)
    records, stop = {}, None
    deadline = time.monotonic() + 180
    try:
        with requests.Session() as session:
            session.trust_env = False
            session.headers.update(
                {"Accept-Encoding": "identity", "User-Agent": "ChipmunkVerifyResearch/1.0"}
            )
            for name, asset in SOURCE_FILES.items():
                path = SOURCE / name
                path.parent.mkdir(mode=0o700, exist_ok=True)
                records[name] = fetch_file(session, asset, path, deadline)
    except AssetStop as error:
        stop = str(error)
    except (requests.RequestException, OSError, ValueError):
        stop = "publisher_transport_or_storage_error"
    receipt = SOURCE / "acquisition.json"
    write(receipt, {"freeze_sha256": sha(PUBLIC), "stop": stop, "records": records})
    return {
        "status": "complete" if stop is None else "incomplete",
        "verified_assets": len(records),
        "stop": stop,
        "acquisition_sha256": sha(receipt),
    }


def verified_assets():
    require(SOURCE.resolve(strict=True) == SOURCE, "Direct archived source required")
    receipt = read(SOURCE / "acquisition.json")
    require(
        receipt["freeze_sha256"] == sha(PUBLIC)
        and receipt["stop"] is None
        and set(receipt["records"]) == set(SOURCE_FILES),
        "Exact complete acquisition required",
    )
    for name, (_, size, checksum) in SOURCE_FILES.items():
        path = SOURCE / name
        require(
            path.resolve(strict=True) == path
            and path.is_file()
            and path.stat().st_size == size
            and sha(path) == checksum
            and receipt["records"][name] == {"bytes": size, "sha256": checksum},
            "Archived publisher asset differs from freeze",
        )
    config, acoustic = read(SOURCE / "onnx/config.json"), read(SOURCE / "acoustic.json")
    require(
        (config["sample_rate"], config["hop"], config["latent_dim"]) == (NATIVE_RATE, HOP, LATENTS)
        and config["times"] == list(STEPS)
        and config["symbols"] == list(SYMBOLS)
        and config["files"] == {key: value[0] for key, value in GRAPHS.items()}
        and config["bytes"] == {key: value[1] for key, value in GRAPHS.items()}
        and acoustic["speakers"] == [config["voice"]]
        and acoustic["model_config"]["d"] == HIDDEN
        and acoustic["model_config"]["n_speakers"] == 1,
        "Published graph and stock voice configuration differ",
    )


def timeline(phrase, durations):
    count = len(phrase)
    require(
        durations.shape == (1, count)
        and durations.dtype == np.float32
        and np.isfinite(durations).all()
        and np.all(durations >= 0),
        "Finite native letter durations required",
    )
    starts = np.array([0, *(i + 1 for i, char in enumerate(phrase) if char == " ")], dtype=np.int64)
    ends = np.append(starts[1:], count)
    lengths = ends - starts
    word_durations = np.add.reduceat(durations[0].astype(np.float64), starts)
    frame_ends = np.rint(np.cumsum(word_durations))
    frames = int(frame_ends[-1])
    if not MIN_FRAMES <= frames <= MAX_FRAMES:
        return frames, None
    frame_counts = np.diff(frame_ends, prepend=0).astype(np.int64)
    words = np.arange(len(starts), dtype=np.int64)
    return frames, {
        "cw": np.repeat(words, lengths)[None],
        "wstart": np.repeat(starts, lengths)[None],
        "fw": np.repeat(words, frame_counts)[None],
        "fp": np.concatenate([np.arange(n, dtype=np.float64) / n for n in frame_counts if n]).astype(
            np.float32
        )[None],
    }


def noise(seed, frames):
    uniform = prior.random_stream(seed)
    samples = np.empty(len(STEPS) * frames * LATENTS, dtype=np.float32)
    for index in range(0, samples.size, 2):
        radius = math.sqrt(-2.0 * math.log(1.0 - uniform()))
        angle = 2.0 * math.pi * uniform()
        samples[index] = radius * math.cos(angle)
        samples[index + 1] = radius * math.sin(angle)
    return samples.reshape(len(STEPS), 1, frames, LATENTS)


def render(item, run_graph):
    phrase = item["phrase"]
    ids = token_ids(phrase)
    require(digest(ids.astype("<i8").tobytes()) == item["token_sha256"], "Authored letter IDs differ")
    h, dur = run_graph("text", {"ids": ids})
    require(
        h.shape == (1, ids.size, HIDDEN) and h.dtype == np.float32 and np.isfinite(h).all(),
        "Exact finite text-stage hidden tensor required",
    )
    frames, conditioning = timeline(phrase, dur)
    if conditioning is None:
        return frames, None
    (z,) = run_graph("sample", {"h": h, "dur": dur, "noise": noise(item["seed"], frames), **conditioning})
    require(
        z.shape == (1, LATENTS, frames) and z.dtype == np.float32 and np.isfinite(z).all(),
        "Exact finite acoustic latents required",
    )
    (audio,) = run_graph("decoder", {"z": z})
    require(
        audio.shape == (1, frames * HOP)
        and audio.dtype == np.float32
        and np.isfinite(audio).all()
        and np.max(np.abs(audio)) <= 1,
        "Exact finite 48 kHz native waveform required",
    )
    return frames, audio[0]


def load_graphs():
    sys.path.insert(0, str(prior.OPTIONAL_RUNTIME))
    import onnxruntime as ort

    require(
        ort.__version__ == prior.ORT_VERSION
        and Path(ort.__file__).resolve().is_relative_to(prior.OPTIONAL_RUNTIME),
        "Only the committed isolated ONNX Runtime may load",
    )
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.use_deterministic_compute = True
    ort.disable_telemetry_events()
    sessions = {}
    for name, (inputs, outputs) in CONTRACTS.items():
        path = SOURCE / "onnx" / GRAPHS[name][0]
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        require(
            session.get_providers() == ["CPUExecutionProvider"]
            and {node.name: node.type for node in session.get_inputs()}
            == {k: f"tensor({t})" for k, t in inputs.items()}
            and {node.name: node.type for node in session.get_outputs()}
            == {k: f"tensor({t})" for k, t in outputs.items()},
            "Exact published CPU ONNX graph contract required",
        )
        sessions[name] = session
    return sessions


NATIVE = {"NATIVE_RATE": NATIVE_RATE, "MIN_NATIVE": MIN_FRAMES * HOP, "MAX_NATIVE": MAX_FRAMES * HOP}
native_wave = scoped(prior.native_wave, NATIVE)
prepare_waveform = scoped(prior.prepare_waveform, {**NATIVE, "native_wave": native_wave})
read_prepared_pcm = scoped(prior.read_prepared_pcm, {**NATIVE, "OUTPUT": OUTPUT})


def prepare_case(item, slot, sessions):
    calls = []

    def run_graph(name, feed):
        calls.append(name)
        return sessions[name].run(list(CONTRACTS[name][1]), feed)

    frames, audio = render(item, run_graph)
    row = {"slot": slot, "graph_calls": len(calls), "native_frames": frames}
    if audio is None:
        return {**row, "status": "unavailable", "reason": "predicted_duration_outside_three_to_four_seconds"}
    wav, evidence = prepare_waveform(audio, OUTPUT / "native" / f"{slot}.wav")
    with (OUTPUT / "prepared" / f"{slot}.wav").open("xb") as stream:
        stream.write(wav)
    return {**row, **evidence, "status": "qualified"}


def summarize(rows, repeat, planned):
    require(
        [row["slot"] for row in rows] == [item["slot"] for item in planned], "Every slot requires an outcome"
    )
    qualified = [row for row in rows if row["status"] == "qualified"]
    require(
        len({row["pcm_sha256"] for row in qualified}) == len(qualified)
        and all(
            row["graph_calls"] == 3 and MIN_FRAMES <= row["native_frames"] <= MAX_FRAMES for row in qualified
        )
        and all(
            row["status"] in ("qualified", "unavailable")
            and (
                row["status"] == "qualified"
                or (row["graph_calls"] == 1 and not MIN_FRAMES <= row["native_frames"] <= MAX_FRAMES)
            )
            for row in rows
        ),
        "Complete accounted calls and distinct qualified waveforms required",
    )
    if qualified:
        require(
            repeat == {**qualified[0], "slot": "synthetic-99"},
            "Exact first-qualified native and prepared replay required",
        )
    else:
        require(repeat is None, "Unexpected replay without a qualified case")
    return {
        "planned_slots": len(planned),
        "case_outcomes": dict(Counter(row["status"] for row in rows)),
        "unavailable_reasons": dict(Counter(row["reason"] for row in rows if row["status"] == "unavailable")),
        "stock_voice_groups": 1,
        "exact_replays": int(repeat is not None),
        "new_neural_calls": sum(row["graph_calls"] for row in rows)
        + (repeat["graph_calls"] if repeat else 0),
    }


def worker():
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline neural guard required",
    )
    planned = verify()
    verified_assets()
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
    rows, repeat = [], None
    journal = OUTPUT / "private-cases.jsonl"
    with journal.open("x") as stream:
        for item in planned:
            row = prepare_case(item, item["slot"], sessions)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if row["status"] == "qualified" and repeat is None:
                repeat = prepare_case(item, "synthetic-99", sessions)
                require(repeat == {**row, "slot": "synthetic-99"}, "Exact first-case replay differs")
                write(OUTPUT / "private-repeat.json", repeat)
    require(not network_attempts, "Unexpected source-worker network access")
    write(
        OUTPUT / "result.json",
        {
            "schema": "chipmunk.audio-okur-turkish-result/240",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "acquisition_sha256": sha(SOURCE / "acquisition.json"),
            "private_case_journal_sha256": sha(journal),
            "private_repeat_sha256": sha(OUTPUT / "private-repeat.json") if repeat else None,
            **summarize(rows, repeat, planned),
            "network_attempts": 0,
            "detector_calls": 0,
        },
    )


def run():
    verified_assets()
    scoped(
        prior.run,
        {"TRIAL": TRIAL, "OUTPUT": OUTPUT, "verify": verify, "MODULE": MODULE, "__file__": __file__},
    )()


def readout():
    planned = verify()
    verified_assets()
    path, profile_path = OUTPUT / "result.json", STAGE / "profile.json"
    profile = read(profile_path)
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "authored"
        and profile["batch"] is None
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["resource_journal_sha256"] == sha(STAGE / "resources.jsonl"),
        "Exact monitored stage profile required",
    )
    public_profile = {
        "resource_profile_sha256": sha(profile_path),
        "worker_duration_seconds": profile["duration_s"],
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
    }
    if profile["child_exit_code"] != 0 or profile["guard"] is not None:
        return {
            "status": "incomplete",
            "freeze_sha256": sha(PUBLIC),
            "planned_slots": len(planned),
            "child_exit_code": profile["child_exit_code"],
            "guard": profile["guard"],
            **public_profile,
        }
    journal, replay = OUTPUT / "private-cases.jsonl", OUTPUT / "private-repeat.json"
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    repeat = read(replay) if replay.is_file() else None
    result = read(path)
    require(
        profile["child_started"]
        and profile["sample_errors"] == 0
        and profile["worker_receipt_sha256"] == sha(path)
        and result["schema"] == "chipmunk.audio-okur-turkish-result/240"
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["acquisition_sha256"] == sha(SOURCE / "acquisition.json")
        and result["private_case_journal_sha256"] == sha(journal)
        and result["private_repeat_sha256"] == (sha(replay) if repeat else None)
        and all(result[key] == value for key, value in summarize(rows, repeat, planned).items())
        and result["new_neural_calls"] == profile["new_neural_calls"]
        and result["network_attempts"] == result["detector_calls"] == 0,
        "Exact completed local preparation receipts required",
    )
    for row in [*rows, *([repeat] if repeat else [])]:
        if row["status"] == "qualified":
            read_prepared_pcm(row)
    return {**result, **public_profile}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {
        "preflight": identity,
        "freeze": freeze,
        "acquire": acquire,
        "run": run,
        "worker": worker,
        "readout": readout,
    }
    parser.add_argument("command", choices=commands)
    result = commands[parser.parse_args().command]()
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
