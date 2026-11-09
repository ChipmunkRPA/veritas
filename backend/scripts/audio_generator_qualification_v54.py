"""Qualify a pinned offline CPU speech-generation path on authored prompts only."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

from scripts.audio_generator_checks_v54 import need

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / "backend/data/independent-evaluation/audio-generator-qualification-v54"
PREPARATION = ROOT / "backend/data/independent-evaluation/audio-generator-runtime-v54"
RUNTIME = Path("/private/tmp/chipmunk-audio-generator-runtime-v54")
PUBLIC = ROOT / "docs/evaluation-panels/audio-generator-qualification-v54-freeze.json"
PROMPTS = ("This is a local audio test. The yellow boat crossed the quiet lake before sunrise.",
           "A small bird rested near the open window while the rain stopped outside.")
SEED = 5407


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, body):
    with Path(path).open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")


def installed_record():
    need(Path(sys.prefix) == RUNTIME and platform.python_version() == "3.12.6", "The dedicated pinned interpreter is required")
    packages, files, total = {}, {}, 0
    for dist in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = dist.metadata["Name"]
        packages[name] = {"version": dist.version,
                          "license_expression": dist.metadata.get("License-Expression"),
                          "license_first_line": (dist.metadata.get("License") or "").split("\n", 1)[0][:160]}
        need(dist.files is not None, "Every installed distribution needs a file record")
        for item in dist.files:
            path = Path(dist.locate_file(item)).resolve()
            need(path.is_relative_to(RUNTIME) and path.is_file(), "Installed package file escaped or is missing")
            data = path.read_bytes()
            if item.hash is not None:
                actual = base64.urlsafe_b64encode(hashlib.new(item.hash.mode, data).digest()).rstrip(b"=").decode()
                need(actual == item.hash.value, "Installed package differs from its wheel file record")
            key = str(path.relative_to(RUNTIME))
            digest = hashlib.sha256(data).hexdigest()
            need(key not in files or files[key] == digest, "Installed distributions give conflicting source files")
            if key not in files:
                total += len(data)
            files[key] = digest
    return {"python": platform.python_version(), "packages": packages, "files": files,
            "installed_file_count": len(files), "installed_bytes": total}


def prepare():
    TRIAL.mkdir(exist_ok=False)
    inventory = installed_record()
    write(TRIAL / "installed-files.json", inventory)
    assets = read(PREPARATION / "asset-receipt.json")
    root = Path(assets["private_root"])
    need(assets["all_publisher_bindings_match"], "Publisher-bound assets required")
    for asset in assets["assets"]:
        path = root / asset["file"]
        need(path.stat().st_size == asset["bytes"] and sha(path) == asset["sha256"], "Downloaded generator asset changed")
    paths = {"assets_root": str(root), "model": "kokoro-v1_0.pth", "voice": "voices/af_heart.pt"}
    write(TRIAL / "private-paths.json", paths)
    code = [ROOT / path for path in (
        "backend/scripts/audio_generator_checks_v54.py", "backend/scripts/audio_generator_qualification_v54.py",
        "backend/scripts/audio_generator_observer_v54.py", "backend/tests/test_audio_generator_checks_v54.py",
        "backend/scripts/local_probe_limits_v24.py", "backend/scripts/data_resource_guard_v21.py",
        "backend/scripts/image_regional_continuation_runtime_v41.py", "docs/audio-generator-qualification-v54-protocol.md")]
    frozen = {"schema": "chipmunk.audio-generator-qualification/54", "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in code},
              "runtime_inventory_sha256": sha(TRIAL / "installed-files.json"),
              "runtime_packages": {k: v["version"] for k, v in inventory["packages"].items()},
              "installed_file_count": inventory["installed_file_count"],
              "base_requirements_sha256": sha(PREPARATION / "requirements-a02.lock"),
              "public_requirements_sha256": sha(ROOT / "docs/evaluation-panels/audio-generator-runtime-v54-requirements.lock"),
              "english_language_package": read(PREPARATION / "english-model-a02-download.json"),
              "english_package_download_sha256": sha(PREPARATION / "english-model-a02-download.json"),
              "private_paths_sha256": sha(TRIAL / "private-paths.json"),
              "model_source_revision": assets["revision"],
              "assets": [{"role": "voice" if a["file"].startswith("voices/") else "config" if a["file"].endswith("json") else "weights",
                          "bytes": a["bytes"], "sha256": a["sha256"], "publisher_binding": a["publisher_binding"]} for a in assets["assets"]],
              "prompts": list(PROMPTS), "seed": SEED, "device": "cpu", "dtype": "float32", "threads": 2,
              "generator_forward_calls": 3, "g2p_calls": 2, "detector_calls_permitted": 0,
              "expected_sample_rate": 24000, "maximum_phonemes": 510, "repeat_exact_required": True,
              "scope": "Authored runtime smoke, not fresh detection validation. Full G2P, strict loaded-state coverage, waveform/clock and repeat checks.",
              "defaults_changed": False, "protected_audio_sha256": sha(ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json")}
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    print(json.dumps({"packages": len(inventory["packages"]), "verified_files": inventory["installed_file_count"], "new_model_calls": 0}), flush=True)


def admission():
    frozen, start = read(TRIAL / "freeze.json"), read(TRIAL / "execution-start.json")
    need(sha(PUBLIC) == sha(TRIAL / "freeze.json") == start["freeze_sha256"]
         and start["remote_verified_before_generation"] is True, "Published exact qualification freeze required")
    need(all(sha(ROOT / k) == v for k, v in frozen["code_sha256"].items()), "Frozen qualification code changed")
    need(sha(TRIAL / "installed-files.json") == frozen["runtime_inventory_sha256"], "Runtime inventory changed")
    for name, value in read(TRIAL / "installed-files.json")["files"].items():
        need(sha(RUNTIME / name) == value, "Installed runtime source/data changed")
    need(sha(TRIAL / "private-paths.json") == frozen["private_paths_sha256"], "Private asset paths changed")
    profile = TRIAL.parent / (TRIAL.name + "-preparation-profile/profile.json")
    p = read(profile)
    need(sha(profile) == start["preparation_profile_sha256"] and p["child_exit_code"] == 0
         and p["guard"] is None and not p["sample_errors"] and not p["resource_limit_observations"], "Bounded preparation must pass")
    return frozen


def smoke():
    frozen = admission()
    import numpy as np
    import torch

    from scripts.audio_generator_checks_v54 import checked_phonemes, strict_loaded_state, waveform_summary

    folder = TRIAL / "smoke"
    folder.mkdir(exist_ok=False)
    paths = read(TRIAL / "private-paths.json")
    root = Path(paths["assets_root"])
    need(sha(root / paths["model"]) == next(a["sha256"] for a in frozen["assets"] if a["role"] == "weights"), "Generator weights changed")
    need(sha(root / paths["voice"]) == next(a["sha256"] for a in frozen["assets"] if a["role"] == "voice"), "Generator profile changed")
    need(sha(root / "config.json") == next(a["sha256"] for a in frozen["assets"] if a["role"] == "config"), "Generator config changed")
    attempts = []

    def offline(event, args):
        if event == "socket.connect":
            attempts.append(event)
            raise RuntimeError("Offline qualification refused a network connection")

    sys.addaudithook(offline)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    need(not torch.serialization.get_unsafe_globals_in_checkpoint(root / paths["model"]), "Generator checkpoint has unexpected pickle globals")
    need(not torch.serialization.get_unsafe_globals_in_checkpoint(root / paths["voice"]), "Speaker checkpoint has unexpected pickle globals")
    from kokoro import KModel, KPipeline

    started = time.monotonic()
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(root / "config.json"), model=str(root / paths["model"])).cpu().float().eval()
    saved = torch.load(root / paths["model"], map_location="cpu", weights_only=True)
    strict = strict_loaded_state(model, saved)
    del saved
    voice = torch.load(root / paths["voice"], map_location="cpu", weights_only=True)
    need(isinstance(voice, torch.Tensor) and voice.dtype == torch.float32 and voice.ndim == 3
         and voice.shape[1:] == (1, 256) and bool(torch.isfinite(voice).all()), "Invalid stock profile tensor")
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
    phonemes = []
    for text in PROMPTS:
        _, tokens = pipeline.g2p(text)
        phonemes.append(checked_phonemes(text, tokens, model.vocab))
    rows, outputs = [], []
    for number, index in enumerate((0, 1, 0)):
        ps = phonemes[index]
        need(len(ps) <= voice.shape[0], "Stock profile does not cover this input length")
        torch.manual_seed(SEED)
        before = time.monotonic()
        with torch.inference_mode():
            output = model(ps, voice[len(ps) - 1], speed=1.0, return_output=True)
        wave = output.audio.numpy().astype(np.float32, copy=False)
        summary = waveform_summary(wave, 24000)
        np.save(folder / f"authored-{number:02d}.npy", wave, allow_pickle=False)
        outputs.append(summary["float32_sha256"])
        rows.append({"call": number, "prompt": index, "phoneme_count": len(ps), "duration_s": time.monotonic() - before, **summary})
    need(outputs[0] == outputs[2], "Seeded CPU generator repeat differs")
    need(not attempts, "Generator attempted network access")
    write(folder / "completed.json", {"schema": "chipmunk.audio-generator-smoke-result/54", "status": "complete",
          "strict_loaded_state": strict, "calls": rows, "repeat_exact": True, "g2p_calls": 2,
          "new_generator_forwards": 3, "new_detector_calls": 0, "network_attempts": len(attempts),
          "device": "cpu", "dtype": "float32", "load_and_generation_duration_s": time.monotonic() - started,
          "scope": "Two authored English prompts through complete G2P and pinned generator. No research source, customer data, detector accuracy, intelligibility review or serving default is qualified."})
    print(json.dumps({"status": "complete", "new_generator_forwards": 3, "new_detector_calls": 0, "repeat_exact": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke"))
    args = parser.parse_args()
    {"prepare": prepare, "smoke": smoke}[args.command]()
