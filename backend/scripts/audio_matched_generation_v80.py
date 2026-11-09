"""Dependency-free native serialization successor; preserve the stopped v79 attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from scripts import audio_matched_generation_v79 as prior
from scripts.audio_generator_checks_v54 import need
from scripts.audio_matched_assets_v78 import DESTINATION as PROFILE_ROOT
from scripts.audio_matched_assets_v78 import MANIFEST, MANIFEST_SHA

ROOT, BASE, RUNTIME = prior.ROOT, prior.BASE, prior.RUNTIME
TRIAL = BASE / "audio-matched-generation-v80"
OUTPUT = PROFILE_ROOT.parent / "generated-v80"
PUBLIC = ROOT / "docs/evaluation-panels/audio-matched-generation-v80-freeze.json"
sha, read, write, offline, profile_pass = prior.sha, prior.read, prior.write, prior.offline, prior.profile_pass
qualified = prior.qualified


def native_save(path, wave):
    import numpy as np

    need(wave.dtype == np.float32 and wave.ndim == 1 and bool(np.isfinite(wave).all()),
         "Complete finite native float32 mono output required")
    with path.open("xb") as stream:
        np.save(stream, wave, allow_pickle=False)
    readback = np.load(path, allow_pickle=False)
    need(readback.dtype == np.float32 and readback.ndim == 1
         and readback.tobytes() == wave.tobytes(), "Native serialization changed sample bytes")
    return readback


def bindings():
    names = set(prior.code_bindings()) | {
        "backend/scripts/audio_matched_generation_v80.py",
        "backend/scripts/audio_matched_runtime_v80.py",
        "backend/tests/test_audio_native_source_v80.py",
        "docs/audio-matched-generation-v80-protocol.md",
    }
    return {name: sha(ROOT / name) for name in sorted(names)}


def prepare():
    parent = prior.admission()
    profile_pass(prior.TRIAL / "preflight-profile/profile.json")
    preflight = read(prior.TRIAL / "preflight.json")
    need(preflight["status"] == "complete" and preflight["source_count"] == 64
         and sha(prior.TRIAL / "phoneme-plan.jsonl") == preflight["phoneme_plan_sha256"],
         "All original fixed phonemes must remain complete")
    failed = read(prior.TRIAL / "batch-00-profile/profile.json")
    log = (prior.TRIAL / "batch-00-profile/worker.log").read_text()
    need(failed["child_exit_code"] == 1 and failed["guard"] is None
         and failed["sample_errors"] == 0 and not failed["resource_limit_observations"]
         and "ModuleNotFoundError: No module named 'soundfile'" in log
         and not (prior.TRIAL / "batch-00.jsonl").exists()
         and not list(prior.OUTPUT.iterdir()), "Exact before-generation v79 dependency failure required")
    need(not TRIAL.exists() and not OUTPUT.exists() and not PUBLIC.exists(), "Never generated successor required")
    TRIAL.mkdir()
    OUTPUT.mkdir(parents=True)
    for name in ("private-plan.json", "private-paths.json", "phoneme-plan.jsonl"):
        (TRIAL / name).write_bytes((prior.TRIAL / name).read_bytes())
    (TRIAL / "preflight-profile").mkdir()
    (TRIAL / "preflight-profile/profile.json").write_bytes((prior.TRIAL / "preflight-profile/profile.json").read_bytes())
    (TRIAL / "preflight.json").write_bytes((prior.TRIAL / "preflight.json").read_bytes())
    frozen = {**parent, "schema": "chipmunk.matched-source-generation-freeze/80",
              "code_sha256": bindings(), "parent_freeze_sha256": sha(prior.PUBLIC),
              "parent_failed_profile_sha256": sha(prior.TRIAL / "batch-00-profile/profile.json"),
              "parent_failed_log_sha256": sha(prior.TRIAL / "batch-00-profile/worker.log"),
              "preflight_sha256": sha(TRIAL / "preflight.json"),
              "phoneme_plan_sha256": sha(TRIAL / "phoneme-plan.jsonl"),
              "container": "NumPy native float32 mono, exact byte readback; native clock 24000 Hz is recorded separately",
              "technical_change": "Only replace the unavailable soundfile import/WAV container with installed NumPy serialization. Same runtime, source/phoneme order, model, stock profiles, seeds, CPU math and resource limits. Preserved failed v79 attempt generated no audio.",
              "detector_calls_permitted": 0, "calibration_fitting_permitted": False}
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    print(json.dumps({"sources": 64, "preflight_sources_preserved": 64, "new_generator_forwards": 0,
                      "freeze_sha256": sha(PUBLIC)}), flush=True)


def admission():
    prior.admission()
    lock, start = read(TRIAL / "freeze.json"), read(TRIAL / "execution-start.json")
    need(Path(sys.prefix) == RUNTIME and sha(PUBLIC) == sha(TRIAL / "freeze.json") == start["freeze_sha256"]
         and start["remote_verified_before_generation"] is True, "Published exact successor freeze required")
    need(lock["code_sha256"] == bindings() and sha(prior.PUBLIC) == lock["parent_freeze_sha256"]
         and sha(MANIFEST) == lock["source_manifest_sha256"] == MANIFEST_SHA,
         "Frozen successor code/source/parent changed")
    for name, field in (("private-plan.json", "private_prompt_plan_sha256"),
                        ("private-paths.json", "private_paths_sha256"),
                        ("preflight.json", "preflight_sha256"), ("phoneme-plan.jsonl", "phoneme_plan_sha256")):
        need(sha(TRIAL / name) == lock[field], "Unchanged source/prompt/phoneme binding required")
    need(sha(prior.TRIAL / "batch-00-profile/profile.json") == lock["parent_failed_profile_sha256"]
         and sha(prior.TRIAL / "batch-00-profile/worker.log") == lock["parent_failed_log_sha256"],
         "Original failure evidence changed")
    profile_pass(TRIAL.parent / (TRIAL.name + "-preparation-profile/profile.json"))
    return lock


def generate(index):
    need(type(index) is int and 0 <= index < 4, "One fixed batch index required")
    lock = admission()
    profile_pass(TRIAL / "preflight-profile/profile.json")
    preflight_result = read(TRIAL / "preflight.json")
    need(preflight_result["status"] == "complete" and preflight_result["source_count"] == 64
         and sha(TRIAL / "phoneme-plan.jsonl") == preflight_result["phoneme_plan_sha256"], "Complete phoneme preflight required")
    for earlier in range(index):
        profile_pass(TRIAL / f"batch-{earlier:02d}-profile/profile.json")
        need(read(TRIAL / f"batch-{earlier:02d}-complete.json")["sources"] == 16, "Every earlier fixed batch must finish")
    attempts = offline()
    import numpy as np
    import torch
    from kokoro import KModel

    from scripts.audio_generator_checks_v54 import waveform_summary
    from scripts.audio_generator_checks_v57 import state_fingerprint, strict_loaded_state

    origins = {name: qualified.checked_module_origin(__import__(name)) for name in
               ("torch", "numpy", "kokoro", "misaki", "transformers", "en_core_web_sm")}
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    paths = read(TRIAL / "private-paths.json")
    root, model_path = Path(paths["assets_root"]), Path(paths["assets_root"]) / paths["model"]
    need(not torch.serialization.get_unsafe_globals_in_checkpoint(model_path), "Generator checkpoint has unsafe pickle globals")
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(root / "config.json"), model=str(model_path)).cpu().float().eval().requires_grad_(False)
    saved = torch.load(model_path, map_location="cpu", weights_only=True)
    strict = strict_loaded_state(model, saved)
    del saved
    original_state = state_fingerprint(model)
    folder = OUTPUT / f"batch-{index:02d}"
    folder.mkdir(exist_ok=False)
    prompts = read(TRIAL / "private-plan.json")[index * 16:(index + 1) * 16]
    phonemes = [json.loads(line) for line in (TRIAL / "phoneme-plan.jsonl").read_text().splitlines()]
    rows, profiles, forward_calls = [], {}, 0
    with (TRIAL / f"batch-{index:02d}.jsonl").open("x") as stream:
        for prompt in prompts:
            voice_path = PROFILE_ROOT / prompt["voice_asset"]
            if str(voice_path) not in profiles:
                need(sha(voice_path) == prompt["voice_asset_sha256"]
                     and not torch.serialization.get_unsafe_globals_in_checkpoint(voice_path), "Unsafe or changed profile checkpoint")
                profile = torch.load(voice_path, map_location="cpu", weights_only=True)
                need(isinstance(profile, torch.Tensor) and profile.dtype == torch.float32 and profile.ndim == 3
                     and profile.shape[1:] == (1, 256) and bool(torch.isfinite(profile).all()), "Finite native profile tensor required")
                profiles[str(voice_path)] = profile
            profile = profiles[str(voice_path)]
            before_profile = hashlib.sha256(profile.numpy().tobytes()).hexdigest()
            chunks = phonemes[prompt["index"]]
            need(chunks["index"] == prompt["index"], "Phoneme source ordering changed")
            started = time.monotonic()

            def render(current_chunks, current_profile, seed):
                nonlocal forward_calls
                outputs = []
                for ordinal, chunk in enumerate(current_chunks["chunks"]):
                    ps = chunk["phonemes"]
                    need(0 < len(ps) <= min(510, current_profile.shape[0]), "Native profile/source length contract changed")
                    torch.manual_seed(seed + ordinal)
                    with torch.inference_mode():
                        result = model(ps, current_profile[len(ps) - 1], speed=1.0, return_output=True)
                    forward_calls += 1
                    outputs.append(result.audio.cpu().numpy().astype(np.float32, copy=False))
                return np.concatenate(outputs).astype(np.float32, copy=False)

            wave = render(chunks, profile, prompt["seed"])
            summary = waveform_summary(wave, 24000)
            need(summary["seconds"] <= lock["maximum_native_seconds"], "Native output exceeded the frozen source bound")
            path = folder / f"source-{prompt['index']:02d}.npy"
            readback = native_save(path, wave)
            need(np.array_equal(wave, readback)
                 and hashlib.sha256(np.asarray(readback, "<f4").tobytes()).hexdigest() == summary["float32_sha256"],
                 "Native array readback changed generated sample bytes")
            repeat = None
            if prompt["index"] == 0:
                repeated = render(chunks, profile, prompt["seed"])
                repeat = np.array_equal(wave, repeated)
                np.save(folder / "first-source-repeat.npy", repeated, allow_pickle=False)
                need(repeat, "Seeded first source generation is not exactly repeatable")
            need(hashlib.sha256(profile.numpy().tobytes()).hexdigest() == before_profile, "Generator changed its profile input")
            row = {"index": prompt["index"], "role": prompt["role"], "path": str(path), "sha256": sha(path),
                   "voice_asset_sha256": prompt["voice_asset_sha256"], "chunks": len(chunks["chunks"]),
                   "seed": prompt["seed"], "repeat_exact": repeat, "source_four_second_eligible": summary["samples"] >= 96000,
                   "generation_duration_s": time.monotonic() - started, **summary}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            print(json.dumps({"batch": index, "generated_sources": len(rows), "planned": 16,
                              "native_seconds": round(summary["seconds"], 3)}), flush=True)
    need(not attempts and state_fingerprint(model) == original_state, "Network access or generator state changed")
    write(TRIAL / f"batch-{index:02d}-complete.json", {"sources": len(rows), "generator_forwards": forward_calls,
          "source_journal_sha256": sha(TRIAL / f"batch-{index:02d}.jsonl"), "strict_state": strict,
          "qualified_module_origins": origins,
          "state_unchanged_after_inference": True, "network_attempts": len(attempts), "detector_calls": 0})
    print(json.dumps({"batch": index, "sources": len(rows), "generator_forwards": forward_calls,
                      "detector_calls": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "generate"))
    parser.add_argument("index", type=int, nargs="?")
    args = parser.parse_args()
    if args.command == "prepare":
        need(args.index is None, "Preparation has no batch index")
        prepare()
    else:
        generate(args.index)
