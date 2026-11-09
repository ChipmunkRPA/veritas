"""Generate every fixed matched source with the qualified offline CPU speech model."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from scripts import audio_generator_qualification_v57 as qualified
from scripts.audio_generator_checks_v54 import checked_phonemes, need
from scripts.audio_matched_assets_v78 import DESTINATION as PROFILE_ROOT
from scripts.audio_matched_assets_v78 import MANIFEST, MANIFEST_SHA

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "audio-matched-generation-v79"
OUTPUT = PROFILE_ROOT.parent / "generated-v79"
PUBLIC = ROOT / "docs/evaluation-panels/audio-matched-generation-v79-freeze.json"
RUNTIME = qualified.RUNTIME


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, body):
    with path.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_seed(prompt):
    value = "chipmunk-matched-source-v79\0" + prompt["canonical_speaker"] + "\0" + prompt["matched_genuine_source_id"]
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def complete_chunks(text, pipeline, vocab):
    _, tokens = pipeline.g2p(text)
    need(bool(tokens), "Complete source tokens required")
    for token in tokens:
        if any(character.isalnum() for character in token.text):
            need(isinstance(token.phonemes, str) and bool(token.phonemes), "G2P dropped a spoken source token")
    before = [(token.text, token.whitespace, token.phonemes or "") for token in tokens]
    chunks = list(pipeline.en_tokenize(tokens))
    after = [(token.text, token.whitespace, token.phonemes or "") for _, _, group in chunks for token in group]
    need(before == after, "Upstream chunking dropped, duplicated or changed source tokens")
    need(0 < len(chunks) <= 4, "Bounded complete upstream English chunks required")
    result = []
    for graphemes, phonemes, group in chunks:
        need(checked_phonemes(graphemes, group, vocab) == phonemes,
             "Unknown, changed or overlong phonemes cannot be silently filtered or truncated")
        result.append({"phonemes": phonemes, "token_count": len(group)})
    return result


def code_bindings():
    old = read(qualified.TRIAL / "freeze.json")["code_sha256"]
    names = set(old) | {
        "backend/scripts/audio_matched_assets_v78.py", "backend/tests/test_audio_matched_assets_v78.py",
        "backend/scripts/audio_matched_generation_v79.py", "backend/scripts/audio_matched_runtime_v79.py",
        "backend/tests/test_audio_matched_generation_v79.py", "docs/audio-matched-generation-v79-protocol.md",
        "backend/scripts/audio_tree_sampler_v64.py", "backend/scripts/local_probe_limits_v24.py",
        "backend/scripts/image_regional_continuation_runtime_v41.py",
        "backend/scripts/audio_source_ancestry_v75.py", "backend/scripts/review_audio_sources_v76.py",
        "backend/scripts/review_audio_context_v77.py",
    }
    return {name: sha(ROOT / name) for name in sorted(names)}


def paths_and_assets():
    paths = read(qualified.TRIAL / "private-paths.json")
    root = Path(paths["assets_root"])
    original = read(qualified.TRIAL / "freeze.json")
    for role, path in (("weights", root / paths["model"]), ("config", root / "config.json")):
        need(sha(path) == next(row["sha256"] for row in original["assets"] if row["role"] == role),
             "Qualified model/config bytes changed")
    receipt = read(BASE / "audio-matched-assets-v78/receipt.json")
    for row in receipt["assets"]:
        path = PROFILE_ROOT / row["file"]
        need(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Stock-profile bytes changed")
    return paths, root, receipt


def prepare():
    need(Path(sys.prefix) == RUNTIME and platform.python_version() == "3.12.6", "Qualified dedicated interpreter required")
    old = qualified.admission()
    need(sha(MANIFEST) == MANIFEST_SHA, "Exact corrected source plan required")
    manifest = read(MANIFEST)
    review_path = BASE / "audio-source-ancestry-v75/primary-context-review-v77.json"
    review = read(review_path)
    need(review["status"] == "complete" and review["manifest_sha256"] == MANIFEST_SHA
         and review["source_graph_review"]["cross_role_components"] == 0, "Complete primary-context and ancestry review required")
    paths, root, receipt = paths_and_assets()
    need(not TRIAL.exists() and not OUTPUT.exists() and not PUBLIC.exists(), "Never generated destination required")
    parents = {row["id"]: row for row in manifest["foregrounds"] if row["kind"] == "genuine"}
    prompts = []
    for index, prompt in enumerate(manifest["pending_matched_generation"]):
        parent = parents[prompt["matched_genuine_source_id"]]
        need(parent["role"] == prompt["role"] and parent["transcript"] == prompt["transcript"]
             and parent["pcm_sha256"] == prompt["matched_genuine_pcm_sha256"]
             and sha(Path(parent["source_path"])) == parent["raw_sha256"], "Matched source text/role/bytes changed")
        prompts.append({**prompt, "index": index, "seed": source_seed(prompt)})
    need(len(prompts) == 64 and len({row["seed"] for row in prompts}) == 64, "All fixed sources and distinct source seeds required")
    TRIAL.mkdir()
    OUTPUT.mkdir(parents=True)
    write(TRIAL / "private-plan.json", prompts)
    write(TRIAL / "private-paths.json", {"assets_root": str(root), "model": paths["model"], "profiles_root": str(PROFILE_ROOT)})
    frozen = {"schema": "chipmunk.matched-source-generation-freeze/79", "source_manifest_sha256": MANIFEST_SHA,
              "private_prompt_plan_sha256": sha(TRIAL / "private-plan.json"),
              "private_paths_sha256": sha(TRIAL / "private-paths.json"), "qualified_runtime_freeze_sha256": sha(qualified.PUBLIC),
              "runtime_inventory_sha256": old["runtime_inventory_sha256"],
              "stock_profile_receipt_sha256": sha(BASE / "audio-matched-assets-v78/receipt.json"),
              "primary_context_review_sha256": sha(review_path), "code_sha256": code_bindings(),
              "base_assets": [row for row in old["assets"] if row["role"] in {"weights", "config"}],
              "stock_profiles": len(receipt["assets"]), "source_count": 64, "sources_per_batch": 16, "batches": 4,
              "role_counts": {role: sum(row["role"] == role for row in prompts) for role in
                              ("fusion_fit", "confidence_cal", "selection", "confirmation")},
              "device": "cpu", "dtype": "float32", "threads": 2, "interop_threads": 1,
              "native_sample_rate": 24000, "container": "WAV IEEE float32 with exact readback",
              "speed": 1.0, "max_phonemes_per_chunk": 510, "maximum_chunks_per_source": 4,
              "maximum_native_seconds": 120, "seed_scheme": "SHA256 fixed namespace, stock identity and genuine source identity; first32 bits, plus chunk ordinal. No role/truth as generator input.",
              "chunking": "Pinned upstream English token chunking; complete unchanged source tokens, no skipped spoken tokens, unknown phonemes or truncation. Native chunks concatenated with no added gap or processing.",
              "first_source_exact_repeat_required": True, "generator_started": False,
              "detector_calls_permitted": 0, "calibration_fitting_permitted": False,
              "short_output_policy": "Preserve native audio and report four-second eligibility; no padding, repetition, speed retry, prompt replacement or detector-based source selection.",
              "scope": "Source generation only. Matched intended text is not independently verified spoken-word alignment, quality or detection accuracy. Historical failures and original source roles remain preserved; corrected v75 roles are authoritative."}
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    print(json.dumps({"sources": 64, "profiles": 16, "batches": 4, "new_generator_forwards": 0,
                      "freeze_sha256": sha(PUBLIC)}), flush=True)


def profile_pass(path):
    value = read(path)
    need(value["child_exit_code"] == 0 and value["guard"] is None and value["sample_errors"] == 0
         and not value["resource_limit_observations"], "Preceding bounded stage failed")


def admission():
    need(Path(sys.prefix) == RUNTIME and platform.python_version() == "3.12.6", "Exact dedicated runtime required")
    qualified.admission()
    lock, start = read(TRIAL / "freeze.json"), read(TRIAL / "execution-start.json")
    need(sha(PUBLIC) == sha(TRIAL / "freeze.json") == start["freeze_sha256"]
         and start["remote_verified_before_preprocessing_or_generation"] is True, "Published exact generation freeze required")
    need(lock["code_sha256"] == code_bindings() and sha(MANIFEST) == lock["source_manifest_sha256"] == MANIFEST_SHA
         and sha(TRIAL / "private-plan.json") == lock["private_prompt_plan_sha256"]
         and sha(TRIAL / "private-paths.json") == lock["private_paths_sha256"]
         and sha(BASE / "audio-matched-assets-v78/receipt.json") == lock["stock_profile_receipt_sha256"],
         "Frozen generator, source, prompt or profile binding changed")
    paths_and_assets()
    profile_pass(TRIAL.parent / (TRIAL.name + "-preparation-profile-a02/profile.json"))
    return lock


def offline():
    attempts = []

    def audit(event, arguments):
        if event == "socket.connect":
            attempts.append(event)
            raise RuntimeError("Offline source generation refused network access")

    sys.addaudithook(audit)
    return attempts


def preflight():
    admission()
    attempts = offline()
    from kokoro import KPipeline

    origins = {name: qualified.checked_module_origin(__import__(name)) for name in
               ("torch", "numpy", "kokoro", "misaki", "transformers", "en_core_web_sm")}
    pipelines = {language: KPipeline(lang_code=language, repo_id="hexgrad/Kokoro-82M", model=False) for language in ("a", "b")}
    paths = read(TRIAL / "private-paths.json")
    vocab = read(Path(paths["assets_root"]) / "config.json")["vocab"]
    planned, total_chunks = read(TRIAL / "private-plan.json"), 0
    with (TRIAL / "phoneme-plan.jsonl").open("x") as stream:
        for prompt in planned:
            chunks = complete_chunks(prompt["transcript"], pipelines[prompt["language"]], vocab)
            total_chunks += len(chunks)
            stream.write(json.dumps({"index": prompt["index"], "chunks": chunks}, ensure_ascii=False) + "\n")
            stream.flush()
    need(not attempts, "Preprocessing attempted network access")
    write(TRIAL / "preflight.json", {"status": "complete", "source_count": len(planned), "phoneme_chunks": total_chunks,
                                    "phoneme_plan_sha256": sha(TRIAL / "phoneme-plan.jsonl"),
                                    "g2p_calls": len(planned), "generator_forwards": 0, "detector_calls": 0,
                                    "network_attempts": len(attempts), "all_source_tokens_and_chunks_checked": True,
                                    "qualified_module_origins": origins,
                                    "scope": "Pinned offline G2P/source-token qualification, not synthesized speech or detector accuracy."})
    print(json.dumps({"preprocessed_sources": len(planned), "phoneme_chunks": total_chunks,
                      "generator_forwards": 0}), flush=True)


def generate(index):
    need(type(index) is int and 0 <= index < 4, "One fixed batch index required")
    lock = admission()
    profile_pass(TRIAL / "preflight-profile/profile.json")
    preflight_result = read(TRIAL / "preflight.json")
    need(preflight_result["status"] == "complete" and preflight_result["source_count"] == 64
         and sha(TRIAL / "phoneme-plan.jsonl") == preflight_result["phoneme_plan_sha256"], "Complete phoneme preflight required")
    for prior in range(index):
        profile_pass(TRIAL / f"batch-{prior:02d}-profile/profile.json")
        need(read(TRIAL / f"batch-{prior:02d}-complete.json")["sources"] == 16, "Every earlier fixed batch must finish")
    attempts = offline()
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KModel

    from scripts.audio_generator_checks_v54 import waveform_summary
    from scripts.audio_generator_checks_v57 import state_fingerprint, strict_loaded_state

    origins = {name: qualified.checked_module_origin(__import__(name)) for name in
               ("torch", "numpy", "soundfile", "kokoro", "misaki", "transformers", "en_core_web_sm")}
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
            path = folder / f"source-{prompt['index']:02d}.wav"
            sf.write(path, wave, 24000, format="WAV", subtype="FLOAT")
            readback, rate = sf.read(path, dtype="float32")
            need(rate == 24000 and np.array_equal(wave, readback)
                 and hashlib.sha256(np.asarray(readback, "<f4").tobytes()).hexdigest() == summary["float32_sha256"],
                 "Native WAV readback changed generated sample bytes")
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
    parser.add_argument("stage", choices=("prepare", "preflight", "generate"))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    if args.stage == "generate":
        generate(args.index)
    elif args.index is not None:
        parser.error("Only a generation batch takes an index")
    else:
        {"prepare": prepare, "preflight": preflight}[args.stage]()
