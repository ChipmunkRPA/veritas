"""Qualify all matched native outputs and preserve their production decoding lineage."""
from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from app.detectors.audio.decode import decode_audio
from scripts import audio_matched_generation_v80 as generation
from scripts.audio_calibration_roles_v71 import ROLES
from scripts.audio_expressive_inventory_v66 import sha, write
from scripts.audio_source_qualification_v72 import eligibility, prefix_hash
from scripts.review_audio_sources_v76 import grouped, require

ROOT, BASE = generation.ROOT, generation.BASE
TRIAL = BASE / "audio-complete-sources-v81"
DESTINATION = generation.OUTPUT.parent / "qualified-v81"
HISTORY = BASE / "audio-history-v58/catalogue.json"
HISTORY_SHA = "530df71b6f6443a71233bb56ccd5ed9743f4874d34fccc50a7172f2bf050d2c1"


def float_wav(wave):
    """Wrap native samples without clipping or integer quantization."""
    require(wave.dtype == np.float32 and wave.ndim == 1 and wave.size > 0
            and bool(np.isfinite(wave).all()), "Complete finite native float32 samples required")
    output = io.BytesIO()
    sf.write(output, wave, 24000, format="WAV", subtype="FLOAT")
    raw = output.getvalue()
    restored, rate = sf.read(io.BytesIO(raw), dtype="float32")
    require(rate == 24000 and restored.dtype == np.float32 and restored.ndim == 1
            and restored.tobytes() == wave.tobytes(), "Float WAV wrapping changed native samples")
    return raw


def connected_roles(rows):
    components = grouped(rows)
    observed_groups = set()
    for indices in components:
        members = [rows[index] for index in indices]
        require(len({row["role"] for row in members}) == 1, "Actual source ancestry crosses roles")
        declared = {row["study_group"] for row in members}
        require(len(declared) == 1, "New sample/context alias merged previously separate source groups")
        identifier = declared.pop()
        require(identifier not in observed_groups, "Declared source group is disconnected")
        observed_groups.add(identifier)
        require({row["foreground"] for row in members} != {True, False},
                "Foreground and background sources share ancestry")
    return len(components)


def qualify():
    require(not TRIAL.exists() and not DESTINATION.exists(), "Never qualified complete source output required")
    source_manifest = generation.MANIFEST
    require(sha(source_manifest) == generation.MANIFEST_SHA, "Exact corrected source roles required")
    original = json.loads(source_manifest.read_text())
    freeze = generation.read(generation.TRIAL / "freeze.json")
    require(sha(generation.PUBLIC) == sha(generation.TRIAL / "freeze.json")
            and freeze["code_sha256"] == generation.bindings(), "Frozen generation code changed")
    require(sha(generation.TRIAL / "private-plan.json") == freeze["private_prompt_plan_sha256"],
            "Frozen matched source plan changed")
    require(sha(HISTORY) == HISTORY_SHA, "Exact historical exclusion catalogue required")
    history = set(generation.read(HISTORY)["keys"])
    prompts = generation.read(generation.TRIAL / "private-plan.json")
    records, stage_bindings, forward_calls = [], {}, 0
    for index in range(4):
        profile = generation.TRIAL / f"batch-{index:02d}-profile/profile.json"
        generation.profile_pass(profile)
        complete_path = generation.TRIAL / f"batch-{index:02d}-complete.json"
        complete = generation.read(complete_path)
        journal = generation.TRIAL / f"batch-{index:02d}.jsonl"
        require(complete["sources"] == 16 and complete["detector_calls"] == 0
                and complete["network_attempts"] == 0 and complete["state_unchanged_after_inference"]
                and sha(journal) == complete["source_journal_sha256"], "Complete unchanged offline generation required")
        batch = [json.loads(line) for line in journal.read_text().splitlines()]
        require([row["index"] for row in batch] == list(range(index * 16, (index + 1) * 16)),
                "Every ordered generated source required once")
        records.extend(batch)
        forward_calls += complete["generator_forwards"]
        stage_bindings[str(index)] = {"profile_sha256": sha(profile), "complete_sha256": sha(complete_path),
                                      "source_journal_sha256": sha(journal)}
    require(len(records) == len(prompts) == 64 and forward_calls == 65 and records[0]["repeat_exact"] is True,
            "All 64 native sources plus one exact repeat required")
    repeat = np.load(generation.OUTPUT / "batch-00/first-source-repeat.npy", allow_pickle=False)
    first = np.load(generation.OUTPUT / "batch-00/source-00.npy", allow_pickle=False)
    require(repeat.dtype == first.dtype == np.float32 and repeat.tobytes() == first.tobytes(),
            "Preserved first-source repeat differs")
    originals = [*original["foregrounds"], *original["backgrounds"]]
    used_bytes, used_pcm = set(), set()
    for row in originals:
        require(sha(Path(row["source_path"])) == row["raw_sha256"], "Previously qualified original bytes changed")
        used_bytes.add(row["raw_sha256"])
        used_pcm.update(row[name] for name in ("pcm_sha256", "first4_pcm_sha256", "prefix_pcm_sha256"))
    TRIAL.mkdir()
    (DESTINATION / "sources").mkdir(parents=True)
    bindings = {"source_manifest_sha256": sha(source_manifest), "generation_freeze_sha256": sha(generation.PUBLIC),
                "generator_stages": stage_bindings, "history_catalogue_sha256": HISTORY_SHA,
                "implementation_sha256": sha(Path(__file__)),
                "protocol_sha256": sha(ROOT / "docs/audio-complete-sources-v81-protocol.md"),
                "decoder_sha256": sha(ROOT / "backend/app/detectors/audio/decode.py")}
    write(TRIAL / "start.json", bindings)
    generated = []
    with (TRIAL / "qualification-journal.jsonl").open("x") as journal:
        for prompt, row in zip(prompts, records, strict=True):
            index = prompt["index"]
            path = generation.OUTPUT / f"batch-{index // 16:02d}/source-{index:02d}.npy"
            require(Path(row["path"]) == path and sha(path) == row["sha256"]
                    and row["index"] == index and row["role"] == prompt["role"]
                    and row["seed"] == prompt["seed"]
                    and row["voice_asset_sha256"] == prompt["voice_asset_sha256"]
                    and row["sample_rate"] == 24000, "Native generation identity/sample clock changed")
            wave = np.load(path, allow_pickle=False)
            require(wave.size == row["samples"] and prefix_hash(wave, len(wave)) == row["float32_sha256"],
                    "Native generated sample payload changed")
            raw = float_wav(wave)
            raw_sha = hashlib.sha256(raw).hexdigest()
            source = DESTINATION / "sources" / (raw_sha + ".wav")
            with source.open("xb") as stream:
                stream.write(raw)
            audio = decode_audio(source, source_sha256=raw_sha)
            full_sha, prefix_sha = audio.mapping["pcm_sha256"], prefix_hash(audio.pcm, 64000)
            reason = eligibility(audio.pcm, 64000, full_sha, prefix_sha, raw_sha,
                                 history, used_bytes, used_pcm)
            require(bool(row["source_four_second_eligible"]) == (wave.size >= 96000),
                    "Recorded native four-second eligibility changed")
            record = {**prompt, "id": f"matched-v80:{index:02d}", "source_id": f"matched-v80:{index:02d}",
                      "kind": "stock", "foreground": True, "source_path": str(source), "source_bytes": len(raw),
                      "raw_sha256": raw_sha, "samples": len(audio.pcm), "pcm_sha256": full_sha,
                      "prefix_pcm_sha256": prefix_sha, "first4_pcm_sha256": prefix_sha,
                      "minimum_samples": 64000, "production_input": audio.mapping,
                      "native_source_path": str(path), "native_source_sha256": row["sha256"],
                      "native_sample_sha256": row["float32_sha256"], "native_samples": int(wave.size),
                      "native_sample_rate": 24000, "native_seconds": wave.size / 24000,
                      "native_out_of_pcm16_range_samples": int((np.abs(wave) > 1).sum()),
                      "status": "ready" if reason is None else "ineligible", "ineligible_reason": reason,
                      "license": "Apache-2.0", "license_scope": "Generator code/weights and stock profile assets; source text remains subject to source permissions."}
            generated.append(record)
            used_bytes.add(raw_sha)
            used_pcm.update({full_sha, prefix_sha})
            journal.write(json.dumps(record, allow_nan=False) + "\n")
            journal.flush()
            if (index + 1) % 16 == 0:
                print(json.dumps({"qualified_generated_sources": index + 1, "planned": 64}), flush=True)
    foregrounds = [*original["foregrounds"], *generated]
    rows = [*foregrounds, *original["backgrounds"]]
    groups = connected_roles(rows)
    require(groups == 157 and len(rows) == 512, "All original planned source components required")
    manifest = {"schema": "chipmunk.audio-confidence-complete-sources/81", **bindings,
                "foregrounds": foregrounds, "backgrounds": original["backgrounds"],
                "pending_matched_generation": [], "detector_calls": 0, "confidence_fitting_calls": 0}
    write(TRIAL / "manifest.json", manifest)
    result = {"schema": "chipmunk.audio-complete-source-summary/81", "status": "complete",
              **bindings, "manifest_sha256": sha(TRIAL / "manifest.json"),
              "qualification_journal_sha256": sha(TRIAL / "qualification-journal.jsonl"),
              "generated_native_sources": 64, "generator_forwards": forward_calls,
              "native_first_source_repeat_exact": True, "original_source_files_reverified": 448,
              "foregrounds_retained": len(foregrounds), "backgrounds_retained": 48,
              "qualified_foregrounds": sum(row["status"] == "ready" for row in foregrounds),
              "ineligible_generated_sources": dict(Counter(row["ineligible_reason"] for row in generated if row["status"] != "ready")),
              "connected_source_components": groups, "cross_role_components": 0,
              "native_duration_s": {"min": min(row["native_seconds"] for row in generated),
                                    "max": max(row["native_seconds"] for row in generated),
                                    "total": sum(row["native_seconds"] for row in generated)},
              "native_out_of_pcm16_range_samples": sum(row["native_out_of_pcm16_range_samples"] for row in generated),
              "roles": {role: {"foregrounds": dict(Counter(row["kind"] for row in foregrounds if row["role"] == role)),
                               "backgrounds": sum(row["role"] == role for row in original["backgrounds"])} for role in ROLES},
              "detector_calls": 0, "confidence_fitting_calls": 0, "defaults_changed": False,
              "scope": "Complete source byte/sample/timeline and connected-role qualification. Four-second foreground study scope. No detector accuracy, listening review, exact spoken-word verification, model-training novelty or calibrated confidence is established. Generated media, transcripts, source identities, voice features and learned artifacts remain private."}
    write(TRIAL / "summary.json", result)
    print(json.dumps({"status": "complete", "foregrounds": len(foregrounds), "backgrounds": 48,
                      "qualified_foregrounds": result["qualified_foregrounds"], "detector_calls": 0}), flush=True)


if __name__ == "__main__":
    qualify()
