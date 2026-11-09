"""Prospective role/candidate recipe before source qualification or new generation."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audio_expressive_inventory_v66 import sha, write

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "audio-calibration-roles-v70"
ROLES = ("fusion_fit", "confidence_cal", "selection", "confirmation")
SEED = "chipmunk-independent-audio-confidence-roles-20261109-v70"


def read(path):
    return json.loads(path.read_text())


def stable(value):
    import hashlib

    return hashlib.sha256((SEED + "\0" + value).encode()).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def round_robin(voices, each):
    ordered = sorted(set(voices), key=stable)
    require(len(ordered) >= each * len(ROLES), "Complete prespecified role support required")
    return {voice: ROLES[index % len(ROLES)] for index, voice in enumerate(ordered[:each * len(ROLES)])}


def stock_assignments(entries):
    require(len(entries) == 16 and len({row["canonical_speaker"] for row in entries}) == 16,
            "Sixteen distinct declared new English profiles required")
    buckets = {role: [] for role in ROLES}
    for language in ("b", "a"):
        for entry in sorted((row for row in entries if row["language"] == language), key=lambda r: stable(r["canonical_speaker"])):
            role = min(ROLES, key=lambda r: (len(buckets[r]), ROLES.index(r)))
            buckets[role].append(entry)
    require(all(len(rows) == 4 for rows in buckets.values()), "Four stock profiles per role required")
    require(all(any(row["language"] == "b" for row in rows) for rows in buckets.values()),
            "Every role needs declared British as well as American profiles")
    return [{**row, "role": role} for role, rows in buckets.items() for row in rows]


def assert_voice_roles(groups):
    seen = {}
    for row in groups:
        require(row["role"] in ROLES, "Known study role required")
        for voice in [row["canonical_speaker"], *([row["matched_genuine_voice"]] if row.get("matched_genuine_voice") else [])]:
            require(seen.setdefault(voice, row["role"]) == row["role"], "Known voice/parent crosses source roles")


def choose_scripts(rows, count, occupied):
    chosen = []
    for row in sorted(rows, key=lambda r: stable(r["id"])):
        keys = set(row["text_identity"])
        if keys & occupied:
            continue
        chosen.append(row)
        occupied.update(keys)
        if len(chosen) == count:
            return chosen
    raise ValueError("Insufficient prespecified distinct declared scripts")


def prepare():
    TRIAL.mkdir(exist_ok=False)
    history_path = BASE / "audio-history-v58/catalogue.json"
    history = set(read(history_path)["keys"])
    joint = BASE / "audio-source-inventory-v68"
    summary = read(joint / "summary.json")
    require(sha(joint / "genuine-candidates.json") == summary["genuine_catalogue_sha256"]
            and sha(joint / "clone-candidates.json") == summary["clone_catalogue_sha256"],
            "Completed joint source inventory changed")
    genuine, clones = read(joint / "genuine-candidates.json"), read(joint / "clone-candidates.json")
    expressive_root = BASE / "audio-expressive-inventory-v66"
    expressive_summary = read(expressive_root / "summary.json")
    require(sha(expressive_root / "candidates.json") == expressive_summary["candidate_catalogue_sha256"],
            "Completed expressive source inventory changed")
    expressive = read(expressive_root / "candidates.json")
    groups, by_genuine, by_clone, by_expressive = [], defaultdict(list), defaultdict(list), defaultdict(list)
    for row in genuine:
        if not any(row["novelty"].values()) and 14 <= len(row["transcript"].split()) <= 60:
            by_genuine[row["canonical_speaker"]].append(row)
    genuine_roles = round_robin([voice for voice, rows in by_genuine.items() if len(rows) >= 4], 14)
    occupied = set()
    for voice, role in genuine_roles.items():
        pool = sorted(by_genuine[voice], key=lambda r: stable(r["id"]))
        groups.append({"kind": "genuine", "canonical_speaker": voice, "role": role,
                       "required_sources": 4, "ordered_candidates": pool,
                       "eligibility": "First four distinct byte/PCM/text-qualified original utterances with >=64000 production samples and nonsilent first64000; no model or VAD scores."})
        # Other banks must not select a script from any candidate in this pool.
        occupied.update(key for row in pool for key in row["text_identity"])
    genuine_bank_voices = {row["canonical_speaker"] for row in genuine}
    for row in clones:
        if (row["audio_header_status"] == "ready" and row["seconds"] >= 4 and row["channels"] in (1, 2)
                and not any(row["novelty"].values()) and row["canonical_speaker"] not in genuine_bank_voices):
            by_clone[row["canonical_speaker"]].append(row)
    clone_chosen = []
    rejected_voice_pools = 0
    for voice in sorted(by_clone, key=stable):
        copied = set(occupied)
        try:
            selected = choose_scripts(by_clone[voice], 3, copied)
        except ValueError:
            rejected_voice_pools += 1
            continue
        clone_chosen.append((voice, selected))
        occupied = copied
        if len(clone_chosen) == 40:
            break
    require(len(clone_chosen) == 40, "Forty qualifying new cloned parent voices required")
    for index, (voice, selected) in enumerate(clone_chosen):
        groups.append({"kind": "clone", "canonical_speaker": voice, "role": ROLES[index % 4],
                       "required_sources": 3, "selected_candidates": selected})
    for row in expressive:
        if (not row["novelty"]["recorded_bytes_seen"] and not row["novelty"]["declared_text_seen"]
                and row["duration_estimate_s"] >= 4):
            by_expressive[row["canonical_speaker"]].append(row)
    expressive_roles = round_robin(by_expressive, 6)
    for voice, role in expressive_roles.items():
        selected = choose_scripts(by_expressive[voice], 4, occupied)
        groups.append({"kind": "expressive", "canonical_speaker": voice, "role": role,
                       "required_sources": 4, "selected_candidates": selected,
                       "historical_voice_exposure": "seen; recordings and scripts absent from prior inventory"})
    stock_path = BASE / "audio-stock-voice-preparation-v69/publisher-metadata.json"
    stock = read(stock_path)
    require(stock["sha"] == "f3ff3571791e39611d31c381e3a41a3af07b4987"
            and stock["cardData"]["license"] == "apache-2.0", "Pinned Apache model metadata required")
    profiles = []
    for row in stock["siblings"]:
        if re.fullmatch(r"voices/[ab][fm]_[a-z0-9_]+\.pt", row["rfilename"]):
            voice = "kokoro-82m:" + Path(row["rfilename"]).stem
            if "voice:" + voice not in history:
                profiles.append({"canonical_speaker": voice, "asset": row["rfilename"], "bytes": row["size"],
                                 "sha256": row["lfs"]["sha256"], "language": row["rfilename"][7]})
    stock_groups = stock_assignments(profiles)
    for role in ROLES:
        real = sorted([voice for voice, assigned in genuine_roles.items() if assigned == role], key=stable)
        synthetic = sorted([row for row in stock_groups if row["role"] == role], key=lambda r: stable(r["canonical_speaker"]))
        for row, parent in zip(synthetic, real[:4], strict=True):
            groups.append({**row, "kind": "stock_generation", "required_sources": 4,
                           "matched_genuine_voice": parent,
                           "recipe": "Render all four exact qualified genuine transcripts with this fixed stock profile; paired sources share role/group. No text truncation or resynthesis selected from detector results."})
    speech_pool = defaultdict(list)
    for row in genuine:
        if (row["novelty"]["declared_voice_seen"] and not row["novelty"]["declared_text_seen"]
                and len(row["transcript"].split()) >= 30):
            speech_pool[row["canonical_speaker"]].append(row)
    speech_roles = round_robin(speech_pool, 4)
    backgrounds = [{"kind": "speech", "canonical_speaker": voice, "role": role,
                    "required_sources": 1, "ordered_candidates": sorted(speech_pool[voice], key=lambda r: stable(r["id"])),
                    "historical_voice_exposure": "seen; fresh recording/script; disjoint from all study foreground parents",
                    "eligibility": "First byte/PCM/text-qualified nonsilent original with >=128000 production samples"}
                   for voice, role in speech_roles.items()]
    noise_path = BASE / "audio-v8/source-planning/additional-noise-inventory.json"
    noise = [row for row in read(noise_path)["rows"] if row["source"] == "free-sound"
             and row["ref"]["decoded_samples"] >= 128000
             and "bytes:" + row["ref"]["sha256"] not in history
             and "pcm:" + row["ref"]["decoded_pcm_sha256"] not in history]
    require(len(noise) >= 16, "Sixteen unused public-domain noise originals required")
    for index, row in enumerate(sorted(noise, key=lambda r: stable(r["path"]))[:16]):
        backgrounds.append({"kind": "noise", "role": ROLES[index % 4], "required_sources": 1,
                            "source_id": row["path"], "license": "Public-domain", "selected_candidate": row["ref"]})
    old_plan_path = BASE / "audio-v9/source-plan.json"
    music = [row for row in read(old_plan_path)["backgrounds"] if row["kind"] == "music"]
    require(len(music) == len({row["source_id"] for row in music}) == 16
            and all(row["license"] == "CC-BY-3.0" and row["full_samples"] >= 128000 for row in music),
            "Sixteen distinct recorded permitted music originals required")
    for index, row in enumerate(sorted(music, key=lambda r: stable(r["source_id"]))):
        backgrounds.append({"kind": "music", "role": ROLES[index % 4], "required_sources": 1,
                            "source_id": row["source_id"], "license": row["license"], "selected_candidate": row,
                            "historical_recording_exposure": "seen; role-disjoint within new study; no new-background claim"})
    assert_voice_roles([*groups, *[row for row in backgrounds if row["kind"] == "speech"]])
    inputs = [history_path, joint / "summary.json", joint / "genuine-candidates.json", joint / "clone-candidates.json",
              expressive_root / "summary.json", expressive_root / "candidates.json", stock_path, noise_path, old_plan_path]
    plan = {"schema": "chipmunk.audio-confidence-source-role-recipe/70", "seed": SEED,
            "roles": ROLES, "source_groups": groups, "backgrounds": backgrounds,
            "input_bindings": {str(path.relative_to(ROOT)): sha(path) for path in inputs},
            "code_sha256": sha(Path(__file__)), "minimum_foreground_samples": 64000,
            "minimum_background_samples": 128000, "sample_rate": 16000,
            "source_qualification_policy": "Verify exact selected bytes and production PCM. Preserve all failures and candidate rejection reasons. Duration/nonsilence/integrity and known ancestry only, no annotation, VAD or detector score selection. No silent substitutions for explicitly selected clone/expressive rows. Ordered genuine/speech pools permit deterministic hard eligibility before any model calls.",
            "short_audio_limit": "This source recipe targets four-second confidence first. Native shorter speech remains an explicit separate qualification need; do not pad, repeat or silently claim support.",
            "neural_calls": 0, "roles_frozen_before_new_generation": True}
    write(TRIAL / "plan.json", plan)
    totals = {role: {kind: sum(row["required_sources"] for row in groups if row["role"] == role and row["kind"] == kind)
                     for kind in ("genuine", "clone", "expressive", "stock_generation")} for role in ROLES}
    result = {"schema": "chipmunk.audio-confidence-source-role-summary/70", "roles": totals,
              "planned_foreground_sources": sum(row["required_sources"] for row in groups),
              "planned_actual_source_voices": len(groups), "historically_unseen_foreground_voices": 112,
              "historically_seen_expressive_voices": 24, "genuine_stock_matched_voice_pairs": 16,
              "backgrounds": {role: dict(Counter(row["kind"] for row in backgrounds if row["role"] == role)) for role in ROLES},
              "rejected_clone_voice_pools_before_selection": rejected_voice_pools,
              "recipe_sha256": sha(TRIAL / "plan.json"), "code_sha256": sha(Path(__file__)),
              "production_source_qualification_complete": False, "stock_generation_complete": False,
              "new_model_calls": 0,
              "scope": "Prospective source roles and deterministic candidate eligibility, not qualified source bytes, fitted confidence or model validation. Historical expressive-voice/music exposure is explicit. Exact selected source and prompt freeze remains required before generator/detector calls."}
    write(TRIAL / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    prepare()
