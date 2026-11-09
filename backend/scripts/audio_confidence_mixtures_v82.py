"""Prepare the full fixed clean/background panel without model-based selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio, read_wav, wav_bytes
from scripts import audio_complete_sources_v81 as source
from scripts.audio_calibration_roles_v71 import ROLES
from scripts.audio_confirmation_v7 import CONDITIONS
from scripts.audio_expressive_inventory_v66 import sha, write
from scripts.audio_matched_generation_v79 import profile_pass
from scripts.audio_source_qualification_v72 import prefix_hash
from scripts.prepare_audio_confirmation_variants_v7 import mixtures
from scripts.review_audio_sources_v76 import require

ROOT, BASE = source.ROOT, source.BASE
TRIAL = BASE / "audio-confidence-mixtures-v82"
DESTINATION = source.DESTINATION.parent / "mixtures-v82"
MANIFEST = source.TRIAL / "manifest.json"
MANIFEST_SHA = "8b667675a803bd03851c27251e08a63b040b2bd29474462f9f0a4e3af9108ea6"
PUBLIC = ROOT / "docs/evaluation-panels/audio-confidence-mixtures-v82-freeze.json"


def stable(value):
    return hashlib.sha256(("chipmunk-audio-mixtures-v82\0" + value).encode()).hexdigest()


def make_plan(body):
    require(len(body["foregrounds"]) == 464 and len(body["backgrounds"]) == 48
            and all(row["status"] == "ready" for row in body["foregrounds"]),
            "All qualified foreground sources required; no omitted or replacement source")
    jobs = []
    for role in ROLES:
        foregrounds = sorted([row for row in body["foregrounds"] if row["role"] == role],
                             key=lambda row: stable(row["raw_sha256"]))
        pools = {kind: sorted([row for row in body["backgrounds"] if row["role"] == role and row["kind"] == kind],
                              key=lambda row: stable(row["raw_sha256"])) for kind in ("speech", "noise", "music")}
        require(len(foregrounds) == 116 and all(len(pool) == 4 for pool in pools.values()),
                "Complete source counts required in every role")
        counts = Counter()
        for row in foregrounds:
            require(row["label"] in {0, 1} and row["samples"] >= 64000, "Qualified four-second foreground required")
            ordinal = counts[row["label"]]
            counts[row["label"]] += 1
            backgrounds = {}
            for kind, pool in pools.items():
                background = pool[ordinal % len(pool)]
                require(background["samples"] >= 128000 and background["study_group"] != row["study_group"],
                        "Whole separate qualified background source required")
                offset = int(stable(row["raw_sha256"] + "\0" + kind)[:16], 16) % (background["samples"] - 64000 + 1)
                backgrounds[kind] = {"raw_sha256": background["raw_sha256"], "start_sample": offset,
                                     "end_sample": offset + 64000, "group": background["study_group"]}
            jobs.append({"base_id": "foreground:" + row["raw_sha256"], "role": role,
                         "source_sha256": row["raw_sha256"], "group": row["study_group"],
                         "label": row["label"], "kind": row["kind"], "source_interval_samples": [0, 64000],
                         "backgrounds": backgrounds})
        require(counts == {0: 56, 1: 60}, "Both truth classes and every planned generator required")
    return jobs


def implementation():
    names = (
        "backend/scripts/audio_confidence_mixtures_v82.py", "backend/scripts/audio_mixtures_runtime_v82.py",
        "backend/tests/test_audio_confidence_mixtures_v82.py", "docs/audio-confidence-mixtures-v82-protocol.md",
        "backend/scripts/prepare_audio_confirmation_variants_v7.py", "backend/scripts/audio_confirmation_v7.py",
        "backend/scripts/audio_complete_sources_v81.py", "backend/app/detectors/audio/decode.py",
        "backend/app/detectors/video/decode.py", "backend/scripts/audio_tree_sampler_v64.py",
        "backend/scripts/local_probe_limits_v24.py", "backend/scripts/image_regional_continuation_runtime_v41.py",
    )
    return {name: sha(ROOT / name) for name in names}


def prepare():
    require(sha(MANIFEST) == MANIFEST_SHA, "Exact complete source manifest required")
    profile = source.TRIAL.parent / (source.TRIAL.name + "-profile/profile.json")
    profile_pass(profile)
    body = json.loads(MANIFEST.read_text())
    jobs = make_plan(body)
    source.connected_roles([*body["foregrounds"], *body["backgrounds"]])
    require(not TRIAL.exists() and not DESTINATION.exists() and not PUBLIC.exists(), "Never mixed source panel required")
    TRIAL.mkdir()
    DESTINATION.mkdir(parents=True)
    write(TRIAL / "plan.json", jobs)
    frozen = {"schema": "chipmunk.audio-confidence-mixture-freeze/82", "source_manifest_sha256": MANIFEST_SHA,
              "source_profile_sha256": sha(profile), "private_plan_sha256": sha(TRIAL / "plan.json"),
              "code_sha256": implementation(), "roles": list(ROLES), "conditions": list(CONDITIONS),
              "foregrounds": 464, "backgrounds": 48, "planned_variants": 4176, "variants_per_role": 1044,
              "foreground_interval_samples": [0, 64000], "sample_rate": 16000,
              "mixture_rule": "Unchanged v7 clean/background mixture helper: foreground RMS 0.1, requested background SNR, one common peak gain across all nine variants, then normal PCM16 encoding and exact production decode.",
              "background_allocation": "Four within-role sources per kind, ordered by fixed hash; cycle independently within each truth class in fixed foreground hash order. Offset is hash of foreground raw bytes and kind. No detector, VAD or ASR selection.",
              "detector_calls_permitted": 0, "confidence_fitting_permitted": False,
              "scope": "Fixed four-second clean and controlled background variants. Correlated variants/shared backgrounds; no new accuracy, real-world SNR distribution or confidence claim."}
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    print(json.dumps({"planned_variants": 4176, "new_model_calls": 0, "freeze_sha256": sha(PUBLIC)}), flush=True)


def admission():
    lock, start = json.loads((TRIAL / "freeze.json").read_text()), json.loads((TRIAL / "execution-start.json").read_text())
    require(sha(PUBLIC) == sha(TRIAL / "freeze.json") == start["freeze_sha256"]
            and start["remote_verified_before_mixing"] is True, "Published exact mixture freeze required")
    require(sha(MANIFEST) == lock["source_manifest_sha256"] == MANIFEST_SHA
            and sha(TRIAL / "plan.json") == lock["private_plan_sha256"] and implementation() == lock["code_sha256"],
            "Fixed sources, allocation or mixture code changed")
    profile_pass(TRIAL.parent / (TRIAL.name + "-preparation-profile/profile.json"))
    return lock


def generate(index):
    require(type(index) is int and 0 <= index < 4, "One fixed role index required")
    admission()
    for previous in range(index):
        profile_pass(TRIAL / f"role-{previous:02d}-profile/profile.json")
        require(json.loads((TRIAL / f"role-{previous:02d}-complete.json").read_text())["variants"] == 1044,
                "Every preceding role must finish")
    role = ROLES[index]
    body = json.loads(MANIFEST.read_text())
    sources = {row["raw_sha256"]: row for row in [*body["foregrounds"], *body["backgrounds"]]}
    jobs = [row for row in json.loads((TRIAL / "plan.json").read_text()) if row["role"] == role]
    require(len(jobs) == 116, "All fixed role sources required")
    cache = {}
    for row in [row for row in sources.values() if row["role"] == role]:
        audio = decode_audio(Path(row["source_path"]), source_sha256=row["raw_sha256"])
        require(len(audio.pcm) == row["samples"] and audio.mapping["pcm_sha256"] == row["pcm_sha256"],
                "Previously qualified production source changed")
        cache[row["raw_sha256"]] = audio.pcm
    folder = DESTINATION / f"role-{index:02d}/inputs"
    folder.mkdir(parents=True, exist_ok=False)
    destination = TRIAL / f"role-{index:02d}.jsonl"
    completed = 0
    with destination.open("x") as journal:
        for ordinal, job in enumerate(jobs):
            backgrounds = job["backgrounds"]
            views = {kind: cache[row["raw_sha256"]][row["start_sample"]:row["end_sample"]]
                     for kind, row in backgrounds.items()}
            variants, gain = mixtures(cache[job["source_sha256"]][:64000], views)
            clean = None
            for condition in CONDITIONS:
                raw = wav_bytes(variants[condition])
                raw_sha = hashlib.sha256(raw).hexdigest()
                path = folder / (raw_sha + ".wav")
                if path.exists():
                    require(path.read_bytes() == raw, "Existing encoded variant differs")
                else:
                    with path.open("xb") as stream:
                        stream.write(raw)
                audio = decode_audio(path, source_sha256=raw_sha)
                require(len(audio.pcm) == 64000 and np.array_equal(audio.pcm, read_wav(raw)),
                        "Variant production decoder altered exact native PCM16 samples")
                kind = condition.split("_")[0]
                if condition == "clean":
                    clean = audio.pcm
                    measured = None
                else:
                    require(clean is not None, "Clean variant must precede background variants")
                    noise = audio.pcm.astype(np.float64) - clean.astype(np.float64)
                    measured = float(20 * np.log10(np.sqrt(np.mean(clean.astype(np.float64) ** 2))
                                                  / np.sqrt(np.mean(noise ** 2))))
                    require(np.isfinite(measured), "Finite measured production SNR required")
                row = {"id": stable(job["base_id"] + "\0" + condition),
                       **{key: job[key] for key in ("base_id", "role", "label", "kind", "group", "source_sha256", "source_interval_samples")},
                       "condition": condition, "sha256": raw_sha, "pcm_sha256": audio.mapping["pcm_sha256"],
                       "path": str(path.relative_to(DESTINATION)), "production_input": audio.mapping,
                       "samples": 64000, "sample_rate": 16000, "common_peak_gain": gain,
                       "background": backgrounds.get(kind), "requested_snr_db": int(condition.rsplit("_", 1)[1]) if kind != "clean" else None,
                       "measured_snr_db": measured,
                       "background_interval_pcm_sha256": prefix_hash(views[kind], 64000) if kind != "clean" else None}
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                completed += 1
            if (ordinal + 1) % 20 == 0:
                print(json.dumps({"role_index": index, "foregrounds": ordinal + 1, "variants": completed}), flush=True)
    require(completed == 1044, "Every planned variant required")
    write(TRIAL / f"role-{index:02d}-complete.json", {"variants": completed, "journal_sha256": sha(destination)})
    print(json.dumps({"role_index": index, "variants": completed, "detector_calls": 0}), flush=True)


def finish():
    lock = admission()
    rows, stage_hashes = [], {}
    for index in range(4):
        profile = TRIAL / f"role-{index:02d}-profile/profile.json"
        profile_pass(profile)
        path = TRIAL / f"role-{index:02d}.jsonl"
        require(json.loads((TRIAL / f"role-{index:02d}-complete.json").read_text())
                == {"variants": 1044, "journal_sha256": sha(path)}, "All unchanged fixed role journals required")
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
        stage_hashes[str(index)] = {"journal_sha256": sha(path), "profile_sha256": sha(profile)}
    require(len(rows) == len({row["id"] for row in rows}) == 4176, "All unique planned variant identities required")
    expected = [(job, condition) for job in json.loads((TRIAL / "plan.json").read_text()) for condition in CONDITIONS]
    owners = {}
    for row, (job, condition) in zip(rows, expected, strict=True):
        require(row["condition"] == condition and row["id"] == stable(job["base_id"] + "\0" + condition)
                and all(row[name] == job[name] for name in
                        ("base_id", "role", "label", "kind", "group", "source_sha256", "source_interval_samples")),
                "Every planned source, role, truth and condition must remain in exact order")
        path = DESTINATION / row["path"]
        require(path.resolve().is_relative_to(DESTINATION.resolve()) and sha(path) == row["sha256"],
                "Complete variant bytes changed or escaped their output root")
        pcm = read_wav(path.read_bytes())
        require(len(pcm) == 64000 and prefix_hash(pcm, 64000) == row["pcm_sha256"],
                "Complete encoded variant PCM changed")
        for name in ("sha256", "pcm_sha256"):
            value = row[name]
            require(value not in owners or owners[value] == row["role"], "Encoded/PCM variant alias crosses roles")
            owners[value] = row["role"]
    body = {"schema": "chipmunk.audio-confidence-controlled-variants/82", "freeze_sha256": sha(PUBLIC),
            "source_manifest_sha256": MANIFEST_SHA, "source_root": str(DESTINATION), "variants": rows}
    write(TRIAL / "variants.json", body)
    result = {"schema": "chipmunk.audio-confidence-mixture-summary/82", "status": "complete",
              "freeze_sha256": sha(PUBLIC), "variants_manifest_sha256": sha(TRIAL / "variants.json"),
              "source_manifest_sha256": MANIFEST_SHA, "stages_sha256": stage_hashes,
              "variants": len(rows), "foregrounds": len({row["base_id"] for row in rows}),
              "backgrounds": lock["backgrounds"], "conditions": dict(Counter(row["condition"] for row in rows)),
              "role_counts": dict(Counter(row["role"] for row in rows)),
              "truth_counts": dict(Counter(str(row["label"]) for row in rows)),
              "cross_role_byte_or_pcm_aliases": 0, "detector_calls": 0, "confidence_fitting_calls": 0,
              "defaults_changed": False,
              "scope": "Four-second controlled variants with fixed group roles, measured SNR and exact production PCM. Correlated variants/shared backgrounds, historical expressive profiles/music, intended-script control without independent word alignment. No detector performance or calibrated confidence established."}
    write(TRIAL / "summary.json", result)
    write(ROOT / "docs/evaluation-panels/audio-confidence-mixtures-v82-result.json", result)
    print(json.dumps({"variants": len(rows), "detector_calls": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "generate", "finish"))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.index)
    else:
        require(args.index is None, "Only one fixed role-generation stage takes an index")
        {"prepare": prepare, "finish": finish}[args.command]()
