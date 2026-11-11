"""Compare the cached rollback classifier on the consumed v210 river crossover."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import math
import os
import subprocess
import wave
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from scripts import audio_demand_crossover_score_v214 as crossover
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = crossover.ROOT, crossover.BASE, crossover.PANEL
read, write, sha, require = crossover.read, crossover.write, crossover.sha, crossover.require
PILOT = crossover.serving.prior.pilot
TRIAL = BASE / "audio-rollback-crossover-v220"
PUBLIC = PANEL / "audio-rollback-crossover-freeze-v220.json"
PRIOR_RESULT = PANEL / "audio-demand-crossover-score-complete-v214.json"
MANIFEST = ROOT / "backend/app/models.yaml"
MODULE = "scripts.audio_rollback_crossover_v220"
MODEL_ID = "hf.gustking.wav2vec2_xlsr_deepfake"
REPO = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
REVISION = "f7050b586236dc910d1157f430def2d0647b02b4"
CACHE = ROOT / "models/hf/hub/models--Gustking--wav2vec2-large-xlsr-deepfake-audio-classification"
SNAPSHOT = CACHE / "snapshots" / REVISION
ASSETS = {
    "config.json": ("bc86d455d78aa7c256844038e342e2beea9399bf0efc48e1748fdd918ddf3934", 2339),
    "preprocessor_config.json": ("a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335", 212),
    "model.safetensors": ("8bff7244579a5c0f22f9994848c7eee2e038a07358bfebfbe75b38bd11d39b90", 1262859296),
}
CONDITIONS, ORIGINS = crossover.CONDITIONS, crossover.ORIGINS
LABELS = {"0": "real", "1": "fake"}
RAW_POINT = 0.5
RATE, MIN_SAMPLES, MAX_SAMPLES = 16000, 40000, 64000
CASE_KEYS = ("case", "pair", "condition", "origin")
CODE = (
    "backend/scripts/audio_rollback_crossover_v220.py",
    "backend/tests/test_audio_rollback_crossover_v220.py",
    "docs/audio-rollback-crossover-v220-method.md",
    "backend/app/detectors/video/audio.py",
    "backend/app/detectors/video/adapters/audio.py",
    "backend/app/detectors/video/adapters/loaders.py",
    "backend/scripts/video_appearance_runtime_v108.py",
)


def assets():
    manifest = yaml.safe_load(MANIFEST.read_bytes())
    row = next(item for item in manifest["models"] if item["id"] == MODEL_ID)
    source = row["source"]
    require(
        (source["type"], source["repo"], source["revision"]) == ("hf", REPO, REVISION)
        and {item["path"]: (item["sha256"], item["bytes"]) for item in source["files"]} == ASSETS
        and row["licence"]["spdx"] == "Apache-2.0"
        and row["licence"]["class"] == "commercial"
        and SNAPSHOT.is_dir()
        and SNAPSHOT.resolve() == SNAPSHOT,
        "Exact commercial manifest entry and pinned local snapshot required",
    )
    blobs = CACHE / "blobs"
    for name, (checksum, length) in ASSETS.items():
        path = SNAPSHOT / name
        require(
            path.is_file()
            and (not path.is_symlink() or path.resolve().is_relative_to(blobs))
            and path.stat().st_size == length
            and sha(path) == checksum,
            "Exact frozen rollback asset bytes required",
        )
    config, preprocessor = read(SNAPSHOT / "config.json"), read(SNAPSHOT / "preprocessor_config.json")
    require(
        config["id2label"] == LABELS
        and preprocessor["sampling_rate"] == RATE
        and preprocessor["do_normalize"] is preprocessor["return_attention_mask"] is True,
        "Native 16 kHz preprocessing and positive label 'fake' required",
    )
    return {
        "revision": REVISION,
        "weight_license": "Apache-2.0",
        "asset_sha256": {name: digest for name, (digest, _) in ASSETS.items()},
        "positive_index": 1,
        "positive_label": LABELS["1"],
        "raw_comparison_point": RAW_POINT,
    }


def source_context():
    PILOT.require_model_free()
    prepared = crossover.verify()
    batches = [crossover.completed_batch(batch, prepared) for batch in range(len(CONDITIONS))]
    aggregate = scoped(
        crossover.readout,
        {"verify": lambda: prepared, "completed_batch": lambda batch, _prepared: batches[batch]},
    )()
    require(
        aggregate == read(PRIOR_RESULT)
        and aggregate["schema"] == "chipmunk.audio-demand-crossover-score-result/214"
        and aggregate["status"] == "complete",
        "Exact published and guarded v214 aggregate required",
    )
    pairs, bindings = {}, {}
    for batch, (name, (rows, profile, _calls)) in enumerate(zip(CONDITIONS, batches, strict=True)):
        stage = crossover.TRIAL / f"extract-{batch:02d}"
        pairs[name] = list(zip(prepared[batch :: len(CONDITIONS)], rows, strict=True))
        bindings[name] = {
            "profile_sha256": profile,
            "private_journal_sha256": sha(stage / "output/private-observations.jsonl"),
        }
    PILOT.require_model_free()
    return pairs, bindings


def identity():
    pairs, bindings = source_context()
    lock = {
        "schema": "chipmunk.audio-rollback-crossover-freeze/220",
        "v210_freeze_sha256": sha(crossover.PREP_FREEZE),
        "v210_result_sha256": sha(crossover.PREP),
        "v214_freeze_sha256": sha(crossover.PUBLIC),
        "v214_result_sha256": sha(PRIOR_RESULT),
        "baseline_batches": bindings,
        "model_manifest_sha256": sha(MANIFEST),
        "rollback": assets(),
        "runtime_packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "safetensors")
        },
        "runtime_device": "cpu",
        "runtime_dtype": "float32",
        "runtime_torch_threads": 4,
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "conditions": list(CONDITIONS),
        "planned_cases": sum(len(items) for items in pairs.values()),
        "original_gate_only": True,
        "new_calibration_permitted": False,
        "fresh_panel_access_permitted": False,
    }
    PILOT.require_model_free()
    return lock, pairs


def preflight():
    lock, pairs = identity()
    return {
        "status": "ready_for_freeze",
        "planned_cases": lock["planned_cases"],
        "original_gate_admitted": {
            name: sum(row["route_status"] == "ok" for _, row in items) for name, items in pairs.items()
        },
        "model_calls": 0,
    }


def verify():
    lock, pairs = identity()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == lock, "Exact prospective v220 freeze required")
    return pairs


def freeze():
    require(not PUBLIC.exists() and not TRIAL.exists(), "Preserve every prior attempt")
    lock, _ = identity()
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    return {"status": "frozen", "planned_cases": lock["planned_cases"], "model_calls": 0}


def waveform(case):
    body = crossover.waveform(case)
    with wave.open(io.BytesIO(body), "rb") as stream:
        values = np.frombuffer(stream.readframes(case["samples"]), dtype="<i2").astype(np.float32)
    return values / np.float32(32768)


def start(stack, count):
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    from app.detectors.video.adapters.audio import GustkingVoiceDetector
    from app.detectors.video.audio import AudioClassifier

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    extractor = AutoFeatureExtractor.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    model, info = AutoModelForAudioClassification.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
        dtype=torch.float32,
    )
    model = model.to("cpu").eval().requires_grad_(False)
    require(
        all(not info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
        and all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
        and LABELS["1"] == GustkingVoiceDetector.POSITIVE,
        "Exact pretrained CPU float32 classifier and native positive label required",
    )
    predictor = AudioClassifier.from_parts(extractor, model, GustkingVoiceDetector.POSITIVE, "cpu", repo=REPO)
    require(predictor.pos == 1, "Rollback classifier polarity changed")
    stack.callback(model.register_forward_pre_hook(count).remove)

    def score(pcm):
        (raw,) = predictor.predict([pcm])
        return float(raw)

    return score


def row_from(case, baseline, score):
    raw = None
    if baseline["route_status"] == "ok":
        pcm = waveform(case)
        require(
            MIN_SAMPLES <= len(pcm) <= MAX_SAMPLES and baseline["windows"] == [[0, len(pcm)]],
            "Rollback scores only the original admitted complete window",
        )
        raw = score(pcm)
    return {
        **{key: case[key] for key in CASE_KEYS},
        "route_status": baseline["route_status"],
        "recorded_direction": baseline["decision"],
        "windows": baseline["windows"],
        "recorded_raw": baseline.get("raw"),
        "rollback_raw": raw,
    }


def valid(raw):
    return type(raw) is float and math.isfinite(raw) and 0 <= raw <= 1


def check_row(row, case, baseline):
    assessed = baseline["route_status"] == "ok"
    require(
        all(row[key] == case[key] == baseline[key] for key in CASE_KEYS)
        and row["route_status"] == baseline["route_status"]
        and row["recorded_direction"] == baseline["decision"]
        and row["windows"] == baseline["windows"]
        and row["recorded_raw"] == baseline.get("raw")
        and (
            valid(row["recorded_raw"]) and valid(row["rollback_raw"])
            if assessed
            else row["rollback_raw"] is None
        )
        and row["rollback_forwards"] == int(assessed),
        "Exact paired original and rollback outcomes required",
    )


def worker(batch, output):
    pairs = verify()[CONDITIONS[batch]]
    stage = TRIAL / f"extract-{batch:02d}"
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and not (stage / "profile.json").exists()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact guarded offline rollback worker required",
    )
    PILOT.require_model_free()
    output.mkdir(mode=0o700)
    journal = output / "private-observations.jsonl"
    forwards = 0

    def count(_module, _args):
        nonlocal forwards
        forwards += 1

    with ExitStack() as stack, journal.open("x") as stream:
        score = start(stack, count)
        for case, baseline in pairs:
            before = forwards
            row = row_from(case, baseline, score)
            row["rollback_forwards"] = forwards - before
            check_row(row, case, baseline)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    write(
        output / "result.json",
        {
            "status": "complete",
            "batch": batch,
            "condition": CONDITIONS[batch],
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "new_neural_calls": forwards,
        },
    )


def run(batch):
    def launch(argv, **kwargs):
        output = TRIAL / f"extract-{batch:02d}/output"
        require(
            argv[3:] == ["scripts.video_appearance_native_v108", "extract", str(output), str(batch)],
            "Exact v220 owned scoring command required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "_worker", str(batch), str(output)], **kwargs)

    proxy = SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "subprocess": proxy,
            "prerequisites": lambda *_: True,
            "__file__": __file__,
        },
    )("extract", batch)


def movement(before, after):
    return "up" if after > before else "down" if after < before else "tie"


def rank_pairs(pairs):
    counts = Counter(movement(real, synthetic) for real, synthetic in pairs)
    wins, ties, losses = counts["up"], counts["tie"], counts["down"]
    return {
        "comparisons": wins + ties + losses,
        "synthetic_higher": wins,
        "ties": ties,
        "genuine_higher": losses,
        "concordance": (wins + 0.5 * ties) / (wins + ties + losses) if counts else None,
    }


def summarize_condition(rows):
    common = [row for row in rows if row["route_status"] == "ok"]
    genuine, synthetic = ([row for row in common if row["origin"] == origin] for origin in ORIGINS)
    matched = {row["pair"]: row for row in synthetic}
    comparison = {}
    for key in ("recorded_raw", "rollback_raw"):
        comparison[key] = {
            "all_cross_class_rank": rank_pairs((g[key], a[key]) for g in genuine for a in synthetic),
            "same_pair_rank": rank_pairs(
                (g[key], matched[g["pair"]][key]) for g in genuine if g["pair"] in matched
            ),
            "raw_fake_side_at_legacy_point": {
                origin: sum(row[key] >= RAW_POINT for row in group)
                for origin, group in zip(ORIGINS, (genuine, synthetic), strict=True)
            },
        }
    return {
        "planned": len(rows),
        "common_assessed": {origin: sum(row["origin"] == origin for row in common) for origin in ORIGINS},
        "route_outcomes": {
            origin: dict(Counter(row["route_status"] for row in rows if row["origin"] == origin))
            for origin in ORIGINS
        },
        "recorded_directions": {
            origin: dict(Counter(row["recorded_direction"] for row in rows if row["origin"] == origin))
            for origin in ORIGINS
        },
        "comparison": comparison,
    }


def paired_movements(clean, noisy):
    by_key = {(row["pair"], row["origin"]): row for row in clean}
    results = {}
    for origin in ORIGINS:
        rows = [(by_key[row["pair"], origin], row) for row in noisy if row["origin"] == origin]
        pairs = [
            (before, after)
            for before, after in rows
            if before["route_status"] == after["route_status"] == "ok"
        ]
        results[origin] = {
            "planned_pairs": len(rows),
            "both_assessed": len(pairs),
            "score_movements": {
                key: dict(Counter(movement(before[key], after[key]) for before, after in pairs))
                for key in ("recorded_raw", "rollback_raw")
            },
        }
    return results


def readout():
    expected = verify()
    groups, profiles, journals = {}, [], []
    calls = 0
    for batch, name in enumerate(CONDITIONS):
        stage = TRIAL / f"extract-{batch:02d}"
        journal = stage / "output/private-observations.jsonl"
        rows = crossover.serving.records(journal)
        total = sum(row["rollback_forwards"] for row in rows)
        receipt = PILOT.speech.qualification_profile(stage, total, {}, Path(__file__).name, "extract", batch)
        require(
            receipt["batch"] == batch
            and receipt["condition"] == name
            and receipt["private_observations_sha256"] == sha(journal)
            and receipt["freeze_sha256"] == sha(PUBLIC),
            "Exact completed v220 condition receipt required",
        )
        for row, (case, baseline) in zip(rows, expected[name], strict=True):
            check_row(row, case, baseline)
        groups[name] = rows
        profiles.append(sha(stage / "profile.json"))
        journals.append(sha(journal))
        calls += total
    return {
        "schema": "chipmunk.audio-rollback-crossover-result/220",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "profile_sha256": profiles,
        "private_journal_sha256": journals,
        "new_neural_calls": calls,
        "raw_comparison_point": RAW_POINT,
        "condition_results": {name: summarize_condition(rows) for name, rows in groups.items()},
        "movements_from_matched_clean": {
            name: paired_movements(groups[CONDITIONS[0]], groups[name]) for name in CONDITIONS[1:]
        },
        "fresh_panel_scored": False,
        "defaults_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("preflight", "freeze", "readout"):
        commands.add_parser(name)
    for name in ("run", "_worker"):
        action = commands.add_parser(name)
        action.add_argument("batch", type=int, choices=range(len(CONDITIONS)))
        if name == "_worker":
            action.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "_worker":
        worker(args.batch, args.output)
    elif args.action == "run":
        run(args.batch)
    else:
        actions = {"preflight": preflight, "freeze": freeze, "readout": readout}
        print(json.dumps(actions[args.action](), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
