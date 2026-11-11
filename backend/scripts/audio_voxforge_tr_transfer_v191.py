"""Measure the frozen English-reference voice route on genuine Turkish controls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import re
import subprocess
import sys
import types
import wave
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from scripts import grid_source_speech_v147 as speech
from scripts import video_appearance_runtime_v108 as runtime
from scripts.prepare_audio_specialists_v15 import ASSETS
from scripts.video_appearance_corrected_v112 import scoped
from scripts.video_appearance_native_v108 import BASE, ROOT, read, require, sha, write

PANEL = ROOT / "docs/evaluation-panels"
TRIAL = BASE / "audio-voxforge-tr-transfer-v191"
OUTPUT = TRIAL / "authored/output"
PUBLIC = PANEL / "audio-voxforge-tr-transfer-freeze-v191.json"
MODULE = "scripts.audio_voxforge_tr_transfer_v191"
PREPARATION = BASE / "audio-voxforge-tr-genuine-v192"
PREPARATION_STAGE = PREPARATION / "bind-00"
PREPARATION_PUBLIC = PANEL / "audio-voxforge-tr-genuine-freeze-v192.json"
INTAKE = BASE / "audio-voxforge-tr-genuine-v188"
INTAKE_PUBLIC = PANEL / "audio-voxforge-tr-genuine-freeze-v188.json"
ALANIA_STAGE = BASE / "audio-alania-transfer-v179/authored"
ALANIA_RESULT = PANEL / "audio-alania-transfer-result-v179.json"
ALANIA_FREEZE = PANEL / "audio-alania-transfer-freeze-v179.json"
VOICE_FREEZE = PANEL / "audio-matched-edit-diagnostic-freeze-v174.json"
WEIGHTS = ROOT / "models/weights/audio-upgrade"
RATE, MIN_SAMPLES, MAX_SAMPLES, PLANNED = 16000, 40000, 64000, 12
QUALIFIED = 3
DIRECTIONS = ("ai", "original", "inconclusive", "unavailable")
CODE = (
    "backend/scripts/audio_voxforge_tr_transfer_v191.py",
    "backend/tests/test_audio_voxforge_tr_transfer_v191.py",
    "docs/audio-voxforge-tr-transfer-v191-method.md",
)


def require_model_free():
    require(
        not {"app.core.config", "app.detectors.audio.upgraded", "torch", "transformers", "onnxruntime"}
        & sys.modules.keys(),
        "Fresh model- and settings-free process required before scoring",
    )


def verify_preparation_freeze():
    lock, intake = read(PREPARATION_PUBLIC), read(INTAKE_PUBLIC)
    require(
        lock == read(PREPARATION / "freeze.json")
        and intake == read(INTAKE / "freeze.json")
        and lock["schema"] == "chipmunk.audio-voxforge-tr-genuine-freeze/192"
        and lock["language"] == "tr"
        and lock["origin"] == "genuine"
        and lock["planned_cases"] == intake["planned_contributor_slots"] == PLANNED
        and lock["prior_preparation_freeze_sha256"] == sha(INTAKE_PUBLIC)
        and lock["prior_preparation_result_sha256"] == sha(INTAKE / "bind-00/output/result.json")
        and lock["prior_preparation_profile_sha256"] == sha(INTAKE / "bind-00/profile.json")
        and lock["prior_private_attempts_sha256"] == sha(INTAKE / "bind-00/output/private-attempts.json")
        and lock["private_acquisition_sha256"] == sha(INTAKE / "acquired/private-acquisition.jsonl")
        and lock["holdout_access_permitted"] is lock["model_calls_permitted"] is False
        and intake["production_rate"] == RATE
        and intake["production_sample_count_bounds"] == [MIN_SAMPLES, MAX_SAMPLES]
        and all(sha(ROOT / name) == digest for name, digest in lock["code_sha256"].items()),
        "Exact corrected frozen genuine Turkish preparation required",
    )
    return lock


def verified_alania():
    alania, voice = read(ALANIA_FREEZE), read(VOICE_FREEZE)
    require(
        alania["schema"] == "chipmunk.audio-alania-transfer-freeze/179"
        and alania == read(ALANIA_STAGE.parent / "freeze.json")
        and alania["voice_route_freeze_sha256"] == sha(VOICE_FREEZE)
        and alania["positive_only_language"] == "tr"
        and alania["production_rate"] == RATE
        and alania["production_sample_count_bounds"] == [MIN_SAMPLES, MAX_SAMPLES]
        and alania["local_asset_commitments"] == voice["local_asset_commitments"] == ASSETS
        and voice["serving_original_side_probe_only"] is True
        and all(
            sha(ROOT / name) == digest
            for name, digest in (voice["code_sha256"] | alania["code_sha256"]).items()
        ),
        "Exact previously completed v179/v174 English-reference route required",
    )
    packages = {name: importlib.metadata.version(name) for name in alania["runtime_packages"]}
    require(packages == alania["runtime_packages"], "Exact frozen voice runtime packages required")
    public = read(ALANIA_RESULT)
    result = speech.qualification_profile(
        ALANIA_STAGE, public["new_neural_calls"], {}, "audio_alania_transfer_v179.py"
    )
    profile_path = ALANIA_STAGE / "profile.json"
    profile = read(profile_path)
    require(
        public
        == {
            **result,
            "resource_profile_sha256": sha(profile_path),
            "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
            "worker_duration_seconds": profile["duration_s"],
        }
        and result["schema"] == "chipmunk.audio-alania-transfer-result/179"
        and result["language"] == "tr"
        and result["freeze_sha256"] == sha(ALANIA_FREEZE)
        and result["private_observations_sha256"] == sha(ALANIA_STAGE / "output/private-observations.jsonl")
        and result["all_positive_cases"]["planned_positives"] == alania["planned_cases"] == 24
        and result["holdout_scored"] is result["defaults_changed"] is False,
        "Exact published guarded Alania result required",
    )
    return public, packages


def read_prepared_wave(row):
    samples = row["samples"]
    require(
        row["status"] == "qualified"
        and re.fullmatch(r"genuine-[0-9]{2}", row["slot"])
        and type(samples) is int
        and MIN_SAMPLES <= samples <= MAX_SAMPLES
        and row["sample_rate"] == RATE
        and row["source_range"]
        == {
            "start_sample": 0,
            "end_sample": samples,
            "source_start_pts": 0,
            "source_end_pts": samples,
            "time_base": f"1/{RATE}",
        },
        "Exact admitted genuine Turkish production sample clock required",
    )
    folder = PREPARATION_STAGE / "output/prepared"
    path = folder / f"{row['slot']}.wav"
    require(
        folder.resolve(strict=True) == folder and path.is_file() and not path.is_symlink(),
        "Only the direct qualified v192 WAV is permitted",
    )
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == row["wav_sha256"], "Prepared WAV bytes differ")
    with wave.open(io.BytesIO(data), "rb") as wav:
        require(
            (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes(), wav.getcomptype())
            == (1, 2, RATE, samples, "NONE"),
            "Prepared WAV format or length differs",
        )
        frames = wav.readframes(samples)
    require(len(frames) == 2 * samples, "Prepared WAV has truncated samples")
    pcm = np.frombuffer(frames, dtype="<i2").astype("<f4") / np.float32(32768)
    require(hashlib.sha256(pcm.tobytes()).hexdigest() == row["pcm_sha256"], "Prepared production PCM differs")


def verified_genuine():
    lock = verify_preparation_freeze()
    stage = PREPARATION_STAGE
    output = stage / "output"
    result_path, profile_path = output / "result.json", stage / "profile.json"
    resources = stage / "resources.jsonl"
    cases, attempts = output / "private-cases.json", output / "private-attempts.json"
    result, profile, rows = read(result_path), read(profile_path), read(cases)
    samples = speech.planning.reviewed.lines(resources)
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "bind"
        and profile["batch"] == 0
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["error_types"] == profile["resource_limit_observations"] == []
        and profile["runtime_sha256"]
        == lock["code_sha256"]["backend/scripts/audio_voxforge_tr_genuine_v192.py"]
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["resource_journal_sha256"] == sha(resources)
        and samples
        and all(
            not item["violations"]
            and not runtime.violations(item, "bind", initial=i == 0)
            and item["archive_free_disk_bytes"] >= (3 * 1024**3 if i == 0 else 1536 * 1024**2)
            for i, item in enumerate(samples)
        )
        and profile["sampled_tree_rss_lower_bound_bytes"] == max(item["tree_rss_bytes"] for item in samples)
        and samples[-1]["elapsed_s"] <= profile["duration_s"] <= 900,
        "Exact successful v192 data-only guard required",
    )
    require(
        result["schema"] == "chipmunk.audio-voxforge-tr-genuine-result/192"
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(PREPARATION_PUBLIC)
        and result["private_cases_sha256"] == sha(cases)
        and result["private_attempts_sha256"] == sha(attempts)
        and result["planned_cases"] == len(rows) == PLANNED
        and result["language"] == "tr"
        and result["origin"] == "genuine"
        and result["holdout_media_opened"] is result["holdout_scored"] is False
        and result["new_neural_calls"] == profile["new_neural_calls"] == 0
        and result["case_outcomes"]
        == {"qualified": QUALIFIED, "unavailable": PLANNED - QUALIFIED}
        == Counter(row["status"] for row in rows)
        and result["qualified_licenses"] == {"GPL-3.0-or-later": QUALIFIED}
        and [row["slot"] for row in rows] == [f"genuine-{i:02d}" for i in range(PLANNED)]
        and all(
            row["status"] in {"qualified", "unavailable"} and row["language"] == "tr" and row["label"] == 0
            for row in rows
        ),
        "Exact three qualified and nine unavailable genuine Turkish slots required",
    )
    qualified = [row for row in rows if row["status"] == "qualified"]
    require(
        all(
            row["license"] == "GPL-3.0-or-later"
            and row["stock_gpl_reference"]["sha256"] in lock["stock_gpl_reference_sha256"]
            for row in qualified
        )
        and all(
            len({row[key] for row in qualified}) == QUALIFIED
            for key in ("prompt_sha256", "source_sha256", "pcm_sha256")
        ),
        "Exact separate permission, prompt and waveform receipts required",
    )
    for row in qualified:
        read_prepared_wave(row)
    return rows, {
        "preparation_freeze_sha256": sha(PREPARATION_PUBLIC),
        "preparation_result_sha256": sha(result_path),
        "preparation_profile_sha256": sha(profile_path),
        "private_genuine_cases_sha256": sha(cases),
    }


def preflight():
    require_model_free()
    verify_preparation_freeze()
    verified_alania()
    pending = not (PREPARATION_STAGE / "output/result.json").is_file()
    selected = None if pending else verified_genuine()[0]
    require_model_free()
    return {
        "status": "preparation_pending" if pending else "ready_for_freeze",
        "planned_genuine_slots": PLANNED,
        "qualified_genuine_slots": None if pending else sum(row["status"] == "qualified" for row in selected),
        "published_synthetic_cases": 24,
        "model_calls": 0,
        "settings_imported": False,
    }


def context():
    require_model_free()
    rows, receipts = verified_genuine()
    alania, packages = verified_alania()
    require_model_free()
    return (
        {
            "schema": "chipmunk.audio-voxforge-tr-transfer-freeze/191",
            "code_sha256": {name: sha(ROOT / name) for name in CODE},
            **receipts,
            "synthetic_result_sha256": sha(ALANIA_RESULT),
            "synthetic_freeze_sha256": sha(ALANIA_FREEZE),
            "voice_route_freeze_sha256": sha(VOICE_FREEZE),
            "local_asset_commitments": ASSETS,
            "runtime_packages": packages,
            "planned_genuine_slots": len(rows),
            "qualified_genuine_slots": sum(row["status"] == "qualified" for row in rows),
            "published_synthetic_cases": alania["all_positive_cases"]["planned_positives"],
            "language": "tr",
            "route_reference_language": "en",
            "new_fitting_permitted": False,
            "holdout_access_permitted": False,
        },
        rows,
        alania,
    )


def verify():
    lock, rows, alania = context()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == lock, "Exact frozen Turkish transfer required")
    return rows, alania


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve previous attempts")
    lock, _, _ = context()
    require(
        all(sha(WEIGHTS / name) == digest for name, digest in ASSETS.items()), "Pinned route assets changed"
    )
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    print(json.dumps({"status": "frozen", "planned_cases": lock["planned_genuine_slots"], "model_calls": 0}))


def unavailable():
    return {
        "route_status": "preparation_unavailable",
        "windows": [],
        "decision": "unavailable",
        "stability_probe": "not_run",
    }


def check_case(row, case):
    qualified = case["status"] == "qualified"
    status = row["route_status"]
    scored = status == "ok"
    calls = row["pella_forwards"]
    probe = row["stability_probe"]
    require(
        row["slot"] == case["slot"]
        and status in {"ok", "not_applicable", "preparation_unavailable"}
        and (status == "preparation_unavailable") == (not qualified)
        and row["windows"] == ([[0, case["samples"]]] if scored and qualified else [])
        and row["decision"] in DIRECTIONS
        and (row["decision"] == "unavailable") == (not scored)
        and (
            not scored
            and probe == "not_run"
            and calls == 0
            or scored
            and (
                probe == "not_required_for_decision"
                and calls == 1
                or probe == "computed"
                and calls == 2
                or probe == "unavailable"
                and calls in {1, 2}
            )
        ),
        "Exact serving direction, sample extent and call count required",
    )


def summary(rows, prepared, alania):
    require(len(rows) == len(prepared), "Every planned genuine slot requires an outcome")
    for row, case in zip(rows, prepared, strict=True):
        check_case(row, case)
    qualified = [row for row, case in zip(rows, prepared, strict=True) if case["status"] == "qualified"]
    planned = len(rows)
    counts = Counter(row["decision"] for row in rows)
    qualified_counts = Counter(row["decision"] for row in qualified)
    scored = sum(row["route_status"] == "ok" for row in rows)
    genuine = {
        "planned_slots": planned,
        "qualified_slots": len(qualified),
        "unqualified_source_slots": planned - len(qualified),
        "speech_gate_rejections": sum(row["route_status"] == "not_applicable" for row in rows),
        "scored_slots": scored,
        "directions_over_planned": {direction: counts[direction] for direction in DIRECTIONS},
        "directions_over_qualified": {direction: qualified_counts[direction] for direction in DIRECTIONS},
        "wrong_ai": counts["ai"],
        "wrong_ai_share_of_qualified": counts["ai"] / len(qualified) if qualified else None,
        "scored_coverage_over_planned": scored / planned,
        "scored_coverage_over_qualified": scored / len(qualified) if qualified else None,
    }
    ai = alania["all_positive_cases"]
    positive = {
        "planned_clips": ai["planned_positives"],
        "scored_clips": ai["scored"],
        "directions_over_planned": ai["directions"],
        "missed_ai": ai["missed_ai"],
        "wrong_original": ai["directions"]["original"],
        "by_render_mode": alania["by_render_mode"],
    }
    return {"genuine_voxforge": genuine, "synthetic_alania_prior_result": positive}


def start_scoring(stack, count):
    # These imports transitively initialize app settings. Do this only after
    # the fresh worker has entered its guarded scoring phase.
    from app.detectors.audio.upgraded import SpeechActivity
    from app.engine.boundary import Deadline
    from scripts import audio_alania_transfer_v179 as alania
    from scripts import audio_voxforge_tr_genuine_v192 as source
    from scripts.run_audio_specialists_v15 import load_voice

    model = load_voice({"weights": str(WEIGHTS), "assets": ASSETS})
    activity = SpeechActivity(WEIGHTS / "silero_vad.jit")
    stack.callback(model.register_forward_pre_hook(count).remove)

    def score(case):
        return alania.score_voice(source.read_prepared_pcm(case), model, activity, Deadline(120))

    return score


def worker(output):
    require_model_free()
    stage = TRIAL / "authored"
    require(
        output == OUTPUT
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact fresh offline guarded worker required",
    )
    prepared, alania = verify()
    require_model_free()
    output.mkdir(mode=0o700, exist_ok=False)
    forwards, rows = 0, []

    def count(_module, _args):
        nonlocal forwards
        forwards += 1

    journal = output / "private-observations.jsonl"
    with ExitStack() as stack, journal.open("x") as stream:
        score = None
        for case in prepared:
            before = forwards
            if case["status"] == "qualified":
                if score is None:
                    score = start_scoring(stack, count)
                outcome = score(case)
            else:
                outcome = unavailable()
            row = {"slot": case["slot"], **outcome, "pella_forwards": forwards - before}
            check_case(row, case)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    raw = sum(len(row["windows"]) for row in rows)
    write(
        output / "result.json",
        {
            "schema": "chipmunk.audio-voxforge-tr-transfer-result/191",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "comparison": summary(rows, prepared, alania),
            "new_neural_calls": forwards,
            "serving_raw_forwards": raw,
            "serving_stability_forwards": forwards - raw,
            "holdout_media_opened": False,
            "holdout_scored": False,
            "defaults_changed": False,
        },
    )


def run():
    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(OUTPUT)],
            "Exact guarded scoring invocation required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "worker", str(OUTPUT)], **kwargs)

    proxy = types.SimpleNamespace(
        Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "subprocess": proxy,
            "prerequisites": lambda *_: True,
            "__file__": __file__,
        },
    )("authored")


def readout():
    prepared, alania = verify()
    stage = TRIAL / "authored"
    journal = OUTPUT / "private-observations.jsonl"
    rows = speech.planning.reviewed.lines(journal)
    compared = summary(rows, prepared, alania)
    calls = sum(row["pella_forwards"] for row in rows)
    raw = sum(len(row["windows"]) for row in rows)
    result = speech.qualification_profile(stage, calls, {}, Path(__file__).name)
    require(
        result["schema"] == "chipmunk.audio-voxforge-tr-transfer-result/191"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_observations_sha256"] == sha(journal)
        and result["comparison"] == compared
        and result["serving_raw_forwards"] == raw
        and result["serving_stability_forwards"] == calls - raw
        and result["holdout_media_opened"] is result["holdout_scored"] is result["defaults_changed"] is False,
        "Exact guarded genuine Turkish comparison receipt required",
    )
    profile_path = stage / "profile.json"
    profile = read(profile_path)
    return {
        **result,
        "resource_profile_sha256": sha(profile_path),
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "worker_duration_seconds": profile["duration_s"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "run", "worker", "readout"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require(
        (args.command == "worker") == (args.output is not None),
        "Only the guarded worker takes an output path",
    )
    if args.command == "worker":
        worker(args.output)
    elif args.command in {"preflight", "readout"}:
        print(
            json.dumps(
                {"preflight": preflight, "readout": readout}[args.command](), indent=2, allow_nan=False
            )
        )
    else:
        {"freeze": freeze, "run": run}[args.command]()
