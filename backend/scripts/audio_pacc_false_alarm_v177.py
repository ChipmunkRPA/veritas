"""Measure paired clean-to-noisy false alarms on prepared genuine PACC-P speech."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import types
from collections import Counter
from pathlib import Path

from app.detectors.audio.decode import read_wav
from scripts import audio_matched_edit_diagnostic_v174 as route
from scripts import audio_pacc_sample_v176 as source
from scripts.video_appearance_corrected_v112 import scoped

baseline, runtime = route.baseline, route.runtime
ROOT, BASE, PANEL = route.ROOT, route.BASE, route.PANEL
TRIAL = BASE / "audio-pacc-false-alarm-v177"
PUBLIC = PANEL / "audio-pacc-false-alarm-freeze-v177.json"
PREPARATION = PANEL / "audio-pacc-sample-result-v176.json"
MODULE = "scripts.audio_pacc_false_alarm_v177"
INPUTS = sum(source.POOLS.values())
CASE_COUNT = INPUTS * len(source.CONDITIONS)
SOURCE_RANGE = {
    "start_sample": 0,
    "end_sample": source.SAMPLES,
    "source_start_pts": 0,
    "source_end_pts": source.SAMPLES,
    "time_base": f"1/{source.SAMPLE_RATE}",
}
CODE = (
    "backend/scripts/audio_pacc_false_alarm_v177.py",
    "backend/tests/test_audio_pacc_false_alarm_v177.py",
    "docs/audio-pacc-false-alarm-v177-method.md",
)
read, write, sha, require = route.read, route.write, route.sha, route.require


def prepared():
    result = source.readout()
    require(
        read(PREPARATION) == result
        and result["schema"] == "chipmunk.audio-pacc-sample-result/176"
        and result["status"] == "complete"
        and result["source_inputs"] == INPUTS
        and result["new_neural_calls"] == 0,
        "Exact completed guarded PACC-P preparation required",
    )
    plan = read(source.TRIAL / "private-plan.json")
    rows = read(source.OUTPUT / "private-observations.json")
    require(
        len(plan) == len({row["slot"] for row in plan}) == INPUTS
        and Counter(row["pool"] for row in plan) == source.POOLS
        and result["declared_source_contexts"] == {"AMI": 4, "VCTK": 5}
        and all(
            row["family"] == ("AMI" if row["pool"] == "AMI" else "VCTK")
            and row["known_historical_voice"] == (row["family"] == "VCTK")
            for row in plan
        )
        and len(rows) == CASE_COUNT
        and [(row["slot"], row["family"], row["condition"]) for row in rows]
        == [(item["slot"], item["family"], condition) for item in plan for condition in source.CONDITIONS]
        and all(row["status"] == "qualified" and row["source_range"] == SOURCE_RANGE for row in rows),
        "Exact four AMI contexts and five known VCTK utterances on two microphones required",
    )
    return plan, rows


def context():
    plan, rows = prepared()
    serving = read(route.PUBLIC)
    packages = {name: importlib.metadata.version(name) for name in serving["runtime_packages"]}
    require(
        serving == read(route.TRIAL / "freeze.json")
        and serving["schema"] == "chipmunk.audio-matched-edit-diagnostic-freeze/174"
        and serving["local_asset_commitments"] == baseline.ASSETS
        and serving["serving_original_side_probe_only"] is True
        and packages == serving["runtime_packages"]
        and all(sha(ROOT / path) == digest for path, digest in serving["code_sha256"].items()),
        "Exact v174 serving route, local assets and runtime required",
    )
    return (
        {
            "schema": "chipmunk.audio-pacc-false-alarm-freeze/177",
            "code_sha256": {path: sha(ROOT / path) for path in CODE},
            "v176_public_freeze_sha256": sha(source.PUBLIC),
            "v176_public_result_sha256": sha(PREPARATION),
            "v174_public_freeze_sha256": sha(route.PUBLIC),
            "local_asset_commitments": baseline.ASSETS,
            "runtime_packages": packages,
            "planned_cases": CASE_COUNT,
            "planned_pairs": INPUTS * (len(source.CONDITIONS) - 1),
            "planned_source_inputs": source.POOLS,
            "publisher_full_clip_nominal_snr_db": source.CONDITIONS,
            "source_sample_range": SOURCE_RANGE,
            "role": "genuine_external_stress_control",
            "fitting_permitted": False,
            "holdout_access_permitted": False,
        },
        plan,
        rows,
    )


def verify():
    lock, plan, rows = context()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == lock, "Exact frozen false-alarm study required")
    return plan, rows


def read_pcm(row):
    slot, condition = row["slot"], row["condition"]
    require(
        re.fullmatch(r"source-[0-9]{2}", slot)
        and condition in source.CONDITIONS
        and row["status"] == "qualified"
        and row["source_range"] == SOURCE_RANGE,
        "Exact admitted source, condition and production sample clock required",
    )
    media = source.MEDIA
    path = media / f"{slot}-{condition}.wav"
    require(
        media.resolve(strict=True) == media and path.is_file() and not path.is_symlink(),
        "Only the direct saved PACC-P WAV is permitted",
    )
    data = path.read_bytes()
    require(source.digest(data) == row["wav_sha256"], "Prepared PACC-P WAV bytes differ")
    pcm = read_wav(data)
    require(
        len(pcm) == source.SAMPLES and source.digest(pcm.astype("<f4").tobytes()) == row["pcm_sha256"],
        "Prepared PACC-P production PCM differs",
    )
    return pcm


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every prior attempt")
    lock, _, rows = context()
    for row in rows:
        read_pcm(row)
    require(
        all(sha(baseline.WEIGHTS / name) == digest for name, digest in baseline.ASSETS.items()),
        "Pinned model assets changed",
    )
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    print(json.dumps({"status": "frozen", "planned_cases": CASE_COUNT, "new_neural_calls": 0}))


def identity(row):
    return row["slot"], row["family"], row["condition"]


def condition_metrics(rows):
    planned = len(rows)
    decisions = Counter(row["decision"] for row in rows)
    statuses = Counter(row["route_status"] for row in rows)
    decided = decisions["ai"] + decisions["original"]
    return {
        "planned": planned,
        "scored": statuses["ok"],
        "decided": decided,
        "false_ai": decisions["ai"],
        "directions": {direction: decisions[direction] for direction in route.DIRECTIONS},
        "route_statuses": dict(statuses),
        "false_ai_rate": decisions["ai"] / planned,
        "scored_coverage": statuses["ok"] / planned,
        "decided_coverage": decided / planned,
        "inconclusive_rate": decisions["inconclusive"] / planned,
        "unavailable_rate": decisions["unavailable"] / planned,
    }


def transitions(paired, key):
    counted = Counter((clean[key], noisy[key]) for clean, noisy in paired)
    return [
        {"clean": clean, "noisy": noisy, "count": count} for (clean, noisy), count in sorted(counted.items())
    ]


def paired_metrics(clean, noisy):
    require(
        [row["slot"] for row in clean] == [row["slot"] for row in noisy],
        "Exact clean-to-noisy pairs required",
    )
    paired = list(zip(clean, noisy, strict=True))
    planned = len(paired)
    new = sum(a["decision"] != "ai" and b["decision"] == "ai" for a, b in paired)
    resolved = sum(a["decision"] == "ai" and b["decision"] != "ai" for a, b in paired)
    both_scored = sum(a["route_status"] == b["route_status"] == "ok" for a, b in paired)
    return {
        "planned_pairs": planned,
        "both_scored": both_scored,
        "both_scored_coverage": both_scored / planned,
        "new_false_ai": new,
        "resolved_false_ai": resolved,
        "new_false_ai_rate": new / planned,
        "resolved_false_ai_rate": resolved / planned,
        "net_false_ai_rate_change": (new - resolved) / planned,
        "direction_transitions": transitions(paired, "decision"),
        "route_status_transitions": transitions(paired, "route_status"),
    }


def summary(rows, plan, expected):
    require(
        [identity(row) for row in rows] == [identity(row) for row in expected]
        and all(
            row["decision"] in route.DIRECTIONS
            and (
                row["route_status"] == "ok"
                and row["decision"] != "unavailable"
                or row["route_status"] == "not_applicable"
                and row["decision"] == "unavailable"
            )
            for row in rows
        ),
        "Exact ordered serving outcomes for all planned PACC-P cases required",
    )
    by_input = {(row["slot"], row["condition"]): row for row in rows}
    result = {}
    for group in ("AMI", "VCTK", "VCTK_mic1", "VCTK_mic2"):
        selected = [item for item in plan if group in {item["family"], item["pool"]}]
        by_condition = {
            condition: [by_input[item["slot"], condition] for item in selected]
            for condition in source.CONDITIONS
        }
        result[group] = {
            "microphone_inputs": len(selected),
            "declared_source_contexts": len({item["source_context"] for item in selected}),
            "known_historical_voice_inputs": sum(item["known_historical_voice"] for item in selected),
            "by_condition": {
                condition: condition_metrics(views) for condition, views in by_condition.items()
            },
            "paired_vs_clean": {
                condition: paired_metrics(by_condition["clean"], views)
                for condition, views in by_condition.items()
                if condition != "clean"
            },
        }
    return result


def worker(output):
    stage = TRIAL / "authored"
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline guarded worker required",
    )
    plan, expected = verify()
    from app.detectors.audio.upgraded import SpeechActivity
    from app.engine.boundary import Deadline
    from scripts.run_audio_specialists_v15 import load_voice

    output.mkdir(mode=0o700, exist_ok=False)
    model = load_voice({"weights": str(baseline.WEIGHTS), "assets": baseline.ASSETS})
    activity = SpeechActivity(baseline.WEIGHTS / "silero_vad.jit")
    forwards = 0

    def count(_module, _args):
        nonlocal forwards
        forwards += 1

    handle = model.register_forward_pre_hook(count)
    journal = output / "private-observations.jsonl"
    rows = []
    try:
        with journal.open("x") as stream:
            for case in expected:
                pcm = read_pcm(case)
                before = forwards
                outcome = route.score_voice(pcm, model, activity, Deadline(120))
                calls = forwards - before
                route.check_forwards(outcome, calls)
                row = {
                    **{key: case[key] for key in ("slot", "family", "condition")},
                    **outcome,
                    "pella_forwards": calls,
                }
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
    finally:
        handle.remove()
    raw = sum(len(row["windows"]) for row in rows)
    write(
        output / "result.json",
        {
            "schema": "chipmunk.audio-pacc-false-alarm-result/177",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "new_neural_calls": forwards,
            "serving_raw_forwards": raw,
            "serving_stability_forwards": forwards - raw,
            "planned_cases": len(rows),
            "genuine_controls": summary(rows, plan, expected),
            "holdout_wavs_read": 0,
            "defaults_changed": False,
        },
    )


def routed_popen(argv, **kwargs):
    output = TRIAL / "authored/output"
    require(
        argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(output)],
        "Exact resource monitor command required",
    )
    return subprocess.Popen([*argv[:3], MODULE, "worker", str(output)], **kwargs)


def run():
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
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
    plan, expected = verify()
    stage = TRIAL / "authored"
    journal = stage / "output/private-observations.jsonl"
    rows = baseline.speech.planning.reviewed.lines(journal)
    for row in rows:
        route.check_forwards(row, row["pella_forwards"])
    calls = sum(row["pella_forwards"] for row in rows)
    raw = sum(len(row["windows"]) for row in rows)
    result = baseline.speech.qualification_profile(stage, calls, {}, Path(__file__).name)
    require(
        result["schema"] == "chipmunk.audio-pacc-false-alarm-result/177"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_observations_sha256"] == sha(journal)
        and result["serving_raw_forwards"] == raw
        and result["serving_stability_forwards"] == calls - raw
        and result["planned_cases"] == len(rows) == CASE_COUNT
        and result["genuine_controls"] == summary(rows, plan, expected)
        and result["holdout_wavs_read"] == 0
        and result["defaults_changed"] is False,
        "Exact successful guarded PACC-P false-alarm receipt required",
    )
    profile = read(stage / "profile.json")
    return {
        **result,
        "private_resource_profile_sha256": sha(stage / "profile.json"),
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "worker_duration_seconds": profile["duration_s"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "worker", "readout"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require((args.command == "worker") == (args.output is not None), "Only the worker takes an output path")
    if args.command == "worker":
        worker(args.output)
    elif args.command == "readout":
        print(json.dumps(readout(), indent=2, allow_nan=False))
    else:
        {"freeze": freeze, "run": run}[args.command]()
