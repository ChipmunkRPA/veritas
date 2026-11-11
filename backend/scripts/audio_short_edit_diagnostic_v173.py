"""Score the prepared development-only short speech edit diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import types
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import audio_short_edit_controls as edits
from scripts import grid_audio_repetition_codecs_v170 as repetition
from scripts import grid_audio_splice_probe_v167 as baseline
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = baseline.ROOT, baseline.BASE, baseline.PANEL
TRIAL = BASE / "audio-short-edit-diagnostic-v173"
PUBLIC = PANEL / "audio-short-edit-diagnostic-freeze-v173.json"
MODULE = "scripts.audio_short_edit_diagnostic_v173"
PREPARATION = BASE / "audio-short-edit-materialization-v169c"
SOURCE_FREEZE = PANEL / "audio-short-edit-materialization-freeze-v169c.json"
SOURCE_RESULT = PANEL / "audio-short-edit-materialization-complete-v169c.json"
DEVELOPMENT = baseline.speech.ARCHIVE / PREPARATION.name / "development"
FOREGROUNDS = {"genuine": 14, "clone": 5, "expressive": 12, "stock": 6}
GROUPS = {"genuine": 7, "clone": 5, "expressive": 12, "stock": 6}
FADES = {"untouched": None, "hard_copy": 0, "fade_20_ms": 20, "fade_50_ms": 50}
SUPPORT = {
    "shift_samples": edits.TARGET_SAMPLES[0] - edits.DONOR_SAMPLES[0],
    "output_samples_half_open": edits.TARGET_SAMPLES,
}
CODE = (
    *baseline.CODE,
    "backend/scripts/audio_short_edit_diagnostic_v173.py",
    "backend/tests/test_audio_short_edit_diagnostic_v173.py",
    "docs/audio-short-edit-diagnostic-v173-method.md",
    "backend/scripts/audio_short_edit_controls.py",
    "backend/scripts/grid_audio_repetition_codecs_v170.py",
    "backend/scripts/audio_repetition_similarity.py",
    "backend/app/detectors/audio/decode.py",
    "backend/app/third_party/kaldi/fbank.py",
    "backend/scripts/local_probe_limits_v24.py",
)
read, write, sha, require = baseline.read, baseline.write, baseline.sha, baseline.require


def verified_preparation():
    stage = PREPARATION / "bind-00"
    result = baseline.speech.qualification_profile(
        stage, 0, {}, "audio_short_edit_materialize_v169c.py", command="bind", batch=0
    )
    profile = read(stage / "profile.json")
    published = read(SOURCE_RESULT)
    require(
        published
        == {
            **result,
            "resource_profile_sha256": sha(stage / "profile.json"),
            "max_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
            "duration_s": profile["duration_s"],
        }
        and result["schema"] == "chipmunk.audio-short-edit-materialization-result/169c"
        and result["freeze_sha256"] == sha(SOURCE_FREEZE) == sha(PREPARATION / "freeze.json")
        and result["source_samples"] == edits.SAMPLES
        and result["sample_rate"] == edits.SAMPLE_RATE
        and result["holdout_scored"] is False
        and profile["sampled_tree_rss_lower_bound_bytes"] <= 1536 * 1024**2,
        "Exact successful v169c materialization receipt required",
    )
    development = result["roles"]["development"]
    require(
        development["foregrounds"] == {lane: {"qualified": count} for lane, count in FOREGROUNDS.items()},
        "Every planned development source must have qualified",
    )
    return development["private_manifest_sha256"]


def development_wav(row):
    slot = row["slot"]
    require(re.fullmatch(r"[A-Za-z0-9_-]+", slot) is not None, "Exact safe development slot required")
    path = DEVELOPMENT / f"{slot}.wav"
    require(Path(row["file"]) == path, "Only the direct admitted development WAV is permitted")
    return path


def development_sources(manifest_sha256):
    path = DEVELOPMENT / "private-manifest.json"
    require(
        DEVELOPMENT.resolve(strict=True) == DEVELOPMENT and path.is_file() and not path.is_symlink(),
        "Only the exact development directory and its private manifest are permitted",
    )
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == manifest_sha256, "Development manifest bytes differ")
    body = json.loads(payload)
    require(body["role"] == "development" and body["scored"] is False, "Exact development role required")
    rows = body["foregrounds"]
    require(
        Counter(row["lane"] for row in rows) == FOREGROUNDS
        and len({row["slot"] for row in rows}) == len({row["wav_sha256"] for row in rows}) == len(rows)
        and all(row["status"] == "qualified" for row in rows)
        and {lane: len({row["group_id"] for row in rows if row["lane"] == lane}) for lane in GROUPS}
        == GROUPS,
        "Exact unique qualified development foreground roster required",
    )
    for row in rows:
        development_wav(row)
    return rows


def read_development_pcm(row):
    from app.detectors.audio.decode import read_wav

    path = development_wav(row)
    require(
        DEVELOPMENT.resolve(strict=True) == DEVELOPMENT and path.is_file() and not path.is_symlink(),
        "Development WAV cannot resolve outside its exact directory",
    )
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == row["wav_sha256"], "Prepared development WAV bytes differ")
    pcm = read_wav(data)
    expected = {
        "start_sample": 0,
        "end_sample": edits.SAMPLES,
        "source_start_pts": 0,
        "source_end_pts": edits.SAMPLES,
        "time_base": "1/16000",
    }
    require(
        len(pcm) == edits.SAMPLES
        and row["wav_range"] == expected
        and hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest() == row["prefix_pcm_sha256"],
        "Exact qualified development PCM and sample origin required",
    )
    return baseline.speech.pcm_lattice(pcm)


def arms(lane):
    return FADES if lane == "genuine" else ("untouched",)


def cases(sources):
    return [(row["slot"], row["lane"], arm) for row in sources for arm in arms(row["lane"])]


def context():
    manifest = verified_preparation()
    sources = development_sources(manifest)
    route = read(baseline.PUBLIC)
    code = {path: sha(ROOT / path) for path in CODE}
    packages = {key: importlib.metadata.version(key) for key in route["runtime_packages"]}
    require(
        route["local_asset_commitments"] == baseline.ASSETS
        and all(code[path] == digest for path, digest in route["code_sha256"].items())
        and packages == route["runtime_packages"],
        "Exact qualified v167 score implementation, assets and runtime required",
    )
    return {
        "schema": "chipmunk.audio-short-edit-diagnostic-freeze/173",
        "code_sha256": code,
        "preparation_public_result_sha256": sha(SOURCE_RESULT),
        "private_development_manifest_sha256": manifest,
        "voice_route_public_freeze_sha256": sha(baseline.PUBLIC),
        "local_asset_commitments": baseline.ASSETS,
        "runtime_packages": packages,
        "role": "development",
        "foregrounds_by_family": FOREGROUNDS,
        "groups_by_family": GROUPS,
        "genuine_fade_ms": FADES,
        "sample_rate": edits.SAMPLE_RATE,
        "samples_per_case": edits.SAMPLES,
        "planned_cases": len(cases(sources)),
        "holdout_wavs_permitted": False,
        "new_fitting_permitted": False,
    }, sources


def verify():
    lock, sources = context()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == lock, "Exact frozen development probe required")
    return sources


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every prior attempt")
    lock, sources = context()
    for row in sources:
        read_development_pcm(row)
    require(
        all(sha(baseline.WEIGHTS / name) == digest for name, digest in baseline.ASSETS.items()),
        "Pinned model assets changed",
    )
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    print(json.dumps({"status": "frozen", "planned_cases": lock["planned_cases"], "new_neural_calls": 0}))


def construct(raw, arm):
    fade = FADES[arm]
    if fade is None:
        return raw, 0
    result = edits.splice_pcm(raw, fade_ms=fade)
    return result.pcm_s16le, result.changed_samples


def repeat_direction(decision, cues):
    matched = any(cue["status"] == "measured" and cue["spans"] for cue in cues.values())
    return "inconclusive" if decision == "ai" and matched else decision


def assess_case(raw, arm, score):
    try:
        variant, changed = construct(raw, arm)
    except edits.UnsupportedControl as error:
        return {
            "construction": "unsupported",
            "reason": str(error),
            "status": "not_applicable",
            "windows": [],
            "decision": "unavailable",
            "symmetric": "unavailable",
            "repeat_abstention": "unavailable",
            "cues": {},
        }
    require(len(variant) == 2 * edits.SAMPLES, "Three-second sample extent changed")
    pcm = np.frombuffer(variant, dtype="<i2").astype("<f4") / np.float32(32768)
    outcome = score(pcm)
    fade_samples = edits.FADE_SAMPLES.get(FADES[arm])
    cues = {name: repetition.outcome(variant, SUPPORT, fade_samples, name) for name in repetition.CUES}
    return {
        **outcome,
        "construction": "ready",
        "changed_samples": changed,
        "private_pcm_sha256": hashlib.sha256(variant).hexdigest(),
        "repeat_abstention": repeat_direction(outcome["decision"], cues),
        "cues": cues,
    }


def summary(rows, sources):
    require(
        [(row["slot"], row["lane"], row["arm"]) for row in rows] == cases(sources),
        "Every frozen case is required in its exact order",
    )
    require(
        all(row["repeat_abstention"] == repeat_direction(row["decision"], row["cues"]) for row in rows),
        "Repeat comparator directions differ from the private inspection evidence",
    )
    result = {}
    for lane in FOREGROUNDS:
        result[lane] = {}
        for arm in arms(lane):
            subset = [row for row in rows if row["lane"] == lane and row["arm"] == arm]
            result[lane][arm] = {
                "planned": len(subset),
                "construction_statuses": dict(Counter(row["construction"] for row in subset)),
                "route_statuses": dict(Counter(row["status"] for row in subset)),
                "baseline_directions": dict(Counter(row["decision"] for row in subset)),
                "symmetric_band_directions": dict(Counter(row["symmetric"] for row in subset)),
                "repeat_abstention_directions": dict(Counter(row["repeat_abstention"] for row in subset)),
                "band_ai_abstentions": sum(
                    row["decision"] == "ai" and row["symmetric"] == "inconclusive" for row in subset
                ),
                "repeat_ai_abstentions": sum(
                    row["decision"] == "ai" and row["repeat_abstention"] == "inconclusive" for row in subset
                ),
                "cues": {
                    name: {
                        "statuses": dict(
                            Counter(row["cues"].get(name, {}).get("status", "not_run") for row in subset)
                        ),
                        "cases_with_reported_spans": sum(
                            bool(row["cues"].get(name, {}).get("spans")) for row in subset
                        ),
                        "controls_with_matching_copy_support": sum(
                            row["cues"].get(name, {}).get("witnessed_copy") is True for row in subset
                        ),
                    }
                    for name in repetition.CUES
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
    sources = verify()
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

    def score(pcm):
        return baseline.score(pcm, model, activity, Deadline(120))

    handle = model.register_forward_pre_hook(count)
    journal = output / "private-observations.jsonl"
    rows = []
    try:
        with journal.open("x") as stream:
            for source in sources:
                raw = read_development_pcm(source)
                for arm in arms(source["lane"]):
                    before = forwards
                    outcome = assess_case(raw, arm, score)
                    require(
                        forwards - before == 2 * len(outcome["windows"]),
                        "Unexpected raw or band forward count",
                    )
                    row = {"slot": source["slot"], "lane": source["lane"], "arm": arm, **outcome}
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
    finally:
        handle.remove()
    write(
        output / "result.json",
        {
            "schema": "chipmunk.audio-short-edit-diagnostic-result/173",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "development_sources": len(sources),
            "planned_cases": len(rows),
            "new_neural_calls": forwards,
            "families": summary(rows, sources),
            "holdout_wavs_read": 0,
            "holdout_scored": False,
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
    sources = verify()
    stage = TRIAL / "authored"
    journal = stage / "output/private-observations.jsonl"
    rows = baseline.speech.planning.reviewed.lines(journal)
    calls = sum(2 * len(row["windows"]) for row in rows)
    result = baseline.speech.qualification_profile(stage, calls, {}, Path(__file__).name)
    require(
        result["schema"] == "chipmunk.audio-short-edit-diagnostic-result/173"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_observations_sha256"] == sha(journal)
        and result["families"] == summary(rows, sources)
        and result["development_sources"] == len(sources)
        and result["planned_cases"] == len(rows)
        and result["holdout_wavs_read"] == 0
        and result["holdout_scored"] is False
        and result["defaults_changed"] is False,
        "Exact guarded development diagnostic receipt required",
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
        (freeze if args.command == "freeze" else run)()
