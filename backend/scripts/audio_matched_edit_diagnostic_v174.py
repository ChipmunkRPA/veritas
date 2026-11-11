"""Compare one frozen voice route and repetition abstention on matched development edits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import types
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from scripts import audio_short_edit_cases_v172 as cases
from scripts import audio_short_edit_diagnostic_v173 as prior
from scripts import grid_audio_repetition_codecs_v170 as repetition
from scripts import grid_audio_splice_probe_v167 as baseline
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = baseline.ROOT, baseline.BASE, baseline.PANEL
TRIAL = BASE / "audio-matched-edit-diagnostic-v174"
PUBLIC = PANEL / "audio-matched-edit-diagnostic-freeze-v174.json"
PREPARATION = PANEL / "audio-short-edit-cases-result-v172.json"
DEVELOPMENT = cases.MEDIA / "development"
MODULE = "scripts.audio_matched_edit_diagnostic_v174"
FAMILIES = {"stock": 6, "clone": 5, "expressive": 12}
ARMS = ("genuine_to_genuine", "genuine_to_ai")
DIRECTIONS = ("ai", "original", "inconclusive", "unavailable")
CASE_COUNT = sum(FAMILIES.values()) * len(ARMS) * len(cases.FADES) * len(cases.LEVELS)
CODE = (
    *prior.CODE,
    "backend/app/detectors/video/adapters/common.py",
    "backend/scripts/audio_matched_edit_diagnostic_v174.py",
    "backend/tests/test_audio_matched_edit_diagnostic_v174.py",
    "docs/audio-matched-edit-diagnostic-v174-method.md",
)
read, write, sha, require = baseline.read, baseline.write, baseline.sha, baseline.require


def direct_file(folder, name):
    path = folder / name
    require(
        path.parent == folder
        and folder.resolve(strict=True) == folder
        and path.is_file()
        and not path.is_symlink(),
        "Only a direct development file is permitted",
    )
    return path


def planned_rows(groups):
    for group in groups:
        layouts = [
            (name, fade, condition, level)
            for name, fade in cases.arms(group["family"])
            for condition, level in cases.LEVELS
        ]
        require(
            group["role"] == "development" and len(layouts) == group["planned_cases"],
            "Development layout differs",
        )
        for index, (name, fade, condition, level) in enumerate(layouts):
            yield {
                **group,
                "case": f"{group['comparison_id']}-{index:02d}",
                "arm": name,
                "condition": condition,
                "nominal_snr_db": level,
                "join_ms": fade,
                "label": cases.ai_truth(name, fade),
            }


def select_matched_rows(rows, groups):
    expected = list(planned_rows(groups))
    require(len(rows) == len(expected), "Every prepared development case needs a receipt")
    selected = []
    for row, plan in zip(rows, expected, strict=True):
        require(
            all(row.get(key) == value for key, value in plan.items()),
            "Preparation case differs from its layout",
        )
        if row["family"] == "ai" and row["join_ms"] is not None:
            selected.append(row)
    require(
        len(selected) == CASE_COUNT and len({row["case"] for row in selected}) == CASE_COUNT,
        "Exact matched development case count required",
    )
    for genuine, synthetic in pairs(selected):
        if genuine["status"] == synthetic["status"] == "constructed":
            for key in ("gain", "shared_noise_pcm_sha256", "matched_host_pcm_sha256"):
                require(key in genuine and genuine[key] == synthetic[key], "Matched PCM treatment differs")
            require(
                (genuine["shared_noise_pcm_sha256"] is None) == (genuine["condition"] == "clean"),
                "Clean and mixed residual receipts differ",
            )
    return selected


def prepared_development():
    frozen = read(cases.PUBLIC)
    public = read(PREPARATION)
    require(
        frozen["schema"] == "chipmunk.audio-short-edit-case-freeze/172"
        and frozen["holdout_scored"] is False
        and frozen["model_calls"] == 0
        and frozen["sample_rate"] == cases.RATE
        and frozen["samples_per_case"] == cases.SAMPLES
        and public["schema"] == "chipmunk.audio-short-edit-case-aggregate/172"
        and all(sha(ROOT / path) == digest for path, digest in frozen["code_sha256"].items()),
        "Exact v172 preparation code and public receipts required",
    )
    development = frozen["roles"]["development"]
    plan_path = direct_file(cases.TRIAL / "development", "private-plan.json")
    plan = read(plan_path)
    manifest = plan["source_manifest"]
    groups = plan["comparisons"]
    require(
        sha(plan_path) == development["private_plan_sha256"]
        and plan["role"] == manifest["role"] == "development"
        and manifest["scored"] is False
        and groups == cases.comparisons(manifest)
        and Counter(group["lane"] for group in groups if group["family"] == "ai") == FAMILIES,
        "Exact frozen development plan required",
    )
    stage = cases.stage_folder("development")
    profile_path = direct_file(stage, "profile.json")
    direct_file(stage, "resources.jsonl")
    direct_file(stage / "output", "result.json")
    journal = direct_file(stage / "output", "private-cases.jsonl")
    failures = direct_file(stage / "output", "private-unavailable-inputs.json")
    result = baseline.speech.qualification_profile(
        stage, 0, {}, Path(cases.__file__).name, command="bind", batch=0
    )
    profile = read(profile_path)
    require(
        public["roles"]["development"]
        == {
            **result,
            "resource_profile_sha256": sha(profile_path),
            "worker_duration_s": profile["duration_s"],
        }
        and result["schema"] == "chipmunk.audio-short-edit-case-result/172"
        and result["role"] == "development"
        and result["freeze_sha256"] == sha(cases.PUBLIC)
        and result["private_plan_sha256"] == development["private_plan_sha256"]
        and result["private_case_journal_sha256"] == sha(journal)
        and result["private_unavailable_inputs_sha256"] == sha(failures)
        and result["holdout_scored"] is False,
        "Exact successful development preparation receipt required",
    )
    rows = baseline.speech.planning.reviewed.lines(journal)
    require(
        len(rows) == development["planned_cases"]
        and Counter(row["status"] for row in rows) == result["case_outcomes"],
        "Preparation coverage differs from its receipt",
    )
    return select_matched_rows(rows, groups), {
        "preparation_freeze_sha256": sha(cases.PUBLIC),
        "preparation_development_result_sha256": sha(stage / "output/result.json"),
        "preparation_development_plan_sha256": sha(plan_path),
        "preparation_development_journal_sha256": sha(journal),
        "preparation_development_profile_sha256": sha(profile_path),
    }


def context():
    selected, receipts = prepared_development()
    route = read(baseline.PUBLIC)
    code = {path: sha(ROOT / path) for path in CODE}
    packages = {key: importlib.metadata.version(key) for key in route["runtime_packages"]}
    require(
        route["local_asset_commitments"] == baseline.ASSETS
        and all(code[path] == digest for path, digest in route["code_sha256"].items())
        and packages == route["runtime_packages"],
        "Exact qualified v167 route, calibration, assets and runtime required",
    )
    return {
        "schema": "chipmunk.audio-matched-edit-diagnostic-freeze/174",
        **receipts,
        "code_sha256": code,
        "voice_route_public_freeze_sha256": sha(baseline.PUBLIC),
        "local_asset_commitments": baseline.ASSETS,
        "runtime_packages": packages,
        "role": "development",
        "sample_rate": cases.RATE,
        "samples_per_case": cases.SAMPLES,
        "join_lengths_ms": [fade for fade, _ in cases.FADES],
        "conditions": dict(cases.LEVELS),
        "blocks_by_donor_family": FAMILIES,
        "planned_cases": CASE_COUNT,
        "planned_pairs": CASE_COUNT // 2,
        "serving_original_side_probe_only": True,
        "new_fitting_permitted": False,
        "holdout_access_permitted": False,
    }, selected


def verify():
    frozen, selected = context()
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == frozen, "Exact frozen development comparison required"
    )
    return selected


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every prior attempt")
    frozen, _ = context()
    require(
        all(sha(baseline.WEIGHTS / name) == digest for name, digest in baseline.ASSETS.items()),
        "Pinned model assets changed",
    )
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    print(json.dumps({"status": "frozen", "planned_cases": CASE_COUNT, "new_neural_calls": 0}))


def read_development_pcm(row):
    from app.detectors.audio.decode import read_wav

    name = row["case"]
    require(re.fullmatch(r"comparison-[0-9]{3}-[0-9]{2}", name), "Exact prepared case name required")
    expected = DEVELOPMENT / f"{name}.wav"
    require(
        row["role"] == "development" and row["status"] == "constructed" and Path(row["file"]) == expected,
        "Only an admitted development case WAV is permitted",
    )
    payload = direct_file(DEVELOPMENT, expected.name).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == row["wav_sha256"], "Prepared WAV bytes differ")
    pcm = read_wav(payload)
    require(
        len(pcm) == row["samples"] == cases.SAMPLES
        and row["sample_rate"] == cases.RATE
        and hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest() == row["production_f32_pcm_sha256"],
        "Production PCM differs from its exact preparation receipt",
    )
    raw = baseline.speech.pcm_lattice(pcm)
    require(
        hashlib.sha256(raw).hexdigest() == row["pcm_s16le_sha256"], "Signed16 PCM differs from its receipt"
    )
    return pcm, raw


def score_voice(pcm, model, activity, deadline):
    from app.detectors.audio import calibration
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack

    lease = types.SimpleNamespace(get=lambda _: types.SimpleNamespace(obj=activity))
    outputs = types.SimpleNamespace(audio=AudioTrack("analysed", pcm, len(pcm)))
    context = types.SimpleNamespace(profile="full", lease=lease, outputs=outputs)
    windows = PellaVoiceDetector()._windows(context)
    if not windows:
        return {
            "route_status": "not_applicable",
            "windows": [],
            "decision": "unavailable",
            "stability_probe": "not_run",
        }
    intervals = [
        [round(start * cases.RATE), round(start * cases.RATE) + len(chunk)] for start, chunk in windows
    ]
    require(intervals == [[0, cases.SAMPLES]], "Expected one whole-clip production speech window")
    (raw,) = model.predict([windows[0][1]])
    require(calibration.valid(raw), "Invalid raw voice response")
    result = calibration.evaluate(float(raw), pcm, windows, model, deadline)
    return {
        "route_status": "ok",
        "windows": intervals,
        "decision": result["decision"],
        "raw": float(raw),
        "reference_estimate_ai": result["probability_ai"],
        "stability_probe": result["probe_status"],
        "stability_log_odds_change": result["stability_log_odds_change"],
    }


def assess(row, score):
    if row["status"] != "constructed":
        return {
            "route_status": "not_run",
            "windows": [],
            "decision": "unavailable",
            "stability_probe": "not_run",
            "repeat_abstention": "unavailable",
            "cues": {},
        }
    pcm, raw = read_development_pcm(row)
    result = score(pcm)
    cues = {name: repetition.outcome(raw, None, None, name) for name in repetition.CUES}
    return {**result, "repeat_abstention": prior.repeat_direction(result["decision"], cues), "cues": cues}


def pair_arm(row):
    return row["arm"].rsplit("_", 1)[0]


def pairs(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["comparison_id"], row["join_ms"], row["condition"])
        arm = pair_arm(row)
        require(arm in ARMS and arm not in grouped[key], "Duplicate or unplanned edit arm")
        grouped[key][arm] = row
    for sibling in grouped.values():
        require(set(sibling) == set(ARMS), "One ordinary and one synthetic arm required per pair")
        genuine, synthetic = (sibling[arm] for arm in ARMS)
        require(
            all(genuine[key] == synthetic[key] for key in ("lane", "genuine_host_slot", "noise_slot")),
            "Comparison pair grouping differs",
        )
        yield genuine, synthetic


def arm_metrics(rows, synthetic):
    result = {
        "planned": len(rows),
        "scored": sum(row["route_status"] == "ok" for row in rows),
        "route_statuses": dict(Counter(row["route_status"] for row in rows)),
    }
    for policy in ("decision", "repeat_abstention"):
        counted = Counter(row[policy] for row in rows)
        metrics = {
            **{direction: counted[direction] for direction in DIRECTIONS},
            "decided": counted["ai"] + counted["original"],
        }
        if synthetic:
            metrics["missed_ai"] = len(rows) - counted["ai"]
        else:
            metrics["false_ai"] = counted["ai"]
        result["baseline" if policy == "decision" else policy] = metrics
    result["ai_to_inconclusive"] = sum(
        row["decision"] == "ai" and row["repeat_abstention"] == "inconclusive" for row in rows
    )
    result["cases_with_measured_repetition"] = sum(
        any(cue["status"] == "measured" and cue["spans"] for cue in row["cues"].values()) for row in rows
    )
    return result


def comparison_metrics(matched):
    genuine = [ordinary for ordinary, _ in matched]
    synthetic = [insert for _, insert in matched]
    decisions = {"ai", "original"}
    return {
        "planned_pairs": len(matched),
        "both_scored": sum(a["route_status"] == b["route_status"] == "ok" for a, b in matched),
        "baseline_both_decided": sum(
            a["decision"] in decisions and b["decision"] in decisions for a, b in matched
        ),
        "repeat_both_decided": sum(
            a["repeat_abstention"] in decisions and b["repeat_abstention"] in decisions for a, b in matched
        ),
        "baseline_paired_ai_contrast": sum(
            (b["decision"] == "ai") - (a["decision"] == "ai") for a, b in matched
        ),
        "repeat_paired_ai_contrast": sum(
            (b["repeat_abstention"] == "ai") - (a["repeat_abstention"] == "ai") for a, b in matched
        ),
        "ordinary": arm_metrics(genuine, False),
        "synthetic": arm_metrics(synthetic, True),
    }


def host_balanced(matched):
    groups = defaultdict(list)
    for pair in matched:
        groups[pair[0]["genuine_host_slot"]].append(pair)
    host_metrics = [comparison_metrics(group) for group in groups.values()]
    result = {"hosts": len(groups), "planned_pairs": len(matched)}
    for policy in ("baseline", "repeat_abstention"):
        result[policy] = (
            {
                "ordinary_false_ai_rate": mean(
                    item["ordinary"][policy]["ai"] / item["planned_pairs"] for item in host_metrics
                ),
                "synthetic_ai_recall": mean(
                    item["synthetic"][policy]["ai"] / item["planned_pairs"] for item in host_metrics
                ),
                "ordinary_decided_rate": mean(
                    item["ordinary"][policy]["decided"] / item["planned_pairs"] for item in host_metrics
                ),
                "synthetic_decided_rate": mean(
                    item["synthetic"][policy]["decided"] / item["planned_pairs"] for item in host_metrics
                ),
            }
            if host_metrics
            else None
        )
    return result


def case_key(row):
    return tuple(
        row[key]
        for key in (
            "case",
            "comparison_id",
            "lane",
            "genuine_host_slot",
            "noise_slot",
            "arm",
            "join_ms",
            "condition",
        )
    )


def summary(rows, selected):
    require(
        [case_key(row) for row in rows] == [case_key(row) for row in selected]
        and all(
            row["decision"] in DIRECTIONS
            and row["repeat_abstention"] == prior.repeat_direction(row["decision"], row["cues"])
            for row in rows
        ),
        "Exact selected order and one-way repetition decisions required",
    )
    matched = list(pairs(rows))
    conditions = {
        condition: [pair for pair in matched if pair[0]["condition"] == condition]
        for condition, _ in cases.LEVELS
    }
    strata = {
        lane: {
            condition: {
                str(fade): comparison_metrics(
                    [pair for pair in subset if pair[0]["lane"] == lane and pair[0]["join_ms"] == fade]
                )
                for fade, _ in cases.FADES
            }
            for condition, subset in conditions.items()
        }
        for lane in FAMILIES
    }
    noises = sorted({pair[0]["noise_slot"] for pair in matched if pair[0]["noise_slot"] is not None})
    return {
        "overall": comparison_metrics(matched),
        "by_noise_condition": {
            condition: comparison_metrics(subset) for condition, subset in conditions.items()
        },
        "by_donor_family_noise_condition_join_ms": strata,
        "descriptive_host_balanced": {
            "overall": host_balanced(matched),
            "by_noise_condition": {
                condition: host_balanced(subset) for condition, subset in conditions.items()
            },
        },
        "descriptive_leave_one_noise_parent_out": [
            {
                "omission_index": index,
                "by_noise_condition": {
                    condition: host_balanced([pair for pair in subset if pair[0]["noise_slot"] != noise])
                    for condition, subset in conditions.items()
                    if condition != "clean"
                },
            }
            for index, noise in enumerate(noises, start=1)
        ],
    }


def check_forwards(row, calls):
    probe = row["stability_probe"]
    windows = len(row["windows"])
    expected = {
        "not_run": (0,),
        "not_required_for_decision": (windows,),
        "computed": (windows + 1,),
        "unavailable": (windows, windows + 1),
    }
    require(
        probe in expected and calls in expected[probe] and (windows == 1) == (row["route_status"] == "ok"),
        "Unexpected serving classifier forward count",
    )


def worker(output):
    stage = TRIAL / "authored"
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline guarded worker required",
    )
    selected = verify()
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
        return score_voice(pcm, model, activity, Deadline(120))

    handle = model.register_forward_pre_hook(count)
    journal = output / "private-observations.jsonl"
    rows = []
    try:
        with journal.open("x") as stream:
            for case in selected:
                before = forwards
                outcome = assess(case, score)
                calls = forwards - before
                check_forwards(outcome, calls)
                row = {
                    **{key: value for key, value in case.items() if key in OBSERVATION_FIELDS},
                    **outcome,
                    "pella_forwards": calls,
                }
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
    finally:
        handle.remove()
    write(
        output / "result.json",
        {
            "schema": "chipmunk.audio-matched-edit-diagnostic-result/174",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "new_neural_calls": forwards,
            "serving_raw_forwards": sum(len(row["windows"]) for row in rows),
            "serving_stability_forwards": forwards - sum(len(row["windows"]) for row in rows),
            "comparison": summary(rows, selected),
            "holdout_wavs_read": 0,
            "holdout_scored": False,
            "defaults_changed": False,
        },
    )


OBSERVATION_FIELDS = {
    "case",
    "comparison_id",
    "lane",
    "genuine_host_slot",
    "noise_slot",
    "arm",
    "join_ms",
    "condition",
    "measured_snr_db",
    "status",
    "reason",
}


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
    selected = verify()
    stage = TRIAL / "authored"
    journal = stage / "output/private-observations.jsonl"
    rows = baseline.speech.planning.reviewed.lines(journal)
    for row in rows:
        check_forwards(row, row["pella_forwards"])
    calls = sum(row["pella_forwards"] for row in rows)
    raw = sum(len(row["windows"]) for row in rows)
    result = baseline.speech.qualification_profile(stage, calls, {}, Path(__file__).name)
    require(
        result["schema"] == "chipmunk.audio-matched-edit-diagnostic-result/174"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_observations_sha256"] == sha(journal)
        and result["serving_raw_forwards"] == raw
        and result["serving_stability_forwards"] == calls - raw
        and result["comparison"] == summary(rows, selected)
        and result["holdout_wavs_read"] == 0
        and result["holdout_scored"] is False
        and result["defaults_changed"] is False,
        "Exact guarded matched development diagnostic receipt required",
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
