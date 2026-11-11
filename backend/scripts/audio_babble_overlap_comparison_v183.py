"""Compare the frozen voice route with competing-speech abstention on matched babble."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from app.detectors.audio.decode import read_wav
from scripts import audio_matched_babble_v181 as source
from scripts import audio_matched_edit_diagnostic_v174 as route
from scripts import audio_overlap_context_v180 as overlap
from scripts.video_appearance_corrected_v112 import scoped

baseline, runtime = route.baseline, route.runtime
ROOT, BASE, PANEL = route.ROOT, route.BASE, route.PANEL
TRIAL = BASE / "audio-babble-overlap-comparison-v183"
OUTPUT = TRIAL / "authored/output"
PUBLIC = PANEL / "audio-babble-overlap-comparison-freeze-v183.json"
PREPARATION = PANEL / "audio-matched-babble-result-v181.json"
OVERLAP_RESULT = PANEL / "audio-overlap-context-result-v180.json"
MODULE = "scripts.audio_babble_overlap_comparison_v183"
ARMS = ("genuine", "synthetic")
CONDITIONS = ("clean", "snr_0db", "snr_plus5db")
DIRECTIONS = route.DIRECTIONS
PAIR_COUNT, CASE_COUNT, COMPONENTS = 16, 96, 4
RATE, SAMPLES = 16000, 48000
CODE = (
    "backend/scripts/audio_babble_overlap_comparison_v183.py",
    "backend/tests/test_audio_babble_overlap_comparison_v183.py",
    "docs/audio-babble-overlap-comparison-v183-method.md",
)
read, write, sha, require = route.read, route.write, route.sha, route.require


def case_key(row):
    return tuple(row[key] for key in ("case", "pair", "component", "pool", "arm", "condition", "label"))


def validate_cases(rows):
    layout = [(row["pair"], row["arm"], row["condition"]) for row in rows]
    expected = [
        (f"pair-{index:02d}", arm, condition)
        for index in range(PAIR_COUNT)
        for arm in ARMS
        for condition in CONDITIONS
    ]
    require(
        len(rows) == CASE_COUNT
        and layout == expected
        and all(
            row["case"] == f"{row['pair']}-{row['arm']}-{row['condition']}"
            and type(row["label"]) is int
            and row["label"] == int(row["arm"] == "synthetic")
            and row["ai_support_samples"] == ([0, SAMPLES] if row["label"] else None)
            and row["status"] in {"constructed", "unavailable", "failed"}
            and isinstance(row["component"], str)
            and bool(row["component"])
            and re.fullmatch(r"pool-0[0-3]", row["pool"])
            for row in rows
        ),
        "Exact 96 planned development cases and origin labels required",
    )
    by_key = {key: row for key, row in zip(layout, rows, strict=True)}
    component_counts, component_pools = Counter(), {}
    for index in range(PAIR_COUNT):
        pair = f"pair-{index:02d}"
        cases = [by_key[pair, arm, condition] for arm in ARMS for condition in CONDITIONS]
        component, pool = cases[0]["component"], cases[0]["pool"]
        require(
            all((row["component"], row["pool"]) == (component, pool) for row in cases)
            and component_pools.setdefault(component, pool) == pool,
            "A pair or connected component changed babble pool",
        )
        component_counts[component] += 1
        constructed = [row for row in cases if row["status"] == "constructed"]
        require(
            len({row["gain"] for row in constructed}) <= 1
            and all(
                math.isfinite(row["gain"])
                and row["gain"] > 0
                and row["samples"] == SAMPLES
                and row["sample_rate"] == RATE
                and (row["condition"] == "clean") == (row["shared_noise_pcm_sha256"] is None)
                and (row["condition"] == "clean" or row["complete_paired_treatment"] is True)
                for row in constructed
            ),
            "Constructed cases must share gain and the exact planned sample clock",
        )
        for condition in CONDITIONS:
            siblings = [row for row in constructed if row["condition"] == condition]
            require(
                len({row["shared_noise_pcm_sha256"] for row in siblings}) <= 1,
                "Paired foregrounds must share the exact babble residual",
            )
    require(
        sorted(component_counts.values()) == [4] * COMPONENTS
        and len(set(component_pools.values())) == COMPONENTS,
        "Four dependent components and four distinct babble pools required",
    )


def prepared_cases():
    frozen = read(source.PUBLIC)
    public = read(PREPARATION)
    result_path = source.OUTPUT / "result.json"
    journal = source.OUTPUT / "private-cases.jsonl"
    profile_path = source.STAGE / "profile.json"
    resource_path = source.STAGE / "resources.jsonl"
    source_receipt = source.OUTPUT / "private-source-receipts.json"
    result, profile = read(result_path), read(profile_path)
    samples = baseline.speech.planning.reviewed.lines(resource_path)
    rows = baseline.speech.planning.reviewed.lines(journal)
    require(
        frozen == read(source.TRIAL / "freeze.json")
        and frozen["schema"] == "chipmunk.audio-matched-babble-freeze/181"
        and frozen["role"] == "development"
        and frozen["planned_cases"] == CASE_COUNT
        and frozen["planned_pairs"] == PAIR_COUNT
        and frozen["foreground_components"] == COMPONENTS
        and frozen["sample_rate"] == RATE
        and frozen["samples_per_case"] == SAMPLES
        and frozen["holdout_media_opened"] is frozen["holdout_scored"] is False
        and all(sha(ROOT / name) == digest for name, digest in frozen["code_sha256"].items()),
        "Exact v181 development-only case freeze required",
    )
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["error_types"] == profile["resource_limit_observations"] == []
        and profile["command"] == "bind"
        and profile["batch"] == 0
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["resource_journal_sha256"] == sha(resource_path)
        and profile["runtime_sha256"] == sha(Path(source.__file__))
        and samples
        and all(
            not item["violations"]
            and not runtime.violations(item, "bind", initial=i == 0)
            and item["archive_free_disk_bytes"] >= (3 * 1024**3 if i == 0 else 1536 * 1024**2)
            for i, item in enumerate(samples)
        )
        and profile["sampled_tree_rss_lower_bound_bytes"] == max(item["tree_rss_bytes"] for item in samples)
        and samples[-1]["elapsed_s"] <= profile["duration_s"] <= 900,
        "Exact successful v108 data-only preparation guard required",
    )
    require(
        result["schema"] == "chipmunk.audio-matched-babble-result/181"
        and result["status"] == "complete"
        and result["role"] == "development"
        and result["freeze_sha256"] == sha(source.PUBLIC)
        and result["private_plan_sha256"] == frozen["private_plan_sha256"]
        and result["private_case_journal_sha256"] == sha(journal)
        and result["private_source_receipt_sha256"] == sha(source_receipt)
        and result["planned_cases"] == CASE_COUNT
        and result["case_outcomes"] == Counter(row["status"] for row in rows)
        and result["condition_outcomes"]
        == {
            condition: Counter(row["status"] for row in rows if row["condition"] == condition)
            for condition in CONDITIONS
        }
        and result["holdout_media_opened"] is result["holdout_scored"] is False
        and result["new_neural_calls"] == profile["new_neural_calls"] == 0
        and public
        == {
            **result,
            "resource_profile_sha256": sha(profile_path),
            "worker_duration_s": profile["duration_s"],
        },
        "Exact v181 private and public guarded aggregates required",
    )
    validate_cases(rows)
    return rows


def context(root):
    rows = prepared_cases()
    voice = read(route.PUBLIC)
    packages = {name: importlib.metadata.version(name) for name in voice["runtime_packages"]}
    require(
        voice == read(route.TRIAL / "freeze.json")
        and voice["schema"] == "chipmunk.audio-matched-edit-diagnostic-freeze/174"
        and voice["local_asset_commitments"] == baseline.ASSETS
        and voice["serving_original_side_probe_only"] is True
        and packages == voice["runtime_packages"]
        and all(sha(ROOT / name) == digest for name, digest in voice["code_sha256"].items()),
        "Exact pinned v174 serving route required",
    )
    previous = read(OVERLAP_RESULT)
    require(
        overlap.context(root) == read(overlap.TRIAL / "freeze.json") == read(overlap.PUBLIC)
        and previous["schema"] == "chipmunk.audio-overlap-context-result/180"
        and previous["status"] == "complete"
        and previous["freeze_sha256"] == sha(overlap.PUBLIC)
        and previous["holdout_wavs_read"] == 0
        and overlap.RUN_SAMPLES == RATE // 4,
        "Exact v180 CPU segmentation assets, runtime and marker required",
    )
    return {
        "schema": "chipmunk.audio-babble-overlap-comparison-freeze/183",
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "prepared_freeze_sha256": sha(source.PUBLIC),
        "prepared_result_sha256": sha(PREPARATION),
        "private_case_journal_sha256": sha(source.OUTPUT / "private-cases.jsonl"),
        "voice_route_freeze_sha256": sha(route.PUBLIC),
        "voice_assets": baseline.ASSETS,
        "voice_runtime_packages": packages,
        "overlap_freeze_sha256": sha(overlap.PUBLIC),
        "overlap_result_sha256": sha(OVERLAP_RESULT),
        "overlap_assets": overlap.ASSETS,
        "overlap_runtime_wheel_sha256": overlap.WHEEL_SHA,
        "minimum_contiguous_overlap_samples": overlap.RUN_SAMPLES,
        "comparator": "condition_blind_ai_to_inconclusive_on_measured_overlap_marker",
        "planned_cases": CASE_COUNT,
        "planned_pairs": PAIR_COUNT,
        "connected_components": COMPONENTS,
        "conditions": list(CONDITIONS),
        "role": "development",
        "fitting_permitted": False,
        "holdout_access_permitted": False,
    }, rows


def verify(root):
    lock, rows = context(root)
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == lock, "Exact frozen babble comparison required")
    return rows


def read_pcm(row):
    case = row["case"]
    path = source.MEDIA / f"{case}.wav"
    require(
        re.fullmatch(r"pair-\d{2}-(genuine|synthetic)-(clean|snr_0db|snr_plus5db)", case)
        and row["status"] == "constructed"
        and Path(row["file"]) == path
        and source.MEDIA.resolve(strict=True) == source.MEDIA
        and path.is_file()
        and not path.is_symlink(),
        "Only a direct constructed v181 WAV is permitted",
    )
    data = path.read_bytes()
    require(source.digest(data) == row["wav_sha256"], "Prepared WAV changed")
    pcm = read_wav(data)
    raw = baseline.speech.pcm_lattice(pcm)
    require(
        len(pcm) == row["samples"] == SAMPLES
        and row["sample_rate"] == RATE
        and source.digest(raw) == row["pcm_s16le_sha256"]
        and source.digest(pcm.astype("<f4").tobytes()) == row["production_f32_pcm_sha256"],
        "Exact production PCM commitments required",
    )
    return pcm


def freeze(root):
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every prior attempt")
    lock, rows = context(root)
    require(
        all(sha(baseline.WEIGHTS / name) == digest for name, digest in baseline.ASSETS.items()),
        "Pinned voice assets changed",
    )
    overlap.weights()
    verify_mixtures(rows)
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)


def comparator(decision, marker):
    return "inconclusive" if decision == "ai" and marker is True else decision


def verify_mixtures(rows):
    clean = {}
    for row in rows:
        if row["status"] != "constructed":
            continue
        pcm = read_pcm(row)
        samples = (pcm * np.float32(32768)).astype(np.int32)
        key = (row["pair"], row["arm"])
        if row["condition"] == "clean":
            clean[key] = samples
        elif key in clean:
            residual = samples - clean[key]
            require(
                residual.min() >= -32768
                and residual.max() <= 32767
                and source.digest(residual.astype("<i2").tobytes()) == row["shared_noise_pcm_sha256"],
                "Observed production babble residual differs from its commitment",
            )


def direction_metrics(rows, policy, arm):
    planned = len(rows)
    counts = Counter(row[policy] for row in rows)
    metrics = {
        **{direction: counts[direction] for direction in DIRECTIONS},
        "decided": counts["ai"] + counts["original"],
        "ai_flag_rate_over_planned": counts["ai"] / planned,
    }
    if arm == "genuine":
        metrics["false_ai"] = counts["ai"]
    else:
        metrics["missed_ai"] = planned - counts["ai"]
        metrics["wrong_original"] = counts["original"]
    return metrics


def arm_metrics(rows, arm):
    return {
        "planned": len(rows),
        "scored": sum(row["route_status"] == "ok" for row in rows),
        "route_statuses": dict(Counter(row["route_status"] for row in rows)),
        "baseline": direction_metrics(rows, "decision", arm),
        "overlap_abstention": direction_metrics(rows, "overlap_abstention", arm),
        "ai_to_inconclusive": sum(
            row["decision"] == "ai" and row["overlap_abstention"] == "inconclusive" for row in rows
        ),
    }


def overlap_metrics(rows):
    measured = [row["overlap"] for row in rows if row["overlap"] is not None]
    return {
        "planned": len(rows),
        "measured": len(measured),
        "marker": sum(record["diagnostic_context"] for record in measured),
        "median_overlap_fraction": median(
            record["overlap_frames"] / record["eligible_frames"] for record in measured
        )
        if measured
        else None,
        "median_overlap_mass": median(record["mean_overlap_mass"] for record in measured)
        if measured
        else None,
    }


def paired_metrics(genuine, synthetic):
    ordinary = {row["pair"]: row for row in genuine}
    positive = {row["pair"]: row for row in synthetic}
    require(
        len(ordinary) == len(genuine) == len(synthetic) == len(positive)
        and ordinary.keys() == positive.keys(),
        "Matched foreground pairs required",
    )
    pairs = [(ordinary[key], positive[key]) for key in ordinary]
    scored = [(a, b) for a, b in pairs if a["route_status"] == b["route_status"] == "ok"]
    result = {
        "planned_pairs": len(pairs),
        "both_scored": len(scored),
        "median_raw_synthetic_minus_genuine": median(b["raw"] - a["raw"] for a, b in scored)
        if scored
        else None,
    }
    for policy in ("decision", "overlap_abstention"):
        counted = Counter((a[policy] == "ai", b[policy] == "ai") for a, b in pairs)
        result["baseline" if policy == "decision" else policy] = {
            "synthetic_only_ai": counted[False, True],
            "genuine_only_ai": counted[True, False],
            "both_ai": counted[True, True],
            "neither_ai": counted[False, False],
        }
    return result


def clean_transitions(clean, noisy):
    clean_by_pair = {row["pair"]: row for row in clean}
    require(
        clean_by_pair.keys() == {row["pair"] for row in noisy}, "Exact clean-to-noisy arm matches required"
    )
    output = {"planned_pairs": len(noisy)}
    for policy in ("decision", "overlap_abstention"):
        counted = Counter((clean_by_pair[row["pair"]][policy], row[policy]) for row in noisy)
        output["baseline" if policy == "decision" else policy] = [
            {"clean": a, "noisy": b, "count": count} for (a, b), count in sorted(counted.items())
        ]
    return output


def panel_metrics(rows):
    conditions = {
        condition: {
            arm: [row for row in rows if row["arm"] == arm and row["condition"] == condition] for arm in ARMS
        }
        for condition in CONDITIONS
    }
    return {
        "by_condition": {
            condition: {
                "arms": {arm: arm_metrics(arms[arm], arm) for arm in ARMS},
                "paired_foreground_sensitivity": paired_metrics(*[arms[arm] for arm in ARMS]),
            }
            for condition, arms in conditions.items()
        },
        "paired_vs_clean": {
            condition: {arm: clean_transitions(conditions["clean"][arm], arms[arm]) for arm in ARMS}
            for condition, arms in conditions.items()
            if condition != "clean"
        },
        "overlap_only": {
            condition: {arm: overlap_metrics(arms[arm]) for arm in ARMS}
            for condition, arms in conditions.items()
        },
    }


def summary(rows, expected):
    require(
        [case_key(row) for row in rows] == [case_key(row) for row in expected],
        "Every frozen case needs one ordered scoring observation",
    )
    for row, case in zip(rows, expected, strict=True):
        evidence = row["overlap"]
        ready = case["status"] == "constructed"
        require(
            (evidence is not None) == ready
            and row["overlap_forwards"] == int(ready)
            and row["decision"] in DIRECTIONS
            and row["route_status"] in {"ok", "not_applicable", "preparation_unavailable"}
            and (row["route_status"] == "ok") == (row["decision"] != "unavailable")
            and (row["route_status"] == "preparation_unavailable") == (not ready)
            and row["overlap_abstention"]
            == comparator(row["decision"], evidence["diagnostic_context"] if ready else None),
            "Exact independent evidence and one-way abstention required",
        )
        if ready:
            validate_overlap(evidence)
        route.check_forwards(row, row["pella_forwards"])
    components = sorted({row["component"] for row in rows})
    return {
        "combined_panel": panel_metrics(rows),
        "by_connected_component": {
            f"component_{i:02d}": panel_metrics([row for row in rows if row["component"] == component])
            for i, component in enumerate(components, start=1)
        },
    }


def validate_overlap(record):
    spans = record["private_intervals_samples"]
    frames = (SAMPLES - overlap.RECEPTIVE) // overlap.STEP + 1
    longest = max((end - start for start, end in spans), default=0)
    require(
        record["eligible_frames"] == frames
        and all(0 <= start < end <= SAMPLES and (end - start) % overlap.STEP == 0 for start, end in spans)
        and all(a[1] < b[0] for a, b in zip(spans, spans[1:]))
        and sum(end - start for start, end in spans) == record["overlap_frames"] * overlap.STEP
        and record["longest_overlap_samples"] == longest
        and record["diagnostic_context"] is (longest >= overlap.RUN_SAMPLES)
        and 0 <= record["mean_overlap_mass"] <= 1 + 2e-5,
        "Expected source-supported overlap spans and the exact frozen marker",
    )


def load_overlap(root):
    sys.path.insert(0, str(root / "site"))
    try:
        import onnxruntime as ort
    finally:
        sys.path.pop(0)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        overlap.weights(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    metadata = session.get_modelmeta().custom_metadata_map
    require(
        ort.__version__ == "1.23.2"
        and Path(ort.__file__).resolve().is_relative_to(root / "site")
        and session.get_providers() == ["CPUExecutionProvider"]
        and int(metadata["sample_rate"]) == RATE
        and int(metadata["window_size"]) == overlap.WINDOW
        and int(metadata["receptive_field_size"]) == overlap.RECEPTIVE
        and int(metadata["receptive_field_shift"]) == overlap.STEP
        and int(metadata["num_classes"]) == 7
        and int(metadata["num_speakers"]) == 3,
        "Exact pinned CPU overlap model and source timing required",
    )
    return session


def assess(case, score, overlap_score):
    if case["status"] == "constructed":
        pcm = read_pcm(case)
        outcome = score(pcm)
        evidence = overlap_score(pcm)
    else:
        outcome = {
            "route_status": "preparation_unavailable",
            "windows": [],
            "decision": "unavailable",
            "stability_probe": "not_run",
        }
        evidence = None
    return {
        **outcome,
        "overlap": evidence,
        "overlap_forwards": int(evidence is not None),
        "overlap_abstention": comparator(
            outcome["decision"], evidence["diagnostic_context"] if evidence else None
        ),
    }


def worker(root, output):
    stage = TRIAL / "authored"
    require(
        output == OUTPUT
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline guarded CPU worker required",
    )
    selected = verify(root)
    from app.detectors.audio.upgraded import SpeechActivity
    from app.engine.boundary import Deadline
    from scripts.run_audio_specialists_v15 import load_voice

    session = load_overlap(root)
    model = load_voice({"weights": str(baseline.WEIGHTS), "assets": baseline.ASSETS})
    activity = SpeechActivity(baseline.WEIGHTS / "silero_vad.jit")
    output.mkdir(mode=0o700, exist_ok=False)
    pella_calls = 0

    def count(_module, _args):
        nonlocal pella_calls
        pella_calls += 1

    def score(pcm):
        return route.score_voice(pcm, model, activity, Deadline(120))

    def overlap_score(pcm):
        padded = np.pad(pcm, (0, overlap.WINDOW - len(pcm)))[None, None, :]
        [response] = session.run(["y"], {"x": padded})
        return overlap.overlap_record(response, len(pcm))

    rows = []
    journal = output / "private-observations.jsonl"
    handle = model.register_forward_pre_hook(count)
    try:
        with journal.open("x") as stream:
            for case in selected:
                before = pella_calls
                outcome = assess(case, score, overlap_score)
                forwards = pella_calls - before
                route.check_forwards(outcome, forwards)
                row = {
                    **{
                        key: case[key]
                        for key in ("case", "pair", "component", "pool", "arm", "condition", "label")
                    },
                    **outcome,
                    "pella_forwards": forwards,
                }
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
    finally:
        handle.remove()
    raw = sum(len(row["windows"]) for row in rows)
    overlap_calls = sum(row["overlap_forwards"] for row in rows)
    write(
        output / "result.json",
        {
            "schema": "chipmunk.audio-babble-overlap-comparison-result/183",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "planned_cases": len(rows),
            "new_neural_calls": pella_calls + overlap_calls,
            "serving_raw_forwards": raw,
            "serving_stability_forwards": pella_calls - raw,
            "overlap_forwards": overlap_calls,
            "research_comparison": summary(rows, selected),
            "holdout_media_opened": False,
            "holdout_scored": False,
            "defaults_changed": False,
        },
    )


def run(root):
    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(OUTPUT)],
            "Exact monitored worker invocation required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "worker", str(root), str(OUTPUT)], **kwargs)

    proxy = types.SimpleNamespace(
        Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": lambda: verify(root),
            "subprocess": proxy,
            "prerequisites": lambda *_: True,
            "__file__": __file__,
        },
    )("authored")


def readout(root):
    selected = verify(root)
    stage = TRIAL / "authored"
    journal = OUTPUT / "private-observations.jsonl"
    rows = baseline.speech.planning.reviewed.lines(journal)
    aggregate = summary(rows, selected)
    pella_calls = sum(row["pella_forwards"] for row in rows)
    raw = sum(len(row["windows"]) for row in rows)
    overlap_calls = sum(row["overlap_forwards"] for row in rows)
    result = baseline.speech.qualification_profile(
        stage, pella_calls + overlap_calls, {}, Path(__file__).name
    )
    require(
        result["schema"] == "chipmunk.audio-babble-overlap-comparison-result/183"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_observations_sha256"] == sha(journal)
        and result["planned_cases"] == len(rows) == CASE_COUNT
        and result["serving_raw_forwards"] == raw
        and result["serving_stability_forwards"] == pella_calls - raw
        and result["overlap_forwards"] == overlap_calls
        and result["research_comparison"] == aggregate
        and result["holdout_media_opened"] is result["holdout_scored"] is result["defaults_changed"] is False,
        "Exact guarded matched-overlap comparison receipt required",
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
    parser.add_argument("command", choices=("freeze", "run", "worker", "readout"))
    parser.add_argument("runtime_root", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    require(
        (args.command == "worker") == (args.output is not None),
        "Only the guarded worker takes an output path",
    )
    root = args.runtime_root.resolve(strict=True)
    if args.command == "worker":
        worker(root, args.output)
    elif args.command == "readout":
        print(json.dumps(readout(root), indent=2, allow_nan=False))
    else:
        {"freeze": freeze, "run": run}[args.command](root)
