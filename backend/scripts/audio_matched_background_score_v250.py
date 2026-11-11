"""Score the v248 matched panel with the frozen serving route and second opinion."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import audio_fresh_shared_noise_score_v219 as baseline
from scripts import audio_paired_transfer_v230 as candidate
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = baseline.ROOT, baseline.BASE, baseline.PANEL
TRIAL = BASE / "audio-matched-background-score-v250"
BASELINE = TRIAL / "baseline"
CANDIDATE = TRIAL / "candidate"
PUBLIC = PANEL / "audio-matched-background-score-freeze-v250.json"
PREPARATION = PANEL / "audio-matched-background-panel-complete-v248.json"
PREPARATION_FREEZE = PANEL / "audio-matched-background-panel-freeze-v248.json"
PREPARATION_STAGE = BASE / "audio-matched-background-panel-v248/bind-00"
PREPARATION_MODULE = "scripts.audio_matched_background_panel_v248"
PREVIOUS_CANDIDATE_RESULT = PANEL / "audio-paired-transfer-complete-v230.json"
MODULE = "scripts.audio_matched_background_score_v250"
NATIVE = candidate.native
ARMS, CONDITIONS = ("genuine", "alania", "okur"), baseline.CONDITIONS
TRIPLET_IDS = tuple(f"triplet-{index:02d}" for index in range(24))
ROWS_PER_BATCH = len(ARMS) * len(TRIPLET_IDS)
CASE_JOURNAL = PREPARATION_STAGE / "output/private-cases.jsonl"
NATIVE_HASH = "native_float32_input_sha256"
DIRECTIONS = candidate.DIRECTIONS
PROVISIONAL = "machine_screened_provisional"
EXPECTED_V219 = "f7152a254aa0fde752442798d67de5bd4a0eba298fa0adf6df03081a3fcb6793"
EXPECTED_V230 = "bac0b9144a566522ad867f4bac20e3fadd6f52ef0b2a3ba03bad4c8209550e8b"
EXPECTED_V230_RESULT = "fc02578c5f5e3c12ad982c0606bcb540607fffa1def3b8fd72fb0f829b9b516a"
EXPECTED_V248 = "af7b558a456d4bbb0e97dde96bf0fb8a9e402652cd514d1c061d32c99fa9a792"
CODE = (
    "backend/scripts/audio_matched_background_score_v250.py",
    "backend/tests/test_audio_matched_background_score_v250.py",
    "docs/audio-matched-background-score-v250-method.md",
    "backend/scripts/audio_matched_background_panel_v248.py",
    "backend/scripts/audio_fresh_shared_noise_score_v219.py",
    "backend/scripts/audio_paired_transfer_v230.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/video_appearance_runtime_v108.py",
)
read, write, sha, require = baseline.read, baseline.write, baseline.sha, baseline.require
verified_wav = scoped(baseline.serving.waveform, {"INPUT": PREPARATION_STAGE})


def float_input(pcm):
    require(len(pcm) == 2 * NATIVE.SAMPLES, "Exact three-second signed16 extent required")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768)


def waveform_input(case):
    body = verified_wav(case)
    with wave.open(io.BytesIO(body), "rb") as stream:
        values = float_input(stream.readframes(NATIVE.SAMPLES))
    return body, values


def baseline_waveform(case):
    body, values = waveform_input(case)
    require(
        NATIVE.float_sha(values) == case[NATIVE_HASH],
        "Serving WAV must bind the exact frozen native float32 samples",
    )
    return body


def candidate_waveform(case):
    _, values = waveform_input(case)
    require(NATIVE.float_sha(values) == case[NATIVE_HASH], "Candidate input differs from serving samples")
    return values


def expected_cases():
    return [
        (f"{triplet}-{arm}-{condition}", triplet, arm, condition)
        for triplet in TRIPLET_IDS
        for arm in ARMS
        for condition in CONDITIONS
    ]


def reconstructed_preparation():
    # v248 uses the production decoder import in its data-only reconstruction.
    # Keep those application settings out of the original v219 scoring process.
    completed = subprocess.run(
        [sys.executable, "-B", "-m", PREPARATION_MODULE, "readout"],
        cwd=ROOT / "backend",
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "backend"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    require(completed.returncode == 0, "Independent data-only v248 reconstruction did not complete")
    return json.loads(completed.stdout)


def preparation():
    require(sha(PREPARATION) == EXPECTED_V248, "Exact published v248 aggregate required")
    result = reconstructed_preparation()
    frozen = read(PREPARATION_FREEZE)
    require(
        result == read(PREPARATION)
        and result["schema"] == "chipmunk.audio-matched-background-panel-result/248"
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(PREPARATION_FREEZE)
        and result["private_cases_sha256"] == sha(CASE_JOURNAL)
        and result["planned_cases"] == ROWS_PER_BATCH * len(CONDITIONS)
        and frozen["planned_triplets"] == len(TRIPLET_IDS)
        and tuple(frozen["arms"]) == ARMS
        and tuple(frozen["conditions"]) == CONDITIONS
        and result["background_evidence_type"] == PROVISIONAL
        and result["human_listening_reviewed"] is result["holdout_scored"] is False
        and result["new_neural_calls"] == result["detector_calls"] == 0,
        "Exact published and independently reconstructed v248 materialization required",
    )
    cases = baseline.serving.records(CASE_JOURNAL)
    require(
        [tuple(row[key] for key in candidate.KEYS) for row in cases] == expected_cases()
        and all(row["status"] in {"prepared", "unavailable"} for row in cases),
        "Exact ordered 360-row v248 preparation roster required",
    )
    for case in cases:
        if case["status"] == "prepared":
            _, values = waveform_input(case)
            case[NATIVE_HASH] = NATIVE.float_sha(values)
        else:
            case[NATIVE_HASH] = None
    return result, cases


def unchanged_routes():
    require(
        sha(baseline.PUBLIC) == EXPECTED_V219
        and sha(candidate.PUBLIC) == EXPECTED_V230
        and sha(PREVIOUS_CANDIDATE_RESULT) == EXPECTED_V230_RESULT,
        "Exact published previous serving and candidate commitments required",
    )
    historical, _, head, points = candidate.identity()
    require(
        historical == read(candidate.PUBLIC) == read(candidate.TRIAL / "freeze.json"),
        "The frozen v230 head, points and v226 qualification must replay",
    )
    route = baseline.serving.prior
    route.verify()
    assets = route.pilot.ASSETS
    asset_digest = hashlib.sha256(
        json.dumps(assets, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    baseline_lock = read(baseline.PUBLIC)
    require(
        baseline_lock["route_freeze_sha256"] == sha(route.PUBLIC)
        and baseline_lock["asset_set_commitment_sha256"] == asset_digest
        and all(sha(route.pilot.WEIGHTS / name) == digest for name, digest in assets.items()),
        "Exact unchanged serving route and local speech assets required",
    )
    checkpoint = NATIVE.checkpoint()
    require(checkpoint == read(NATIVE.PUBLIC)["checkpoint"], "Exact frozen v226 native checkpoint required")
    return (
        {
            "v219_freeze_sha256": EXPECTED_V219,
            "serving_route_freeze_sha256": sha(route.PUBLIC),
            "serving_asset_set_sha256": asset_digest,
            "v230_freeze_sha256": EXPECTED_V230,
            "v230_result_sha256": EXPECTED_V230_RESULT,
            **{
                key: historical[key]
                for key in (
                    "v223_freeze_sha256",
                    "v223_result_sha256",
                    "v223_fit_sha256",
                    "v223_calibration_sha256",
                    "v226_freeze_sha256",
                    "v226_result_sha256",
                    "candidate_head",
                    "diagnostic_english_points",
                )
            },
            "native_checkpoint": checkpoint,
        },
        head,
        points,
    )


def identity():
    NATIVE.PILOT.require_model_free()
    prepared, cases = preparation()
    routes, head, points = unchanged_routes()
    NATIVE.PILOT.require_model_free()
    return (
        {
            "schema": "chipmunk.audio-matched-background-score-freeze/250",
            "v248_freeze_sha256": sha(PREPARATION_FREEZE),
            "v248_result_sha256": sha(PREPARATION),
            "v248_case_journal_sha256": prepared["private_cases_sha256"],
            "background_evidence_type": PROVISIONAL,
            "human_listening_reviewed": False,
            "foreground_coverage": prepared["foreground_coverage"],
            "background_coverage": prepared["background_coverage"],
            "planned_cases": len(cases),
            "planned_rows_per_batch": ROWS_PER_BATCH,
            "batch_order": {"baseline": list(CONDITIONS), "candidate_after_all_baseline": list(CONDITIONS)},
            "route_bindings": routes,
            "second_opinion": "promote_only_baseline_inconclusive_when_candidate_english_diagnostic_direction_is_ai",
            "target_language_calibration_permitted": False,
            "new_fitting_permitted": False,
            "serving_change_permitted": False,
            "holdout_access_permitted": False,
            "code_sha256": {name: sha(ROOT / name) for name in CODE},
        },
        cases,
        head,
        points,
    )


def input_index(cases):
    return [{"case": case["case"], NATIVE_HASH: case[NATIVE_HASH]} for case in cases]


def preflight():
    if not PREPARATION.is_file():
        return {"status": "awaiting_published_v248_preparation", "model_calls": 0}
    locked, *_ = identity()
    return {**locked, "status": "ready_for_freeze", "model_calls": 0}


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve earlier scoring attempts")
    locked, cases, _, _ = identity()
    TRIAL.mkdir(mode=0o700)
    for phase in (BASELINE, CANDIDATE):
        phase.mkdir(mode=0o700)
    index = TRIAL / "private-input-index.json"
    write(index, input_index(cases))
    locked["private_input_index_sha256"] = sha(index)
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return {"status": "frozen", "freeze_sha256": sha(PUBLIC), "planned_cases": len(cases), "model_calls": 0}


def verify():
    locked, cases, head, points = identity()
    index = TRIAL / "private-input-index.json"
    locked["private_input_index_sha256"] = sha(index)
    require(
        locked == read(PUBLIC) == read(TRIAL / "freeze.json") and read(index) == input_index(cases),
        "Exact v250 scoring freeze and native input index required",
    )
    return locked, cases, head, points


def verify_baseline():
    return verify()[1]


def stage(phase, batch):
    require(
        phase in (BASELINE, CANDIDATE) and type(batch) is int and 0 <= batch < len(CONDITIONS),
        "One fixed phase and condition batch required",
    )
    return phase / f"extract-{batch:02d}"


def completed_baseline(batch, cases):
    folder = stage(BASELINE, batch)
    journal = folder / "output/private-observations.jsonl"
    rows = baseline.serving.records(journal)
    count = sum(row["pella_forwards"] for row in rows)
    receipt = NATIVE.PILOT.speech.qualification_profile(
        folder, count, {}, Path(__file__).name, "extract", batch
    )
    require(
        receipt
        == {
            "status": "complete",
            "batch": batch,
            "condition": CONDITIONS[batch],
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "new_neural_calls": count,
        }
        and len(rows) == ROWS_PER_BATCH,
        "Exact complete 72-row baseline batch required",
    )
    for row, case in zip(rows, cases[batch :: len(CONDITIONS)], strict=True):
        baseline.audit_case(row, case)
    return rows, sha(folder / "profile.json"), count


def candidate_pairs(cases, baseline_batches):
    require(len(baseline_batches) == len(CONDITIONS), "Complete baseline phase required before candidate")
    result = []
    for batch, (rows, _, _) in enumerate(baseline_batches):
        require(len(rows) == ROWS_PER_BATCH, "Exact 72-row baseline batch required")
        pairs = list(zip(cases[batch :: len(CONDITIONS)], rows, strict=True))
        for case, observed in pairs:
            baseline.audit_case(observed, case)
            candidate.admitted(case, observed)
        result.append(pairs)
    return result


def verify_candidate():
    _, cases, head, points = verify()
    require(
        all((stage(BASELINE, batch) / "profile.json").is_file() for batch in range(len(CONDITIONS))),
        "All five complete baseline batches are required before candidate scoring",
    )
    batches = [completed_baseline(i, cases) for i in range(len(CONDITIONS))]
    return candidate_pairs(cases, batches), head, points


def audit_candidate(row, case, observed, head, points):
    candidate.audit_observation(row, case, observed, head, points)
    packet = row["native_features"]
    if packet is not None:
        require(
            packet["input_pcm_sha256"] == case[NATIVE_HASH],
            "Candidate native feature packet differs from the baseline's frozen input",
        )


def candidate_stage(batch):
    return stage(CANDIDATE, batch)


baseline_worker = scoped(
    baseline.worker,
    {"TRIAL": BASELINE, "PUBLIC": PUBLIC, "verify": verify_baseline, "waveform": baseline_waveform},
)
run_baseline = scoped(
    baseline.run,
    {
        "TRIAL": BASELINE,
        "verify": verify_baseline,
        "completed_batch": completed_baseline,
        "MODULE": MODULE,
        "__file__": __file__,
    },
)
native_worker = SimpleNamespace(
    checkpoint=NATIVE.checkpoint,
    PUBLIC=NATIVE.PUBLIC,
    PILOT=NATIVE.PILOT,
    WEIGHTS=NATIVE.WEIGHTS,
    extract=NATIVE.extract,
    waveform=candidate_waveform,
)
candidate_worker = scoped(
    candidate.worker,
    {
        "PUBLIC": PUBLIC,
        "batch_stage": candidate_stage,
        "verify": verify_candidate,
        "native": native_worker,
        "audit_observation": audit_candidate,
    },
)
completed_candidate = scoped(
    candidate.completed_batch,
    {
        "PUBLIC": PUBLIC,
        "batch_stage": candidate_stage,
        "audit_observation": audit_candidate,
        "__file__": __file__,
    },
)
run_candidate = scoped(
    candidate.run,
    {
        "TRIAL": CANDIDATE,
        "verify": verify_candidate,
        "batch_stage": candidate_stage,
        "completed_batch": completed_candidate,
        "MODULE": MODULE,
        "__file__": __file__,
    },
)


def worker(batch, output):
    if output == stage(BASELINE, batch) / "output":
        baseline_worker(batch, output)
    else:
        require(output == candidate_stage(batch) / "output", "Exact guarded phase output required")
        candidate_worker(batch, output)


def second_opinion(recorded, candidate_direction):
    require(recorded in DIRECTIONS and candidate_direction in DIRECTIONS, "Known directions required")
    return "ai" if recorded == "inconclusive" and candidate_direction == "ai" else recorded


def coverage(group):
    promoted = [row for row in group if row["second_opinion"] != row["recorded_direction"]]
    recovered = [row for row in promoted if row["arm"] != "genuine"]
    added_false = [row for row in promoted if row["arm"] == "genuine"]
    baseline_false = sum(row["arm"] == "genuine" and row["recorded_direction"] == "ai" for row in group)
    baseline_wrong_original = sum(
        row["arm"] != "genuine" and row["recorded_direction"] == "original" for row in group
    )
    return {
        "planned": len(group),
        "preparation_unavailable": sum(row["route_status"] == "preparation_unavailable" for row in group),
        "speech_gate_rejected": sum(row["route_status"] == "not_applicable" for row in group),
        "assessed_by_both": sum(row["route_status"] == "ok" for row in group),
        "baseline_directions": candidate.directions(group, "recorded_direction"),
        "candidate_english_diagnostic_directions": candidate.directions(
            group, "candidate_english_point_direction"
        ),
        "second_opinion_directions": candidate.directions(group, "second_opinion"),
        "recovered_synthetic_variants": len(recovered),
        "recovered_synthetic_voice_groups": len({(row["arm"], row["group"]) for row in recovered}),
        "added_false_ai_variants": len(added_false),
        "added_false_ai_genuine_groups": len({row["group"] for row in added_false}),
        "baseline_false_ai_variants": baseline_false,
        "second_opinion_false_ai_variants": baseline_false + len(added_false),
        "baseline_synthetic_wrong_original_variants": baseline_wrong_original,
        "second_opinion_synthetic_wrong_original_variants": baseline_wrong_original,
    }


def summarize(cases, rows):
    lookup = {row["case"]: row for row in rows}
    require(
        len(lookup) == len(rows) == len(cases) == ROWS_PER_BATCH * len(CONDITIONS),
        "Every planned case requires exactly one paired outcome",
    )
    grouped = {condition: [] for condition in CONDITIONS}
    for case in cases:
        row = lookup[case["case"]]
        require(
            all(row[key] == case[key] for key in candidate.KEYS)
            and row["route_status"] in {"ok", "not_applicable", "preparation_unavailable"}
            and ((case["status"] == "unavailable") == (row["route_status"] == "preparation_unavailable"))
            and ((row["route_status"] == "ok") == (row["recorded_direction"] != "unavailable"))
            and ((row["route_status"] == "ok") == (row["candidate_english_point_direction"] != "unavailable"))
            and row["encoder_forwards"] == int(row["route_status"] == "ok"),
            "Candidate must share the unchanged baseline speech gate and planned disposition",
        )
        direction = second_opinion(row["recorded_direction"], row["candidate_english_point_direction"])
        grouped[case["condition"]].append(
            {**row, "group": case["group"], "cohort": case["cohort"], "second_opinion": direction}
        )
    cohorts = sorted({case["cohort"] for case in cases})
    return {
        "by_condition": {condition: coverage(group) for condition, group in grouped.items()},
        "by_arm_and_condition": {
            arm: {
                condition: coverage([row for row in group if row["arm"] == arm])
                for condition, group in grouped.items()
            }
            for arm in ARMS
        },
        "by_source_cohort_and_condition": {
            cohort: {
                condition: coverage([row for row in group if row["cohort"] == cohort])
                for condition, group in grouped.items()
            }
            for cohort in cohorts
        },
        "rank_by_generator_and_condition": {
            condition: {
                name: {
                    key: candidate.rank_comparison(group, arms, key)
                    for key in ("current_raw", "candidate_score")
                }
                for name, arms in {"all_synthetic": ARMS[1:], **{arm: (arm,) for arm in ARMS[1:]}}.items()
            }
            for condition, group in grouped.items()
        },
    }


def readout():
    locked, cases, head, points = verify()
    baseline_batches = [completed_baseline(i, cases) for i in range(len(CONDITIONS))]
    pairs = candidate_pairs(cases, baseline_batches)
    candidate_batches = [completed_candidate(i, pairs, head, points) for i in range(len(CONDITIONS))]
    rows = [row for batch, _, _ in candidate_batches for row in batch]
    return {
        "schema": "chipmunk.audio-matched-background-score-result/250",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "planned_cases": len(cases),
        "baseline_classifier_calls": sum(count for _, _, count in baseline_batches),
        "candidate_encoder_calls": sum(count for _, _, count in candidate_batches),
        "baseline_profile_sha256": [profile for _, profile, _ in baseline_batches],
        "candidate_profile_sha256": [profile for _, profile, _ in candidate_batches],
        "foreground_coverage": locked["foreground_coverage"],
        "background_coverage": locked["background_coverage"],
        "background_evidence_type": locked["background_evidence_type"],
        "human_listening_reviewed": False,
        **summarize(cases, rows),
        "new_fitting": False,
        "target_language_calibration": False,
        "serving_changed": False,
        "holdout_scored": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "freeze", "baseline", "candidate", "readout", "_worker")
    )
    parser.add_argument("batch", type=int, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    require(
        (args.batch is not None) == (args.command in {"baseline", "candidate", "_worker"})
        and (args.output is not None) == (args.command == "_worker"),
        "Exact scoring phase arguments required",
    )
    if args.command == "_worker":
        worker(args.batch, args.output)
    elif args.command in {"baseline", "candidate"}:
        (run_baseline if args.command == "baseline" else run_candidate)(args.batch)
    else:
        result = {"preflight": preflight, "freeze": freeze, "readout": readout}[args.command]()
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
