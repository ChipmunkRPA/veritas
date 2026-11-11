"""Apply the unchanged v223 paired head to admitted v219 three-second clips."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import expit

from scripts import audio_fresh_shared_noise_score_v219 as prior
from scripts import audio_paired_consistency_v223 as learner
from scripts import audio_wavlm_three_second_v226 as native
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = prior.ROOT, prior.BASE, prior.PANEL
TRIAL = BASE / "audio-paired-transfer-v230"
PUBLIC = PANEL / "audio-paired-transfer-freeze-v230.json"
QUALIFICATION = PANEL / "audio-wavlm-three-second-complete-v226.json"
FIT_RESULT = PANEL / "audio-paired-consistency-complete-v223.json"
MODULE = "scripts.audio_paired_transfer_v230"
METHOD = "paired_wavlm_raw"
CLEAN, *NOISE = prior.CONDITIONS
DIRECTIONS = learner.DIRECTIONS
KEYS = ("case", "triplet", "arm", "condition")
CODE = (
    "backend/scripts/audio_paired_transfer_v230.py",
    "backend/tests/test_audio_paired_transfer_v230.py",
    "docs/audio-paired-transfer-v230-method.md",
)
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require


def exact_lock(public, local, schema):
    locked = read(public)
    require(
        locked == read(local)
        and locked["schema"] == schema
        and all(sha(ROOT / name) == digest for name, digest in locked["code_sha256"].items()),
        f"Exact {schema} identity required",
    )
    return locked


def v219_context():
    prepared = scoped(prior.cases, {"waveform": lambda _case: None})()
    locked = exact_lock(
        prior.PUBLIC, prior.TRIAL / "freeze.json", "chipmunk.audio-fresh-shared-noise-score-freeze/219"
    )
    require(
        locked["v217_preparation_sha256"] == sha(prior.PREP)
        and locked["v217_freeze_sha256"] == sha(prior.PREP_FREEZE)
        and locked["route_freeze_sha256"] == sha(prior.serving.prior.PUBLIC)
        and tuple(locked["conditions"]) == prior.CONDITIONS
        and locked["scoring_batches"] == len(prior.CONDITIONS)
        and locked["holdout_access_permitted"] is locked["calibration_change_permitted"] is False,
        "Exact guarded v217 preparation and unchanged v219 route required",
    )
    batches = [prior.completed_batch(batch, prepared) for batch in range(len(prior.CONDITIONS))]
    aggregate = scoped(
        prior.readout,
        {"verify": lambda: prepared, "completed_batch": lambda batch, _rows: batches[batch]},
    )()
    require(
        aggregate == read(native.PRIOR_RESULT)
        and aggregate["status"] == "complete"
        and aggregate["schema"] == "chipmunk.audio-fresh-shared-noise-score-result/219",
        "Exact published v219 aggregate and guarded baseline journals required",
    )
    return [
        list(zip(prepared[batch :: len(prior.CONDITIONS)], result[0], strict=True))
        for batch, result in enumerate(batches)
    ]


def fit_context():
    locked = exact_lock(
        learner.PUBLIC, learner.TRIAL / "freeze.json", "chipmunk.audio-paired-consistency-freeze/223"
    )
    result = read(FIT_RESULT)
    fit, cal = learner.TRIAL / "fitted.json", learner.TRIAL / "calibration.json"
    head, calibration = read(fit)[METHOD], read(cal)
    require(
        result == read(learner.TRIAL / "summary.json")
        and result["schema"] == "chipmunk.audio-paired-consistency-result/223"
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(learner.PUBLIC)
        and result["private_fit_sha256"] == sha(fit)
        and result["private_calibration_sha256"] == sha(cal)
        and result["confirmation_feature_or_media_read"] is result["serving_admitted"] is False
        and locked["pair_penalty_by_arm"][METHOD] == learner.FITS[METHOD]
        and locked["numeric_versions"]
        == {name: importlib.metadata.version(name) for name in ("numpy", "scipy")}
        and head["arm"] == METHOD,
        "Exact frozen v223 fitted head and English calibration required",
    )
    points = {method: calibration["points"][method] for method in ("current", METHOD)}
    center, scale = (np.asarray(head["transform"][key]) for key in ("center", "scale"))
    coefficients = np.asarray(head["coefficients"])
    require(
        center.shape == scale.shape == (native.DIMENSIONS,)
        and coefficients.shape == (native.DIMENSIONS + 2,)
        and all(np.isfinite(values).all() for values in (center, scale, coefficients))
        and np.all(scale > 0)
        and coefficients[0] >= 0
        and all(0 <= p["original_below"] <= p["ai_above"] <= 1 for p in points.values()),
        "Exact finite v223 head dimensions and ordered diagnostic points required",
    )
    return head, points


def qualification_context(pairs):
    locked = exact_lock(
        native.PUBLIC, native.TRIAL / "freeze.json", "chipmunk.audio-wavlm-three-second-freeze/226"
    )
    index = prior.CONDITIONS.index(native.CONDITION)
    cases, observed = zip(*pairs[index], strict=True)
    plan = native.choose(cases, observed)
    private_plan = native.TRIAL / "private-plan.json"
    stage = native.TRIAL / "authored"
    journal = stage / "output/private-features.jsonl"
    receipt = native.PILOT.speech.qualification_profile(
        stage, native.FORWARDS, {}, Path(native.__file__).name
    )
    result = read(QUALIFICATION)
    profile = read(stage / "profile.json")
    require(
        private_plan.read_bytes() == native.encoded(plan)
        and locked["private_plan_sha256"] == sha(private_plan)
        and locked["v217_freeze_sha256"] == sha(prior.PREP_FREEZE)
        and locked["v217_result_sha256"] == sha(prior.PREP)
        and locked["v219_freeze_sha256"] == sha(prior.PUBLIC)
        and locked["v219_result_sha256"] == sha(native.PRIOR_RESULT)
        and locked["v219_condition_profile_sha256"] == sha(prior.TRIAL / f"extract-{index:02d}/profile.json")
        and locked["v219_condition_journal_sha256"]
        == sha(prior.TRIAL / f"extract-{index:02d}/output/private-observations.jsonl")
        and locked["condition"] == native.CONDITION
        and locked["arms"] == list(prior.ARMS)
        and locked["sample_rate"] == native.RATE
        and locked["pool_layers"] == list(native.LAYERS)
        and locked["checkpoint"]["v13_freeze_sha256"] == sha(native.V13_FREEZE)
        and locked["checkpoint"]["clock"]["frames"] == native.FRAMES
        and locked["candidate_calls_permitted"] is locked["fitting_permitted"] is False
        and receipt["status"] == "complete"
        and receipt["new_neural_calls"] == native.FORWARDS
        and receipt["freeze_sha256"] == sha(native.PUBLIC)
        and receipt["private_features_sha256"] == sha(journal)
        and receipt["repeated_exact"] is True
        and result
        == {
            "schema": "chipmunk.audio-wavlm-three-second-result/226",
            **receipt,
            "resource_profile_sha256": sha(stage / "profile.json"),
            "worker_duration_s": profile["duration_s"],
            "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
            "condition": native.CONDITION,
            "arms_qualified": list(prior.ARMS),
            "input_samples": native.SAMPLES,
            "feature_frames": native.FRAMES,
            "feature_dimensions": native.DIMENSIONS,
            "candidate_scores": 0,
            "new_fitting": False,
            "defaults_changed": False,
        },
        "Exact published guarded v226 qualification required",
    )


def identity():
    native.PILOT.require_model_free()
    pairs = v219_context()
    head, points = fit_context()
    qualification_context(pairs)
    return (
        {
            "schema": "chipmunk.audio-paired-transfer-freeze/230",
            "code_sha256": {name: sha(ROOT / name) for name in CODE},
            "v217_freeze_sha256": sha(prior.PREP_FREEZE),
            "v217_result_sha256": sha(prior.PREP),
            "v219_freeze_sha256": sha(prior.PUBLIC),
            "v219_result_sha256": sha(native.PRIOR_RESULT),
            "v223_freeze_sha256": sha(learner.PUBLIC),
            "v223_result_sha256": sha(FIT_RESULT),
            "v223_fit_sha256": sha(learner.TRIAL / "fitted.json"),
            "v223_calibration_sha256": sha(learner.TRIAL / "calibration.json"),
            "v226_freeze_sha256": sha(native.PUBLIC),
            "v226_result_sha256": sha(QUALIFICATION),
            "batches": list(prior.CONDITIONS),
            "planned_by_batch": [len(batch) for batch in pairs],
            "admitted_by_batch": [sum(admitted(case, row) for case, row in batch) for batch in pairs],
            "candidate_head": METHOD,
            "diagnostic_english_points": points,
            "new_fitting_permitted": False,
            "serving_change_permitted": False,
            "holdout_access_permitted": False,
        },
        pairs,
        head,
        points,
    )


def verify():
    locked, pairs, head, points = identity()
    require(locked == read(PUBLIC) == read(TRIAL / "freeze.json"), "Exact frozen v230 experiment required")
    return pairs, head, points


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve any earlier v230 attempt")
    locked, *_ = identity()
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return {"status": "frozen", "freeze_sha256": sha(PUBLIC)}


def admitted(case, baseline):
    if baseline["route_status"] != "ok":
        return False
    require(
        case["status"] == "prepared"
        and case["samples"] == native.SAMPLES
        and baseline["windows"] == [[0, native.SAMPLES]],
        "Exact original-gate-admitted whole-window support required",
    )
    return True


def paired_score(raw, packet, head):
    design = learner.matrix([{"raw": raw, "wavlm": packet["wavlm"]}], METHOD, head["transform"])
    return float(expit(design[0] @ head["coefficients"]))


def outcome(case, baseline, packet, head, points):
    if admitted(case, baseline):
        require(packet is not None, "An admitted window requires its native feature packet")
        candidate = paired_score(baseline["raw"], packet, head)
        current = baseline["english_reference_estimate_ai"]
        learner.bounded_logit([candidate, current])
    else:
        require(packet is None, "Unavailable input must never acquire native features")
        candidate, current = None, None
    return {
        **{key: case[key] for key in KEYS},
        "route_status": baseline["route_status"],
        "recorded_direction": baseline["decision"],
        "current_raw": baseline["raw"] if packet is not None else None,
        "candidate_score": candidate,
        "current_english_point_direction": learner.score_direction(current, points["current"]),
        "candidate_english_point_direction": learner.score_direction(candidate, points[METHOD]),
        "native_features": packet,
        "encoder_forwards": int(packet is not None),
    }


def audit_observation(row, case, baseline, head, points):
    packet = row["native_features"]
    if packet is not None:
        values = np.asarray(packet["wavlm"], dtype="<f4")
        digests = [packet["input_pcm_sha256"], *packet["pooled_state_sha256"]]
        require(
            packet["feature_frames"] == packet["valid_feature_frames"] == native.FRAMES
            and values.shape == (native.DIMENSIONS,)
            and values.tolist() == packet["wavlm"]
            and np.isfinite(values).all()
            and native.float_sha(values) == packet["wavlm_sha256"]
            and len(digests) == 1 + len(native.LAYERS)
            and all(
                isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
                for value in digests
            ),
            "Exact native three-second feature packet required",
        )
    require(row == outcome(case, baseline, packet, head, points), "Recorded transfer calculation differs")


def batch_stage(batch):
    require(type(batch) is int and 0 <= batch < len(prior.CONDITIONS), "One fixed batch is required")
    return TRIAL / f"extract-{batch:02d}"


def worker(batch, output):
    stage = batch_stage(batch)
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and not (stage / "profile.json").exists()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact owned guarded offline transfer worker required",
    )
    pairs, head, points = verify()
    require(native.checkpoint() == read(native.PUBLIC)["checkpoint"], "v226 checkpoint or runtime differs")
    native.PILOT.require_model_free()
    import torch

    from scripts.audio_wavlm_features_v13 import load

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = load(native.WEIGHTS)
    output.mkdir(mode=0o700)
    journal = output / "private-observations.jsonl"
    calls = 0

    def count(_module, _args):
        nonlocal calls
        calls += 1

    with ExitStack() as stack, journal.open("x") as stream:
        stack.callback(model.register_forward_pre_hook(count).remove)
        for case, baseline in pairs[batch]:
            before = calls
            packet = native.extract(native.waveform(case), model) if admitted(case, baseline) else None
            row = outcome(case, baseline, packet, head, points)
            require(
                calls - before == row["encoder_forwards"], "One encoder forward per admitted case required"
            )
            audit_observation(row, case, baseline, head, points)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    write(
        output / "result.json",
        {
            "status": "complete",
            "batch": batch,
            "condition": prior.CONDITIONS[batch],
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "new_neural_calls": calls,
        },
    )


def completed_batch(batch, pairs, head, points):
    stage = batch_stage(batch)
    journal = stage / "output/private-observations.jsonl"
    rows = prior.serving.records(journal)
    calls = sum(admitted(case, baseline) for case, baseline in pairs[batch])
    receipt = native.PILOT.speech.qualification_profile(
        stage, calls, {}, Path(__file__).name, "extract", batch
    )
    require(
        receipt
        == {
            "status": "complete",
            "batch": batch,
            "condition": prior.CONDITIONS[batch],
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "new_neural_calls": calls,
        },
        "Exact complete transfer batch receipt required",
    )
    for row, (case, baseline) in zip(rows, pairs[batch], strict=True):
        audit_observation(row, case, baseline, head, points)
    summary_rows = [{key: value for key, value in row.items() if key != "native_features"} for row in rows]
    return summary_rows, sha(stage / "profile.json"), calls


def run(batch):
    output = batch_stage(batch) / "output"
    pairs, head, points = verify()
    for previous in range(batch):
        completed_batch(previous, pairs, head, points)

    def launch(argv, **kwargs):
        require(
            argv[3:] == ["scripts.video_appearance_native_v108", "extract", str(output), str(batch)],
            "Exact owned transfer command required",
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


def rank_comparison(group, synthetic_arms, score):
    genuine = [row for row in group if row["arm"] == "genuine" and row[score] is not None]
    synthetic = [row for row in group if row["arm"] in synthetic_arms and row[score] is not None]
    differences = [(a[score] - b[score], a["triplet"] == b["triplet"]) for a in synthetic for b in genuine]
    return {
        "all": concordance([value for value, _ in differences]),
        "within_triplet": concordance([value for value, matched in differences if matched]),
    }


def concordance(differences):
    n = len(differences)
    wins = sum(value > 0 for value in differences)
    ties = sum(value == 0 for value in differences)
    return {
        "comparisons": n,
        "wins": wins,
        "ties": ties,
        "losses": n - wins - ties,
        "concordance": (wins + ties / 2) / n if n else None,
    }


def directions(group, key):
    count = Counter(row[key] for row in group)
    return {direction: count[direction] for direction in DIRECTIONS}


def drift(clean, noisy, score):
    paired = [(clean[row["triplet"]][score], row[score]) for row in noisy]
    paired = [(a, b) for a, b in paired if a is not None and b is not None]
    if paired:
        before, after = np.asarray(paired).T
        delta = after - before
        mean = float(np.abs(delta).mean())
        mean_logit = float(np.abs(learner.bounded_logit(after) - learner.bounded_logit(before)).mean())
        up, down = int(np.count_nonzero(delta > 0)), int(np.count_nonzero(delta < 0))
    else:
        mean = mean_logit = None
        up = down = 0
    return {
        "planned_pairs": len(noisy),
        "both_assessed": len(paired),
        "mean_absolute_score_change": mean,
        "mean_absolute_bounded_logit_change": mean_logit,
        "up": up,
        "down": down,
        "tie": len(paired) - up - down,
    }


def summarize(rows):
    by_condition = {
        condition: [row for row in rows if row["condition"] == condition] for condition in prior.CONDITIONS
    }
    groups = {
        arm: {
            condition: [row for row in group if row["arm"] == arm]
            for condition, group in by_condition.items()
        }
        for arm in prior.ARMS
    }
    scores = ("current_raw", "candidate_score")
    generators = {"all_synthetic": prior.ARMS[1:], "alania": ("alania",), "drifting": ("drifting",)}
    return {
        "coverage_and_directions": {
            arm: {
                condition: {
                    "planned": len(group),
                    "assessed": sum(row["route_status"] == "ok" for row in group),
                    "speech_gate_rejected": sum(row["route_status"] == "not_applicable" for row in group),
                    "preparation_unavailable": sum(
                        row["route_status"] == "preparation_unavailable" for row in group
                    ),
                    "recorded_v219": directions(group, "recorded_direction"),
                    "current_at_english_diagnostic_point": directions(
                        group, "current_english_point_direction"
                    ),
                    "candidate_at_english_diagnostic_point": directions(
                        group, "candidate_english_point_direction"
                    ),
                }
                for condition, group in conditions.items()
            }
            for arm, conditions in groups.items()
        },
        "raw_ranking": {
            condition: {
                generator: {score: rank_comparison(group, arms, score) for score in scores}
                for generator, arms in generators.items()
            }
            for condition, group in by_condition.items()
        },
        "paired_clean_to_noise_score_drift": {
            arm: {
                condition: {
                    score: drift(
                        {row["triplet"]: row for row in conditions[CLEAN]}, conditions[condition], score
                    )
                    for score in scores
                }
                for condition in NOISE
            }
            for arm, conditions in groups.items()
        },
    }


def readout():
    pairs, head, points = verify()
    rows, profiles, calls = [], [], 0
    for batch in range(len(prior.CONDITIONS)):
        observed, profile, count = completed_batch(batch, pairs, head, points)
        rows.extend(observed)
        profiles.append(profile)
        calls += count
    return {
        "schema": "chipmunk.audio-paired-transfer-result/230",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "new_neural_calls": calls,
        "profile_sha256": profiles,
        "planned": len(rows),
        "assessed": calls,
        **summarize(rows),
        "scope": "Observed transfer on the consumed v219 cohorts and two shared recorded tapes. "
        "English score directions are diagnostic only, without target-language or duration calibration.",
        "new_fitting": False,
        "defaults_changed": False,
        "holdout_scored": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "run", "readout", "_worker"))
    parser.add_argument("batch", type=int, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    if args.command == "_worker":
        worker(args.batch, args.output)
    elif args.command == "run":
        run(args.batch)
    elif args.command == "preflight":
        if not QUALIFICATION.is_file():
            print(json.dumps({"status": "waiting_for_published_v226_qualification", "model_calls": 0}))
            return
        locked, *_ = identity()
        print(
            json.dumps(
                {
                    "status": "ready_for_freeze",
                    "admitted_by_batch": locked["admitted_by_batch"],
                    "model_calls": 0,
                }
            )
        )
    else:
        print(json.dumps(freeze() if args.command == "freeze" else readout(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
