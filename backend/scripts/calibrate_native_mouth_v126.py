"""Separate point/file study calibration on the fixed calibration groups only."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import types
from pathlib import Path

from scripts import native_mouth_readout_v126 as method
from scripts.video_appearance_corrected_v112 import scoped

fitted, f = method.fitted, method.f
ROOT, BASE, PANEL = f.ROOT, f.BASE, f.PANEL
TRIAL = BASE / "native-mouth-calibration-v126"
PRIVATE = f.CACHE.parent / "native-mouth-calibration-v126"
PUBLIC = PANEL / "native-mouth-calibration-freeze-v126.json"
CODE = (
    "backend/scripts/native_mouth_readout_v126.py",
    "backend/scripts/calibrate_native_mouth_v126.py",
    "backend/tests/test_native_mouth_readout_v126.py",
    "docs/native-mouth-calibration-v126-method.md",
    "backend/app/detectors/video/adapters/lipsync_xcorr.py",
    "backend/app/detectors/video/adapters/lipsync.py",
    "backend/app/detectors/video/lipsync.py",
    "backend/app/detectors/video/xcorr.py",
    "backend/app/detectors/video/av_evidence.py",
    "backend/app/detectors/video/sampling.py",
    "backend/app/detectors/video/adapters/swap.py",
    "backend/app/detectors/audio/decode.py",
    "backend/app/engine/detector.py",
)
read, write, sha, require = f.read, f.write, f.sha, f.require


def qualified_fit():
    fitted.verify()
    stage = fitted.TRIAL / "bind-00"
    r, p = read(stage / "output/result.json"), read(stage / "profile.json")
    public = read(PANEL / "native-mouth-head-fit-result-v125.json")
    require(
        r["status"] == "complete"
        and r["raw_heads_fitted"] == 3
        and p["child_exit_code"] == 0
        and p["guard"] is None
        and p["sample_errors"] == 0
        and p["worker_receipt_sha256"] == sha(stage / "output/result.json")
        and r["private_learned_artifact_sha256"]
        == public["private_learned_artifact_sha256"]
        == sha(fitted.PRIVATE / "reference-and-heads.json")
        and r["private_labels_coverage_sha256"] == sha(fitted.PRIVATE / "fitting-labels-and-coverage.json"),
        "Exact successful fitting-only learned artifact required",
    )


def records_for(role):
    require(role in {"confidence_cal", "selection", "confirmation"}, "One exact nonfitting role required")
    rows = [r for r in fitted.input_records() if r["study_role"] == role]
    require(
        len(rows) == 60
        and len({r["asset_index"] for r in rows}) == 10
        and len({r["source_group"] for r in rows}) == 10,
        "All ten fixed connected nonfitting groups required",
    )
    constructions = {(r["asset_index"], r["condition"]): r for i in range(8) for r in f.case_rows(i)}
    return rows, constructions


def identity():
    qualified_fit()
    rows, _ = records_for("confidence_cal")
    return {
        "schema": "chipmunk.native-mouth-calibration-freeze/126",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "fitting_result_sha256": sha(PANEL / "native-mouth-head-fit-result-v125.json"),
        "learned_reference_heads_sha256": sha(fitted.PRIVATE / "reference-and-heads.json"),
        "calibration_groups": 10,
        "planned_calibration_cases": len(rows),
        "arms": list(fitted.ARMS),
        "study_calibration": {
            "method": method.CAL_METHOD,
            "slope_bounds": [0, 50],
            "intercept_bounds": [-50, 50],
            "L2_penalty": 0.01,
            "scale_floor": 0.05,
            "max_iterations": 2000,
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
        "research_group_budgets": list(method.BUDGETS),
        "primary_research_group_budget": method.PRIMARY_BUDGET,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "torch", "opencv-python-headless", "pillow")
        },
        "scope": "Calibration-group vectors/labels only. Separate monotone point and clip study calibrations for every frozen head; frozen control-group research points and current correlation adapter on common recorded native faces. No selection/confirmation responses, customer probability or serving change.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "new_neural_calls": 0, "new_calibrations_fitted": 0}), flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged separate calibration protocol required",
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
        "Only fixed calibration stage permitted",
    )
    args = list(argv)
    args[3] = "scripts.calibrate_native_mouth_v126"
    return subprocess.Popen(args, **kwargs)


def run():
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        f.runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda c, b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", 0)


def calibrate_stage(output):
    import torch
    from threadpoolctl import threadpool_limits

    require(
        all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
        "One numerical thread required",
    )
    threadpool_limits(limits=1)
    PRIVATE.mkdir(exist_ok=False)
    rows, constructions = records_for("confidence_cal")
    learned = read(fitted.PRIVATE / "reference-and-heads.json")
    plans = {r["source"]["asset_index"]: r["source"] for r in read(f.clips.TRIAL / "plan.json")}
    scores = []
    point_rows = []
    point_values = {arm: [] for arm in fitted.ARMS}
    forwards = []
    hook = torch.nn.modules.module.register_module_forward_pre_hook(lambda *args: forwards.append(True))
    try:
        with (PRIVATE / "calibration-cases.jsonl").open("x") as journal:
            for row in rows:
                construction = constructions[row["asset_index"], row["condition"]]
                scored = method.score_case(row, learned, construction)
                scored["current"] = method.current_check(row, construction)
                scores.append(scored)
                journal.write(json.dumps(scored, allow_nan=False) + "\n")
                journal.flush()
                bounds = plans[row["asset_index"]]["edit_frames_half_open"]
                point_rows.extend(
                    {
                        **{k: row[k] for k in ("source_group", "condition", "study_role")},
                        "label": int(
                            row["condition"].startswith("ai_")
                            and bounds is not None
                            and bounds[0] <= i < bounds[1]
                        ),
                    }
                    for i in scored["eligible_centres"]
                )
                for arm in fitted.ARMS:
                    point_values[arm].extend(scored["arms"][arm]["point_margins"])
                print(json.dumps({"completed_calibration_cases": len(scores), "planned": 60}), flush=True)
    finally:
        hook.remove()
    require(not forwards, "Unexpected neural forward in data-only calibration; no retry")
    calibrations = {}
    summaries = {}
    point_weights = method.balanced_weights(point_rows, "confidence_cal")
    for arm in fitted.ARMS:
        cases = [
            {
                **{k: r[k] for k in ("source_group", "condition", "study_role", "label")},
                "window_score": r["arms"][arm]["window_score"],
            }
            for r in scores
        ]
        usable = [r for r in cases if r["window_score"] is not None]
        point_cal = method.calibrate(point_values[arm], point_rows)
        clip_cal = method.calibrate([r["window_score"] for r in usable], usable)
        points = method.operating_points(cases)
        calibrations[arm] = {"point": point_cal, "clip": clip_cal, "research_points": points}
        summaries[arm] = {
            "point_calibration_targets": len(point_rows),
            "clip_calibration_targets": len(usable),
            "point_iterations": point_cal["iterations"],
            "clip_iterations": clip_cal["iterations"],
            "point_positive_slope": point_cal["slope"] > 0,
            "clip_positive_slope": clip_cal["slope"] > 0,
            "point_calibration_fit_reliability": method.reliability(
                method.probability(point_values[arm], point_cal),
                [r["label"] for r in point_rows],
                point_weights,
            ),
            "clip_calibration_fit_reliability": method.reliability(
                method.probability([r["window_score"] for r in usable], clip_cal),
                [r["label"] for r in usable],
                method.balanced_weights(usable, "confidence_cal"),
            ),
            "calibration_genuine_groups_flagged": {
                key: sum(
                    max(
                        c["window_score"]
                        for c in cases
                        if c["source_group"] == g and not c["label"] and c["window_score"] is not None
                    )
                    > p["raw_margin"]
                    for g in {c["source_group"] for c in cases if not c["label"]}
                )
                for key, p in points.items()
            },
        }
    write(
        PRIVATE / "calibrations-and-research-points.json",
        {
            "method": method.CAL_METHOD,
            "arms": calibrations,
            "scope": "Balanced study calibration, no population or customer accuracy claim. All operating points are frozen internal research settings.",
        },
    )
    result = {
        "schema": "chipmunk.native-mouth-calibration-result/126",
        "status": "complete",
        "calibration_groups": 10,
        "planned_cases": len(scores),
        "ready_cases": sum(r["input_status"] == "extracted" for r in scores),
        "unavailable_cases": sum(r["input_status"] == "unavailable" for r in scores),
        "eligible_native_centres": len(point_rows),
        "positive_point_targets": sum(r["label"] for r in point_rows),
        "calibrations_fitted": 6,
        "arms": summaries,
        "current_common_input": {
            "assessed_cases": sum(r["current"]["assessed"] for r in scores),
            "flagged_cases": sum(r["current"]["flagged"] for r in scores),
        },
        "private_calibrations_sha256": sha(PRIVATE / "calibrations-and-research-points.json"),
        "private_calibration_case_journal_sha256": sha(PRIVATE / "calibration-cases.jsonl"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "new_neural_calls": len(forwards),
        "defaults_changed": False,
        "scope": "Calibration fitting, not held-out validation. Independent point/clip study calibrations and internal genuine-group research settings. Current adapter uses common recorded dense faces, not a serving-tracker parity claim. No customer confidence or serving contribution.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "run", "bind"))
    p.add_argument("arguments", nargs="*")
    args = p.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze takes no arguments")
        freeze()
    elif args.command == "run":
        require(not args.arguments, "One fixed calibration stage")
        run()
    else:
        verify()
        require(
            len(args.arguments) == 2 and args.arguments[1] == "0", "Exact fixed calibration stage required"
        )
        output = Path(args.arguments[0])
        require(output.resolve().is_relative_to(TRIAL.resolve()), "Exclusive private output required")
        output.mkdir(exist_ok=False)
        calibrate_stage(output)
