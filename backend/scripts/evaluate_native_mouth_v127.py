"""Locked grouped selection/confirmation comparison; no learning or promotion."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import types
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import calibrate_native_mouth_v126 as calibration
from scripts.video_appearance_corrected_v112 import scoped

m, fitted, f = calibration.method, calibration.fitted, calibration.f
ROOT, BASE, PANEL = f.ROOT, f.BASE, f.PANEL
TRIAL = BASE / "native-mouth-evaluation-v127"
PRIVATE = f.CACHE.parent / "native-mouth-evaluation-v127"
PUBLIC = PANEL / "native-mouth-evaluation-freeze-v127.json"
CODE = (
    "backend/scripts/evaluate_native_mouth_v127.py",
    "backend/tests/test_evaluate_native_mouth_v127.py",
    "docs/native-mouth-evaluation-v127-method.md",
)
ROLES = ("selection", "confirmation")
BOOTSTRAPS = 1000
SEED = 127
read, write, sha, require = f.read, f.write, f.sha, f.require


def qualified_calibration():
    calibration.verify()
    stage = calibration.TRIAL / "bind-00"
    r, p = read(stage / "output/result.json"), read(stage / "profile.json")
    public = read(PANEL / "native-mouth-calibration-result-v126.json")
    require(
        r["status"] == "complete"
        and p["child_exit_code"] == 0
        and p["guard"] is None
        and p["sample_errors"] == 0
        and public["private_receipt_sha256"]
        == p["worker_receipt_sha256"]
        == sha(stage / "output/result.json")
        and r["private_calibrations_sha256"]
        == sha(calibration.PRIVATE / "calibrations-and-research-points.json")
        and r["private_calibration_case_journal_sha256"]
        == sha(calibration.PRIVATE / "calibration-cases.jsonl"),
        "Exact successful separate calibrations and immutable research points required",
    )


def balanced_reliability(probabilities, rows):
    if not rows:
        return m.reliability([], [])
    cells = Counter((r["source_group"], r["label"], r["condition"]) for r in rows)
    groups = {y: {g for g, label, _ in cells if label == y} for y in (0, 1)}
    if not all(groups.values()):
        return {
            "n": len(rows),
            "brier": None,
            "log_loss": None,
            "ece_10": None,
            "reason": "Binary class-balanced reliability requires both classes; unweighted slice reliability is reported separately.",
        }
    conditions = defaultdict(set)
    for g, y, c in cells:
        conditions[g, y].add(c)
    weights = [
        1
        / (
            2
            * len(groups[r["label"]])
            * len(conditions[r["source_group"], r["label"]])
            * cells[r["source_group"], r["label"], r["condition"]]
        )
        for r in rows
    ]
    return m.reliability(probabilities, [r["label"] for r in rows], weights)


def outcome(metadata, assessed, intervals, probability, truth, roi=None):
    require(assessed or not intervals, "Unsupported scope cannot create a flag")
    require(
        metadata["input_status"] == "extracted" or (not assessed and not intervals and probability is None),
        "Unavailable construction cannot become an assessed or probability-bearing result",
    )
    return {
        **metadata,
        "assessed": bool(assessed),
        "flagged": bool(intervals),
        "study_probability": probability,
        **m.interval_metrics(intervals, truth),
        "roi_change_mask": roi,
    }


def mask_overlap(feature, construction, selected):
    from PIL import Image

    if selected is None:
        return None
    index = selected["centre_frame_index"]
    bounds = feature["frames"][index]["region_bbox"]
    proof = next((r for r in construction["edit_proofs"] if r["frame_index"] == index), None)
    base = {
        "frame_index": index,
        "inspected_region_bbox": bounds,
        "region_role": "model input mouth rectangle, not predicted edited-pixel mask",
    }
    if proof is None:
        return {**base, "pixel_iou": 0.0, "changed_pixel_coverage": 0.0}
    file = Path(proof["mask_file"])
    require(sha(file) == proof["mask_sha256"], "Actual audited change mask required")
    with Image.open(file) as image:
        mask = np.asarray(image) > 0
    x0, y0, x1, y1 = bounds
    require(
        0 <= x0 < x1 <= mask.shape[1] and 0 <= y0 < y1 <= mask.shape[0],
        "Inspected native region outside mask",
    )
    total = int(mask.sum())
    intersection = int(mask[y0:y1, x0:x1].sum())
    area = (x1 - x0) * (y1 - y0)
    require(
        total == proof["changed_pixels"] and total > 0, "Nonempty exact changed-pixel ground truth required"
    )
    return {
        **base,
        "pixel_iou": intersection / (total + area - intersection),
        "changed_pixel_coverage": intersection / total,
    }


def counts(rows):
    ready = [r for r in rows if r["input_status"] == "extracted"]
    ai = [r for r in ready if r["label"]]
    genuine = [r for r in ready if not r["label"]]
    assessed_ai = [r for r in ai if r["assessed"]]
    assessed_genuine = [r for r in genuine if r["assessed"]]
    probabilities = [r for r in rows if r["study_probability"] is not None]
    rois = [r["roi_change_mask"] for r in ai if r["flagged"] and r["roi_change_mask"] is not None]
    flagged_ai = sum(r["flagged"] for r in ai)
    false_flags = sum(r["flagged"] for r in genuine)

    def rate(n, d):
        return n / d if d else None

    return {
        "planned_cases": len(rows),
        "ready_cases": len(ready),
        "unavailable_cases": len(rows) - len(ready),
        "assessed_cases": sum(r["assessed"] for r in rows),
        "inconclusive_ready_cases": sum(not r["assessed"] for r in ready),
        "ready_ai": len(ai),
        "assessed_ai": len(assessed_ai),
        "ai_flags": flagged_ai,
        "missed_assessed_ai": sum(not r["flagged"] for r in assessed_ai),
        "unassessed_ready_ai": len(ai) - len(assessed_ai),
        "ready_ai_without_flag": len(ai) - flagged_ai,
        "ready_genuine": len(genuine),
        "assessed_genuine": len(assessed_genuine),
        "genuine_false_flags": false_flags,
        "unassessed_ready_genuine": len(genuine) - len(assessed_genuine),
        "ai_detection_rate_ready": rate(flagged_ai, len(ai)),
        "false_alarm_rate_assessed_genuine": rate(false_flags, len(assessed_genuine)),
        "false_alarm_rate_ready_genuine": rate(false_flags, len(genuine)),
        "inconclusive_rate_ready": rate(sum(not r["assessed"] for r in ready), len(ready)),
        "ai_flags_overlapping_edit": sum(r["edit_overlap_seconds"] > 0 for r in ai),
        "ai_localized_iou_0_2": sum(r["localized_iou_0_2"] for r in ai),
        "mean_ai_temporal_union_iou": float(np.mean([r["temporal_union_iou"] for r in ai])) if ai else None,
        "outside_edit_flagged_seconds": sum(r["outside_edit_flagged_seconds"] for r in ready),
        "flagged_ai_with_inspected_roi_mask_comparison": len(rois),
        "mean_inspected_roi_pixel_iou_on_flagged_ai": float(np.mean([r["pixel_iou"] for r in rois]))
        if rois
        else None,
        "mean_changed_pixel_coverage_on_flagged_ai": float(
            np.mean([r["changed_pixel_coverage"] for r in rois])
        )
        if rois
        else None,
        "clip_reliability_unweighted": m.reliability(
            [r["study_probability"] for r in probabilities], [r["label"] for r in probabilities]
        ),
        "clip_reliability_balanced": balanced_reliability(
            [r["study_probability"] for r in probabilities], probabilities
        ),
    }


def paired(candidate, reference):
    def keys(rows):
        return [
            (r["asset_index"], r["source_group"], r["condition"], r["input_status"], r["label"]) for r in rows
        ]

    require(keys(candidate) == keys(reference), "Complete same-input paired outcomes required")
    groups = sorted({r["source_group"] for r in candidate})
    require(len(groups) >= 2, "Multiple connected held-out groups required for uncertainty")
    group_indices = {g: [i for i, r in enumerate(candidate) if r["source_group"] == g] for g in groups}

    def differences(indices):
        positive = [
            i for i in indices if candidate[i]["input_status"] == "extracted" and candidate[i]["label"]
        ]
        negative = [
            i for i in indices if candidate[i]["input_status"] == "extracted" and not candidate[i]["label"]
        ]
        require(positive and negative, "Both ready paired classes required")
        return [
            sum(int(candidate[i][key]) - int(reference[i][key]) for i in selected) / len(selected)
            for key, selected in [
                ("flagged", positive),
                ("localized_iou_0_2", positive),
                ("flagged", negative),
            ]
        ]

    samples = []
    rng = np.random.default_rng(SEED)
    for _ in range(BOOTSTRAPS):
        indices = [i for g in rng.choice(groups, size=len(groups), replace=True) for i in group_indices[g]]
        samples.append(differences(indices))
    observed = differences(list(range(len(candidate))))
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    names = ("ai_detection_rate_ready", "ai_localization_rate_ready", "false_alarm_rate_ready_genuine")
    return {
        "connected_groups": len(groups),
        "bootstrap_replicates": BOOTSTRAPS,
        "common_assessed_cases": sum(
            a["assessed"] and b["assessed"] for a, b in zip(candidate, reference, strict=True)
        ),
        "additional_assessed_cases": sum(
            a["assessed"] and not b["assessed"] for a, b in zip(candidate, reference, strict=True)
        ),
        "lost_assessed_cases": sum(
            not a["assessed"] and b["assessed"] for a, b in zip(candidate, reference, strict=True)
        ),
        "additional_localized_ai": sum(
            a["label"] and a["localized_iou_0_2"] and not b["localized_iou_0_2"]
            for a, b in zip(candidate, reference, strict=True)
        ),
        "lost_localized_ai": sum(
            a["label"] and not a["localized_iou_0_2"] and b["localized_iou_0_2"]
            for a, b in zip(candidate, reference, strict=True)
        ),
        "additional_genuine_flags": sum(
            not a["label"] and a["flagged"] and not b["flagged"]
            for a, b in zip(candidate, reference, strict=True)
        ),
        "removed_genuine_flags": sum(
            not a["label"] and not a["flagged"] and b["flagged"]
            for a, b in zip(candidate, reference, strict=True)
        ),
        "rate_differences": {
            name: {
                "candidate_minus_reference": observed[i],
                "group_bootstrap_ci95": [float(intervals[0, i]), float(intervals[1, i])],
            }
            for i, name in enumerate(names)
        },
        "scope": "Paired connected-source bootstrap, all variants together. Ten-group intervals are exploratory, not a population reliability guarantee; coverage differences are explicit.",
    }


def identity():
    qualified_calibration()
    return {
        "schema": "chipmunk.native-mouth-evaluation-freeze/127",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "calibration_result_sha256": sha(PANEL / "native-mouth-calibration-result-v126.json"),
        "private_calibrations_sha256": sha(calibration.PRIVATE / "calibrations-and-research-points.json"),
        "roles": list(ROLES),
        "planned_cases_per_role": 60,
        "connected_groups_per_role": 10,
        "research_budgets": list(m.BUDGETS),
        "primary_research_budget": m.PRIMARY_BUDGET,
        "group_bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "temporal_union_iou_localization_point": 0.2,
        "scope": "All three locked heads, own point/clip calibrations and all frozen research settings; no learning, retuning or automatic winner/promotion. Actual current correlation adapter on common recorded dense native faces. Report missing scope separately from missed assessed AI.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "held_out_cases_scored": 0}), flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged pre-held-out comparison required",
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
        "Only fixed held-out stages permitted",
    )
    args = list(argv)
    args[3] = "scripts.evaluate_native_mouth_v127"
    return subprocess.Popen(args, **kwargs)


def run(role):
    require(role in ROLES, "One locked held-out role required")
    if role == "confirmation":
        stage = TRIAL / "bind-00"
        r, p = read(stage / "output/result.json"), read(stage / "profile.json")
        require(
            r["status"] == "complete"
            and r["role"] == "selection"
            and p["child_exit_code"] == 0
            and p["guard"] is None
            and p["sample_errors"] == 0
            and p["worker_receipt_sha256"] == sha(stage / "output/result.json"),
            "Successful complete selection readout required; no outcome-driven retuning",
        )
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
    )("bind", ROLES.index(role))


def evaluate(role, output):
    import torch
    from threadpoolctl import threadpool_limits

    from app.detectors.video.decode import FrameTable

    require(
        role in ROLES
        and all(
            os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        ),
        "One fixed role and numerical thread required",
    )
    threadpool_limits(limits=1)
    folder = PRIVATE / role
    folder.mkdir(parents=True, exist_ok=False)
    rows, constructions = calibration.records_for(role)
    learned = read(fitted.PRIVATE / "reference-and-heads.json")
    calibrated = read(calibration.PRIVATE / "calibrations-and-research-points.json")["arms"]
    plans = {r["source"]["asset_index"]: r["source"] for r in read(f.clips.TRIAL / "plan.json")}
    outcomes = {arm: {str(b): [] for b in m.BUDGETS} for arm in fitted.ARMS}
    baseline = []
    point_rows = []
    point_probabilities = {arm: [] for arm in fitted.ARMS}
    forwards = []
    hook = torch.nn.modules.module.register_module_forward_pre_hook(lambda *args: forwards.append(True))
    try:
        with (folder / "scored-cases.jsonl").open("x") as journal:
            for ordinal, row in enumerate(rows):
                construction = constructions[row["asset_index"], row["condition"]]
                scored = m.score_case(row, learned, construction)
                current = m.current_check(row, construction)
                metadata = {
                    k: scored[k]
                    for k in (
                        "asset_index",
                        "source_group",
                        "study_role",
                        "condition",
                        "label",
                        "input_status",
                    )
                }
                bounds = plans[row["asset_index"]]["edit_frames_half_open"]
                truth = None
                if scored["label"] and row["status"] == "extracted":
                    table = FrameTable.from_json(row["table"])
                    truth = [str(Fraction(table.pts[i]) * table.tb) for i in bounds]
                intervals = [[str(r["start_s"]), str(r["end_s"])] for r in current["segments"]]
                require(
                    current["flagged"] == bool(intervals), "Actual baseline flags must have source segments"
                )
                baseline.append(outcome(metadata, current["assessed"], intervals, None, truth))
                targets = [
                    int(scored["label"] and bounds is not None and bounds[0] <= i < bounds[1])
                    for i in scored["eligible_centres"]
                ]
                point_rows.extend(
                    {**{k: metadata[k] for k in ("source_group", "condition")}, "label": y} for y in targets
                )
                for arm in fitted.ARMS:
                    part = scored["arms"][arm]
                    cal = calibrated[arm]
                    probabilities = m.probability(part["point_margins"], cal["point"]).tolist()
                    part["study_point_probabilities"] = probabilities
                    point_probabilities[arm].extend(probabilities)
                    file_p = (
                        None
                        if part["window_score"] is None
                        else float(m.probability(part["window_score"], cal["clip"]))
                    )
                    part["study_clip_probability"] = file_p
                    for budget, point in cal["research_points"].items():
                        selected = [w for w in part["windows"] if w["response"] > point["raw_margin"]]
                        roi = (
                            mask_overlap(row, construction, selected[0])
                            if selected and scored["label"]
                            else None
                        )
                        spans = [[w["start_seconds"], w["end_seconds"]] for w in selected]
                        outcomes[arm][budget].append(
                            outcome(metadata, bool(part["windows"]), spans, file_p, truth, roi)
                        )
                journal.write(
                    json.dumps({**scored, "current": current, "truth_interval": truth}, allow_nan=False)
                    + "\n"
                )
                journal.flush()
                print(json.dumps({"role": role, "completed_cases": ordinal + 1, "planned": 60}), flush=True)
    finally:
        hook.remove()
    require(not forwards, "Unexpected pretrained network forward in locked data readout")
    write(folder / "outcomes.json", {"current_common_input": baseline, "candidates": outcomes})
    primary = str(m.PRIMARY_BUDGET)
    summaries = {
        arm: {
            "research_budgets": {key: counts(values) for key, values in cases.items()},
            "conditions_at_primary": {
                c: counts([r for r in cases[primary] if r["condition"] == c]) for c in f.clips.CONDITIONS
            },
            "point_reliability_unweighted": m.reliability(
                point_probabilities[arm], [r["label"] for r in point_rows]
            ),
            "point_reliability_balanced": balanced_reliability(point_probabilities[arm], point_rows),
        }
        for arm, cases in outcomes.items()
    }
    result = {
        "schema": "chipmunk.native-mouth-locked-evaluation/127",
        "status": "complete",
        "role": role,
        "planned_cases": 60,
        "connected_groups": 10,
        "primary_research_budget": m.PRIMARY_BUDGET,
        "arms": summaries,
        "current_common_input": {
            "overall": counts(baseline),
            "conditions": {
                c: counts([r for r in baseline if r["condition"] == c]) for c in f.clips.CONDITIONS
            },
        },
        "paired_primary_vs_visual": {
            arm: paired(outcomes[arm][primary], outcomes["visual_only"][primary])
            for arm in fitted.ARMS
            if arm != "visual_only"
        },
        "paired_primary_vs_current": {arm: paired(outcomes[arm][primary], baseline) for arm in fitted.ARMS},
        "private_score_journal_sha256": sha(folder / "scored-cases.jsonl"),
        "private_outcomes_sha256": sha(folder / "outcomes.json"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "new_neural_calls": len(forwards),
        "new_head_or_calibration_fits": 0,
        "defaults_changed": False,
        "scope": "Locked connected-source outcomes on one centred construction generator. Common dense native faces for current adapter, not serving-pipeline parity. Study probabilities, ROI input rectangles and indexed speech support are not universal authenticity probabilities or pixel-edit masks. Fresh placement/generator transfer and full app verification remain required.",
    }
    write(output / "result.json", result)
    print(
        json.dumps({"status": "complete", "role": role, "planned_cases": 60, "new_neural_calls": 0}),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "run", "bind"))
    p.add_argument("arguments", nargs="*")
    args = p.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze takes no arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One held-out role required")
        run(args.arguments[0])
    else:
        verify()
        require(
            len(args.arguments) == 2 and args.arguments[1] in {"0", "1"},
            "Fixed output and held-out role required",
        )
        output = Path(args.arguments[0])
        require(output.resolve().is_relative_to(TRIAL.resolve()), "Exclusive private output required")
        output.mkdir(exist_ok=False)
        evaluate(ROLES[int(args.arguments[1])], output)
