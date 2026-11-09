"""Confirmation of the fixed sensitive and shared-region readouts; no refit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from scripts import image_native_confirmation_v46 as producer
from scripts import image_regional_readout_v27 as statistics
from scripts.evaluation_tradeoffs_v44 import observed_frontier
from scripts.image_native_comparison_readout_v45 import source_mask
from scripts.localization_benchmark import mask_metrics
from scripts.replay_regional_calibration_v43 import load_map

read, write, sha, require = producer.read, producer.write, producer.sha, producer.require
ROOT = producer.ROOT
METHODS = (producer.POLICIES[0], "current-and-strict")


def producer_rows(trial, arm, number, expected):
    name = f"{arm}-native-{number:03d}"
    part, profile_root = (trial / name, trial / (name + "-profile"))
    result, profile = (read(part / "completed.json"), read(profile_root / "profile.json"))
    require(
        profile["module"] == "scripts.image_native_confirmation_v46"
        and profile["arguments"] == ["score", str(trial), arm, "native", str(number)],
        "Wrong producer phase",
    )
    require(
        profile["monitor_implementation_sha256"]
        == sha(ROOT / "backend/scripts/image_native_confirmation_runtime_v46.py"),
        "Wrong resource observer",
    )
    require(
        profile["child_exit_code"] == 0
        and profile["guard"] is None
        and (profile["sample_errors"] == 0)
        and (not profile["resource_limit_observations"]),
        "Failed or missing phase must not be excluded",
    )
    require(
        sha(profile_root / "resource-samples.jsonl") == profile["resource_journal_sha256"],
        "Resource observations changed",
    )
    require(
        result["arm"] == arm
        and result["format"] == "native"
        and (result["phase"] == number)
        and (result["selected_views"] == len(expected)),
        "Wrong producer support",
    )
    require(
        result["attempted_model_forwards"]
        == result["completed_model_forwards"]
        == len(expected) + (number == 0)
        and result["exact_repeat_extra_forwards"] == (number == 0),
        "Complete prediction and repeat counts required",
    )
    require(
        result["execution_freeze_sha256"] == sha(trial / "freeze.json")
        and sha(part / "rows.jsonl") == result["rows_sha256"],
        "Producer bindings changed",
    )
    rows = [json.loads(line) for line in (part / "rows.jsonl").read_text().splitlines()]
    require(len(rows) == len(expected), "Complete source rows required")
    for row, task in zip(rows, expected, strict=True):
        require(
            row["status"] == "ok"
            and row["arm"] == arm
            and (row["format"] == "native")
            and all((row[k] == v for k, v in task.items())),
            "Unchecked or mismatched source",
        )
    return rows


def admission(trial):
    _frozen, _chain, plan = producer.lease(trial)
    tasks = producer.comparison_tasks(plan)
    for number in range(10):
        selected = producer.phase_tasks(tasks, number)
        for arm in producer.ARMS:
            producer_rows(trial, arm, number, selected)
    return (plan, tasks, Path(read(trial / "private-paths.json")["raw_map_root"]))


def phase(trial, number):
    from scripts.image_regional_training_v26 import PAIRS, source_views

    cv2.setNumThreads(1)
    plan, tasks, raw_root = admission(trial)
    selected = producer.phase_tasks(tasks, number)
    arms = [producer_rows(trial, arm, number, selected) for arm in producer.ARMS]
    out = trial / f"readout-{number:03d}"
    out.mkdir(exist_ok=False)
    cases = {row["source_id"]: row for row in plan["rows"]}
    qualified = {
        row["source_id"]: row
        for row in (
            json.loads(line) for line in (PAIRS / "qualification/pairs.jsonl").read_text().splitlines()
        )
    }
    previous, views, source_truth = (None, None, None)
    with (out / "rows.jsonl").open("x") as journal:
        for group in zip(*arms, strict=True):
            row = group[0]
            try:
                sid = row["source_id"]
                if previous != sid:
                    edited, controls, source_truth = source_views(cases[sid], qualified[sid])
                    previous, views = (sid, {"ai": edited, **controls})
                data, image = producer.encoded_view(views[row["kind"]], "native")
                truth = source_truth > 0 if row["kind"] == "ai" else np.zeros(source_truth.shape, bool)
                result = measure(
                    group, raw_root, data, np.asarray(image), truth, qualified[sid]["truth_png_sha256"]
                )
                journal.write(json.dumps(result, allow_nan=False) + "\n")
                journal.flush()
            except BaseException as error:
                journal.write(
                    json.dumps(
                        {k: row[k] for k in ("source_id", "component", "role", "kind", "format")}
                        | {"status": "failed", "error_type": type(error).__name__}
                    )
                    + "\n"
                )
                journal.flush()
                raise
    write(
        out / "completed.json",
        {
            "phase": number,
            "views": len(selected),
            "rows_sha256": sha(out / "rows.jsonl"),
            "execution_freeze_sha256": sha(trial / "freeze.json"),
            "neural_forwards": 0,
        },
    )


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (48,) and np.isfinite(values).all(), "All 48 paired source groups required")
    rng = np.random.default_rng(statistics.SEED)
    means = values[rng.integers(0, 48, size=(statistics.BOOTSTRAPS, 48))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975], method="linear")]


def summarize(rows):
    statistics.validate_rows(rows, role="confirmation", groups=48, policy_ids=METHODS)
    require(len(rows) == 236, "All four views of all 59 photos required")
    summaries = {}
    baseline = statistics.group_summary(rows, "baseline")
    for name in ("baseline", *METHODS):
        groups = statistics.group_summary(rows, name)
        summary = statistics.totals(groups)
        ai = [statistics.measurement(row, name) for row in rows if row["kind"] == "ai"]
        summary["edited_photos"] = 59
        summary["usefully_localized_photos"] = sum(m["iou"] >= 0.3 and m["recall"] >= 0.5 for m in ai)
        summary["edited_photos_without_retained_region"] = sum(
            m["predicted_positive_pixels"] == 0 for m in ai
        )
        summary["control_photos"] = {}
        for kind in ("original", "tone", "blur"):
            values = [statistics.measurement(row, name) for row in rows if row["kind"] == kind]
            summary["control_photos"][kind] = {
                "photos": 59,
                "flagged": sum(m["predicted_positive_pixels"] > 0 for m in values),
                "mean_wrongly_highlighted_area_fraction": sum(
                    m["false_positive_pixels"] / m["pixels"] for m in values
                )
                / len(values),
            }
        keys = sorted(groups)
        differences = [groups[key]["useful"] - baseline[key]["useful"] for key in keys]
        summary["paired_useful_fraction_difference"] = sum(differences) / 48
        summary["paired_group_bootstrap_95pct_interval"] = interval(differences)
        summary["paired_false_group_difference_intervals"] = {
            kind: interval(
                [
                    float(groups[key]["false_groups"][kind]) - float(baseline[key]["false_groups"][kind])
                    for key in keys
                ]
            )
            for kind in ("original", "tone", "blur")
        }
        summary["new_useful_photos"] = sum(

                row["kind"] == "ai"
                and statistics.measurement(row, name)["iou"] >= 0.3
                and (statistics.measurement(row, name)["recall"] >= 0.5)
                and (
                    not (
                        statistics.measurement(row, "baseline")["iou"] >= 0.3
                        and statistics.measurement(row, "baseline")["recall"] >= 0.5
                    )
                )
                for row in rows

        )
        summary["lost_useful_photos"] = sum(

                row["kind"] == "ai"
                and statistics.measurement(row, "baseline")["iou"] >= 0.3
                and (statistics.measurement(row, "baseline")["recall"] >= 0.5)
                and (
                    not (
                        statistics.measurement(row, name)["iou"] >= 0.3
                        and statistics.measurement(row, name)["recall"] >= 0.5
                    )
                )
                for row in rows

        )
        summaries[name] = summary
    frontier_values = {
        name: {"useful": s["useful_photo_fraction_mean_over_groups"], **s["false_groups_by_kind"]}
        for name, s in summaries.items()
    }
    return {
        "methods": summaries,
        "tradeoff_comparison": observed_frontier(
            frontier_values, maximize=("useful",), minimize=("original", "tone", "blur")
        ),
        "automatic_zero_false_alarm_increase_rule_applied": False,
        "comparison_source_role": "confirmation",
        "source_groups": 48,
        "photos": 59,
        "final_confirmation_used": True,
        "probability_calibration_performed": False,
        "serving_admitted": False,
        "scope": "Confirmation source groups and readouts fixed before inference; no fitting or model selection uses these labels. Same editing family; creator aliases/pretraining overlap unknown. Conditional pointwise bootstrap intervals are not simultaneous or customer authorship probabilities. No retained region establishes whole-file originality.",
    }


def seal(trial):
    admission(trial)
    rows = []
    for number in range(10):
        part = trial / f"readout-{number:03d}"
        profile_root = trial / f"readout-{number:03d}-profile"
        result, profile = (read(part / "completed.json"), read(profile_root / "profile.json"))
        require(
            profile["module"] == "scripts.image_native_confirmation_readout_v46"
            and profile["arguments"] == ["phase", str(trial), str(number)],
            "Wrong readout phase",
        )
        require(
            profile["monitor_implementation_sha256"]
            == sha(ROOT / "backend/scripts/image_native_confirmation_runtime_v46.py")
            and profile["child_exit_code"] == 0
            and (profile["guard"] is None)
            and (profile["sample_errors"] == 0)
            and (not profile["resource_limit_observations"]),
            "Incomplete readout cannot select a method",
        )
        require(
            sha(profile_root / "resource-samples.jsonl") == profile["resource_journal_sha256"]
            and sha(part / "rows.jsonl") == result["rows_sha256"],
            "Readout observations changed",
        )
        require(
            result["phase"] == number
            and result["views"] == (20 if number == 9 else 24)
            and (result["execution_freeze_sha256"] == sha(trial / "freeze.json"))
            and (result["neural_forwards"] == 0),
            "Wrong readout coverage",
        )
        rows.extend(json.loads(line) for line in (part / "rows.jsonl").read_text().splitlines())
    write(
        trial / "comparison-summary.json",
        {
            "schema": "chipmunk.native-confirmation-result/46",
            "execution_freeze_sha256": sha(trial / "freeze.json"),
            **summarize(rows),
        },
    )


def measure(rows, raw_root, data, pixels, truth, truth_sha):
    from app.detectors.image import splice
    from scripts.score_splice_localizers import score_map

    height, width = truth.shape
    positive = rows[0]["kind"] == "ai"
    require(
        truth.dtype == bool and pixels.shape == (height, width, 3) and bool(truth.any()) == positive,
        "Exact source geometry and truth required",
    )
    require([row["arm"] for row in rows] == list(producer.ARMS), "Both selected detector paths required")
    for row in rows:
        require(
            row["status"] == "ok" and row["role"] == "confirmation" and row["format"] == "native",
            "Checked native confirmation views required",
        )
        require(
            all(row[k] == rows[0][k] for k in ("source_id", "component", "kind")),
            "Same source support required",
        )
        require(
            row["input_sha256"] == hashlib.sha256(data).hexdigest()
            and row["source_rgb_sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest(),
            "Exact input bytes/RGB required",
        )
        require(
            row["source_size"] == [width, height]
            and row["positive_truth_png_sha256"] == truth_sha
            and row["negative_truth_is_zero"] is (not positive),
            "Truth or source coordinates changed",
        )
    raw, current = (
        load_map(row, raw_root, shape) for row, shape in zip(rows, ((height, width), (384, 384)), strict=True)
    )
    current_gate = int(np.count_nonzero(current > splice.PIXEL_THRESHOLD)) >= splice.MINIMUM_ACTIVE_CELLS
    require(
        type(rows[1]["image_gate"]) is bool
        and rows[1]["image_gate"] == current_gate
        and rows[1]["pixel_threshold"] == splice.PIXEL_THRESHOLD,
        "Current production gate changed",
    )

    def mask(response, gate, pixel):
        value = source_mask(response, gate, pixel, data, pixels, truth)
        require(
            mask_metrics(value, truth) == score_map(response, gate, pixel, data, pixels, truth)["metrics"],
            "Source mask differs from frozen production metric",
        )
        return value

    current_mask = mask(current, current_gate, splice.PIXEL_THRESHOLD)
    name = producer.POLICIES[0]
    policy = statistics.policies()[name]
    strict = mask(raw, statistics.image_gate(raw, policy), policy["pixel"])
    result = {k: rows[0][k] for k in ("source_id", "component", "role", "kind", "format")}
    result["baseline"] = {"status": "ok", "metrics": mask_metrics(current_mask, truth)}
    result["candidate"] = {
        name: {"status": "ok", "metrics": mask_metrics(strict, truth)},
        "current-and-strict": {"status": "ok", "metrics": mask_metrics(current_mask & strict, truth)},
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("phase")
    p.add_argument("trial", type=Path)
    p.add_argument("number", type=int, choices=range(10))
    p = sub.add_parser("seal")
    p.add_argument("trial", type=Path)
    args = parser.parse_args()
    if args.command == "phase":
        phase(args.trial.resolve(), args.number)
    else:
        seal(args.trial.resolve())
