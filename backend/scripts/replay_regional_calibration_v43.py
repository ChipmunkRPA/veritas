"""Replay exact private calibration maps through frozen v27 policy selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts.image_regional_inference_v28 import (
    calibration_tasks,
    encoded_view,
    phase_tasks,
    read,
    require,
    sha,
    write,
)
from scripts.image_regional_readout_v27 import calibrate, coverage, image_gate, measurement, policies
from scripts.prepare_regional_calibration_v42 import check_score_admission

ROOT = Path(__file__).resolve().parents[2]
PREPARATION = ROOT / "docs/evaluation-panels/image-regional-calibration-v42-preparation.json"


def admission(trial):
    check_score_admission(trial)
    preparation, start = read(PREPARATION), read(trial / "readout-v43-execution-start.json")
    require(
        start["public_preparation_sha256"] == sha(PREPARATION)
        and start["execution_freeze_sha256"] == sha(trial / "freeze.json")
        and start["remote_verified_before_first_readout"] is True
        and len(start["remote_preparation_blob_sha"]) == 40,
        "Published readout and exact final-state binding required",
    )
    require(
        all(sha(ROOT / path) == value for path, value in preparation["implementation_sha256"].items()),
        "Frozen readout implementation changed",
    )
    paths = read(trial / "private-paths.json")
    plan = read(Path(paths["study"]) / "plan.json")
    tasks = calibration_tasks(plan)
    # Require the full cohort, not just successful/easier producer phases.
    for fmt in ("native", "jpeg75"):
        for number in range(7):
            expected = phase_tasks(tasks, number)
            for arm in ("candidate", "baseline"):
                producer_rows(trial, arm, fmt, number, expected)
    return plan, tasks, Path(paths["raw_map_root"])


def producer_rows(trial, arm, fmt, number, expected):
    require(arm in {"candidate", "baseline"} and fmt in {"native", "jpeg75"}, "Known arm and format required")
    name = f"{arm}-{fmt}-{number:03d}"
    part, profile_root = trial / name, trial / (name + "-profile")
    result, profile = read(part / "completed.json"), read(profile_root / "profile.json")
    require(profile.get("monitor_implementation_sha256") == sha(ROOT / "backend/scripts/regional_calibration_runtime_v42.py"), "Fixed-monitor provenance required for every producer phase")
    require(
        profile["module"] == "scripts.image_regional_inference_v28"
        and profile["arguments"] == [str(trial), arm, fmt, str(number)]
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and not profile["resource_limit_observations"],
        "Missing, failed or mismatched producer phase cannot be excluded",
    )
    require(
        sha(profile_root / "resource-samples.jsonl") == profile["resource_journal_sha256"],
        "Producer resource observations changed",
    )
    repeats = int(number == 0)
    require(
        result["arm"] == arm
        and result["format"] == fmt
        and result["phase"] == number
        and result["selected_views"] == len(expected)
        and result["attempted_model_forwards"]
        == result["completed_model_forwards"]
        == len(expected) + repeats
        and result["exact_repeat_extra_forwards"] == repeats
        and result["execution_freeze_sha256"] == sha(trial / "freeze.json"),
        "Producer identity/count/repeat proof differs",
    )
    require(sha(part / "rows.jsonl") == result["rows_sha256"], "Producer row journal changed")
    rows = [json.loads(line) for line in (part / "rows.jsonl").read_text().splitlines()]
    require(len(rows) == len(expected), "All prescribed producer rows required")
    for row, task in zip(rows, expected, strict=True):
        require(
            {k: row[k] for k in task} == task
            and row["status"] == "ok"
            and row["arm"] == arm
            and row["format"] == fmt,
            "Source/kind identity changed or an unchecked source was dropped",
        )
    return rows


def load_map(row, raw_root, expected_shape):
    path = Path(row["raw_map_file"])
    require(
        path.resolve().is_relative_to(raw_root.resolve())
        and path.is_file()
        and sha(path) == row["raw_map_sha256"],
        "Raw map path or retained bytes changed",
    )
    value = np.load(path, allow_pickle=False)
    require(
        list(value.shape) == row["raw_map_shape"]
        and value.shape == expected_shape
        and value.dtype == np.float32
        and np.isfinite(value).all()
        and value.min() >= 0
        and value.max() <= 1,
        "Complete finite float32 map with exact geometry required",
    )
    return value


def producer_coverage(trial, output):
    """Report recorded unknowns without granting calibration admission."""
    check_score_admission(trial)
    paths = read(trial / "private-paths.json")
    tasks = calibration_tasks(read(Path(paths["study"]) / "plan.json"))
    expected = {(r["source_id"], r["kind"]): r for r in tasks}
    result = {}
    for arm in ("candidate", "baseline"):
        for fmt in ("native", "jpeg75"):
            observed, errors = {}, 0
            for number in range(7):
                journal = trial / f"{arm}-{fmt}-{number:03d}/rows.jsonl"
                if not journal.is_file():
                    continue
                for line in journal.read_text().splitlines():
                    try:
                        row = json.loads(line)
                        key = row["source_id"], row["kind"]
                        require(
                            key in expected
                            and row["arm"] == arm
                            and row["format"] == fmt
                            and {k: row[k] for k in expected[key]} == expected[key],
                            "Unbound recorded source",
                        )
                        status = row["status"]
                        require(
                            status in {"ok", "failed", "unavailable", "skipped", "inconclusive"},
                            "Unknown recorded status",
                        )
                        if key in observed:
                            status = "failed"
                            errors += 1
                        observed[key] = status
                    except (ValueError, KeyError, TypeError):
                        errors += 1
            counts = Counter(observed.get(key, "missing") for key in expected)
            result[f"{arm}-{fmt}"] = {
                "planned_unique_views": len(tasks),
                "recorded_status_counts": dict(counts),
                "not_recorded_ok": len(tasks) - counts["ok"],
                "not_recorded_ok_fraction": (len(tasks) - counts["ok"]) / len(tasks),
                "malformed_or_duplicate_records": errors,
            }
    write(
        output,
        {
            "schema": "chipmunk.regional-calibration-input-coverage/43",
            "arm_formats": result,
            "metadata_only": True,
            "grants_calibration_admission": False,
            "whole_file_originality_established": False,
            "scope": "Recorded per-view statuses only. An ok row is not a substitute for a successful terminal phase/profile, exact-map verification or complete-cohort admission. Missing may mean not yet attempted; no terminal-process inference is made here.",
        },
    )


def pair_metrics(candidate, baseline, raw_root, data, pixels, truth, expected_truth_sha):
    from app.detectors.image import fused, splice
    from scripts.localization_benchmark import mask_metrics
    from scripts.score_splice_localizers import score_map

    require(
        candidate["arm"] == "candidate"
        and baseline["arm"] == "baseline"
        and all(r["role"] == "calibration" and r["status"] == "ok" for r in (candidate, baseline)),
        "Only successfully checked calibration pairs can yield measurements",
    )
    require(
        candidate["format"] == baseline["format"]
        and candidate["kind"] == baseline["kind"]
        and candidate["source_id"] == baseline["source_id"]
        and candidate["component"] == baseline["component"],
        "The two arms must describe the same source and view",
    )
    height, width = pixels.shape[:2]
    input_sha = hashlib.sha256(data).hexdigest()
    rgb_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
    positive = candidate["kind"] == "ai"
    require(
        truth.shape == (height, width)
        and truth.dtype == bool
        and (bool(truth.any()) if positive else not bool(truth.any())),
        "Precompression AI truth or explicit zero control truth required",
    )
    for row in (candidate, baseline):
        require(
            row["input_sha256"] == input_sha
            and row["source_rgb_sha256"] == rgb_sha
            and row["source_size"] == [width, height]
            and row["positive_truth_png_sha256"] == expected_truth_sha
            and row["negative_truth_is_zero"] is (not positive),
            "Input bytes, RGB, truth or source coordinates differ",
        )
    raw = load_map(candidate, raw_root, (height, width))
    baseline_raw = load_map(baseline, raw_root, (384, 384) if candidate["format"] == "native" else (512, 512))
    if candidate["format"] == "native":
        gate = int(np.count_nonzero(baseline_raw > splice.PIXEL_THRESHOLD)) >= splice.MINIMUM_ACTIVE_CELLS
        threshold = splice.PIXEL_THRESHOLD
    else:
        gate = fused.candidate_eligibility(baseline["image_score"])["eligible"]
        threshold = 0.5
    require(
        type(baseline["image_gate"]) is bool
        and baseline["image_gate"] == gate
        and baseline["pixel_threshold"] == threshold,
        "Baseline's actual production gate or threshold changed",
    )
    measured = {
        "status": "ok",
        **score_map(baseline_raw, gate, threshold, data, pixels, truth),
        "image_gate": gate,
    }
    zero = mask_metrics(np.zeros_like(truth), truth)
    by_pixel, candidate_measurements = {}, {}
    for name, policy in policies().items():
        pixel = policy["pixel"]
        if pixel not in by_pixel:
            by_pixel[pixel] = score_map(raw, True, pixel, data, pixels, truth)["metrics"]
        eligible = image_gate(raw, policy)
        candidate_measurements[name] = {
            "status": "ok",
            "image_gate": eligible,
            "metrics": by_pixel[pixel] if eligible else zero,
            "ungated": by_pixel[pixel],
        }
    return {
        "source_id": candidate["source_id"],
        "component": candidate["component"],
        "role": "calibration",
        "kind": candidate["kind"],
        "format": candidate["format"],
        "baseline": measured,
        "candidate": candidate_measurements,
    }


def phase(trial, fmt, number):
    from scripts.image_regional_training_v26 import PAIRS, source_views

    require(fmt in {"native", "jpeg75"}, "Known format required")
    cv2.setNumThreads(1)
    plan, tasks, raw_root = admission(trial)
    selected = phase_tasks(tasks, number)
    candidate, baseline = (
        producer_rows(trial, arm, fmt, number, selected) for arm in ("candidate", "baseline")
    )
    out = trial / f"readout-v43-{fmt}-{number:03d}"
    out.mkdir(exist_ok=False)
    cases = {r["source_id"]: r for r in plan["rows"]}
    qualified = {
        r["source_id"]: r
        for r in (json.loads(line) for line in (PAIRS / "qualification/pairs.jsonl").read_text().splitlines())
    }
    previous, views, source_truth = None, None, None
    with (out / "rows.jsonl").open("x") as journal:
        for a, b in zip(candidate, baseline, strict=True):
            try:
                sid = a["source_id"]
                if sid != previous:
                    edited, controls, source_truth = source_views(cases[sid], qualified[sid])
                    previous, views = sid, {"ai": edited, **controls}
                data, image = encoded_view(views[a["kind"]], fmt)
                truth = source_truth > 0 if a["kind"] == "ai" else np.zeros(source_truth.shape, bool)
                result = pair_metrics(
                    a, b, raw_root, data, np.asarray(image), truth, qualified[sid]["truth_png_sha256"]
                )
                journal.write(json.dumps(result, allow_nan=False) + "\n")
                journal.flush()
            except BaseException as error:
                journal.write(
                    json.dumps(
                        {k: a[k] for k in ("source_id", "component", "kind", "format")}
                        | {"status": "failed", "error_type": type(error).__name__}
                    )
                    + "\n"
                )
                journal.flush()
                raise
    write(
        out / "completed.json",
        {
            "schema": "chipmunk.regional-calibration-map-readout/43",
            "format": fmt,
            "phase": number,
            "views": len(selected),
            "rows_sha256": sha(out / "rows.jsonl"),
            "execution_freeze_sha256": sha(trial / "freeze.json"),
            "neural_forwards": 0,
            "serving_admitted": False,
        },
    )


def outcome_counts(rows, policy):
    if policy is None:
        return {
            "selected_policy": None,
            "outcomes": None,
            "whole_file_originality_established": False,
            "reason": "No false-alarm-matched localization policy was eligible; do not replace this with zero detections.",
        }
    positives = [measurement(row, policy) for row in rows if row["kind"] == "ai"]
    missed = sum(m["predicted_positive_pixels"] == 0 for m in positives)
    not_useful = sum(not (m["iou"] >= 0.30 and m["recall"] >= 0.50) for m in positives)
    controls = {}
    for kind in ("original", "tone", "blur"):
        values = [measurement(row, policy) for row in rows if row["kind"] == kind]
        flagged = sum(m["predicted_positive_pixels"] > 0 for m in values)
        controls[kind] = {
            "views": len(values),
            "flagged": flagged,
            "flagged_fraction": flagged / len(values) if values else None,
        }
    return {
        "selected_policy": policy,
        "outcomes": {
            "edited_photos": len(positives),
            "edited_photos_without_retained_region": missed,
            "missed_region_fraction": missed / len(positives) if positives else None,
            "edited_photos_not_usefully_localized": not_useful,
            "not_usefully_localized_fraction": not_useful / len(positives) if positives else None,
            "controls": controls,
        },
        "whole_file_originality_established": False,
        "scope": "Per-photo descriptive denominators; photos within a source group are correlated. Primary comparisons and uncertainty use the frozen group-level method. No retained region is not an original-file verdict.",
    }


def seal(trial):
    admission(trial)
    summaries = {}
    for fmt in ("native", "jpeg75"):
        rows = []
        for number in range(7):
            name = f"readout-v43-{fmt}-{number:03d}"
            part = trial / name
            result = read(part / "completed.json")
            profile_root = trial / (name + "-profile")
            profile = read(profile_root / "profile.json")
            require(
                profile["module"] == "scripts.replay_regional_calibration_v43"
                and profile["arguments"] == ["phase", str(trial), fmt, str(number)]
                and profile["child_exit_code"] == 0
                and profile["guard"] is None
                and profile["sample_errors"] == 0
                and not profile["resource_limit_observations"],
                "Incomplete or failed map readout cannot select a policy",
            )
            require(
                sha(profile_root / "resource-samples.jsonl") == profile["resource_journal_sha256"],
                "Readout resource observations changed",
            )
            require(
                result["format"] == fmt
                and result["phase"] == number
                and result["views"] == (24 if number < 6 else 12)
                and sha(part / "rows.jsonl") == result["rows_sha256"]
                and result["execution_freeze_sha256"] == sha(trial / "freeze.json")
                and result["neural_forwards"] == 0,
                "Map-readout identity or rows changed",
            )
            rows.extend(json.loads(line) for line in (part / "rows.jsonl").read_text().splitlines())
        require(len(rows) == 156, "All calibration views must remain in selection")
        selected = calibrate(rows)
        summaries[fmt] = {
            "coverage": coverage(rows, policies()),
            "calibration": selected,
            "outcome_counts": {
                "baseline": outcome_counts(rows, "baseline"),
                "candidate": outcome_counts(rows, selected["selected_policy"]),
            },
        }
    write(
        trial / "calibration-v43-summary.json",
        {
            "schema": "chipmunk.regional-calibration-summary/43",
            "formats": summaries,
            "execution_freeze_sha256": sha(trial / "freeze.json"),
            "neural_forwards": 0,
            "probability_calibration_performed": False,
            "serving_admitted": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("phase")
    replay.add_argument("trial", type=Path)
    replay.add_argument("format", choices=("native", "jpeg75"))
    replay.add_argument("number", type=int)
    commands.add_parser("seal").add_argument("trial", type=Path)
    status = commands.add_parser("coverage")
    status.add_argument("trial", type=Path)
    status.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "phase":
        phase(args.trial, args.format, args.number)
    elif args.command == "seal":
        seal(args.trial)
    else:
        producer_coverage(args.trial, args.output)
