"""Read the exact v112 selection packets for a model-free landmark support transfer."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median, quantiles

from scripts import video_appearance_corrected_v112 as old
from scripts import video_appearance_readout_v111 as prior
from scripts.video_appearance_plan_v107 import CONDITIONS

ROOT, PANEL = old.ROOT, old.PANEL
NAME = "video-landmark-transfer-v255"
PUBLIC = PANEL / f"{NAME}-freeze.json"
PREVIOUS = {
    old.PUBLIC: "30d2ae73d9edf48e1f30a7b5e7f90b965fa0190d7f190098b9559f9647e46489",
    old.READOUT_PUBLIC: "1f2750079e22dbf43548879f149f34520bd5cf0e85380b0b1d01a3423a0e2ac4",
    old.RESULT_PANEL
    / "execution-summary-v112.json": "f7319c36332b07cfaefb077822a945d530419fe5175467e4e032e32dfea15245",
    old.RESULT_PANEL
    / "video-appearance-selection-result-v111.json": "c3c9a90917906297578697e728877bbf2ebb6181f9daacf84201873b993dd1bb",
}
CODE = (
    f"backend/scripts/{NAME.replace('-', '_')}.py",
    "backend/scripts/video_landmark_support.py",
    "backend/scripts/video_face_confound_v222.py",
    "backend/scripts/video_appearance_readout_v111.py",
    "backend/tests/test_video_landmark_transfer_v255.py",
    f"docs/{NAME}-method.md",
)
SWAPS = ("swap_a", "swap_b")
CONTROLS = tuple(condition for condition in CONDITIONS if condition not in SWAPS)
PHASES = ("before", "during", "after")
CENTERS = ("before_centers", "middle_centers", "after_centers")
METRICS = (
    *(f"{role}_median_eye_box_width" for role in PHASES),
    "during_minimum_eye_box_width",
    "during_minus_flank_eye_box_width",
    "before_to_during_geometry_rms_eye_units",
    "after_to_during_geometry_rms_eye_units",
    "before_to_after_geometry_rms_eye_units",
)
read, sha, require = prior.read, prior.sha, prior.require


def load_bound_packets():
    require(all(sha(path) == digest for path, digest in PREVIOUS.items()), "Published v112 receipts changed")
    lock = read(old.PUBLIC)
    plan_path = old.PREP / "plan.json"
    original_result = read(old.RESULT_PANEL / "video-appearance-selection-result-v111.json")
    require(
        read(old.TRIAL / "freeze.json") == lock
        and read(old.TRIAL / "readout/freeze.json") == read(old.READOUT_PUBLIC)
        and sha(plan_path) == lock["derived_plan_sha256"]
        and sha(prior.FITTED) == lock["fitted_models_sha256"]
        and sha(prior.HEADS) == lock["local_heads_sha256"]
        and sha(Path(prior.__file__))
        == lock["code_sha256"]["backend/scripts/video_appearance_readout_v111.py"],
        "Exact frozen v112 plan, scalar maps and review code required",
    )
    plan = read(plan_path)
    local = read(prior.HEADS)["specialists"]["face-appearance|single-face"]
    model = read(prior.FITTED)["models"]["appearance_only"]
    cases = []
    receipts = original_result["batches"]
    require(
        len(receipts) == len(plan["batch_group_indices"]) == 8, "Eight exact v112 selection batches required"
    )
    for index, (receipt, group_indices) in enumerate(zip(receipts, plan["batch_group_indices"], strict=True)):
        folder = old.TRIAL / f"extract-{index:02d}"
        binding = old.TRIAL / f"bind-{index:02d}/output"
        profile_path, result_path = folder / "profile.json", folder / "output/result.json"
        journal, bound_path = folder / "output/responses.jsonl", binding / "bound-cases.json"
        profile, result = read(profile_path), read(result_path)
        require(
            receipt["batch"] == result["batch"] == index
            and receipt["result_sha256"] == profile["worker_receipt_sha256"] == sha(result_path)
            and receipt["responses_sha256"] == result["responses_sha256"] == sha(journal)
            and receipt["profile_sha256"] == sha(profile_path)
            and result["freeze_sha256"] == PREVIOUS[old.PUBLIC]
            and result["status"] == "complete"
            and profile["child_exit_code"] == profile["sample_errors"] == 0
            and profile["guard"] is None
            and result["binding_receipt_sha256"] == sha(binding / "result.json")
            and read(binding / "result.json")["bound_cases_sha256"] == sha(bound_path),
            "Published v112 batch and source commitments must match saved files",
        )
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        bound = read(bound_path)
        expected = [case for case in plan["cases"] if case["group_index"] in group_indices]
        require(
            len(rows) == len(bound) == receipt["cases"] == result["cases"]
            and [entry["planned"] for entry in bound] == expected,
            "No missing or substituted v112 selection packets",
        )
        for row, source in zip(rows, bound, strict=True):
            prior.review_case(row, source, local, model, lock["raw_point"], lock["candidate_point"])
            cases.append(row)
    groups = defaultdict(list)
    for case in cases:
        groups[case["group_index"]].append(case["condition"])
    require(
        len(cases) == 138
        and len(groups) == len({row["source_group"] for row in cases}) == 23
        and all(Counter(conditions) == Counter(CONDITIONS) for conditions in groups.values())
        and all(row["source_role"] == "selection" and len(row["units"]) == 5 for row in cases),
        "Exactly 23 selection groups with six variants and five windows each are required",
    )
    return cases, lock["raw_point"]


def protocol():
    return {
        "schema": "chipmunk.video-landmark-transfer-freeze/255",
        "source_receipts": {str(path.relative_to(ROOT)): digest for path, digest in PREVIOUS.items()},
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "source_split": "v112_selection",
        "planned_source_groups": 23,
        "planned_variants": 138,
        "planned_windows": 690,
        "metrics": list(METRICS),
        "paired_comparison": "existing_raw_swap_flag_minus_same_group_and_window_control",
        "new_threshold_or_fit_permitted": False,
        "new_neural_calls_permitted": False,
        "reserved_confirmation_access_permitted": False,
    }


def packet(unit, frames):
    centers = {role: unit[key] for role, key in zip(PHASES, CENTERS, strict=True)}
    by_role = {
        role: {i + offset for i in values for offset in (-1, 0, 1)} for role, values in centers.items()
    }
    location = unit["location"]
    require(
        tuple(map(len, by_role.values())) == (6, 9, 6)
        and by_role["during"] == set(location["sampled_frames"])
        and by_role["before"] | by_role["after"] == set(location["context_frames"])
        and len(set.union(*by_role.values())) == 21,
        "Exact ordered window frame support required",
    )
    return {
        "centers": centers,
        "frame_inputs": [
            {
                "frame_index": index,
                "bbox": dict(zip(("x", "y", "w", "h"), frames[index]["bbox"], strict=True)),
                "landmarks": frames[index]["landmarks"],
            }
            for index in sorted(set.union(*by_role.values()))
        ],
    }


def metric_values(measurement):
    eye = measurement["eye_box_width"]
    return {
        **{f"{role}_median_eye_box_width": eye[role]["median"] for role in PHASES},
        "during_minimum_eye_box_width": eye["during"]["minimum"],
        "during_minus_flank_eye_box_width": measurement["during_minus_flank_eye_box_width"],
        **{
            f"{pair}_geometry_rms_eye_units": value
            for pair, value in measurement["geometry_rms_eye_units"].items()
        },
    }


def observations(cases, point):
    from scripts.video_landmark_support import measure

    windows = {}
    for case in cases:
        frames = {frame["index"]: frame for frame in case["frame_inputs"]}
        for ordinal, unit in enumerate(case["units"]):
            if unit["status"] == "ok":
                measurement = measure(packet(unit, frames))
            else:
                measurement = {"status": "unavailable", "reason": "v112_window_unassessed"}
            windows[case["group_index"], case["condition"], ordinal] = {
                "reason": measurement.get("reason"),
                "metrics": metric_values(measurement) if measurement["status"] == "measured" else None,
                "current_appearance_measured": unit["status"] == "ok",
                "current_flag": unit["raw"] is not None and unit["raw"] > point,
                "location": unit["location"],
            }
    return windows


def distribution(values):
    if not values:
        return None
    low, middle, high = quantiles(values, n=4, method="inclusive") if len(values) > 1 else values * 3
    return {"minimum": min(values), "q25": low, "median": middle, "q75": high, "maximum": max(values)}


def summarize(windows):
    groups = defaultdict(list)
    for (group, _, _), row in windows.items():
        groups[group].append(row)
    measured = {
        group: [row["metrics"] for row in rows if row["metrics"] is not None]
        for group, rows in groups.items()
    }
    return {
        "source_groups": len(groups),
        "groups_with_geometry": sum(bool(rows) for rows in measured.values()),
        "groups_with_geometry_for_all_windows": sum(
            len(measured[group]) == len(rows) for group, rows in groups.items()
        ),
        "planned_windows": len(windows),
        "current_appearance_measured_windows": sum(
            row["current_appearance_measured"] for row in windows.values()
        ),
        "geometry_measured_windows": sum(map(len, measured.values())),
        "unavailable_windows_by_reason": dict(
            Counter(row["reason"] for row in windows.values() if row["reason"])
        ),
        "metrics_across_group_medians": {
            metric: distribution(
                [median([row[metric] for row in rows]) for rows in measured.values() if rows]
            )
            for metric in METRICS
        },
    }


def paired(windows, swap, control):
    pairs = []
    missing = Counter()
    planned = 0
    for (group, condition, ordinal), row in windows.items():
        if condition != swap or not row["current_flag"]:
            continue
        planned += 1
        comparison = windows[group, control, ordinal]
        require(
            row["location"] == comparison["location"], "Paired controls must use the same scheduled window"
        )
        if row["metrics"] is None:
            missing["flagged_swap_geometry_unavailable"] += 1
        if comparison["metrics"] is None:
            missing["control_geometry_unavailable"] += 1
        if row["metrics"] is not None and comparison["metrics"] is not None:
            pairs.append(
                (
                    group,
                    {metric: row["metrics"][metric] - comparison["metrics"][metric] for metric in METRICS},
                )
            )
    return {
        "existing_flag_windows_planned_for_pairing": planned,
        "matched_geometry_windows": len(pairs),
        "matched_source_groups": len({group for group, _ in pairs}),
        "missing_endpoints": dict(missing),
        "swap_minus_control": {
            metric: {
                "negative": sum(row[metric] < 0 for _, row in pairs),
                "zero": sum(row[metric] == 0 for _, row in pairs),
                "positive": sum(row[metric] > 0 for _, row in pairs),
                "distribution": distribution([row[metric] for _, row in pairs]),
            }
            for metric in METRICS
        },
    }


def readout(cases, point, frozen):
    windows = observations(cases, point)
    by_condition = {}
    for condition in CONDITIONS:
        variant_windows = {key: row for key, row in windows.items() if key[1] == condition}
        flags = {key: row for key, row in variant_windows.items() if row["current_flag"]}
        by_condition[condition] = {"all": summarize(variant_windows), "existing_flags": summarize(flags)}
    return {
        "schema": "chipmunk.video-landmark-transfer-result/255",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "source_split": frozen["source_split"],
        "source_groups": frozen["planned_source_groups"],
        "variants": len(cases),
        "windows": len(windows),
        "by_condition": by_condition,
        "paired_at_existing_swap_flags": {
            swap: {control: paired(windows, swap, control) for control in CONTROLS} for swap in SWAPS
        },
        "new_neural_calls": 0,
        "reserved_confirmation_groups_accessed": 0,
        "serving_change": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "readout"))
    command = parser.parse_args().command
    frozen = protocol()
    if command == "readout":
        require(read(PUBLIC) == frozen, "Exact prospective landmark comparison freeze required")
    cases, point = load_bound_packets()
    if command == "freeze":
        prior.write(PUBLIC, frozen)
    value = (
        readout(cases, point, frozen)
        if command == "readout"
        else {
            "status": "frozen" if command == "freeze" else "ready_for_freeze",
            "bound_variants": len(cases),
            "new_neural_calls": 0,
        }
    )
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
