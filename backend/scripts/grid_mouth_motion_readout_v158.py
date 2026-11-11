"""Read fixed mouth-motion curves from the completed geometry observer."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import types
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import grid_mouth_motion_arithmetic_v158 as arithmetic
from scripts import grid_transfer_plan_v143 as planning

ROOT, BASE, PANEL, ARCHIVE = planning.ROOT, planning.BASE, planning.PANEL, planning.ARCHIVE
TRIAL = BASE / "grid-mouth-motion-readout-v158"
CACHE = ARCHIVE / "grid-mouth-motion-readout-v158"
PUBLIC = PANEL / "grid-mouth-motion-readout-freeze-v158.json"
CODE = (
    "backend/scripts/grid_mouth_motion_readout_v158.py",
    "backend/scripts/grid_mouth_motion_arithmetic_v158.py",
    "backend/tests/test_grid_mouth_motion_readout_v158.py",
    "backend/tests/test_grid_mouth_motion_arithmetic_v158.py",
    "docs/grid-mouth-motion-readout-v158-method.md",
    "docs/grid-achieved-motion-validation-design.md",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/audio_tree_sampler_v64.py",
)
BRANCHES = ("musetalk", "liveportrait")
PAIR_SPECS = tuple(
    (f"{branch}_earlier_320ms", target)
    for branch in BRANCHES
    for target in (f"{branch}_source_time", f"resample_{branch}", f"brightness_{branch}")
)
FAMILIES = {
    "original": ("original",),
    **{branch: (f"{branch}_source_time", f"{branch}_earlier_320ms") for branch in BRANCHES},
    "resample": tuple(f"resample_{branch}" for branch in BRANCHES),
    "brightness": tuple(f"brightness_{branch}" for branch in BRANCHES),
    "genuine_frame_retime": ("genuine_frame_retime",),
    "audio_delay": ("audio_delay_320ms",),
}
MOTION_STATES = (
    "unavailable",
    "weak_reference",
    "static_output",
    "source_time_compatible",
    "earlier_compatible",
    "search_boundary",
    "ambiguous",
    "other_or_poor_match",
)
PAIR_STATES = ("unavailable", "gain", "loss", "within_dead_band")
UNAVAILABLE_REASONS = {
    "construction_unavailable",
    "original_reference_unavailable",
    "insufficient_full_support",
}
GEOMETRY_STATES = {
    "geometry_available",
    "unavailable_case",
    "unavailable_no_face",
    "unavailable_multiple_faces",
    "unavailable_mesh",
    "unavailable_geometry",
}
EYE_PATCH_STATES = {
    "identical_eye_pixels",
    "different_eye_pixels",
    "patch_outside_picture",
    "unavailable_geometry",
}
REFERENCE_FRAMES = tuple(range(15, 59))
VARIANT_FRAMES = tuple(range(25, 49))
EDIT = tuple(range(*planning.EDIT))
CONTROLS = ("fixed_repeat", "genuine_frame_retime", "audio_delay_320ms")
CONTROL_COUNTS = (
    "planned_pairs",
    "unavailable_pairs",
    "mismatched_pairs",
    "matching_geometry_pairs",
    "matching_unavailable_geometry_pairs",
)
MIN_PUBLIC_COMPONENTS = 5
read, write, sha, require = planning.read, planning.write, planning.sha, planning.require


def exact_time(value):
    require(type(value) in (int, str), "Exact rational timestamp required")
    return Fraction(value)


def source_times(source):
    row = source["input"]
    if row["status"] != planning.reviewed.AVAILABLE:
        return None
    table = row["native_video_table"]
    require(
        len(table["pts"]) == 75
        and not table["interpolated"]
        and not table["truncated"]
        and all(type(pts) is int for pts in table["pts"]),
        "All observed native picture timestamps required",
    )
    times = tuple(exact_time(pts) * exact_time(table["time_base"]) for pts in table["pts"])
    require(
        all(b - a == arithmetic.STEP for a, b in zip(times, times[1:])),
        "Frozen native 25 Hz support required throughout the clip",
    )
    return times


def empty_motion(reason):
    return {
        "status": "unavailable",
        "unavailable_reason": reason,
        "near_best_lags": [],
        "full_support_times": [],
        "target_support_times": [],
        "curve": {"measured": False, "supported_positions": 0, "adjacent_pairs": 0},
        "target_only_contrast": {"measured": False, "supported_positions": 0, "adjacent_pairs": 0},
        "entry_velocity_norm": None,
        "exit_velocity_norm": None,
    }


def assess_trajectories(source, geometry):
    """Measure all fixed trajectories and six pairs on their admitted clocks."""
    times = source_times(source)
    require(tuple(geometry) == planning.CASES, "Every planned case in canonical order required")
    reference = geometry["original"]
    edit = tuple(times[index] for index in EDIT) if times is not None else ()
    rows = []
    for condition, item in geometry.items():
        if times is None or item["construction_status"] == "unavailable":
            motion = empty_motion("construction_unavailable")
        elif reference["construction_status"] == "unavailable":
            motion = empty_motion("original_reference_unavailable")
        else:
            motion = arithmetic.measure(item["trajectory"], reference["trajectory"], edit)
            motion["unavailable_reason"] = (
                None if motion["curve"]["measured"] else "insufficient_full_support"
            )
        rows.append(
            {
                "condition": condition,
                "clip_ordinal": source["ordinal"],
                "source_slot": source["source_slot"],
                "component_index": source["component_index"],
                **{
                    key: item[key]
                    for key in (
                        "case_number",
                        "construction_status",
                        "construction_reason",
                        "neural_provenance_verified",
                        "observed_neural_pixel_change",
                        "geometry_counts",
                        "eye_drift_px",
                        "eye_patch_counts",
                    )
                },
                **motion,
            }
        )
    first = rows[0]
    for row in rows:
        for boundary in ("entry", "exit"):
            key = f"{boundary}_velocity_norm"
            row[f"{boundary}_velocity_delta_vs_original"] = (
                row[key] - first[key] if row[key] is not None and first[key] is not None else None
            )
    pairs = []
    for earlier, comparator in PAIR_SPECS:
        left, right = geometry[earlier], geometry[comparator]
        if times is None or any(
            item["construction_status"] == "unavailable" for item in (left, right, reference)
        ):
            contrast = {"status": "unavailable", "supported_positions": 0, "adjacent_pairs": 0}
        else:
            contrast = arithmetic.paired_preference(
                left["trajectory"], right["trajectory"], reference["trajectory"], edit
            )
        pairs.append(
            {
                "earlier": earlier,
                "comparator": comparator,
                "component_index": source["component_index"],
                "clip_ordinal": source["ordinal"],
                "source_slot": source["source_slot"],
                **contrast,
            }
        )
    return rows, pairs


def distribution(values):
    """Quartiles use linear interpolation over sufficient grouped support."""
    records = list(values)
    numbers = [number for _, number in records]
    require(all(math.isfinite(number) for number in numbers), "Finite aggregate values required")
    components = len({component for component, _ in records})
    q1, median, q3 = (
        map(float, np.quantile(numbers, (0.25, 0.5, 0.75), method="linear"))
        if components >= MIN_PUBLIC_COMPONENTS
        else (None, None, None)
    )
    return {"observations": len(numbers), "components": components, "q1": q1, "median": median, "q3": q3}


def case_aggregate(rows):
    states = Counter(row["status"] for row in rows)
    reason_counts = Counter(
        row["unavailable_reason"] for row in rows if row["unavailable_reason"] is not None
    )
    target_only = [
        row for row in rows if row["target_only_contrast"]["measured"] and not row["curve"]["measured"]
    ]
    eligible_components = defaultdict(list)
    for row in rows:
        require(
            row["status"] in MOTION_STATES
            and (row["unavailable_reason"] is None or row["unavailable_reason"] in UNAVAILABLE_REASONS)
            and row["geometry_counts"].keys() <= GEOMETRY_STATES
            and row["eye_patch_counts"].keys() <= EYE_PATCH_STATES,
            "Only fixed aggregate status keys may be published",
        )
        eligible_components[row["component_index"]].append(row["curve"]["measured"])
    return {
        "planned_cases": len(rows),
        "components": len(eligible_components),
        "construction_unavailable": sum(row["construction_status"] == "unavailable" for row in rows),
        "neural_provenance_cases": sum(row["neural_provenance_verified"] for row in rows),
        "observable_neural_cases": sum(row["observed_neural_pixel_change"] for row in rows),
        "full_curves_measured": sum(row["curve"]["measured"] for row in rows),
        "component_weighted_full_curve_rate": (
            float(np.mean([sum(values) / len(values) for values in eligible_components.values()]))
            if eligible_components
            else None
        ),
        "full_support_positions": distribution(
            (row["component_index"], row["curve"]["supported_positions"]) for row in rows
        ),
        "full_support_adjacent_pairs": distribution(
            (row["component_index"], row["curve"]["adjacent_pairs"]) for row in rows
        ),
        "target_contrasts_measured": sum(row["target_only_contrast"]["measured"] for row in rows),
        "target_only_measured_without_full_curve": len(target_only),
        "target_only_weak_reference": sum(
            row["target_only_contrast"]["weak_reference"] for row in target_only
        ),
        "target_only_earlier_preference": distribution(
            (row["component_index"], row["target_only_contrast"]["earlier_preference"]) for row in target_only
        ),
        "states": {status: states[status] for status in MOTION_STATES},
        "unavailable_reasons": dict(sorted(reason_counts.items())),
        "compatible_lag_bounds_frames": {
            status: {
                name: distribution(
                    (row["component_index"], row["near_best_lags"][index])
                    for row in rows
                    if row["status"] == status
                )
                for name, index in (("first", 0), ("last", -1))
            }
            for status in ("source_time_compatible", "earlier_compatible")
        },
        "boundaries": {
            key: distribution((row["component_index"], row[key]) for row in rows if row[key] is not None)
            for key in (
                "entry_velocity_norm",
                "exit_velocity_norm",
                "entry_velocity_delta_vs_original",
                "exit_velocity_delta_vs_original",
            )
        },
        "geometry_observations": dict(
            sorted(sum((Counter(row["geometry_counts"]) for row in rows), Counter()).items())
        ),
        "eye_patch_observations": dict(
            sorted(sum((Counter(row["eye_patch_counts"]) for row in rows), Counter()).items())
        ),
        "unchanged_eye_patch_drift_px": distribution(
            (row["component_index"], value) for row in rows for value in row["eye_drift_px"]
        ),
    }


def pair_aggregate(rows):
    counts = Counter(row["status"] for row in rows)
    measured = [row for row in rows if row["status"] != "unavailable"]
    by_component = defaultdict(list)
    for row in measured:
        by_component[row["component_index"]].append(row["earlier_preference_difference"])
    return {
        "planned_pairs": len(rows),
        "measured_pairs": len(measured),
        "statuses": {status: counts[status] for status in PAIR_STATES},
        "weak_reference_measured_pairs": sum(row["weak_reference"] for row in measured),
        "weak_reference_statuses": {
            status: sum(row["status"] == status and row["weak_reference"] for row in measured)
            for status in PAIR_STATES
            if status != "unavailable"
        },
        "shared_support_positions": distribution(
            (row["component_index"], row["supported_positions"]) for row in rows
        ),
        "shared_support_adjacent_pairs": distribution(
            (row["component_index"], row["adjacent_pairs"]) for row in rows
        ),
        "earlier_preference_difference": distribution(
            (row["component_index"], row["earlier_preference_difference"]) for row in measured
        ),
        "component_medians": distribution(
            (component, float(np.median(values))) for component, values in by_component.items()
        ),
    }


def aggregate_rows(rows, pairs):
    return {
        "planned_cases": len(rows),
        "planned_pairs": len(pairs),
        "source_components": len({row["component_index"] for row in rows}),
        "by_condition": {
            name: case_aggregate([row for row in rows if row["condition"] == name]) for name in planning.CASES
        },
        "by_family": {
            family: case_aggregate([row for row in rows if row["condition"] in conditions])
            for family, conditions in FAMILIES.items()
        },
        "paired_comparisons": [
            {
                "earlier": earlier,
                "comparator": comparator,
                **pair_aggregate(
                    [row for row in pairs if row["earlier"] == earlier and row["comparator"] == comparator]
                ),
            }
            for earlier, comparator in PAIR_SPECS
        ],
    }


def observer_module():
    from scripts import grid_mouth_observer_v157

    return grid_mouth_observer_v157


def native_modules():
    from scripts import grid_native_clips_v149, grid_source_speech_v147

    return grid_native_clips_v149, grid_source_speech_v147


def sealed_plan():
    bindings = {}
    freeze = planning.frozen_context(planning.TRIAL, planning.PUBLIC, bindings)
    stage = planning.TRIAL / "bind-00"
    result = planning.reviewed.stage(stage, sha(planning.PUBLIC), "grid_transfer_plan_v143.py", 0, bindings)
    plan_file = stage / "output/private-plan.json"
    plan = planning.bind_metadata(plan_file, bindings)
    all_sources = plan["sources"]
    sources = [source for source in all_sources if source["role"] == "pilot"]
    require(
        result["schema"] == "chipmunk.grid-transfer-plan-result/143"
        and result["new_media_reads"] == 0
        and result["private_plan_sha256"] == sha(plan_file)
        and freeze["case_order"] == plan["case_order"] == list(planning.CASES)
        and plan["schema"] == "chipmunk.grid-native-transfer-plan/143"
        and plan["edit_frames_half_open"] == list(planning.EDIT)
        and [s["ordinal"] for s in all_sources] == list(range(32))
        and len(sources) == freeze["actual_pilot_clips"] == 16
        and result["pilot_cases"] == freeze["actual_pilot_cases"] == 176
        and all([c["id"] for c in source["cases"]] == list(planning.CASES) for source in sources)
        and len({s["component_index"] for s in sources}) == 8
        and not {s["component_index"] for s in sources}.intersection(
            s["component_index"] for s in all_sources if s["role"] == "reserve"
        ),
        "Sealed fixed v143 pilot roster and component closure required",
    )
    return sources, bindings


def identity():
    observer = observer_module()
    clips, _ = native_modules()
    sources, bindings = sealed_plan()
    observed = planning.frozen_context(observer.TRIAL, observer.PUBLIC, bindings)
    planned = planning.frozen_context(clips.TRIAL, clips.PUBLIC, bindings)
    complete = planning.bind_metadata(observer.CONSTRUCTION_COMPLETE, bindings)
    require(
        observed["schema"] == "chipmunk.grid-mouth-observer-freeze/157"
        and observed["transfer_plan_freeze_sha256"] == sha(planning.PUBLIC)
        and observed["private_transfer_plan_sha256"]
        == sha(planning.TRIAL / "bind-00/output/private-plan.json")
        and observed["construction_freeze_sha256"] == complete["freeze_sha256"] == sha(clips.PUBLIC)
        and observed["construction_complete_sha256"] == sha(observer.CONSTRUCTION_COMPLETE)
        and complete["schema"] == "chipmunk.grid-native-clips-complete/149"
        and complete["status"] == "complete"
        and complete["complete_guarded_stages"] == 16
        and complete["planned_cases"] == 176
        and complete["guard_violations"] == complete["monitor_sample_errors"] == 0
        and observed["pilot_clips"] == len(sources) == 16
        and planned["pilot_clip_ordinals"] == [source["ordinal"] for source in sources]
        and observed["cases_per_pilot"] == len(planning.CASES) == 11
        and observed["reference_frames_half_open"] == [15, 59]
        and observed["variant_frames_half_open"] == [25, 49]
        and observed["fixed_original_repeat_frame"] == 27
        and observed["planned_observations_per_pilot"] == 285,
        "Exact observer freeze and completed matched construction contract required",
    )
    monitor = planning.reviewed.stage(
        BASE / "video-appearance-native-v108/monitor",
        None,
        "video_appearance_runtime_v108.py",
        None,
        bindings,
        monitor=True,
    )
    require(
        monitor["owned_child_exit_code"] == 0
        and monitor["owned_descendant_sampled"] is True
        and monitor["sample_helper_sha256"] == sha(ROOT / "backend/scripts/audio_tree_sampler_v64.py"),
        "Actual unchanged owned monitor qualification required",
    )
    return {
        "schema": "chipmunk.grid-mouth-motion-readout-freeze/158",
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "completed_context_metadata_sha256": bindings,
        "pilot_clips": len(sources),
        "pilot_components": 8,
        "planned_cases": 176,
        "pairs_per_clip": len(PAIR_SPECS),
        "paired_comparisons": [list(pair) for pair in PAIR_SPECS],
        "case_order": list(planning.CASES),
        "native_lags_frames": list(arithmetic.LAGS),
        "edit_frames_half_open": list(planning.EDIT),
        "minimum_supported_positions": arithmetic.MIN_FRAMES,
        "minimum_adjacent_pairs": arithmetic.MIN_PAIRS,
        "minimum_public_components_for_quartiles": MIN_PUBLIC_COMPONENTS,
        "numpy_version": importlib.metadata.version("numpy"),
        "scope": "Saved verified geometry and exact rational source-clock arithmetic only. No media, weights, detector scores, reserve scoring or phonetic claim.",
    }


def verify():
    frozen = read(PUBLIC)
    require(frozen == read(TRIAL / "freeze.json") == identity(), "Unchanged v158 frozen readout required")
    return frozen


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "planned_cases": 176, "planned_pairs": 96}), flush=True)


def source_for(ordinal):
    sources, _ = sealed_plan()
    require(type(ordinal) is int, "Original pilot ordinal required")
    source = next((source for source in sources if source["ordinal"] == ordinal), None)
    require(source is not None, "Reserve or unknown source cannot enter motion readout")
    return source


def pilot_ordinals():
    sources, _ = sealed_plan()
    return [source["ordinal"] for source in sources]


def construction_metadata(source, clips):
    ordinal = source["ordinal"]
    tag = f"clip-{ordinal:02d}"
    stage, folder = clips.TRIAL / tag / "bind-00", clips.CACHE / tag
    result = planning.reviewed.stage(stage, sha(clips.PUBLIC), "grid_native_clips_v149.py", 0, {})
    cases_file, input_file = folder / "cases.jsonl", folder / "input-binding.json"
    cases, binding = planning.reviewed.lines(cases_file), read(input_file)
    require(
        result["schema"] == "chipmunk.grid-native-clips-result/149"
        and result["clip_ordinal"] == ordinal
        and result["private_case_journal_sha256"] == sha(cases_file)
        and result["private_input_binding_sha256"] == sha(input_file)
        and binding["source"] == source,
        "Exact completed v149 case and source binding required",
    )
    clips.validate_case_inventory(cases, binding, folder)
    require(
        all(result[k] == v for k, v in clips.case_counts(cases).items()),
        "Actual constructed case totals differ",
    )
    return cases, {
        "construction_case_journal_sha256": sha(cases_file),
        "construction_result_sha256": sha(stage / "output/result.json"),
        "construction_profile_sha256": sha(stage / "profile.json"),
        "construction_resource_journal_sha256": sha(stage / "resources.jsonl"),
    }


def observation_metadata(source):
    observer = observer_module()
    clips, speech = native_modules()
    cases, construction = construction_metadata(source, clips)
    expected = observer.schedule(source, cases)
    expected["bindings"] = construction
    tag = f"clip-{source['ordinal']:02d}"
    stage, folder = observer.TRIAL / tag / "authored", observer.CACHE / tag
    out = stage / "output"
    calls = read(out / "result.json")["new_neural_calls"]
    require(type(calls) is int and 0 <= calls <= 285, "Actual observer task count required")
    result = speech.qualification_profile(stage, calls, {}, "grid_mouth_observer_v157.py")
    artifacts = {
        "private_input_binding_sha256": folder / "input-binding.json",
        "private_observation_journal_sha256": folder / "observations.jsonl",
        "private_control_checks_sha256": folder / "control-checks.json",
        "private_face_progress_sha256": out / "face-progress.jsonl",
    }
    digests = {key: sha(file) for key, file in artifacts.items()}
    require(
        result["schema"] == "chipmunk.grid-mouth-observer-result/157"
        and result["freeze_sha256"] == sha(observer.PUBLIC)
        and result["clip_ordinal"] == source["ordinal"]
        and result["defaults_changed"] is False
        and all(result[k] == v for k, v in digests.items())
        and read(artifacts["private_input_binding_sha256"]) == expected,
        "Exact successful v157 journal, source clock and input schedule required",
    )
    rows = planning.reviewed.lines(artifacts["private_observation_journal_sha256"])
    progress = planning.reviewed.lines(artifacts["private_face_progress_sha256"])
    require(
        observer.validate_observations(expected, rows, progress) == calls, "Observer call journal differs"
    )
    controls = observer.control_checks(expected, rows)
    require(
        controls == read(artifacts["private_control_checks_sha256"])
        and all(c["mismatched_pairs"] == 0 for c in controls.values())
        and all(result[k] == v for k, v in observer.observed_counts(rows, calls, controls).items()),
        "Actual observation statuses, repeat and exact control outcomes differ",
    )
    return (
        cases,
        rows,
        controls,
        {
            **construction,
            "observer_result_sha256": sha(out / "result.json"),
            "observer_profile_sha256": sha(stage / "profile.json"),
            "observer_resource_journal_sha256": sha(stage / "resources.jsonl"),
            "observer_artifact_sha256": digests,
        },
    )


def finite_hex(value):
    require(type(value) is str, "Canonical hexadecimal measurement required")
    measured = float.fromhex(value)
    require(
        math.isfinite(measured) and measured.hex() == value,
        "Canonical finite hexadecimal measurement required",
    )
    return measured


def trajectories(source, cases, observations):
    times = source_times(source)
    require(
        times is not None
        and [case["condition"] for case in cases] == list(planning.CASES)
        and {case["case_number"] for case in cases} == set(range(11)),
        "Complete exact source clock and v149 construction roster required",
    )
    require(
        len(observations) == 285
        and [row["request_ordinal"] for row in observations] == list(range(285))
        and all(type(row["repeat"]) is bool for row in observations),
        "Every observed request must be retained exactly once",
    )
    repeats = [row for row in observations if row["repeat"]]
    require(
        len(repeats) == 1
        and repeats[0] is observations[-1]
        and repeats[0]["case_number"] == cases[0]["case_number"]
        and repeats[0]["frame_index"] == 27,
        "One original repeat must remain outside the trajectory",
    )
    observed = defaultdict(list)
    for row in observations[:-1]:
        observed[row["case_number"]].append(row)
    geometry = {}
    require(set(observed) == set(range(11)), "Observed cases must match the complete v149 roster")
    for case in cases:
        name, number = case["condition"], case["case_number"]
        rows = observed[number]
        require(
            [row["frame_index"] for row in rows]
            == list(REFERENCE_FRAMES if name == "original" else VARIANT_FRAMES),
            "Observed native indices differ from the fixed case schedule",
        )
        trajectory, drift, eye = {}, [], Counter()
        for row in rows:
            index = row["frame_index"]
            clock = times[index]
            require(
                row["source_time_seconds"] == str(clock)
                and exact_time(row["source_pts"]) * exact_time(row["source_time_base"]) == clock
                and row["case_status"] == case["status"]
                and row["case_reason"] == case["reason"],
                "Observer row clock or construction availability differs",
            )
            if case["status"] == "ready":
                require(
                    exact_time(row["pts"]) * exact_time(row["time_base"]) == clock
                    and row["rgb_sha256"] == case["expected"]["rgb_sha256"][index],
                    "Observer picture does not match its bound native construction",
                )
            available = row["status"] == "geometry_available"
            trajectory[clock] = (
                arithmetic.MouthShape(
                    tuple(finite_hex(value) for value in row["descriptor_hex"]),
                    finite_hex(row["eye_distance_px_hex"]),
                )
                if available
                else None
            )
            if "eye_patch_drift" in row:
                patch = row["eye_patch_drift"]
                eye[patch["status"]] += 1
                if patch["status"] == "identical_eye_pixels":
                    value = finite_hex(patch["rms_px_hex"])
                    require(value >= 0, "Nonnegative eye drift required")
                    drift.append(value)
        geometry[name] = {
            "case_number": number,
            "construction_status": case["status"],
            "construction_reason": case["reason"],
            "neural_provenance_verified": case["neural_provenance_verified"],
            "observed_neural_pixel_change": case.get("observed_neural_pixel_change", False),
            "geometry_counts": dict(Counter(row["status"] for row in rows)),
            "eye_drift_px": drift,
            "eye_patch_counts": dict(eye),
            "trajectory": trajectory,
        }
    return geometry


def private_folder(ordinal):
    folder = CACHE / stage_name(ordinal)
    require(
        folder.resolve().is_relative_to(CACHE.resolve())
        and CACHE.resolve().is_relative_to(ARCHIVE.resolve()),
        "Private mouth geometry readout must stay on the owned archive",
    )
    return folder


def stage_name(ordinal):
    return "summary" if ordinal == "summary" else f"clip-{ordinal:02d}"


def stage_folder(ordinal):
    return TRIAL / stage_name(ordinal) / "bind-00"


def counts(payload):
    rows, pairs = payload["cases"], payload["pairs"]
    return {
        "planned_cases": len(rows),
        "planned_pairs": len(pairs),
        "full_curves_measured": sum(row["curve"]["measured"] for row in rows),
        "target_only_measured_without_full_curve": sum(
            row["target_only_contrast"]["measured"] and not row["curve"]["measured"] for row in rows
        ),
        "pairs_measured": sum(row["status"] != "unavailable" for row in pairs),
    }


def measure_clip(ordinal, output):
    source = source_for(ordinal)
    cases, observations, controls, bindings = observation_metadata(source)
    rows, pairs = assess_trajectories(source, trajectories(source, cases, observations))
    payload = {
        "schema": "chipmunk.grid-mouth-motion-private/158",
        "cases": rows,
        "pairs": pairs,
        "observer_exact_control_checks": controls,
        "observer_bindings": bindings,
    }
    folder = private_folder(ordinal)
    folder.mkdir(parents=True, exist_ok=False)
    file = folder / "motion.json"
    # Persisted curve dicts use canonical string keys for exact signed lags.
    write(file, payload)
    require(read(file) == json.loads(json.dumps(payload, allow_nan=False)), "Saved motion curves differ")
    result = {
        "schema": "chipmunk.grid-mouth-motion-clip-result/158",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "clip_ordinal": ordinal,
        "private_motion_sha256": sha(file),
        **counts(payload),
        "new_neural_calls": 0,
        "new_media_reads": 0,
        "defaults_changed": False,
    }
    write(output / "result.json", result)
    print(json.dumps({"status": "complete", **counts(payload)}), flush=True)


def validate_payload(source, cases, controls, bindings, payload):
    require(
        payload["schema"] == "chipmunk.grid-mouth-motion-private/158"
        and payload["observer_bindings"] == bindings
        and payload["observer_exact_control_checks"] == controls,
        "Private curves must match their admitted observer receipts",
    )
    rows, pairs = payload["cases"], payload["pairs"]
    require(
        [row["condition"] for row in rows] == list(planning.CASES)
        and [(row["earlier"], row["comparator"]) for row in pairs] == list(PAIR_SPECS)
        and all(
            row["clip_ordinal"] == source["ordinal"]
            and row["source_slot"] == source["source_slot"]
            and row["component_index"] == source["component_index"]
            for row in [*rows, *pairs]
        ),
        "All private case and pair rows must remain on their exact component",
    )
    for row, case in zip(rows, cases, strict=True):
        require(
            row["case_number"] == case["case_number"]
            and row["construction_status"] == case["status"]
            and row["construction_reason"] == case["reason"]
            and row["neural_provenance_verified"] is case["neural_provenance_verified"]
            and row["observed_neural_pixel_change"] is case.get("observed_neural_pixel_change", False)
            and row["status"] in MOTION_STATES
            and row["curve"]["measured"] is (row["status"] != "unavailable")
            and (case["status"] == "ready" or row["status"] == "unavailable")
            and row["curve"]["supported_positions"] == len(row["full_support_times"])
            and row["target_only_contrast"]["supported_positions"] == len(row["target_support_times"]),
            "Motion outcome or original case identity changed",
        )
        if row["curve"]["measured"]:
            require(
                set(row["curve"]["lags"]) == {str(lag) for lag in arithmetic.LAGS}
                and set(row["curve"]["target_movement_rms"]) == {str(lag) for lag in arithmetic.TARGETS},
                "Full curve must retain all exact signed lag measurements",
            )
        if row["target_only_contrast"]["measured"]:
            require(
                set(row["target_only_contrast"]["lags"]) == {str(lag) for lag in arithmetic.TARGETS},
                "Target-only contrast must keep its own exact support",
            )
    for pair in pairs:
        require(
            pair["status"] in PAIR_STATES
            and (
                pair["status"] == "unavailable"
                or (
                    pair["supported_positions"] == len(pair["shared_support_times"])
                    and pair["supported_positions"] >= arithmetic.MIN_FRAMES
                    and pair["adjacent_pairs"] >= arithmetic.MIN_PAIRS
                )
            ),
            "Matched preference must retain its common support",
        )


def load_completed(ordinal):
    """Read a private completed motion packet and bind all upstream saved observations."""
    verify()
    source = source_for(ordinal)
    cases, _, controls, bindings = observation_metadata(source)
    stage = stage_folder(ordinal)
    result = planning.reviewed.stage(stage, sha(PUBLIC), "grid_mouth_motion_readout_v158.py", 0, {})
    file = private_folder(ordinal) / "motion.json"
    payload = read(file)
    require(
        result["schema"] == "chipmunk.grid-mouth-motion-clip-result/158"
        and result["clip_ordinal"] == ordinal
        and result["private_motion_sha256"] == sha(file)
        and result["new_media_reads"] == 0
        and all(result[k] == v for k, v in counts(payload).items()),
        "Actual completed bounded private readout required",
    )
    validate_payload(source, cases, controls, bindings, payload)
    return payload


def exact_controls(payloads):
    return {
        name: {
            key: sum(payload["observer_exact_control_checks"][name][key] for payload in payloads)
            for key in CONTROL_COUNTS
        }
        for name in CONTROLS
    }


def summarize(output):
    frozen = verify()
    ordinals = pilot_ordinals()
    payloads, manifest = [], []
    for ordinal in ordinals:
        payloads.append(load_completed(ordinal))
        stage = stage_folder(ordinal)
        manifest.append(
            {
                "clip_ordinal": ordinal,
                "motion_result_sha256": sha(stage / "output/result.json"),
                "motion_profile_sha256": sha(stage / "profile.json"),
                "motion_resource_journal_sha256": sha(stage / "resources.jsonl"),
                "private_motion_sha256": sha(private_folder(ordinal) / "motion.json"),
            }
        )
    rows = [row for payload in payloads for row in payload["cases"]]
    pairs = [row for payload in payloads for row in payload["pairs"]]
    require(
        len(payloads) == 16
        and len(rows) == 176
        and len(pairs) == 96
        and len({row["component_index"] for row in rows}) == frozen["pilot_components"],
        "Complete pilot roster and paired denominator required",
    )
    folder = private_folder("summary")
    folder.mkdir(parents=True, exist_ok=False)
    manifest_file = folder / "clip-artifact-manifest.json"
    write(manifest_file, manifest)
    result = {
        "schema": "chipmunk.grid-mouth-motion-readout-result/158",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "complete_guarded_clip_stages": len(payloads),
        "private_clip_artifact_manifest_sha256": sha(manifest_file),
        "exact_observer_control_checks": exact_controls(payloads),
        **aggregate_rows(rows, pairs),
        "new_neural_calls": 0,
        "new_media_reads": 0,
        "defaults_changed": False,
        "scope": "Pilot visible trajectory compatibility and control measurements under one observer. Phonetic alignment and detector performance remain unassessed.",
    }
    write(output / "result.json", result)
    print(
        json.dumps({"status": "complete", "planned_cases": len(rows), "planned_pairs": len(pairs)}),
        flush=True,
    )


def prerequisites(ordinal, ordinals):
    if ordinal == "summary":
        return all(
            (stage_folder(clip) / name).is_file()
            for clip in ordinals
            for name in ("profile.json", "output/result.json")
        )
    observer = observer_module()
    observed = observer.TRIAL / stage_name(ordinal) / "authored"
    index = ordinals.index(ordinal)
    stages = [observed]
    if index:
        stages.append(stage_folder(ordinals[index - 1]))
    return all(
        (stage / name).is_file() for stage in stages for name in ("profile.json", "output/result.json")
    )


def routed_popen(ordinal, argv, **kwargs):
    output = stage_folder(ordinal) / "output"
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"]
        and argv[5:] == [str(output), "0"],
        "Only the exclusive data-only motion readout is permitted",
    )
    return subprocess.Popen(
        [*argv[:3], "scripts.grid_mouth_motion_readout_v158", "_worker", str(ordinal), str(output)], **kwargs
    )


def run(ordinal):
    from scripts import video_appearance_runtime_v108 as runtime
    from scripts.video_appearance_corrected_v112 import scoped

    verify()
    ordinals = pilot_ordinals()
    require(
        ordinal == "summary" or (type(ordinal) is int and ordinal in ordinals),
        "Fixed pilot or summary required",
    )
    stage = stage_folder(ordinal)
    stage.parent.mkdir(exist_ok=True)
    scoped(
        runtime.run,
        {
            "TRIAL": stage.parent,
            "verify": verify,
            "prerequisites": lambda _command, _batch: prerequisites(ordinal, ordinals),
            "subprocess": types.SimpleNamespace(
                Popen=lambda argv, **kw: routed_popen(ordinal, argv, **kw),
                STDOUT=subprocess.STDOUT,
                TimeoutExpired=subprocess.TimeoutExpired,
            ),
            "__file__": str(Path(__file__)),
        },
    )("bind", 0)


def parse_target(value):
    require(
        value == "summary" or value in {str(i) for i in range(32)},
        "Exact original ordinal or summary required",
    )
    return value if value == "summary" else int(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    commands.add_parser("run").add_argument("target", type=parse_target)
    worker = commands.add_parser("_worker")
    worker.add_argument("target", type=parse_target)
    worker.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "run":
        run(args.target)
    else:
        ordinals = pilot_ordinals()
        stage = stage_folder(args.target)
        resources, log = stage / "resources.jsonl", stage / "worker.log"
        require(
            (args.target == "summary" or args.target in ordinals)
            and args.output == stage / "output"
            and resources.is_file()
            and resources.stat().st_size > 0
            and log.is_file()
            and not (stage / "profile.json").exists()
            and all(os.path.samestat(os.fstat(fd), log.stat()) for fd in (1, 2)),
            "Exact exclusive guarded data-only worker output required",
        )
        verify()
        args.output.mkdir(exist_ok=False)
        if args.target == "summary":
            summarize(args.output)
        else:
            index = ordinals.index(args.target)
            if index:
                load_completed(ordinals[index - 1])
            measure_clip(args.target, args.output)


if __name__ == "__main__":
    main()
