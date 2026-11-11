"""Independent native picture observations for the fixed GRID pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import types
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import grid_native_clips_v149 as construction
from scripts import grid_portrait_geometry_v144 as geometry
from scripts import grid_transfer_plan_v143 as planning

ROOT, BASE, PANEL, ARCHIVE = planning.ROOT, planning.BASE, planning.PANEL, planning.ARCHIVE
TRIAL = BASE / "grid-mouth-observer-v157"
CACHE = ARCHIVE / "grid-mouth-observer-v157"
PUBLIC = PANEL / "grid-mouth-observer-freeze-v157.json"
CONSTRUCTION_COMPLETE = PANEL / "grid-native-clips-complete-v149.json"
REFERENCE = tuple(range(15, 59))
VARIANT = tuple(range(25, 49))
REPEAT = 27
POINTS = (1, 13, 14, 33, 78, 82, 87, 133, 263, 308, 312, 317, 362)
EYES = (33, 133, 362, 263)
GAPS = ((82, 87), (13, 14), (312, 317))
AVAILABLE = "geometry_available"
PLANNED = len(REFERENCE) + 10 * len(VARIANT) + 1
GEOMETRY_FIELDS = (
    "landmarks_xy_hex",
    "descriptor_hex",
    "eye_distance_px_hex",
    "eye_roll_radians_hex",
    "nose_offset_hex",
)
CODE = (
    "backend/scripts/grid_mouth_observer_v157.py",
    "backend/tests/test_grid_mouth_observer_v157.py",
    "docs/grid-mouth-observer-v157-method.md",
    "docs/grid-achieved-motion-validation-design.md",
    "backend/scripts/grid_native_clips_v149.py",
    "backend/scripts/grid_portrait_geometry_v144.py",
    "backend/scripts/grid_transfer_plan_v143.py",
    "backend/app/detectors/video/decode.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/scripts/local_probe_limits_v24.py",
)
read, write, sha, require = planning.read, planning.write, planning.sha, planning.require


def pilots(plan):
    sources = [source for source in plan["sources"] if source["role"] == "pilot"]
    require(
        len(sources) == 16 and [s["ordinal"] for s in sources] == sorted({s["ordinal"] for s in sources}),
        "Sixteen unique ordered original pilot sources required",
    )
    return sources


def pilot(ordinal):
    require(type(ordinal) is int and 0 <= ordinal < 32, "Original integer pilot ordinal required")
    source = next((row for row in pilots(planning.load_completed()) if row["ordinal"] == ordinal), None)
    require(source is not None, "Reserve or unknown clip cannot be observed")
    return source


def identity():
    plan = planning.load_completed()
    frozen = planning.frozen_context(construction.TRIAL, construction.PUBLIC, {})
    original_runtime = planning.frozen_context(geometry.TRIAL, geometry.PUBLIC, {})
    complete = read(CONSTRUCTION_COMPLETE)
    require(
        complete["schema"] == "chipmunk.grid-native-clips-complete/149"
        and complete["status"] == "complete"
        and complete["complete_guarded_stages"] == 16
        and complete["planned_cases"] == 176
        and complete["guard_violations"] == complete["monitor_sample_errors"] == 0
        and complete["freeze_sha256"] == sha(construction.PUBLIC),
        "Complete published v149 bank required",
    )
    runtime = geometry.runtime_verified()
    require(
        original_runtime["geometry"] == geometry.GEOMETRY
        and all(original_runtime[key] == value for key, value in runtime.items()),
        "Unchanged v144 qualified isolated observer settings and assets required",
    )
    return {
        "schema": "chipmunk.grid-mouth-observer-freeze/157",
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "transfer_plan_freeze_sha256": sha(planning.PUBLIC),
        "private_transfer_plan_sha256": sha(planning.TRIAL / "bind-00/output/private-plan.json"),
        "construction_freeze_sha256": sha(construction.PUBLIC),
        "construction_complete_sha256": sha(CONSTRUCTION_COMPLETE),
        "original_observer_contract_sha256": sha(geometry.PUBLIC),
        "binaries_sha256": frozen["binaries_sha256"],
        "observer_runtime": runtime,
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "opencv-python-headless", "psutil")
        },
        "observer_settings": geometry.GEOMETRY,
        "pilot_clips": len(pilots(plan)),
        "cases_per_pilot": 11,
        "reference_frames_half_open": [REFERENCE[0], REFERENCE[-1] + 1],
        "variant_frames_half_open": [VARIANT[0], VARIANT[-1] + 1],
        "fixed_original_repeat_frame": REPEAT,
        "planned_observations_per_pilot": PLANNED,
        "landmark_indices": list(POINTS),
        "eye_patch_side_px": 7,
        "scope": "Fresh finished native picture geometry under the qualified observer. Independent trajectory interpretation follows in a separate context.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(
        json.dumps(
            {
                "status": "frozen",
                "pilot_clips": body["pilot_clips"],
                "planned_observations_per_pilot": PLANNED,
            }
        ),
        flush=True,
    )


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged published observer contract required",
    )


def source_clock(source):
    table = source["input"]["native_video_table"]
    pts, time_base = table["pts"], table["time_base"]
    require(
        type(time_base) is str
        and Fraction(time_base) > 0
        and len(pts) == 75
        and all(type(t) is int for t in pts)
        and all(a < b for a, b in zip(pts, pts[1:])),
        "Complete increasing rational original native picture clock required",
    )
    require(
        all(type(table[k]) is int and 0 < table[k] <= 640 for k in ("width", "height")),
        "Bounded exact native picture shape required",
    )
    return {"time_base": time_base, "pts": pts}, [table["height"], table["width"], 3]


def control_indices(condition, index):
    return index - 8 if condition == "genuine_frame_retime" and index in range(*planning.EDIT) else index


def schedule(source, cases):
    """Build complete fixed observation requests using metadata only."""
    clock, shape = source_clock(source)
    times = [Fraction(t) * Fraction(clock["time_base"]) for t in clock["pts"]]
    require(
        source["role"] == "pilot"
        and [case["condition"] for case in cases] == list(planning.CASES)
        and {c["case_number"] for c in cases} == set(range(11)),
        "All eleven original numbered pilot cases required",
    )
    requests, summaries = [], []
    for case in cases:
        ready = case["status"] == "ready"
        require(
            case["asset_index"] == source["ordinal"]
            and case["source_slot"] == source["source_slot"]
            and case["component_index"] == source["component_index"]
            and case["study_role"] == "pilot"
            and type(case["case_number"]) is int
            and case["status"] in {"ready", "unavailable"},
            "Exact original pilot/source/component case required",
        )
        if ready:
            proof = case["expected"]
            require(
                case["source_video_pts"] == clock["pts"]
                and case["source_video_time_base"] == clock["time_base"]
                and len(proof["picture_pts_ms"]) == len(proof["rgb_sha256"]) == 75
                and all(type(t) is int for t in proof["picture_pts_ms"])
                and [Fraction(t, 1000) for t in proof["picture_pts_ms"]] == times
                and all(type(h) is str and re.fullmatch(r"[0-9a-f]{64}", h) for h in proof["rgb_sha256"]),
                "Exact constructed picture clocks and hashes required",
            )
        summaries.append({k: case[k] for k in ("case_number", "condition", "status", "reason")})
        indices = REFERENCE if case["condition"] == "original" else VARIANT
        for index in indices:
            requests.append(
                {
                    "request_ordinal": len(requests),
                    "case_number": case["case_number"],
                    "frame_index": index,
                    "repeat": False,
                    "case_status": case["status"],
                    "case_reason": case["reason"],
                    "source_pts": clock["pts"][index],
                    "source_time_base": clock["time_base"],
                    "source_time_seconds": str(times[index]),
                    "pts": case["expected"]["picture_pts_ms"][index] if ready else None,
                    "time_base": "1/1000" if ready else None,
                    "rgb_sha256": case["expected"]["rgb_sha256"][index] if ready else None,
                }
            )
    original = cases[0]
    fixed = requests[REFERENCE.index(REPEAT)]
    requests.append({**fixed, "request_ordinal": len(requests), "repeat": True})
    require(
        len(requests) == PLANNED and original["condition"] == "original",
        "Fixed reference and repeat schedule required",
    )
    if original["status"] == "ready":
        for control in (
            c
            for c in cases
            if c["condition"] in {"audio_delay_320ms", "genuine_frame_retime"} and c["status"] == "ready"
        ):
            for index in VARIANT:
                reference_index = control_indices(control["condition"], index)
                require(
                    times[index] - times[reference_index]
                    == (Fraction(8, 25) if reference_index != index else 0)
                    and control["expected"]["rgb_sha256"][index]
                    == original["expected"]["rgb_sha256"][reference_index],
                    "Exact original or earlier-copy control picture binding required",
                )
    return {
        "clip_ordinal": source["ordinal"],
        "source_slot": source["source_slot"],
        "component_index": source["component_index"],
        "picture_shape": shape,
        "source_clock": clock,
        "cases": summaries,
        "requests": requests,
    }


def input_binding(ordinal):
    source = pilot(ordinal)
    cases = construction.load_completed(ordinal)
    plan = schedule(source, cases)
    clip = f"clip-{ordinal:02d}"
    stage = construction.TRIAL / clip / "bind-00"
    plan["bindings"] = {
        "construction_case_journal_sha256": sha(construction.CACHE / clip / "cases.jsonl"),
        "construction_result_sha256": sha(stage / "output/result.json"),
        "construction_profile_sha256": sha(stage / "profile.json"),
        "construction_resource_journal_sha256": sha(stage / "resources.jsonl"),
    }
    return plan, cases


def describe(points, shape):
    """Use only the selected native 2D points in the frozen descriptor."""
    values = np.asarray(points, dtype=np.float64)
    height, width, channels = shape
    require(
        channels == 3
        and values.shape == (len(POINTS), 2)
        and np.isfinite(values).all()
        and (values >= 0).all()
        and (values < [width, height]).all(),
        "Finite in-picture native landmarks required",
    )
    p = dict(zip(POINTS, values, strict=True))
    left, right = (p[33] + p[133]) / 2, (p[362] + p[263]) / 2
    eye = right - left
    scale = float(np.linalg.norm(eye))
    require(scale >= 12, "Insufficient native eye separation")
    ex = eye / scale
    ey = np.array([-ex[1], ex[0]])
    nose = p[1] - (left + right) / 2
    require(float(nose @ ey) > 0, "Nose must lie below the oriented eye axis")
    gaps = [float((p[lower] - p[upper]) @ ey) for upper, lower in GAPS]
    mouth_width = abs(float((p[308] - p[78]) @ ex))
    require(mouth_width >= 8 and min(gaps) >= -1, "Insufficient lip width or crossing inner lip points")
    descriptor = np.asarray([*(max(0.0, gap) for gap in gaps), mouth_width]) / scale
    return {
        "landmarks_xy_hex": geometry.canonical_hex(values, (len(POINTS), 2)),
        "descriptor_hex": geometry.canonical_hex(descriptor, (4,)),
        "eye_distance_px_hex": scale.hex(),
        "eye_roll_radians_hex": math.atan2(float(ex[1]), float(ex[0])).hex(),
        "nose_offset_hex": geometry.canonical_hex([float(nose @ ex) / scale, float(nose @ ey) / scale], (2,)),
    }


def rgb_commitment(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def task_event(kind, request, observed_faces=None):
    row = {
        "event": kind,
        "request_ordinal": request["request_ordinal"],
        "case_number": request["case_number"],
        "frame_index": request["frame_index"],
        "repeat": request["repeat"],
    }
    if observed_faces is not None:
        row["observed_faces"] = observed_faces
    return row


def unavailable(request):
    require(
        request["case_status"] == "unavailable" and request["case_reason"],
        "Explicit planned case absence required",
    )
    return {**request, "status": "unavailable_case", "reason": request["case_reason"], "observed_faces": None}


def observe_picture(request, frame, shape, detect, event):
    """The observer callback receives only verified RGB."""
    rgb = frame.rgb
    require(
        request["case_status"] == "ready"
        and frame.index == request["frame_index"]
        and frame.pts == request["pts"]
        and tuple(rgb.shape) == tuple(shape)
        and rgb.dtype == np.uint8
        and rgb_commitment(rgb) == request["rgb_sha256"],
        "Exact selected native RGB/index/PTS required before face task",
    )
    event(task_event("face_task_begin", request))
    faces = detect(np.ascontiguousarray(rgb).copy()).face_landmarks
    count = len(faces)
    event(task_event("face_task_end", request, count))
    base = {**request, "observed_faces": count}
    if count != 1:
        status = "unavailable_no_face" if count == 0 else "unavailable_multiple_faces"
        return {**base, "status": status, "reason": "One unambiguous observed face required"}
    try:
        mesh = np.asarray([(p.x, p.y, p.z) for p in faces[0]], dtype=np.float64)
        require(mesh.shape == (478, 3) and np.isfinite(mesh).all(), "Finite complete face mesh required")
    except (ValueError, TypeError, AttributeError):
        return {**base, "status": "unavailable_mesh", "reason": "Finite complete face mesh unavailable"}
    points = mesh[list(POINTS), :2] * [shape[1], shape[0]]
    try:
        descriptor = describe(points, shape)
    except ValueError as exc:
        return {**base, "status": "unavailable_geometry", "reason": str(exc)}
    return {**base, "status": AVAILABLE, "reason": None, **descriptor}


def eye_points(row):
    points = geometry.decode_hex(row["landmarks_xy_hex"], (len(POINTS), 2))
    return points[[POINTS.index(i) for i in EYES]]


def patch_bounds(points, shape):
    result = []
    for x, y in points:
        x, y = round(float(x)), round(float(y))
        if x < 3 or y < 3 or x + 4 > shape[1] or y + 4 > shape[0]:
            return None
        result.append((slice(y - 3, y + 4), slice(x - 3, x + 4)))
    return result


def eye_rms(original, observed):
    return float(np.sqrt(np.square(eye_points(original) - eye_points(observed)).sum(axis=1).mean())).hex()


def eye_drift(original, observed, original_rgb, rgb, shape):
    if original["status"] != AVAILABLE or observed["status"] != AVAILABLE:
        return {"status": "unavailable_geometry", "rms_px_hex": None}
    patches = patch_bounds(eye_points(original), shape)
    if patches is None:
        return {"status": "patch_outside_picture", "rms_px_hex": None}
    if any(not np.array_equal(original_rgb[patch], rgb[patch]) for patch in patches):
        return {"status": "different_eye_pixels", "rms_px_hex": None}
    return {"status": "identical_eye_pixels", "rms_px_hex": eye_rms(original, observed)}


def same_measurement(first, second):
    return all(
        first.get(k) == second.get(k) for k in ("status", "observed_faces", "reason", *GEOMETRY_FIELDS)
    )


def control_checks(plan, rows):
    cases = {case["condition"]: case["case_number"] for case in plan["cases"]}
    by_key = {(row["case_number"], row["frame_index"]): row for row in rows if not row["repeat"]}
    repeated = [row for row in rows if row["repeat"]]
    require(len(repeated) == 1, "Exact one original observer repeat required")
    original = cases["original"]

    def compare(pairs):
        counts = Counter({"planned_pairs": len(pairs)})
        for reference, actual in pairs:
            if reference["case_status"] != "ready" or actual["case_status"] != "ready":
                counts["unavailable_pairs"] += 1
            elif reference["rgb_sha256"] != actual["rgb_sha256"] or not same_measurement(reference, actual):
                counts["mismatched_pairs"] += 1
            elif reference["status"] == AVAILABLE:
                counts["matching_geometry_pairs"] += 1
            else:
                counts["matching_unavailable_geometry_pairs"] += 1
        return {
            k: counts[k]
            for k in (
                "planned_pairs",
                "unavailable_pairs",
                "mismatched_pairs",
                "matching_geometry_pairs",
                "matching_unavailable_geometry_pairs",
            )
        }

    controls = {"fixed_repeat": compare([(by_key[original, REPEAT], repeated[0])])}
    for condition in ("audio_delay_320ms", "genuine_frame_retime"):
        controls[condition] = compare(
            [(by_key[original, control_indices(condition, i)], by_key[cases[condition], i]) for i in VARIANT]
        )
    return controls


def validate_observations(plan, rows, events):
    require(len(rows) == len(plan["requests"]) == PLANNED, "Every fixed observation must remain recorded")
    original = plan["cases"][0]["case_number"]
    originals, expected = {}, []
    for request, row in zip(plan["requests"], rows, strict=True):
        require(
            all(k in row and type(row[k]) is type(v) and row[k] == v for k, v in request.items()),
            "Observation request/clock/hash correspondence differs",
        )
        status, count = row["status"], row["observed_faces"]
        if request["case_status"] == "unavailable":
            require(
                status == "unavailable_case" and count is None and row["reason"] == request["case_reason"],
                "Unavailable construction cannot supply a face observation",
            )
        else:
            require(type(count) is int and count >= 0, "Observed face count required")
            allowed = (
                {"unavailable_no_face"}
                if count == 0
                else (
                    {"unavailable_multiple_faces"}
                    if count > 1
                    else {AVAILABLE, "unavailable_mesh", "unavailable_geometry"}
                )
            )
            require(status in allowed, "Face observation status and observed count differ")
            expected.extend(
                (task_event("face_task_begin", request), task_event("face_task_end", request, count))
            )
        if status == AVAILABLE:
            points = geometry.decode_hex(row["landmarks_xy_hex"], (len(POINTS), 2))
            require(
                row["reason"] is None
                and all(row[k] == v for k, v in describe(points, plan["picture_shape"]).items()),
                "Native descriptor/pose values differ from observed coordinates",
            )
        else:
            require(
                row["reason"] and not set(GEOMETRY_FIELDS).intersection(row),
                "Unavailable geometry cannot carry a descriptor",
            )
        if row["case_number"] == original:
            require("eye_patch_drift" not in row, "Original and fixed repeat carry no variant eye drift")
            if not row["repeat"]:
                originals[row["frame_index"]] = row
        else:
            validate_drift(originals[row["frame_index"]], row, plan["picture_shape"])
    require(
        json.dumps(events, sort_keys=True) == json.dumps(expected, sort_keys=True),
        "Exact ordered pre/post face-task observations required",
    )
    return len(expected) // 2


def validate_drift(original, row, shape):
    drift = row["eye_patch_drift"]
    if original["status"] != AVAILABLE or row["status"] != AVAILABLE:
        expected = {"status": "unavailable_geometry", "rms_px_hex": None}
    elif patch_bounds(eye_points(original), shape) is None:
        expected = {"status": "patch_outside_picture", "rms_px_hex": None}
    elif drift["status"] == "identical_eye_pixels" or original["rgb_sha256"] == row["rgb_sha256"]:
        expected = {"status": "identical_eye_pixels", "rms_px_hex": eye_rms(original, row)}
    else:
        expected = {"status": "different_eye_pixels", "rms_px_hex": None}
    require(drift == expected, "Saved eye patch observation or geometry arithmetic differs")


def decoded_pictures(case, requests, shape):
    from app.detectors.video.decode import decode_frames, probe_frames

    path = Path(case["file"])
    expected = construction.CACHE / f"clip-{case['asset_index']:02d}" / f"{case['case_number']:02d}.mkv"
    construction.owned_file(path, construction.CACHE, expected)
    require(
        path.resolve().parent == construction.CACHE.resolve() / expected.parent.name,
        "Constructed native file escaped its owned clip folder",
    )
    require(sha(path) == case["file_sha256"], "Actual constructed native file changed")
    table = probe_frames(path, cap=75, timeout_s=60)
    require(
        table.time_base == "1/1000"
        and list(table.pts) == case["expected"]["picture_pts_ms"]
        and len(table) == 75
        and [table.height, table.width, 3] == shape
        and not table.truncated
        and not table.interpolated,
        "Actual complete decoded native clock and dimensions required",
    )
    return decode_frames(path, table, [row["frame_index"] for row in requests], timeout_s=120)


def observer_options(mp):
    settings = dict(geometry.GEOMETRY)
    settings.pop("delegate")
    settings["running_mode"] = mp.tasks.vision.RunningMode.IMAGE
    return mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(geometry.ASSETS / "permissions/face_landmarker.task"),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        **settings,
    )


def observe(ordinal, output):
    import cv2

    from scripts import portrait_geometry_runtime_v132 as runtime

    plan, cases = input_binding(ordinal)
    frozen = read(PUBLIC)
    require(
        geometry.runtime_verified() == frozen["observer_runtime"],
        "Qualified isolated observer runtime changed",
    )
    for name, digest in frozen["binaries_sha256"].items():
        path = shutil.which(name)
        require(path is not None and sha(path) == digest, "Qualified native decoder binary changed")
    require("mediapipe" not in sys.modules, "Fresh isolated observer import required")
    sys.path.insert(0, str(runtime.SITE))
    import mediapipe as mp

    require(
        Path(mp.__file__).resolve().is_relative_to(runtime.SITE.resolve())
        and importlib.metadata.version("mediapipe") == runtime.PINS["mediapipe"][0],
        "Qualified isolated face task required",
    )
    cv2.setNumThreads(1)
    folder = CACHE / f"clip-{ordinal:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "input-binding.json", plan)
    rows, events, original_frames, original_rows = [], [], {}, {}
    shape = plan["picture_shape"]
    original_number = cases[0]["case_number"]
    with (
        (output / "face-progress.jsonl").open("x") as progress,
        (folder / "observations.jsonl").open("x") as journal,
    ):

        def event(value):
            progress.write(json.dumps(value, allow_nan=False) + "\n")
            progress.flush()
            events.append(value)

        def record(row):
            rows.append(row)
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()

        with mp.tasks.vision.FaceLandmarker.create_from_options(observer_options(mp)) as task:

            def detect(rgb):
                return task.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            for case in cases:
                number = case["case_number"]
                requests = [r for r in plan["requests"] if r["case_number"] == number and not r["repeat"]]
                frames = (
                    decoded_pictures(case, requests, shape)
                    if case["status"] == "ready"
                    else [None] * len(requests)
                )
                for request, frame in zip(requests, frames, strict=True):
                    row = (
                        unavailable(request)
                        if frame is None
                        else observe_picture(request, frame, shape, detect, event)
                    )
                    index = request["frame_index"]
                    if number == original_number:
                        original_rows[index] = row
                        if frame is not None:
                            original_frames[index] = frame
                    else:
                        original_frame = original_frames.get(index)
                        row["eye_patch_drift"] = eye_drift(
                            original_rows[index],
                            row,
                            None if original_frame is None else original_frame.rgb,
                            None if frame is None else frame.rgb,
                            shape,
                        )
                    record(row)
            fixed = plan["requests"][-1]
            require(
                fixed["repeat"] is True and fixed["frame_index"] == REPEAT,
                "Fixed original repeat request required",
            )
            record(
                observe_picture(fixed, original_frames[REPEAT], shape, detect, event)
                if fixed["case_status"] == "ready"
                else unavailable(fixed)
            )
    calls = validate_observations(plan, rows, events)
    controls = control_checks(plan, rows)
    write(folder / "control-checks.json", controls)
    require(
        all(c["mismatched_pairs"] == 0 for c in controls.values()),
        "Fresh repeat or exact-RGB control measurement differs",
    )
    result = {
        "schema": "chipmunk.grid-mouth-observer-result/157",
        "status": "complete",
        "clip_ordinal": ordinal,
        **observed_counts(rows, calls, controls),
        "private_input_binding_sha256": sha(folder / "input-binding.json"),
        "private_observation_journal_sha256": sha(folder / "observations.jsonl"),
        "private_control_checks_sha256": sha(folder / "control-checks.json"),
        "private_face_progress_sha256": sha(output / "face-progress.jsonl"),
        "freeze_sha256": sha(PUBLIC),
        "defaults_changed": False,
        "scope": "Fresh full-native CPU IMAGE observer measurements and exact control checks. A separate context evaluates visible trajectories.",
    }
    write(output / "result.json", result)
    print(
        json.dumps(
            {"status": "complete", "planned_observations": PLANNED, "observed_face_task_inputs": calls}
        ),
        flush=True,
    )


def observed_counts(rows, calls, controls):
    return {
        "planned_observations": len(rows),
        "observed_face_task_inputs_started": calls,
        "observed_face_task_inputs_completed": calls,
        "new_neural_calls": calls,
        "availability": dict(Counter(row["status"] for row in rows)),
        "eye_patch_drift_availability": dict(
            Counter(row["eye_patch_drift"]["status"] for row in rows if "eye_patch_drift" in row)
        ),
        "controls": controls,
    }


def load_completed(ordinal):
    """Bind actual completed observation metadata for a separate arithmetic worker."""
    verify()
    plan, _ = input_binding(ordinal)
    clip = f"clip-{ordinal:02d}"
    stage, folder = TRIAL / clip / "authored", CACHE / clip
    out = stage / "output"
    result, profile = read(out / "result.json"), read(stage / "profile.json")
    require(
        result["schema"] == "chipmunk.grid-mouth-observer-result/157"
        and result["status"] == "complete"
        and result["clip_ordinal"] == ordinal
        and result["defaults_changed"] is False
        and result["freeze_sha256"] == sha(PUBLIC)
        and profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "authored"
        and profile["batch"] is None
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["error_types"] == []
        and profile["resource_limit_observations"] == []
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["worker_receipt_sha256"] == sha(out / "result.json")
        and profile["new_neural_calls"] == result["new_neural_calls"],
        "Actual complete bounded observer receipt required",
    )
    geometry.resource_verified(stage, profile)
    for key, path in {
        "private_input_binding_sha256": folder / "input-binding.json",
        "private_observation_journal_sha256": folder / "observations.jsonl",
        "private_control_checks_sha256": folder / "control-checks.json",
        "private_face_progress_sha256": out / "face-progress.jsonl",
    }.items():
        require(result[key] == sha(path), "Actual observer artifact changed")
    require(read(folder / "input-binding.json") == plan, "Actual complete observer input binding required")
    rows = planning.reviewed.lines(folder / "observations.jsonl")
    calls = validate_observations(plan, rows, planning.reviewed.lines(out / "face-progress.jsonl"))
    controls = control_checks(plan, rows)
    require(
        read(folder / "control-checks.json") == controls
        and all(c["mismatched_pairs"] == 0 for c in controls.values())
        and all(result[k] == v for k, v in observed_counts(rows, calls, controls).items()),
        "Actual observer outcomes or counters differ",
    )
    return {**{k: v for k, v in plan.items() if k != "requests"}, "observations": rows, "controls": controls}


def routed_popen(ordinal, argv, **kwargs):
    expected = TRIAL / f"clip-{ordinal:02d}/authored/output"
    require(
        type(ordinal) is int
        and 0 <= ordinal < 32
        and argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"]
        and argv[5:] == [str(expected)],
        "Only exact exclusive per-pilot neural observer permitted",
    )
    return subprocess.Popen(
        [*argv[:3], "scripts.grid_mouth_observer_v157", "observe", str(expected), str(ordinal)], **kwargs
    )


def run(ordinal):
    from scripts import video_appearance_runtime_v108 as runtime
    from scripts.video_appearance_corrected_v112 import scoped

    input_binding(ordinal)
    folder = TRIAL / f"clip-{ordinal:02d}"
    folder.mkdir(exist_ok=True)
    proxy = types.SimpleNamespace(
        Popen=lambda argv, **kw: routed_popen(ordinal, argv, **kw),
        STDOUT=subprocess.STDOUT,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    scoped(
        runtime.run,
        {
            "TRIAL": folder,
            "verify": verify,
            "prerequisites": lambda _c, _b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("authored")


def worker(ordinal, output):
    require(type(ordinal) is int and 0 <= ordinal < 32, "Original integer pilot ordinal required")
    stage = TRIAL / f"clip-{ordinal:02d}/authored"
    require(output == stage / "output", "Exact exclusive observer output required")
    resources, log = stage / "resources.jsonl", stage / "worker.log"
    require(
        resources.is_file()
        and resources.stat().st_size > 0
        and log.is_file()
        and not (stage / "profile.json").exists()
        and all(os.path.samestat(os.fstat(fd), log.stat()) for fd in (1, 2)),
        "Active outer monitor files and attached unfinished worker log required",
    )
    verify()
    output.mkdir(exist_ok=False)
    observe(ordinal, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "observe"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "No freeze arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One explicit pilot ordinal required")
        run(int(args.arguments[0]))
    else:
        require(len(args.arguments) == 2, "Exclusive output and pilot ordinal required")
        worker(int(args.arguments[1]), Path(args.arguments[0]))
