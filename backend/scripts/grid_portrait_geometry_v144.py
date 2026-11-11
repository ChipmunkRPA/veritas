"""Bound pilot-only isolated face geometry; no generator or detector execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
PANEL = ROOT / "docs/evaluation-panels"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
TRIAL = BASE / "grid-portrait-geometry-v144"
CACHE = ARCHIVE / "grid-portrait-geometry-v144"
PUBLIC = PANEL / "grid-portrait-geometry-freeze-v144.json"
ASSETS = BASE / "lipsync-v4"
FRAME_INDICES = tuple(range(19, 47))
REPEAT_FRAME = 27
TASK_INPUTS = 29
AVAILABLE = "geometry_available"
UNAVAILABLE = {
    "unavailable_no_face",
    "unavailable_multiple_faces",
    "unavailable_mesh",
    "unavailable_geometry",
    "unavailable_picture_decode",
}
GEOMETRY = {
    "running_mode": "IMAGE",
    "delegate": "CPU",
    "num_faces": 2,
    "min_face_detection_confidence": 0.5,
    "min_face_presence_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "output_face_blendshapes": False,
    "output_facial_transformation_matrixes": False,
}
CODE = (
    "backend/scripts/grid_portrait_geometry_v144.py",
    "backend/tests/test_grid_portrait_geometry_v144.py",
    "docs/grid-portrait-geometry-v144-method.md",
    "backend/scripts/portrait_geometry_runtime_v132.py",
    "backend/scripts/fetch_mouth_benchmark_assets.py",
    "backend/scripts/liveportrait_mouth_benchmark.py",
    "backend/scripts/prepare_mouth_edit_v4.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/video_appearance_corrected_v112.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def planner():
    from scripts import grid_transfer_plan_v143

    return grid_transfer_plan_v143


def plan():
    body = planner().load_completed()
    rows = body["sources"]
    require(
        len(rows) == 32 and [r["ordinal"] for r in rows] == list(range(32)),
        "Actual complete fixed v143 cohort required",
    )
    pilots = [r for r in rows if r["role"] == "pilot"]
    require(pilots and all(r["role"] in {"pilot", "reserve"} for r in rows), "Frozen pilot roles required")
    for row in pilots:
        requests(row)
    return {"schema": "chipmunk.grid-portrait-geometry-plan/144", "pilots": pilots}


def requests(planned):
    """Exact original requests; no content-dependent choice or source replacement."""
    require(
        type(planned["ordinal"]) is int and 0 <= planned["ordinal"] < 32 and planned["role"] == "pilot",
        "One original pilot ordinal required",
    )
    geometry = planned["geometry"]
    if geometry["status"] != "ready":
        require(geometry["status"] == "unavailable", "Explicit geometry availability required")
        return []
    source = planned["input"]
    pictures = geometry["pictures"]
    by_index = {p["frame_index"]: p for p in source["pictures"]}
    require(
        geometry["frame_indices"] == list(FRAME_INDICES)
        and geometry["repeat_frame_index"] == REPEAT_FRAME
        and [p["frame_index"] for p in pictures] == list(FRAME_INDICES)
        and all(p == by_index.get(p["frame_index"]) for p in pictures)
        and all(p["pts"] == source["native_video_table"]["pts"][p["frame_index"]] for p in pictures),
        "All original frames19..46 and repeat27 must match frozen native bank",
    )
    return [*pictures, by_index[REPEAT_FRAME]]


def runtime_verified():
    """Recheck saved isolated runtime and commercial asset records before inference."""
    from scripts import fetch_mouth_benchmark_assets as assets
    from scripts import portrait_geometry_runtime_v132 as runtime

    runtime.verify()
    folder = runtime.TRIAL / "bind-00"
    result, profile = read(folder / "output/result.json"), read(folder / "profile.json")
    manifest = folder / "output/installed-files.json"
    require(
        result["schema"] == "chipmunk.portrait-geometry-runtime-result/132"
        and result["status"] == "complete"
        and result["new_neural_calls"] == 0
        and result["application_packages_unchanged"] is True
        and result["defaults_changed"] is False
        and result["freeze_sha256"] == sha(runtime.PUBLIC)
        and result["private_installed_manifest_sha256"] == sha(manifest)
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["worker_receipt_sha256"] == sha(folder / "output/result.json")
        and profile["resource_journal_sha256"] == sha(folder / "resources.jsonl"),
        "Actual successful isolated v132 runtime and installed manifest required",
    )
    wheels = []
    for row in runtime.plan():
        wheel = runtime.TRIAL / "wheels" / row["filename"]
        require(sha(wheel) == row["sha256"], "Unchanged publisher wheel bytes required")
        wheels.append(wheel)
    require(
        runtime.installed_manifest(wheels, runtime.SITE) == read(manifest),
        "Installed isolated source/native bytes changed",
    )
    catalogue = ASSETS / "permissions/liveportrait-verified-assets.json"
    licence = ASSETS / "permissions/liveportrait-source-LICENSE.txt"
    task = ASSETS / "permissions/face_landmarker.task"
    require(
        sha(catalogue) == assets.CATALOGUE_SHA
        and sha(licence) == assets.LICENCE_SHA
        and sha(task) == assets.MESH_SHA,
        "Reviewed commercial asset catalogue, licence and exact face task required",
    )
    record = read(catalogue)
    require(
        record["publisher_checkpoint_license"] == "MIT"
        and record["restricted_face_models_downloaded"] is False,
        "Only reviewed unrestricted asset records admitted",
    )
    cards = {
        name: {"sha256": sha(ASSETS / "permissions" / name), "reviewed_license": "Apache-2.0"}
        for name in ("blazeface-card.pdf", "facemesh-card.pdf", "blendshape-card.pdf")
    }
    return {
        "runtime_freeze_sha256": sha(runtime.PUBLIC),
        "runtime_result_sha256": sha(folder / "output/result.json"),
        "runtime_profile_sha256": sha(folder / "profile.json"),
        "installed_manifest_sha256": sha(manifest),
        "asset_catalogue_sha256": assets.CATALOGUE_SHA,
        "source_license_sha256": assets.LICENCE_SHA,
        "face_task_sha256": assets.MESH_SHA,
        "bundled_face_model_cards": cards,
    }


def identity():
    expected = plan()
    require(read(TRIAL / "plan.json") == expected, "Unchanged completed v143 pilot plan required")
    p = planner()
    return {
        "schema": "chipmunk.grid-portrait-geometry-freeze/144",
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "private_plan_sha256": sha(TRIAL / "plan.json"),
        "transfer_plan_freeze_sha256": sha(p.PUBLIC),
        "transfer_plan_result_sha256": sha(p.TRIAL / "bind-00/output/result.json"),
        "private_transfer_plan_sha256": sha(p.TRIAL / "bind-00/output/private-plan.json"),
        **runtime_verified(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "opencv-python-headless", "psutil")
        },
        "geometry": GEOMETRY,
        "pilot_clips": len(expected["pilots"]),
        "distinct_frame_inputs_per_ready_clip": 28,
        "face_task_inputs_per_ready_clip": TASK_INPUTS,
        "frame_indices": list(FRAME_INDICES),
        "repeat_frame_index": REPEAT_FRAME,
        "stage_policy": "One exclusive neural authored stage per original pilot;900s,6GiB initial available RAM/sampled descendant RSS,3GiB initial project/archive disk,1.5GiB running reserves;no retries or substitutions.",
        "scope": "Face placement/crop/support geometry only;no generator,encoder,detector,achieved motion,identity recognition or confidence change. Missing/ambiguous/invalid observations retained unavailable;all individual vectors and media private.",
    }


def freeze():
    expected = plan()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "plan.json", expected)
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(
        json.dumps(
            {
                "status": "frozen",
                "pilot_clips": len(expected["pilots"]),
                "face_task_inputs_per_ready_clip": TASK_INPUTS,
            }
        ),
        flush=True,
    )


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged published geometry preparation contract required",
    )


def pilot(ordinal):
    require(type(ordinal) is int and 0 <= ordinal < 32, "One explicit original clip ordinal required")
    rows = [r for r in read(TRIAL / "plan.json")["pilots"] if r["ordinal"] == ordinal]
    require(len(rows) == 1, "Only the actual frozen pilot can be processed")
    return rows[0]


def canonical_hex(value, shape):
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    require(array.shape == shape and np.isfinite(array).all(), "Complete finite geometry array required")
    return np.asarray([float(v).hex() for v in array.flat], dtype=object).reshape(shape).tolist()


def decode_hex(value, shape):
    import numpy as np

    raw = np.asarray(value, dtype=object)
    require(
        raw.shape == shape and all(type(v) is str for v in raw.flat), "Canonical geometry hex array required"
    )
    decoded = np.asarray([float.fromhex(v) for v in raw.flat], dtype=np.float64).reshape(shape)
    require(canonical_hex(decoded, shape) == value, "Noncanonical or nonfinite cached geometry")
    return decoded


def upscale_matrix(matrix):
    import numpy as np

    matrix = np.asarray(matrix, dtype=np.float64)
    require(
        matrix.shape == (2, 3) and np.isfinite(matrix).all() and abs(np.linalg.det(matrix[:, :2])) > 1e-12,
        "Invertible source-to-256 geometry required",
    )
    result = matrix * 2
    result[:, 2] += 0.5
    return result


def base_record(picture, time_base):
    return {
        "frame_index": picture["frame_index"],
        "pts": picture["pts"],
        "time_base": time_base,
        "source_file": picture["file"],
        "source_file_sha256": picture["file_sha256"],
        "source_rgb_sha256": picture["rgb_sha256"],
    }


def observe_geometry(picture, time_base, rgb, detect, crop_fn, alpha_fn, event):
    """Mockable one-call observation; only explicit known availability is caught."""
    import numpy as np

    row = base_record(picture, time_base)
    event({"event": "face_task_begin", "frame_index": picture["frame_index"]})
    result = detect(rgb)
    faces = result.face_landmarks
    event({"event": "face_task_end", "frame_index": picture["frame_index"], "observed_faces": len(faces)})
    if len(faces) != 1:
        row["status"] = "unavailable_no_face" if not faces else "unavailable_multiple_faces"
        return row, None, None
    mesh = faces[0]
    try:
        values = np.asarray([(p.x, p.y, p.z) for p in mesh], dtype=np.float64)
        require(values.shape == (478, 3) and np.isfinite(values).all(), "Finite 478-point mesh required")
    except (ValueError, TypeError, AttributeError):
        row["status"] = "unavailable_mesh"
        return row, None, None
    row["mesh_xyz_hex"] = canonical_hex(values, (478, 3))
    try:
        crop, matrix, ratio = crop_fn(rgb, mesh)
        alpha = alpha_fn(rgb.shape, mesh)
        full = upscale_matrix(matrix)
        require(crop.shape == (256, 256, 3) and crop.dtype == np.uint8, "Exact full256 uint8 crop required")
        require(np.isfinite(ratio) and 0 <= ratio <= 1, "Finite source aperture required")
        require(
            alpha.shape == rgb.shape[:2]
            and alpha.dtype == np.float32
            and np.isfinite(alpha).all()
            and ((alpha >= 0) & (alpha <= 1)).all()
            and np.any(alpha > 0),
            "Finite nonempty float32 native mouth support required",
        )
    except (ValueError, TypeError, AttributeError):
        row["status"] = "unavailable_geometry"
        return row, None, None
    row.update(
        status=AVAILABLE,
        source_to_256_matrix_hex=canonical_hex(matrix, (2, 3)),
        source_to_512_matrix_hex=canonical_hex(full, (2, 3)),
        source_ratio_hex=float(ratio).hex(),
    )
    return row, crop, alpha


def repeat_check(current, repeated):
    if current["status"] != AVAILABLE or repeated["status"] != AVAILABLE:
        return None
    fields = ("mesh_xyz_hex", "source_to_256_matrix_hex", "source_to_512_matrix_hex", "source_ratio_hex")
    return all(current[k] == repeated[k] for k in fields) and all(
        current[k]["sha256"] == repeated[k]["sha256"] for k in ("crop", "alpha")
    )


def owned_paths(row, folder):
    for field, name in (("crop", "crop.png"), ("alpha", "alpha.npy")):
        path = Path(row[field]["path"])
        require(
            path.resolve() == (folder / name).resolve() and path.resolve().is_relative_to(CACHE.resolve()),
            "Geometry payload escaped its exclusive owned cache",
        )


def resource_verified(folder, profile):
    rows = [json.loads(line) for line in (folder / "resources.jsonl").read_text().splitlines()]
    require(
        rows and profile["resource_journal_sha256"] == sha(folder / "resources.jsonl"),
        "Actual full resource journal required",
    )
    for i, row in enumerate(rows):
        require(
            row["violations"] == []
            and type(row["elapsed_s"]) in (int, float)
            and math.isfinite(row["elapsed_s"])
            and 0 <= row["elapsed_s"] <= 900
            and (i == 0 or rows[i - 1]["elapsed_s"] <= row["elapsed_s"])
            and all(
                type(row[k]) is int and row[k] >= 0
                for k in (
                    "tree_rss_bytes",
                    "ram_available_bytes",
                    "disk_free_bytes",
                    "archive_free_disk_bytes",
                )
            )
            and row["tree_rss_bytes"] <= 6 * 1024**3
            and row["ram_available_bytes"] >= (6 * 1024**3 if i == 0 else 1536 * 1024**2)
            and min(row["disk_free_bytes"], row["archive_free_disk_bytes"])
            >= (3 * 1024**3 if i == 0 else 1536 * 1024**2),
            "Original complete neural-stage resources required",
        )
    require(
        type(profile["duration_s"]) in (int, float)
        and math.isfinite(profile["duration_s"])
        and rows[-1]["elapsed_s"] <= profile["duration_s"] <= 900
        and profile["sampled_tree_rss_lower_bound_bytes"] == max(r["tree_rss_bytes"] for r in rows)
        and profile["minimum_available_ram_bytes"] == min(r["ram_available_bytes"] for r in rows),
        "Profile/resource observations differ",
    )


def load_completed(ordinal):
    """Metadata-only admission; payload hashes/pixels are checked by their consumer."""
    verify()
    planned = pilot(ordinal)
    folder = TRIAL / f"clip-{ordinal:02d}/authored"
    out = folder / "output"
    result, profile = read(out / "result.json"), read(folder / "profile.json")
    records = read(out / "geometry-records.json")
    cache_records = CACHE / f"clip-{ordinal:02d}/geometry-records.json"
    require(
        result["schema"] == "chipmunk.grid-portrait-geometry-result/144"
        and result["status"] == "complete"
        and result["task_delegate"] == "CPU"
        and result["running_mode"] == "IMAGE"
        and result["num_faces"] == 2
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["defaults_changed"] is False
        and result["clip_ordinal"] == ordinal
        and result["generator_calls"] == result["encoder_calls"] == result["detector_calls"] == 0
        and result["private_geometry_records_sha256"] == sha(out / "geometry-records.json")
        and result["private_geometry_cache_sha256"] == sha(cache_records)
        and result["private_face_progress_sha256"] == sha(out / "face-progress.jsonl")
        and result["private_repeat_check_sha256"] == sha(out / "repeat-check.json")
        and profile["command"] == "authored"
        and profile["batch"] is None
        and profile["child_started"] is True
        and profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["error_types"] == []
        and profile["resource_limit_observations"] == []
        and profile["worker_receipt_sha256"] == sha(out / "result.json")
        and profile["resource_journal_sha256"] == sha(folder / "resources.jsonl")
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["new_neural_calls"] == result["new_neural_calls"] == result["face_task_inputs_completed"]
        and records["schema"] == "chipmunk.grid-portrait-geometry-records/144"
        and records["clip_ordinal"] == ordinal
        and records["planned"] == planned,
        "Actual bound successful per-pilot geometry stage required",
    )
    require(read(cache_records) == records, "Private Seagate geometry cache differs from stage record")
    resource_verified(folder, profile)
    requested = requests(planned)
    all_rows = [*records["pictures"], records["repeat"]] if requested else []
    require(
        len(all_rows) == len(requested)
        and len(records["pictures"]) == (28 if requested else 0)
        and (bool(requested) or records["repeat"] is None),
        "Every original geometry request must remain recorded",
    )
    time_base = planned["input"].get("native_video_table", {}).get("time_base")
    for index, (row, picture) in enumerate(zip(all_rows, requested, strict=True)):
        require(
            all(row[k] == v for k, v in base_record(picture, time_base).items()),
            "Cached geometry lost its exact native picture correspondence",
        )
        require(
            row["status"] in {AVAILABLE, *UNAVAILABLE}, "Unknown geometry availability cannot be promoted"
        )
        if row["status"] == AVAILABLE:
            decode_hex(row["mesh_xyz_hex"], (478, 3))
            matrix = decode_hex(row["source_to_256_matrix_hex"], (2, 3))
            require(
                canonical_hex(upscale_matrix(matrix), (2, 3)) == row["source_to_512_matrix_hex"],
                "Cached512 pixel-centre matrix differs",
            )
            ratio = float.fromhex(row["source_ratio_hex"])
            require(
                0 <= ratio <= 1 and ratio.hex() == row["source_ratio_hex"],
                "Canonical finite cached aperture required",
            )
            cache_folder = (
                CACHE
                / f"clip-{ordinal:02d}"
                / (f"repeat-{REPEAT_FRAME:04d}" if index == 28 else f"frame-{row['frame_index']:04d}")
            )
            owned_paths(row, cache_folder)
    repeated = (
        repeat_check(records["pictures"][REPEAT_FRAME - FRAME_INDICES[0]], records["repeat"])
        if requested
        else None
    )
    require(
        records["repeat_bitwise"] is repeated
        and read(out / "repeat-check.json")["repeat_bitwise"] is repeated
        and repeated is not False,
        "Available geometry own repeat must be exact",
    )
    qualified = bool(requested) and all(r["status"] == AVAILABLE for r in all_rows) and repeated is True
    require(
        records["geometry_qualified"] is result["geometry_qualified"] is qualified,
        "Geometry availability cannot become an inferred verdict",
    )
    require(
        result["repeat_bitwise"] is repeated
        and result["availability"] == dict(Counter(r["status"] for r in all_rows))
        and result["retained_distinct_picture_records"] == len(records["pictures"]),
        "Actual geometry aggregates differ from private observations",
    )
    events = [json.loads(line) for line in (out / "face-progress.jsonl").read_text().splitlines()]
    active, starts, ends, observed, request_ends, face_count = None, 0, 0, [], 0, None
    for event in events:
        if event["event"] == "face_task_begin":
            require(
                active is None
                and request_ends == 0
                and len(observed) < len(all_rows)
                and event["frame_index"] == all_rows[len(observed)]["frame_index"],
                "Unmatched planned face-task begin",
            )
            active = event["frame_index"]
            starts += 1
        elif event["event"] == "face_task_end":
            require(
                active is not None
                and active == event["frame_index"]
                and type(event["observed_faces"]) is int
                and event["observed_faces"] >= 0,
                "Unmatched face-task completion",
            )
            active = None
            ends += 1
            request_ends += 1
            face_count = event["observed_faces"]
        else:
            require(
                event["event"] == "geometry_record"
                and active is None
                and len(observed) < len(all_rows)
                and type(event["request_ordinal"]) is int
                and event["request_ordinal"] == len(observed),
                "Complete ordered geometry journal required",
            )
            row = all_rows[len(observed)]
            require(
                event["frame_index"] == row["frame_index"] and event["status"] == row["status"],
                "Geometry journal/record correspondence differs",
            )
            require(
                request_ends == int(row["status"] != "unavailable_picture_decode"),
                "Each safe request needs its own face-task completion",
            )
            expected_faces = (
                0
                if row["status"] == "unavailable_no_face"
                else (None if row["status"] == "unavailable_picture_decode" else 1)
            )
            require(
                face_count > 1
                if row["status"] == "unavailable_multiple_faces"
                else face_count == expected_faces,
                "Observed face count differs from geometry availability",
            )
            observed.append(event)
            request_ends = 0
            face_count = None
    require(
        active is None
        and len(observed) == len(all_rows)
        and starts
        == ends
        == result["face_task_inputs_started"]
        == result["face_task_inputs_completed"]
        == sum(r["status"] != "unavailable_picture_decode" for r in all_rows)
        and result["planned_face_task_inputs"] == len(requested),
        "Every safe planned face-task call must be complete and retained",
    )
    return records


def completed(ordinal):
    folder = TRIAL / f"clip-{ordinal:02d}/authored"
    if not all(p.is_file() for p in (folder / "output/result.json", folder / "profile.json")):
        return False
    load_completed(ordinal)
    return True


def prepare(ordinal, output):
    import cv2
    import numpy as np

    from scripts import portrait_geometry_runtime_v132 as runtime

    runtime_verified()  # Includes actual installed bytes/assets, before the first face forward.
    require("mediapipe" not in sys.modules, "Fresh isolated task import required")
    sys.path.insert(0, str(runtime.SITE))
    import mediapipe as mp

    from scripts.liveportrait_mouth_benchmark import crop_from_mesh
    from scripts.prepare_mouth_edit_v4 import mouth_alpha

    require(
        Path(mp.__file__).resolve().is_relative_to(runtime.SITE.resolve()),
        "Qualified isolated face API required",
    )
    require(
        importlib.metadata.version("mediapipe") == runtime.PINS["mediapipe"][0],
        "Exact isolated face package version required",
    )
    cv2.setNumThreads(1)
    planned = pilot(ordinal)
    requested = requests(planned)
    cache = CACHE / f"clip-{ordinal:02d}"
    cache.mkdir(parents=True, exist_ok=False)
    rows, counts = [], Counter()
    with (output / "face-progress.jsonl").open("x") as journal:

        def event(row):
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            counts[row["event"]] += 1

        options_dict = dict(GEOMETRY)
        options_dict.pop("delegate")
        options_dict["running_mode"] = mp.tasks.vision.RunningMode.IMAGE
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(ASSETS / "permissions/face_landmarker.task"),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            **options_dict,
        )
        if requested:
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as task:

                def detect(rgb):
                    return task.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

                for position, picture in enumerate(requested):
                    row = base_record(picture, planned["input"]["native_video_table"]["time_base"])
                    source = Path(picture["file"])
                    expected = (
                        ARCHIVE
                        / "grid-native-inputs-v137"
                        / f"source-{planned['source_slot']:02d}"
                        / f"clip-{planned['within_source_ordinal']:02d}"
                        / f"frame-{picture['frame_index']:04d}.png"
                    )
                    require(
                        source.resolve() == expected.resolve()
                        and source.resolve().is_relative_to((ARCHIVE / "grid-native-inputs-v137").resolve())
                        and sha(source) == picture["file_sha256"],
                        "Exact bound original PNG bytes required before decode",
                    )
                    bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
                    if bgr is None:
                        row["status"] = "unavailable_picture_decode"
                    else:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        table = planned["input"]["native_video_table"]
                        require(
                            rgb.shape == (table["height"], table["width"], 3)
                            and rgb.dtype == np.uint8
                            and hashlib.sha256(rgb.tobytes()).hexdigest() == picture["rgb_sha256"],
                            "Decoded native RGB bytes differ from frozen plain-byte commitment",
                        )
                        row, crop, alpha = observe_geometry(
                            picture, table["time_base"], rgb, detect, crop_from_mesh, mouth_alpha, event
                        )
                        if row["status"] == AVAILABLE:
                            destination = cache / (
                                f"repeat-{REPEAT_FRAME:04d}"
                                if position == 28
                                else f"frame-{picture['frame_index']:04d}"
                            )
                            destination.mkdir(exist_ok=False)
                            crop_path, alpha_path = destination / "crop.png", destination / "alpha.npy"
                            require(
                                cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)),
                                "Could not retain exact geometry crop",
                            )
                            recovered = cv2.cvtColor(cv2.imread(str(crop_path)), cv2.COLOR_BGR2RGB)
                            require(np.array_equal(recovered, crop), "Saved geometry crop differs")
                            with alpha_path.open("xb") as stream:
                                np.save(stream, alpha, allow_pickle=False)
                            require(
                                np.array_equal(np.load(alpha_path, allow_pickle=False), alpha),
                                "Saved exact float alpha differs",
                            )
                            row.update(
                                crop={"path": str(crop_path), "sha256": sha(crop_path)},
                                alpha={"path": str(alpha_path), "sha256": sha(alpha_path)},
                            )
                    rows.append(row)
                    event(
                        {
                            "event": "geometry_record",
                            "request_ordinal": position,
                            "frame_index": picture["frame_index"],
                            "status": row["status"],
                        }
                    )
                    write(output / f"request-{position:02d}.json", row)
        repeat = repeat_check(rows[REPEAT_FRAME - FRAME_INDICES[0]], rows[-1]) if requested else None
        qualified = bool(requested) and all(row["status"] == AVAILABLE for row in rows) and repeat is True
        records = {
            "schema": "chipmunk.grid-portrait-geometry-records/144",
            "clip_ordinal": ordinal,
            "planned": planned,
            "pictures": rows[:28],
            "repeat": rows[-1] if requested else None,
            "repeat_bitwise": repeat,
            "geometry_qualified": qualified,
        }
        write(output / "geometry-records.json", records)
        write(cache / "geometry-records.json", records)
        write(output / "repeat-check.json", {"repeat_bitwise": repeat, "geometry_qualified": qualified})
        require(repeat is not False, "Available IMAGE-mode own repeat differs; preserve failed stage")
        require(
            counts["face_task_begin"] == counts["face_task_end"]
            and counts["face_task_begin"] == sum(r["status"] != "unavailable_picture_decode" for r in rows),
            "Every safe planned input needs exactly one completed face-task API call",
        )
        require(
            "insightface" not in sys.modules
            and not any(k.startswith("chipmunk_benchmark_portrait") for k in sys.modules),
            "No restricted face or generator model import permitted",
        )
        result = {
            "schema": "chipmunk.grid-portrait-geometry-result/144",
            "status": "complete",
            "clip_ordinal": ordinal,
            "task_delegate": "CPU",
            "running_mode": "IMAGE",
            "num_faces": 2,
            "planned_face_task_inputs": TASK_INPUTS if requested else 0,
            "retained_distinct_picture_records": len(records["pictures"]),
            "face_task_inputs_started": counts["face_task_begin"],
            "face_task_inputs_completed": counts["face_task_end"],
            "new_neural_calls": counts["face_task_end"],
            "generator_calls": 0,
            "encoder_calls": 0,
            "detector_calls": 0,
            "availability": dict(Counter(r["status"] for r in rows)),
            "geometry_qualified": qualified,
            "repeat_bitwise": repeat,
            "private_geometry_records_sha256": sha(output / "geometry-records.json"),
            "private_geometry_cache_sha256": sha(cache / "geometry-records.json"),
            "private_face_progress_sha256": sha(output / "face-progress.jsonl"),
            "private_repeat_check_sha256": sha(output / "repeat-check.json"),
            "freeze_sha256": sha(PUBLIC),
            "defaults_changed": False,
            "scope": "Original pilot geometry/crop/support only. Unavailable observations retained without replacement;no generation,identity recognition,detector score,achieved alignment or original-file verdict. Individual face vectors and media private.",
        }
        write(output / "result.json", result)
        print(json.dumps(result), flush=True)


def routed_popen(ordinal, argv, **kwargs):
    expected = TRIAL / f"clip-{ordinal:02d}/authored/output"
    require(
        type(ordinal) is int
        and 0 <= ordinal < 32
        and argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"]
        and argv[5:] == [str(expected)],
        "Only exact exclusive per-pilot neural geometry stage permitted",
    )
    return subprocess.Popen(
        [*argv[:3], "scripts.grid_portrait_geometry_v144", "authored", str(expected), str(ordinal)], **kwargs
    )


def run(ordinal):
    from scripts import video_appearance_runtime_v108 as runtime
    from scripts.video_appearance_corrected_v112 import scoped

    pilot(ordinal)
    stage = TRIAL / f"clip-{ordinal:02d}"
    stage.mkdir(exist_ok=True)
    proxy = types.SimpleNamespace(
        Popen=lambda argv, **kw: routed_popen(ordinal, argv, **kw),
        STDOUT=subprocess.STDOUT,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    scoped(
        runtime.run,
        {
            "TRIAL": stage,
            "verify": verify,
            "prerequisites": lambda c, b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("authored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "authored"))
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
        output, ordinal = Path(args.arguments[0]), int(args.arguments[1])
        verify()
        pilot(ordinal)
        require(
            output.resolve() == (TRIAL / f"clip-{ordinal:02d}/authored/output").resolve(),
            "Exact exclusive output required",
        )
        output.mkdir(exist_ok=False)
        prepare(ordinal, output)
