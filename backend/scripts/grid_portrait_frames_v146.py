"""Private per-frame full-expression GRID construction; no detection/timing claim."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import types
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
PANEL = ROOT / "docs/evaluation-panels"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
TRIAL = BASE / "grid-portrait-frames-v146"
CACHE = ARCHIVE / "grid-portrait-frames-v146"
PUBLIC = PANEL / "grid-portrait-frames-freeze-v146.json"
ASSETS = BASE / "lipsync-v4"
CONDITIONS = ("source_time", "earlier_320ms")
FRAMES = tuple(range(27, 47))
ROOT_NAMES = (
    "appearance_feature_extractor",
    "motion_extractor",
    "warping_module",
    "spade_generator",
    "lip",
)
CODE = (
    "backend/scripts/grid_portrait_frames_v146.py",
    "backend/tests/test_grid_portrait_frames_v146.py",
    "docs/grid-portrait-frames-v146-method.md",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/scripts/video_appearance_corrected_v112.py",
)
FROZEN_CODE = (
    "backend/scripts/portrait_expression_drive_v138.py",
    "backend/scripts/spatial_conv_cpu_v139.py",
    "backend/scripts/portrait_tiled_runtime_v139.py",
    "backend/scripts/portrait_expression_qualifier_v140.py",
    "backend/scripts/spatial_conv_install_v141.py",
)
UNAVAILABLE = {
    "unavailable_no_face",
    "unavailable_multiple_faces",
    "unavailable_mesh",
    "unavailable_geometry",
    "unavailable_picture_decode",
    "unavailable_planned_input",
}


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


def fingerprint(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def exact(value):
    require(type(value) in (int, str, Fraction), "Exact native rational clock required")
    return Fraction(value)


def dependencies():
    from scripts import grid_portrait_geometry_v144 as geometry
    from scripts import grid_transfer_plan_v143 as planning

    return planning, geometry


def completed_renderer():
    """Only bind published actual JSON/journals here, not old media or neural calls."""
    folder = BASE / "portrait-expression-qualifier-v140"
    frozen = read(PANEL / "portrait-expression-qualifier-freeze-v140.json")
    require(frozen == read(folder / "freeze.json"), "Same published v140 freeze required")
    for path, digest in frozen["code_sha256"].items():
        require(sha(ROOT / path) == digest, "Qualified historical renderer code changed")
    require(all(path in frozen["code_sha256"] for path in FROZEN_CODE), "Exact frozen renderer path required")
    output = folder / "authored/output"
    result, profile = read(output / "result.json"), read(folder / "authored/profile.json")
    public = read(PANEL / "portrait-expression-qualifier-result-v140.json")
    counts = dict(zip(ROOT_NAMES, (3, 6, 3, 3, 0), strict=True))
    require(
        result["schema"] == "chipmunk.portrait-expression-qualifier-result/140"
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(folder / "freeze.json")
        and result["strict_exact_checkpoint_loads"] == 5
        and result["renderer_inputs"] == 3
        and result["new_face_task_inputs"] == 0
        and result["new_neural_calls"] == 15
        and result["top_level_module_forwards_started"] == counts
        and result["top_level_module_forwards_completed"] == counts
        and result["renderer_repeat_bitwise"] is True
        and result["renderer_response_qualified"] is True
        and result["native_coordinate_containment_qualified"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["new_neural_calls"] == 15
        and profile["worker_receipt_sha256"] == sha(output / "result.json")
        and public["status"] == "complete"
        and public["measurement"] == result
        and public["profile"] == profile
        and public["result_sha256"] == sha(output / "result.json")
        and public["profile_sha256"] == sha(folder / "authored/profile.json"),
        "Actual successful published full-expression CPU qualification required",
    )
    bindings = {}
    for name, key in (
        ("private-render-records.json", "private_render_records_sha256"),
        ("private-input-files.json", "private_input_files_sha256"),
        ("render-records.jsonl", "private_render_journal_sha256"),
        ("forward-progress.jsonl", "private_forward_progress_sha256"),
    ):
        path = output / name
        require(sha(path) == result[key], "Qualified renderer private journal changed")
        bindings[str(path.relative_to(ROOT))] = sha(path)
    for path in (
        folder / "freeze.json",
        PANEL / "portrait-expression-qualifier-freeze-v140.json",
        PANEL / "portrait-expression-qualifier-result-v140.json",
        output / "result.json",
        folder / "authored/profile.json",
    ):
        bindings[str(path.relative_to(ROOT))] = sha(path)
    return bindings


def pilot_source(ordinal):
    planning, _ = dependencies()
    plan = planning.load_completed()
    require(type(ordinal) is int and 0 <= ordinal < 32, "Original integer clip ordinal required")
    require(
        plan["schema"] == "chipmunk.grid-native-transfer-plan/143"
        and len(plan["sources"]) == 32
        and [row["ordinal"] for row in plan["sources"]] == list(range(32)),
        "Complete original ordered GRID plan required",
    )
    source = plan["sources"][ordinal]
    require(source["role"] == "pilot", "Reserve clips may not be consumed by this pilot")
    return source


def identity():
    planning, geometry = dependencies()
    plan = planning.load_completed()
    frozen = read(geometry.PUBLIC)
    require(frozen == read(geometry.TRIAL / "freeze.json"), "Published geometry procedure required")
    for path, digest in frozen["code_sha256"].items():
        require(sha(ROOT / path) == digest, "Geometry procedure changed")
    pilots = [r["ordinal"] for r in plan["sources"] if r["role"] == "pilot"]
    return {
        "schema": "chipmunk.grid-portrait-frames-freeze/146",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "renderer_bindings_sha256": completed_renderer(),
        "plan_freeze_sha256": sha(planning.PUBLIC),
        "plan_private_sha256": sha(planning.TRIAL / "bind-00/output/private-plan.json"),
        "geometry_freeze_sha256": sha(geometry.PUBLIC),
        "pilot_clip_ordinals": pilots,
        "per_clip_planned_condition_frames": 40,
        "per_clip_planned_repeat_frames": 1,
        "per_clip_maximum_renders": 41,
        "per_clip_maximum_top_module_forward_units": 205,
        "conditions": list(CONDITIONS),
        "source_frames_half_open": [27, 47],
        "earlier_native_frame_offset": -8,
        "required_earlier_time_offset_seconds": "-8/25",
        "device": "cpu",
        "precision": "float32",
        "seed": "146 + original clip ordinal",
        "threads": {"torch": 1, "interop": 1, "opencv": 1},
        "deterministic_algorithms": True,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "opencv-python-headless", "Pillow", "PyYAML")
        },
        "scope": "Fixed pilot frame construction only; per-pair unavailable geometry retained. No new face/speech inputs, activation caches, achieved phonetic timing, face-swap accuracy, detector score or serving change.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(
        json.dumps(
            {"status": "frozen", "planned_clips": len(body["pilot_clip_ordinals"]), "new_neural_calls": 0}
        ),
        flush=True,
    )


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged published v146 procedure required",
    )


def hex_array(value, shape):
    raw = np.asarray(value, dtype=object)
    require(
        raw.shape == shape and all(type(v) is str for v in raw.flat),
        "Exact cached canonical hexadecimal array required",
    )
    result = np.asarray([float.fromhex(v) for v in raw.flat], np.float64).reshape(shape)
    require(
        np.isfinite(result).all()
        and all(float(v).hex() == s for v, s in zip(result.flat, raw.flat, strict=True)),
        "Finite canonical geometry values required",
    )
    return result


def upscale_matrix(matrix):
    require(matrix.shape == (2, 3) and np.isfinite(matrix).all(), "Finite source-to-crop matrix required")
    output = matrix * 2
    output[:, 2] += 0.5
    return output


def row_contract(row, source):
    """Pure source/crop geometry binding, including rows that are unavailable."""
    index = row["frame_index"]
    require(type(index) is int and index in range(19, 47), "Fixed native geometry index required")
    original = source["input"]
    pictures = {p["frame_index"]: p for p in original.get("pictures", [])}
    if row["status"] == "unavailable_planned_input":
        require(index not in pictures, "Available source picture cannot disappear from a geometry record")
        return None
    require(index in pictures, "Geometry record must correspond to a bound source picture")
    expected, table = pictures[index], original["native_video_table"]
    require(
        row["pts"] == expected["pts"] == table["pts"][index]
        and row["time_base"] == table["time_base"]
        and row["source_file"] == expected["file"]
        and row["source_file_sha256"] == expected["file_sha256"]
        and row["source_rgb_sha256"] == expected["rgb_sha256"],
        "Geometry/source native picture and clock correspondence changed",
    )
    require(row["status"] in UNAVAILABLE | {"geometry_available"}, "Known geometry status required")
    if row["status"] != "geometry_available":
        return None
    mesh = hex_array(row["mesh_xyz_hex"], (478, 3))
    matrix = hex_array(row["source_to_256_matrix_hex"], (2, 3))
    output = hex_array(row["source_to_512_matrix_hex"], (2, 3))
    require(np.array_equal(output, upscale_matrix(matrix)), "Exact 512 pixel-centre mapping required")
    ratio = float.fromhex(row["source_ratio_hex"])
    require(
        np.isfinite(ratio) and 0 <= ratio <= 1 and float(ratio).hex() == row["source_ratio_hex"],
        "Finite canonical source aperture required",
    )
    return mesh, matrix, output


def jobs_for(source, records):
    """Keep all 40 source/driver pairs; unrelated unavailable geometry never vetoes them."""
    require(
        records["schema"] == "chipmunk.grid-portrait-geometry-records/144"
        and records["clip_ordinal"] == source["ordinal"]
        and records["planned"] == source,
        "Complete exact per-clip geometry metadata required",
    )
    require(
        records["repeat_bitwise"] is True or records["repeat_bitwise"] is None,
        "Failed available geometry repeat cannot be used",
    )
    if not records["pictures"]:
        require(
            source["geometry"]["status"] == "unavailable"
            and source["geometry"]["pictures"] == []
            and records["repeat"] is None
            and records["repeat_bitwise"] is None
            and records["geometry_qualified"] is False,
            "Empty geometry must be the bound explicitly unavailable request set",
        )
        return [
            {
                "condition": condition,
                "frame_index": index,
                "driver_frame_index": index if condition == "source_time" else index - 8,
                "status": "unavailable",
                "reason": {"planned_geometry": source["geometry"]["unavailable_reason"]},
            }
            for condition in CONDITIONS
            for index in FRAMES
        ], {}
    require(
        [r["frame_index"] for r in records["pictures"]] == list(range(19, 47)),
        "Every requested source/driver geometry row must remain recorded",
    )
    by_index = {r["frame_index"]: r for r in records["pictures"]}
    for row in records["pictures"]:
        row_contract(row, source)
    frames = source["frames"]
    require([f["frame_index"] for f in frames] == list(FRAMES), "All planned source picture routes required")
    plans = {f["frame_index"]: f for f in frames}
    jobs = []
    for condition in CONDITIONS:
        for index in FRAMES:
            driver_index = index if condition == "source_time" else index - 8
            current, driver = by_index[index], by_index[driver_index]
            available = current["status"] == driver["status"] == "geometry_available"
            route = plans[index]
            if available:
                t = exact(current["pts"]) * exact(current["time_base"])
                d = exact(driver["pts"]) * exact(driver["time_base"])
                require(
                    t == exact(route["source_time_seconds"])
                    and d
                    == exact(
                        route["source_time_seconds"]
                        if condition == "source_time"
                        else route["earlier_time_seconds"]
                    )
                    and d - t == (0 if condition == "source_time" else Fraction(-8, 25)),
                    "Actual source/driver PTS must prove the exact frozen offset",
                )
                expected = route["source_picture"] if condition == "source_time" else route["earlier_picture"]
                require(
                    expected["frame_index"] == driver_index
                    and expected["file"] == driver["source_file"]
                    and expected["pts"] == driver["pts"],
                    "Exact planned driver picture required",
                )
            jobs.append(
                {
                    "condition": condition,
                    "frame_index": index,
                    "driver_frame_index": driver_index,
                    "status": "available" if available else "unavailable",
                    "reason": None
                    if available
                    else {"source": current["status"], "driver": driver["status"]},
                }
            )
    return jobs, by_index


def opacity(index):
    require(type(index) is int and index in FRAMES, "Fixed planned source frame required")
    return 0.5 if index in (27, 46) else 1.0


def composite(source, transformed, alpha, weight):
    require(
        source.dtype == transformed.dtype == np.uint8
        and source.shape == transformed.shape
        and source.ndim == 3
        and source.shape[2] == 3
        and alpha.dtype == np.float32
        and alpha.shape == source.shape[:2]
        and np.isfinite(alpha).all()
        and ((alpha >= 0) & (alpha <= 1)).all()
        and weight in (0.5, 1.0),
        "Corresponding finite native RGB and exact mouth alpha required",
    )
    a = alpha[:, :, None] * weight
    edited = (
        np.rint(source.astype(np.float32) * (1 - a) + transformed.astype(np.float32) * a)
        .clip(0, 255)
        .astype(np.uint8)
    )
    changed = np.any(edited != source, axis=2)
    require(not changed[alpha == 0].any(), "Native pixels changed outside exact mouth support")
    return edited, changed.astype(np.uint8) * 255


def generated_contract(value):
    require(
        isinstance(value, np.ndarray) and value.shape == (512, 512, 3) and value.dtype == np.uint8,
        "Complete finite uint8 512-square generator output required",
    )


def expected_roots(renders):
    require(type(renders) is int and 0 <= renders <= 41, "Bounded actual render count required")
    return dict(zip(ROOT_NAMES, (renders, 2 * renders, renders, renders, 0), strict=True))


def save_png(path, value):
    """Exclusive save plus an independent Pillow pixel round trip."""
    import cv2
    from PIL import Image

    require(
        value.dtype == np.uint8 and (value.ndim == 2 or value.ndim == 3 and value.shape[2] == 3),
        "Byte image required",
    )
    encoded_input = cv2.cvtColor(value, cv2.COLOR_RGB2BGR) if value.ndim == 3 else value
    ok, encoded = cv2.imencode(".png", encoded_input)
    require(ok, "Private PNG encoding failed")
    with Path(path).open("xb") as stream:
        stream.write(encoded.tobytes())
    with Image.open(path) as image:
        recovered = np.asarray(image.convert("RGB") if value.ndim == 3 else image)
    require(np.array_equal(recovered, value), "Saved private PNG pixels changed")
    return {"path": str(path), "sha256": sha(path), "raw_sha256": fingerprint(value)}


def alpha_from_mesh(mesh, shape):
    """Independent pure reconstruction of the frozen native feathered support."""
    height, width = shape[:2]
    points = mesh[:, :2] * np.array([width, height])
    center = (points[13] + points[14]) / 2
    axis = points[308] - points[78]
    span = float(np.linalg.norm(axis))
    require(np.isfinite(points).all() and span >= 8, "Sufficient measured mouth support required")
    axis /= span
    normal = np.array([-axis[1], axis[0]])
    y, x = np.indices((height, width))
    dx, dy = x - center[0], y - center[1]
    u, v = dx * axis[0] + dy * axis[1], dx * normal[0] + dy * normal[1]
    radius = np.sqrt((u / (0.7 * span)) ** 2 + (v / (0.45 * span)) ** 2)
    return np.clip((1 - radius) / 0.2, 0, 1).astype(np.float32)


def load_frame(row, source, geometry_cache):
    """Guarded worker only: validate actual saved input/crop/alpha bytes and geometry."""
    import cv2
    from PIL import Image

    mesh, matrix, output_matrix = row_contract(row, source)
    original = source["input"]
    shape = (original["native_video_table"]["height"], original["native_video_table"]["width"], 3)
    file = Path(row["source_file"])
    require(sha(file) == row["source_file_sha256"], "Original source PNG bytes changed")
    with Image.open(file) as image:
        rgb = np.asarray(image.convert("RGB"))
    require(
        rgb.shape == shape and fingerprint(rgb) == row["source_rgb_sha256"], "Original native raw RGB changed"
    )
    values = {}
    for key in ("crop", "alpha"):
        path = Path(row[key]["path"])
        require(
            path.is_absolute()
            and path.resolve().is_relative_to(geometry_cache.resolve())
            and sha(path) == row[key]["sha256"],
            "Exact owned cached geometry payload required",
        )
        if key == "crop":
            with Image.open(path) as image:
                values[key] = np.asarray(image.convert("RGB"))
        else:
            values[key] = np.load(path, allow_pickle=False)
    crop, alpha = values["crop"], values["alpha"]
    require(
        crop.shape == (256, 256, 3) and crop.dtype == np.uint8, "Exact cached 256-square RGB crop required"
    )
    reconstructed = cv2.warpAffine(
        rgb, matrix, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    require(np.array_equal(crop, reconstructed), "Cached crop differs from its original pixels/matrix")
    require(
        alpha.dtype == np.float32 and np.array_equal(alpha, alpha_from_mesh(mesh, shape)),
        "Cached native alpha differs from exact bound mesh support",
    )
    require(abs(float(np.linalg.det(matrix[:, :2]))) > 0, "Invertible crop matrix required")
    proof = {
        "frame_index": row["frame_index"],
        "pts": row["pts"],
        "time_base": row["time_base"],
        "source_png_sha256": sha(file),
        "source_rgb_sha256": fingerprint(rgb),
        "crop_png_sha256": row["crop"]["sha256"],
        "crop_rgb_sha256": fingerprint(crop),
        "alpha_file_sha256": row["alpha"]["sha256"],
        "alpha_f32_sha256": fingerprint(alpha),
        "matrix256_f64_sha256": fingerprint(matrix),
        "matrix512_f64_sha256": fingerprint(output_matrix),
    }
    return rgb, crop, alpha, output_matrix, proof


def input_binding(ordinal):
    source = pilot_source(ordinal)
    _, geometry = dependencies()
    records = geometry.load_completed(ordinal)
    jobs, by_index = jobs_for(source, records)
    folder = geometry.TRIAL / f"clip-{ordinal:02d}/authored"
    files = {
        str(path): sha(path)
        for path in (
            folder / "profile.json",
            folder / "resources.jsonl",
            folder / "output/result.json",
            folder / "output/geometry-records.json",
            folder / "output/face-progress.jsonl",
            folder / "output/repeat-check.json",
        )
    }
    return source, records, jobs, by_index, files


def build(ordinal, output):
    import cv2

    source, geometry_records, jobs, by_index, bindings = input_binding(ordinal)
    folder = CACHE / f"clip-{ordinal:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    write(
        output / "private-input-binding.json",
        {"source": source, "geometry": geometry_records, "bindings": bindings},
    )
    counts_started, counts_completed = expected_roots(0), expected_roots(0)
    bounded = {
        "spatial_convolutions": 0,
        "functional_tile_calls": 0,
        "maximum_estimated_tile_columns_bytes": 0,
    }
    state_loads, patches, handles, output_references = [], [], [], []
    repeat_reference = None
    executed, zero_effect, paired, unchanged_driver = 0, 0, 0, 0
    available = sum(j["status"] == "available" for j in jobs)
    repeat_status = "unassessed_source_time_frame27_unavailable"
    repeat_renders = 0
    active = {}
    _, geometry = dependencies()
    geometry_cache = geometry.CACHE / f"clip-{ordinal:02d}"
    renderer = None
    with (
        (output / "root-progress.jsonl").open("x") as root_journal,
        (output / "operator-progress.jsonl").open("x") as operator_journal,
        (output / "render-records.jsonl").open("x") as render_journal,
    ):

        def emit(journal, row):
            journal.write(
                json.dumps({"elapsed_monotonic_seconds": time.monotonic(), **active, **row}, allow_nan=False)
                + "\n"
            )
            journal.flush()

        def observe(row):
            emit(operator_journal, row)
            if row["event"] == "spatial_conv_begin":
                bounded["spatial_convolutions"] += 1
                bounded["maximum_estimated_tile_columns_bytes"] = max(
                    bounded["maximum_estimated_tile_columns_bytes"], row["estimated_tile_columns_bytes"]
                )
            elif row["event"] == "spatial_conv_end":
                bounded["functional_tile_calls"] += row["tiles"]

        def hook(name, completed):
            def observed(module, args, *values):
                (counts_completed if completed else counts_started)[name] += 1
                emit(
                    root_journal,
                    {"event": "top_module_end" if completed else "top_module_begin", "module": name},
                )

            return observed

        try:
            if available:
                import torch

                from scripts import spatial_conv_install_v141 as tiling
                from scripts.fetch_mouth_benchmark_assets import fetch
                from scripts.portrait_expression_drive_v138 import ExpressionRenderer

                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
                torch.use_deterministic_algorithms(True)
                torch.manual_seed(146 + ordinal)
                cv2.setNumThreads(1)
                fetch(ASSETS / "permissions/liveportrait-verified-assets.json", ASSETS, verify_only=True)
                actual_loader = torch.nn.Module.load_state_dict

                def exact_load(module, state, strict=True, assign=False):
                    require(
                        strict is True and assign is False, "Strict tensor-exact checkpoint loading required"
                    )
                    result = actual_loader(module, state, strict=strict, assign=assign)
                    loaded = module.state_dict()
                    require(
                        loaded.keys() == state.keys()
                        and all(torch.equal(loaded[k].cpu(), value.cpu()) for k, value in state.items()),
                        "Loaded checkpoint tensor differs",
                    )
                    state_loads.append({"strict": True, "exact": True, "tensors": len(state)})
                    emit(root_journal, {"event": "strict_state_load_complete", "tensors": len(state)})
                    return result

                emit(root_journal, {"event": "renderer_load_begin", "new_face_task_inputs": 0})
                try:
                    torch.nn.Module.load_state_dict = exact_load
                    renderer = ExpressionRenderer(ASSETS)
                finally:
                    torch.nn.Module.load_state_dict = actual_loader
                require(
                    len(state_loads) == 5 and set(renderer.renderer.models) == set(ROOT_NAMES),
                    "Five strict tensor-exact source model loads required",
                )
                patches = tiling.install(renderer.renderer.models, observe)
                for name, model in renderer.renderer.models.items():
                    handles.extend(
                        [
                            model.register_forward_pre_hook(hook(name, False)),
                            model.register_forward_hook(hook(name, True)),
                        ]
                    )

            saved_source_outputs = {}
            for job in jobs:
                active.update(
                    condition=job["condition"],
                    frame_index=job["frame_index"],
                    driver_frame_index=job["driver_frame_index"],
                    repeat=False,
                )
                if job["status"] != "available":
                    record = {**job, "rendered": False}
                    emit(render_journal, record)
                    output_references.append(record)
                    continue
                rgb, crop, alpha, matrix, source_proof = load_frame(
                    by_index[job["frame_index"]], source, geometry_cache
                )
                _, driver, _, _, driver_proof = load_frame(
                    by_index[job["driver_frame_index"]], source, geometry_cache
                )
                started = time.monotonic()
                emit(render_journal, {"event": "render_begin", "rendered": False})
                generated = renderer.render(crop, driver)
                generated_contract(generated)
                native = cv2.warpAffine(
                    generated,
                    cv2.invertAffineTransform(matrix),
                    (rgb.shape[1], rgb.shape[0]),
                    flags=cv2.INTER_LINEAR,
                )
                edited, mask = composite(rgb, native, alpha, opacity(job["frame_index"]))
                changed = int(np.count_nonzero(mask))
                files = {}
                for kind, image in (("crop", generated), ("native", edited), ("mask", mask)):
                    files[kind] = save_png(
                        folder / f"{job['condition']}-{job['frame_index']:04d}-{kind}.png", image
                    )
                record = {
                    **job,
                    "rendered": True,
                    "event": "render_complete",
                    "source": source_proof,
                    "driver": driver_proof,
                    "opacity": opacity(job["frame_index"]),
                    "changed_native_pixels": changed,
                    "outside_alpha_changed_pixels": 0,
                    "files": files,
                    "render_seconds": time.monotonic() - started,
                }
                if job["condition"] == "source_time":
                    saved_source_outputs[job["frame_index"]] = files["crop"]
                elif job["frame_index"] in saved_source_outputs:
                    from PIL import Image

                    other = saved_source_outputs[job["frame_index"]]
                    require(
                        sha(other["path"]) == other["sha256"], "Completed source-time output cache changed"
                    )
                    with Image.open(other["path"]) as image:
                        earlier_source = np.asarray(image.convert("RGB"))
                    response = int(np.any(generated != earlier_source, axis=2).sum())
                    record["earlier_vs_source_crop_changed_pixels"] = response
                    paired += 1
                    unchanged_driver += response == 0
                emit(render_journal, record)
                output_references.append(record)
                executed += 1
                zero_effect += changed == 0
                if job["condition"] == "source_time" and job["frame_index"] == 27:
                    active["repeat"] = True
                    emit(render_journal, {"event": "fixed_repeat_begin"})
                    repeated = renderer.render(crop.copy(), driver.copy())
                    generated_contract(repeated)
                    repeat_renders += 1
                    same = bool(np.array_equal(generated, repeated))
                    repeated_native = cv2.warpAffine(
                        repeated,
                        cv2.invertAffineTransform(matrix),
                        (rgb.shape[1], rgb.shape[0]),
                        flags=cv2.INTER_LINEAR,
                    )
                    repeated_edit, repeated_mask = composite(rgb, repeated_native, alpha, opacity(27))
                    same = (
                        same and np.array_equal(edited, repeated_edit) and np.array_equal(mask, repeated_mask)
                    )
                    repeat_files = {
                        kind: save_png(folder / f"source_time-repeat-0027-{kind}.png", image)
                        for kind, image in (
                            ("crop", repeated),
                            ("native", repeated_edit),
                            ("mask", repeated_mask),
                        )
                    }
                    repeat_status = "bitwise_equal" if same else "failed"
                    repeat_reference = {"frame_index": 27, "status": repeat_status, "files": repeat_files}
                    emit(
                        render_journal,
                        {"event": "fixed_repeat_complete", "repeat_bitwise": same, "files": repeat_files},
                    )
                    write(
                        output / "repeat-check.json",
                        {"status": repeat_status, "source_frame_index": 27, "repeat_bitwise": same},
                    )
                    require(
                        same, "Available source_time frame27 own repeat failed; preserve stage without retry"
                    )
        finally:
            for handle in handles:
                handle.remove()
            if patches:
                tiling.restore(patches)
    renders = executed + repeat_renders
    repeat_available = any(
        job["condition"] == "source_time" and job["frame_index"] == 27 and job["status"] == "available"
        for job in jobs
    )
    require(
        len(output_references) == 40 and repeat_renders == int(repeat_available),
        "Every planned output and exactly the supported fixed repeat must remain recorded",
    )
    require(
        counts_started == counts_completed == expected_roots(renders),
        "Observed complete root forwards differ from actual render inputs",
    )
    require(
        executed == available and len(state_loads) == (5 if available else 0),
        "Every and only supported pair must execute",
    )
    require(
        not renders
        or bounded["spatial_convolutions"] > 0
        and bounded["maximum_estimated_tile_columns_bytes"] <= tiling.WORKSPACE,
        "Qualified bounded spatial convolution execution required",
    )
    require(
        not any(
            k == "mediapipe"
            or k.startswith("mediapipe.")
            or k == "insightface"
            or k.startswith("insightface.")
            for k in sys.modules
        ),
        "No face task/dependency may be imported during frame generation",
    )
    if repeat_renders == 0:
        write(
            output / "repeat-check.json",
            {"status": repeat_status, "source_frame_index": 27, "repeat_bitwise": None},
        )
    write(
        folder / "outputs.json",
        {
            "schema": "chipmunk.grid-portrait-frames-private/146",
            "clip_ordinal": ordinal,
            "planned": source,
            "frames": output_references,
            "repeat": repeat_reference,
            "state_loads": state_loads,
        },
    )
    table = source["input"].get("native_video_table")
    interval = (
        None
        if table is None or len(table["pts"]) <= 47
        else [str(exact(table["pts"][i]) * exact(table["time_base"])) for i in (27, 47)]
    )
    result = {
        "schema": "chipmunk.grid-portrait-frames-result/146",
        "status": "complete",
        "clip_ordinal": ordinal,
        "planned_condition_frames": 40,
        "available_condition_frames": available,
        "executed_condition_frames": executed,
        "skipped_condition_frames": 40 - available,
        "planned_repeat_frames": 1,
        "executed_repeat_frames": repeat_renders,
        "repeat_status": repeat_status,
        "renderer_repeat_bitwise": True if repeat_status == "bitwise_equal" else None,
        "renderer_inputs": renders,
        "strict_exact_checkpoint_loads": len(state_loads),
        "top_level_module_forwards_started": counts_started,
        "top_level_module_forwards_completed": counts_completed,
        "new_neural_calls": sum(counts_started.values()),
        "new_face_task_inputs": 0,
        "new_speech_encoder_inputs": 0,
        "zero_effect_native_outputs": zero_effect,
        "source_earlier_response_pairs": paired,
        "source_earlier_zero_pixel_responses": unchanged_driver,
        "bounded_convolutions": bounded,
        "source_frames_half_open": [27, 47],
        "actual_native_interval_seconds": interval,
        "geometry_global_qualified": geometry_records["geometry_qualified"],
        "device": "cpu",
        "precision": "float32",
        "seed": 146 + ordinal,
        "private_input_binding_sha256": sha(output / "private-input-binding.json"),
        "private_render_journal_sha256": sha(output / "render-records.jsonl"),
        "private_root_progress_sha256": sha(output / "root-progress.jsonl"),
        "private_operator_progress_sha256": sha(output / "operator-progress.jsonl"),
        "private_repeat_check_sha256": sha(output / "repeat-check.json"),
        "private_output_references_sha256": sha(folder / "outputs.json"),
        "freeze_sha256": sha(PUBLIC),
        "achieved_motion_assessed": False,
        "achieved_phonetic_alignment_qualified": False,
        "defaults_changed": False,
        "scope": "Private fixed-pilot native mouth construction, completed-output caching only. Zero pixel effects and unavailable pairs retained. Pixel changes are not achieved motion, phonetic alignment, face swapping or detector accuracy. No serving change.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def routed_popen(argv, ordinal, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"] and len(argv) == 6,
        "Only one fixed authored worker invocation can be routed",
    )
    routed = [*argv[:3], "scripts.grid_portrait_frames_v146", *argv[4:], str(ordinal)]
    return subprocess.Popen(routed, **kwargs)


def run(ordinal):
    from scripts import video_appearance_runtime_v108 as runtime
    from scripts.video_appearance_corrected_v112 import scoped

    pilot_source(ordinal)
    _, geometry = dependencies()
    stage = TRIAL / f"clip-{ordinal:02d}"
    stage.mkdir(exist_ok=True)
    proxy = types.SimpleNamespace(
        Popen=lambda argv, **kwargs: routed_popen(argv, ordinal, **kwargs),
        STDOUT=subprocess.STDOUT,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    scoped(
        runtime.run,
        {
            "TRIAL": stage,
            "ARCHIVE": ARCHIVE,
            "verify": verify,
            "prerequisites": lambda c, b: geometry.completed(ordinal),
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
        require(not args.arguments, "Freeze has no arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One explicit original pilot ordinal required")
        run(int(args.arguments[0]))
    else:
        require(len(args.arguments) == 2, "Exact guarded output path and pilot ordinal required")
        output, ordinal = Path(args.arguments[0]), int(args.arguments[1])
        verify()
        require(
            output.resolve() == (TRIAL / f"clip-{ordinal:02d}/authored/output").resolve(),
            "Exact exclusive per-clip authored output required",
        )
        pilot_source(ordinal)
        output.mkdir(exist_ok=False)
        build(ordinal, output)
