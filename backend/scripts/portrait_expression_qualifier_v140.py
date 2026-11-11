"""Full-expression CPU response/native-support qualification, using bound meshes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

from scripts import portrait_tiled_runtime_v139 as baseline
from scripts import spatial_conv_install_v141 as tiling
from scripts.video_appearance_corrected_v112 import scoped

original = baseline.original
ROOT, BASE, PANEL = baseline.ROOT, baseline.BASE, baseline.PANEL
TRIAL = BASE / "portrait-expression-qualifier-v140"
CACHE = baseline.CACHE.parent / "portrait-expression-qualifier-v140"
PUBLIC = PANEL / "portrait-expression-qualifier-freeze-v140.json"
BASELINE_EXECUTION = PANEL / "portrait-tiled-runtime-result-v139.json"
CODE = baseline.CODE + (
    "backend/scripts/spatial_conv_install_v141.py",
    "backend/tests/test_spatial_conv_install_v141.py",
    "docs/spatial-conv-install-v141-method.md",
    "docs/evaluation-panels/portrait-tiled-runtime-result-v139.json",
    "backend/scripts/portrait_expression_drive_v138.py",
    "backend/tests/test_portrait_expression_drive_v138.py",
    "docs/portrait-expression-drive-v138-design.md",
    "backend/scripts/portrait_expression_qualifier_v140.py",
    "backend/tests/test_portrait_expression_qualifier_v140.py",
    "docs/portrait-expression-qualifier-v140-method.md",
)
EXPECTED = {
    "appearance_feature_extractor": 3,
    "motion_extractor": 6,
    "warping_module": 3,
    "spade_generator": 3,
    "lip": 0,
}
CONDITIONS = ("source_expression", "current_repeat", "earlier_expression")
read, write, sha, require = baseline.read, baseline.write, baseline.sha, baseline.require


def completion_binding(result, profile, *, result_sha, freeze_sha, records_sha, progress_sha):
    """Require a completed actual v139 response, not a readiness/file-exists claim."""
    require(
        result["status"] == "complete"
        and result["execution_context_schema"] == "chipmunk.portrait-tiled-cpu-result/139"
        and result["freeze_sha256"] == freeze_sha
        and result["strict_exact_checkpoint_loads"] == 5
        and result["stateless_geometry_repeat_bitwise"] is True
        and result["renderer_repeat_bitwise"] is True
        and result["native_coordinate_containment_qualified"] is True
        and result["face_task_inputs"] == 3
        and result["renderer_inputs"] == 4
        and result["top_level_module_forwards"] == {name: 4 for name in EXPECTED}
        and result["new_neural_calls"] == 23
        and result["private_render_records_sha256"] == records_sha
        and result["private_forward_progress_sha256"] == progress_sha
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["new_neural_calls"] == 23
        and profile["worker_receipt_sha256"] == result_sha,
        "Actual successful unchanged v139 CPU/geometry/counter bindings required",
    )


def cached_geometry(record):
    """Decode only finite canonical hexadecimal values; never run a face task."""

    def array(value, shape):
        raw = np.asarray(value, dtype=object)
        require(
            raw.shape == shape and all(type(v) is str for v in raw.flat),
            "Exact cached geometry shape/hex strings required",
        )
        decoded = np.asarray([float.fromhex(v) for v in raw.flat], np.float64).reshape(shape)
        require(
            np.isfinite(decoded).all()
            and all(float(v).hex() == s for v, s in zip(decoded.flat, raw.flat, strict=True)),
            "Finite canonical cached geometry required",
        )
        return decoded

    meshes = array(record["mesh_xyz_hex"], (3, 478, 3))
    matrix = array(record["source_to_256_matrix_hex"], (2, 3))
    output_matrix = array(record["source_to_512_matrix_hex"], (2, 3))
    require(np.array_equal(meshes[0], meshes[2]), "Qualified cached current mesh repeat must remain exact")
    require(
        np.array_equal(output_matrix, original.upscale_matrix(matrix)),
        "Cached 512 pixel-centre matrix differs",
    )
    require(type(record["source_ratio_hex"]) is str, "Cached source aperture hex string required")
    ratio = float.fromhex(record["source_ratio_hex"])
    require(np.isfinite(ratio) and 0 <= ratio <= 1, "Finite cached source aperture required")
    return meshes, matrix, ratio


def baseline_verified():
    baseline.context()["verify"]()
    folder = baseline.TRIAL / "authored"
    result, profile = read(folder / "output/result.json"), read(folder / "profile.json")
    private = folder / "output/private-render-records.json"
    completion_binding(
        result,
        profile,
        result_sha=sha(folder / "output/result.json"),
        freeze_sha=sha(baseline.PUBLIC),
        records_sha=sha(private),
        progress_sha=sha(folder / "output/forward-progress.jsonl"),
    )
    public = read(BASELINE_EXECUTION)
    require(
        public["status"] == "complete"
        and public["measurement"] == result
        and public["profile"] == profile
        and public["result_sha256"] == sha(folder / "output/result.json")
        and public["profile_sha256"] == sha(folder / "profile.json"),
        "Actual baseline artifacts must match their published execution receipt",
    )
    record = read(private)
    cached_geometry(record)
    selected = read(baseline.TRIAL / "selection.json")
    require(selected == original.selection(), "Same original source/picture/driver clock selection required")
    for name in ("current", "earlier"):
        picture = selected[name]
        require(sha(Path(picture["file"])) == picture["file_sha256"], "Qualified source picture bytes differ")
    require(
        [r["condition"] for r in record["records"]] == read(baseline.PUBLIC)["conditions"],
        "Complete original scalar response correspondence required",
    )
    for row in record["records"]:
        for file in row["files"].values():
            path = Path(file["path"])
            require(
                path.resolve().is_relative_to(baseline.CACHE.resolve()) and sha(path) == file["sha256"],
                "Exact owned scalar response files required",
            )
    return selected, record


def response_metrics(images):
    require(
        len(images) == 3
        and all(image.shape == (512, 512, 3) and image.dtype == np.uint8 for image in images),
        "Three full-resolution uint8 expression outputs required",
    )
    return {
        "renderer_repeat_bitwise": bool(np.array_equal(images[0], images[1])),
        "earlier_expression_changed_crop_pixels": int(np.any(images[0] != images[2], axis=2).sum()),
    }


def require_response(metrics):
    require(metrics["renderer_repeat_bitwise"] is True, "Expression renderer own repeat differs")
    require(
        metrics["earlier_expression_changed_crop_pixels"] > 0,
        "Earlier expression produced no observed pixel response; do not substitute another source",
    )


def support_metrics(source, edited, alpha):
    require(
        source.dtype == edited.dtype == np.uint8
        and source.shape == edited.shape
        and source.ndim == 3
        and source.shape[2] == 3
        and alpha.shape == source.shape[:2]
        and np.isfinite(alpha).all()
        and ((alpha >= 0) & (alpha <= 1)).all(),
        "Corresponding source/edit/support arrays required",
    )
    changed = np.any(source != edited, axis=2)
    return {
        "changed_native_pixels": int(changed.sum()),
        "outside_support_changed_pixels": int(changed[alpha == 0].sum()),
    }


def identity():
    selected, _ = baseline_verified()
    require(read(TRIAL / "selection.json") == selected, "Unchanged source/current/earlier binding required")
    return {
        "schema": "chipmunk.portrait-expression-qualifier-freeze/140",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "baseline_freeze_sha256": sha(baseline.PUBLIC),
        "baseline_public_execution_sha256": sha(BASELINE_EXECUTION),
        "baseline_result_sha256": sha(baseline.TRIAL / "authored/output/result.json"),
        "baseline_profile_sha256": sha(baseline.TRIAL / "authored/profile.json"),
        "private_cached_geometry_records_sha256": sha(
            baseline.TRIAL / "authored/output/private-render-records.json"
        ),
        "private_selection_sha256": sha(TRIAL / "selection.json"),
        "asset_catalogue_sha256": original.assets.CATALOGUE_SHA,
        "source_license_sha256": original.assets.LICENCE_SHA,
        "packages": {
            p: importlib.metadata.version(p) for p in ("torch", "numpy", "opencv-python-headless", "PyYAML")
        },
        "device": "cpu",
        "precision": "float32",
        "seed": 140,
        "torch_threads": 1,
        "interop_threads": 1,
        "opencv_threads": 1,
        "deterministic_algorithms": True,
        "convolution_workspace_estimate_budget_bytes": tiling.WORKSPACE,
        "installer_policy": "v141 transactional whole-graph validation and identical forward-time inference guards on fast/tiled paths; unchanged v139 spatial arithmetic.",
        "conditions": list(CONDITIONS),
        "expected_top_module_forwards": EXPECTED,
        "new_face_task_inputs": 0,
        "expected_top_module_forward_units": 15,
        "driver_requested_offset_seconds": selected["requested_offset_seconds"],
        "driver_actual_offset_seconds": selected["actual_offset_seconds"],
        "native_mapping": "Source-to-512 matrix is twice source-to-256 plus0.5 in both translations; inverse-map full512 output and composite only cached source-mouth feathered support.",
        "scope": "One consumed fitting source; full-expression renderer response/repeat and native-support containment only. Reused qualified cached meshes, no new face task. No achieved motion, phonetic timing, unseen-source detection or customer-confidence claim.",
    }


def freeze():
    selected, _ = baseline_verified()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "selection.json", selected)
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print('{"status":"frozen","new_face_task_inputs":0,"expected_top_module_forwards":15}', flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged expression response contract required",
    )


def qualify(output):
    import cv2
    import torch

    from scripts.liveportrait_mouth_benchmark import crop_from_mesh
    from scripts.portrait_expression_drive_v138 import ExpressionRenderer
    from scripts.prepare_mouth_edit_v4 import composite, mouth_alpha

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(140)
    cv2.setNumThreads(1)
    selected, cached = baseline_verified()
    meshes, cached_matrix, cached_ratio = cached_geometry(cached)
    pictures = []
    for name in ("current", "earlier"):
        row = selected[name]
        image = cv2.imread(row["file"])
        require(image is not None, "Complete bound source PNG required")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        require(
            original.inputs.speech.speech.fingerprint(rgb) == row["rgb_sha256"],
            "Decoded source picture pixels differ",
        )
        pictures.append(rgb)
    points = [
        [types.SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in mesh] for mesh in meshes
    ]
    source_crop, matrix, ratio = crop_from_mesh(pictures[0], points[0])
    earlier_crop, earlier_matrix, _ = crop_from_mesh(pictures[1], points[1])
    require(
        np.array_equal(matrix, cached_matrix) and float(ratio).hex() == float(cached_ratio).hex(),
        "Reconstructed source crop geometry differs from qualified baseline",
    )
    alpha = mouth_alpha(pictures[0].shape, points[0])
    inverse = cv2.invertAffineTransform(original.upscale_matrix(matrix))
    CACHE.mkdir(exist_ok=False)

    def save_png(path, value):
        require(
            cv2.imwrite(str(path), cv2.cvtColor(value, cv2.COLOR_RGB2BGR) if value.ndim == 3 else value),
            "Could not preserve private expression support PNG",
        )
        recovered = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        require(recovered is not None, "Complete saved expression support PNG required")
        if recovered.ndim == 3:
            recovered = cv2.cvtColor(recovered, cv2.COLOR_BGR2RGB)
        require(np.array_equal(recovered, value), "Saved expression support PNG differs from its pixel array")
        return {"path": str(path), "sha256": sha(path)}

    input_files = {
        "current_crop": save_png(CACHE / "current-input-crop.png", source_crop),
        "earlier_crop": save_png(CACHE / "earlier-input-crop.png", earlier_crop),
    }
    alpha_file = CACHE / "source-mouth-alpha.npy"
    with alpha_file.open("xb") as stream:
        np.save(stream, alpha, allow_pickle=False)
    input_files["source_mouth_alpha"] = {"path": str(alpha_file), "sha256": sha(alpha_file)}
    write(output / "private-input-files.json", input_files)
    counts_started, counts_completed = dict.fromkeys(EXPECTED, 0), dict.fromkeys(EXPECTED, 0)
    bounded = {
        "spatial_convolutions": 0,
        "functional_tile_calls": 0,
        "maximum_estimated_tile_columns_bytes": 0,
    }
    patches, handles, state_loads, images, records = [], [], [], [], []
    with (output / "forward-progress.jsonl").open("x") as journal:

        def observe(row):
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            if row["event"] == "spatial_conv_begin":
                bounded["spatial_convolutions"] += 1
                bounded["maximum_estimated_tile_columns_bytes"] = max(
                    bounded["maximum_estimated_tile_columns_bytes"], row["estimated_tile_columns_bytes"]
                )
            elif row["event"] == "spatial_conv_end":
                bounded["functional_tile_calls"] += row["tiles"]

        def hook(name, finished):
            def callback(module, args, *values):
                (counts_completed if finished else counts_started)[name] += 1
                observe({"event": "top_module_end" if finished else "top_module_begin", "module": name})

            return callback

        actual_loader = torch.nn.Module.load_state_dict

        def exact_load(module, state, strict=True, assign=False):
            require(strict is True and assign is False, "Strict exact checkpoint loading required")
            result = actual_loader(module, state, strict=strict, assign=assign)
            loaded = module.state_dict()
            require(
                loaded.keys() == state.keys()
                and all(torch.equal(loaded[k].cpu(), value.cpu()) for k, value in state.items()),
                "Loaded checkpoint tensor differs",
            )
            state_loads.append({"strict": True, "exact": True, "tensors": len(state)})
            observe({"event": "strict_state_load_complete", "tensor_count": len(state)})
            return result

        observe({"event": "loader_begin_with_reused_geometry", "new_face_task_inputs": 0})
        try:
            torch.nn.Module.load_state_dict = exact_load
            renderer = ExpressionRenderer(original.ASSETS)
        finally:
            torch.nn.Module.load_state_dict = actual_loader
        require(len(state_loads) == 5, "Five exact checkpoint state loads required")
        require(set(renderer.renderer.models) == set(EXPECTED), "Exact five source-model roots required")
        try:
            patches = tiling.install(renderer.renderer.models, observe)
            for name, model in renderer.renderer.models.items():
                handles.extend(
                    [
                        model.register_forward_pre_hook(hook(name, False)),
                        model.register_forward_hook(hook(name, True)),
                    ]
                )
            with (output / "render-records.jsonl").open("x") as render_journal:
                for name, driver in zip(
                    CONDITIONS, (source_crop, source_crop.copy(), earlier_crop), strict=True
                ):
                    started = time.monotonic()
                    generated = renderer.render(source_crop, driver)
                    require(
                        generated.shape == (512, 512, 3) and generated.dtype == np.uint8,
                        "Full finite uint8 expression output required",
                    )
                    native = cv2.warpAffine(
                        generated,
                        inverse,
                        (pictures[0].shape[1], pictures[0].shape[0]),
                        flags=cv2.INTER_LINEAR,
                    )
                    edited = composite(pictures[0], native, alpha)
                    support = support_metrics(pictures[0], edited, alpha)
                    files = {}
                    for suffix, value in (
                        ("crop", generated),
                        ("native", edited),
                        ("mask", np.any(edited != pictures[0], axis=2).astype(np.uint8) * 255),
                    ):
                        path = CACHE / f"{name}-{suffix}.png"
                        files[suffix] = save_png(path, value)
                    record = {
                        "condition": name,
                        "support": support,
                        "files": files,
                        "render_seconds": time.monotonic() - started,
                    }
                    records.append(record)
                    images.append(generated)
                    render_journal.write(json.dumps(record, allow_nan=False) + "\n")
                    render_journal.flush()
                    observe({"event": "render_support_observed", "condition": name, **support})
                    require(
                        support["changed_native_pixels"] > 0
                        and support["outside_support_changed_pixels"] == 0,
                        "Empty native edit or changed pixels outside source-mouth support",
                    )
        finally:
            for handle in handles:
                handle.remove()
            tiling.restore(patches)
        require(
            counts_started == counts_completed == EXPECTED,
            "Exactly fifteen complete expected top-module forwards required",
        )
        require(
            bounded["spatial_convolutions"] > 0
            and bounded["maximum_estimated_tile_columns_bytes"] <= tiling.WORKSPACE,
            "Explicit bounded convolution execution required",
        )
        metrics = response_metrics(images)
        write(output / "response-check.json", metrics)
        require_response(metrics)
        require(
            "insightface" not in sys.modules and "mediapipe" not in sys.modules,
            "No restricted face dependency or new face-task import is permitted",
        )
        scalar_images = [
            cv2.cvtColor(cv2.imread(cached["records"][i]["files"]["crop"]["path"]), cv2.COLOR_BGR2RGB)
            for i in (0, 2)
        ]
        comparison = {
            "scalar_earlier_changed_crop_pixels": int(
                np.any(scalar_images[0] != scalar_images[1], axis=2).sum()
            ),
            "expression_earlier_changed_crop_pixels": metrics["earlier_expression_changed_crop_pixels"],
            "between_path_source_changed_crop_pixels": int(
                np.any(images[0] != scalar_images[0], axis=2).sum()
            ),
            "between_path_earlier_changed_crop_pixels": int(
                np.any(images[2] != scalar_images[1], axis=2).sum()
            ),
            "interpretation": "Pixel response magnitudes only; neither method is ranked for achieved speech/motion alignment or detection accuracy.",
        }
        require(
            comparison["scalar_earlier_changed_crop_pixels"]
            == read(baseline.TRIAL / "authored/output/result.json")["counterfactual_changed_crop_pixels"][
                "earlier_native_aperture"
            ],
            "Cached scalar comparison differs from actual completed baseline",
        )
        write(
            output / "private-render-records.json",
            {
                "records": records,
                "state_loads": state_loads,
                "input_files": input_files,
                "source_to_256_matrix_hex": [[float(x).hex() for x in row] for row in matrix],
                "earlier_to_256_matrix_hex": [[float(x).hex() for x in row] for row in earlier_matrix],
                "source_to_512_matrix_hex": [
                    [float(x).hex() for x in row] for row in original.upscale_matrix(matrix)
                ],
            },
        )
        result = {
            "schema": "chipmunk.portrait-expression-qualifier-result/140",
            "status": "complete",
            "device": "cpu",
            "precision": "float32",
            "strict_exact_checkpoint_loads": len(state_loads),
            "renderer_inputs": 3,
            "new_face_task_inputs": 0,
            "cached_qualified_mesh_records_checked": 3,
            "cached_meshes_used_for_cropping": 2,
            "top_level_module_forwards_started": counts_started,
            "top_level_module_forwards_completed": counts_completed,
            "new_neural_calls": 15,
            "neural_counter_scope": "Fifteen hooked top-level generator forwards; zero new face-task calls. Functional convolution tile calls are separate units.",
            **metrics,
            "bounded_convolutions": bounded,
            "scalar_comparison": comparison,
            "native_coordinate_containment_qualified": True,
            "renderer_response_qualified": True,
            "achieved_motion_assessed": False,
            "achieved_phonetic_alignment_qualified": False,
            "driver_requested_offset_seconds": selected["requested_offset_seconds"],
            "driver_actual_offset_seconds": selected["actual_offset_seconds"],
            "changed_native_pixels": {r["condition"]: r["support"]["changed_native_pixels"] for r in records},
            "render_seconds": {r["condition"]: r["render_seconds"] for r in records},
            "private_render_records_sha256": sha(output / "private-render-records.json"),
            "private_input_files_sha256": sha(output / "private-input-files.json"),
            "private_render_journal_sha256": sha(output / "render-records.jsonl"),
            "private_forward_progress_sha256": sha(output / "forward-progress.jsonl"),
            "freeze_sha256": sha(PUBLIC),
            "defaults_changed": False,
            "scope": "One consumed source and two bound pictures. Full-expression response, repeat and native-mouth containment only. Pixel differences are not measured lip trajectories, phonetic retiming, fresh detector accuracy or customer confidence.",
        }
        write(output / "result.json", result)
        print(json.dumps(result), flush=True)


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"],
        "Only fixed expression qualification permitted",
    )
    args = list(argv)
    args[3] = "scripts.portrait_expression_qualifier_v140"
    return subprocess.Popen(args, **kwargs)


def run():
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        original.geometry.acquisition.runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda command, batch: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("authored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "authored"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    if args.command in ("freeze", "run"):
        require(args.output is None, "No output argument for freeze/run")
        (freeze if args.command == "freeze" else run)()
    else:
        verify()
        require(
            args.output is not None and args.output.resolve() == (TRIAL / "authored/output").resolve(),
            "Exact exclusive guarded expression output required",
        )
        args.output.mkdir(exist_ok=False)
        qualify(args.output)
