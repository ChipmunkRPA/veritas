"""CPU-only second-generator native-coordinate response qualification."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
import types
from fractions import Fraction
from pathlib import Path

from scripts import fetch_mouth_benchmark_assets as assets
from scripts import native_mouth_inputs_v116 as inputs
from scripts import native_mouth_renderer_v117 as prior
from scripts import portrait_geometry_runtime_v132 as geometry
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = geometry.ROOT, geometry.BASE, geometry.PANEL
TRIAL = BASE / "portrait-native-qualifier-v133"
CACHE = geometry.acquisition.runtime.ARCHIVE / "portrait-native-qualifier-v133"
ASSETS = BASE / "lipsync-v4"
PUBLIC = PANEL / "portrait-native-qualifier-freeze-v133.json"
CODE = (
    "backend/scripts/portrait_native_qualifier_v133.py",
    "backend/tests/test_portrait_native_qualifier_v133.py",
    "docs/portrait-native-qualifier-v133-method.md",
    "backend/scripts/liveportrait_mouth_benchmark.py",
    "backend/scripts/prepare_mouth_edit_v4.py",
    "backend/scripts/fetch_mouth_benchmark_assets.py",
)
read, write, sha, require = geometry.read, geometry.write, geometry.sha, geometry.require


def selection():
    chosen = prior.selection()
    rows = read(inputs.TRIAL / "plan.json")
    ordinal = rows.index(chosen["source"])
    batch, offset = divmod(ordinal, 6)
    bound = read(inputs.TRIAL / f"bind-{batch:02d}/output/bindings.json")[offset]
    current_time = Fraction(chosen["aligned_plan"]["source_time_seconds"])
    wanted = current_time - Fraction(8, 25)
    by_index = {r["video_frame_index"]: r for r in chosen["source"]["clock_frames"]}
    earlier = min(
        bound["pictures"],
        key=lambda p: (
            abs(Fraction(by_index[p["frame_index"]]["source_time_seconds"]) - wanted),
            p["frame_index"],
        ),
    )
    actual = Fraction(by_index[earlier["frame_index"]]["source_time_seconds"])
    require(
        abs(actual - wanted) <= Fraction(1, 25) and actual < current_time,
        "Bound native picture within 40ms of fixed earlier target required",
    )
    return {
        "current": chosen["picture"],
        "earlier": earlier,
        "requested_offset_seconds": "-8/25",
        "actual_offset_seconds": str(actual - current_time),
        "source_binding_sha256": chosen["binding_result_sha256"],
    }


def runtime_verified():
    geometry.verify()
    folder = geometry.TRIAL / "bind-00"
    result, profile = read(folder / "output/result.json"), read(folder / "profile.json")
    require(
        result["status"] == "complete"
        and result["new_neural_calls"] == 0
        and result["application_packages_unchanged"]
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["worker_receipt_sha256"] == sha(folder / "output/result.json"),
        "Actual successful isolated runtime stage required",
    )
    wheels = [geometry.TRIAL / "wheels" / row["filename"] for row in geometry.plan()]
    for wheel, row in zip(wheels, geometry.plan(), strict=True):
        require(sha(wheel) == row["sha256"], "Retained original wheel bytes required")
    require(
        geometry.installed_manifest(wheels, geometry.SITE) == read(folder / "output/installed-files.json"),
        "Unchanged isolated installed source/native files required",
    )


def identity():
    inputs.verify()
    runtime_verified()
    assets.fetch(ASSETS / "permissions/liveportrait-verified-assets.json", ASSETS, verify_only=True)
    require(
        read(TRIAL / "selection.json") == selection(), "Unchanged first-fitting-source selection required"
    )
    return {
        "schema": "chipmunk.portrait-native-qualifier-freeze/133",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "private_selection_sha256": sha(TRIAL / "selection.json"),
        "source_bank_freeze_sha256": sha(inputs.PUBLIC),
        "geometry_runtime_freeze_sha256": sha(geometry.PUBLIC),
        "geometry_runtime_result_sha256": sha(geometry.TRIAL / "bind-00/output/result.json"),
        "asset_catalogue_sha256": assets.CATALOGUE_SHA,
        "face_task_sha256": assets.MESH_SHA,
        "source_license_sha256": assets.LICENCE_SHA,
        "packages": {
            p: importlib.metadata.version(p) for p in ("torch", "numpy", "opencv-python-headless", "PyYAML")
        },
        "device": "cpu",
        "precision": "float32",
        "torch_threads": 1,
        "interop_threads": 1,
        "deterministic_algorithms": True,
        "seed": 133,
        "geometry": {
            "running_mode": "IMAGE",
            "delegate": "CPU",
            "num_faces": 2,
            "min_face_detection_confidence": 0.5,
            "min_face_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "output_face_blendshapes": False,
            "output_facial_transformation_matrixes": False,
        },
        "calls": {"face_task_inputs": 3, "renderer_inputs": 4, "expected_top_module_forwards": 20},
        "conditions": ["same_aperture", "repeat", "earlier_native_aperture", "authored_aperture_0.5"],
        "native_mapping": "256 to 512 pixel-centre affine: multiply matrix by 2, then add 0.5 to both translations. Invert that full matrix for warp to the original native image. Composite only fixed source-mouth feathered support.",
        "scope": "One consumed fitting source; benchmark-generator response, exact strict state loading, stateless geometry repeat and native-coordinate containment only. No new-source detector accuracy or achieved audio/phonetic synchronization claim.",
    }


def freeze():
    runtime_verified()
    chosen = selection()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "selection.json", chosen)
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print('{"status":"frozen","face_task_inputs":3,"renderer_inputs":4}', flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Published unchanged CPU qualifier required",
    )


def upscale_matrix(matrix):
    import numpy as np

    value = np.asarray(matrix, dtype=np.float64)
    require(value.shape == (2, 3) and np.isfinite(value).all(), "Finite source-to-256 affine required")
    result = value * 2
    result[:, 2] += 0.5
    require(abs(np.linalg.det(result[:, :2])) > 1e-12, "Invertible native affine required")
    return result


def response_summary(images):
    import numpy as np

    require(
        len(images) == 4 and all(im.shape == (512, 512, 3) and im.dtype == np.uint8 for im in images),
        "Four finite byte output images required",
    )
    require(np.array_equal(images[0], images[1]), "Repeated CPU render differs")
    changes = {
        name: int(np.any(images[0] != other, axis=2).sum())
        for name, other in zip(("earlier_native_aperture", "authored_aperture_0.5"), images[2:], strict=True)
    }
    require(changes["authored_aperture_0.5"] > 0, "Authored aperture has no pixel effect")
    return changes


def qualify(output):
    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, str(geometry.SITE))
    import mediapipe as mp

    from scripts.liveportrait_mouth_benchmark import crop_from_mesh, load
    from scripts.prepare_mouth_edit_v4 import composite, mouth_alpha

    require(
        Path(mp.__file__).resolve().is_relative_to(geometry.SITE.resolve()),
        "Qualified isolated face API required",
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(133)
    cv2.setNumThreads(1)
    CACHE.mkdir(exist_ok=False)
    chosen = read(TRIAL / "selection.json")
    pictures = []
    for name in ("current", "earlier"):
        record = chosen[name]
        require(sha(Path(record["file"])) == record["file_sha256"], "Bound source PNG differs")
        bgr = cv2.imread(record["file"])
        require(bgr is not None, "Native source PNG required")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        require(inputs.speech.speech.fingerprint(rgb) == record["rgb_sha256"], "Decoded source pixels differ")
        pictures.append(rgb)
    meshes = []
    mesh_options = dict(read(PUBLIC)["geometry"])
    mesh_options.pop("delegate")
    mesh_options["running_mode"] = mp.tasks.vision.RunningMode.IMAGE
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(ASSETS / "permissions/face_landmarker.task"),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        **mesh_options,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as task:
        for rgb in (pictures[0], pictures[1], pictures[0]):
            result = task.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            require(len(result.face_landmarks) == 1, "One unambiguous face required; no source replacement")
            mesh = result.face_landmarks[0]
            require(
                len(mesh) == 478 and np.isfinite([(p.x, p.y, p.z) for p in mesh]).all(),
                "Complete finite 478-point mesh required",
            )
            meshes.append(mesh)
    mesh_values = [np.array([(p.x, p.y, p.z) for p in mesh], dtype=np.float64) for mesh in meshes]
    require(np.array_equal(mesh_values[0], mesh_values[2]), "IMAGE-mode geometry repeat differs")
    crop, matrix, source_ratio = crop_from_mesh(pictures[0], meshes[0])
    _, _, earlier_ratio = crop_from_mesh(pictures[1], meshes[1])
    require(source_ratio != 0.5, "Fixed authored counterfactual must differ from measured source")
    alpha = mouth_alpha(pictures[0].shape, meshes[0])
    inverse = cv2.invertAffineTransform(upscale_matrix(matrix))
    original_loader = torch.nn.Module.load_state_dict
    state_loads = []

    def exact_load(module, state, strict=True, assign=False):
        require(strict is True and assign is False, "Strict exact checkpoint load required")
        outcome = original_loader(module, state, strict=strict, assign=assign)
        actual = module.state_dict()
        require(
            actual.keys() == state.keys()
            and all(torch.equal(actual[k].cpu(), v.cpu()) for k, v in state.items()),
            "Loaded checkpoint tensor differs",
        )
        state_loads.append({"tensors": len(state), "strict": True, "exact": True})
        return outcome

    try:
        torch.nn.Module.load_state_dict = exact_load
        renderer = load(ASSETS, "cpu")
    finally:
        torch.nn.Module.load_state_dict = original_loader
    require(
        len(state_loads) == 5
        and all(
            p.device.type == "cpu" and p.dtype == torch.float32
            for model in renderer.models.values()
            for p in model.parameters()
        ),
        "Five exact CPU float32 models required",
    )
    counts = dict.fromkeys(renderer.models, 0)

    def counter(name):
        def hook(module, args):
            counts[name] += 1

        return hook

    hooks = [model.register_forward_pre_hook(counter(name)) for name, model in renderer.models.items()]
    images, records = [], []
    try:
        for name, ratio in zip(
            read(PUBLIC)["conditions"], (source_ratio, source_ratio, earlier_ratio, 0.5), strict=True
        ):
            started = time.monotonic()
            generated = renderer.render(crop, source_ratio, ratio)
            native = cv2.warpAffine(
                generated, inverse, (pictures[0].shape[1], pictures[0].shape[0]), flags=cv2.INTER_LINEAR
            )
            edited = composite(pictures[0], native, alpha)
            changed = np.any(edited != pictures[0], axis=2)
            require(
                changed.any() and not changed[alpha == 0].any(),
                "Empty edit or changed pixels outside source support",
            )
            files = {}
            for suffix, value in (
                ("crop", generated),
                ("native", edited),
                ("mask", changed.astype(np.uint8) * 255),
            ):
                path = CACHE / f"{name}-{suffix}.png"
                require(
                    cv2.imwrite(
                        str(path), cv2.cvtColor(value, cv2.COLOR_RGB2BGR) if value.ndim == 3 else value
                    ),
                    "Could not preserve private render support",
                )
                files[suffix] = {"path": str(path), "sha256": sha(path)}
            records.append(
                {
                    "condition": name,
                    "target_ratio_hex": float(ratio).hex(),
                    "changed_native_pixels": int(changed.sum()),
                    "render_seconds": time.monotonic() - started,
                    "files": files,
                }
            )
            images.append(generated)
    finally:
        for hook in hooks:
            hook.remove()
    changes = response_summary(images)
    require(
        all(value == 4 for value in counts.values()) and sum(counts.values()) == 20,
        "Exactly twenty top-level generator module forwards required",
    )
    require("insightface" not in sys.modules, "Restricted face dependency must not be imported")
    write(
        output / "private-render-records.json",
        {
            "records": records,
            "state_loads": state_loads,
            "source_to_256_matrix_hex": [[float(x).hex() for x in row] for row in matrix],
            "source_to_512_matrix_hex": [[float(x).hex() for x in row] for row in upscale_matrix(matrix)],
            "source_ratio_hex": float(source_ratio).hex(),
            "mesh_xyz_hex": [[[float(x).hex() for x in point] for point in mesh] for mesh in mesh_values],
        },
    )
    result = {
        "schema": "chipmunk.portrait-native-qualifier-result/133",
        "status": "complete",
        "device": "cpu",
        "strict_exact_checkpoint_loads": len(state_loads),
        "stateless_geometry_repeat_bitwise": True,
        "face_task_inputs": 3,
        "renderer_inputs": 4,
        "renderer_repeat_bitwise": True,
        "top_level_module_forwards": counts,
        "new_neural_calls": 23,
        "neural_counter_scope": "3 face-task API inputs plus 20 hooked top-level generator module forwards; internal face-task graph invocations are not independently instrumented.",
        "counterfactual_changed_crop_pixels": changes,
        "changed_native_pixels": {r["condition"]: r["changed_native_pixels"] for r in records},
        "render_seconds": {r["condition"]: r["render_seconds"] for r in records},
        "private_render_records_sha256": sha(output / "private-render-records.json"),
        "freeze_sha256": sha(PUBLIC),
        "native_coordinate_containment_qualified": True,
        "achieved_audio_synchronization_qualified": False,
        "defaults_changed": False,
        "scope": "One previously consumed fitting source, not new-source transfer or detector performance. CPU generator aperture response and native support containment only; earlier aperture does not establish phonetic retiming.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"],
        "Only fixed neural qualification required",
    )
    routed = list(argv)
    routed[3] = "scripts.portrait_native_qualifier_v133"
    return subprocess.Popen(routed, **kwargs)


def run():
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        geometry.acquisition.runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda c, b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("authored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "authored"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    if args.command in {"freeze", "run"}:
        require(args.output is None, "No extra arguments required")
        (freeze if args.command == "freeze" else run)()
    else:
        verify()
        require(
            args.output is not None and args.output.resolve() == (TRIAL / "authored/output").resolve(),
            "Exact exclusive guarded output required",
        )
        args.output.mkdir(exist_ok=False)
        qualify(args.output)
