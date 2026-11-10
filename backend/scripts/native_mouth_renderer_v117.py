"""Native source/state mouth-render response qualification; no detector score."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import types
from fractions import Fraction
from pathlib import Path

from scripts import av_conditioning_clock_v109 as clock
from scripts import native_mouth_inputs_v116 as inputs
from scripts import video_appearance_runtime_v108 as runtime
from scripts.fetch_mouth_generator_v7 import verify as verify_asset
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = inputs.ROOT, inputs.BASE, inputs.PANEL
TRIAL = BASE / "native-mouth-renderer-v117"
CACHE = inputs.CACHE.parent / "native-mouth-renderer-v117"
PUBLIC = PANEL / "native-mouth-renderer-freeze-v117.json"
ASSETS = inputs.speech.speech.ASSETS
MANIFEST = PANEL / "mouth-v7-generator-assets.json"
GENERATOR_MANIFEST_SHA = "bc823fb8f21e21c8c1cd722d225c927b0102b724b76942457f166ef047caf9fc"
CODE = (
    "backend/scripts/native_mouth_renderer_v117.py",
    "backend/tests/test_native_mouth_renderer_v117.py",
    "docs/native-mouth-renderer-v117-method.md",
    "backend/scripts/musetalk_benchmark_v7.py",
    "backend/scripts/smoke_mouth_generator_v7.py",
    "backend/scripts/prepare_mouth_edit_v4.py",
    "backend/app/detectors/image/faces.py",
)
read, write, sha, require = inputs.read, inputs.write, inputs.sha, inputs.require


def earlier_plan(frame):
    result = clock.frame_plan(
        frame["video_frame_index"],
        str(Fraction(frame["source_time_seconds"]) - Fraction(8, 25)),
        frame["source_audio_origin_seconds"],
        frame["source_video_origin_seconds"],
        frame["observed_mono_samples"],
        frame["native_clock_allowance_seconds"],
    )
    require(
        result["source_sample_status"] == "supported" and all(result["retained_tick_slots"]),
        "Complete fixed 320-ms earlier conditioning required",
    )
    return result


def selection():
    rows = read(inputs.TRIAL / "plan.json")
    ordinal = next(i for i, r in enumerate(rows) if r["study_role"] == "fusion_fit" and r["frame_indices"])
    row = rows[ordinal]
    index = sum(row["edit_frames_half_open"]) // 2
    frame = next(f for f in row["clock_frames"] if f["video_frame_index"] == index)
    require(
        frame["source_sample_status"] == "supported" and all(frame["retained_tick_slots"]),
        "Complete central source conditioning required",
    )
    batch, offset = divmod(ordinal, 6)
    folder = inputs.TRIAL / f"bind-{batch:02d}"
    result, profile = read(folder / "output/result.json"), read(folder / "profile.json")
    require(
        result["status"] == "complete"
        and result["bindings_sha256"] == sha(folder / "output/bindings.json")
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["worker_receipt_sha256"] == sha(folder / "output/result.json"),
        "Qualified exact source pictures required",
    )
    bound = read(folder / "output/bindings.json")[offset]
    require(bound["planned"] == row, "Exact planned fitting input required")
    picture = next(p for p in bound["pictures"] if p["frame_index"] == index)
    return {
        "source": row,
        "picture": picture,
        "aligned_plan": frame,
        "earlier_plan": earlier_plan(frame),
        "binding_result_sha256": sha(folder / "output/result.json"),
    }


def identity():
    inputs.verify()
    complete = read(inputs.TRIAL / "result.json")
    require(
        complete == read(PANEL / "native-mouth-input-result-v116.json")
        and complete["sources"] == 48
        and complete["pictures"] == 1128,
        "Complete published native input bank required",
    )
    manifest = read(MANIFEST)
    require(
        sha(MANIFEST) == GENERATOR_MANIFEST_SHA
        and len(manifest["assets"]) == 7
        and all(r["license"] in {"mit", "apache-2.0"} for r in manifest["assets"]),
        "Only retained permitted assets required",
    )
    actual = [verify_asset(ASSETS / r["path"], r) for r in manifest["assets"]]
    require(
        read(ASSETS / "verified-assets.json")["assets"] == actual, "Exact acquired generator assets required"
    )
    face = read(PANEL / "video-appearance-native-freeze-v108.json")["model_records"]["yunet.2023mar"]
    require(
        face["licence"]["class_"] == "commercial"
        and all(sha(ROOT / p) == h for p, h in face["files"].items()),
        "Cached permitted face geometry required",
    )
    require(
        read(TRIAL / "selection.json") == selection(), "Deterministic unchanged first-fitting input required"
    )
    return {
        "schema": "chipmunk.native-mouth-renderer-freeze/117",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "input_freeze_sha256": sha(inputs.PUBLIC),
        "input_result_sha256": sha(inputs.TRIAL / "result.json"),
        "private_selection_sha256": sha(TRIAL / "selection.json"),
        "generator_manifest_sha256": sha(MANIFEST),
        "generator_assets": actual,
        "face_model_record": face,
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "diffusers",
                "transformers",
                "safetensors",
                "accelerate",
                "numpy",
                "opencv-python-headless",
            )
        },
        "device": "mps",
        "seed": 117,
        "face_min_size": 48,
        "conditions": ["aligned", "repeat", "earlier_320ms", "zero"],
        "planned_render_calls": 4,
        "planned_new_neural_calls": 17,
        "scope": "One deterministic fitting-source native picture/state render response contract only. Changed pixels do not establish achieved mouth synchronization, realistic manipulation or detector performance.",
    }


def freeze():
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "selection.json", selection())
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "planned_new_neural_calls": 17, "new_neural_calls": 0}), flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Published unchanged native renderer contract required",
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"],
        "Only fixed native render qualification can be routed",
    )
    routed = list(argv)
    routed[3] = "scripts.native_mouth_renderer_v117"
    return subprocess.Popen(routed, **kwargs)


def run():
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda command, batch: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("authored")


def response_counts(images):
    import numpy as np

    require(
        len(images) == 4 and all(im.shape == (256, 256, 3) and im.dtype == np.uint8 for im in images),
        "Complete native render images required",
    )
    require(images[0].tobytes() == images[1].tobytes(), "Seeded native render repeat changed")
    counts = {
        name: int(np.any(images[0] != other, axis=2).sum())
        for name, other in zip(("earlier_320ms", "zero"), images[2:], strict=True)
    }
    require(all(v > 0 for v in counts.values()), "Changed conditioning has no pixel effect")
    return counts


def qualify(output):
    import cv2
    import numpy as np
    import torch

    from app.detectors.image.faces import detect_faces, load_yunet
    from scripts.musetalk_benchmark_v7 import load
    from scripts.prepare_mouth_edit_v4 import composite
    from scripts.smoke_mouth_generator_v7 import alpha_from_face, crop_from_face

    require(torch.backends.mps.is_available(), "Frozen explicit MPS renderer required; no fallback")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.mps.set_per_process_memory_fraction(0.65)
    cv2.setNumThreads(1)
    CACHE.mkdir(exist_ok=False)
    chosen = read(TRIAL / "selection.json")
    row = chosen["source"]
    picture = chosen["picture"]
    require(
        sha(Path(picture["file"])) == picture["file_sha256"]
        and sha(Path(row["hidden_file"])) == row["hidden_file_sha256"],
        "Bound picture and source states changed",
    )
    bgr = cv2.imread(picture["file"])
    require(bgr is not None, "Native input picture required")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    require(
        inputs.speech.speech.fingerprint(rgb) == picture["rgb_sha256"], "Actual renderer source RGB differs"
    )
    detector = load_yunet(ROOT / "models/weights/face_detection_yunet_2023mar.onnx")
    faces = detect_faces(bgr, detector=detector, min_size=48)
    require(len(faces) == 1, "Unique measurable source face required; no source replacement")
    crop, matrix, geometry = crop_from_face(rgb, faces[0], "landmark-box")
    alpha = alpha_from_face(rgb.shape, faces[0])
    inverse = cv2.invertAffineTransform(matrix)
    hidden = np.load(row["hidden_file"], allow_pickle=False)
    aligned = clock.select_chunk(hidden, chosen["aligned_plan"])
    earlier = clock.select_chunk(hidden, chosen["earlier_plan"])
    require(aligned.tobytes() != earlier.tobytes(), "Distinct fixed source conditioning required")
    renderer = load(MANIFEST, ASSETS, "mps")
    counts = dict.fromkeys(("vae_encode", "vae_decode", "unet", "speech_encoder"), 0)

    def count(name):
        def hook(module, args):
            counts[name] += 1

        return hook

    handles = [
        module.register_forward_pre_hook(count(name))
        for name, module in (
            ("vae_encode", renderer.vae.encoder),
            ("vae_decode", renderer.vae.decoder),
            ("unet", renderer.unet),
            ("speech_encoder", renderer.encoder),
        )
    ]
    images, records, gpu = [], [], []
    try:
        for name, condition in (
            ("aligned", aligned),
            ("repeat", aligned.copy()),
            ("earlier_320ms", earlier),
            ("zero", np.zeros_like(aligned)),
        ):
            generated, proof = renderer.render(crop, torch.from_numpy(condition.copy()), 117)
            native = cv2.warpAffine(generated, inverse, (rgb.shape[1], rgb.shape[0]), flags=cv2.INTER_LINEAR)
            edited = composite(rgb, native, alpha)
            changed = np.any(edited != rgb, axis=2)
            require(
                changed.any() and not changed[alpha == 0].any(), "Generated edit empty or outside source mask"
            )
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
                    "Could not retain private render support",
                )
            images.append(generated)
            records.append(
                {
                    "condition": name,
                    "proof": proof,
                    "conditioning_sha256": inputs.speech.speech.fingerprint(condition),
                    "changed_native_pixels": int(changed.sum()),
                }
            )
            torch.mps.synchronize()
            gpu.append(
                {
                    "current_allocated_bytes": torch.mps.current_allocated_memory(),
                    "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
                }
            )
    finally:
        for handle in handles:
            handle.remove()
    changes = response_counts(images)
    require(records[0]["proof"] == records[1]["proof"], "Native repeat model proof differs")
    require(
        counts == {"vae_encode": 8, "vae_decode": 4, "unet": 4, "speech_encoder": 0},
        "Exact predeclared module forwards required",
    )
    write(
        output / "private-render-records.json",
        {
            "records": records,
            "geometry": geometry,
            "source_to_crop_matrix_hex": [[float(x).hex() for x in r] for r in matrix],
        },
    )
    result = {
        "schema": "chipmunk.native-mouth-renderer-result/117",
        "status": "complete",
        "device": "mps",
        "fitting_sources": 1,
        "render_calls": 4,
        "seeded_repeat_bitwise": True,
        "counterfactual_changed_crop_pixels": changes,
        "changed_native_pixels": {r["condition"]: r["changed_native_pixels"] for r in records},
        "native_module_forward_counts": counts,
        "face_detection_inputs": 1,
        "new_neural_calls": 1 + sum(counts.values()),
        "maximum_post_render_mps_driver_bytes": max(r["driver_allocated_bytes"] for r in gpu),
        "private_render_records_sha256": sha(output / "private-render-records.json"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "renderer_response_qualified": True,
        "achieved_mouth_synchronization_qualified": False,
        "defaults_changed": False,
        "scope": "One fitting-source native renderer/state response only. No generated-clip quality, phonetic alignment, authenticity detection, fitting or serving confidence claim.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "authored"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        require(args.output is None, "Freeze has no output argument")
        freeze()
    elif args.command == "run":
        require(args.output is None, "Run has no output argument")
        run()
    else:
        verify()
        require(
            args.output is not None and args.output.resolve().is_relative_to(TRIAL.resolve()),
            "Private exclusive guarded output required",
        )
        args.output.mkdir(exist_ok=False)
        qualify(args.output)
