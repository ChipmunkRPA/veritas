"""Fixed-pilot MuseTalk/YuNet frames with original cached speech; research only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import types
from collections import Counter
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import av_conditioning_clock_v109 as clock
from scripts.grid_portrait_frames_v146 import (
    ARCHIVE,
    BASE,
    PANEL,
    ROOT,
    composite,
    exact,
    fingerprint,
    hex_array,
    opacity,
    read,
    require,
    save_png,
    sha,
    write,
)

TRIAL = BASE / "grid-musetalk-frames-v148"
CACHE = ARCHIVE / "grid-musetalk-frames-v148"
PUBLIC = PANEL / "grid-musetalk-frames-freeze-v148.json"
ASSETS = BASE / "lipsync-v7/assets"
MANIFEST = PANEL / "mouth-v7-generator-assets.json"
OVERLAY = BASE / "lipsync-v7/runtime"
CONDITIONS = ("source_time", "earlier_320ms")
FRAMES = tuple(range(27, 47))
MODULES = ("vae_encode", "vae_decode", "unet", "speech_encoder")
CODE = (
    "backend/scripts/grid_musetalk_frames_v148.py",
    "backend/tests/test_grid_musetalk_frames_v148.py",
    "docs/grid-musetalk-frames-v148-method.md",
    "backend/scripts/grid_portrait_frames_v146.py",
    "backend/scripts/grid_source_speech_v147.py",
    "backend/scripts/av_conditioning_clock_v109.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/audio_tree_sampler_v64.py",
)
GENERATOR_SHA = "bc823fb8f21e21c8c1cd722d225c927b0102b724b76942457f166ef047caf9fc"


def dependencies():
    from scripts import grid_source_speech_v147 as speech
    from scripts import grid_transfer_plan_v143 as planning

    return planning, speech


@contextmanager
def isolated_overlay():
    before = sys.path[:]
    try:
        sys.path.insert(0, str(OVERLAY))
        yield
    finally:
        sys.path[:] = before


def package_identity():
    with isolated_overlay():
        return {
            n: importlib.metadata.version(n)
            for n in (
                "torch",
                "numpy",
                "opencv-python-headless",
                "Pillow",
                "diffusers",
                "transformers",
                "safetensors",
                "accelerate",
            )
        }


def baseline():
    """Metadata/code/installed-runtime checks only; no historical source media opened."""
    folder = BASE / "native-mouth-renderer-v118"
    frozen = read(PANEL / "native-mouth-renderer-freeze-v118.json")
    require(frozen == read(folder / "freeze.json"), "Published isolated v118 contract required")
    for path, digest in frozen["code_sha256"].items():
        require(sha(ROOT / path) == digest, "Qualified frozen MuseTalk geometry/renderer code changed")
    old = read(PANEL / "native-mouth-clips-freeze-v120.json")
    for path, digest in old["code_sha256"].items():
        require(sha(ROOT / path) == digest, "Historical native construction code changed")
    require(
        sha(MANIFEST) == frozen["generator_manifest_sha256"] == GENERATOR_SHA,
        "Exact permitted seven-asset manifest required",
    )
    manifest = read(MANIFEST)
    require(
        manifest["schema"] == "chipmunk.mouth-generator-assets/7"
        and len(manifest["assets"]) == 7
        and all(r["license"] in {"mit", "apache-2.0"} for r in manifest["assets"]),
        "Commercially permitted publisher assets required",
    )
    out = folder / "authored/output"
    result, profile = read(out / "result.json"), read(folder / "authored/profile.json")
    public = read(PANEL / "native-mouth-renderer-result-v118.json")
    require(
        result["schema"] == "chipmunk.native-mouth-renderer-result/117"
        and result["status"] == "complete"
        and result["device"] == "mps"
        and result["render_calls"] == 4
        and result["seeded_repeat_bitwise"] is True
        and result["native_module_forward_counts"] == dict(zip(MODULES, (8, 4, 4, 0), strict=True))
        and result["face_detection_inputs"] == 1
        and result["new_neural_calls"] == 17
        and result["renderer_response_qualified"] is True
        and result["freeze_sha256"] == sha(folder / "freeze.json")
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["new_neural_calls"] == 17
        and profile["worker_receipt_sha256"] == sha(out / "result.json")
        and profile["resource_journal_sha256"] == sha(folder / "authored/resources.jsonl")
        and result["private_render_records_sha256"] == sha(out / "private-render-records.json")
        and public["status"] == "complete"
        and public["measurement"] == result
        and public["profile"] == profile
        and public["result_sha256"] == sha(out / "result.json")
        and public["profile_sha256"] == sha(folder / "authored/profile.json"),
        "Actual successful isolated native renderer qualification required",
    )
    with isolated_overlay():
        from scripts import native_mouth_runtime_v118 as runtime

        require(
            runtime.overlay_identity() == frozen["retained_runtime"]
            and sha(PANEL / "mouth-v7-runtime.json") == frozen["runtime_plan_sha256"]
            and sha(PANEL / "mouth-v7-runtime.lock") == frozen["runtime_lock_sha256"]
            and all(importlib.metadata.version(n) == v for n, v in frozen["packages"].items()),
            "Same qualified isolated runtime sources required",
        )
    receipt = read(ASSETS / "verified-assets.json")
    require(
        receipt["schema"] == "chipmunk.mouth-generator-assets-receipt/7"
        and receipt["manifest_sha256"] == GENERATOR_SHA
        and receipt["assets"] == frozen["generator_assets"]
        and receipt["restricted_components_loaded"] == 0,
        "Same permitted acquisition receipt required",
    )
    paths = [
        folder / "freeze.json",
        folder / "authored/profile.json",
        folder / "authored/resources.jsonl",
        out / "result.json",
        out / "private-render-records.json",
        MANIFEST,
        PANEL / "native-mouth-renderer-freeze-v118.json",
        PANEL / "native-mouth-renderer-result-v118.json",
        PANEL / "native-mouth-clips-freeze-v120.json",
        PANEL / "mouth-v7-runtime.json",
        PANEL / "mouth-v7-runtime.lock",
        ASSETS / "verified-assets.json",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def pilot(ordinal):
    planning, _ = dependencies()
    plan = planning.load_completed()
    require(
        type(ordinal) is int and 0 <= ordinal < 32 and len(plan["sources"]) == 32,
        "Original pilot ordinal required",
    )
    source = plan["sources"][ordinal]
    require(source["ordinal"] == ordinal and source["role"] == "pilot", "Reserve clips cannot be consumed")
    return source


def identity():
    planning, speech = dependencies()
    plan = planning.load_completed()
    require(
        read(speech.PUBLIC) == read(speech.TRIAL / "freeze.json"),
        "Published original/delay encoder procedure required",
    )
    for path, digest in read(speech.PUBLIC)["code_sha256"].items():
        require(sha(ROOT / path) == digest, "Source encoder procedure changed")
    return {
        "schema": "chipmunk.grid-musetalk-frames-freeze/148",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "baseline_bindings_sha256": baseline(),
        "plan_freeze_sha256": sha(planning.PUBLIC),
        "plan_private_sha256": sha(planning.TRIAL / "bind-00/output/private-plan.json"),
        "speech_freeze_sha256": sha(speech.PUBLIC),
        "pilot_clip_ordinals": [s["ordinal"] for s in plan["sources"] if s["role"] == "pilot"],
        "source_frames_half_open": [27, 47],
        "conditions": list(CONDITIONS),
        "speech_waveform_id": "original_pcm",
        "planned_condition_frames_per_clip": 40,
        "planned_fixed_render_repeat": 1,
        "planned_face_inputs_per_clip": 21,
        "maximum_render_root_units_per_clip": 164,
        "device": "mps",
        "precision": "float32",
        "mps_memory_fraction": 0.65,
        "torch_interop_opencv_threads": [1, 1, 1],
        "deterministic_algorithms": True,
        "packages": package_identity(),
        "seed_rule": "SHA256(chipmunk.grid-musetalk-frames/148:original_clip_ordinal:source_frame), first8bytes big-endian modulo2**63, shared across driver conditions",
        "scope": "Fixed-pilot per-pair neural construction with native YuNet branch geometry, original whole speech only; unavailable/zero-effect retained. No achieved synchronization, phonetics, face swapping, detection accuracy or default change.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(
        json.dumps(
            {"status": "frozen", "pilot_clips": len(body["pilot_clip_ordinals"]), "new_neural_calls": 0}
        ),
        flush=True,
    )


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged published MuseTalk frame procedure required",
    )


def seed(ordinal, index):
    require(
        type(ordinal) is int and 0 <= ordinal < 32 and type(index) is int and index in FRAMES,
        "Fixed source/frame seed coordinates required",
    )
    return (
        int.from_bytes(
            hashlib.sha256(f"chipmunk.grid-musetalk-frames/148:{ordinal}:{index}".encode()).digest()[:8],
            "big",
        )
        % 2**63
    )


def expected_roots(renders):
    require(type(renders) is int and 0 <= renders <= 41, "Bounded observed render inputs required")
    return dict(zip(MODULES, (2 * renders, renders, renders, 0), strict=True))


def source_states(source, record, cache):
    """Worker-only payload verification; delayed states cannot enter generation."""
    require(
        record["source_ordinal"] == source["ordinal"]
        and record["waveform_id"] == "original_pcm"
        and record["role"] == "pilot"
        and record["status"] in {"encoded", "unavailable"},
        "Only bound original-pilot states permitted",
    )
    if record["status"] == "unavailable":
        return None
    require(
        record["whole_input_encoded"] is True
        and record["hidden_shape"] == [1500, 5, 384]
        and record["hidden_dtype"] == "float32",
        "Whole qualified five-state speech array required",
    )
    wave = record["waveform"]
    require(
        wave["id"] == "original_pcm"
        and wave["changed_samples"] == 0
        and wave["original_pcm_mapping"] == source["input"]["pcm_mapping"]
        and wave["audio_origin_seconds"] == source["input"]["audio_origin_seconds"],
        "Exact untouched original PCM/mapping required",
    )
    arrays = []
    clip_cache = cache / f"batch-{record['batch']:02d}/clip-{source['ordinal']:02d}"
    for path, digest, name in (
        (record["hidden_file"], record["hidden_file_sha256"], "original_pcm.hidden.npy"),
        (wave["pcm_file"], wave["pcm_file_sha256"], "original_pcm.pcm.npy"),
    ):
        file = Path(path)
        require(
            file.is_absolute()
            and file.resolve() == (clip_cache / name).resolve()
            and file.resolve().is_relative_to(cache.resolve())
            and sha(file) == digest,
            "Exact owned original state/PCM payload required",
        )
        arrays.append(np.load(file, allow_pickle=False))
    hidden, pcm = arrays
    require(
        hidden.shape == (1500, 5, 384)
        and hidden.dtype == np.float32
        and np.isfinite(hidden).all()
        and fingerprint(hidden) == record["proof"]["full_hidden_sha256"]
        and record["proof"]["layer_readout_matches"] is True,
        "Finite complete hidden states/raw proof required",
    )
    require(
        pcm.ndim == 1
        and pcm.dtype == np.dtype("<f4")
        and len(pcm) == wave["samples"] == source["input"]["pcm_mapping"]["samples"]
        and 640 <= len(pcm) <= 160000
        and np.isfinite(pcm).all()
        and fingerprint(pcm) == wave["pcm_sha256"] == source["input"]["pcm_mapping"]["pcm_sha256"],
        "Complete unchanged original waveform required",
    )
    scaled = pcm * np.float32(32768)
    require(
        np.equal(scaled, np.rint(scaled)).all() and (scaled >= -32768).all() and (scaled <= 32767).all(),
        "Original signed16 sample lattice required",
    )
    return hidden


def conditioning(source, record, hidden):
    """All40 routes remain explicit; independently reconstruct actual tick/layer copies."""
    require(
        [f["frame_index"] for f in source["frames"]] == list(FRAMES),
        "All20 fixed source-clock frame routes required",
    )
    routes = {(r["frame_index"], r["view"]): r for r in record.get("routing", [])}
    if hidden is not None:
        require(len(routes) == len(record["routing"]) == 40, "Unique full original routing journal required")
    jobs, chunks = [], {}
    for condition in CONDITIONS:
        key = "source_time_conditioning" if condition == "source_time" else "earlier_conditioning"
        for frame in source["frames"]:
            i, selected = frame["frame_index"], frame[key]
            route = routes.get((i, key))
            supported = hidden is not None and route is not None and route["status"] == "verified"
            if supported:
                table = source["input"]["native_video_table"]
                t = exact(table["pts"][i]) * exact(table["time_base"])
                rebuilt = clock.frame_plan(
                    i,
                    str(t - (Fraction(8, 25) if condition == "earlier_320ms" else 0)),
                    source["input"]["audio_origin_seconds"],
                    source["input"]["video_origin_seconds"],
                    source["input"]["pcm_mapping"]["samples"],
                    selected["native_clock_allowance_seconds"],
                )
                require(
                    rebuilt == selected == route["conditioning_plan"]
                    and selected["source_sample_status"] == "supported"
                    and all(selected["retained_tick_slots"]),
                    "Exact full ten-slot source-time plan required",
                )
                chunk = clock.select_chunk(hidden, selected)
                independent = np.stack(
                    [hidden[tick, layer] for tick in selected["tick_indices"] for layer in range(5)]
                )
                require(
                    chunk.shape == (50, 384)
                    and chunk.dtype == np.float32
                    and np.isfinite(chunk).all()
                    and chunk.tobytes() == independent.tobytes()
                    and fingerprint(chunk) == route["chunk_sha256"],
                    "Qualified tick-major/layer-major selected values changed",
                )
                chunks[condition, i] = chunk
            jobs.append(
                {
                    "condition": condition,
                    "frame_index": i,
                    "status": "available" if supported else "unavailable",
                    "reason": None
                    if supported
                    else "Original whole speech or verified full source-time route unavailable",
                    "seed": seed(source["ordinal"], i),
                }
            )
    return jobs, chunks


def face_proof(face):
    points = np.asarray(face.landmarks, np.float64)
    box = [face.x, face.y, face.w, face.h]
    require(
        points.shape == (5, 2)
        and np.isfinite(points).all()
        and np.isfinite(face.score)
        and 0 <= face.score <= 1
        and all(type(v) is int for v in box)
        and min(box[:2]) >= 0
        and min(box[2:]) >= 48,
        "Finite eligible five-point native face geometry required",
    )
    return {
        "box": box,
        "score_hex": float(face.score).hex(),
        "landmarks_xy_hex": [[float(x).hex() for x in row] for row in points],
    }


def observe_geometry(rgb, detect, crop_fn, alpha_fn, event):
    """Authored-callable geometry protocol, without a model implementation here."""
    event({"event": "face_begin"})
    faces = detect(rgb)
    event({"event": "face_end", "observed_faces": len(faces)})
    if len(faces) != 1:
        return {
            "status": "unavailable_no_face" if not faces else "unavailable_multiple_faces",
            "faces": len(faces),
        }, None
    proof = None
    try:
        proof = face_proof(faces[0])
        crop, matrix, geometry = crop_fn(rgb, faces[0], "landmark-box")
        alpha = alpha_fn(rgb.shape, faces[0])
        require(
            isinstance(crop, np.ndarray)
            and crop.shape == (256, 256, 3)
            and crop.dtype == np.uint8
            and isinstance(matrix, np.ndarray)
            and matrix.shape == (2, 3)
            and matrix.dtype == np.float64
            and np.isfinite(matrix).all()
            and abs(np.linalg.det(matrix[:, :2])) > 0
            and isinstance(alpha, np.ndarray)
            and alpha.dtype == np.float32
            and alpha.shape == rgb.shape[:2]
            and np.isfinite(alpha).all()
            and ((alpha >= 0) & (alpha <= 1)).all()
            and (alpha > 0).any(),
            "Unsupported native branch crop, transform or mouth support",
        )
    except ValueError as exc:
        return {"status": "unavailable_geometry", "face": proof, "reason": str(exc)}, None
    return {
        "status": "geometry_available",
        "face": proof,
        "geometry": geometry,
        "source_to_crop_matrix_hex": [[float(x).hex() for x in row] for row in matrix],
        "crop_rgb_sha256": fingerprint(crop),
        "alpha_f32_sha256": fingerprint(alpha),
    }, (crop, matrix, alpha)


def geometry_repeat(first, repeated):
    if first["status"] != "geometry_available":
        return None
    return first == repeated


def render_repeat(first, repeated, first_proof, repeated_proof):
    return bool(np.array_equal(first, repeated) and first_proof == repeated_proof)


def same_noise(first, earlier):
    keys = ("seed", "masked_input_f32_sha256", "original_input_f32_sha256", "sampled_latents_f32_sha256")
    return all(first[k] == earlier[k] for k in keys)


def input_binding(ordinal):
    source = pilot(ordinal)
    _, speech = dependencies()
    record = speech.source_record(ordinal, "original_pcm")
    folder = speech.TRIAL / f"batch-{record['batch']:02d}/authored"
    paths = [
        folder / "profile.json",
        folder / "resources.jsonl",
        folder / "output/result.json",
        folder / "output/source-states.jsonl",
        folder / "output/forward-progress.jsonl",
        folder / "output/fixed-repeat.json",
    ]
    return source, record, {str(p): sha(p) for p in paths}


def source_rgb(picture, source):
    from PIL import Image, UnidentifiedImageError

    table = source["input"]["native_video_table"]
    require(
        picture["pts"] == table["pts"][picture["frame_index"]]
        and sha(picture["file"]) == picture["file_sha256"],
        "Exact original source PNG and native PTS required",
    )
    try:
        with Image.open(picture["file"]) as image:
            rgb = np.asarray(image.convert("RGB"))
    except (OSError, UnidentifiedImageError):
        return None
    require(
        rgb.shape == (table["height"], table["width"], 3) and fingerprint(rgb) == picture["rgb_sha256"],
        "Actual original native raw RGB changed",
    )
    return rgb


def build(ordinal, output):
    with isolated_overlay():
        build_with_overlay(ordinal, output)


def build_with_overlay(ordinal, output):
    import cv2
    from PIL import Image

    source, state, bindings = input_binding(ordinal)
    _, speech = dependencies()
    hidden = source_states(source, state, speech.CACHE)
    jobs, chunks = conditioning(source, state, hidden)
    folder = CACHE / f"clip-{ordinal:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    write(
        output / "private-input-binding.json",
        {"source": source, "original_state": state, "bindings": bindings},
    )
    pictures = {p["frame_index"]: p for p in source["input"].get("pictures", [])}
    geometries, raw, data, references, geometry_repeated = {}, {}, {}, [], None
    face_counts = Counter()
    started, completed = expected_roots(0), expected_roots(0)
    model_handles, active, gpu = [], {}, []
    geometry_repeat_bitwise, render_repeat_bitwise = None, None
    render_repeated = None
    renders = executed = zero_effect = paired = zero_response = 0
    renderer = None
    with (
        (output / "face-progress.jsonl").open("x") as faces_log,
        (output / "root-progress.jsonl").open("x") as roots_log,
        (output / "render-records.jsonl").open("x") as renders_log,
    ):

        def emit(stream, row):
            stream.write(json.dumps({**active, **row}, allow_nan=False) + "\n")
            stream.flush()

        def face_event(row):
            face_counts[row["event"]] += 1
            emit(faces_log, row)

        requested = [i for i in FRAMES if i in pictures]
        if requested:
            import torch

            from app.detectors.image.faces import detect_faces, load_yunet

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.manual_seed(148 + ordinal)
            torch.use_deterministic_algorithms(True)
            cv2.setNumThreads(1)
            from scripts.smoke_mouth_generator_v7 import alpha_from_face, crop_from_face

            frozen = read(PANEL / "native-mouth-renderer-freeze-v118.json")
            require(
                frozen["face_model_record"]["licence"]["class_"] == "commercial",
                "Commercial YuNet geometry required",
            )
            for name, digest in frozen["face_model_record"]["files"].items():
                require(sha(ROOT / name) == digest, "Qualified YuNet weight bytes changed")
            detector = load_yunet(ROOT / "models/weights/face_detection_yunet_2023mar.onnx")
        for i in FRAMES:
            active.update(frame_index=i, repeat=False)
            if i not in pictures:
                geometries[i] = {"frame_index": i, "status": "unavailable_picture"}
                continue
            rgb = source_rgb(pictures[i], source)
            if rgb is None:
                geometries[i] = {"frame_index": i, "status": "unavailable_picture_decode"}
                continue
            raw[i] = rgb
            row, values = observe_geometry(
                rgb,
                lambda x: detect_faces(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), detector=detector, min_size=48),
                crop_from_face,
                alpha_from_face,
                face_event,
            )
            geometries[i] = {"frame_index": i, "picture": pictures[i], **row}
            if values is not None:
                crop, matrix, alpha = values
                destination = folder / f"geometry-{i:04d}"
                destination.mkdir(exist_ok=False)
                crop_file = save_png(destination / "crop.png", crop)
                with (destination / "alpha.npy").open("xb") as stream:
                    np.save(stream, alpha, allow_pickle=False)
                require(
                    np.array_equal(np.load(destination / "alpha.npy", allow_pickle=False), alpha),
                    "Saved native alpha changed",
                )
                geometries[i].update(
                    crop=crop_file,
                    alpha={"path": str(destination / "alpha.npy"), "sha256": sha(destination / "alpha.npy")},
                )
                data[i] = values
            emit(faces_log, {"event": "geometry_observed", "status": row["status"]})
        if 27 in raw:
            active.update(frame_index=27, repeat=True)
            repeated, _ = observe_geometry(
                raw[27].copy(),
                lambda x: detect_faces(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), detector=detector, min_size=48),
                crop_from_face,
                alpha_from_face,
                face_event,
            )
            first = {
                k: v
                for k, v in geometries[27].items()
                if k not in {"frame_index", "picture", "crop", "alpha"}
            }
            geometry_repeat_bitwise = geometry_repeat(first, repeated)
            geometry_repeated = repeated
        write(
            output / "geometry-repeat-check.json",
            {"frame_index": 27, "repeat_bitwise": geometry_repeat_bitwise, "repeated": geometry_repeated},
        )
        require(
            geometry_repeat_bitwise is not False,
            "Available fixed YuNet geometry repeat failed; preserve stage",
        )
        expected_available = sum(j["status"] == "available" and j["frame_index"] in data for j in jobs)
        expect_repeat = 27 in data and ("source_time", 27) in chunks
        source_outputs = {}
        try:
            for job in jobs:
                i, mode = job["frame_index"], job["condition"]
                active.update(frame_index=i, condition=mode, repeat=False)
                if job["status"] != "available" or i not in data:
                    reason = (
                        job["reason"]
                        if job["status"] != "available"
                        else "Native YuNet mouth geometry unavailable"
                    )
                    references.append(
                        {
                            **job,
                            "status": "unavailable",
                            "rendered": False,
                            "reason": reason,
                            "geometry_status": geometries[i]["status"],
                        }
                    )
                    emit(renders_log, references[-1])
                    continue
                if renderer is None:
                    from scripts.musetalk_benchmark_v7 import load

                    require(
                        torch.backends.mps.is_available(), "Qualified MPS generation required; no fallback"
                    )
                    torch.mps.set_per_process_memory_fraction(0.65)
                    emit(roots_log, {"event": "renderer_load_begin", "device": "mps", "precision": "float32"})
                    renderer = load(MANIFEST, ASSETS, "mps")
                    require(
                        renderer.device == "mps"
                        and renderer.load_audit["all_three_modules_strict_loaded"] is True
                        and renderer.load_audit["vae_all_cpu_tensors_exact"] is True,
                        "Unchanged strict renderer loading required",
                    )
                    emit(roots_log, {"event": "renderer_load_end", "strict_load_audit": renderer.load_audit})
                    modules = {
                        "vae_encode": renderer.vae.encoder,
                        "vae_decode": renderer.vae.decoder,
                        "unet": renderer.unet,
                        "speech_encoder": renderer.encoder,
                    }
                    for model in (renderer.vae, renderer.unet, renderer.encoder):
                        require(
                            not model.training
                            and all(
                                p.device.type == "mps" and p.dtype == torch.float32
                                for p in model.parameters()
                            ),
                            "Explicit eval MPS float32 modules required",
                        )

                    def hook(name, finished):
                        def counted(module, args, *rest):
                            (completed if finished else started)[name] += 1
                            emit(
                                roots_log, {"event": "root_end" if finished else "root_begin", "module": name}
                            )

                        return counted

                    for name, model in modules.items():
                        model_handles.extend(
                            [
                                model.register_forward_pre_hook(hook(name, False)),
                                model.register_forward_hook(hook(name, True)),
                            ]
                        )
                crop, matrix, alpha = data[i]
                require(
                    sha(pictures[i]["file"]) == pictures[i]["file_sha256"],
                    "Original source changed after geometry",
                )
                chunk = chunks[mode, i]
                emit(renders_log, {"event": "render_begin", "seed": job["seed"]})
                generated, proof = renderer.render(crop, torch.from_numpy(chunk.copy()), job["seed"])
                require(
                    generated.shape == (256, 256, 3)
                    and generated.dtype == np.uint8
                    and proof["seed"] == job["seed"]
                    and proof["output_rgb_sha256"] == fingerprint(generated),
                    "Complete256-square RGB output/seed proof required",
                )
                renders += 1
                gpu.append(torch.mps.driver_allocated_memory())
                native = cv2.warpAffine(
                    generated,
                    cv2.invertAffineTransform(matrix),
                    (raw[i].shape[1], raw[i].shape[0]),
                    flags=cv2.INTER_LINEAR,
                )
                edited, mask = composite(raw[i], native, alpha, opacity(i))
                files = {
                    kind: save_png(folder / f"{mode}-{i:04d}-{kind}.png", image)
                    for kind, image in (("crop", generated), ("native", edited), ("mask", mask))
                }
                record = {
                    **job,
                    "rendered": True,
                    "status": "rendered",
                    "files": files,
                    "proof": proof,
                    "conditioning_sha256": fingerprint(chunk),
                    "conditioning_plan": next(f for f in source["frames"] if f["frame_index"] == i)[
                        "source_time_conditioning" if mode == "source_time" else "earlier_conditioning"
                    ],
                    "opacity": opacity(i),
                    "changed_native_pixels": int(np.count_nonzero(mask)),
                    "outside_alpha_changed_pixels": 0,
                    "post_render_mps_driver_bytes": gpu[-1],
                }
                if mode == "source_time":
                    source_outputs[i] = record
                elif i in source_outputs:
                    first = source_outputs[i]
                    require(
                        same_noise(first["proof"], proof),
                        "Paired drivers must share identical source inputs and sampled latents",
                    )
                    other_file = first["files"]["crop"]
                    require(
                        sha(other_file["path"]) == other_file["sha256"],
                        "Completed source-time crop cache changed",
                    )
                    with Image.open(other_file["path"]) as image:
                        other_crop = np.asarray(image.convert("RGB"))
                    changed = int(np.any(generated != other_crop, axis=2).sum())
                    record["earlier_vs_source_crop_changed_pixels"] = changed
                    paired += 1
                    zero_response += changed == 0
                references.append(record)
                emit(renders_log, {"event": "render_complete", **record})
                executed += 1
                zero_effect += not mask.any()
                if mode == "source_time" and i == 27:
                    active["repeat"] = True
                    emit(renders_log, {"event": "fixed_repeat_begin", "seed": job["seed"]})
                    repeated, other = renderer.render(
                        crop.copy(), torch.from_numpy(chunk.copy()), job["seed"]
                    )
                    require(
                        repeated.shape == (256, 256, 3) and repeated.dtype == np.uint8,
                        "Complete fixed-repeat output required",
                    )
                    renders += 1
                    gpu.append(torch.mps.driver_allocated_memory())
                    render_repeat_bitwise = render_repeat(generated, repeated, proof, other)
                    repeated_native = cv2.warpAffine(
                        repeated,
                        cv2.invertAffineTransform(matrix),
                        (raw[i].shape[1], raw[i].shape[0]),
                        flags=cv2.INTER_LINEAR,
                    )
                    repeated_edit, repeated_mask = composite(raw[i], repeated_native, alpha, opacity(i))
                    render_repeat_bitwise = (
                        render_repeat_bitwise
                        and np.array_equal(edited, repeated_edit)
                        and np.array_equal(mask, repeated_mask)
                    )
                    repeat_files = {
                        kind: save_png(folder / f"source_time-repeat-0027-{kind}.png", image)
                        for kind, image in (
                            ("crop", repeated),
                            ("native", repeated_edit),
                            ("mask", repeated_mask),
                        )
                    }
                    render_repeated = {
                        "frame_index": 27,
                        "repeat_bitwise": render_repeat_bitwise,
                        "files": repeat_files,
                        "proof": other,
                        "post_render_mps_driver_bytes": gpu[-1],
                    }
                    write(output / "render-repeat-check.json", render_repeated)
                    emit(renders_log, {"event": "fixed_repeat_complete", **render_repeated})
                    require(
                        render_repeat_bitwise is True, "Available fixed render repeat failed; preserve stage"
                    )
        finally:
            for handle in model_handles:
                handle.remove()
    repeat_executed = int(render_repeated is not None)
    require(
        len(references) == 40
        and started == completed == expected_roots(renders)
        and executed == expected_available
        and repeat_executed == int(expect_repeat)
        and face_counts["face_begin"] == face_counts["face_end"] == len(raw) + int(27 in raw),
        "Every planned pair and observed neural input must remain counted",
    )
    if render_repeated is None:
        write(
            output / "render-repeat-check.json",
            {
                "frame_index": 27,
                "repeat_bitwise": None,
                "reason": "Fixed original frame/crop/source-time route unavailable; no replacement",
            },
        )
    canonical = {
        "schema": "chipmunk.grid-musetalk-frames-private/148",
        "clip_ordinal": ordinal,
        "planned": source,
        "frames": references,
        "geometry": [geometries[i] for i in FRAMES],
        "geometry_repeat": geometry_repeated,
        "render_repeat": render_repeated,
    }
    write(folder / "outputs.json", canonical)
    table = source["input"].get("native_video_table")
    interval = (
        None
        if table is None or len(table["pts"]) <= 47
        else [str(exact(table["pts"][i]) * exact(table["time_base"])) for i in (27, 47)]
    )
    result = {
        "schema": "chipmunk.grid-musetalk-frames-result/148",
        "status": "complete",
        "clip_ordinal": ordinal,
        "planned_condition_frames": 40,
        "available_condition_frames": executed,
        "executed_condition_frames": executed,
        "skipped_condition_frames": 40 - executed,
        "planned_face_inputs": 21,
        "face_inputs_started": face_counts["face_begin"],
        "face_inputs_completed": face_counts["face_end"],
        "geometry_available_frames": len(data),
        "geometry_repeat_bitwise": geometry_repeat_bitwise,
        "renderer_repeat_bitwise": render_repeat_bitwise,
        "planned_render_repeat_frames": 1,
        "executed_render_repeat_frames": repeat_executed,
        "renderer_inputs": renders,
        "top_level_module_forwards_started": started,
        "top_level_module_forwards_completed": completed,
        "new_speech_encoder_inputs": 0,
        "new_neural_calls": face_counts["face_begin"] + sum(started.values()),
        "zero_effect_native_outputs": int(zero_effect),
        "device": "mps",
        "precision": "float32",
        "source_earlier_response_pairs": paired,
        "source_earlier_zero_pixel_responses": zero_response,
        "maximum_post_render_mps_driver_bytes": max(gpu) if gpu else None,
        "source_frames_half_open": [27, 47],
        "actual_native_interval_seconds": interval,
        "private_output_references_sha256": sha(folder / "outputs.json"),
        "private_input_binding_sha256": sha(output / "private-input-binding.json"),
        "private_face_progress_sha256": sha(output / "face-progress.jsonl"),
        "private_root_progress_sha256": sha(output / "root-progress.jsonl"),
        "private_render_journal_sha256": sha(output / "render-records.jsonl"),
        "private_geometry_repeat_sha256": sha(output / "geometry-repeat-check.json"),
        "private_render_repeat_sha256": sha(output / "render-repeat-check.json"),
        "freeze_sha256": sha(PUBLIC),
        "achieved_motion_assessed": False,
        "achieved_synchronization_qualified": False,
        "defaults_changed": False,
        "scope": "Fixed-pilot native YuNet/MuseTalk construction with original whole speech only. Missing geometry/routes and zero effects retained. Pixel changes are not achieved motion, phonetics, face swapping or detector accuracy; no serving change. MPS driver samples are separate from CPU RSS and not peaks.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def prerequisite(ordinal):
    planning, speech = dependencies()
    for batch, sources in enumerate(speech.batches(planning.load_completed())):
        if any(s["ordinal"] == ordinal for s in sources):
            folder = speech.TRIAL / f"batch-{batch:02d}/authored"
            if not all(p.is_file() for p in (folder / "output/result.json", folder / "profile.json")):
                return False
            speech.source_record(ordinal, "original_pcm")
            return True
    raise ValueError("Only original pilot clips have a source-state prerequisite")


def check_output_inventory(records, result, source, cache):
    """Pure/metadata-only artifact coverage, repeat and ownership checks."""
    ordinal = source["ordinal"]
    require(
        records["schema"] == "chipmunk.grid-musetalk-frames-private/148"
        and records["planned"] == source
        and records["clip_ordinal"] == ordinal
        and [(r["condition"], r["frame_index"]) for r in records["frames"]]
        == [(c, i) for c in CONDITIONS for i in FRAMES]
        and [r["frame_index"] for r in records["geometry"]] == list(FRAMES),
        "All40 condition rows and20 native geometry rows required",
    )

    def owned(file, relative):
        path = Path(file["path"])
        require(
            path.is_absolute()
            and path.resolve() == (cache / relative).resolve()
            and path.resolve().is_relative_to(cache.resolve())
            and len(file["sha256"]) == 64,
            "Exact owned construction reference required",
        )

    geometry = {r["frame_index"]: r for r in records["geometry"]}
    original = {r["frame_index"]: r for r in source["input"].get("pictures", [])}
    available_geometry = [r for r in records["geometry"] if r["status"] == "geometry_available"]
    for row in available_geometry:
        i = row["frame_index"]
        require(row["picture"] == original[i], "Branch geometry lost native picture correspondence")
        matrix = hex_array(row["source_to_crop_matrix_hex"], (2, 3))
        require(abs(np.linalg.det(matrix[:, :2])) > 0, "Invertible recorded crop matrix required")
        owned(row["crop"], f"geometry-{i:04d}/crop.png")
        owned(row["alpha"], f"geometry-{i:04d}/alpha.npy")
    rendered = []
    plans = {f["frame_index"]: f for f in source["frames"]}
    for row in records["frames"]:
        i, mode = row["frame_index"], row["condition"]
        require(row["seed"] == seed(ordinal, i), "Same original clip/picture seed required for both drivers")
        if row["rendered"] is False:
            require(
                row["status"] == "unavailable" and row["reason"] and "files" not in row,
                "Unavailable pair cannot invent an output",
            )
            continue
        require(
            row["rendered"] is True
            and row["status"] == "rendered"
            and geometry[i]["status"] == "geometry_available"
            and row["conditioning_plan"]
            == plans[i]["source_time_conditioning" if mode == "source_time" else "earlier_conditioning"]
            and row["proof"]["seed"] == row["seed"]
            and row["outside_alpha_changed_pixels"] == 0
            and row["proof"]["output_rgb_sha256"] == row["files"]["crop"]["raw_sha256"]
            and set(row["files"]) == {"crop", "native", "mask"},
            "Rendered pair lacks its actual geometry/conditioning/proof",
        )
        for kind, file in row["files"].items():
            owned(file, f"{mode}-{i:04d}-{kind}.png")
        rendered.append(row)
    first = records["frames"][0]
    repeat = records["render_repeat"]
    repeated = int(first["rendered"] is True)
    if repeated:
        require(
            repeat["frame_index"] == 27
            and repeat["repeat_bitwise"] is True
            and repeat["proof"] == first["proof"]
            and set(repeat["files"]) == {"crop", "native", "mask"},
            "Exact available fixed render repeat required",
        )
        for kind, file in repeat["files"].items():
            owned(file, f"source_time-repeat-0027-{kind}.png")
            require(file["raw_sha256"] == first["files"][kind]["raw_sha256"], "Fixed repeat pixels differ")
    else:
        require(repeat is None, "Unavailable fixed repeat cannot be replaced")
    roots = expected_roots(len(rendered) + repeated)
    require(
        result["planned_condition_frames"] == 40
        and result["executed_condition_frames"] == result["available_condition_frames"] == len(rendered)
        and result["skipped_condition_frames"] == 40 - len(rendered)
        and result["geometry_available_frames"] == len(available_geometry)
        and result["executed_render_repeat_frames"] == repeated
        and result["renderer_repeat_bitwise"] is (True if repeated else None)
        and result["geometry_repeat_bitwise"]
        is (True if geometry[27]["status"] == "geometry_available" else None)
        and result["renderer_inputs"] == len(rendered) + repeated
        and result["top_level_module_forwards_started"]
        == result["top_level_module_forwards_completed"]
        == roots
        and 0 <= result["face_inputs_started"] == result["face_inputs_completed"] <= 21
        and result["new_neural_calls"] == result["face_inputs_started"] + sum(roots.values())
        and result["new_speech_encoder_inputs"] == 0
        and result["zero_effect_native_outputs"] == sum(r["changed_native_pixels"] == 0 for r in rendered),
        "Observed availability/forwards/repeats do not match actual output inventory",
    )


def load_completed(ordinal):
    """Metadata-only consumer admission; consumers must verify referenced PNG/alpha bytes."""
    verify()
    source = pilot(ordinal)
    _, speech = dependencies()
    folder = TRIAL / f"clip-{ordinal:02d}/authored"
    result = read(folder / "output/result.json")
    speech.qualification_profile(
        folder, result["new_neural_calls"], {}, caller="grid_musetalk_frames_v148.py", command="authored"
    )
    references = CACHE / f"clip-{ordinal:02d}/outputs.json"
    records = read(references)
    require(
        result["schema"] == "chipmunk.grid-musetalk-frames-result/148"
        and result["clip_ordinal"] == ordinal
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_output_references_sha256"] == sha(references),
        "Actual bound complete frame construction required",
    )
    for name, key in (
        ("private-input-binding.json", "private_input_binding_sha256"),
        ("face-progress.jsonl", "private_face_progress_sha256"),
        ("root-progress.jsonl", "private_root_progress_sha256"),
        ("render-records.jsonl", "private_render_journal_sha256"),
        ("geometry-repeat-check.json", "private_geometry_repeat_sha256"),
        ("render-repeat-check.json", "private_render_repeat_sha256"),
    ):
        require(sha(folder / "output" / name) == result[key], "Completed construction journal changed")
    check_output_inventory(records, result, source, references.parent)
    root_events = [
        json.loads(line) for line in (folder / "output/root-progress.jsonl").read_text().splitlines()
    ]
    face_events = [
        json.loads(line) for line in (folder / "output/face-progress.jsonl").read_text().splitlines()
    ]
    require(
        all(
            dict(Counter(r["module"] for r in root_events if r["event"] == event))
            == {k: v for k, v in result["top_level_module_forwards_started"].items() if v}
            for event in ("root_begin", "root_end")
        )
        and all(
            sum(r["event"] == event for r in face_events) == result["face_inputs_started"]
            for event in ("face_begin", "face_end")
        ),
        "Observed flushed face/root journal counts changed",
    )
    return records


def routed_popen(argv, ordinal, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"] and len(argv) == 6,
        "Only fixed authored CLI can be routed",
    )
    return subprocess.Popen(
        [*argv[:3], "scripts.grid_musetalk_frames_v148", *argv[4:], str(ordinal)], **kwargs
    )


def run(ordinal):
    from scripts import video_appearance_runtime_v108 as runtime
    from scripts.video_appearance_corrected_v112 import scoped

    pilot(ordinal)
    folder = TRIAL / f"clip-{ordinal:02d}"
    folder.mkdir(exist_ok=True)
    proxy = types.SimpleNamespace(
        Popen=lambda argv, **kwargs: routed_popen(argv, ordinal, **kwargs),
        STDOUT=subprocess.STDOUT,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    scoped(
        runtime.run,
        {
            "TRIAL": folder,
            "ARCHIVE": ARCHIVE,
            "verify": verify,
            "prerequisites": lambda c, b: prerequisite(ordinal),
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
        require(not args.arguments, "Freeze takes no arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One original pilot ordinal required")
        run(int(args.arguments[0]))
    else:
        require(len(args.arguments) == 2, "Exact guarded output path and pilot ordinal required")
        output, ordinal = Path(args.arguments[0]), int(args.arguments[1])
        verify()
        pilot(ordinal)
        require(
            output.resolve() == (TRIAL / f"clip-{ordinal:02d}/authored/output").resolve(),
            "Exact exclusive per-clip output required",
        )
        output.mkdir(exist_ok=False)
        build(ordinal, output)
