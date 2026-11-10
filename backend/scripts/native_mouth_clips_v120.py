"""Known native mouth edits and matched ordinary controls; no detector fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import types
import wave
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import av_conditioning_clock_v109 as clock
from scripts import native_av_mux_v119 as mux
from scripts import native_mouth_inputs_v116 as inputs
from scripts import native_mouth_runtime_v118 as qualified
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = inputs.ROOT, inputs.BASE, inputs.PANEL
TRIAL = BASE / "native-mouth-clips-v120"
CACHE = inputs.CACHE.parent / "native-mouth-clips-v120"
PUBLIC = PANEL / "native-mouth-clips-freeze-v120.json"
CONDITIONS = ("original", "resample", "blur2", "blur4", "ai_aligned", "ai_earlier_320ms")
CODE = (
    "backend/scripts/native_mouth_clips_v120.py",
    "backend/tests/test_native_mouth_clips_v120.py",
    "docs/native-mouth-clips-v120-method.md",
)
read, write, sha, require = inputs.read, inputs.write, inputs.sha, inputs.require


def seed(index, frame):
    require(
        type(index) is int and type(frame) is int and index >= 0 and frame >= 0,
        "Native integer source/frame seed required",
    )
    return int.from_bytes(
        hashlib.sha256(f"chipmunk.native-mouth-clips/120:{index}:{frame}".encode()).digest()[:8], "big"
    ) % (2**63)


def opacity(index, bounds):
    start, end = bounds
    require(
        type(index) is int and end - start == 20 and start <= index < end,
        "Planned native edit frame required",
    )
    return min(1.0, (index - start + 1) / 2, (end - index) / 2)


def shifted(frame):
    result = clock.frame_plan(
        frame["video_frame_index"],
        str(Fraction(frame["source_time_seconds"]) - Fraction(8, 25)),
        frame["source_audio_origin_seconds"],
        frame["source_video_origin_seconds"],
        frame["observed_mono_samples"],
        frame["native_clock_allowance_seconds"],
    )
    return (
        result
        if result["source_sample_status"] == "supported" and all(result["retained_tick_slots"])
        else None
    )


def planned():
    rows = read(inputs.TRIAL / "plan.json")
    native = {r["index"]: r for r in read(inputs.native.TRIAL / "qualified-plan.json")["assets"]}
    waves = {p["asset_index"]: w for p, w in inputs.speech.source_records()}
    result = []
    for row in rows:
        edit = row["edit_frames_half_open"]
        frames = (
            []
            if edit is None
            else [f for f in row["clock_frames"] if edit[0] <= f["video_frame_index"] < edit[1]]
        )
        require(
            edit is None or len(frames) == 20 and all(all(f["retained_tick_slots"]) for f in frames),
            "All aligned native edit positions required",
        )
        earlier = [shifted(f) for f in frames]
        result.append(
            {
                "source": row,
                "native_rgb_sha256": native[row["asset_index"]]["rgb_sha256"],
                "waveform": waves[row["asset_index"]],
                "edit_clock_frames": frames,
                "earlier_clock_frames": earlier if frames and all(f is not None for f in earlier) else None,
                "conditions": list(CONDITIONS),
            }
        )
    require(len(result) == 48, "All original sources required")
    return result


def identity():
    mux.verify()
    stage = mux.TRIAL / "bind-00"
    r, p = read(stage / "output/result.json"), read(stage / "profile.json")
    require(
        r["status"] == "complete"
        and r["cases"] == 3
        and p["child_exit_code"] == 0
        and p["guard"] is None
        and p["sample_errors"] == 0
        and p["worker_receipt_sha256"] == sha(stage / "output/result.json"),
        "Successful full native assembly qualification required",
    )
    require(
        read(TRIAL / "plan.json") == planned(), "Unchanged complete source-only construction plan required"
    )
    rows = read(TRIAL / "plan.json")
    return {
        "schema": "chipmunk.native-mouth-clips-freeze/120",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "private_plan_sha256": sha(TRIAL / "plan.json"),
        "mux_freeze_sha256": sha(mux.PUBLIC),
        "mux_result_sha256": sha(stage / "output/result.json"),
        "renderer_freeze_sha256": sha(qualified.PUBLIC),
        "sources": 48,
        "planned_cases": 288,
        "native20_edit_sources": 47,
        "earlier_conditioning_available_sources": sum(r["earlier_clock_frames"] is not None for r in rows),
        "role_counts": dict(Counter(r["source"]["study_role"] for r in rows)),
        "conditions": list(CONDITIONS),
        "batch_sources": 6,
        "batches": 8,
        "scope": "Complete source-clock-preserving generated clips and ordinary controls. Planned missing conditions remain unavailable, not negative detector results. No detector inference/fitting, achieved synchronization or serving promotion.",
    }


def freeze():
    rows = planned()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "plan.json", rows)
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(
        json.dumps(
            {
                "status": "frozen",
                "sources": 48,
                "planned_cases": 288,
                "earlier_supported_sources": body["earlier_conditioning_available_sources"],
                "new_neural_calls": 0,
            }
        ),
        flush=True,
    )


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Published unchanged native generation required",
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "extract"],
        "Only fixed generation worker can be routed",
    )
    routed = list(argv)
    routed[3] = "scripts.native_mouth_clips_v120"
    return subprocess.Popen(routed, **kwargs)


def run(batch):
    require(type(batch) is int and 0 <= batch < 8, "One fixed source batch required")
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
    )("extract", batch)


class UnsupportedGeometry(ValueError):
    pass


def build(batch, output):
    import cv2
    import torch

    from app.detectors.image.faces import detect_faces, load_yunet
    from app.detectors.video.decode import FrameTable, decode_frames
    from scripts.musetalk_benchmark_v7 import load
    from scripts.prepare_mouth_edit_v4 import composite
    from scripts.smoke_mouth_generator_v7 import alpha_from_face, crop_from_face

    require(torch.backends.mps.is_available(), "Explicit qualified MPS required; no fallback")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.mps.set_per_process_memory_fraction(0.65)
    cv2.setNumThreads(1)
    cache = CACHE / f"batch-{batch:02d}"
    cache.mkdir(parents=True, exist_ok=False)
    plans = read(TRIAL / "plan.json")[batch * 6 : (batch + 1) * 6]
    require(len(plans) == 6, "Complete source batch required")
    detector = load_yunet(ROOT / "models/weights/face_detection_yunet_2023mar.onnx")
    counts = dict.fromkeys(("vae_encode", "vae_decode", "unet", "speech_encoder"), 0)
    face_inputs = 0
    render_calls = 0
    renderer = None
    handles = []
    repeat = False
    cases = []

    def hook(name):
        def entered(module, args):
            counts[name] += 1

        return entered

    try:
        with (output / "cases.jsonl").open("x") as journal:
            for offset, plan in enumerate(plans):
                source = plan["source"]
                folder = cache / f"source-{offset:02d}"
                folder.mkdir(exist_ok=False)
                table = FrameTable.from_json(source["table"])
                path = Path(source["source_path"])
                require(
                    sha(path) == source["input_sha256"]
                    and sha(Path(source["hidden_file"])) == source["hidden_file_sha256"],
                    "Exact source media/states required",
                )
                wav = plan["waveform"]
                wavepath = inputs.speech.clock.CACHE / wav["file"]
                require(sha(wavepath) == wav["sha256"], "Exact qualified waveform required")
                with wave.open(str(wavepath), "rb") as stream:
                    require(
                        (
                            stream.getframerate(),
                            stream.getnchannels(),
                            stream.getsampwidth(),
                            stream.getnframes(),
                        )
                        == (16000, 1, 2, wav["sample_frames"]),
                        "Qualified mono16k WAV required",
                    )
                    pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()
                origin = Fraction(
                    inputs.speech.source_records()[batch * 6 + offset][0]["frames"][0][
                        "source_audio_origin_seconds"
                    ]
                )
                edited = {c: {} for c in CONDITIONS if c != "original"}
                proofs = {c: [] for c in edited}
                geometry = {}
                unsupported = {}
                bounds = source["edit_frames_half_open"]
                if bounds is None:
                    unsupported = {
                        c: "Source lacks the frozen 20-picture edit with native context; no padding or replacement."
                        for c in edited
                    }
                else:
                    bound = read(inputs.TRIAL / f"bind-{batch:02d}/output/bindings.json")[offset]
                    require(bound["planned"] == source, "Exact qualified picture binding required")
                    try:
                        for picture in bound["pictures"]:
                            file = Path(picture["file"])
                            require(sha(file) == picture["file_sha256"], "Bound native picture changed")
                            bgr = cv2.imread(str(file))
                            require(bgr is not None, "Native picture required")
                            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            require(
                                inputs.speech.speech.fingerprint(rgb) == picture["rgb_sha256"],
                                "Model RGB differs from bound source",
                            )
                            faces = detect_faces(bgr, detector=detector, min_size=48)
                            face_inputs += 1
                            if len(faces) != 1:
                                raise UnsupportedGeometry(
                                    "A planned edit/context picture lacks a unique eligible face."
                                )
                            i = picture["frame_index"]
                            if bounds[0] <= i < bounds[1]:
                                try:
                                    crop, matrix, g = crop_from_face(rgb, faces[0], "landmark-box")
                                    alpha = alpha_from_face(rgb.shape, faces[0])
                                except ValueError as exc:
                                    raise UnsupportedGeometry(str(exc)) from exc
                                geometry[i] = (rgb, crop, matrix, alpha, g)
                    except UnsupportedGeometry as exc:
                        unsupported = {c: str(exc) for c in edited}
                    if not unsupported:
                        require(len(geometry) == 20, "All native edit geometries required")
                        if plan["earlier_clock_frames"] is None:
                            unsupported["ai_earlier_320ms"] = (
                                "The fixed 320-ms earlier source context is not fully observed for all 20 edit pictures."
                            )
                        hidden = np.load(source["hidden_file"], allow_pickle=False)
                        if renderer is None:
                            renderer = load(qualified.core.MANIFEST, qualified.core.ASSETS, "mps")
                            handles = [
                                module.register_forward_pre_hook(hook(name))
                                for name, module in (
                                    ("vae_encode", renderer.vae.encoder),
                                    ("vae_decode", renderer.vae.decoder),
                                    ("unet", renderer.unet),
                                    ("speech_encoder", renderer.encoder),
                                )
                            ]
                        aligned = {f["video_frame_index"]: f for f in plan["edit_clock_frames"]}
                        earlier = (
                            {}
                            if plan["earlier_clock_frames"] is None
                            else {f["video_frame_index"]: f for f in plan["earlier_clock_frames"]}
                        )
                        for i in range(*bounds):
                            rgb, crop, matrix, alpha, g = geometry[i]
                            inverse = cv2.invertAffineTransform(matrix)
                            variants = {
                                "resample": cv2.warpAffine(
                                    crop, inverse, (table.width, table.height), flags=cv2.INTER_LINEAR
                                ),
                                "blur2": cv2.GaussianBlur(rgb, (0, 0), 2.0),
                                "blur4": cv2.GaussianBlur(rgb, (0, 0), 4.0),
                            }
                            model_proofs = {}
                            for condition, f in [("ai_aligned", aligned[i])] + (
                                [] if not earlier else [("ai_earlier_320ms", earlier[i])]
                            ):
                                chunk = clock.select_chunk(hidden, f)
                                generated, proof = renderer.render(
                                    crop, torch.from_numpy(chunk.copy()), seed(source["asset_index"], i)
                                )
                                render_calls += 1
                                if not repeat and condition == "ai_aligned":
                                    other, other_proof = renderer.render(
                                        crop, torch.from_numpy(chunk.copy()), seed(source["asset_index"], i)
                                    )
                                    render_calls += 1
                                    require(
                                        generated.tobytes() == other.tobytes() and proof == other_proof,
                                        "Native batch repeat differs",
                                    )
                                    repeat = True
                                variants[condition] = cv2.warpAffine(
                                    generated, inverse, (table.width, table.height), flags=cv2.INTER_LINEAR
                                )
                                model_proofs[condition] = {
                                    "model_proof": proof,
                                    "conditioning_plan": f,
                                    "conditioning_sha256": inputs.speech.speech.fingerprint(chunk),
                                }
                            for condition, value in variants.items():
                                composite_rgb = composite(rgb, value, alpha, opacity(i, bounds))
                                changed = np.any(composite_rgb != rgb, axis=2)
                                require(
                                    not changed[alpha == 0].any(), "Composite changes outside mouth support"
                                )
                                require(
                                    not condition.startswith("ai_") or changed.any(),
                                    "A required AI edit picture has no changed pixels",
                                )
                                image = folder / f"{condition}-{i:04d}.png"
                                mask = folder / f"{condition}-mask-{i:04d}.png"
                                require(
                                    cv2.imwrite(str(image), cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR))
                                    and cv2.imwrite(str(mask), changed.astype(np.uint8) * 255),
                                    "Could not retain native edits/masks",
                                )
                                edited[condition][i] = {
                                    "file": str(image),
                                    "file_sha256": sha(image),
                                    "rgb_sha256": inputs.speech.speech.fingerprint(composite_rgb),
                                }
                                proofs[condition].append(
                                    {
                                        "frame_index": i,
                                        "pts": table.pts[i],
                                        "image_sha256": sha(image),
                                        "mask_file": str(mask),
                                        "mask_sha256": sha(mask),
                                        "changed_pixels": int(changed.sum()),
                                        "geometry": g,
                                        "source_to_crop_matrix_hex": [
                                            [float(x).hex() for x in r] for r in matrix
                                        ],
                                        **model_proofs.get(condition, {}),
                                    }
                                )
                        del hidden

                def complete_frames(
                    condition, path=path, table=table, expected=plan["native_rgb_sha256"], edited=edited
                ):
                    for frame in decode_frames(path, table, range(len(table)), timeout_s=120):
                        require(
                            inputs.speech.speech.fingerprint(frame.rgb) == expected[frame.index],
                            "Native source pixels changed during assembly",
                        )
                        image = edited.get(condition, {}).get(frame.index)
                        if image is None:
                            yield frame.rgb
                        else:
                            require(
                                sha(Path(image["file"])) == image["file_sha256"], "Saved edit bytes changed"
                            )
                            bgr = cv2.imread(image["file"])
                            require(bgr is not None, "Generated native picture missing")
                            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            require(
                                inputs.speech.speech.fingerprint(rgb) == image["rgb_sha256"],
                                "Saved edit RGB differs from composite/mask",
                            )
                            yield rgb

                for condition in CONDITIONS:
                    case = {
                        "asset_index": source["asset_index"],
                        "source_group": source["source_group"],
                        "study_role": source["study_role"],
                        "condition": condition,
                        "known_ai_provenance": condition.startswith("ai_"),
                        "achieved_synchronization_qualified": False,
                        "edit_frames_half_open": None if condition == "original" else bounds,
                        "status": "unavailable",
                    }
                    if condition in unsupported:
                        case["reason"] = unsupported[condition]
                    else:
                        require(
                            condition == "original" or len(edited[condition]) == 20,
                            "Complete planned edit pictures required",
                        )
                        file = folder / f"{condition}.mkv"
                        expected = mux.encode(
                            file,
                            complete_frames(condition),
                            table.pts,
                            (table.height, table.width, 3),
                            pcm,
                            origin,
                        )
                        review = mux.verify_media(file, expected, (table.height, table.width, 3))
                        case.update(
                            status="ready",
                            file=str(file),
                            file_sha256=sha(file),
                            expected=expected,
                            review=review,
                            edit_proofs=proofs.get(condition, []),
                            before_during_after=None
                            if bounds is None
                            else {
                                "before": [bounds[0] - 2, bounds[0] - 1],
                                "during": [bounds[0], sum(bounds) // 2, bounds[1] - 1],
                                "after": [bounds[1], bounds[1] + 1],
                            },
                        )
                    cases.append(case)
                    journal.write(json.dumps(case, allow_nan=False) + "\n")
                    journal.flush()
                require(sha(path) == source["input_sha256"], "Original source changed during construction")
                print(
                    json.dumps(
                        {
                            "completed_source_ordinal": batch * 6 + offset,
                            "ready_conditions": sum(r["status"] == "ready" for r in cases[-6:]),
                            "planned_conditions": 6,
                        }
                    ),
                    flush=True,
                )
        require(
            counts
            == {
                "vae_encode": render_calls * 2,
                "vae_decode": render_calls,
                "unet": render_calls,
                "speech_encoder": 0,
            },
            "Actual fixed module counts differ",
        )
    finally:
        for handle in handles:
            handle.remove()
    result = {
        "schema": "chipmunk.native-mouth-clips-result/120",
        "status": "complete",
        "batch": batch,
        "sources": 6,
        "planned_cases": 36,
        "ready_cases": sum(r["status"] == "ready" for r in cases),
        "unavailable_cases": sum(r["status"] != "ready" for r in cases),
        "by_condition": {
            c: dict(Counter(r["status"] for r in cases if r["condition"] == c)) for c in CONDITIONS
        },
        "face_detection_inputs": face_inputs,
        "render_calls": render_calls,
        "bitwise_render_repeat": repeat,
        "native_module_forward_counts": counts,
        "new_neural_calls": face_inputs + sum(counts.values()),
        "cases_sha256": sha(output / "cases.jsonl"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "defaults_changed": False,
        "scope": "Native known-provenance generation and exact lossless video/PCM/timing verification only. Construction unavailability is not an original-file verdict. No detector response, fitting, achieved lip synchronization or accuracy qualification.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "extract"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze has no arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One fixed batch required")
        run(int(args.arguments[0]))
    else:
        verify()
        require(len(args.arguments) == 2, "Exact fixed output and batch required")
        output = Path(args.arguments[0])
        batch = int(args.arguments[1])
        require(
            0 <= batch < 8 and output.resolve().is_relative_to(TRIAL.resolve()),
            "Exclusive private stage required",
        )
        output.mkdir(exist_ok=False)
        build(batch, output)
