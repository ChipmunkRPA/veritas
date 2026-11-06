"""Source-clock-correct ordinary video calibration on the four reserved point actors."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.detectors.audio.decode import decode_audio, wav_bytes
from app.detectors.image.adapters.loaders import load_yunet
from app.detectors.image.faces import detect_faces
from app.detectors.video.decode import decode_frames, probe_frames
from app.engine.fetchers import ensure_verified
from app.engine.registry import get_registry
from scripts.liveportrait_mouth_benchmark import crop_from_mesh, digest
from scripts.prepare_mouth_edit_v4 import composite, encode, mouth_alpha
from scripts.smoke_mouth_generator_v7 import alpha_from_face, crop_from_face

POLICY = "chipmunk.mouth-context-ordinary-preparation/10"
CONDITIONS = ("original", "local-light", "local-resample", "muse-resample", "local-blur-2", "local-blur-4")
TASK_SHA = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def clock_span(table, audio):
    blocks = audio.mapping["timeline"]
    if len(blocks) != 1 or blocks[0]["start_sample"] != 0 or audio.mapping["sample_rate"] != 16000:
        raise ValueError("Contiguous source-bound mono16k clock required")
    offset = Fraction(table.pts[0]) * table.tb * 16000 - blocks[0]["source_pts"]
    if offset.denominator != 1 or offset < 0:
        raise ValueError("Picture origin has no exact supported PCM sample")
    offset = int(offset)
    n = min(len(table), (len(audio.pcm) - offset) // 640)
    if not 50 <= n <= 60 or any(
        (table.pts[i] - table.pts[0]) * table.tb != Fraction(i, 25) for i in range(n)
    ):
        raise ValueError("The fixed four sources require50–60 exact complete common25Hz frames")
    return {
        "frames": n,
        "source_video_frames": len(table),
        "source_frame_range": [0, n],
        "source_sample_range": [offset, offset + n * 640],
        "source_origin_s": table.seconds(0),
        "source_pts": list(table.pts[:n]),
        "source_time_base": table.time_base,
        "excluded_end_picture_frames": len(table) - n,
    }


def code_pins():
    root = Path(__file__).resolve().parents[2]
    names = [
        "backend/scripts/prepare_mouth_context_v10.py",
        "backend/scripts/liveportrait_mouth_benchmark.py",
        "backend/scripts/prepare_mouth_edit_v4.py",
        "backend/scripts/smoke_mouth_generator_v7.py",
        "backend/app/detectors/image/faces.py",
        "backend/app/detectors/audio/decode.py",
        "backend/app/detectors/video/decode.py",
        "docs/mouth-context-v10-development.md",
    ]
    return {p: digest(root / p) for p in names}


def plan(source_plan, task, output):
    recipe = json.loads(source_plan.read_text())
    if digest(task) != TASK_SHA:
        raise ValueError("Permitted MediaPipe task changed")
    selected = [r for r in recipe["sources"] if r["role"] == "point"]
    if [r["group"] for r in selected] != ["CREMA-D:1043", "CREMA-D:1075", "CREMA-D:1086", "CREMA-D:1083"]:
        raise ValueError("Keep all four original point actors and roles")
    source_root = Path(recipe["source_root"])
    for row in selected:
        path = source_root / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("Reserved source bytes changed")
        row["clock_span"] = clock_span(probe_frames(path), decode_audio(path, source_sha256=row["sha256"]))
    body = {
        "policy": POLICY,
        "source_plan": str(source_plan.resolve()),
        "source_plan_sha256": digest(source_plan),
        "source_root": str(source_root.resolve()),
        "source_manifest_sha256": digest(source_root / "manifest.json"),
        "task": str(task.resolve()),
        "task_sha256": TASK_SHA,
        "sources": selected,
        "conditions": CONDITIONS,
        "planned_cases": 24,
        "implementation_sha256": code_pins(),
        "face_min_size": settings.face_min_size,
        "deployment_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "plan_sha256": digest(output),
                "planned_cases": 24,
                "common_frames": [r["clock_span"]["frames"] for r in selected],
            }
        )
    )


def operations(rgb, mesh, face):
    crop, matrix, _ = crop_from_mesh(rgb, mesh)
    height, width = rgb.shape[:2]
    portrait_alpha = mouth_alpha(rgb.shape, mesh)
    speech_crop, speech_matrix, geometry = crop_from_face(rgb, face, "landmark-box")
    speech_alpha = alpha_from_face(rgb.shape, face)
    values = {
        "local-light": (np.rint(rgb.astype(np.float32) * 1.12).clip(0, 255).astype(np.uint8), portrait_alpha),
        "local-resample": (
            cv2.warpAffine(
                cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR),
                cv2.invertAffineTransform(matrix * 2),
                (width, height),
                flags=cv2.INTER_LINEAR,
            ),
            portrait_alpha,
        ),
        "muse-resample": (
            cv2.warpAffine(
                speech_crop, cv2.invertAffineTransform(speech_matrix), (width, height), flags=cv2.INTER_LINEAR
            ),
            speech_alpha,
        ),
        "local-blur-2": (cv2.GaussianBlur(rgb, (0, 0), 2.0), speech_alpha),
        "local-blur-4": (cv2.GaussianBlur(rgb, (0, 0), 4.0), speech_alpha),
    }
    binding = {
        "portrait_source_to_crop_matrix_hex": [[float(v).hex() for v in r] for r in matrix],
        "speech_source_to_crop_matrix_hex": [[float(v).hex() for v in r] for r in speech_matrix],
        "portrait_crop_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
        "speech_crop_rgb_sha256": hashlib.sha256(speech_crop.tobytes()).hexdigest(),
        "speech_geometry": geometry,
    }
    return values, binding


def geometry_frames(frames, table, task):
    import mediapipe as mp

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(task), delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=2,
    )
    mesh = {}
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for i, rgb in enumerate(frames):
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(table.seconds(i) * 1000)
            )
            if len(result.face_landmarks) == 1:
                mesh[i] = result.face_landmarks[0]
    return mesh


def build(plan_path, output):
    recipe = json.loads(plan_path.read_text())
    task = Path(recipe["task"])
    source_root = Path(recipe["source_root"])
    if (
        recipe["implementation_sha256"] != code_pins()
        or digest(task) != TASK_SHA
        or digest(source_root / "manifest.json") != recipe["source_manifest_sha256"]
        or settings.face_min_size != recipe["face_min_size"]
    ):
        raise ValueError("Frozen source construction/task/face policy changed")
    output.mkdir(parents=True, exist_ok=False)
    spec = get_registry().manifest.model("yunet.2023mar")
    if spec.licence.class_ != "commercial":
        raise ValueError("Permitted face geometry required")
    ensure_verified(spec, force=True)
    detector = load_yunet("yunet.2023mar", "cpu", "float32").obj
    records = []
    for source in recipe["sources"]:
        started = time.monotonic()
        folder = output / source["id"]
        folder.mkdir()
        planned = [
            {
                "id": source["id"] + ":v10:" + c,
                "group": source["group"],
                "role": "point",
                "condition": c,
                "label": 0,
                "status": "unavailable",
                "source_sha256": source["sha256"],
                "source_url": source["download_url"],
                "source_license": source["source_license"],
            }
            for c in CONDITIONS
        ]
        try:
            path = source_root / source["path"]
            if digest(path) != source["sha256"]:
                raise ValueError("Reserved source file changed")
            table = probe_frames(path)
            audio = decode_audio(path, source_sha256=source["sha256"])
            span = clock_span(table, audio)
            if span != source["clock_span"]:
                raise ValueError("Declared common source clocks changed")
            n = span["frames"]
            frames = [f.rgb for f in decode_frames(path, table, range(n))]
            if len(frames) != n:
                raise ValueError("Missing retained source picture")
            mesh = geometry_frames(frames, table, task)
            start, end = (n - 20) // 2, (n - 20) // 2 + 20
            if any(i not in mesh for i in range(start, end)):
                raise ValueError("A planned operation lacks unique original mesh geometry")
            sound_path = folder / "source-common.wav"
            a, b = span["source_sample_range"]
            sound_path.write_bytes(wav_bytes(audio.pcm[a:b]))
            treatments = {c: list(frames) for c in CONDITIONS}
            masks = {c: [] for c in CONDITIONS}
            for i in range(start, end):
                rgb = frames[i]
                faces = detect_faces(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=settings.face_min_size
                )
                if len(faces) != 1:
                    raise ValueError("A planned operation lacks unique source face")
                variants, geometry = operations(rgb, mesh[i], faces[0])
                opacity = min(1.0, (i - start + 1) / 2, (end - i) / 2)
                for c in CONDITIONS:
                    native = (
                        rgb if c == "original" else composite(rgb, variants[c][0], variants[c][1], opacity)
                    )
                    changed = np.any(native != rgb, axis=2)
                    if bool(changed.any()) != (c != "original"):
                        raise ValueError("Empty ordinary edit or changed original")
                    mask_path = folder / f"{c}-mask-{i:04d}.png"
                    if not cv2.imwrite(str(mask_path), changed.astype(np.uint8) * 255):
                        raise ValueError("Could not record exact calibration mask")
                    masks[c].append(
                        {
                            "frame_index": i,
                            "pts": table.pts[i],
                            "source_time_s": table.seconds(i),
                            "path": str(mask_path.relative_to(output)),
                            "sha256": digest(mask_path),
                            "source_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                            "changed_pixels": int(changed.sum()),
                            "opacity": opacity,
                            **geometry,
                        }
                    )
                    treatments[c][i] = native
            for case in planned:
                c = case["condition"]
                target = folder / (c + ".mkv")
                case.update(
                    status="ready",
                    path=str(target.relative_to(output)),
                    **encode(treatments[c], sound_path, target),
                    edit_frame_range=[start, end],
                    edit_time_range_s=[start / 25, end / 25],
                    masks=masks[c],
                    source_ranges={"before": [0, start], "during": [start, end], "after": [end, n]},
                    original_rgb_sha256=[hashlib.sha256(f.tobytes()).hexdigest() for f in frames],
                    source_clock_span=span,
                    original_audio_mapping=audio.mapping,
                )
        except Exception as error:
            for case in planned:
                if case["status"] != "ready":
                    case["reason"] = f"{type(error).__name__}: {error}"
        records.extend(planned)
        with (output / "partial-receipts.jsonl").open("a") as stream:
            for case in planned:
                stream.write(json.dumps(case, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "source": source["id"],
                    "planned": 6,
                    "ready": sum(r["status"] == "ready" for r in planned),
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    with (output / "manifest.json").open("x") as stream:
        json.dump(
            {
                "policy": POLICY,
                "plan_sha256": digest(plan_path),
                "items": records,
                "deployment_eligible": False,
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    print(
        json.dumps(
            {
                "planned": 24,
                "ready": sum(r["status"] == "ready" for r in records),
                "manifest_sha256": digest(output / "manifest.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["plan", "build"])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--task", type=Path)
    a = p.parse_args()
    if a.stage == "plan" and a.task:
        plan(a.input, a.task, a.output)
    elif a.stage == "build":
        build(a.input, a.output)
    else:
        p.error("plan requires --task")
