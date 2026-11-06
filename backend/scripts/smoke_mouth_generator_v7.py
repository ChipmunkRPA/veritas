"""Bounded deterministic generator smoke on one already-consumed source frame."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from app.core.config import settings
from app.detectors.audio.decode import decode_audio
from app.detectors.image import gend
from app.detectors.image.adapters.loaders import load_yunet
from app.detectors.image.faces import detect_faces
from app.detectors.video.decode import decode_frames, probe_frames
from app.engine.fetchers import ensure_verified
from app.engine.registry import get_registry
from scripts.liveportrait_mouth_benchmark import digest
from scripts.musetalk_benchmark_v7 import load, tensor_sha
from scripts.prepare_mouth_edit_v4 import composite


def alpha_from_face(shape, face):
    height, width = shape[:2]
    a, b = np.asarray(face.landmarks[3:5], np.float64)
    axis = b - a
    mouth_width = float(np.linalg.norm(axis))
    if mouth_width < 8:
        raise ValueError("Insufficient source mouth support")
    axis /= mouth_width
    normal = np.array([-axis[1], axis[0]])
    center = (a + b) / 2
    yy, xx = np.indices((height, width))
    dx, dy = xx - center[0], yy - center[1]
    radius = np.sqrt(
        ((dx * axis[0] + dy * axis[1]) / (0.7 * mouth_width)) ** 2
        + ((dx * normal[0] + dy * normal[1]) / (0.45 * mouth_width)) ** 2
    )
    return np.clip((1 - radius) / 0.2, 0, 1).astype(np.float32)


def crop_from_face(rgb, face, mode="landmark-box"):
    if mode == "aligned":
        matrix, side = gend.align_transform(face.landmarks)
        matrix = matrix * (256 / side)
        crop = cv2.warpAffine(
            rgb, matrix, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        return crop, matrix, {"mode": mode}
    if mode != "landmark-box":
        raise ValueError("Unknown benchmark geometry rule")
    eyes = np.asarray(face.landmarks[:2], np.float64).mean(axis=0)
    nose = np.asarray(face.landmarks[2], np.float64)
    half_y = float(eyes[1] + 0.75 * (nose[1] - eyes[1]))
    height, width = rgb.shape[:2]
    x0, y0 = max(0, int(face.x)), max(0, int(np.floor(2 * half_y - (face.y + face.h))))
    x1, y1 = min(width, int(face.x + face.w)), min(height, int(face.y + face.h))
    if min(x1 - x0, y1 - y0) < 64:
        raise ValueError("Insufficient native source for the permitted face-box crop")
    sx, sy = 256 / (x1 - x0), 256 / (y1 - y0)
    matrix = np.array([[sx, 0, (0.5 - x0) * sx - 0.5], [0, sy, (0.5 - y0) * sy - 0.5]], np.float64)
    crop = cv2.resize(rgb[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_LANCZOS4)
    return (
        crop,
        matrix,
        {
            "mode": mode,
            "bbox": [x0, y0, x1, y1],
            "approximate_half_face_y": half_y,
            "interpretation": "YuNet five-point approximation, not measured 68-point landmark 29",
        },
    )


def smoke(manifest, assets, source_plan, output, device, crop_mode):
    recipe = json.loads(source_plan.read_text())
    source = next(r for r in recipe["sources"] if r["role"] == "fit")
    path = Path(recipe["source_root"]) / source["path"]
    if digest(path) != source["sha256"]:
        raise ValueError("Consumed smoke source changed")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.65)
    ensure_verified(get_registry().manifest.model("yunet.2023mar"), force=True)
    detector = load_yunet("yunet.2023mar", "cpu", "float32").obj
    table = probe_frames(path)
    index = source["selected_frames"][len(source["selected_frames"]) // 2]
    frame = next(decode_frames(path, table, [index]))
    faces = detect_faces(
        cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=settings.face_min_size
    )
    transform = gend.align_transform(faces[0].landmarks) if faces else None
    if transform is None:
        raise ValueError("Consumed smoke input has no measurable face")
    crop, matrix, geometry = crop_from_face(frame.rgb, faces[0], crop_mode)
    cv2.imwrite(str(output / "source.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / "crop.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    started = time.monotonic()
    renderer = load(manifest, assets, device)
    print(
        json.dumps({"stage": "loaded", "seconds": time.monotonic() - started, "device": device}), flush=True
    )
    audio = decode_audio(path, source_sha256=source["sha256"])
    pcm = audio.pcm[: source["frames"] * 640]
    chunks = renderer.conditioning(pcm)
    if len(chunks) != source["frames"]:
        raise ValueError("Speech conditioning does not share complete source frame bounds")
    alpha = alpha_from_face(frame.rgb.shape, faces[0])
    records, images = [], []
    for tag, i in (("aligned", index), ("repeat", index), ("earlier", index - 8)):
        started = time.monotonic()
        rgb, proof = renderer.render(crop, chunks[i], 717)
        images.append(rgb)
        native = cv2.warpAffine(
            rgb, cv2.invertAffineTransform(matrix), (table.width, table.height), flags=cv2.INTER_LINEAR
        )
        edited = composite(frame.rgb, native, alpha)
        changed = np.any(edited != frame.rgb, axis=2)
        if not changed.any() or np.any(changed[alpha == 0]):
            raise ValueError("Benchmark edit is empty or changes outside its measured support")
        cv2.imwrite(str(output / (tag + "-crop.png")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output / (tag + "-native.png")), cv2.cvtColor(edited, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output / (tag + "-mask.png")), changed.astype(np.uint8) * 255)
        record = {
            "tag": tag,
            "conditioning_frame": i,
            "conditioning_f32_sha256": tensor_sha(chunks[i]),
            "proof": proof,
            "seconds": time.monotonic() - started,
            "changed_native_pixels": int(changed.sum()),
            "native_rgb_sha256": hashlib.sha256(edited.tobytes()).hexdigest(),
        }
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "proof"}), flush=True)
    exact_repeat = np.array_equal(images[0], images[1])
    changed_with_audio = int(np.count_nonzero(np.any(images[0] != images[2], axis=2)))
    if not exact_repeat or changed_with_audio == 0:
        raise ValueError("Seeded repeat is not exact or changed conditioning has no pixel effect")
    body = {
        "purpose": __doc__,
        "device": device,
        "source_id": source["id"],
        "source_sha256": source["sha256"],
        "frame_index": index,
        "pts": frame.pts,
        "time_base": table.time_base,
        "native_rgb_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
        "alignment_matrix_hex": [[float(x).hex() for x in row] for row in matrix],
        "aligned_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
        "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
        "seeded_repeat_exact": exact_repeat,
        "audio_conditioning_changed_crop_pixels": changed_with_audio,
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "diffusers", "huggingface_hub")
        },
        "records": records,
        "accuracy_or_deployment_validation": False,
        "strict_load_audit": renderer.load_audit,
        "geometry": geometry,
        "implementation_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("musetalk_benchmark_v7.py", "smoke_mouth_generator_v7.py")
        },
    }
    with (output / "receipt.json").open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "passed": True,
                "seeded_repeat_exact": exact_repeat,
                "audio_conditioning_changed_crop_pixels": changed_with_audio,
                "role": "Engineering smoke, not detector validation",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("assets", type=Path)
    p.add_argument("source_plan", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    p.add_argument("--crop", choices=["aligned", "landmark-box"], default="landmark-box")
    a = p.parse_args()
    smoke(a.manifest, a.assets, a.source_plan, a.output, a.device, a.crop)
