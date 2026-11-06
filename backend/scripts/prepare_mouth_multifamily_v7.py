"""Known second-family neural mouth edits and matched ordinary controls; research only."""

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
from app.detectors.audio.decode import decode_audio, wav_bytes
from app.detectors.image.adapters.loaders import load_yunet
from app.detectors.image.faces import detect_faces
from app.detectors.video.decode import decode_frames, probe_frames
from app.engine.fetchers import ensure_verified
from app.engine.registry import get_registry
from scripts.liveportrait_mouth_benchmark import digest
from scripts.musetalk_benchmark_v7 import ASSET_MANIFEST_SHA, load, tensor_sha
from scripts.prepare_mouth_edit_v4 import composite, encode, interval
from scripts.smoke_mouth_generator_v7 import alpha_from_face, crop_from_face

POLICY = "chipmunk.mouth-multifamily-extension/7"
CONDITIONS = ("muse-resample", "local-blur-2", "local-blur-4", "muse-ai-aligned", "muse-ai-earlier")


def code_pins():
    root = Path(__file__).resolve().parents[2]
    paths = [
        "backend/scripts/prepare_mouth_multifamily_v7.py",
        "backend/scripts/musetalk_benchmark_v7.py",
        "backend/scripts/smoke_mouth_generator_v7.py",
        "backend/scripts/fetch_mouth_generator_v7.py",
        "backend/scripts/prepare_mouth_edit_v4.py",
        "backend/app/detectors/image/faces.py",
        "backend/app/detectors/image/gend.py",
        "backend/app/detectors/video/decode.py",
        "docs/mouth-multifamily-v7-development.md",
        "docs/evaluation-panels/mouth-v7-generator-assets.json",
    ]
    return {p: digest(root / p) for p in paths}


def plan(parent, output):
    data = json.loads(parent.read_text())
    sources = [row for row in data["items"] if row["condition"] == "original"]
    if (
        len(sources) != 8
        or len({r["group"] for r in sources}) != 8
        or any(r["status"] != "ready" for r in sources)
    ):
        raise ValueError("All eight existing consumed original groups are required")
    for source in sources:
        if digest(parent.parent / source["path"]) != source["sha256"]:
            raise ValueError("Existing lossless original changed")
        interval(source["frames"])
    body = {
        "policy": POLICY,
        "role": "Consumed-source second-family development extension; not fresh confirmation",
        "parent_manifest": str(parent.resolve()),
        "parent_manifest_sha256": digest(parent),
        "sources": sources,
        "conditions": CONDITIONS,
        "planned_cases": 40,
        "face_min_size": settings.face_min_size,
        "implementation_sha256": code_pins(),
        "asset_manifest_sha256": ASSET_MANIFEST_SHA,
        "deployment_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"plan_sha256": digest(output), "source_groups": 8, "planned_cases": 40}))


def build(plan_path, asset_manifest, assets, output):
    recipe = json.loads(plan_path.read_text())
    if (
        recipe["implementation_sha256"] != code_pins()
        or recipe["asset_manifest_sha256"] != digest(asset_manifest)
        or recipe["face_min_size"] != settings.face_min_size
    ):
        raise ValueError("Frozen generation implementation, assets or face policy changed")
    parent = Path(recipe["parent_manifest"])
    if digest(parent) != recipe["parent_manifest_sha256"]:
        raise ValueError("Parent input manifest changed")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise ValueError("Frozen generator requires MPS; no silent fallback")
    torch.mps.set_per_process_memory_fraction(0.65)
    spec = get_registry().manifest.model("yunet.2023mar")
    if spec.licence.class_ != "commercial":
        raise ValueError("Commercial face geometry required")
    ensure_verified(spec, force=True)
    detector = load_yunet("yunet.2023mar", "cpu", "float32").obj
    renderer = load(asset_manifest, assets, "mps")
    results = []
    for source in recipe["sources"]:
        started = time.monotonic()
        prefix = source["id"].rsplit(":", 1)[0]
        folder = output / prefix
        folder.mkdir()
        planned = [
            {
                "id": prefix + ":" + c,
                "group": source["group"],
                "condition": c,
                "label": int(c.startswith("muse-ai-")),
                "source_sha256": source["sha256"],
                "upstream_normalized_source_sha256": source["source_sha256"],
                "source_url": source["source_url"],
                "source_license": source["source_license"],
                "status": "unavailable",
            }
            for c in CONDITIONS
        ]
        try:
            path = parent.parent / source["path"]
            if digest(path) != source["sha256"]:
                raise ValueError("Parent original bytes changed")
            table = probe_frames(path)
            n = source["frames"]
            if len(table) != n or any(abs(table.seconds(i) - i / 25) > 1e-9 for i in range(n)):
                raise ValueError("Parent is not the exact complete 25 Hz clock")
            frames = [f.rgb for f in decode_frames(path, table, range(n))]
            if len(frames) != n:
                raise ValueError("A planned source frame is unavailable")
            start, end = interval(n)
            sound = decode_audio(path, source_sha256=source["sha256"])
            if len(sound.pcm) != n * 640:
                raise ValueError("Parent audio does not share the exact picture span")
            audio = folder / "source.wav"
            audio.write_bytes(wav_bytes(sound.pcm))
            chunks = renderer.conditioning(sound.pcm)
            if len(chunks) != n:
                raise ValueError("Speech conditioning frame clock differs")
            treatments = {c: list(frames) for c in CONDITIONS}
            masks = {c: [] for c in CONDITIONS}
            for i in range(start, end):
                faces = detect_faces(
                    cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR),
                    detector=detector,
                    min_size=settings.face_min_size,
                )
                if len(faces) != 1:
                    raise ValueError("A planned edit frame lacks unique eligible face geometry")
                crop, matrix, geometry = crop_from_face(frames[i], faces[0], "landmark-box")
                inverse = cv2.invertAffineTransform(matrix)
                height, width = frames[i].shape[:2]
                alpha = alpha_from_face(frames[i].shape, faces[0])
                opacity = min(1.0, (i - start + 1) / 2, (end - i) / 2)
                variants = {
                    "muse-resample": cv2.warpAffine(crop, inverse, (width, height), flags=cv2.INTER_LINEAR),
                    "local-blur-2": cv2.GaussianBlur(frames[i], (0, 0), 2.0),
                    "local-blur-4": cv2.GaussianBlur(frames[i], (0, 0), 4.0),
                }
                proofs = {}
                seed = int.from_bytes(
                    hashlib.sha256(f"chipmunk.mouth-multifamily/7:{prefix}:{i}".encode()).digest()[:8], "big"
                ) % (2**63)
                for c, k in (("muse-ai-aligned", i), ("muse-ai-earlier", i - 8)):
                    generated, proof = renderer.render(crop, chunks[k], seed)
                    variants[c] = cv2.warpAffine(generated, inverse, (width, height), flags=cv2.INTER_LINEAR)
                    proofs[c] = {
                        **proof,
                        "conditioning_frame": k,
                        "conditioning_f32_sha256": tensor_sha(chunks[k]),
                    }
                for c, transformed in variants.items():
                    edited = composite(frames[i], transformed, alpha, opacity)
                    changed = np.any(edited != frames[i], axis=2)
                    if not changed.any():
                        raise ValueError("A planned edited frame has zero changed pixels")
                    name = f"{c}-mask-{i:04d}.png"
                    mask_path = folder / name
                    if not cv2.imwrite(str(mask_path), changed.astype(np.uint8) * 255):
                        raise ValueError("Could not retain exact edit mask")
                    masks[c].append(
                        {
                            "frame_index": i,
                            "pts": table.pts[i],
                            "path": str(mask_path.relative_to(output)),
                            "sha256": digest(mask_path),
                            "changed_pixels": int(changed.sum()),
                            "source_rgb_sha256": hashlib.sha256(frames[i].tobytes()).hexdigest(),
                            "source_crop_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
                            "source_to_crop_matrix_hex": [[float(v).hex() for v in row] for row in matrix],
                            "geometry": geometry,
                            "model_proof": proofs.get(c),
                        }
                    )
                    treatments[c][i] = edited
            for case in planned:
                c = case["condition"]
                target = folder / (c + ".mkv")
                case.update(
                    status="ready",
                    path=str(target.relative_to(output)),
                    **encode(treatments[c], audio, target),
                    edit_frame_range=[start, end],
                    edit_time_range_s=[start / 25, end / 25],
                    masks=masks[c],
                    source_ranges={"before": [0, start], "during": [start, end], "after": [end, n]},
                    original_rgb_sha256=[hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames],
                )
        except Exception as error:
            for case in planned:
                if case["status"] != "ready":
                    case["reason"] = f"{type(error).__name__}: {error}"
        results.extend(planned)
        with (output / "partial-receipts.jsonl").open("a") as stream:
            for case in planned:
                stream.write(json.dumps(case, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "source": source["id"],
                    "seconds": time.monotonic() - started,
                    "ready": sum(r["status"] == "ready" for r in planned),
                    "planned": len(planned),
                }
            ),
            flush=True,
        )
    body = {
        "policy": POLICY,
        "plan_sha256": digest(plan_path),
        "parent_manifest_sha256": digest(parent),
        "generator_assets_sha256": ASSET_MANIFEST_SHA,
        "generator_strict_load_audit": renderer.load_audit,
        "libraries": {
            **{
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "diffusers", "huggingface_hub")
            },
            "opencv": cv2.__version__,
        },
        "generator_device": "mps",
        "items": results,
        "role": recipe["role"],
        "deployment_eligible": False,
    }
    with (output / "manifest.json").open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "planned": len(results),
                "ready": sum(r["status"] == "ready" for r in results),
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
    p.add_argument("--asset-manifest", type=Path)
    p.add_argument("--assets", type=Path)
    a = p.parse_args()
    if a.stage == "plan":
        plan(a.input, a.output)
    else:
        if not a.asset_manifest or not a.assets:
            p.error("build requires --asset-manifest and --assets")
        build(a.input, a.asset_manifest, a.assets, a.output)
