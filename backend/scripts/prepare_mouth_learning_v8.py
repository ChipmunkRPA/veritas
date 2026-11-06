"""Extend immutable v6 stills with source-clock-bound second-family examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.detectors.audio.decode import decode_audio
from app.detectors.image.adapters.loaders import load_yunet
from app.detectors.image.faces import detect_faces
from app.detectors.video.decode import decode_frames, probe_frames
from app.engine.fetchers import ensure_verified
from app.engine.registry import get_registry
from scripts.liveportrait_mouth_benchmark import digest
from scripts.musetalk_benchmark_v7 import ASSET_MANIFEST_SHA, load, tensor_sha
from scripts.prepare_mouth_edit_v4 import composite
from scripts.prepare_mouth_multifamily_v7 import CONDITIONS
from scripts.prepare_mouth_training_v6 import save_rgb, selected_indices
from scripts.smoke_mouth_generator_v7 import alpha_from_face, crop_from_face

POLICY = "chipmunk.mouth-learning-preparation/8"


def code_pins():
    root = Path(__file__).resolve().parents[2]
    paths = [
        "backend/scripts/prepare_mouth_learning_v8.py",
        "backend/scripts/musetalk_benchmark_v7.py",
        "backend/scripts/smoke_mouth_generator_v7.py",
        "backend/scripts/prepare_mouth_training_v6.py",
        "backend/scripts/prepare_mouth_multifamily_v7.py",
        "backend/scripts/prepare_mouth_edit_v4.py",
        "backend/app/detectors/audio/decode.py",
        "backend/app/detectors/image/faces.py",
        "backend/app/detectors/video/decode.py",
        "docs/mouth-learning-v8-development.md",
    ]
    return {p: digest(root / p) for p in paths}


def validate_parent(recipe, rows):
    sources = recipe["sources"]
    if len(sources) != 20 or len({s["group"] for s in sources}) != 20:
        raise ValueError("Exactly the twenty existing source actors are required")
    expected = []
    for source in sources:
        start, indices = selected_indices(source["frames"])
        if start != source["edit_start"] or list(indices) != source["selected_frames"]:
            raise ValueError("Source frame selection changed")
        for i in indices:
            for c in ("original", "local-light", "local-resample", "ai-redraw", "ai-retime"):
                expected.append(
                    (f"{source['id']}:{i}:{c}", source["group"], source["role"], int(c.startswith("ai-")))
                )
    actual = [(r["id"], r["group"], r["role"], r["label"]) for r in rows]
    if actual != expected or any(r["status"] != "ready" for r in rows):
        raise ValueError("All immutable parent frames, labels and roles are required")
    if {s["group"] for s in sources} & set(recipe["excluded_development_groups"]):
        raise ValueError("Development actors overlap fitting roles")
    if [s["role"] for s in sources] != ["fit"] * 16 + ["point"] * 4:
        raise ValueError("Existing 16/4 actor roles must remain fixed")


def conditioning_index(time_s, mapping, available):
    blocks = mapping["timeline"]
    if len(blocks) != 1 or blocks[0]["start_sample"] != 0 or mapping["sample_rate"] != 16000:
        raise ValueError("A contiguous, source-bound 16 kHz audio clock is required")
    value = (time_s * 16000 - blocks[0]["source_pts"]) / 640
    index = round(value)
    if abs(value - index) > 1e-6 or not 8 <= index < available:
        raise ValueError("Picture time has no exact supported conditioning tick")
    return index


def plan(parent_plan, parent_manifest, output):
    recipe = json.loads(parent_plan.read_text())
    parent = json.loads(parent_manifest.read_text())
    if parent["plan_sha256"] != digest(parent_plan):
        raise ValueError("Parent manifest uses a different source plan")
    validate_parent(recipe, parent["items"])
    for row in parent["items"]:
        if digest(parent_manifest.parent / row["path"]) != row["sha256"]:
            raise ValueError("An immutable parent PNG changed")
    body = {
        "policy": POLICY,
        "parent_plan": str(parent_plan.resolve()),
        "parent_plan_sha256": digest(parent_plan),
        "parent_manifest": str(parent_manifest.resolve()),
        "parent_manifest_sha256": digest(parent_manifest),
        "source_root": recipe["source_root"],
        "source_manifest_sha256": recipe["source_manifest_sha256"],
        "sources": recipe["sources"],
        "excluded_development_groups": recipe["excluded_development_groups"],
        "conditions": CONDITIONS,
        "planned_frames": 800,
        "face_min_size": settings.face_min_size,
        "implementation_sha256": code_pins(),
        "asset_manifest_sha256": ASSET_MANIFEST_SHA,
        "deployment_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"plan_sha256": digest(output), "planned_additions": 800, "combined_stills": 1600}))


def build(plan_path, asset_manifest, assets, output):
    import torch

    recipe = json.loads(plan_path.read_text())
    if recipe["implementation_sha256"] != code_pins() or recipe["face_min_size"] != settings.face_min_size:
        raise ValueError("Frozen preparation implementation or face policy changed")
    for key in ("parent_plan", "parent_manifest"):
        if digest(Path(recipe[key])) != recipe[key + "_sha256"]:
            raise ValueError("Immutable parent changed")
    source_root = Path(recipe["source_root"])
    if digest(source_root / "manifest.json") != recipe["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    if digest(asset_manifest) != recipe["asset_manifest_sha256"]:
        raise ValueError("Generator asset plan changed")
    if not torch.backends.mps.is_available():
        raise ValueError("MPS generation required; no silent fallback")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(0.65)
    spec = get_registry().manifest.model("yunet.2023mar")
    if spec.licence.class_ != "commercial":
        raise ValueError("Permitted face geometry required")
    ensure_verified(spec, force=True)
    detector = load_yunet("yunet.2023mar", "cpu", "float32").obj
    renderer = load(asset_manifest, assets, "mps")
    records = []
    for source in recipe["sources"]:
        started = time.monotonic()
        folder = output / source["id"]
        folder.mkdir()
        planned = [
            {
                "id": f"{source['id']}:{i}:{c}",
                "group": source["group"],
                "role": source["role"],
                "condition": c,
                "label": int(c.startswith("muse-ai-")),
                "source_sha256": source["sha256"],
                "source_url": source["download_url"],
                "source_license": source["source_license"],
                "frame_index": i,
                "status": "unavailable",
            }
            for i in source["selected_frames"]
            for c in CONDITIONS
        ]
        try:
            path = source_root / source["path"]
            if digest(path) != source["sha256"]:
                raise ValueError("Original source bytes changed")
            table = probe_frames(path)
            frames = {f.index: f.rgb for f in decode_frames(path, table, source["selected_frames"])}
            if set(frames) != set(source["selected_frames"]):
                raise ValueError("A selected native frame is missing")
            audio = decode_audio(path, source_sha256=source["sha256"])
            chunks = renderer.conditioning(audio.pcm)
            for i in source["selected_frames"]:
                rgb = frames[i]
                faces = detect_faces(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=settings.face_min_size
                )
                if len(faces) != 1:
                    raise ValueError("Unique eligible source face required")
                crop, matrix, geometry = crop_from_face(rgb, faces[0], "landmark-box")
                inverse = cv2.invertAffineTransform(matrix)
                height, width = rgb.shape[:2]
                alpha = alpha_from_face(rgb.shape, faces[0])
                offset = i - source["edit_start"]
                opacity = min(1.0, (offset + 1) / 2, (20 - offset) / 2)
                tick = conditioning_index(table.seconds(i), audio.mapping, len(chunks))
                variants = {
                    "muse-resample": cv2.warpAffine(crop, inverse, (width, height), flags=cv2.INTER_LINEAR),
                    "local-blur-2": cv2.GaussianBlur(rgb, (0, 0), 2.0),
                    "local-blur-4": cv2.GaussianBlur(rgb, (0, 0), 4.0),
                }
                seed = int.from_bytes(
                    hashlib.sha256(f"chipmunk.mouth-learning/8:{source['id']}:{i}".encode()).digest()[:8],
                    "big",
                ) % (2**63)
                proofs = {}
                for c, k in (("muse-ai-aligned", tick), ("muse-ai-earlier", tick - 8)):
                    generated, proof = renderer.render(crop, chunks[k], seed)
                    variants[c] = cv2.warpAffine(generated, inverse, (width, height), flags=cv2.INTER_LINEAR)
                    proofs[c] = {
                        **proof,
                        "conditioning_frame": k,
                        "conditioning_f32_sha256": tensor_sha(chunks[k]),
                    }
                for row in [r for r in planned if r["frame_index"] == i]:
                    c = row["condition"]
                    native = composite(rgb, variants[c], alpha, opacity)
                    changed = np.any(native != rgb, axis=2)
                    if not changed.any() or not np.array_equal(native[alpha == 0], rgb[alpha == 0]):
                        raise ValueError("Empty edit or change outside support")
                    image_path = folder / f"{i:04d}-{c}.png"
                    commitment = save_rgb(image_path, native)
                    mask = folder / f"{i:04d}-{c}-mask.png"
                    if not cv2.imwrite(str(mask), changed.astype(np.uint8) * 255):
                        raise ValueError("Could not retain exact mask")
                    row.update(
                        status="ready",
                        path=str(image_path.relative_to(output)),
                        **commitment,
                        pts=table.pts[i],
                        time_base=table.time_base,
                        original_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        mask_path=str(mask.relative_to(output)),
                        mask_sha256=digest(mask),
                        changed_pixels=int(changed.sum()),
                        outside_support_exact=True,
                        source_crop_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                        source_to_crop_matrix_hex=[[float(v).hex() for v in r] for r in matrix],
                        geometry=geometry,
                        opacity=opacity,
                        model_proof=proofs.get(c),
                        aligned_conditioning_tick=tick,
                        audio_mapping=audio.mapping,
                    )
        except Exception as error:
            for row in planned:
                if row["status"] != "ready":
                    row["reason"] = f"{type(error).__name__}: {error}"
        records.extend(planned)
        with (output / "partial-receipts.jsonl").open("a") as stream:
            for row in planned:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "source": source["id"],
                    "role": source["role"],
                    "ready": sum(r["status"] == "ready" for r in planned),
                    "planned": 40,
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    body = {
        "policy": POLICY,
        "plan_sha256": digest(plan_path),
        "generator_assets_sha256": ASSET_MANIFEST_SHA,
        "generator_strict_load_audit": renderer.load_audit,
        "items": records,
        "deployment_eligible": False,
    }
    with (output / "manifest.json").open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "planned": len(records),
                "ready": sum(r["status"] == "ready" for r in records),
                "manifest_sha256": digest(output / "manifest.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "build"])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    if args.stage == "plan":
        if not args.parent_manifest:
            parser.error("plan requires --parent-manifest")
        plan(args.input, args.parent_manifest, args.output)
    else:
        if not args.asset_manifest or not args.assets:
            parser.error("build requires --asset-manifest and --assets")
        build(args.input, args.asset_manifest, args.assets, args.output)
