"""Prepare complete, clock-bound fitting videos for temporal discrimination."""

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
from scripts.liveportrait_mouth_benchmark import load as load_portrait
from scripts.musetalk_benchmark_v7 import ASSET_MANIFEST_SHA, tensor_sha
from scripts.musetalk_benchmark_v7 import load as load_speech
from scripts.prepare_mouth_context_v10 import TASK_SHA, geometry_frames, operations
from scripts.prepare_mouth_edit_v4 import composite, encode, mouth_alpha
from scripts.prepare_mouth_learning_v8 import conditioning_index
from scripts.smoke_mouth_generator_v7 import alpha_from_face, crop_from_face

POLICY = "chipmunk.mouth-temporal-preparation/11"
CONDITIONS = (
    "original",
    "local-light",
    "local-resample",
    "muse-resample",
    "local-blur-2",
    "local-blur-4",
    "ai-redraw",
    "ai-retime",
    "muse-ai-aligned",
    "muse-ai-earlier",
)
AI_CONDITIONS = frozenset(CONDITIONS[6:])


def seed_for(identifier, index):
    return int.from_bytes(
        hashlib.sha256(f"chipmunk.mouth-temporal/11:{identifier}:{index}".encode()).digest()[:8], "big"
    ) % (2**63)


def clock_span(table, audio):
    blocks = audio.mapping["timeline"]
    if len(blocks) != 1 or blocks[0]["start_sample"] != 0 or audio.mapping["sample_rate"] != 16000:
        raise ValueError("Contiguous source-bound mono16k clock required")
    offset = Fraction(table.pts[0]) * table.tb * 16000 - blocks[0]["source_pts"]
    if offset.denominator != 1 or offset < 0:
        raise ValueError("Picture origin has no exact PCM sample")
    offset = int(offset)
    n = min(len(table), (len(audio.pcm) - offset) // 640)
    if not 51 <= n <= 71 or any(
        (table.pts[i] - table.pts[0]) * table.tb != Fraction(i, 25) for i in range(n)
    ):
        raise ValueError("All fixed fitting sources require51–71 complete25Hz frames")
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
    from scripts.prepare_mouth_context_v10 import code_pins as parent

    root = Path(__file__).resolve().parents[2]
    names = [
        "backend/scripts/prepare_mouth_temporal_v11.py",
        "backend/scripts/musetalk_benchmark_v7.py",
        "backend/scripts/prepare_mouth_learning_v8.py",
        "docs/mouth-temporal-v11-development.md",
    ]
    return {**parent(), **{p: digest(root / p) for p in names}}


def plan(source_plan, asset_manifest, portrait_root, output):
    recipe = json.loads(source_plan.read_text())
    selected = [dict(s) for s in recipe["sources"] if s["role"] == "fit"]
    others = {s["group"] for s in recipe["sources"] if s["role"] != "fit"} | set(
        recipe["excluded_development_groups"]
    )
    if (
        len(selected) != 16
        or len({s["group"] for s in selected}) != 16
        or any(s["group"] in others for s in selected)
    ):
        raise ValueError("Keep the sixteen original disjoint fitting actors")
    task = portrait_root / "permissions/face_landmarker.task"
    if digest(task) != TASK_SHA or digest(asset_manifest) != ASSET_MANIFEST_SHA:
        raise ValueError("Permitted generator/task declarations changed")
    source_root = Path(recipe["source_root"])
    for s in selected:
        path = source_root / s["path"]
        if digest(path) != s["sha256"]:
            raise ValueError("Original fitting source changed")
        s["clock_span"] = clock_span(probe_frames(path), decode_audio(path, source_sha256=s["sha256"]))
    body = {
        "policy": POLICY,
        "source_plan": str(source_plan.resolve()),
        "source_plan_sha256": digest(source_plan),
        "source_root": str(source_root),
        "source_manifest_sha256": digest(source_root / "manifest.json"),
        "sources": selected,
        "excluded_groups": sorted(others),
        "conditions": CONDITIONS,
        "planned_cases": 160,
        "task": str(task.resolve()),
        "task_sha256": TASK_SHA,
        "portrait_root": str(portrait_root.resolve()),
        "portrait_assets_sha256": digest(portrait_root / "permissions/liveportrait-verified-assets.json"),
        "speech_asset_manifest": str(asset_manifest.resolve()),
        "speech_asset_manifest_sha256": ASSET_MANIFEST_SHA,
        "implementation_sha256": code_pins(),
        "face_min_size": settings.face_min_size,
        "deployment_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, body)
    print(
        json.dumps(
            {
                "plan_sha256": digest(output),
                "planned_cases": 160,
                "common_frames": [s["clock_span"]["frames"] for s in selected],
            }
        ),
        flush=True,
    )


def write(path, body):
    with path.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")


def build(plan_path, speech_assets, output):
    import torch

    recipe = json.loads(plan_path.read_text())
    if recipe["implementation_sha256"] != code_pins() or recipe["face_min_size"] != settings.face_min_size:
        raise ValueError("Frozen preparation implementation/face policy changed")
    for k in ("source_plan", "task", "speech_asset_manifest"):
        if digest(Path(recipe[k])) != recipe[k + "_sha256"]:
            raise ValueError("Frozen input changed: " + k)
    source_root = Path(recipe["source_root"])
    if digest(source_root / "manifest.json") != recipe["source_manifest_sha256"]:
        raise ValueError("Consumed source manifest changed")
    portrait_root = Path(recipe["portrait_root"])
    if (
        digest(portrait_root / "permissions/liveportrait-verified-assets.json")
        != recipe["portrait_assets_sha256"]
    ):
        raise ValueError("Portrait permissions/assets changed")
    if not torch.backends.mps.is_available():
        raise ValueError("MPS required; no silent device fallback")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(0.65)
    spec = get_registry().manifest.model("yunet.2023mar")
    if spec.licence.class_ != "commercial":
        raise ValueError("Permitted geometry weights required")
    ensure_verified(spec, force=True)
    face_detector = load_yunet("yunet.2023mar", "cpu", "float32").obj
    portrait = load_portrait(portrait_root, "mps")
    speech = load_speech(Path(recipe["speech_asset_manifest"]), speech_assets, "mps")
    records = []
    for source in recipe["sources"]:
        started = time.monotonic()
        folder = output / source["id"]
        folder.mkdir()
        cases = [
            {
                "id": source["id"] + ":v11:" + c,
                "group": source["group"],
                "role": "fit",
                "condition": c,
                "label": int(c in AI_CONDITIONS),
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
                raise ValueError("Original source bytes changed")
            table = probe_frames(path)
            audio = decode_audio(path, source_sha256=source["sha256"])
            span = clock_span(table, audio)
            if span != source["clock_span"]:
                raise ValueError("Declared common clocks changed")
            n = span["frames"]
            frames = [f.rgb for f in decode_frames(path, table, range(n))]
            if len(frames) != n:
                raise ValueError("Missing original picture")
            mesh = geometry_frames(frames, table, Path(recipe["task"]))
            start, end = (n - 20) // 2, (n - 20) // 2 + 20
            required = set(range(start - 8, end))
            if not required <= set(mesh):
                raise ValueError("Planned frame lacks unique original mesh")
            geom = {i: crop_from_mesh(frames[i], mesh[i]) for i in required}
            chunks = speech.conditioning(audio.pcm)
            a, b = span["source_sample_range"]
            sound_path = folder / "source-common.wav"
            sound_path.write_bytes(wav_bytes(audio.pcm[a:b]))
            treatments = {c: list(frames) for c in CONDITIONS}
            masks = {c: [] for c in CONDITIONS}
            for i in range(start, end):
                rgb = frames[i]
                faces = detect_faces(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    detector=face_detector,
                    min_size=settings.face_min_size,
                )
                if len(faces) != 1:
                    raise ValueError("Unique original source face required")
                variants, geometry = operations(rgb, mesh[i], faces[0])
                crop, matrix, ratio = geom[i]
                height, width = rgb.shape[:2]
                alpha = mouth_alpha(rgb.shape, mesh[i])
                proof = {}
                for c, target in (("ai-redraw", ratio), ("ai-retime", geom[i - 8][2])):
                    generated = portrait.render(crop, ratio, target)
                    variants[c] = (
                        cv2.warpAffine(
                            generated,
                            cv2.invertAffineTransform(matrix * 2),
                            (width, height),
                            flags=cv2.INTER_LINEAR,
                        ),
                        alpha,
                    )
                    proof[c] = {
                        "source_aperture": ratio,
                        "target_aperture": target,
                        "target_frame_index": i if c == "ai-redraw" else i - 8,
                        "generated_rgb_sha256": hashlib.sha256(generated.tobytes()).hexdigest(),
                    }
                scrop, smatrix, _ = crop_from_face(rgb, faces[0], "landmark-box")
                tick = conditioning_index(table.seconds(i), audio.mapping, len(chunks))
                seed = seed_for(source["id"], i)
                for c, k in (("muse-ai-aligned", tick), ("muse-ai-earlier", tick - 8)):
                    generated, p = speech.render(scrop, chunks[k], seed)
                    variants[c] = (
                        cv2.warpAffine(
                            generated,
                            cv2.invertAffineTransform(smatrix),
                            (width, height),
                            flags=cv2.INTER_LINEAR,
                        ),
                        alpha_from_face(rgb.shape, faces[0]),
                    )
                    proof[c] = {
                        **p,
                        "conditioning_frame": k,
                        "conditioning_f32_sha256": tensor_sha(chunks[k]),
                        "seed": seed,
                    }
                opacity = min(1.0, (i - start + 1) / 2, (end - i) / 2)
                for c in CONDITIONS:
                    native = rgb if c == "original" else composite(rgb, *variants[c], opacity)
                    changed = np.any(native != rgb, axis=2)
                    if bool(changed.any()) != (c != "original"):
                        raise ValueError("Empty edit or changed original")
                    mp = folder / f"{c}-mask-{i:04d}.png"
                    if not cv2.imwrite(str(mp), changed.astype(np.uint8) * 255):
                        raise ValueError("Could not retain exact mask")
                    masks[c].append(
                        {
                            "frame_index": i,
                            "pts": table.pts[i],
                            "source_time_s": table.seconds(i),
                            "path": str(mp.relative_to(output)),
                            "sha256": digest(mp),
                            "source_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                            "changed_pixels": int(changed.sum()),
                            "opacity": opacity,
                            **geometry,
                            "model_proof": proof.get(c),
                        }
                    )
                    treatments[c][i] = native
            for case in cases:
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
            for case in cases:
                if case["status"] != "ready":
                    case["reason"] = f"{type(error).__name__}: {error}"
        records.extend(cases)
        with (output / "partial-receipts.jsonl").open("a") as stream:
            for case in cases:
                stream.write(json.dumps(case, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "source": source["id"],
                    "planned": 10,
                    "ready": sum(c["status"] == "ready" for c in cases),
                    "reason": next((c.get("reason") for c in cases if c.get("reason")), None),
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    write(
        output / "manifest.json",
        {"policy": POLICY, "plan_sha256": digest(plan_path), "items": records, "deployment_eligible": False},
    )
    print(
        json.dumps(
            {
                "planned": 160,
                "ready": sum(c["status"] == "ready" for c in records),
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
    p.add_argument("--portrait-root", type=Path)
    p.add_argument("--speech-assets", type=Path)
    a = p.parse_args()
    if a.stage == "plan" and a.asset_manifest and a.portrait_root:
        plan(a.input, a.asset_manifest, a.portrait_root, a.output)
    elif a.stage == "build" and a.speech_assets:
        build(a.input, a.speech_assets, a.output)
    else:
        p.error("plan needs --asset-manifest/--portrait-root; build needs --speech-assets")
