"""Create source-bound AI mouth edits and non-AI controls for consumed development."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from app.detectors.audio.decode import decode_audio, wav_bytes
from app.detectors.video.decode import decode_frames, probe_frames
from scripts.liveportrait_mouth_benchmark import crop_from_mesh, digest, load

POLICY = "chipmunk.mouth-edit-development/4.1"
SELECTION_SEED = "chipmunk.mouth-edit-development/4"
CONDITIONS = ("original", "local-light", "local-resample", "constant-audio-delay", "ai-redraw", "ai-retime")
RATE, FPS, EDIT_FRAMES = 16000, 25, 20


def order(value):
    return hashlib.sha256((SELECTION_SEED + ":" + value).encode()).hexdigest()


def interval(frames):
    if not 63 <= frames <= 100:
        raise ValueError("Development clips require 63–100 source frames at 25 Hz")
    start = (frames - EDIT_FRAMES) // 2
    return start, start + EDIT_FRAMES


def mouth_alpha(shape, landmarks):
    h, w = shape[:2]
    p = np.array([(x.x * w, x.y * h) for x in landmarks], np.float64)
    center = (p[13] + p[14]) / 2
    axis = p[308] - p[78]
    width = float(np.linalg.norm(axis))
    if not np.isfinite(p).all() or width < 8:
        raise ValueError("Insufficient mouth geometry for a controlled edit")
    axis /= width
    normal = np.array([-axis[1], axis[0]])
    y, x = np.indices((h, w))
    dx, dy = x - center[0], y - center[1]
    u, v = dx * axis[0] + dy * axis[1], dx * normal[0] + dy * normal[1]
    radius = np.sqrt((u / (0.7 * width)) ** 2 + (v / (0.45 * width)) ** 2)
    return np.clip((1 - radius) / 0.2, 0, 1).astype(np.float32)


def composite(original, transformed, alpha, opacity=1.0):
    if original.dtype != np.uint8 or transformed.shape != original.shape or alpha.shape != original.shape[:2]:
        raise ValueError("Invalid native composite inputs")
    a = alpha[:, :, None] * opacity
    result = (
        np.rint(original.astype(np.float32) * (1 - a) + transformed.astype(np.float32) * a)
        .clip(0, 255)
        .astype(np.uint8)
    )
    if not np.array_equal(result[alpha == 0], original[alpha == 0]):
        raise ValueError("Composite changed pixels outside its recorded support")
    return result


def plan(old, output, amend_from=None):
    m = json.loads((old / "manifest.json").read_text())
    rows = []
    for row in m["items"]:
        if row["category"] != "original" or not row["group"].startswith("CREMA-D:"):
            continue
        p = old / row["path"]
        if digest(p) != row["sha256"]:
            raise ValueError("Consumed original source changed")
        t = probe_frames(p)
        audio = decode_audio(p, source_sha256=row["sha256"])
        common = min(100, len(t), len(audio.pcm) // (RATE // FPS))
        if common >= 63:
            rows.append({**row, "frames": common, "source_pcm_samples": len(audio.pcm)})
    selected = sorted(rows, key=lambda r: order(r["id"]))[:8]
    if len(selected) != 8:
        raise ValueError("Insufficient consumed originals; no score-dependent substitution is allowed")
    if amend_from is not None:
        prior = json.loads(amend_from.read_text())
        if [r["id"] for r in prior["sources"]] != [r["id"] for r in selected]:
            raise ValueError("Common-clock amendment must preserve the exact selected actor/source IDs")
    root = Path(__file__).resolve().parents[2]
    pins = [
        "backend/scripts/liveportrait_mouth_benchmark.py",
        "backend/scripts/prepare_mouth_edit_v4.py",
        "docs/mouth-edit-v4-development.md",
    ]
    body = {
        "policy": POLICY,
        "selection_seed": SELECTION_SEED,
        "amended_generation_plan_sha256": digest(amend_from) if amend_from else None,
        "role": "consumed-source development; not fresh confirmation",
        "source_root": str(old.resolve()),
        "source_manifest_sha256": digest(old / "manifest.json"),
        "sources": selected,
        "conditions": CONDITIONS,
        "planned_cases": len(selected) * len(CONDITIONS),
        "excluded_prior_actors": sorted(
            {r["group"] for r in m["items"] if r["group"].startswith("CREMA-D:")}
        ),
        "implementation_sha256": {n: digest(root / n) for n in pins},
        "deployment_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "planned_cases": body["planned_cases"],
                "eligible_consumed_sources": len(rows),
                "plan_sha256": digest(output),
            }
        )
    )


def encode(frames, audio, target):
    h, w = frames[0].shape[:2]
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        "25",
        "-i",
        "pipe:0",
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map_metadata",
        "-1",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "bgr0",
        "-level",
        "3",
        "-c:a",
        "pcm_s16le",
        "-shortest",
        "-threads",
        "2",
        str(target),
    ]
    subprocess.run(
        command, input=b"".join(f.tobytes() for f in frames), check=True, capture_output=True, timeout=120
    )
    table = probe_frames(target)
    if len(table) != len(frames) or any(abs(table.seconds(i) - i / FPS) > 1e-9 for i in range(len(table))):
        raise ValueError("Encoded video did not preserve the planned frame clock")
    for f in decode_frames(target, table, range(len(table))):
        if not np.array_equal(f.rgb, frames[f.index]):
            raise ValueError("Lossless encoding changed the planned native RGB pixels")
    sound = decode_audio(target, source_sha256=digest(target))
    expected = decode_audio(audio, source_sha256=digest(audio))
    if not np.array_equal(sound.pcm, expected.pcm):
        raise ValueError("Encoded soundtrack differs from its declared PCM")
    return {
        "sha256": digest(target),
        "frames": len(table),
        "time_base": table.time_base,
        "pts": list(map(int, table.pts)),
        "pcm_sha256": sound.mapping["pcm_sha256"],
        "audio_samples": len(sound.pcm),
    }


def build(plan_path, assets, output, prepared_cache=None):
    import mediapipe as mp
    import torch

    p = json.loads(plan_path.read_text())
    root = Path(__file__).resolve().parents[2]
    for name, sha in p["implementation_sha256"].items():
        if digest(root / name) != sha:
            raise ValueError("Generation implementation changed after freeze")
    old = Path(p["source_root"])
    if digest(old / "manifest.json") != p["source_manifest_sha256"]:
        raise ValueError("Consumed input manifest changed")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    renderer = load(assets, "mps")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(assets / "permissions/face_landmarker.task"),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=2,
    )
    results = []
    cache = json.loads((prepared_cache / "manifest.json").read_text()) if prepared_cache else None
    if cache and cache["renderer_asset_receipt_sha256"] != digest(
        assets / "permissions/liveportrait-verified-assets.json"
    ):
        raise ValueError("Prepared cache uses different generator assets")
    for source in p["sources"]:
        started = time.monotonic()
        folder = output / source["id"]
        folder.mkdir()
        planned = [
            {
                "id": source["id"] + ":" + condition,
                "group": source["group"],
                "condition": condition,
                "label": int(condition.startswith("ai-")),
                "source_sha256": source["sha256"],
                "source_url": source["download_url"],
                "source_license": source["source_license"],
            }
            for condition in CONDITIONS
        ]
        reusable = [r for r in cache["items"] if r["group"] == source["group"]] if cache else []
        if (
            len(reusable) == len(CONDITIONS)
            and {r["condition"] for r in reusable} == set(CONDITIONS)
            and all(
                r.get("status") == "ready"
                and r["source_sha256"] == source["sha256"]
                and r["frames"] == source["frames"]
                and r["audio_samples"] == source["frames"] * RATE // FPS
                for r in reusable
            )
        ):
            for row in reusable:
                if digest(prepared_cache / row["path"]) != row["sha256"]:
                    raise ValueError("Prepared cache bytes changed")
                for mask in row["masks"]:
                    if digest(prepared_cache / mask["path"]) != mask["sha256"]:
                        raise ValueError("Prepared cache mask changed")
            shutil.copytree(prepared_cache / source["id"], folder, dirs_exist_ok=True)
            planned = [
                {**row, "prepared_from_manifest_sha256": digest(prepared_cache / "manifest.json")}
                for row in reusable
            ]
            results.extend(planned)
            with (output / "partial-receipts.jsonl").open("a") as stream:
                for case in planned:
                    stream.write(json.dumps(case, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {"source": source["id"], "ready": len(planned), "reused_verified_preparation": True}
                ),
                flush=True,
            )
            continue
        try:
            path = old / source["path"]
            if digest(path) != source["sha256"]:
                raise ValueError("Consumed source bytes changed")
            table = probe_frames(path)
            n = source["frames"]
            frames = [f.rgb for f in decode_frames(path, table, range(n))]
            if len(frames) != n:
                raise ValueError("A planned source frame is unavailable")
            start, end = interval(n)
            sound = decode_audio(path, source_sha256=source["sha256"])
            pcm = sound.pcm[: n * RATE // FPS]
            if len(pcm) != n * RATE // FPS:
                raise ValueError("Source audio is too short for the frozen frame interval")
            audio = folder / "source.wav"
            audio.write_bytes(wav_bytes(pcm))
            seen = {}
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
                for i, rgb in enumerate(frames):
                    if not max(0, start - 8) <= i < end:
                        continue
                    result = detector.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), i * 40
                    )
                    if len(result.face_landmarks) == 1:
                        seen[i] = result.face_landmarks[0]
            required = range(start - 8, end)
            if not all(i in seen for i in required):
                raise ValueError(
                    "Missing unique-face geometry for a planned edit frame; source is not replaced"
                )
            geometry = {i: crop_from_mesh(frames[i], seen[i]) for i in required}
            treatments = {c: list(frames) for c in CONDITIONS}
            masks = {c: [] for c in CONDITIONS if c not in ("original", "constant-audio-delay")}
            for i in range(start, end):
                crop, matrix, source_ratio = geometry[i]
                output_matrix = matrix * 2
                inverse = cv2.invertAffineTransform(output_matrix)
                h, w = frames[i].shape[:2]
                alpha = mouth_alpha(frames[i].shape, seen[i])
                opacity = min(1.0, (i - start + 1) / 2, (end - i) / 2)
                variants = {
                    "local-light": (frames[i].astype(np.float32) * 1.12)
                    .round()
                    .clip(0, 255)
                    .astype(np.uint8),
                    "local-resample": cv2.warpAffine(
                        cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR),
                        inverse,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                    ),
                }
                for c, ratio in (("ai-redraw", source_ratio), ("ai-retime", geometry[i - 8][2])):
                    generated = renderer.render(crop, source_ratio, ratio)
                    variants[c] = cv2.warpAffine(generated, inverse, (w, h), flags=cv2.INTER_LINEAR)
                for c, transformed in variants.items():
                    edited = composite(frames[i], transformed, alpha, opacity)
                    changed = np.any(edited != frames[i], axis=2).astype(np.uint8) * 255
                    name = f"{c}-mask-{i:04}.png"
                    cv2.imwrite(str(folder / name), changed)
                    masks[c].append(
                        {
                            "frame_index": i,
                            "pts_ms": i * 40,
                            "path": name,
                            "sha256": digest(folder / name),
                            "changed_pixels": int(np.count_nonzero(changed)),
                        }
                    )
                    treatments[c][i] = edited
                if (i - start + 1) % 5 == 0:
                    print(
                        json.dumps({"source": source["id"], "generated_edit_frames": i - start + 1}),
                        flush=True,
                    )
            delayed = np.r_[np.zeros(5120, np.float32), pcm][: (len(pcm))]
            delay_audio = folder / "delayed.wav"
            delay_audio.write_bytes(wav_bytes(delayed))
            for case in planned:
                c = case["condition"]
                target = folder / (c + ".mkv")
                provenance = encode(
                    treatments[c], delay_audio if c == "constant-audio-delay" else audio, target
                )
                case.update(
                    status="ready",
                    path=str(target.relative_to(output)),
                    **provenance,
                    edit_frame_range=[start, end] if c not in ("original", "constant-audio-delay") else None,
                    edit_time_range_s=[start / FPS, end / FPS]
                    if c not in ("original", "constant-audio-delay")
                    else None,
                    masks=[
                        {**r, "path": str((folder / r["path"]).relative_to(output))} for r in masks.get(c, [])
                    ],
                    source_ranges={"before": [0, start], "during": [start, end], "after": [end, n]},
                    original_rgb_sha256=[hashlib.sha256(f.tobytes()).hexdigest() for f in frames],
                    renderer="Pinned MIT human portrait model with permitted mesh preprocessing"
                    if case["label"]
                    else None,
                )
        except Exception as error:
            for case in planned:
                if case.get("status") != "ready":
                    case.update(status="unavailable", reason=f"{type(error).__name__}: {error}")
        results.extend(planned)
        with (output / "partial-receipts.jsonl").open("a") as stream:
            for case in planned:
                stream.write(json.dumps(case, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "source": source["id"],
                    "seconds": time.monotonic() - started,
                    "ready": sum(c["status"] == "ready" for c in planned),
                }
            ),
            flush=True,
        )
    body = {
        "policy": POLICY,
        "plan_sha256": digest(plan_path),
        "items": results,
        "role": p["role"],
        "deployment_eligible": False,
        "renderer_asset_receipt_sha256": digest(assets / "permissions/liveportrait-verified-assets.json"),
        "note": "AI provenance and exact changed pixels are known; timing agreement is not an authenticity verdict. Zero labels include ordinary non-AI edits.",
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
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["plan", "build"])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--assets", type=Path)
    p.add_argument("--amend-from", type=Path)
    p.add_argument("--prepared-cache", type=Path)
    a = p.parse_args()
    if a.stage == "plan":
        plan(a.input, a.output, a.amend_from)
    else:
        if not a.assets:
            p.error("build requires --assets")
        build(a.input, a.assets, a.output, a.prepared_cache)
