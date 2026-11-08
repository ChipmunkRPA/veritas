"""Frozen authored image/clip localization test of the explicitly MIT WAM release."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import public_visual_watermark_v1 as method

ROOT = Path(__file__).resolve().parents[2]
CODE = [
    "backend/scripts/public_visual_watermark_v1.py",
    "backend/scripts/public_visual_watermark_assets_v1.json",
    "backend/scripts/run_public_visual_watermark_v1.py",
    "backend/scripts/review_public_visual_watermark_v1.py",
    "backend/scripts/public_visual_watermark_runtime_v1.py",
    "backend/scripts/specialist_runtime_v4.py",
    "backend/tests/test_public_visual_watermark_v1.py",
    "docs/public-visual-watermark-v1-protocol.md",
]
STATES = ["unmarked", "marked", "partial", "ordinary_edit"]
TRANSFORMS = ["native", "jpeg70", "down_up128"]
POLICY = {
    "schema": "chipmunk.public-visual-watermark-authored-policy/1",
    "scope": "Non-neural authored carriers only; not real-media or production-provider accuracy.",
    "checkpoint": "Explicit MIT SA-1B checkpoint only, matched to pinned publisher mirror SHA256. Original noncommercial COCO weights excluded.",
    "storage": "Download checkpoint directly into bounded RAM; verify full length/SHA256 before restricted tensor loading. Retain identity/loaded-tensor receipts; replay requires reacquisition. No source checkpoint is stored on disk.",
    "images": "Six deterministic256x256 RGB authored carriers; unmarked/full mark/center128x128 partial mark/ordinary rotated center crop; native/JPEG70/down128-up256 bilinear variants.",
    "preprocessing": "Publisher ImageNet transform; native F32 generation with fixed alternating32bit message, scaling_w2, original JND config. Unnormalize/clamp/half-up RGB8 quantization; partial copies marked center pixels only.",
    "mask_readout": "Native first-channel logits strictly>0; retain every logit and pixel.8-connected components>=500pixels are inspection regions; this is a proposed inspection rule, not a validated false-alarm guarantee.",
    "message_readout": "Publisher-supported hard method: each raw bit logit>0, pack all32 binary maps; majority over selected mask pixels. Empty mask is unavailable. Raw continuous bit logits are unused and not retained.",
    "video": "Three authored moving8frame clips at2fps. Unmarked/partial states; partial center mark on frames2:6 only (1–3s), raw frames and H264CRF23; preserve8 decoded frame PTS and exact source coordinates.",
    "image_cases": 72,
    "video_frame_cases": 96,
    "generator_forwards": 18,
    "detector_forwards": 173,
    "parity": "Four first-scene wrapper/direct detector comparisons; first partial repeat bit-exact. No response-based point change or case removal.",
    "inference": "CPUfloat32, twoTorch threads, seed20261108; guard900s/6GiB sampledRSS/1536MiB availableRAM/disk.",
    "provider_authenticated": False,
    "serving_admitted": False,
    "used_in_confidence": False,
    "early_stop_eligible": False,
}


def freeze(main, source, trial):
    method.verify(main, source)
    trial.mkdir(exist_ok=False)
    lock = {
        "policy": POLICY,
        "main": str(main),
        "source": str(source),
        "code_sha256": {p: method.digest(ROOT / p) for p in CODE},
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["torch", "torchvision", "numpy", "scipy", "PyYAML", "einops", "opencv-python-headless", "Pillow"]
        },
        "source_assets_sha256": method.digest(method.LOCK),
        "neural_forwards_before_freeze": 0,
        "customer_inputs": 0,
    }
    method.write(trial / "freeze.json", lock)
    return {"frozen": True, "image_cases": 72, "video_frame_cases": 96, "native_forwards": 191}


def check(trial):
    lock = method.read(trial / "freeze.json")
    method.require(
        lock["policy"] == POLICY and lock["code_sha256"] == {p: method.digest(ROOT / p) for p in CODE},
        "Frozen policy/code changed",
    )
    method.require(
        lock["packages"] == {p: importlib.metadata.version(p) for p in lock["packages"]},
        "Frozen environment changed",
    )
    main, source = Path(lock["main"]), Path(lock["source"])
    method.verify(main, source)
    return main, source


def transform(pixels, name):
    im = Image.fromarray(pixels)
    if name == "jpeg70":
        stream = io.BytesIO()
        im.save(stream, "JPEG", quality=70)
        stream.seek(0)
        return np.asarray(Image.open(stream).convert("RGB")).copy()
    if name == "down_up128":
        return np.asarray(
            im.resize((128, 128), Image.Resampling.BILINEAR).resize((256, 256), Image.Resampling.BILINEAR)
        ).copy()
    method.require(name == "native", "Unknown fixed transform")
    return pixels.copy()


def truth(state):
    mask = np.zeros((256, 256), bool)
    if state == "marked":
        mask[:] = True
    elif state == "partial":
        mask[64:192, 64:192] = True
    return mask


def packet(logits, binary_bits):
    mask = logits > 0
    selected = int(mask.sum())
    message = (
        None if selected == 0 else (binary_bits[:, mask].sum(axis=1) > selected / 2).astype(np.uint8).tolist()
    )
    return {
        **method.spatial(logits),
        "decoded_message": message,
        "decoded_bits_correct_of32": None
        if message is None
        else sum(v == i % 2 for i, v in enumerate(message)),
    }


def encode_video(frames, path):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "256x256",
            "-framerate",
            "2",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            str(path),
        ],
        input=frames.tobytes(),
        check=True,
        timeout=30,
    )
    data = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    method.require(len(data) == 8 * 256 * 256 * 3, "Encoded frame count differs; no trimming/padding")
    probe = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_frames",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    )
    base = probe["streams"][0]["time_base"]
    anchors = []
    for i, row in enumerate(probe["frames"]):
        pts = int(row["best_effort_timestamp"])
        method.require(Fraction(pts) * Fraction(base) == Fraction(i, 2), "Decoded source PTS differs")
        anchors.append({"frame_index": i, "source_pts": pts, "time_base": base, "source_time_s": i / 2})
    method.require(len(anchors) == 8, "Exact8 source anchors required")
    return np.frombuffer(data, np.uint8).reshape(8, 256, 256, 3).copy(), anchors


def run(trial):
    main, source = check(trial)
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261108)
    model, load_receipt = method.load_model(main, source)
    from watermark_anything.data.metrics import msg_predict_inference
    from watermark_anything.data.transforms import default_transform, unnormalize_img

    output = trial / "private"
    output.mkdir(exist_ok=False)
    message = torch.tensor([[i % 2 for i in range(32)]], dtype=torch.float32)
    rows = []
    generators = detectors = 0

    def embed(pixels):
        nonlocal generators
        tensor = default_transform(Image.fromarray(pixels))[None]
        with torch.inference_mode():
            marked = model.embed(tensor, message)["imgs_w"]
        generators += 1
        values = unnormalize_img(marked)[0].permute(1, 2, 0).numpy()
        return np.floor(np.clip(values, 0, 1) * 255 + 0.5).astype(np.uint8)

    def detect(pixels, identity, state, anchor=None, parity=False, repeat=False):
        nonlocal detectors
        tensor = default_transform(Image.fromarray(pixels))[None]
        with torch.inference_mode():
            pred = model.detect(tensor)["preds"]
        detectors += 1
        method.require(
            tuple(pred.shape) == (1, 33, 256, 256)
            and pred.dtype == torch.float32
            and torch.isfinite(pred).all(),
            "Native pixel/time geometry changed",
        )
        if parity:
            with torch.inference_mode():
                direct = model.detector(tensor)
            detectors += 1
            method.require(torch.equal(pred, direct), "Wrapper/direct native outputs differ")
        if repeat:
            with torch.inference_mode():
                again = model.detect(tensor)["preds"]
            detectors += 1
            method.require(torch.equal(pred, again), "Native repeat differs")
        logits = pred[0, 0].numpy().copy()
        bits = (pred[0, 1:] > 0).numpy().copy()
        e = packet(logits, bits)
        if e["decoded_message"] is not None:
            with torch.inference_mode():
                native = msg_predict_inference(
                    pred[:, 1:].clone(),
                    torch.from_numpy((logits > 0).astype(np.float32))[None, None],
                    method="hard",
                )
            method.require(
                native[0].int().tolist() == e["decoded_message"], "Publisher hard message readout differs"
            )
        path = output / (identity + ".npz")
        packed = np.packbits(bits.reshape(32, -1), axis=1, bitorder="little")
        np.savez_compressed(path, pixels=pixels, presence_logits=logits, binary_bit_maps=packed)
        row = {
            "id": identity,
            "state": state,
            "anchor": anchor,
            "private_array": str(path),
            "array_sha256": method.digest(path),
            "source_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "evidence": e,
            "overlap": method.overlap(logits > 0, truth(state)),
        }
        rows.append(row)

    for number, (name, original) in enumerate(method.authored().items()):
        full = embed(original)
        partial = original.copy()
        partial[64:192, 64:192] = full[64:192, 64:192]
        ordinary = original.copy()
        ordinary[64:192, 64:192] = np.rot90(original[64:192, 64:192])
        base = output / (name + "-base.npz")
        np.savez_compressed(base, unmarked=original, marked=full, partial=partial, ordinary_edit=ordinary)
        states = {"unmarked": original, "marked": full, "partial": partial, "ordinary_edit": ordinary}
        for state in STATES:
            for alteration in TRANSFORMS:
                identity = name + "-" + state + "-" + alteration
                detect(
                    transform(states[state], alteration),
                    identity,
                    state,
                    parity=number == 0
                    and (state, alteration)
                    in [
                        ("unmarked", "native"),
                        ("marked", "native"),
                        ("partial", "native"),
                        ("partial", "jpeg70"),
                    ],
                    repeat=number == 0 and (state, alteration) == ("partial", "native"),
                )
                rows[-1].update(
                    kind="image",
                    carrier=name,
                    transform=alteration,
                    base_array=str(base),
                    base_sha256=method.digest(base),
                )
        print(json.dumps({"images_completed": number + 1, "cases_completed": len(rows)}), flush=True)
    for name in ["gradient", "waves", "texture"]:
        base = method.authored()[name]
        original = np.stack([np.roll(base, i * 3, axis=1) for i in range(8)])
        partial = original.copy()
        for i in range(2, 6):
            full = embed(original[i])
            partial[i, 64:192, 64:192] = full[64:192, 64:192]
        base_path = output / (name + "-video-base.npz")
        np.savez_compressed(base_path, unmarked=original, partial=partial)
        for state, frames in [("unmarked", original), ("partial", partial)]:
            video = output / (name + "-" + state + ".mp4")
            decoded, anchors = encode_video(frames, video)
            for container, values in [("raw", frames), ("h264", decoded)]:
                for i, frame in enumerate(values):
                    actual_state = "partial" if state == "partial" and 2 <= i < 6 else "unmarked"
                    anchor = (
                        anchors[i]
                        if container == "h264"
                        else {"frame_index": i, "source_pts": i, "time_base": "1/2", "source_time_s": i / 2}
                    )
                    detect(frame, f"{name}-{state}-{container}-{i}", actual_state, anchor)
                    rows[-1].update(
                        kind="video",
                        carrier=name,
                        clip_state=state,
                        container=container,
                        base_array=str(base_path),
                        base_sha256=method.digest(base_path),
                        video_path=str(video) if container == "h264" else None,
                    )
        print(json.dumps({"video_carrier": name, "cases_completed": len(rows)}), flush=True)
    method.require(
        len(rows) == 168 and generators == 18 and detectors == 173, "Every fixed case/native forward required"
    )
    method.write(trial / "cases-private.json", {"cases": rows})
    result = {
        "schema": "chipmunk.public-visual-watermark-result/1",
        "policy": POLICY,
        "case_count": len(rows),
        "generator_forwards": generators,
        "detector_forwards": detectors,
        "load_receipt": load_receipt,
        "groups": aggregate(rows),
        "provider_authenticated": False,
        "used_in_confidence": False,
        "early_stop_eligible": False,
        "serving_admitted": False,
    }
    method.write(trial / "result.json", result)
    print(json.dumps({"complete": len(rows), "groups": result["groups"]}), flush=True)


def aggregate(rows):
    keys = sorted(
        {(r["kind"], r.get("transform", r.get("container")), r.get("clip_state", r["state"])) for r in rows}
    )
    result = []
    for kind, alteration, state in keys:
        members = [
            r
            for r in rows
            if (r["kind"], r.get("transform", r.get("container")), r.get("clip_state", r["state"]))
            == (kind, alteration, state)
        ]
        marked = [r for r in members if r["state"] in ["partial", "marked"]]
        unmarked = [r for r in members if r["state"] in ["unmarked", "ordinary_edit"]]
        result.append(
            {
                "kind": kind,
                "alteration": alteration,
                "state": state,
                "cases": len(members),
                "flagged": sum(r["evidence"]["flagged_for_inspection"] for r in members),
                "unmarked_cases": len(unmarked),
                "unmarked_flagged": sum(r["evidence"]["flagged_for_inspection"] for r in unmarked),
                "marked_cases": len(marked),
                "marked_missed": sum(not r["evidence"]["flagged_for_inspection"] for r in marked),
                "mean_marked_iou": float(np.mean([r["overlap"]["iou"] for r in marked])) if marked else None,
                "mean_selected_pixels": float(np.mean([r["evidence"]["selected_pixels"] for r in members])),
            }
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("main", type=Path)
    f.add_argument("source", type=Path)
    f.add_argument("trial", type=Path)
    f = sub.add_parser("run")
    f.add_argument("trial", type=Path)
    a = p.parse_args()
    if a.action == "freeze":
        print(json.dumps(freeze(a.main, a.source, a.trial), indent=2))
    else:
        run(a.trial)
