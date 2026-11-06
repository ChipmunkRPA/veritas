"""Exact source/mask/control audit and fixed sampled neural replay of v7 clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from app.core.config import settings
from app.detectors.audio.decode import decode_audio
from app.detectors.image.adapters.loaders import load_yunet
from app.detectors.image.faces import detect_faces
from app.detectors.video.decode import decode_frames, probe_frames
from app.engine.fetchers import ensure_verified
from app.engine.registry import get_registry
from scripts.liveportrait_mouth_benchmark import digest
from scripts.musetalk_benchmark_v7 import image_inputs, load, tensor_sha
from scripts.prepare_mouth_edit_v4 import composite, interval
from scripts.smoke_mouth_generator_v7 import alpha_from_face, crop_from_face


def audit(plan_path, manifest_path, asset_manifest, assets, output):
    recipe = json.loads(plan_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    parent = Path(recipe["parent_manifest"])
    expected_ids = [
        source["id"].rsplit(":", 1)[0] + ":" + condition
        for source in recipe["sources"]
        for condition in recipe["conditions"]
    ]
    if (
        manifest["plan_sha256"] != digest(plan_path)
        or digest(parent) != recipe["parent_manifest_sha256"]
        or [row["id"] for row in manifest["items"]] != expected_ids
        or len(set(expected_ids)) != 40
    ):
        raise ValueError("Frozen sources or complete planned panel changed")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise ValueError("Fixed neural replay requires the recorded MPS device")
    torch.mps.set_per_process_memory_fraction(0.65)
    ensure_verified(get_registry().manifest.model("yunet.2023mar"), force=True)
    detector = load_yunet("yunet.2023mar", "cpu", "float32").obj
    renderer = load(asset_manifest, assets, "mps")
    records = []
    for source in recipe["sources"]:
        original_path = parent.parent / source["path"]
        if digest(original_path) != source["sha256"]:
            raise ValueError("Parent original changed")
        table = probe_frames(original_path)
        frames = [f.rgb for f in decode_frames(original_path, table, range(len(table)))]
        audio = decode_audio(original_path, source_sha256=source["sha256"]).pcm
        start, end = interval(len(table))
        chosen = {start, start + 10}
        chunks = renderer.conditioning(audio)
        geometry = {}
        for i in range(start, end):
            faces = detect_faces(
                cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), detector=detector, min_size=settings.face_min_size
            )
            if len(faces) != 1:
                raise ValueError("Source geometry cannot be independently replayed")
            crop, matrix, meta = crop_from_face(frames[i], faces[0], "landmark-box")
            geometry[i] = (crop, matrix, meta, alpha_from_face(frames[i].shape, faces[0]))
        cases = [r for r in manifest["items"] if r["group"] == source["group"]]
        for case in cases:
            result = {
                "id": case["id"],
                "status": "failed",
                "verified_frames": 0,
                "verified_masks": 0,
                "verified_neural_inputs": 0,
                "neural_output_replays": 0,
            }
            try:
                path = manifest_path.parent / case["path"]
                if (
                    case["status"] != "ready"
                    or digest(path) != case["sha256"]
                    or case["source_sha256"] != source["sha256"]
                    or case["label"] != int(case["condition"].startswith("muse-ai-"))
                ):
                    raise ValueError("Prepared bytes or source lineage changed")
                actual = probe_frames(path)
                if (
                    actual.pts != table.pts
                    or actual.time_base != table.time_base
                    or (actual.width, actual.height) != (table.width, table.height)
                ):
                    raise ValueError("Exact source picture clock or dimensions changed")
                if case["edit_frame_range"] != [start, end] or case["edit_time_range_s"] != [
                    start / 25,
                    end / 25,
                ]:
                    raise ValueError("Declared edit interval changed")
                masks = {r["frame_index"]: r for r in case["masks"]}
                if set(masks) != set(range(start, end)):
                    raise ValueError("Missing or extra native edit masks")
                for frame in decode_frames(path, actual, range(len(actual))):
                    changed = np.any(frame.rgb != frames[frame.index], axis=2)
                    result["verified_frames"] += 1
                    if frame.index not in masks:
                        if changed.any():
                            raise ValueError("Surrounding unedited source frame changed")
                        continue
                    i = frame.index
                    mask = masks[i]
                    p = manifest_path.parent / mask["path"]
                    if (
                        digest(p) != mask["sha256"]
                        or not np.array_equal(changed, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0)
                        or int(changed.sum()) != mask["changed_pixels"]
                    ):
                        raise ValueError("Exact changed-pixel mask differs from native replay")
                    crop, matrix, meta, alpha = geometry[i]
                    opacity = min(1.0, (i - start + 1) / 2, (end - i) / 2)
                    if (
                        changed[alpha == 0].any()
                        or mask["source_rgb_sha256"] != hashlib.sha256(frames[i].tobytes()).hexdigest()
                        or mask["source_crop_rgb_sha256"] != hashlib.sha256(crop.tobytes()).hexdigest()
                        or mask["source_to_crop_matrix_hex"]
                        != [[float(v).hex() for v in row] for row in matrix]
                        or mask["geometry"] != meta
                    ):
                        raise ValueError("Source crop/support binding differs from freshly detected geometry")
                    c = case["condition"]
                    inverse = cv2.invertAffineTransform(matrix)
                    if not case["label"]:
                        raw = (
                            cv2.warpAffine(crop, inverse, (table.width, table.height), flags=cv2.INTER_LINEAR)
                            if c == "muse-resample"
                            else cv2.GaussianBlur(frames[i], (0, 0), 2.0 if c == "local-blur-2" else 4.0)
                        )
                        if not np.array_equal(frame.rgb, composite(frames[i], raw, alpha, opacity)):
                            raise ValueError("Ordinary control does not match its exact declared operation")
                    else:
                        seed = int.from_bytes(
                            hashlib.sha256(
                                f"chipmunk.mouth-multifamily/7:{source['id'].rsplit(':', 1)[0]}:{i}".encode()
                            ).digest()[:8],
                            "big",
                        ) % (2**63)
                        k = i if c == "muse-ai-aligned" else i - 8
                        proof = mask["model_proof"]
                        masked, original = image_inputs(crop)
                        if (
                            proof["seed"] != seed
                            or proof["conditioning_frame"] != k
                            or proof["conditioning_f32_sha256"] != tensor_sha(chunks[k])
                            or proof["masked_input_f32_sha256"] != tensor_sha(masked)
                            or proof["original_input_f32_sha256"] != tensor_sha(original)
                        ):
                            raise ValueError(
                                "Neural source pixels, conditioning or seed differs from exact replay"
                            )
                        result["verified_neural_inputs"] += 1
                        if i in chosen:
                            generated, replay = renderer.render(crop, chunks[k], seed)
                            if any(replay[key] != proof[key] for key in replay):
                                raise ValueError(
                                    "Sampled neural output/proof is not bitwise repeatable on this runtime"
                                )
                            native = cv2.warpAffine(
                                generated, inverse, (table.width, table.height), flags=cv2.INTER_LINEAR
                            )
                            if not np.array_equal(frame.rgb, composite(frames[i], native, alpha, opacity)):
                                raise ValueError("Sampled neural source composite differs")
                            result["neural_output_replays"] += 1
                    result["verified_masks"] += 1
                if result["verified_frames"] != len(table) or not np.array_equal(
                    decode_audio(path, source_sha256=case["sha256"]).pcm, audio
                ):
                    raise ValueError("Incomplete frame replay or changed original PCM")
                result["status"] = "ok"
            except Exception as error:
                result["reason"] = f"{type(error).__name__}: {error}"
            records.append(result)
    report = {
        "schema": "chipmunk.mouth-extension-source-audit/7",
        "manifest_sha256": digest(manifest_path),
        "implementation_sha256": digest(Path(__file__)),
        "planned": 40,
        "passed": sum(r["status"] == "ok" for r in records),
        "verified_native_frames": sum(r["verified_frames"] for r in records),
        "verified_masks": sum(r["verified_masks"] for r in records),
        "verified_neural_input_bindings": sum(r["verified_neural_inputs"] for r in records),
        "sampled_neural_output_replays": sum(r["neural_output_replays"] for r in records),
        "sampling_rule": "First and eleventh edited frame, both neural conditions, every actor",
        "scope": "Local source/clock/mask/control audit and explicitly sampled neural replay; not outside review or detection accuracy",
        "cases": records,
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))
    if report["passed"] != 40:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "manifest", "asset_manifest", "assets", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    audit(a.plan, a.manifest, a.asset_manifest, a.assets, a.output)
