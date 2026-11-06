"""Replay source pixels, masks, source clocks, fitting views and sampled neural outputs."""

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
from app.detectors.image import gend
from app.detectors.image.adapters.loaders import load_yunet
from app.detectors.image.faces import detect_faces
from app.detectors.video.decode import decode_frames, probe_frames
from app.engine.fetchers import ensure_verified
from app.engine.registry import get_registry
from scripts import mouth_learner_v8 as method
from scripts.liveportrait_mouth_benchmark import digest
from scripts.mouth_probe_v6 import mouth_chip, preprocessed
from scripts.musetalk_benchmark_v7 import load, tensor_sha
from scripts.prepare_mouth_edit_v4 import composite
from scripts.prepare_mouth_learning_v8 import conditioning_index
from scripts.run_mouth_learning_v8 import check_lock, load_cache, write
from scripts.smoke_mouth_generator_v7 import alpha_from_face, crop_from_face


def audit(plan_path, lock_path, cache_path, asset_manifest, assets, output):
    if not torch.backends.mps.is_available():
        raise ValueError("Sampled neural replay requires the recorded MPS runtime")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(0.65)
    recipe = json.loads(plan_path.read_text())
    check_lock(lock_path)
    _, rows, receipt, _ = load_cache(lock_path, cache_path)
    source_root = Path(recipe["source_root"])
    if digest(source_root / "manifest.json") != recipe["source_manifest_sha256"]:
        raise ValueError("Original source manifest changed")
    ensure_verified(get_registry().manifest.model("yunet.2023mar"), force=True)
    detector = load_yunet("yunet.2023mar", "cpu", "float32").obj
    renderer = load(asset_manifest, assets, "mps")
    bindings = {r["id"]: r for r in receipt["bindings"]}
    records = []
    totals = {
        "native_stills": 0,
        "exact_masks": 0,
        "fitting_views": 0,
        "ordinary_reconstructions": 0,
        "neural_input_bindings": 0,
        "sampled_neural_output_replays": 0,
    }
    for source in recipe["sources"]:
        path = source_root / source["path"]
        if digest(path) != source["sha256"]:
            raise ValueError("Original source bytes changed")
        table = probe_frames(path)
        frames = {f.index: f.rgb for f in decode_frames(path, table, source["selected_frames"])}
        audio = decode_audio(path, source_sha256=source["sha256"])
        chunks = renderer.conditioning(audio.pcm)
        source_geometry = {}
        for i, original in frames.items():
            faces = detect_faces(
                cv2.cvtColor(original, cv2.COLOR_RGB2BGR), detector=detector, min_size=settings.face_min_size
            )
            if len(faces) != 1:
                raise ValueError("Source geometry is not unique")
            source_geometry[i] = (faces[0], crop_from_face(original, faces[0], "landmark-box"))
        for row in [r for r in rows if r["group"] == source["group"]]:
            record = {"id": row["id"], "status": "failed"}
            try:
                i = row["frame_index"]
                original = frames[i]
                if (
                    row["role"] != source["role"]
                    or row["source_sha256"] != source["sha256"]
                    or row["pts"] != table.pts[i]
                    or row["time_base"] != table.time_base
                ):
                    raise ValueError("Source, actor role or native picture clock changed")
                if hashlib.sha256(original.tobytes()).hexdigest() != row["original_rgb_sha256"]:
                    raise ValueError("Original RGB binding changed")
                image = Path(row["path"])
                if digest(image) != row["sha256"]:
                    raise ValueError("PNG file changed")
                rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
                if hashlib.sha256(rgb.tobytes()).hexdigest() != row["native_rgb_sha256"]:
                    raise ValueError("PNG decoded pixels changed")
                mask = Path(row["mask_path"])
                changed = np.any(rgb != original, axis=2)
                if (
                    digest(mask) != row["mask_sha256"]
                    or not np.array_equal(changed, cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE) > 0)
                    or int(changed.sum()) != row["changed_pixels"]
                ):
                    raise ValueError("Exact changed-pixel mask differs from source")
                if bool(changed.any()) != (row["condition"] != "original"):
                    raise ValueError("Native ordinary/edit status changed")
                positive = row["condition"].startswith(("ai-", "muse-ai-"))
                if row["label"] != int(positive):
                    raise ValueError("AI provenance label changed")
                totals["native_stills"] += 1
                totals["exact_masks"] += 1
                binding = bindings[row["id"]]
                faces = detect_faces(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=settings.face_min_size
                )
                transform = gend.align_transform(faces[0].landmarks) if faces else None
                if (
                    transform is None
                    or transform[1] != binding["alignment_side"]
                    or [[float(v).hex() for v in r] for r in transform[0]] != binding["alignment_matrix_hex"]
                ):
                    raise ValueError("Fresh fitting alignment differs")
                aligned = gend.warp(rgb, *transform)
                chip, bounds = mouth_chip(aligned)
                if (
                    bounds != binding["chip_bounds"]
                    or hashlib.sha256(aligned.tobytes()).hexdigest() != binding["aligned_rgb_sha256"]
                    or hashlib.sha256(chip.tobytes()).hexdigest() != binding["mouth_chip_rgb_sha256"]
                ):
                    raise ValueError("Native mouth measurement support differs")
                for v in binding["views"]:
                    variant = method.view(chip, v["view"])
                    if (
                        hashlib.sha256(variant.tobytes()).hexdigest() != v["view_rgb_sha256"]
                        or hashlib.sha256(preprocessed([variant])[0].tobytes()).hexdigest()
                        != v["input_f32_sha256"]
                    ):
                        raise ValueError("Augmented/native model input differs from source replay")
                    totals["fitting_views"] += 1
                if row["family"] == "musetalk":
                    face, (crop, matrix, geometry) = source_geometry[i]
                    alpha = alpha_from_face(original.shape, face)
                    offset = i - source["edit_start"]
                    opacity = min(1.0, (offset + 1) / 2, (20 - offset) / 2)
                    if (
                        row["opacity"] != opacity
                        or row["geometry"] != geometry
                        or row["source_to_crop_matrix_hex"] != [[float(v).hex() for v in r] for r in matrix]
                        or hashlib.sha256(crop.tobytes()).hexdigest() != row["source_crop_rgb_sha256"]
                        or not np.array_equal(rgb[alpha == 0], original[alpha == 0])
                    ):
                        raise ValueError("Generation crop/support/source pixels differ")
                    tick = conditioning_index(table.seconds(i), audio.mapping, len(chunks))
                    if row["audio_mapping"] != audio.mapping or row["aligned_conditioning_tick"] != tick:
                        raise ValueError("Neural conditioning uses a different source-audio clock")
                    inverse = cv2.invertAffineTransform(matrix)
                    h, w = original.shape[:2]
                    c = row["condition"]
                    if not positive:
                        transformed = (
                            cv2.warpAffine(crop, inverse, (w, h), flags=cv2.INTER_LINEAR)
                            if c == "muse-resample"
                            else cv2.GaussianBlur(original, (0, 0), float(c.rsplit("-", 1)[1]))
                        )
                        if not np.array_equal(rgb, composite(original, transformed, alpha, opacity)):
                            raise ValueError("Ordinary transformation does not reproduce exactly")
                        totals["ordinary_reconstructions"] += 1
                    else:
                        k = tick if c == "muse-ai-aligned" else tick - 8
                        seed = int.from_bytes(
                            hashlib.sha256(f"chipmunk.mouth-learning/8:{source['id']}:{i}".encode()).digest()[
                                :8
                            ],
                            "big",
                        ) % (2**63)
                        proof = row["model_proof"]
                        if (
                            proof["seed"] != seed
                            or proof["conditioning_frame"] != k
                            or proof["conditioning_f32_sha256"] != tensor_sha(chunks[k])
                        ):
                            raise ValueError("Neural input/seed binding changed")
                        totals["neural_input_bindings"] += 1
                        if i in (source["selected_frames"][0], source["selected_frames"][4]):
                            generated, fresh_proof = renderer.render(crop, chunks[k], seed)
                            if any(proof[key] != value for key, value in fresh_proof.items()):
                                raise ValueError("Sampled neural output proof differs")
                            transformed = cv2.warpAffine(generated, inverse, (w, h), flags=cv2.INTER_LINEAR)
                            if not np.array_equal(rgb, composite(original, transformed, alpha, opacity)):
                                raise ValueError("Sampled neural native composite differs")
                            totals["sampled_neural_output_replays"] += 1
                record["status"] = "ok"
            except Exception as error:
                record["reason"] = f"{type(error).__name__}: {error}"
            records.append(record)
        print(
            json.dumps(
                {
                    "source": source["id"],
                    "checked": len(records),
                    "failed": sum(r["status"] != "ok" for r in records),
                }
            ),
            flush=True,
        )
    body = {
        "schema": "chipmunk.mouth-learning-source-audit/8",
        "plan_sha256": digest(plan_path),
        "lock_sha256": digest(lock_path),
        "cache_receipt_sha256": digest(cache_path / "receipt.json"),
        "audit_implementation_sha256": digest(Path(__file__)),
        "planned": 1600,
        "passed": len(records) == 1600 and all(r["status"] == "ok" for r in records),
        "totals": totals,
        "cases": records,
        "scope": "All native masks/source times/fitting view tensors; all 320 new neural input bindings; 80/320 neural output replays. First-family generation was audited in v6, not rerun here.",
        "outside_review": False,
    }
    write(output, body)
    print(json.dumps({k: v for k, v in body.items() if k != "cases"}), flush=True)
    if not body["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("lock", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--asset-manifest", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    a = p.parse_args()
    audit(a.plan, a.lock, a.cache, a.asset_manifest, a.assets, a.output)
