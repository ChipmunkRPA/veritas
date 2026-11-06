"""Replay complete fitting clocks, masks, ordinary edits and neural bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.detectors.audio.decode import decode_audio
from app.detectors.image.faces import detect_faces
from app.detectors.video.decode import decode_frames, probe_frames
from scripts.audit_mouth_probe_v6 import detector
from scripts.liveportrait_mouth_benchmark import crop_from_mesh, digest
from scripts.liveportrait_mouth_benchmark import load as load_portrait
from scripts.musetalk_benchmark_v7 import image_inputs, tensor_sha
from scripts.musetalk_benchmark_v7 import load as load_speech
from scripts.prepare_mouth_context_v10 import geometry_frames, operations
from scripts.prepare_mouth_edit_v4 import composite, mouth_alpha
from scripts.prepare_mouth_learning_v8 import conditioning_index
from scripts.prepare_mouth_temporal_v11 import AI_CONDITIONS, CONDITIONS, clock_span, code_pins, seed_for
from scripts.run_mouth_learning_v8 import write
from scripts.smoke_mouth_generator_v7 import alpha_from_face, crop_from_face


def audit(plan_path, manifest_path, assets, output):
    import torch

    recipe = json.loads(plan_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    expected = [(s["id"] + ":v11:" + c, s["group"], c) for s in recipe["sources"] for c in CONDITIONS]
    if (
        recipe["implementation_sha256"] != code_pins()
        or manifest["plan_sha256"] != digest(plan_path)
        or [(r["id"], r["group"], r["condition"]) for r in manifest["items"]] != expected
    ):
        raise ValueError("Wrong or incomplete frozen construction panel")
    if not torch.backends.mps.is_available():
        raise ValueError("Frozen neural replay requires MPS")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(0.65)
    face_detector = detector()
    portrait = load_portrait(Path(recipe["portrait_root"]), "mps")
    speech = load_speech(Path(recipe["speech_asset_manifest"]), assets, "mps")
    records = []
    native_frames = exact_masks = pcm_samples = neural_bindings = neural_replays = 0
    for source in recipe["sources"]:
        source_path = Path(recipe["source_root"]) / source["path"]
        if digest(source_path) != source["sha256"]:
            raise ValueError("Original source changed")
        table = probe_frames(source_path)
        audio = decode_audio(source_path, source_sha256=source["sha256"])
        span = clock_span(table, audio)
        if span != source["clock_span"]:
            raise ValueError("Common source clocks changed")
        n = span["frames"]
        original = [f.rgb for f in decode_frames(source_path, table, range(n))]
        mesh = geometry_frames(original, table, Path(recipe["task"]))
        start, end = (n - 20) // 2, (n - 20) // 2 + 20
        geom = {i: crop_from_mesh(original[i], mesh[i]) for i in range(start - 8, end)}
        chunks = speech.conditioning(audio.pcm)
        fresh = {}
        for i in range(start, end):
            rgb = original[i]
            faces = detect_faces(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=face_detector, min_size=settings.face_min_size
            )
            if len(faces) != 1:
                raise ValueError("Fresh source face is not unique")
            variants, binding = operations(rgb, mesh[i], faces[0])
            crop, matrix, ratio = geom[i]
            scrop, smatrix, _ = crop_from_face(rgb, faces[0], "landmark-box")
            tick = conditioning_index(table.seconds(i), audio.mapping, len(chunks))
            masked, unmasked = image_inputs(scrop)
            proofs = {}
            for c, target in (("ai-redraw", ratio), ("ai-retime", geom[i - 8][2])):
                proofs[c] = {
                    "source_aperture": ratio,
                    "target_aperture": target,
                    "target_frame_index": i if c == "ai-redraw" else i - 8,
                }
                if i - start in (0, 10):
                    generated = portrait.render(crop, ratio, target)
                    proofs[c]["generated_rgb_sha256"] = hashlib.sha256(generated.tobytes()).hexdigest()
                    variants[c] = (
                        cv2.warpAffine(
                            generated,
                            cv2.invertAffineTransform(matrix * 2),
                            (table.width, table.height),
                            flags=cv2.INTER_LINEAR,
                        ),
                        mouth_alpha(rgb.shape, mesh[i]),
                    )
            for c, k in (("muse-ai-aligned", tick), ("muse-ai-earlier", tick - 8)):
                seed = seed_for(source["id"], i)
                proofs[c] = {
                    "conditioning_frame": k,
                    "seed": seed,
                    "conditioning_f32_sha256": tensor_sha(chunks[k]),
                    "masked_input_f32_sha256": tensor_sha(masked),
                    "original_input_f32_sha256": tensor_sha(unmasked),
                    "condition_with_position_f32_sha256": tensor_sha(
                        chunks[k].unsqueeze(0).to("mps") + speech.positions
                    ),
                }
                if i - start in (0, 10):
                    generated, proof = speech.render(scrop, chunks[k], seed)
                    proofs[c].update(proof)
                    variants[c] = (
                        cv2.warpAffine(
                            generated,
                            cv2.invertAffineTransform(smatrix),
                            (table.width, table.height),
                            flags=cv2.INTER_LINEAR,
                        ),
                        alpha_from_face(rgb.shape, faces[0]),
                    )
            fresh[i] = (variants, binding, proofs, faces[0])
        for case in [c for c in manifest["items"] if c["group"] == source["group"]]:
            record = {
                "id": case["id"],
                "status": "failed",
                "native_frames": 0,
                "masks": 0,
                "neural_bindings": 0,
                "neural_replays": 0,
            }
            try:
                movie = manifest_path.parent / case["path"]
                c = case["condition"]
                if (
                    case["status"] != "ready"
                    or case["role"] != "fit"
                    or case["label"] != int(c in AI_CONDITIONS)
                    or digest(movie) != case["sha256"]
                    or case["source_clock_span"] != span
                    or case["original_audio_mapping"] != audio.mapping
                ):
                    raise ValueError("Prepared role/label/bytes/source-clock mismatch")
                if case["edit_frame_range"] != [start, end] or case["edit_time_range_s"] != [
                    start / 25,
                    end / 25,
                ]:
                    raise ValueError("Known edit boundaries changed")
                clock = probe_frames(movie)
                if len(clock) != n or any(abs(clock.seconds(i) - i / 25) > 1e-9 for i in range(n)):
                    raise ValueError("Encoded source-clock mismatch")
                masks = {m["frame_index"]: m for m in case["masks"]}
                if set(masks) != set(range(start, end)):
                    raise ValueError("Missing or extra edit mask")
                frames = list(decode_frames(movie, clock, range(n)))
                if len(frames) != n:
                    raise ValueError("Missing decoded picture")
                for i, frame in enumerate(frames):
                    rgb = original[i]
                    if case["original_rgb_sha256"][i] != hashlib.sha256(rgb.tobytes()).hexdigest():
                        raise ValueError("Original RGB binding changed")
                    if i not in masks:
                        if not np.array_equal(frame.rgb, rgb):
                            raise ValueError("Pixels outside the edit interval changed")
                    else:
                        m = masks[i]
                        variants, binding, proofs, face = fresh[i]
                        opacity = min(1.0, (i - start + 1) / 2, (end - i) / 2)
                        if (
                            any(m[k] != v for k, v in binding.items())
                            or m["opacity"] != opacity
                            or m["pts"] != table.pts[i]
                            or m["source_time_s"] != table.seconds(i)
                            or m["source_rgb_sha256"] != hashlib.sha256(rgb.tobytes()).hexdigest()
                        ):
                            raise ValueError("Operation/input geometry or original time changed")
                        mask_path = manifest_path.parent / m["path"]
                        changed = np.any(frame.rgb != rgb, axis=2)
                        if (
                            digest(mask_path) != m["sha256"]
                            or not np.array_equal(
                                changed, cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
                            )
                            or int(changed.sum()) != m["changed_pixels"]
                            or bool(changed.any()) != (c != "original")
                        ):
                            raise ValueError("Changed-pixel mask mismatch")
                        if c in AI_CONDITIONS:
                            if any(m["model_proof"][k] != v for k, v in proofs[c].items()):
                                raise ValueError("Neural construction input/output binding mismatch")
                            alpha = (
                                mouth_alpha(rgb.shape, mesh[i])
                                if c.startswith("ai-")
                                else alpha_from_face(rgb.shape, face)
                            )
                            if not np.array_equal(frame.rgb[alpha == 0], rgb[alpha == 0]):
                                raise ValueError("Neural edit changed pixels outside alpha support")
                            record["neural_bindings"] += 1
                        if c == "original":
                            replay = rgb
                        elif c in variants:
                            replay = composite(rgb, *variants[c], opacity)
                        else:
                            replay = None
                        if replay is not None:
                            if not np.array_equal(frame.rgb, replay):
                                raise ValueError("Ordinary or sampled neural pixels did not reproduce")
                            record["neural_replays"] += int(c in AI_CONDITIONS)
                        record["masks"] += 1
                    record["native_frames"] += 1
                sound = decode_audio(movie, source_sha256=case["sha256"])
                a, b = span["source_sample_range"]
                if not np.array_equal(sound.pcm, audio.pcm[a:b]) or len(sound.pcm) != n * 640:
                    raise ValueError("Original selected PCM differs")
                pcm_samples += len(sound.pcm)
                record["status"] = "ok"
            except Exception as error:
                record["reason"] = f"{type(error).__name__}: {error}"
            native_frames += record["native_frames"]
            exact_masks += record["masks"]
            neural_bindings += record["neural_bindings"]
            neural_replays += record["neural_replays"]
            records.append(record)
        print(
            json.dumps(
                {
                    "source": source["id"],
                    "checked": len(records),
                    "passed": sum(r["status"] == "ok" for r in records),
                }
            ),
            flush=True,
        )
    body = {
        "schema": "chipmunk.mouth-temporal-source-audit/11",
        "plan_sha256": digest(plan_path),
        "manifest_sha256": digest(manifest_path),
        "implementation_sha256": digest(Path(__file__)),
        "planned": 160,
        "passed": sum(r["status"] == "ok" for r in records),
        "native_frames": native_frames,
        "exact_masks": exact_masks,
        "common_pcm_samples": pcm_samples,
        "neural_input_bindings": neural_bindings,
        "sampled_neural_output_replays": neural_replays,
        "cases": records,
        "scope": "All native pixels/clocks/masks/PCM; every ordinary operation and neural input binding; neural outputs at offsets0/10 per source/condition. Not every neural output rerun or outside validation.",
    }
    write(output, body)
    print(json.dumps({k: v for k, v in body.items() if k != "cases"}), flush=True)
    if body["passed"] != 160 or neural_bindings != 1280 or neural_replays != 128:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("assets", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    audit(a.plan, a.manifest, a.assets, a.output)
