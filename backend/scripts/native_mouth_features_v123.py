"""Blind full-native mouth-image and cached speech features; no classifier fit."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import subprocess
import types
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import mouth_articulation_core_v122 as interaction
from scripts import native_mouth_clips_v120 as clips
from scripts import review_native_mouth_clips_v121 as audit
from scripts import video_appearance_runtime_v108 as runtime
from scripts.mouth_probe_v6 import features as native_features
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = clips.ROOT, clips.BASE, clips.PANEL
TRIAL = BASE / "native-mouth-features-v123"
CACHE = clips.CACHE.parent / "native-mouth-features-v123"
PUBLIC = PANEL / "native-mouth-feature-freeze-v123.json"
CODE = (
    "backend/scripts/native_mouth_features_v123.py",
    "backend/tests/test_native_mouth_features_v123.py",
    "docs/native-mouth-features-v123-method.md",
    "backend/scripts/mouth_probe_v6.py",
    "backend/scripts/mouth_articulation_core_v122.py",
    "backend/app/detectors/image/faces.py",
)
read, write, sha, require = clips.read, clips.write, clips.sha, clips.require


class UnsupportedMouth(ValueError):
    pass


def mouth_crop(rgb, face):
    require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, "Native RGB picture required")
    points = np.asarray(face.landmarks, np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise UnsupportedMouth("Five finite native mouth/face landmarks unavailable")
    distance = float(np.linalg.norm(points[4] - points[3]))
    if distance < 8:
        raise UnsupportedMouth("Native mouth-corner distance below eight pixels")
    cx, cy = (points[3] + points[4]) / 2
    height, width = rgb.shape[:2]
    x0, y0 = max(0, math.floor(cx - 0.9 * distance)), max(0, math.floor(cy - 0.6 * distance))
    x1, y1 = min(width, math.ceil(cx + 0.9 * distance)), min(height, math.ceil(cy + 0.6 * distance))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise UnsupportedMouth("Clipped native mouth region lacks eight-pixel support")
    return np.ascontiguousarray(rgb[y0:y1, x0:x1]), [x0, y0, x1, y1]


def face_record(face):
    return {
        "box": list(face.box()),
        "landmarks": [list(p) for p in face.landmarks],
        "score_hex": float(face.score).hex(),
    }


def model_records():
    from app.engine.fetchers import file_path
    from app.engine.registry import get_registry

    result = {}
    for name in ("dinov2.small", "yunet.2023mar"):
        entry = get_registry().manifest.model(name)
        require(entry.licence.class_ == "commercial", "Pinned commercial model record required")
        files = []
        for f in entry.source.files:
            path = file_path(entry, f)
            require(
                path.is_file() and path.stat().st_size == f.bytes and sha(path) == f.sha256,
                "Exact cached model bytes required; no download",
            )
            files.append({"name": f.path, "bytes": f.bytes, "sha256": f.sha256})
        result[name] = {
            "license": entry.licence.model_dump(mode="json"),
            "revision": entry.source.revision,
            "files": files,
        }
    return result


class Encoder:
    def __init__(self):
        import torch
        from safetensors.torch import load_file
        from transformers import Dinov2Config, Dinov2Model

        from app.engine.fetchers import file_path
        from app.engine.registry import get_registry

        model_records()
        require(torch.backends.mps.is_available(), "Explicit MPS feature path required; no fallback")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.mps.set_per_process_memory_fraction(0.65)
        entry = get_registry().manifest.model("dinov2.small")
        paths = {f.path: file_path(entry, f) for f in entry.source.files}
        config = Dinov2Config.from_dict(read(paths["config.json"]))
        config._attn_implementation = "eager"
        model = Dinov2Model(config).to(device="cpu", dtype=torch.float32)
        state = load_file(paths["model.safetensors"], device="cpu")
        require(
            len(state) == 223 and all(v.dtype == torch.float32 for v in state.values()),
            "Pinned 223 float32 image tensors required",
        )
        require(set(state) == set(model.state_dict()), "Complete exact image encoder tensor keys required")
        model.load_state_dict(state, strict=True)
        require(
            all(torch.equal(value, model.state_dict()[key]) for key, value in state.items()),
            "Strict image encoder load changed tensor values",
        )
        self.tensor_count = len(state)
        del state
        self.model = model.to(device="mps", dtype=torch.float32).eval().requires_grad_(False)
        self.device = "mps"
        self.image_inputs = 0

    def encode(self, crop):
        self.image_inputs += 1
        values, hashes = native_features([crop], self)
        require(
            values.shape == (1, 768) and values.dtype == np.float32 and np.isfinite(values).all(),
            "Complete finite native mouth-image representation required",
        )
        return values[0].copy(), hashes[0]


def detect(rgb, detector, encoder):
    import cv2

    from app.detectors.image.faces import detect_faces

    faces = detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
    if len(faces) != 1:
        return {"status": "unassessed", "reason": "Unique eligible native face unavailable"}, None
    try:
        crop, bounds = mouth_crop(rgb, faces[0])
    except UnsupportedMouth as exc:
        return {"status": "unassessed", "reason": str(exc)}, None
    vector, input_hash = encoder.encode(crop)
    return {
        "status": "ok",
        "region_bbox": bounds,
        "face": face_record(faces[0]),
        "crop_rgb_sha256": clips.inputs.speech.speech.fingerprint(crop),
        "model_input_float32_sha256": input_hash,
        "feature_float32_sha256": clips.inputs.speech.speech.fingerprint(vector),
    }, vector


def qualified_bank():
    clips.verify()
    audit.verify()
    monitor = runtime.TRIAL / "monitor"
    m, p = read(monitor / "output/result.json"), read(monitor / "profile.json")
    require(
        m["status"] == "complete"
        and m["owned_descendant_sampled"]
        and m["sample_helper_sha256"] == sha(ROOT / "backend/scripts/audio_tree_sampler_v64.py")
        and p["child_exit_code"] == 0
        and p["guard"] is None
        and p["sample_errors"] == 0
        and p["worker_receipt_sha256"] == sha(monitor / "output/result.json"),
        "Qualified exact owned-descendant monitor required",
    )
    generation = read(PANEL / "native-mouth-clips-complete-v120.json")
    reviewed = read(PANEL / "native-mouth-clips-audit-complete-v121.json")
    require(
        generation["status"] == reviewed["status"] == "complete"
        and generation["ready_cases"] == reviewed["ready_cases"] == 283
        and reviewed["edit_picture_masks"] == 4700,
        "Complete source/mask audited input bank required",
    )
    for i in range(8):
        f = audit.TRIAL / f"bind-{i:02d}"
        r, p = read(f / "output/result.json"), read(f / "profile.json")
        require(
            r["status"] == "complete"
            and p["child_exit_code"] == 0
            and p["guard"] is None
            and p["sample_errors"] == 0
            and p["worker_receipt_sha256"] == sha(f / "output/result.json"),
            "Successful exact source replay required",
        )


def identity():
    qualified_bank()
    return {
        "schema": "chipmunk.native-mouth-feature-freeze/123",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "construction_complete_sha256": sha(PANEL / "native-mouth-clips-complete-v120.json"),
        "source_mask_audit_complete_sha256": sha(PANEL / "native-mouth-clips-audit-complete-v121.json"),
        "model_records": model_records(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "safetensors", "numpy", "opencv-python-headless")
        },
        "image_device": "mps",
        "image_dtype": "float32",
        "attention": "eager",
        "image_batch_size": 1,
        "qualification_image_inputs": 5,
        "qualification_face_inputs": 5,
        "feature_kind": "768 image variables; 3840 cached speech summary variables; no probability",
        "scope": "All native pictures, content-only per-source memoization and exact audio-source routes. No true edit interval/mask/label selects inputs or enters vectors. No classifier fit, detector accuracy or serving contribution.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "new_neural_calls": 0}), flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Published unchanged full-native feature contract required",
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:4] == ["-B", "-m", "scripts.video_appearance_native_v108"]
        and argv[4] in {"authored", "extract"},
        "Only fixed feature qualification/extraction can be routed",
    )
    routed = list(argv)
    routed[3] = "scripts.native_mouth_features_v123"
    return subprocess.Popen(routed, **kwargs)


def run(command, batch=None):
    require(command in {"authored", "extract"}, "One fixed feature stage required")
    if command == "extract":
        require(type(batch) is int and 0 <= batch < 8, "One fixed source batch required")
        f = TRIAL / "authored"
        r, p = read(f / "output/result.json"), read(f / "profile.json")
        require(
            r["status"] == "complete"
            and p["child_exit_code"] == 0
            and p["guard"] is None
            and p["sample_errors"] == 0
            and p["worker_receipt_sha256"] == sha(f / "output/result.json"),
            "Successful native encoder qualification required",
        )
    else:
        require(batch is None, "Qualification has no batch")
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda command, batch: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )(command, batch)


def detector():
    import cv2

    from app.detectors.image.faces import load_yunet

    cv2.setNumThreads(1)
    return load_yunet(ROOT / "models/weights/face_detection_yunet_2023mar.onnx")


def case_rows(batch):
    require(type(batch) is int and 0 <= batch < 8, "One fixed source batch required")
    f = clips.TRIAL / f"extract-{batch:02d}/output/cases.jsonl"
    receipt = read(f.parent / "result.json")
    require(receipt["cases_sha256"] == sha(f), "Exact constructed condition journal required")
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    require(len(rows) == 36, "All planned conditions required")
    return rows


def verified_frames(case, table, indices):
    from app.detectors.video.decode import decode_frames

    require(
        case["status"] == "ready" and sha(Path(case["file"])) == case["file_sha256"],
        "Exact audited condition clip required",
    )
    wanted = list(indices)
    seen = []
    for frame in decode_frames(Path(case["file"]), table, wanted, timeout_s=120):
        require(
            clips.inputs.speech.speech.fingerprint(frame.rgb) == case["expected"]["rgb_sha256"][frame.index],
            "Audited native RGB picture changed",
        )
        seen.append(frame.index)
        yield frame
    require(seen == wanted, "Complete requested native pictures required")


def qualify(output):
    from app.detectors.video.decode import FrameTable

    encoder, face_detector = Encoder(), detector()
    source = read(clips.TRIAL / "plan.json")[0]["source"]
    require(source["study_role"] == "fusion_fit", "Fixed fitting-source qualification required")
    rows = case_rows(0)
    original = rows[0]
    edited = rows[4]
    require(
        original["condition"] == "original" and edited["condition"] == "ai_aligned",
        "Fixed two qualification inputs required",
    )
    table = FrameTable.from_json(source["table"])
    center = len(table) // 2
    chosen = list(range(center - 1, center + 2))
    vectors = []
    proofs = []
    face_inputs = 0
    for frame in verified_frames(original, table, chosen):
        meta, vector = detect(frame.rgb, face_detector, encoder)
        face_inputs += 1
        require(meta["status"] == "ok", "Qualification source mouth unavailable; no replacement")
        vectors.append(vector)
        proofs.append(meta)
    frame = list(verified_frames(original, table, [center]))[0]
    again, repeat = detect(frame.rgb, face_detector, encoder)
    face_inputs += 1
    require(
        again == proofs[1] and repeat.tobytes() == vectors[1].tobytes(),
        "Native face/crop/input/feature repeat differs",
    )
    frame = list(verified_frames(edited, table, [center]))[0]
    altered, other = detect(frame.rgb, face_detector, encoder)
    face_inputs += 1
    require(altered["status"] == "ok", "Fixed constructed mouth unavailable; no replacement")
    clocks = clips.inputs.speech.source_records()[0][0]["frames"]
    plan = next(f for f in clocks if f["video_frame_index"] == center)
    require(
        sha(Path(source["hidden_file"])) == source["hidden_file_sha256"],
        "Qualified speech state bytes changed",
    )
    hidden = np.load(source["hidden_file"], allow_pickle=False)
    speech = interaction.speech_summary(hidden, plan)
    visual = interaction.temporal_visual(*vectors, [str(Fraction(table.pts[i]) * table.tb) for i in chosen])
    write(output / "private-proof.json", {"original": proofs, "repeat": again, "altered": altered})
    require(encoder.image_inputs == face_inputs == 5, "Exactly five image and five face inputs required")
    result = {
        "schema": "chipmunk.native-mouth-feature-contract/123",
        "status": "complete",
        "strict_encoder_tensor_count": encoder.tensor_count,
        "native_image_inputs": encoder.image_inputs,
        "face_inputs": face_inputs,
        "new_neural_calls": encoder.image_inputs + face_inputs,
        "repeat_face_crop_input_features_bitwise": True,
        "visual_descriptor_shape": list(visual.shape),
        "speech_descriptor_shape": list(speech.shape),
        "source_to_constructed_feature_l2": float(np.linalg.norm(vectors[1].astype(np.float64) - other)),
        "private_proof_sha256": sha(output / "private-proof.json"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "defaults_changed": False,
        "scope": "One-fitting-source native encoder/crop/repeat/descriptor contract. Image response distance is not an AI score or detector accuracy. No head/calibration fitted.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def extract(batch, output):
    from app.detectors.video.decode import FrameTable
    from scripts.av_conditioning_clock_v109 import frame_plan

    require(type(batch) is int and 0 <= batch < 8, "One fixed source batch required")
    encoder, face_detector = Encoder(), detector()
    folder = CACHE / f"batch-{batch:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    rows = case_rows(batch)
    plans = read(clips.TRIAL / "plan.json")[batch * 6 : (batch + 1) * 6]
    parents = {
        r["asset_index"]: r for r in read(BASE / "av-sentence-support-v104/preparation/plan.json")["records"]
    }
    face_inputs = 0
    cache_hits = 0
    case_records = []
    with (output / "feature-cases.jsonl").open("x") as journal:
        for offset, plan in enumerate(plans):
            source = plan["source"]
            sub = folder / f"source-{offset:02d}"
            sub.mkdir(exist_ok=False)
            table = FrameTable.from_json(source["table"])
            hidden_file = Path(source["hidden_file"])
            require(sha(hidden_file) == source["hidden_file_sha256"], "Qualified speech states changed")
            hidden = np.load(hidden_file, allow_pickle=False)
            c = parents[source["asset_index"]]["clock_support"]
            audio_indices = []
            audio_rows = []
            audio_proofs = []
            for i, t in enumerate(table.pts):
                ap = frame_plan(
                    i,
                    str(Fraction(t) * table.tb),
                    c["source_audio_origin_seconds"],
                    c["source_video_origin_seconds"],
                    plan["waveform"]["sample_frames"],
                    c["native_audio_clock_quantization_allowance_seconds"],
                )
                if ap["source_sample_status"] == "supported" and all(ap["retained_tick_slots"]):
                    audio_indices.append(i)
                    audio_rows.append(interaction.speech_summary(hidden, ap))
                    audio_proofs.append(ap)
            audio_file = sub / "speech.npy"
            with audio_file.open("xb") as file:
                np.save(
                    file,
                    np.stack(audio_rows) if audio_rows else np.empty((0, interaction.AUDIO_DIM)),
                    allow_pickle=False,
                )
            audio_ref = {
                "file": str(audio_file),
                "sha256": sha(audio_file),
                "indices": audio_indices,
                "plans": audio_proofs,
                "shape": [len(audio_rows), interaction.AUDIO_DIM],
            }
            del hidden, audio_rows
            cache = {}
            for local, case in enumerate(rows[offset * 6 : (offset + 1) * 6]):
                require(
                    (case["asset_index"], case["source_group"], case["study_role"], case["condition"])
                    == (
                        source["asset_index"],
                        source["source_group"],
                        source["study_role"],
                        clips.CONDITIONS[local],
                    ),
                    "Exact source/role/condition order required",
                )
                record = {
                    "asset_index": source["asset_index"],
                    "source_group": source["source_group"],
                    "study_role": source["study_role"],
                    "condition": case["condition"],
                    "input_status": case["status"],
                    "speech": audio_ref,
                }
                if case["status"] != "ready":
                    record.update(status="unavailable", reason=case["reason"])
                else:
                    records = []
                    values = []
                    for frame in verified_frames(case, table, range(len(table))):
                        rgb_hash = clips.inputs.speech.speech.fingerprint(frame.rgb)
                        if rgb_hash in cache:
                            meta, vector = cache[rgb_hash]
                            cache_hits += 1
                        else:
                            meta, vector = detect(frame.rgb, face_detector, encoder)
                            face_inputs += 1
                            cache[rgb_hash] = (meta, vector)
                        row = {
                            **meta,
                            "frame_index": frame.index,
                            "pts": frame.pts,
                            "rgb_sha256": rgb_hash,
                            "feature_row": None,
                        }
                        if vector is not None:
                            row["feature_row"] = len(values)
                            values.append(vector)
                        records.append(row)
                    require(
                        [r["frame_index"] for r in records] == list(range(len(table))),
                        "Complete blind native picture sampling required",
                    )
                    file = sub / f"condition-{local:02d}.npy"
                    with file.open("xb") as stream:
                        np.save(
                            stream,
                            np.stack(values) if values else np.empty((0, 768), np.float32),
                            allow_pickle=False,
                        )
                    record.update(
                        status="extracted",
                        file=str(file),
                        file_sha256=sha(file),
                        table=table.to_json(),
                        frames=records,
                        source_clip_sha256=case["file_sha256"],
                        sampling="all_native_pictures",
                        shape=[len(values), 768],
                    )
                case_records.append(record)
                journal.write(json.dumps(record, allow_nan=False) + "\n")
                journal.flush()
            del cache
            print(
                json.dumps({"completed_source_ordinal": batch * 6 + offset, "condition_records": 6}),
                flush=True,
            )
    result = {
        "schema": "chipmunk.native-mouth-feature-extraction/123",
        "status": "complete",
        "batch": batch,
        "planned_cases": 36,
        "extracted_cases": sum(r["status"] == "extracted" for r in case_records),
        "unavailable_cases": sum(r["status"] == "unavailable" for r in case_records),
        "native_pictures": sum(len(r.get("frames", [])) for r in case_records),
        "assessed_mouth_pictures": sum(r.get("shape", [0])[0] for r in case_records),
        "face_inputs": face_inputs,
        "image_inputs": encoder.image_inputs,
        "exact_rgb_cache_hits": cache_hits,
        "new_neural_calls": face_inputs + encoder.image_inputs,
        "feature_case_journal_sha256": sha(output / "feature-cases.jsonl"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "defaults_changed": False,
        "scope": "Blind all-native image/cached-speech features only; no edit masks/bounds/labels select inputs or enter vectors. No head, detector score, calibration or accuracy. Missing scope is not originality.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "authored", "extract"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze has no arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) in (1, 2), "Exact stage/batch required")
        run(args.arguments[0], None if len(args.arguments) == 1 else int(args.arguments[1]))
    else:
        verify()
        require(
            len(args.arguments) == (1 if args.command == "authored" else 2), "Exact guarded output required"
        )
        output = Path(args.arguments[0])
        require(output.resolve().is_relative_to(TRIAL.resolve()), "Exclusive private output required")
        output.mkdir(exist_ok=False)
        qualify(output) if args.command == "authored" else extract(int(args.arguments[1]), output)
