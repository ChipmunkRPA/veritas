"""Data-only replay of native mouth pixels, array rows and speech-time routes."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import types
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import native_mouth_features_v123 as features
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = features.ROOT, features.BASE, features.PANEL
TRIAL = BASE / "native-mouth-feature-review-v124"
PUBLIC = PANEL / "native-mouth-feature-review-freeze-v124.json"
CODE = (
    "backend/scripts/review_native_mouth_features_v124.py",
    "backend/tests/test_review_native_mouth_features_v124.py",
    "docs/native-mouth-feature-review-v124-method.md",
)
read, write, sha, require = features.read, features.write, features.sha, features.require
fingerprint = features.clips.inputs.speech.speech.fingerprint


def array_file(record, dtype):
    path = Path(record["file"])
    require(sha(path) == record.get("file_sha256", record.get("sha256")), "Saved feature array bytes changed")
    values = np.load(path, allow_pickle=False)
    require(
        values.dtype == dtype and list(values.shape) == record["shape"] and np.isfinite(values).all(),
        "Exact finite array dtype/dimensions required",
    )
    return values


def pixel_replay(rgb, metadata, vector):
    """Rebuild crop/input from recorded native landmarks; no face inference."""
    import cv2

    points = np.asarray(metadata["face"]["landmarks"], np.float64)
    require(points.shape == (5, 2) and np.isfinite(points).all(), "Complete native landmarks required")
    x, y, w, h = metadata["face"]["box"]
    require(w >= 48 and h >= 48 and x >= 0 and y >= 0, "Recorded eligible native face required")
    distance = math.hypot(*(points[4] - points[3]))
    require(distance >= 8, "Sufficient recorded native mouth separation required")
    cx, cy = (points[3] + points[4]) / 2
    bounds = [
        max(0, math.floor(cx - 0.9 * distance)),
        max(0, math.floor(cy - 0.6 * distance)),
        min(rgb.shape[1], math.ceil(cx + 0.9 * distance)),
        min(rgb.shape[0], math.ceil(cy + 0.6 * distance)),
    ]
    require(
        bounds == metadata["region_bbox"] and bounds[2] - bounds[0] >= 8 and bounds[3] - bounds[1] >= 8,
        "Recorded region differs from native landmark crop",
    )
    crop = rgb[bounds[1] : bounds[3], bounds[0] : bounds[2]]
    require(fingerprint(crop) == metadata["crop_rgb_sha256"], "Recorded crop pixels differ")
    pixels = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], np.float32)
    std = np.asarray([0.229, 0.224, 0.225], np.float32)
    tensor = np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1))
    require(fingerprint(tensor) == metadata["model_input_float32_sha256"], "Normalized image input differs")
    require(
        vector.shape == (768,)
        and vector.dtype == np.float32
        and np.isfinite(vector).all()
        and fingerprint(vector) == metadata["feature_float32_sha256"],
        "Saved picture/feature row differs",
    )
    require(
        all(abs(float(np.linalg.norm(v)) - 1) < 5e-6 for v in (vector[:384], vector[384:])),
        "Normalized feature branches required",
    )


def speech_replay(hidden, plan):
    """Independent direct ten-tick/five-state mean/std, without the v122 helper."""
    ticks = plan["tick_indices"]
    require(
        plan["source_sample_status"] == "supported"
        and len(ticks) == 10
        and all(plan["retained_tick_slots"])
        and all(0 <= t < plan["retained_tick_prefix"] <= len(hidden) for t in ticks),
        "Observed full speech support required",
    )
    selected = np.asarray([hidden[t] for t in ticks], np.float64)
    return np.r_[selected.mean(axis=0).reshape(-1), selected.std(axis=0).reshape(-1)]


def identity():
    features.verify()
    return {
        "schema": "chipmunk.native-mouth-feature-review-freeze/124",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "feature_freeze_sha256": sha(features.PUBLIC),
        "scope": "Full native picture/crop/input hashes, array rows, cache identity and cached speech clocks/summaries. Data-only; no independent face or image/speech encoder forwards, detector scores or accuracy.",
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
        "Unchanged feature replay contract required",
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
        "Only fixed data replay permitted",
    )
    args = list(argv)
    args[3] = "scripts.review_native_mouth_features_v124"
    return subprocess.Popen(args, **kwargs)


def completed_batch(batch):
    require(type(batch) is int and 0 <= batch < 8, "One fixed batch required")
    folder = features.TRIAL / f"extract-{batch:02d}"
    r, p = read(folder / "output/result.json"), read(folder / "profile.json")
    require(
        r["status"] == "complete"
        and r["batch"] == batch
        and p["child_exit_code"] == 0
        and p["guard"] is None
        and p["sample_errors"] == 0
        and p["worker_receipt_sha256"] == sha(folder / "output/result.json")
        and r["feature_case_journal_sha256"] == sha(folder / "output/feature-cases.jsonl")
        and r["freeze_sha256"] == sha(features.TRIAL / "freeze.json"),
        "Successful exact feature extraction required",
    )
    return folder, r


def run(batch):
    completed_batch(batch)
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        features.runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda c, b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", batch)


def review(batch, output):
    from app.detectors.video.decode import FrameTable, decode_frames
    from scripts.av_conditioning_clock_v109 import frame_plan

    folder, receipt = completed_batch(batch)
    rows = [json.loads(line) for line in (folder / "output/feature-cases.jsonl").read_text().splitlines()]
    cases = features.case_rows(batch)
    plans = read(features.clips.TRIAL / "plan.json")[batch * 6 : (batch + 1) * 6]
    parents = {
        r["asset_index"]: r for r in read(BASE / "av-sentence-support-v104/preparation/plan.json")["records"]
    }
    require(len(rows) == len(cases) == 36, "Complete condition journal required")
    picture_count = assessed = face_inputs = image_inputs = cache_hits = speech_positions = 0
    with (output / "reviewed-cases.jsonl").open("x") as journal:
        for offset, source_plan in enumerate(plans):
            source = source_plan["source"]
            table = FrameTable.from_json(source["table"])
            hidden_path = Path(source["hidden_file"])
            require(sha(hidden_path) == source["hidden_file_sha256"], "Cached speech states changed")
            hidden = np.load(hidden_path, allow_pickle=False)
            require(
                hidden.dtype == np.float32
                and hidden.ndim == 3
                and hidden.shape[1:] == (5, 384)
                and np.isfinite(hidden).all(),
                "Qualified speech states required",
            )
            c = parents[source["asset_index"]]["clock_support"]
            expected_plans = []
            for i, t in enumerate(table.pts):
                p = frame_plan(
                    i,
                    str(Fraction(t) * table.tb),
                    c["source_audio_origin_seconds"],
                    c["source_video_origin_seconds"],
                    source_plan["waveform"]["sample_frames"],
                    c["native_audio_clock_quantization_allowance_seconds"],
                )
                if p["source_sample_status"] == "supported" and all(p["retained_tick_slots"]):
                    expected_plans.append(p)
            audio_ref = rows[offset * 6]["speech"]
            audio = array_file(audio_ref, np.float64)
            require(
                audio_ref["plans"] == expected_plans
                and audio_ref["indices"] == [p["video_frame_index"] for p in expected_plans]
                and audio.shape == (len(expected_plans), 3840),
                "Exact all-native supported speech positions required",
            )
            for n, p in enumerate(expected_plans):
                require(
                    np.array_equal(audio[n], speech_replay(hidden, p)),
                    "Cached speech feature/time route differs",
                )
            speech_positions += len(expected_plans)
            cache = {}
            for local in range(6):
                row, case = rows[offset * 6 + local], cases[offset * 6 + local]
                require(
                    all(row[k] == case[k] for k in ("asset_index", "source_group", "study_role", "condition"))
                    and row["condition"] == features.clips.CONDITIONS[local]
                    and row["input_status"] == case["status"]
                    and row["speech"] == audio_ref,
                    "Exact source/condition/role and shared soundtrack required",
                )
                count = 0
                if case["status"] != "ready":
                    require(
                        row["status"] == "unavailable"
                        and row["reason"] == case["reason"]
                        and "file" not in row,
                        "Unavailable input must stay unavailable",
                    )
                else:
                    require(
                        row["status"] == "extracted"
                        and row["sampling"] == "all_native_pictures"
                        and row["table"] == table.to_json()
                        and row["source_clip_sha256"] == case["file_sha256"]
                        and sha(Path(case["file"])) == case["file_sha256"],
                        "Exact full native input and clock required",
                    )
                    values = array_file(row, np.float32)
                    used = []
                    seen = []
                    require(len(row["frames"]) == len(table), "Every native picture must have a record")
                    for frame in decode_frames(Path(case["file"]), table, range(len(table)), timeout_s=120):
                        seen.append(frame.index)
                        r = row["frames"][frame.index]
                        rgb_hash = fingerprint(frame.rgb)
                        require(
                            (r["frame_index"], r["pts"], r["rgb_sha256"])
                            == (frame.index, table.pts[frame.index], rgb_hash)
                            and rgb_hash == case["expected"]["rgb_sha256"][frame.index],
                            "Native picture, index or time differs",
                        )
                        if r["status"] == "ok":
                            n = r["feature_row"]
                            require(type(n) is int and 0 <= n < len(values), "Actual feature row required")
                            pixel_replay(frame.rgb, r, values[n])
                            used.append(n)
                            count += 1
                        else:
                            require(
                                r["status"] == "unassessed"
                                and r["feature_row"] is None
                                and bool(r["reason"]),
                                "Missing scope must remain unassessed",
                            )
                        signature = {
                            k: v
                            for k, v in r.items()
                            if k not in {"frame_index", "pts", "rgb_sha256", "feature_row"}
                        }
                        if rgb_hash in cache:
                            require(
                                cache[rgb_hash] == signature,
                                "Byte-identical input cache changed metadata/features",
                            )
                            cache_hits += 1
                        else:
                            cache[rgb_hash] = signature
                            face_inputs += 1
                            image_inputs += int(r["status"] == "ok")
                        picture_count += 1
                    require(
                        seen == list(range(len(table))) and used == list(range(len(values))),
                        "Complete native decode and one-to-one eligible feature rows required",
                    )
                assessed += count
                journal.write(json.dumps({"status": row["status"], "assessed_pictures": count}) + "\n")
                journal.flush()
            del hidden, audio, cache
    require(
        (picture_count, assessed, face_inputs, image_inputs, cache_hits)
        == tuple(
            receipt[k]
            for k in (
                "native_pictures",
                "assessed_mouth_pictures",
                "face_inputs",
                "image_inputs",
                "exact_rgb_cache_hits",
            )
        ),
        "Full extraction counters differ",
    )
    result = {
        "schema": "chipmunk.native-mouth-feature-review/124",
        "status": "complete",
        "batch": batch,
        "cases": 36,
        "native_pictures": picture_count,
        "assessed_mouth_pictures": assessed,
        "unique_source_speech_positions": speech_positions,
        "face_inputs_reconciled": face_inputs,
        "image_inputs_reconciled": image_inputs,
        "cache_hits_reconciled": cache_hits,
        "private_review_journal_sha256": sha(output / "reviewed-cases.jsonl"),
        "extraction_receipt_sha256": sha(folder / "output/result.json"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "Complete native RGB/crop/preprocessor/array-row and speech-summary/time replay. Recorded landmarks and vectors are not independently re-inferred; no model accuracy, achieved sync or calibration claim.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "run", "bind"))
    p.add_argument("arguments", nargs="*")
    args = p.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze takes no arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One batch required")
        run(int(args.arguments[0]))
    else:
        verify()
        require(len(args.arguments) == 2, "Exact output and batch required")
        out = Path(args.arguments[0])
        batch = int(args.arguments[1])
        require(
            0 <= batch < 8 and out.resolve().is_relative_to(TRIAL.resolve()),
            "Fixed exclusive output required",
        )
        out.mkdir(exist_ok=False)
        review(batch, out)
