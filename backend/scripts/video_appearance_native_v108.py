"""Bind and measure the fixed selection-source appearance transfer comparison.

Private source records/features; public aggregates only. No serving changes.
Neural commands are called only by the separately guarded runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
SOURCE = BASE / "video-native-inserts-v95"
PREP = BASE / "video-appearance-plan-v107/preparation"
TRIAL = BASE / "video-appearance-native-v108"
PANEL = ROOT / "docs/evaluation-panels"
PUBLIC = PANEL / "video-appearance-native-freeze-v108.json"
FITTED = BASE / "video-joint-scalars-v106/evaluation/fitted-models.json"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
CODE = (
    "backend/scripts/video_appearance_native_v108.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/tests/test_video_appearance_native_v108.py",
    "docs/video-appearance-native-v108-method.md",
    "backend/scripts/video_appearance_plan_v107.py",
    "backend/scripts/video_joint_scalar_v106.py",
    "backend/scripts/measure_video_specialists_v18.py",
    "backend/scripts/review_video_native_inserts_v96.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/app/detectors/video/decode.py",
    "backend/app/detectors/video/face_appearance.py",
    "backend/app/detectors/image/faces.py",
    "backend/app/bench/specialists.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprint(value):
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def identity():
    previous = read(PANEL / "video-appearance-plan-freeze-v107.json")
    require(
        all(sha(ROOT / p) == h for p, h in previous["code_sha256"].items()),
        "Frozen support implementation changed",
    )
    require(
        sha(SOURCE / "plan.json") == previous["constructor_plan_sha256"]
        and sha(SOURCE / "freeze.json") == previous["constructor_freeze_sha256"],
        "Support constructor inputs changed",
    )
    require(
        sha(PANEL / "video-face-specialists-v20-all-artifact.json") == previous["fixed_local_heads_sha256"],
        "Fixed local heads changed",
    )
    require(
        sha(PREP / "plan.json") == read(PREP / "result.json")["private_plan_sha256"],
        "Prepared support changed",
    )
    require(sha(FITTED) == previous["candidate_fitted_models_sha256"], "Sealed candidate changed")
    for record in previous["model_records"].values():
        require(record["licence"]["class_"] == "commercial", "Commercial model record required")
        require(all(sha(ROOT / p) == h for p, h in record["files"].items()), "Cached model changed")
    return {
        "schema": "chipmunk.video-appearance-native-freeze/108",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "support_plan_sha256": sha(PREP / "plan.json"),
        "support_result_sha256": sha(PREP / "result.json"),
        "support_freeze_sha256": sha(PANEL / "video-appearance-plan-freeze-v107.json"),
        "source_plan_sha256": sha(SOURCE / "plan.json"),
        "source_freeze_sha256": sha(SOURCE / "freeze.json"),
        "fitted_models_sha256": sha(FITTED),
        "local_heads_sha256": sha(PANEL / "video-face-specialists-v20-all-artifact.json"),
        "model_records": previous["model_records"],
        "packages": {
            p: importlib.metadata.version(p) for p in ("numpy", "scipy", "opencv-python-headless", "psutil")
        },
        "binary_sha256": {p: sha(Path(shutil.which(p))) for p in ("ffmpeg", "ffprobe")},
        "evaluated_role": "selection",
        "planned_cases": 138,
        "planned_batches": 8,
        "raw_point": read(PREP / "plan.json")["raw_point"],
        "candidate_point": read(PREP / "plan.json")["candidate_point"],
        "scope": "Fixed single-visible-face cue on planned native support. Sealed file-map transfer, no fit, no full production tracking/fallback, lip/audio inference or customer probability.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "planned_cases": 138, "new_neural_calls": 0}), flush=True)


def verify():
    lock = read(TRIAL / "freeze.json")
    require(lock == read(PUBLIC) == identity(), "Unchanged published native-method freeze required")
    return lock


def selected(batch):
    require(type(batch) is int and 0 <= batch < 8, "One fixed batch required")
    plan = read(PREP / "plan.json")
    groups = plan["batch_group_indices"][batch]
    cases = [c for c in plan["cases"] if c["group_index"] in groups]
    require(
        len(cases) == len(groups) * 6 and all(c["source_role"] == "selection" for c in cases),
        "Complete selection batch required",
    )
    return cases


def ready(batch):
    groups = {c["group_index"] for c in selected(batch)}
    for i in groups:
        folder = SOURCE / f"group-{i:03d}-command-review-v96"
        if not all(
            p.is_file()
            for p in (
                SOURCE / f"group-{i:03d}-complete.json",
                folder / "result.json",
                folder / "profile.json",
            )
        ):
            return False
        if (
            read(folder / "result.json").get("status") != "complete"
            or read(folder / "profile.json").get("status") != "complete"
        ):
            return False
    return True


def validate_case(planned, case):
    clock = planned["expected_clock"]
    table = case["frame_table"]
    n = clock["frame_count"]
    require(
        case["condition"] == planned["condition"]
        and case["label"] == planned["publisher_manipulation_label"],
        "Condition/provenance differs",
    )
    require(
        Path(case["file"]).name == case["file"] and case["file"] == planned["condition"] + ".mp4",
        "Unsafe or unexpected media name",
    )
    require(
        not table["truncated"]
        and not table["interpolated"]
        and table["rotation"] == 0
        and (table["width"], table["height"]) == (clock["width"], clock["height"])
        and (table["coded_width"], table["coded_height"]) == (clock["width"], clock["height"])
        and len(table["pts"]) == n
        and len(case["expected_rgb_sha256"]) == n
        and len(table["pict_types"]) == n,
        "Complete unrotated native geometry required",
    )
    require(
        Fraction(table["time_base"]) > 0
        and all(
            p * Fraction(table["time_base"]) == e * Fraction(clock["time_base"])
            for p, e in zip(table["pts"], clock["pts"], strict=True)
        )
        and Fraction(case["encoded_duration_seconds"]) == Fraction(clock["duration_seconds"]),
        "Observed constructed clock differs from planned source clock",
    )
    require(case["construction"]["frames"] == n, "Constructed extent differs")
    changed = case["actual_changed_frames"]
    require(
        all(type(i) is int and 0 <= i < n for i in changed) and changed == sorted(set(changed)),
        "Invalid changed-frame support",
    )
    if planned["publisher_manipulation_label"]:
        require(changed == planned["intended_changed_ai_frames"], "Swap changed-source support differs")
    if planned["condition"] == "original":
        require(not changed, "Original condition changed")
    if planned["known_insert_frames"] is not None:
        a, b = planned["known_insert_frames"]
        require(
            case["construction"]["edit_frames"] == [a, b] and all(a <= i < b for i in changed),
            "Edit boundary differs",
        )
    require(
        all(0 <= i < n for i in planned["model_frame_indices"]), "Scheduled frame outside observed extent"
    )


def load_group(index):
    source = read(SOURCE / "plan.json")
    group = source["groups"][index]
    receipt_path = SOURCE / f"group-{index:03d}-complete.json"
    receipt = read(receipt_path)
    build_path = SOURCE / f"group-{index:03d}-profile/profile.json"
    review_dir = SOURCE / f"group-{index:03d}-command-review-v96"
    build, review, profile = (
        read(build_path),
        read(review_dir / "result.json"),
        read(review_dir / "profile.json"),
    )
    require(
        receipt["index"] == index
        and receipt["group"] == group["group"]
        and receipt["role"] == group["role"] == "selection",
        "Wrong source group/role",
    )
    require(receipt["freeze_sha256"] == sha(SOURCE / "freeze.json"), "Constructor freeze differs")
    require(
        build["child_exit_code"] == 0 and build["guard"] is None and build["sample_errors"] == 0,
        "Unqualified constructor",
    )
    require(profile["status"] == "complete" and profile["sample_errors"] == 0, "Unqualified command review")
    require(
        review["status"] == "complete"
        and review["cases"] == 6
        and review["private_group_receipt_sha256"] == sha(receipt_path)
        and review["constructor_profile_sha256"] == sha(build_path)
        and review["constructor_freeze_sha256"] == sha(SOURCE / "freeze.json")
        and review["command_journal_sha256"] == sha(review_dir / "command-frames.jsonl")
        and review["reviewer_sha256"] == sha(ROOT / "backend/scripts/review_video_native_inserts_v96.py")
        and review["sample_helper_sha256"] == sha(ROOT / "backend/scripts/audio_tree_sampler_v64.py"),
        "Exact complete command-review binding required",
    )
    media = Path(source["media_root"]) / f"group-{index:03d}"
    require(media.resolve().is_relative_to(ARCHIVE.resolve()), "Media must stay in the approved archive")
    return (
        group,
        receipt,
        media,
        {
            "constructor_receipt_sha256": sha(receipt_path),
            "command_review_sha256": sha(review_dir / "result.json"),
            "command_profile_sha256": sha(review_dir / "profile.json"),
        },
    )


def bind(batch, output):
    import cv2

    from app.detectors.video.decode import decode_frames, probe_frames

    require(ready(batch), "Complete source batches required before a binding attempt")
    rows = []
    for planned in selected(batch):
        group, receipt, media, commitments = load_group(planned["group_index"])
        require(
            group["group"] == planned["source_group"] and receipt["role"] == planned["source_role"],
            "Prepared source identity differs",
        )
        matches = [c for c in receipt["cases"] if c["condition"] == planned["condition"]]
        require(len(matches) == 1, "One exact constructed condition required")
        case = matches[0]
        validate_case(planned, case)
        path = media / case["file"]
        require(
            path.stat().st_size == case["bytes"] and sha(path) == case["sha256"],
            "Constructed media bytes changed",
        )
        table = probe_frames(path, timeout_s=120)
        require(table.to_json() == case["frame_table"], "Fresh native frame table differs")
        argv = [
            shutil.which("ffprobe"),
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-show_entries",
            "stream=codec_type,time_base,duration_ts",
            "-of",
            "json",
            "file:" + str(path.resolve()),
        ]
        clock = subprocess.run(
            argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=True
        )
        streams = json.loads(clock.stdout)["streams"]
        require(
            not clock.stderr.strip() and len(streams) == 1 and streams[0]["codec_type"] == "video",
            "One silent video stream required",
        )
        require(
            int(streams[0]["duration_ts"]) * Fraction(streams[0]["time_base"])
            == Fraction(case["encoded_duration_seconds"]),
            "Fresh display duration differs",
        )
        seen = []
        for frame in decode_frames(path, table, planned["model_frame_indices"], timeout_s=120):
            require(
                fingerprint(frame.rgb) == case["expected_rgb_sha256"][frame.index],
                "Scheduled native model-input RGB changed",
            )
            seen.append(frame.index)
        require(seen == planned["model_frame_indices"], "Complete ordered scheduled frame set required")
        a, b = case["construction"]["edit_frames"]
        expected_exhibits = {"before": a - 1, "during": (a + b - 1) // 2, "after": b}
        require(
            set(case["representatives"]) == set(expected_exhibits), "Three construction exhibits required"
        )
        for name, exhibit in case["representatives"].items():
            i = exhibit["frame_index"]
            require(
                i == expected_exhibits[name] and Path(exhibit["file"]).name == exhibit["file"],
                "Wrong or unsafe exhibit",
            )
            image = media / exhibit["file"]
            require(sha(image) == exhibit["sha256"], "Construction exhibit bytes changed")
            pixels = cv2.imread(str(image))
            require(
                pixels is not None and pixels.shape == (table.height, table.width, 3),
                "Construction exhibit geometry differs",
            )
            require(
                fingerprint(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)) == case["expected_rgb_sha256"][i]
                and exhibit["pts"] * Fraction(exhibit["time_base"]) == table.pts[i] * table.tb,
                "Exhibit pixels/timestamp differ",
            )
        require(sha(path) == case["sha256"], "Media changed during binding")
        rows.append(
            {
                "planned": planned,
                "path": str(path),
                "input_sha256": case["sha256"],
                "bytes": case["bytes"],
                "table": table.to_json(),
                "expected_rgb_sha256": {str(i): case["expected_rgb_sha256"][i] for i in seen},
                "constructed_representatives": case["representatives"],
                **commitments,
            }
        )
    write(output / "bound-cases.json", rows)
    result = {
        "schema": "chipmunk.video-appearance-binding/108",
        "status": "complete",
        "batch": batch,
        "cases": len(rows),
        "native_model_frame_inputs_verified": sum(len(r["planned"]["model_frame_indices"]) for r in rows),
        "construction_pngs_verified": 3 * len(rows),
        "bound_cases_sha256": sha(output / "bound-cases.json"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "new_neural_calls": 0,
        "scope": "Actual bytes/clock/duration and planned RGB/exhibit binding. Existing codec library; no neural or detector accuracy result.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def fixed_response(unit, values, boxes):
    import numpy as np

    from app.detectors.video.face_appearance import distance
    from app.detectors.video.face_appearance import unit as normalize
    from scripts.measure_video_specialists_v18 import supported_window

    needed = {i for k in ("sampled_frames", "context_frames") for i in unit["location"][k]}
    if not needed <= set(values) or not supported_window(unit, boxes):
        return None
    vectors = {i: normalize(values[i]) for i in needed}
    pooled = [
        np.mean([np.mean([vectors[i + k] for k in (-1, 0, 1)], axis=0) for i in unit[key]], axis=0)
        for key in ("before_centers", "middle_centers", "after_centers")
    ]
    left, middle, right = pooled
    raw = min(distance(middle, left), distance(middle, right)) - 0.35 * distance(left, right)
    require(math.isfinite(raw) and -4 <= raw <= 4, "Finite appearance response required")
    return float(raw)


def models():
    import cv2

    from app.detectors.image.faces import load_yunet

    cv2.setNumThreads(1)
    return load_yunet(ROOT / "models/weights/face_detection_yunet_2023mar.onnx"), cv2.FaceRecognizerSF.create(
        str(ROOT / "models/weights/face_recognition_sface_2021dec.onnx"),
        "",
        cv2.dnn.DNN_BACKEND_OPENCV,
        cv2.dnn.DNN_TARGET_CPU,
    )


def aligned_feature(bgr, landmarks, recognizer):
    import cv2
    import numpy as np

    from app.detectors.video import face_appearance as appearance

    row = appearance.face_row(landmarks)
    matrix = appearance.cpp_alignment(landmarks)
    actual = recognizer.alignCrop(bgr, row)
    separate = cv2.warpAffine(bgr, matrix, (112, 112), flags=cv2.INTER_LINEAR)
    require(np.array_equal(actual, separate), "Native/manual SFace crop differs")
    feature = recognizer.feature(actual).reshape(-1).copy()
    require(feature.dtype == np.float32, "Native float32 feature required")
    appearance.unit(feature)
    return feature, actual


def authored(output):
    import numpy as np

    from app.detectors.image.faces import detect_faces
    from app.detectors.video.face_appearance import TARGET

    yunet, recognizer = models()
    yy, xx = np.mgrid[:112, :112]
    bgr = np.stack([(xx * 3 + yy * 7) % 256, (xx * 11 + yy) % 256, (xx + yy * 5) % 256], axis=-1).astype(
        np.uint8
    )
    landmarks = TARGET * np.float32(0.87) + np.array([4, 7], np.float32)
    first, crop = aligned_feature(bgr, landmarks, recognizer)
    second, repeated = aligned_feature(bgr, landmarks, recognizer)
    require(
        np.array_equal(first, second) and np.array_equal(crop, repeated),
        "Authored native feature repeat differs",
    )
    blank = np.zeros((320, 320, 3), np.uint8)
    a = [f.to_dict() for f in detect_faces(blank, detector=yunet, min_size=48)]
    b = [f.to_dict() for f in detect_faces(blank, detector=yunet, min_size=48)]
    require(a == b, "Authored face repeat differs")
    result = {
        "schema": "chipmunk.video-appearance-authored/108",
        "status": "complete",
        "alignment_matches": True,
        "feature_shape": list(first.shape),
        "feature_dtype": str(first.dtype),
        "feature_repeat_bitwise": True,
        "face_repeat_equal": True,
        "native_input_counts": {"face_detection_inputs": 2, "appearance_inputs": 2},
        "new_neural_calls": 4,
        "scope": "Authored CPU alignment/feature/repeat contract only, not source accuracy or a claim that an authored pattern is a face.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def extract(batch, output):
    import cv2
    import numpy as np

    from app.detectors.image.faces import detect_faces
    from app.detectors.video.decode import FrameTable, decode_frames
    from scripts.video_joint_scalar_v106 import predict, raw_appearance

    require(
        read(TRIAL / "authored/output/result.json")["status"] == "complete",
        "Successful authored native contract required",
    )
    qualification = read(TRIAL / "authored/profile.json")
    require(
        qualification["child_exit_code"] == 0
        and qualification["guard"] is None
        and qualification["sample_errors"] == 0,
        "Successful authored resource profile required",
    )
    binding = TRIAL / f"bind-{batch:02d}/output"
    receipt = read(binding / "result.json")
    require(
        receipt["status"] == "complete"
        and receipt["batch"] == batch
        and receipt["bound_cases_sha256"] == sha(binding / "bound-cases.json"),
        "Complete exact source binding required",
    )
    bp = read(binding.parent / "profile.json")
    require(
        bp["child_exit_code"] == 0 and bp["guard"] is None and bp["sample_errors"] == 0,
        "Successful binding resource profile required",
    )
    bound = read(binding / "bound-cases.json")
    require([r["planned"] for r in bound] == selected(batch), "Exact complete prepared batch required")
    candidate = read(FITTED)["models"]["appearance_only"]
    heads = read(PANEL / "video-face-specialists-v20-all-artifact.json")["specialists"]
    yunet, recognizer = models()
    counts = dict.fromkeys(("face_detection_inputs", "appearance_inputs", "appearance_repeat_inputs"), 0)
    with (output / "responses.jsonl").open("x") as responses:
        for row in bound:
            planned, path = row["planned"], Path(row["path"])
            require(
                path.stat().st_size == row["bytes"] and sha(path) == row["input_sha256"],
                "Bound input bytes changed",
            )
            table = FrameTable.from_json(row["table"])
            vectors, boxes, records, seen = {}, {}, [], []
            repeated = False
            for frame in decode_frames(path, table, planned["model_frame_indices"], timeout_s=120):
                require(
                    fingerprint(frame.rgb) == row["expected_rgb_sha256"][str(frame.index)],
                    "Actual model RGB differs from bound input",
                )
                seen.append(frame.index)
                bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                faces = detect_faces(bgr, detector=yunet, min_size=48)
                counts["face_detection_inputs"] += 1
                proof = {
                    "index": frame.index,
                    "pts": frame.pts,
                    "time_base": table.time_base,
                    "rgb_sha256": fingerprint(frame.rgb),
                    "faces": len(faces),
                    "status": "unassessed",
                }
                records.append(proof)
                if len(faces) != 1:
                    proof["reason"] = "Single-visible-face requirement not met; no originality verdict."
                    continue
                face = faces[0]
                # Alignment degeneracy is unsupported; native/model contract errors fail the stage.
                from app.detectors.video.face_appearance import cpp_alignment

                try:
                    cpp_alignment(face.landmarks)
                except ValueError:
                    proof["reason"] = "Degenerate alignment; no originality verdict."
                    continue
                vector, crop = aligned_feature(bgr, face.landmarks, recognizer)
                counts["appearance_inputs"] += 1
                if not repeated:
                    other = recognizer.feature(crop).reshape(-1).copy()
                    counts["appearance_repeat_inputs"] += 1
                    require(np.array_equal(vector, other), "Actual-input native repeat differs")
                    repeated = True
                vectors[frame.index], boxes[frame.index] = vector, face.box()
                proof.update(
                    status="ok",
                    bbox=list(face.box()),
                    landmarks=face.landmarks,
                    aligned_crop_sha256=fingerprint(crop),
                    feature_sha256=fingerprint(vector),
                )
            require(
                seen == planned["model_frame_indices"] and sha(path) == row["input_sha256"],
                "Complete unchanged native input required",
            )
            units = []
            for unit in planned["units"]:
                raw = fixed_response(unit, vectors, boxes)
                units.append(
                    {
                        **unit,
                        "specialist": "face-appearance",
                        "context": "single-face",
                        "status": "ok" if raw is not None else "not_applicable",
                        "raw": raw,
                        "reason": None
                        if raw is not None
                        else "Planned face/alignment/geometric context unavailable; no originality verdict.",
                    }
                )
            file = {"units": units}
            p, old = predict(file, heads, candidate), raw_appearance(file)
            result = {
                "group_index": planned["group_index"],
                "condition": planned["condition"],
                "source_group": planned["source_group"],
                "source_role": "selection",
                "publisher_manipulation_label": planned["publisher_manipulation_label"],
                "planned_scope_label": planned["planned_scope_label"],
                "known_insert_frames": planned["known_insert_frames"],
                "known_insert_seconds": planned["known_insert_seconds"],
                "intended_changed_ai_frames": planned["intended_changed_ai_frames"],
                "input_sha256": row["input_sha256"],
                "units": units,
                "frame_inputs": records,
                "raw_file_flag": None if old is None else bool(old),
                "candidate_file_reference_probability": p,
                "candidate_file_flag": None if p is None else p > read(PREP / "plan.json")["candidate_point"],
                "scope": "Identical fixed source support for raw cue and sealed file calibration. Local RGB/source evidence is not AI-pixel/identity truth; file reference probability is not customer confidence.",
            }
            responses.write(json.dumps(result, allow_nan=False) + "\n")
            responses.flush()
    result = {
        "schema": "chipmunk.video-appearance-extraction/108",
        "status": "complete",
        "batch": batch,
        "cases": len(bound),
        "responses_sha256": sha(output / "responses.jsonl"),
        "binding_receipt_sha256": sha(binding / "result.json"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "native_input_counts": counts,
        "new_neural_calls": sum(counts.values()),
        "defaults_changed": False,
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def monitor(output):
    import psutil

    from scripts.audio_tree_sampler_v64 import sample_tree

    code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True); sys.stdin.readline(); p.terminate(); p.wait()"
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        descendant_pid = int(child.stdout.readline().strip())
        root = psutil.Process(child.pid)
        descendants = root.children(recursive=True)
        require(descendant_pid in {p.pid for p in descendants}, "Owned live descendant not enumerated")
        rss, events, terminal = sample_tree(root, child)
        require(
            not terminal and rss > 0 and not events and psutil.Process(descendant_pid).memory_info().rss > 0,
            "Complete live descendant sample unavailable",
        )
        child.communicate("stop\n", timeout=10)
        require(child.returncode == 0, "Owned sampling child failed")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    result = {
        "schema": "chipmunk.video-appearance-monitor/108",
        "status": "complete",
        "owned_child_exit_code": child.returncode,
        "owned_descendant_sampled": True,
        "combined_rss_bytes": rss,
        "sample_helper_sha256": sha(ROOT / "backend/scripts/audio_tree_sampler_v64.py"),
        "new_neural_calls": 0,
        "scope": "Owned standard-library child/grandchild process-access preflight; not model execution.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "monitor", "bind", "authored", "extract"))
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("batch", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "freeze":
        require(args.output is None and args.batch is None, "Freeze has no stage arguments")
        freeze()
    else:
        verify()
        require(
            args.output is not None and args.output.resolve().is_relative_to(TRIAL.resolve()),
            "Private guarded stage output required",
        )
        args.output.mkdir(exist_ok=False)
        if args.command in {"bind", "extract"}:
            (bind if args.command == "bind" else extract)(args.batch, args.output)
        else:
            require(args.batch is None, "No source batch for qualification")
            {"monitor": monitor, "authored": authored}[args.command](args.output)
