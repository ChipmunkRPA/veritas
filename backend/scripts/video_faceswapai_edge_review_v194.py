"""Read-only replay of saved native face crops around each v190 review end."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from scripts import video_faceswapai_full_app_v190 as prior

ROOT = prior.original.ROOT
PUBLIC = prior.original.PANEL / "video-faceswapai-edge-review-freeze-v194.json"
PRIOR_RESULT = prior.original.PANEL / "video-faceswapai-full-app-complete-v190.json"
WEIGHT = ROOT / "models/weights/face_recognition_sface_2021dec.onnx"
WEIGHT_SHA = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
SEPARATION = 0.11243900750879454
REPLAY_TOLERANCE = 1e-6
SCHEMA = "chipmunk.video-faceswapai-edge-review/194"
SCOPE = "One consumed synthetic source component. Same-track appearance context; no authenticity or serving claim."
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require


class ReplayUnavailable(ValueError):
    def __init__(self, reason, forwards):
        super().__init__(reason)
        self.forwards = forwards


def triplet(center):
    return tuple(range(center - 1, center + 2))


def positions(centers):
    return [index for center in centers for index in triplet(center)]


def retained(end, reason):
    return {
        "status": reason,
        "proposed_last_supported_frame": end,
        "first_measured_return_frame": None,
        "observations": [],
        "reference_distance": None,
    }


def appearance_distance(a, b):
    # Reuse the exact same normalized cosine implementation as the published lane.
    from app.detectors.video.face_appearance import distance

    return distance(a, b)


def refine(centers, embeddings):
    """Decide solely from measured same-track crops and a held-out end context."""
    before, during, after = (centers[role] for role in ("before", "during", "after"))
    require(len(before) == len(after) == 2 and len(during) >= 3, "Measured flank and core triplets required")
    old_end = during[-1]
    baseline = retained(old_end, "no_stable_return_bracket")
    before_ref, core_ref = positions(before), positions(during[:-1])
    tested = [triplet(center) for center in (old_end, *after)]
    required = [*before_ref, *core_ref, *(index for group in tested for index in group)]
    if required != sorted(set(required)):
        return retained(old_end, "overlapping_reference_or_test_triplets")
    require(set(required) <= embeddings.keys(), "Every committed reference and test crop must be replayed")
    before_mean = np.mean([embeddings[index] for index in before_ref], axis=0)
    core_mean = np.mean([embeddings[index] for index in core_ref], axis=0)
    separation = appearance_distance(before_mean, core_mean)
    baseline["reference_distance"] = round(separation, 6)
    if separation <= SEPARATION:
        return {**baseline, "status": "reference_appearance_not_distinct"}
    labels = []
    for group in tested:
        group_labels = []
        for index in group:
            a = appearance_distance(embeddings[index], before_mean)
            b = appearance_distance(embeddings[index], core_mean)
            label = "ambiguous" if abs(a - b) <= REPLAY_TOLERANCE else "before_like" if a < b else "core_like"
            group_labels.append(label)
            baseline["observations"].append(
                {
                    "frame": index,
                    "label": label,
                    "distance_to_before": round(a, 6),
                    "distance_to_core": round(b, 6),
                }
            )
        labels.append(group_labels[0] if len(set(group_labels)) == 1 else "ambiguous")
    if labels[0] != "core_like" or labels[2] != "before_like" or labels[1] == "ambiguous":
        return baseline
    last, first_return = (
        (tested[1][-1], tested[2][0]) if labels[1] == "core_like" else (tested[0][-1], tested[1][0])
    )
    return {
        **baseline,
        "status": "measured_return_bracket",
        "proposed_last_supported_frame": last,
        "first_measured_return_frame": first_return,
    }


def interval_metrics(start, last, changed):
    interval = set(range(start, last + 1))
    truth = set(changed)
    overlap = interval & truth
    return {
        "covered_native_positions": len(interval),
        "overlap_native_positions": len(overlap),
        "missed_native_positions": len(truth - interval),
        "outside_native_positions": len(interval - truth),
        "iou": round(len(overlap) / len(interval | truth), 6),
    }


def summarize(start, old_end, proposal, changed, is_swap, pts):
    end = proposal["proposed_last_supported_frame"]
    first_return = proposal["first_measured_return_frame"]
    require(
        start <= old_end <= end < len(pts) and (first_return is None or end < first_return < len(pts)),
        "Monotonic source-bound interval and return bracket required",
    )

    def anchor(index):
        return {"native_frame": index, "pts": pts[index], "time_seconds": float(pts[index] * prior.BASE)}

    appended = set(range(old_end + 1, end + 1))
    measured = {row["frame"] for row in proposal["observations"] if row["label"] == "core_like"}
    existing = interval_metrics(start, old_end, changed)
    candidate = interval_metrics(start, end, changed)
    return {
        "status": proposal["status"],
        "existing": {"start": anchor(start), "last_supported": anchor(old_end), **existing},
        "proposed": {"start": anchor(start), "last_supported": anchor(end), **candidate},
        "first_measured_return": anchor(first_return) if first_return is not None else None,
        "unmeasured_positions_in_return_bracket": first_return - end - 1
        if first_return is not None
        else None,
        "appended_directly_measured_core_like": [anchor(index) for index in sorted(appended & measured)],
        "appended_unmeasured_positions": len(appended - measured),
        "comparison_target": "publisher_swap" if is_swap else "ordinary_changes_only",
        "reference_distance": proposal["reference_distance"],
        "observations": [
            {**anchor(row["frame"]), **{k: v for k, v in row.items() if k != "frame"}}
            for row in proposal["observations"]
        ],
    }


def database_path(work):
    path = work / "data/veritas.db"
    wal, journal = Path(f"{path}-wal"), Path(f"{path}-journal")
    require(
        path.is_file()
        and path.resolve() == path
        and not journal.exists()
        and not journal.is_symlink()
        and not wal.is_symlink()
        and (not wal.exists() or wal.stat().st_size == 0),
        "A finalized, checkpointed read-only v190 database required",
    )
    return path


def identity():
    code = (
        Path(__file__),
        ROOT / "backend/tests/test_video_faceswapai_edge_review_v194.py",
        ROOT / "docs/video-faceswapai-edge-review-v194-method.md",
        ROOT / "backend/app/detectors/video/face_appearance.py",
        ROOT / "backend/app/evidence/face_interval_binding.py",
        ROOT / "backend/app/detectors/video/decode.py",
    )
    require(
        WEIGHT.is_file() and sha(WEIGHT) == WEIGHT_SHA,
        "Exact local commercial appearance checkpoint required",
    )
    prior_result = read(PRIOR_RESULT)
    require(
        prior_result["schema"] == "chipmunk.video-faceswapai-full-app-readout/190"
        and prior_result["status"] == "complete"
        and prior_result["freeze_sha256"] == sha(prior.TRIAL / "freeze.json")
        and read(prior.TRIAL / "freeze.json") == read(prior.PUBLIC)
        and tuple(row["condition"] for row in prior_result["cases"]) == prior.CONDITIONS,
        "Complete exact v190 comparison required",
    )
    cases = {}
    for condition in prior.CONDITIONS:
        _, _, receipt = prior.verify(condition)
        _, work = prior.qualification(condition)
        cases[condition] = {
            "input_sha256": receipt["file_sha256"],
            "database_sha256": sha(database_path(work)),
            "case_sha256": sha(work / "case.json"),
        }
    binaries = {}
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        require(path is not None, f"Local {name} required")
        binaries[name] = sha(Path(path).resolve())
    return {
        "schema": "chipmunk.video-faceswapai-edge-review-freeze/194",
        "prior_freeze_sha256": sha(prior.PUBLIC),
        "prior_result_sha256": sha(PRIOR_RESULT),
        "code_sha256": {path.name: sha(path) for path in code},
        "weight_sha256": WEIGHT_SHA,
        "binary_sha256": binaries,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "opencv-python-headless", "sqlalchemy")
        },
        "cases": cases,
        "separation_point": SEPARATION,
        "appearance_replay_tolerance": REPLAY_TOLERANCE,
        "scope": SCOPE,
    }


def enforce_read_only():
    programs = {Path(shutil.which(name)).resolve() for name in ("ffmpeg", "ffprobe")}

    def audit(event, args):
        if event == "open":
            path, _, flags = args
            writes = flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC)
            if writes and not isinstance(path, int) and os.fsdecode(path) != os.devnull:
                raise PermissionError("Edge replay may not write files")
        if event in {
            "os.mkdir",
            "os.remove",
            "os.rmdir",
            "os.rename",
            "os.link",
            "os.symlink",
            "os.chmod",
            "os.truncate",
            "os.utime",
            "os.system",
        } or event.startswith("socket."):
            raise PermissionError("Edge replay may not mutate storage or access networking")
        if event == "subprocess.Popen":
            path = shutil.which(os.fsdecode(args[0]))
            require(path and Path(path).resolve() in programs, "Only frozen native decoder commands allowed")

    sys.addaudithook(audit)


def replay_features(path, record, expected, recognizer):
    import cv2

    from app.detectors.video import face_appearance as appearance
    from app.detectors.video.decode import FrameTable, decode_frames

    table = FrameTable.from_json(expected["frame_table"])
    inputs = {row["frame_index"]: row for row in record["frame_inputs"]}
    embeddings = {}
    for frame in decode_frames(path, table, sorted(inputs), timeout_s=90):
        row = inputs[frame.index]
        require(
            frame.pts == row["pts"]
            and hashlib.sha256(frame.rgb.tobytes()).hexdigest()
            == row["decoded_rgb_sha256"]
            == expected["rgb_sha256"][frame.index],
            "Measured native picture or its clock changed",
        )
        detection = appearance.face_row(row["landmarks"])
        detection[0, :4] = [row["bbox"][key] for key in ("x", "y", "w", "h")]
        crop = recognizer.alignCrop(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), detection)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if hashlib.sha256(crop_rgb.tobytes()).hexdigest() != row["aligned_rgb_sha256"]:
            raise ReplayUnavailable("aligned_crop_did_not_reproduce", len(embeddings))
        embeddings[frame.index] = appearance.unit(recognizer.feature(crop).reshape(-1))
    require(embeddings.keys() == inputs.keys(), "All source-bound crop measurements required")
    before, during, after = [
        np.mean(
            [
                np.mean([embeddings[index] for index in triplet(center)], axis=0)
                for center in record["centers"][role]
            ],
            axis=0,
        )
        for role in ("before", "during", "after")
    ]
    left, right, flank = (
        appearance.distance(during, before),
        appearance.distance(during, after),
        appearance.distance(before, after),
    )
    for key, value in {
        "middle_to_before": left,
        "middle_to_after": right,
        "flank_change": flank,
        "response": min(left, right) - appearance.FLANK_PENALTY * flank,
    }.items():
        if not math.isclose(
            value, float.fromhex(record["measurement_hex"][key]), rel_tol=0, abs_tol=REPLAY_TOLERANCE
        ):
            raise ReplayUnavailable("published_appearance_arithmetic_did_not_reproduce", len(embeddings))
    return embeddings


def case_record(condition, recognizer):
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.evidence.face_interval_binding import APPEARANCE_METHOD
    from app.evidence.service import asset_refs, load_packet_set
    from app.storage.db import Analysis
    from app.storage.objects import ObjectStore

    _, file, receipt = prior.verify(condition)
    _, work = prior.qualification(condition)
    case = read(work / "case.json")
    expected = read(prior.TRIAL / condition / "inputs.json")
    db_file = database_path(work)
    engine = create_engine(f"sqlite:///{db_file.as_uri()}?mode=ro&immutable=1&uri=true")
    try:
        with Session(engine) as db:
            analysis = db.scalars(select(Analysis)).one()
            require(
                analysis.status == "done"
                and analysis.sha256 == receipt["file_sha256"]
                and analysis.result_hash == case["result_hash"],
                "Exact completed v190 case required",
            )
            store = ObjectStore(work / "data/objects")
            store.verify(analysis.result_hash)
            evidence = load_packet_set(db, analysis, store)
            require(evidence and evidence["status"] == "ready", "Committed source evidence required")
            for digest in asset_refs(evidence):
                store.verify(digest)
            packets = [packet for packet in evidence["packets"] if packet["semantics"] == APPEARANCE_METHOD]
            require(len(packets) <= 1, "At most one existing appearance interval per case")
            if not packets:
                return {
                    "condition": condition,
                    "status": "no_existing_appearance_interval",
                    "crop_forwards": 0,
                }
            packet = packets[0]
            old = prior.review_packet(packet, evidence["source"], expected)
    finally:
        engine.dispose()
    record = packet["face_sequence"]
    try:
        embeddings = replay_features(file, record, expected, recognizer)
        forwards = len(embeddings)
        proposal = refine(record["centers"], embeddings)
    except ReplayUnavailable as error:
        forwards = error.forwards
        proposal = retained(old["core_last_frame"], str(error))
    summary = summarize(
        old["core_start_frame"],
        old["core_last_frame"],
        proposal,
        expected["actual_changed_frames"],
        condition.startswith("swap_"),
        expected["frame_table"]["pts"],
    )
    return {
        "condition": condition,
        **summary,
        "existing_representatives": old["representatives"],
        "crop_forwards": forwards,
    }


def readout():
    import cv2

    frozen = identity()
    require(read(PUBLIC) == frozen, "Exact v194 freeze required")
    enforce_read_only()
    cv2.setNumThreads(1)
    recognizer = cv2.FaceRecognizerSF.create(
        str(WEIGHT), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
    )
    rows = [case_record(condition, recognizer) for condition in prior.CONDITIONS]
    for condition in prior.CONDITIONS:
        path = database_path(prior.WORK / condition / prior.ARM)
        require(
            sha(path) == frozen["cases"][condition]["database_sha256"], "v190 database changed during replay"
        )
    return {
        "schema": SCHEMA,
        "freeze_sha256": sha(PUBLIC),
        "status": "complete",
        "scope": SCOPE,
        "independent_source_components": 1,
        "crop_forwards": sum(row["crop_forwards"] for row in rows),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "readout"))
    args = parser.parse_args()
    if args.action == "freeze":
        write(PUBLIC, identity())
    else:
        print(json.dumps(readout(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
