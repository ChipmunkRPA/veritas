"""Measure the normal app on a frozen matched panel with a one-face adapter."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from scripts import video_faceswapai_matched_v186 as preparation
from scripts import video_frame_full_app_v168 as original
from scripts import video_frame_full_app_v171 as runtime
from scripts.video_appearance_corrected_v112 import scoped

TRIAL = original.BASE / "video-faceswapai-full-app-v187"
WORK = original.BASE / "video-faceswapai-full-app-work-v187"
MANIFEST = original.BASE / "video-faceswapai-matched-v186/bind-00/output/private-manifest.json"
ARCHIVE = original.baseline.ARCHIVE / "video-faceswapai-matched-v186"
PREPARED = original.PANEL / "video-faceswapai-matched-complete-v186.json"
MODULE = "scripts.video_faceswapai_full_app_v187"
ARM = "batch-one"
CONDITIONS = (
    "unchanged", "recompression_only", "swap_tv_bt601", "swap_pc_bt601", "swap_tv_bt709", "swap_pc_bt709"
)
EDIT_FRAMES = range(72, 144)
PICTURE_PTS = list(range(0, 122880, 512))
BASE = Fraction(1, 12288)
SCOPE = "Six matched renderings of one silent synthetic source. No independent-camera or lip-sync accuracy estimate."
read, write, sha, require = runtime.read, runtime.write, runtime.sha, runtime.require


def inputs():
    verified = preparation.readout()
    require(read(PREPARED) == verified, "Exact independently verified prepared-panel receipt required")
    manifest = read(MANIFEST)
    require(
        manifest["schema"] == "chipmunk.video-faceswapai-matched-manifest/186"
        and sha(MANIFEST) == verified["private_manifest_sha256"]
        and tuple(row["condition"] for row in manifest["cases"]) == CONDITIONS,
        "Complete ordered matched panel required",
    )
    rows = {row["condition"]: row for row in manifest["cases"]}
    for name, row in rows.items():
        path = ARCHIVE / row["file"]
        table = row["frame_table"]
        changed = row["actual_changed_frames"]
        require(
            row["file"] == f"{name}.mp4"
            and table["pts"] == PICTURE_PTS
            and Fraction(table["time_base"]) == BASE
            and (table["width"], table["height"]) == (1280, 720)
            and not table["interpolated"]
            and not table["truncated"]
            and len(row["rgb_sha256"]) == len(PICTURE_PTS)
            and changed == sorted(set(changed))
            and (not changed if name == "unchanged" else bool(changed) and set(changed).issubset(EDIT_FRAMES))
            and path.stat().st_size == row["bytes"],
            "Expected prepared RGB, zero-origin clock and interval-bound controls",
        )
    return rows


def identity():
    prior, _, _ = runtime.verify()
    code = (
        Path(__file__),
        original.ROOT / "backend/tests/test_video_faceswapai_full_app_v187.py",
        original.ROOT / "docs/video-faceswapai-full-app-v187-method.md",
    )
    return {
        "schema": "chipmunk.video-faceswapai-full-app-freeze/187",
        "conditions": list(CONDITIONS),
        "edit_frames_half_open": [EDIT_FRAMES.start, EDIT_FRAMES.stop],
        "time_base": str(BASE),
        "prepared_receipt_sha256": sha(PREPARED),
        "private_manifest_sha256": sha(MANIFEST),
        "runtime_freeze_sha256": sha(runtime.TRIAL / "freeze.json"),
        "arm_manifest_sha256": sha(runtime.TRIAL / f"{ARM}.yaml"),
        "device": prior["device"],
        "model_manager_budget_bytes": prior["model_manager_budget_bytes"],
        "code_sha256": {path.name: sha(path) for path in code},
        "scope": SCOPE,
    }


def freeze():
    require(not TRIAL.exists() and not WORK.exists(), "Preserve every prior or partial attempt")
    rows = inputs()
    frozen = identity()
    TRIAL.mkdir(mode=0o700)
    WORK.mkdir(mode=0o700)
    for condition, row in rows.items():
        stage = TRIAL / condition
        stage.mkdir(mode=0o700)
        write(stage / "inputs.json", row)
        (stage / f"{ARM}.yaml").write_bytes((runtime.TRIAL / f"{ARM}.yaml").read_bytes())
        write(stage / "freeze.json", frozen)
    write(TRIAL / "freeze.json", frozen)
    return frozen


def verify(condition):
    frozen = identity()
    stage = TRIAL / condition
    row = read(stage / "inputs.json")
    manifest = read(MANIFEST)
    require(
        read(TRIAL / "freeze.json") == read(stage / "freeze.json") == frozen
        and row == next(item for item in manifest["cases"] if item["condition"] == condition)
        and sha(stage / f"{ARM}.yaml") == frozen["arm_manifest_sha256"],
        "Frozen panel, condition and exact serving manifest required",
    )
    file = original.baseline.clips.owned_file(ARCHIVE / f"{condition}.mp4", ARCHIVE)
    require(file.stat().st_size == row["bytes"] and sha(file) == row["sha256"], "Prepared input changed")
    return frozen, file, {"file_sha256": row["sha256"], "bytes": row["bytes"]}


def verify_input(outputs, expected, digest):
    probe, decoded = outputs.get("probe") or {}, outputs.get("decode") or {}
    table, frames = probe.get("frame_table"), decoded.get("frames") or []
    wanted = ((probe.get("plan") or {}).get("legacy") or {}).get("frames") or []
    clock_exact = table == expected["frame_table"]
    rgb_exact = bool(
        clock_exact
        and frames
        and not decoded.get("missing_frames")
        and [frame["frame_index"] for frame in frames] == wanted
        and all(
            frame["index"] == i
            and 0 <= frame["frame_index"] < len(PICTURE_PTS)
            and frame["pts"] == PICTURE_PTS[frame["frame_index"]]
            and frame["sha256"] == expected["rgb_sha256"][frame["frame_index"]]
            for i, frame in enumerate(frames)
        )
    )
    audio = decoded.get("audio") or {}
    faces = (outputs.get("faces") or {}).get("faces") or []
    require(
        clock_exact and rgb_exact and len(faces) == len(frames)
        and (outputs.get("ingest") or {}).get("sha256") == digest
        and (probe.get("info") or {}).get("has_audio") is False
        and audio.get("status") == "absent" and audio.get("samples") == 0 and audio.get("source_mapping") is None,
        "Expected exact sampled RGB and an absent soundtrack",
    )
    return {
        "sampled_frames": len(frames),
        "face_frame_indices": [frame["frame_index"] for frame, boxes in zip(frames, faces, strict=True) if boxes],
        "sampled_indices": wanted,
    }


def worker(condition, output):
    inherited = SimpleNamespace(
        serving=original.serving,
        worker=scoped(original.worker, {"record_case": scoped(original.record_case, {"verify_input": verify_input})}),
    )
    scoped(
        runtime.worker,
        {
            "TRIAL": TRIAL / condition,
            "WORK": WORK / condition,
            "verify": lambda: verify(condition),
            "prior": inherited,
        },
    )(ARM, output)


def run(condition):
    if condition != CONDITIONS[0]:
        previous = CONDITIONS[CONDITIONS.index(condition) - 1]
        qualification(previous)
    parent = TRIAL / condition / ARM
    verify(condition)
    parent.mkdir(exist_ok=True)

    def popen(argv, **kwargs):
        output = parent / "authored/output"
        require(argv[3:] == ["scripts.video_appearance_native_v108", "authored", str(output)], "Exact worker required")
        return subprocess.Popen([*argv[:3], MODULE, "_worker", condition, str(output)], **kwargs)

    scoped(
        original.runtime.run,
        {
            "TRIAL": parent,
            "verify": lambda: verify(condition),
            "prerequisites": lambda *_: True,
            "subprocess": SimpleNamespace(
                Popen=popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
            ),
            "__file__": __file__,
        },
    )("authored")


def qualification(condition):
    stage = TRIAL / condition / ARM / "authored"
    work = WORK / condition / ARM
    receipt = original.baseline.speech.qualification_profile(stage, None, {}, Path(__file__).name)
    require(
        receipt["arm"] == ARM
        and receipt["freeze_sha256"] == sha(TRIAL / "freeze.json")
        and receipt["manifest_sha256"] == sha(TRIAL / condition / f"{ARM}.yaml")
        and receipt["private_case_sha256"] == sha(work / "case.json")
        and receipt["private_runtime_sha256"] == sha(work / "runtime.json"),
        "Exact successful monitored condition required",
    )
    return read(stage / "profile.json"), work


def partition(index):
    if index < EDIT_FRAMES.start:
        return "before"
    if index < EDIT_FRAMES.stop:
        return "inside"
    return "after"


def score_summary(indexed_scores):
    buckets = {name: [] for name in ("before", "inside", "after")}
    for index, score in indexed_scores:
        require(math.isfinite(score) and 0 <= score <= 1, "Expected a finite raw frame score")
        buckets[partition(index)].append(score)
    return {
        name: {"frames": len(values), "mean": round(mean(values), 6) if values else None,
               "maximum": round(max(values), 6) if values else None}
        for name, values in buckets.items()
    }


def overlap(frames, expected):
    expected = set(expected)
    common = len(frames & expected)
    union = len(frames | expected)
    return {"intersection_frames": common, "union_frames": union, "iou": round(common / union, 6) if union else None}


def review_packet(packet, source, expected):
    from app.evidence.face_interval_binding import validate_packet

    validate_packet(packet, source, table_pts=tuple(PICTURE_PTS))
    record = packet["face_sequence"]
    for row in record["frame_inputs"]:
        require(row["decoded_rgb_sha256"] == expected["rgb_sha256"][row["frame_index"]], "Evidence RGB differs")
    during = record["centers"]["during"]
    frames = [
        {"role": item["interval_role"], "native_frame": item["anchor"]["frame_index"],
         "time_seconds": float(item["anchor"]["pts"] * BASE)}
        for item in packet["frames"]
    ]
    return {
        "method": packet["semantics"],
        "start_seconds": float(PICTURE_PTS[during[0]] * BASE),
        "last_supported_seconds": float(PICTURE_PTS[during[-1]] * BASE),
        "core_start_frame": during[0],
        "core_last_frame": during[-1],
        "verified_native_inputs": len(record["frame_inputs"]),
        "representatives": frames,
    }


def audit_case(condition):
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.evidence.service import asset_refs, load_packet_set
    from app.storage.db import Analysis, Artefact, DetectorRun
    from app.storage.objects import ObjectStore

    _, _, input_receipt = verify(condition)
    profile, work = qualification(condition)
    private = read(work / "case.json")
    expected = read(TRIAL / condition / "inputs.json")
    require(private["analysis_status"] == private["pipeline_status"] == "done", "Serving analysis did not complete")
    data = work / "data"
    engine = create_engine(f"sqlite:///{(data / 'veritas.db').as_uri()}?mode=ro&uri=true")
    try:
        with Session(engine) as db:
            analysis = db.scalars(select(Analysis)).one()
            store = ObjectStore(data / "objects")
            require(
                analysis.result_hash and analysis.result_hash == private["result_hash"]
                and analysis.sha256 == input_receipt["file_sha256"] and analysis.profile == "full",
                "Persisted full result identity differs",
            )
            store.verify(analysis.result_hash)
            evidence = load_packet_set(db, analysis, store)
            require(evidence and evidence["status"] == "ready", "Persisted evidence unavailable")
            assets = asset_refs(evidence)
            for digest in assets:
                store.verify(digest)
            packets = [review_packet(packet, evidence["source"], expected) for packet in evidence["packets"]
                       if packet.get("face_sequence")]
            target = db.scalars(select(DetectorRun).where(DetectorRun.detector_id == original.DETECTOR)).one()
            raw = db.get(Artefact, target.raw_output_artefact_id) if target.raw_output_artefact_id else None
            units = []
            if raw is not None:
                store.verify(raw.sha256)
                units = json.loads(store.read_bytes(raw.sha256))["unit_scores"]
            indices = private["input"]["sampled_indices"]
            indexed = [(indices[int(unit["unit_ref"].split(":")[1])], unit["raw"]) for unit in units]
            if target.status == "ok":
                require(
                    [index for index, _ in indexed] == private["input"]["face_frame_indices"]
                    and private["target_series_published"] and private["target_signal_published"],
                    "Stored points must match the published series and every measured face",
                )
            else:
                require(not indexed, "An unavailable route cannot contribute frame scores")
    finally:
        engine.dispose()
    # The report's end is a measured frame timestamp. Count that frame in discrete overlap.
    by_method = {}
    for packet in packets:
        covered = by_method.setdefault(packet["method"], set())
        covered.update(range(packet["core_start_frame"], packet["core_last_frame"] + 1))
    return {
        "condition": condition,
        "role": expected["role"],
        "actual_changed_frames": len(expected["actual_changed_frames"]),
        "sampled_frames": len(indices),
        "face_frames": len(private["input"]["face_frame_indices"]),
        "frame_route_status": target.status,
        "raw_frame_response": score_summary(indexed),
        "customer_direction": private["customer_assessment"]["direction"],
        "review_intervals": packets,
        "interval_overlap": {name: overlap(frames, expected["actual_changed_frames"])
                             for name, frames in by_method.items()} if condition.startswith("swap_") else {},
        "verified_evidence_assets": len(assets),
        "worker_seconds": profile["duration_s"],
        "sampled_peak_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
    }, indices


def readout():
    results, positions = zip(*(audit_case(condition) for condition in CONDITIONS), strict=True)
    return {
        "schema": "chipmunk.video-faceswapai-full-app-readout/187",
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "status": "complete",
        "same_native_sample_positions": all(indices == positions[0] for indices in positions),
        "independent_source_components": 1,
        "cases": list(results),
        "scope": SCOPE,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("freeze", "readout"):
        commands.add_parser(action)
    for action in ("run", "_worker", "case-readout"):
        command = commands.add_parser(action)
        command.add_argument("condition", choices=CONDITIONS)
        if action == "_worker":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.condition)
    elif args.command == "_worker":
        worker(args.condition, args.output)
    else:
        result = audit_case(args.condition)[0] if args.command == "case-readout" else globals()[args.command]()
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
