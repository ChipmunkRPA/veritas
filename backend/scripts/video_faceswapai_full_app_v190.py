"""Measure the normal app on a frozen matched panel with a one-face adapter."""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from scripts import video_frame_full_app_v168 as original
from scripts import video_frame_full_app_v171 as runtime
from scripts.video_appearance_corrected_v112 import scoped

TRIAL = original.BASE / "video-faceswapai-full-app-v190"
WORK = original.BASE / "video-faceswapai-full-app-work-v190"
MANIFEST = original.BASE / "video-faceswapai-matched-v186/bind-00/output/private-manifest.json"
ARCHIVE = original.baseline.ARCHIVE / "video-faceswapai-matched-v186"
PREPARED = original.PANEL / "video-faceswapai-matched-complete-v186.json"
PUBLIC = original.PANEL / "video-faceswapai-full-app-freeze-v190.json"
HISTORY = original.BASE / "video-faceswapai-full-app-v187/unchanged/batch-one/authored"
MODULE = "scripts.video_faceswapai_full_app_v190"
ARM = "batch-one"
CONDITIONS = (
    "unchanged",
    "recompression_only",
    "swap_tv_bt601",
    "swap_pc_bt601",
    "swap_tv_bt709",
    "swap_pc_bt709",
)
EDIT_FRAMES = range(72, 144)
PICTURE_PTS = list(range(0, 122880, 512))
BASE = Fraction(1, 12288)
SCOPE = "Six matched renderings of one silent synthetic source. No independent-camera or lip-sync accuracy estimate."
read, write, sha, require = runtime.read, runtime.write, runtime.sha, runtime.require


def previous_failure():
    profile, log = HISTORY / "profile.json", HISTORY / "worker.log"
    recorded = read(profile)
    prior_freeze = original.PANEL / "video-faceswapai-full-app-freeze-v187.json"
    require(
        recorded["child_exit_code"] == 1
        and recorded["guard"] is None
        and recorded["sample_errors"] == 0
        and recorded["worker_receipt_sha256"] is None
        and recorded["resource_journal_sha256"] == sha(HISTORY / "resources.jsonl")
        and recorded["runtime_sha256"]
        == read(prior_freeze)["code_sha256"]["video_faceswapai_full_app_v187.py"]
        and "Fresh settings process required" in log.read_text()
        and not (HISTORY / "output/result.json").exists(),
        "The preserved pre-inference v187 stop is required",
    )
    return {
        "previous_freeze_sha256": sha(prior_freeze),
        "profile_sha256": sha(profile),
        "log_sha256": sha(log),
    }


def identity():
    prior, _, _ = runtime.verify()
    code = (
        Path(__file__),
        Path(__file__).with_name("video_faceswapai_full_app_v187.py"),
        original.ROOT / "backend/tests/test_video_faceswapai_full_app_v190.py",
        original.ROOT / "docs/video-faceswapai-full-app-v190-method.md",
    )
    return {
        "schema": "chipmunk.video-faceswapai-full-app-freeze/190",
        "previous_startup_stop": previous_failure(),
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
    from scripts import video_faceswapai_full_app_v187 as prior

    return scoped(prior.freeze, {"TRIAL": TRIAL, "WORK": WORK, "identity": identity})()


def verify(condition):
    frozen = identity()
    stage = TRIAL / condition
    row = read(stage / "inputs.json")
    manifest = read(MANIFEST)
    require(
        read(TRIAL / "freeze.json") == read(stage / "freeze.json") == read(PUBLIC) == frozen
        and row == next(item for item in manifest["cases"] if item["condition"] == condition)
        and sha(stage / f"{ARM}.yaml") == frozen["arm_manifest_sha256"],
        "Frozen panel, condition and exact serving manifest required",
    )
    file = original.baseline.clips.owned_file(ARCHIVE / f"{condition}.mp4", ARCHIVE)
    require(file.stat().st_size == row["bytes"] and sha(file) == row["sha256"], "Prepared input changed")
    return frozen, file, {"file_sha256": row["sha256"], "bytes": row["bytes"]}


def verify_input(outputs, expected, digest):
    # Preparation dependencies may enter only after the isolated app has initialized.
    from scripts.video_faceswapai_full_app_v187 import verify_input as check

    return check(outputs, expected, digest)


def preflight():
    for condition in CONDITIONS:
        verify(condition)
    require("app.core.config" not in sys.modules, "Preparation must not import the worker settings")
    return runtime.preflight(ARM)


def worker(condition, output):
    verify(condition)
    require("app.core.config" not in sys.modules, "Fresh isolated settings required before worker entry")
    inherited = SimpleNamespace(
        serving=original.serving,
        worker=scoped(
            original.worker, {"record_case": scoped(original.record_case, {"verify_input": verify_input})}
        ),
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
    from scripts import video_faceswapai_full_app_v187 as prior

    scoped(
        prior.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "qualification": qualification,
            "MODULE": MODULE,
            "__file__": __file__,
        },
    )(condition)


def qualification(condition):
    from scripts import video_faceswapai_full_app_v187 as prior

    return scoped(prior.qualification, {"TRIAL": TRIAL, "WORK": WORK, "__file__": __file__})(condition)


def review_packet(packet, source, expected):
    from app.evidence.face_interval_binding import validate_packet

    validate_packet(packet, source, table_pts=tuple(PICTURE_PTS))
    sequence = packet.get("face_sequence")
    anchors = [item["anchor"] for item in packet["frames"]]
    for row in anchors + (sequence["frame_inputs"] if sequence else []):
        require(
            row["decoded_rgb_sha256"] == expected["rgb_sha256"][row["frame_index"]], "Evidence RGB differs"
        )
    if sequence:
        during = sequence["centers"]["during"]
        full_rgb_bindings = len(sequence["frame_inputs"])
    else:
        during = [sample["frame_index"] for sample in packet["face_interval"]["samples"]["during"]]
        full_rgb_bindings = len(anchors)
    frames = [
        {
            "role": item["interval_role"],
            "native_frame": item["anchor"]["frame_index"],
            "time_seconds": float(item["anchor"]["pts"] * BASE),
        }
        for item in packet["frames"]
    ]
    return {
        "method": packet["semantics"],
        "start_seconds": float(PICTURE_PTS[during[0]] * BASE),
        "last_supported_seconds": float(PICTURE_PTS[during[-1]] * BASE),
        "core_start_frame": during[0],
        "core_last_frame": during[-1],
        "verified_native_rgb_inputs": full_rgb_bindings,
        "representatives": frames,
    }


def audit_case(condition):
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.evidence.service import asset_refs, load_packet_set
    from app.storage.db import Analysis, Artefact, DetectorRun
    from app.storage.objects import ObjectStore
    from scripts.video_faceswapai_full_app_v187 import overlap, score_summary

    _, _, input_receipt = verify(condition)
    profile, work = qualification(condition)
    private = read(work / "case.json")
    expected = read(TRIAL / condition / "inputs.json")
    require(
        private["analysis_status"] == private["pipeline_status"] == "done",
        "Serving analysis did not complete",
    )
    data = work / "data"
    engine = create_engine(f"sqlite:///{(data / 'veritas.db').as_uri()}?mode=ro&uri=true")
    try:
        with Session(engine) as db:
            analysis = db.scalars(select(Analysis)).one()
            store = ObjectStore(data / "objects")
            require(
                analysis.result_hash
                and analysis.result_hash == private["result_hash"]
                and analysis.sha256 == input_receipt["file_sha256"]
                and analysis.profile == "full",
                "Persisted full result identity differs",
            )
            store.verify(analysis.result_hash)
            evidence = load_packet_set(db, analysis, store)
            require(evidence and evidence["status"] == "ready", "Persisted evidence unavailable")
            assets = asset_refs(evidence)
            for digest in assets:
                store.verify(digest)
            packets = [
                review_packet(packet, evidence["source"], expected)
                for packet in evidence["packets"]
                if packet.get("face_sequence") or packet.get("face_interval")
            ]
            appearance = db.scalars(
                select(DetectorRun).where(DetectorRun.detector_id == "vid.review.face_appearance")
            ).one()
            appearance_raw = db.get(Artefact, appearance.raw_output_artefact_id)
            store.verify(appearance_raw.sha256)
            details = json.loads(store.read_bytes(appearance_raw.sha256)).get("details") or {}
            review_scope = {
                "route_status": appearance.status,
                "appearance_status": details.get("appearance_check_status", appearance.status),
                "window_method": details.get("face_sequence_binding_version"),
                "measured_windows": len(details.get("face_sequence_windows") or []),
                "selected_window": bool(details.get("face_sequence")),
            }
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
                    and private["target_series_published"]
                    and private["target_signal_published"],
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
        "appearance_review_coverage": review_scope,
        "review_intervals": packets,
        "interval_overlap": {
            name: overlap(frames, expected["actual_changed_frames"]) for name, frames in by_method.items()
        }
        if condition.startswith("swap_")
        else {},
        "verified_evidence_assets": len(assets),
        "worker_seconds": profile["duration_s"],
        "sampled_peak_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
    }, indices


def readout():
    results, positions = zip(*(audit_case(condition) for condition in CONDITIONS), strict=True)
    return {
        "schema": "chipmunk.video-faceswapai-full-app-readout/190",
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "status": "complete",
        "same_legacy_frame_score_positions": all(indices == positions[0] for indices in positions),
        "independent_source_components": 1,
        "cases": list(results),
        "scope": SCOPE,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("freeze", "readout"):
        commands.add_parser(action)
    commands.add_parser("preflight")
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
        result = (
            audit_case(args.condition)[0] if args.command == "case-readout" else globals()[args.command]()
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
