"""Audit the frozen full video app on matched face-swap-negative webcam controls."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from scripts import video_faceswapai_edge_review_v194 as edge
from scripts import video_faceswapai_full_app_v190 as prior
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = prior.original.ROOT, prior.original.BASE, prior.original.PANEL
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require
TRIAL = BASE / "video-vcd2-full-app-v202"
WORK = BASE / "video-vcd2-full-app-work-v202"
ARCHIVE = prior.original.baseline.ARCHIVE
PREPARATION = ARCHIVE / "video-vcd2-controls-v200"
SEALED = ARCHIVE / "video-vcd2-full-app-v202"
PUBLIC = PANEL / "video-vcd2-full-app-freeze-v202.json"
SOURCE_FREEZE = PANEL / "video-vcd2-genuine-roster-freeze-v197.json"
MODULE = "scripts.video_vcd2_full_app_v202"
ARM = prior.ARM
SLOTS = range(1, 13)
ROLES = ("unchanged", "chroma_conversion_only", "recompression_only")
CONDITIONS = tuple(f"slot-{slot:02d}-{role}" for slot in SLOTS for role in ROLES)
DIRECTIONS = ("ai", "original", "inconclusive", "unavailable", "record")
REVIEW_METHODS = ("face-track-response/1", "face-sequence-change/1", "face-appearance-change/1")
SHA = re.compile(r"[a-f0-9]{64}")
GIB, MIB = 1024**3, 1024**2
MAX_APP_VIDEO_BYTES = 500 * MIB
CODE = (
    "backend/scripts/video_vcd2_full_app_v202.py",
    "backend/scripts/video_vcd2_case_archive_v202.py",
    "backend/tests/test_video_vcd2_full_app_v202.py",
    "docs/video-vcd2-full-app-v202-method.md",
)


def require_fresh():
    require(
        not {"app.core.config", "app.engine.pipelines", "torch", "cv2"} & sys.modules.keys(),
        "Fresh settings and model-free process required before app worker entry",
    )


def direct_file(path):
    require(path.is_file() and path.resolve() == path, "Expected direct owned case file")
    return path


def prepared_roster():
    document = read(direct_file(PREPARATION / "freeze.json"))
    identity, slots = document["identity"], document["slots"]
    require(
        identity["schema"] == "chipmunk.video-vcd2-controls/200"
        and identity["source_freeze_sha256"] == sha(SOURCE_FREEZE)
        and identity["model_calls"] == 0
        and identity["roles"] == list(ROLES)
        and all(sha(ROOT / name) == value for name, value in identity["code_sha256"].items())
        and [row["slot"] for row in slots] == list(SLOTS),
        "Exact v200 construction freeze and v197 roster required",
    )
    return slots


def qualify_slot(entry, receipt):
    count = receipt["frame_count"]
    pts = receipt["relative_pts"]
    tb = Fraction(receipt["time_base"])
    start, end = receipt["frame_interval"]
    require(
        receipt["slot"] == entry["slot"]
        and receipt["source_sha256"] == entry["selected"]["source_sha256"]
        and receipt["model_calls"] == 0
        and type(count) is int
        and count == len(pts)
        and 3 <= count <= 1200
        and pts[0] == 0
        and all(type(value) is int for value in pts)
        and all(a < b for a, b in zip(pts, pts[1:]))
        and tb > 0
        and 0 < start < end < count
        and [i for i, value in enumerate(pts) if 3 <= value * tb < 6] == list(range(start, end))
        and min(receipt["width"], receipt["height"]) > 0
        and set(receipt["cases"]) == set(ROLES),
        "Exact source-relative v200 picture clock and ordinary-edit interval required",
    )
    baseline = receipt["cases"]["unchanged"]["rgb_sha256"]
    for role, case in receipt["cases"].items():
        hashes = case["rgb_sha256"]
        actual = case["changed_frames"]
        require(
            case["file"] == f"{role}.mp4"
            and SHA.fullmatch(case["sha256"])
            and case["verified_frames"] == len(hashes) == count
            and all(SHA.fullmatch(digest) for digest in hashes)
            and [row["index"] for row in actual]
            == [i for i, (left, right) in enumerate(zip(baseline, hashes, strict=True)) if left != right]
            and all(
                start <= row["index"] < end and row["pts"] == pts[row["index"]] and 0 < row["rgb_mae"] <= 255
                for row in actual
            ),
            "All exact v200 picture hashes and ordinary changes required",
        )


def case_name(slot, role):
    return f"slot-{slot:02d}-{role}"


def case_inventory():
    rows = {}
    for entry in prepared_roster():
        slot = entry["slot"]
        folder = PREPARATION / f"slot-{slot:02d}-attempt-01"
        qualification = folder / "qualification.json"
        if entry["status"] == "unavailable":
            require(not qualification.exists() and entry["selected"] is None, "Unexpected available control")
            status = "source_unavailable"
        elif entry["status"] == "qualified":
            status = "prepared" if qualification.is_file() else "preparation_pending"
        else:
            raise ValueError("Unknown frozen source status")
        if status != "prepared":
            for role in ROLES:
                rows[case_name(slot, role)] = {"slot": slot, "role": role, "status": status}
            continue
        receipt = read(direct_file(qualification))
        qualify_slot(entry, receipt)
        for role in ROLES:
            case = receipt["cases"][role]
            size = direct_file(folder / case["file"]).stat().st_size
            require(size > 0, "Empty prepared control")
            rows[case_name(slot, role)] = {
                "slot": slot,
                "role": role,
                "status": "prepared" if size <= MAX_APP_VIDEO_BYTES else "unsupported_upload_size",
                "qualification_sha256": sha(qualification),
                "sha256": case["sha256"],
                "bytes": size,
            }
    require(tuple(rows) == CONDITIONS, "Exact fixed slot and role roster required")
    return rows


def identity(plan):
    require_fresh()
    frozen = prior.identity()
    edge_lock = read(edge.PUBLIC)
    require(
        frozen == read(prior.PUBLIC)
        and sha(edge.WEIGHT) == edge.WEIGHT_SHA
        and edge_lock["weight_sha256"] == edge.WEIGHT_SHA
        and edge_lock["separation_point"] == edge.SEPARATION
        and edge_lock["appearance_replay_tolerance"] == edge.REPLAY_TOLERANCE
        and edge_lock["code_sha256"][Path(edge.__file__).name] == sha(Path(edge.__file__)),
        "Exact v190 app and v194 refinement required",
    )
    tool_hashes = read(direct_file(PREPARATION / "freeze.json"))["identity"]["tool_sha256"]
    require(
        all(
            shutil.which(name) and sha(Path(shutil.which(name)).resolve()) == digest
            for name, digest in tool_hashes.items()
        ),
        "The exact v200 decoder binaries must remain available",
    )
    require_fresh()
    return {
        "schema": "chipmunk.video-vcd2-full-app-freeze/202",
        "planned_cases": len(CONDITIONS),
        "planned_slots": len(SLOTS),
        "roles": list(ROLES),
        "app_freeze_sha256": sha(prior.PUBLIC),
        "edge_freeze_sha256": sha(edge.PUBLIC),
        "preparation_freeze_sha256": sha(direct_file(PREPARATION / "freeze.json")),
        "roster_freeze_sha256": sha(SOURCE_FREEZE),
        "app_manifest_sha256": frozen["arm_manifest_sha256"],
        "model_manager_budget_bytes": frozen["model_manager_budget_bytes"],
        "device": frozen["device"],
        "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "max_app_video_bytes": MAX_APP_VIDEO_BYTES,
        "all_face_swap_negatives": True,
        "serving_change_permitted": False,
    }


def disk_required(size):
    # Upload and object-store original account for two complete local copies.
    return max(3 * GIB, 1536 * MIB + 2 * size + 256 * MIB)


def preflight():
    plan = case_inventory()
    identity(plan)
    free = shutil.disk_usage(BASE).free
    counts = Counter(row["status"] for row in plan.values())
    return {
        "status": "waiting_for_preparation" if counts["preparation_pending"] else "ready_for_freeze",
        "planned_cases": len(plan),
        "input_outcomes": dict(counts),
        "available_internal_bytes": free,
        "prepared_fitting_current_disk": sum(
            row["status"] == "prepared" and free >= disk_required(row["bytes"]) for row in plan.values()
        ),
        "model_calls": 0,
    }


def freeze():
    require(not any(path.exists() for path in (TRIAL, WORK, PUBLIC)), "Preserve every earlier study attempt")
    plan = case_inventory()
    require(
        all(row["status"] != "preparation_pending" for row in plan.values()),
        "Wait for all v200 control receipts",
    )
    frozen = identity(plan)
    TRIAL.mkdir(mode=0o700)
    WORK.mkdir(mode=0o700)
    write(TRIAL / "private-plan.json", plan)
    for condition, row in plan.items():
        if row["status"] != "prepared":
            continue
        folder = TRIAL / condition
        folder.mkdir(mode=0o700)
        write(folder / "inputs.json", row)
        write(folder / "freeze.json", frozen)
        with (folder / f"{ARM}.yaml").open("xb") as stream:
            stream.write((prior.runtime.TRIAL / f"{ARM}.yaml").read_bytes())
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    return {"status": "frozen", "planned_cases": len(plan)}


def verify_plan():
    plan = read(TRIAL / "private-plan.json")
    frozen = identity(plan)
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == frozen and tuple(plan) == CONDITIONS,
        "Exact prospective app and camera-control freeze required",
    )
    return frozen, plan


def verify(condition):
    frozen, plan = verify_plan()
    row = plan[condition]
    folder = TRIAL / condition
    require(
        row["status"] == "prepared"
        and read(folder / "freeze.json") == frozen
        and read(folder / "inputs.json") == row
        and sha(folder / f"{ARM}.yaml") == frozen["app_manifest_sha256"],
        "Exact frozen case inputs and app manifest required",
    )
    source = PREPARATION / f"slot-{row['slot']:02d}-attempt-01"
    qualification = direct_file(source / "qualification.json")
    file = direct_file(source / f"{row['role']}.mp4")
    require(
        sha(qualification) == row["qualification_sha256"]
        and file.stat().st_size == row["bytes"]
        and sha(file) == row["sha256"],
        "Prepared case pixels or metadata changed",
    )
    return frozen, file, {"file_sha256": row["sha256"], "bytes": row["bytes"]}


def expected_case(row):
    source = PREPARATION / f"slot-{row['slot']:02d}-attempt-01"
    qualification = direct_file(source / "qualification.json")
    require(sha(qualification) == row["qualification_sha256"], "Prepared qualification changed")
    doc = read(qualification)
    return {
        "time_base": doc["time_base"],
        "pts": doc["relative_pts"],
        "size": (doc["width"], doc["height"]),
        "rgb": doc["cases"][row["role"]]["rgb_sha256"],
        "ordinary": doc["frame_interval"],
        "changed": [item["index"] for item in doc["cases"][row["role"]]["changed_frames"]],
    }


def verify_input(outputs, row, digest):
    expected = expected_case(row)
    probe, decoded = outputs.get("probe") or {}, outputs.get("decode") or {}
    table = probe.get("frame_table")
    frames = decoded.get("frames") or []
    wanted = ((probe.get("plan") or {}).get("legacy") or {}).get("frames") or []
    pts, rgb = expected["pts"], expected["rgb"]
    tb = Fraction(expected["time_base"])
    require((outputs.get("ingest") or {}).get("sha256") == digest, "App ingest differs from the v200 source")
    if table is not None:
        require(
            table["pts"] == pts
            and Fraction(table["time_base"]) == tb
            and (table["width"], table["height"]) == expected["size"]
            and not table["truncated"]
            and not table["interpolated"]
            and table["rotation"] == 0
            and (probe.get("info") or {}).get("has_audio") is False,
            "Production app used a different webcam picture clock",
        )
    if frames:
        audio = decoded["audio"]
        require(
            table is not None
            and not decoded.get("missing_frames")
            and [frame["frame_index"] for frame in frames] == wanted == sorted(set(wanted))
            and all(
                frame["index"] == i
                and 0 <= frame["frame_index"] < len(pts)
                and frame["pts"] == pts[frame["frame_index"]]
                and frame["sha256"] == rgb[frame["frame_index"]]
                for i, frame in enumerate(frames)
            )
            and audio["status"] == "absent"
            and audio["samples"] == 0
            and audio["source_mapping"] is None,
            "Every sampled native RGB frame, timestamp and absent soundtrack must reproduce",
        )
    faces = (outputs.get("faces") or {}).get("faces")
    if faces is not None:
        require(len(faces) == len(frames), "Face observations differ from decoded sample count")
    anchors = (outputs.get("render") or {}).get("frame_anchors") or []
    native = validate_anchors(anchors, expected)
    require(not anchors or set(wanted) <= native, "Legacy samples missing from the full native anchor set")
    return {
        "sampled_frames": len(frames),
        "face_frame_indices": [frame["frame_index"] for frame, boxes in zip(frames, faces or []) if boxes],
        "sampled_indices": [frame["frame_index"] for frame in frames],
        "legacy_samples": [
            {key: frame[key] for key in ("index", "frame_index", "pts", "sha256")} for frame in frames
        ],
        "native_render_anchors": len(native),
    }


def validate_anchors(anchors, expected):
    ids = set()
    for anchor in anchors:
        index = anchor["frame_index"]
        require(
            type(index) is int
            and 0 <= index < len(expected["pts"])
            and anchor["pts"] == expected["pts"][index]
            and anchor["decoded_rgb_sha256"] == expected["rgb"][index],
            "Native evidence RGB or picture timestamp differs from v200",
        )
        ids.add(index)
    require(len(ids) == len(anchors), "Duplicate native source anchor")
    return ids


def worker(condition, output):
    scoped(prior.worker, {"TRIAL": TRIAL, "WORK": WORK, "verify": verify, "verify_input": verify_input})(
        condition, output
    )


def authored(condition):
    return TRIAL / condition / ARM / "authored"


def work_dir(condition):
    return WORK / condition / ARM


@contextmanager
def active_case():
    path = TRIAL / "active.lock"
    with path.open("x"):
        try:
            yield
        finally:
            path.unlink()


def run(condition):
    frozen, plan = verify_plan()
    row = plan[condition]
    if row["status"] != "prepared":
        return {"status": row["status"]}
    with active_case():
        require(
            not any(WORK.glob(f"*/{ARM}")), "Archive and release the existing local case before another run"
        )
        free = shutil.disk_usage(BASE).free
        minimum = disk_required(row["bytes"])
        if free < minimum:
            return {"status": "pending_disk_headroom", "free_bytes": free, "required_bytes": minimum}
        _, file, _ = verify(condition)
        require(file.is_file() and frozen["planned_cases"] == 36, "Exact panel required")
        stage = authored(condition)
        stage.parent.mkdir(exist_ok=True)

        def launch(argv, **kwargs):
            output = stage / "output"
            require(
                argv[3:] == ["scripts.video_appearance_native_v108", "authored", str(output)],
                "Exact v202 worker invocation required",
            )
            return subprocess.Popen([*argv[:3], MODULE, "_worker", condition, str(output)], **kwargs)

        scoped(
            prior.original.runtime.run,
            {
                "TRIAL": stage.parent,
                "verify": lambda: verify(condition),
                "prerequisites": lambda *_: True,
                "subprocess": SimpleNamespace(
                    Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
                ),
                "__file__": __file__,
            },
        )("authored")
    return None


def qualification(condition):
    from scripts import video_faceswapai_full_app_v187 as original

    return scoped(original.qualification, {"TRIAL": TRIAL, "WORK": WORK, "__file__": __file__})(condition)


def overlap(start, last, expected):
    frames = set(range(start, last + 1))
    a, b = expected["ordinary"]
    ordinary = set(range(a, b))
    changed = set(expected["changed"])
    return {
        "start_frame": start,
        "last_supported_frame": last,
        "covered_frames": len(frames),
        "overlap_designated_frames": len(frames & ordinary),
        "overlap_changed_frames": len(frames & changed),
        "outside_changed_frames": len(frames - changed),
        "changed_frames_not_covered": len(changed - frames),
    }


def packet_inputs(packet, source, expected, native):
    from app.evidence.face_interval_binding import validate_packet

    pts, tb = expected["pts"], Fraction(expected["time_base"])
    validate_packet(packet, source, table_pts=tuple(pts))
    frames = [frame["anchor"] for frame in packet["frames"]]
    indices = validate_anchors(frames, expected)
    sequence = packet.get("face_sequence")
    if sequence:
        indices |= validate_anchors(sequence["frame_inputs"], expected)
    if interval := packet.get("face_interval"):
        indices.update(
            sample["frame_index"] for samples in interval["samples"].values() for sample in samples
        )
    if window := packet.get("window"):
        wanted = window["frame_indices"]
        require(
            bool(wanted)
            and set(wanted) <= native
            and Fraction(window["time_base"]) == tb
            and (window["start_pts"], window["end_pts"]) == (pts[wanted[0]], pts[wanted[-1]]),
            "Evidence window differs from the exact native input clocks",
        )
        indices.update(wanted)
    if (clip := packet.get("clip")) and clip["status"] == "ready":
        a, b = clip["start_frame_index"], clip["end_frame_index_exclusive"]
        require(
            clip["source_pts"] == pts[a:b],
            "Context clip does not reproduce the v200 source picture times",
        )
    require(indices <= native, "Evidence cited a frame absent from the persisted native observations")
    return indices


def verify_packet_pixels(packet, store, size):
    from PIL import Image

    from app.evidence.video_packets import _bounds, _context

    def image(ref):
        asset = packet["artifacts"][ref]
        with Image.open(io.BytesIO(store.read_bytes(asset["sha256"]))) as decoded:
            require(
                decoded.format == "PNG" and decoded.size == (asset["width"], asset["height"]),
                "Evidence PNG geometry differs",
            )
            return decoded.convert("RGB")

    for pos, frame in enumerate(packet["frames"], 1):
        anchor, refs = frame["anchor"], frame["artifact_refs"]
        original = image(refs["original"])
        require(
            original.size == size
            and hashlib.sha256(original.tobytes()).hexdigest() == anchor["decoded_rgb_sha256"],
            "Original evidence still does not contain its committed native RGB",
        )
        region = frame["context_region"]
        time_s = float(anchor["pts"] * Fraction(anchor["time_base"]))
        label = f"{packet['id']}.{pos} - frame {anchor['frame_index']}, {time_s:.3f} s"
        rendered = _context(original, region, label)
        actual = image(refs["context"])
        require(
            rendered.size == actual.size and rendered.tobytes() == actual.tobytes(),
            "Annotated evidence differs",
        )
        if region:
            crop = original.crop(_bounds(region, *size))
            actual_crop = image(refs["crop"])
            require(
                crop.size == actual_crop.size and crop.tobytes() == actual_crop.tobytes(),
                "Evidence crop differs",
            )


def inspect_refinement(file, packet, table, expected):
    import cv2

    record = packet["face_sequence"]
    centers = record["centers"]
    start, last = centers["during"][0], centers["during"][-1]
    cv2.setNumThreads(1)
    recognizer = cv2.FaceRecognizerSF.create(
        str(edge.WEIGHT), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
    )
    try:
        replay = edge.replay_features(
            file, record, {"frame_table": table, "rgb_sha256": expected["rgb"]}, recognizer
        )
        forwards = len(replay)
        refined = edge.refine(centers, replay)
    except edge.ReplayUnavailable as error:
        forwards = error.forwards
        refined = edge.retained(last, str(error))
    new_last = refined["proposed_last_supported_frame"]
    require(start <= last <= new_last < len(expected["pts"]), "Refinement exceeded the frozen picture clock")
    return {
        "status": refined["status"],
        "existing": overlap(start, last, expected),
        "proposed": overlap(start, new_last, expected),
        "last_frame_delta": new_last - last,
        "crop_forwards": forwards,
    }


def verify_clip_asset(stream, asset):
    from app.evidence.mp4 import inspect_mp4

    clock = inspect_mp4(stream)
    duration = clock.pop("duration_s")
    require(
        abs(asset["duration_s"] - duration) <= 0.000001
        and all(asset[key] == value for key, value in clock.items()),
        "Context MP4 clock differs",
    )


def audit_case(condition):
    _, file, bound = verify(condition)
    profile, work = qualification(condition)
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.evidence.face_interval_binding import APPEARANCE_METHOD, validate_sequence_record
    from app.evidence.service import asset_refs, load_packet_set
    from app.evidence.video_packets import _stage
    from app.storage.db import Analysis, Artefact, DetectorRun
    from app.storage.objects import ObjectStore

    private = read(work / "case.json")
    row = read(TRIAL / condition / "inputs.json")
    expected = expected_case(row)
    db_path = edge.database_path(work)
    db_sha = sha(db_path)
    engine = create_engine(f"sqlite:///{db_path.as_uri()}?mode=ro&immutable=1&uri=true")
    try:
        with Session(engine) as db:
            analysis = db.scalars(select(Analysis)).one()
            store = ObjectStore(work / "data/objects")
            require(
                analysis.sha256 == bound["file_sha256"] and analysis.profile == "full",
                "Exact owned analysis required",
            )
            if private["analysis_status"] != "done" or private["pipeline_status"] != "done":
                return {"status": "analysis_unavailable", "condition": condition, "role": row["role"]}
            require(
                analysis.result_hash == private["result_hash"] and bool(analysis.result_hash),
                "Stored app result changed",
            )
            store.verify(analysis.result_hash)
            published = json.loads(store.read_bytes(analysis.result_hash))
            direction = prior.original.baseline.customer_readout(analysis, published)["direction"]
            require(direction == private["customer_assessment"]["direction"], "Customer direction changed")
            stages = {}
            for name in ("ingest", "probe", "faces", "render"):
                staged = _stage(db, store, analysis.id, name)
                require(
                    staged is not None or name == "faces", "A completed case is missing source stage records"
                )
                if staged is not None:
                    stages[name] = staged[0]
            # The live decode stage is transient. The hash-bound worker case
            # preserves its exact legacy sample positions and RGB.
            stages["decode"] = {
                "frames": private["input"]["legacy_samples"],
                "audio": {
                    "status": stages["render"]["audio_status"],
                    "samples": 0,
                    "source_mapping": stages["render"]["audio_mapping"],
                },
            }
            runtime_input = verify_input(stages, row, bound["file_sha256"])
            require(
                runtime_input == private["input"]
                and runtime_input["sampled_frames"] > 0
                and runtime_input["native_render_anchors"] > 0,
                "Missing persisted coverage or mismatch with worker observations",
            )
            native = validate_anchors(stages["render"]["frame_anchors"], expected)
            evidence = load_packet_set(db, analysis, store)
            appearance = db.scalars(
                select(DetectorRun).where(DetectorRun.detector_id == "vid.review.face_appearance")
            ).one_or_none()
            if not evidence or evidence["status"] != "ready":
                return {
                    "status": "evidence_unavailable",
                    "condition": condition,
                    "role": row["role"],
                    "customer_direction": direction,
                }
            assets = asset_refs(evidence)
            for digest, asset in assets.items():
                store.verify(digest)
                if asset["media_type"] == "video/mp4":
                    with store.path_for(digest).open("rb") as stream:
                        verify_clip_asset(stream, asset)
            source = evidence["source"]
            require(
                source["sha256"] == bound["file_sha256"]
                and Fraction(source["time_base"]) == Fraction(expected["time_base"])
                and (source["width"], source["height"]) == expected["size"]
                and source["frame_count"] == len(expected["pts"])
                and evidence["coverage"]["decoded_frame_indices"] == sorted(native),
                "Evidence source or coverage differs from the frozen v200 sample",
            )
            verified, intervals, appearance_packet = set(), {}, None
            for packet in evidence["packets"]:
                verified |= packet_inputs(packet, source, expected, native)
                verify_packet_pixels(packet, store, expected["size"])
                semantics = packet["semantics"]
                if sequence := packet.get("face_sequence"):
                    during = sequence["centers"]["during"]
                elif interval := packet.get("face_interval"):
                    during = [sample["frame_index"] for sample in interval["samples"]["during"]]
                else:
                    continue
                intervals.setdefault(semantics, []).append(overlap(during[0], during[-1], expected))
                if semantics == APPEARANCE_METHOD:
                    require(appearance_packet is None, "One published appearance interval required")
                    appearance_packet = packet
            app_status = appearance.status if appearance else "not_run"
            details = {}
            if appearance and appearance.raw_output_artefact_id:
                raw = db.get(Artefact, appearance.raw_output_artefact_id)
                store.verify(raw.sha256)
                details = json.loads(store.read_bytes(raw.sha256)).get("details") or {}
            windows = details.get("face_sequence_windows") or []
            for window in windows:
                validate_sequence_record(window, source, table_pts=tuple(expected["pts"]), require_flag=False)
                require(
                    validate_anchors(window["frame_inputs"], expected) <= native,
                    "Measured window was not bound",
                )
            if appearance_packet:
                refinement = inspect_refinement(
                    file, appearance_packet, stages["probe"]["frame_table"], expected
                )
            else:
                refinement = {"status": "no_existing_appearance_interval", "crop_forwards": 0}
            result = {
                "status": "scored",
                "condition": condition,
                "role": row["role"],
                "customer_direction": direction,
                "legacy_frame_status": private["target_status"] or "not_run",
                "appearance_route_status": app_status,
                "appearance_check_status": details.get("appearance_check_status", app_status),
                "appearance_measured_windows": len(windows),
                "appearance_selected_window": bool(details.get("face_sequence")),
                "verified_sampled_frames": runtime_input["sampled_frames"],
                "verified_native_anchors": len(native),
                "verified_evidence_frames": len(verified),
                "verified_evidence_assets": len(assets),
                "face_review_intervals": intervals,
                "edge_refinement": refinement,
                "actual_changed_frames": len(expected["changed"]),
                "worker_seconds": profile["duration_s"],
            }
    finally:
        engine.dispose()
    require(sha(db_path) == db_sha, "Read-only analysis database changed during audit")
    return result


def audit_path(condition):
    return authored(condition) / "audit.json"


def review(condition):
    verify(condition)
    stage = authored(condition)
    require(
        (stage / "profile.json").is_file() and not audit_path(condition).exists(),
        "One completed guard review required",
    )
    profile = read(stage / "profile.json")
    if profile["child_exit_code"] != 0 or profile["guard"] is not None:
        row = read(TRIAL / "private-plan.json")[condition]
        result = {"condition": condition, "role": row["role"], "status": "guard_or_worker_failed"}
    else:
        finished = subprocess.run(
            [sys.executable, "-B", "-m", MODULE, "_audit", condition],
            cwd=ROOT / "backend",
            check=True,
            capture_output=True,
            text=True,
            timeout=360,
        )
        result = json.loads(finished.stdout)
    write(audit_path(condition), result)
    return {"status": result["status"]}


def archive_paths(condition):
    archive = SEALED / f"{condition}.tar"
    return archive, archive.with_suffix(".tar.receipt.json")


def allowed_data_links(work):
    inventory = read(prior.original.INVENTORY)
    links = {}
    for asset in inventory["local_fuser_files"]:
        relative = Path(asset.get("slot", "data")) / asset["relative"]
        if (work / relative).is_symlink():
            links[relative.as_posix()] = {
                "target": str(Path(asset["source_file"]).resolve(strict=True)),
                "sha256": asset["sha256"],
            }
    return links


def seal(condition):
    from scripts.video_vcd2_case_archive_v202 import MARKER, WORK_SCHEMA, seal_case

    verify(condition)
    stage = authored(condition)
    audit = read(direct_file(audit_path(condition)))
    require(audit["condition"] == condition, "Review this case before sealing its bytes")
    with active_case():
        require(
            ARCHIVE.is_dir() and ARCHIVE.parent.parent.is_mount(), "Seagate archive volume must be mounted"
        )
        archive, receipt = archive_paths(condition)
        require(not receipt.exists() and not archive.exists(), "Preserve every prior seal attempt")
        SEALED.mkdir(mode=0o700, exist_ok=True)
        work = work_dir(condition)
        if not work.exists():
            require(audit["status"] == "guard_or_worker_failed", "Completed work directory is missing")
            work.mkdir(mode=0o700, parents=True)
        marker = work / MARKER
        owner = {"schema": WORK_SCHEMA, "case_id": work.name, "work_path": str(work)}
        if marker.is_file():
            require(read(direct_file(marker)) == owner, "Unexpected workspace owner")
        else:
            write(marker, owner)
        for name in ("inputs.json", "freeze.json", f"{ARM}.yaml"):
            data = (TRIAL / condition / name).read_bytes()
            copy = stage / name
            if copy.is_file():
                require(direct_file(copy).read_bytes() == data, "Archived stage binding changed")
            else:
                with copy.open("xb") as stream:
                    stream.write(data)
        seal_case(work, stage, archive, allowed_data_links=allowed_data_links(work))
    return {"status": "sealed"}


def release(condition):
    from scripts.video_vcd2_case_archive_v202 import release_local_work, verify_archive

    verify(condition)
    archive, receipt = archive_paths(condition)
    sealed = read(direct_file(receipt))
    require(sealed["stage_path"] == str(authored(condition)), "Exact case archive required")
    with active_case():
        if work_dir(condition).is_dir():
            release_local_work(work_dir(condition), archive, sealed)
        else:
            verify_archive(archive, sealed)
    return {"status": "local_workspace_released"}


def metric_totals(intervals):
    fields = (
        "covered_frames",
        "overlap_designated_frames",
        "overlap_changed_frames",
        "outside_changed_frames",
        "changed_frames_not_covered",
    )
    return {field: sum(row[field] for row in intervals) for field in fields}


def describe(rows):
    scored = [row for row in rows if row["status"] == "scored"]
    customer = [row.get("customer_direction") for row in rows if row.get("customer_direction")]
    methods = sorted(
        set(REVIEW_METHODS) | {method for row in scored for method in row["face_review_intervals"]}
    )
    by_method = {}
    for method in methods:
        flags = [row for row in scored if row["face_review_intervals"].get(method)]
        by_method[method] = {
            "negative_cases_flagged": len(flags),
            "intervals": sum(len(row["face_review_intervals"][method]) for row in flags),
            "interval_frame_totals": metric_totals(
                [item for row in flags for item in row["face_review_intervals"][method]]
            ),
        }
    with_refinement = [row["edge_refinement"] for row in scored if "existing" in row["edge_refinement"]]
    verified = [row for row in scored if row["appearance_check_status"] == "ok"]
    flagged = sum(bool(row["face_review_intervals"]) for row in scored)
    return {
        "planned_negative_cases": len(rows),
        "case_outcomes": dict(Counter(row["status"] for row in rows)),
        "customer_direction_counts": {direction: customer.count(direction) for direction in DIRECTIONS},
        "customer_direction_unobserved": len(rows) - len(customer),
        "full_app_evidence_verified_cases": len(scored),
        "appearance_route_outcomes": dict(Counter(row["appearance_route_status"] for row in scored)),
        "appearance_check_outcomes": dict(Counter(row["appearance_check_status"] for row in scored)),
        "legacy_frame_route_outcomes": dict(Counter(row["legacy_frame_status"] for row in scored)),
        "appearance_measured_cases": len(verified),
        "appearance_measured_windows": sum(row["appearance_measured_windows"] for row in scored),
        "appearance_selected_cases": sum(row["appearance_selected_window"] for row in scored),
        "negative_cases_with_face_review_interval": flagged,
        "face_review_share_of_planned": flagged / len(rows),
        "face_review_share_of_evidence_verified": flagged / len(scored) if scored else None,
        "face_review_methods": by_method,
        "refinement_statuses": dict(Counter(row["edge_refinement"]["status"] for row in scored)),
        "matched_refinement_intervals": len(with_refinement),
        "refinement_existing_frame_totals": metric_totals([row["existing"] for row in with_refinement]),
        "refinement_proposed_frame_totals": metric_totals([row["proposed"] for row in with_refinement]),
        "refinement_additional_frames": sum(row["last_frame_delta"] for row in with_refinement),
        "refinement_crop_forwards": sum(row["edge_refinement"]["crop_forwards"] for row in scored),
        "verified_sampled_frames": sum(row["verified_sampled_frames"] for row in scored),
        "verified_native_anchors": sum(row["verified_native_anchors"] for row in scored),
        "verified_evidence_frames": sum(row["verified_evidence_frames"] for row in scored),
        "verified_evidence_assets": sum(row["verified_evidence_assets"] for row in scored),
    }


def paired_flag_changes(by_slot):
    paired = {}
    for role in ROLES[1:]:
        states = Counter()
        for group in by_slot.values():
            before = group[0]
            edited = next(row for row in group if row["role"] == role)
            if before["status"] != "scored" or edited["status"] != "scored":
                states["comparison_unavailable"] += 1
                continue
            initial, final = bool(before["face_review_intervals"]), bool(edited["face_review_intervals"])
            states[("flagged" if initial else "clear") + "_to_" + ("flagged" if final else "clear")] += 1
        paired[role] = dict(states)
    return paired


def readout():
    _, plan = verify_plan()
    rows, seals = [], {}
    for condition, case in plan.items():
        archive, receipt = archive_paths(condition)
        status = case["status"]
        if status == "prepared":
            if receipt.is_file():
                from scripts.video_vcd2_case_archive_v202 import verify_archive

                sealed = read(direct_file(receipt))
                verify_archive(archive, sealed)
                path = direct_file(audit_path(condition))
                audited = read(path)
                require(
                    sealed["stage_path"] == str(authored(condition))
                    and [
                        entry["sha256"] for entry in sealed["entries"] if entry["path"] == "stage/audit.json"
                    ]
                    == [sha(path)]
                    and audited["condition"] == condition
                    and audited["role"] == case["role"],
                    "Audited result differs from the sealed case",
                )
                seals[condition] = {
                    "receipt_sha256": sha(receipt),
                    "archive_sha256": sealed["archive_sha256"],
                }
                rows.append(audited)
                continue
            status = "attempt_pending_review_or_seal" if authored(condition).exists() else "run_pending"
        rows.append({"condition": condition, "role": case["role"], "status": status})
    by_slot = {slot: [row for row in rows if plan[row["condition"]]["slot"] == slot] for slot in SLOTS}
    pending = {"attempt_pending_review_or_seal", "run_pending"}
    return {
        "schema": "chipmunk.video-vcd2-full-app-result/202",
        "status": "partial" if any(row["status"] in pending for row in rows) else "complete",
        "freeze_sha256": sha(PUBLIC),
        "sealed_cases": len(seals),
        "archive_set_commitment_sha256": hashlib.sha256(
            json.dumps(seals, sort_keys=True).encode()
        ).hexdigest(),
        "planned_source_groups": len(SLOTS),
        "complete_matched_source_groups": sum(
            all(row["status"] == "scored" for row in group) for group in by_slot.values()
        ),
        "source_groups_with_any_face_review_interval": sum(
            any(row.get("face_review_intervals") for row in group) for group in by_slot.values()
        ),
        "paired_any_face_review_interval_changes": paired_flag_changes(by_slot),
        "overall": describe(rows),
        "by_role": {role: describe([row for row in rows if row["role"] == role]) for role in ROLES},
        "all_face_swap_negatives": True,
        "serving_defaults_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "freeze", "readout"):
        commands.add_parser(name)
    for name in ("run", "review", "seal", "release", "_worker", "_audit"):
        operation = commands.add_parser(name)
        operation.add_argument("condition", choices=CONDITIONS)
        if name == "_worker":
            operation.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "_worker":
        worker(args.condition, args.output)
        return
    if args.command == "_audit":
        edge.enforce_read_only()
        result = audit_case(args.condition)
    else:
        command = globals()[args.command]
        result = command(args.condition) if hasattr(args, "condition") else command()
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
